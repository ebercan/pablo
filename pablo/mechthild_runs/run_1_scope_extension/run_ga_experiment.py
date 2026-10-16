"""Core BO execution for the run_1 scope-extension PlanAhead benchmark sweep."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from pablo.optimize import (
    optimize_acqf_mixed_GA,
    optimize_plan_ahead_qlogEI_fantasized,
    optimize_plan_ahead_qlogEI_myopic,
)
from pablo.utils.acq_utils import get_joint_log_ei
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from pablo.mechthild_runs.run_1_scope_extension.config import (
    BENCHMARK_CLASS_MAP,
    DEFAULT_BATCH_CONSTRAINTS,
    DEPENDENT_BLOCK_SIZES,
    DEPENDENT_SOURCE_CARDINALITIES,
    FANTASIZED_OPERATING_MODES,
    INITIAL_AVAILABLE_SOURCE_A_COUNT,
    LOOKAHEAD_HORIZON,
    MYOPIC_OPERATING_MODE,
    N_BO_BATCHES,
    N_CONTINUOUS_DIMS,
    N_DISCRETE_DIMS,
    N_EVALS_GA,
    N_OPTIMIZE_ACQF_REPETITIONS,
    N_STARTING_POINTS,
    NUM_FANTASIES,
    PLAN_AHEAD_STRATEGIES,
    POP_SIZE,
    Q_BATCH_SIZE,
    Q_EXPAND_SIZE,
    QMC_SAMPLES,
    SOURCE_A_AVAILABILITY_DELAY,
    TOTAL_DIM,
)
from pablo.mechthild_runs.run_1_scope_extension.run_state import update_run_state


def print_progress(
    label: str,
    completed: int,
    total: int,
    extra: str = "",
) -> None:
    """Print a single-line dynamic progress bar.

    Args:
        label: Prefix label shown at the left of the bar.
        completed: Number of completed steps.
        total: Total number of steps.
        extra: Optional suffix text for additional metrics.
    """
    bar_width = 24
    safe_total = max(total, 1)
    filled = int(bar_width * completed / safe_total)
    bar = "#" * filled + "." * (bar_width - filled)
    line = f"\r{label} [{bar}] {completed}/{total}"
    if extra:
        line += f" | {extra}"

    sys.stdout.write(line)
    sys.stdout.flush()
    if completed >= total:
        sys.stdout.write("\n")


def _sample_uniform_block(
    lower: torch.Tensor,
    upper: torch.Tensor,
    n_rows: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a dense block uniformly within per-dimension bounds.

    Args:
        lower: Tensor with shape (d_block,) containing lower bounds.
        upper: Tensor with shape (d_block,) containing upper bounds.
        n_rows: Number of rows to sample.
        generator: Random generator controlling reproducibility.

    Returns:
        Tensor with shape (n_rows, d_block).
    """
    span = upper - lower
    return lower.unsqueeze(0) + span.unsqueeze(0) * torch.rand(
        (n_rows, lower.shape[0]), dtype=lower.dtype, generator=generator
    )


def build_benchmark(benchmark_name: str):
    """Build a benchmark instance in TOTAL_DIM = 26 dimensions.

    Args:
        benchmark_name: Key from BENCHMARK_CLASS_MAP.

    Returns:
        A synthetic benchmark object with total dimension 26.

    Raises:
        ValueError: If benchmark_name is unknown.
    """
    if benchmark_name not in BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    benchmark_class = BENCHMARK_CLASS_MAP[benchmark_name]
    return benchmark_class(d=TOTAL_DIM)


def generate_dependent_choice_sources(
    bounds: torch.Tensor,
    seed_discrete: int,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dependent source tensors for the three discrete blocks.

    The full search space has 26 dimensions: 4 continuous followed by 22
    discrete. The 22 discrete dimensions are split into dependent blocks
    (``DEPENDENT_BLOCK_SIZES``), each driven by one latent source whose rows
    are drawn uniformly within the block's bounds:
    - block A: 18 dimensions, 225 choices by default.
    - block B: 2 dimensions, 14 choices by default.
    - block C: 2 dimensions, 8 choices by default.

    Args:
        bounds: Tensor with shape (2, 26), row 0 lower and row 1 upper bounds.
        seed_discrete: Seed controlling all random draws for discrete source tensors.
        source_cardinalities: Tuple (n_a, n_b, n_c) for source row counts.

    Returns:
        Tuple (source_a, source_b, source_c) with shapes:
        - source_a: (n_a, 18)
        - source_b: (n_b, 2)
        - source_c: (n_c, 2)

    Raises:
        ValueError: If bounds do not have shape (2, 26), the block sizes do not
            sum to 22, or the cardinalities are not three positive integers.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] != TOTAL_DIM:
        raise ValueError("bounds must have shape (2, 33).")
    if len(source_cardinalities) != 3:
        raise ValueError("source_cardinalities must have length 3.")
    if sum(DEPENDENT_BLOCK_SIZES) != N_DISCRETE_DIMS:
        raise ValueError("Dependent block sizes do not sum to 30 discrete dimensions.")

    n_a, n_b, n_c = source_cardinalities
    if n_a <= 0 or n_b <= 0 or n_c <= 0:
        raise ValueError("All source cardinalities must be positive integers.")

    bounds_cpu = bounds.detach().cpu().to(torch.double)
    discrete_bounds = bounds_cpu[:, N_CONTINUOUS_DIMS:]

    split_a = DEPENDENT_BLOCK_SIZES[0]
    split_b = split_a + DEPENDENT_BLOCK_SIZES[1]

    lower_a = discrete_bounds[0, :split_a]
    upper_a = discrete_bounds[1, :split_a]

    lower_b = discrete_bounds[0, split_a:split_b]
    upper_b = discrete_bounds[1, split_a:split_b]

    lower_c = discrete_bounds[0, split_b:]
    upper_c = discrete_bounds[1, split_b:]

    generator = torch.Generator()
    generator.manual_seed(seed_discrete)

    source_a = _sample_uniform_block(lower_a, upper_a, n_a, generator)
    source_b = _sample_uniform_block(lower_b, upper_b, n_b, generator)
    source_c = _sample_uniform_block(lower_c, upper_c, n_c, generator)

    return source_a, source_b, source_c


def build_x_allowed_from_sources(
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    source_c: torch.Tensor,
) -> torch.Tensor:
    """Create X_allowed by cartesian composition of three source tensors.

    Args:
        source_a: Tensor with shape (n_a, 18).
        source_b: Tensor with shape (n_b, 2).
        source_c: Tensor with shape (n_c, 2).

    Returns:
        Tensor with shape (n_a * n_b * n_c, 22), ordered with source_a slowest
        and source_c fastest, so row ``i`` belongs to source-A row
        ``i // (n_b * n_c)``.

    Raises:
        ValueError: If source column counts are not (18, 2, 2).
    """
    if source_a.ndim != 2 or source_a.shape[1] != DEPENDENT_BLOCK_SIZES[0]:
        raise ValueError("source_a must have shape (n_a, 26).")
    if source_b.ndim != 2 or source_b.shape[1] != DEPENDENT_BLOCK_SIZES[1]:
        raise ValueError("source_b must have shape (n_b, 2).")
    if source_c.ndim != 2 or source_c.shape[1] != DEPENDENT_BLOCK_SIZES[2]:
        raise ValueError("source_c must have shape (n_c, 2).")

    n_a = source_a.shape[0]
    n_b = source_b.shape[0]
    n_c = source_c.shape[0]

    block_a = source_a.repeat_interleave(n_b * n_c, dim=0)
    block_b = source_b.repeat_interleave(n_c, dim=0).repeat(n_a, 1)
    block_c = source_c.repeat(n_b, 1).repeat(n_a, 1)

    return torch.cat([block_a, block_b, block_c], dim=1)


def decompose_x_allowed_to_sources(
    x_subset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a ``(n, 22)`` x_allowed-like tensor into per-source unique rows.

    Used to pass the active per-component embedding tables to the GA in
    component mode.

    Args:
        x_subset: ``(n, 22)`` tensor whose columns are
            ``[source_a (18) | source_b (2) | source_c (2)]``.

    Returns:
        Three-tuple of ``(cat_emb, sol_emb, bas_emb)`` containing the unique
        per-source rows present in the subset.
    """
    split_a = DEPENDENT_BLOCK_SIZES[0]
    split_b = split_a + DEPENDENT_BLOCK_SIZES[1]
    cat_emb = x_subset[:, :split_a].unique(dim=0)
    sol_emb = x_subset[:, split_a:split_b].unique(dim=0)
    bas_emb = x_subset[:, split_b:].unique(dim=0)
    return cat_emb, sol_emb, bas_emb


def build_x_allowed_tensor(
    bounds: torch.Tensor,
    seed_discrete: int,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
) -> torch.Tensor:
    """Generate a full X_allowed tensor for one benchmark and seed.

    Args:
        bounds: Tensor with shape (2, 26).
        seed_discrete: Seed for generating dependent source tensors.
        source_cardinalities: Tuple (n_a, n_b, n_c).

    Returns:
        Tensor with shape (n_a * n_b * n_c, 22) on the dtype and device of
        ``bounds``.
    """
    source_a, source_b, source_c = generate_dependent_choice_sources(
        bounds=bounds,
        seed_discrete=seed_discrete,
        source_cardinalities=source_cardinalities,
    )
    x_allowed = build_x_allowed_from_sources(
        source_a=source_a, source_b=source_b, source_c=source_c
    )
    return x_allowed.to(dtype=bounds.dtype, device=bounds.device)


def generate_starting_set(
    bounds: torch.Tensor,
    x_allowed: torch.Tensor,
    seed_starting: int,
    n_starting: int = N_STARTING_POINTS,
) -> torch.Tensor:
    """Generate initial mixed-design points from discrete choices.

    Draws ``n_starting`` distinct rows of ``x_allowed`` without replacement and
    pairs each with continuous values sampled uniformly within ``bounds``.

    Args:
        bounds: Tensor with shape (2, 26).
        x_allowed: Tensor with shape (n_allowed, 22).
        seed_starting: Seed from SEED_LIST_2.
        n_starting: Number of starting points.

    Returns:
        Tensor with shape (n_starting, 26), columns ``[continuous (4) | discrete (22)]``.

    Raises:
        ValueError: If shapes are inconsistent, n_starting is not positive, or
            n_starting exceeds n_allowed.
    """
    if bounds.ndim != 2 or bounds.shape != (2, TOTAL_DIM):
        raise ValueError("bounds must have shape (2, 33).")
    if x_allowed.ndim != 2 or x_allowed.shape[1] != N_DISCRETE_DIMS:
        raise ValueError("x_allowed must have shape (n_allowed, 30).")
    if n_starting <= 0:
        raise ValueError("n_starting must be positive.")
    if n_starting > x_allowed.shape[0]:
        raise ValueError("n_starting cannot exceed number of allowed discrete rows.")

    generator = torch.Generator()
    generator.manual_seed(seed_starting)

    indices = torch.randperm(x_allowed.shape[0], generator=generator)[:n_starting]
    x_discrete = x_allowed[indices]

    # Sample on CPU for reproducible generator behavior and then move to target device.
    cont_lower_cpu = bounds[0, :N_CONTINUOUS_DIMS].detach().cpu()
    cont_upper_cpu = bounds[1, :N_CONTINUOUS_DIMS].detach().cpu()
    x_continuous_cpu = cont_lower_cpu.unsqueeze(0) + (
        cont_upper_cpu - cont_lower_cpu
    ).unsqueeze(0) * torch.rand(
        (n_starting, N_CONTINUOUS_DIMS),
        dtype=bounds.dtype,
        generator=generator,
    )
    x_continuous = x_continuous_cpu.to(device=bounds.device, dtype=bounds.dtype)

    return torch.cat([x_continuous, x_discrete], dim=1)


def _fit_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
) -> SingleTaskGP:
    """Fit a SingleTaskGP model with standard transforms.

    Args:
        train_x: Tensor with shape (n, 26).
        train_y: Tensor with shape (n, 1).
        bounds: Tensor with shape (2, 26).

    Returns:
        Fitted SingleTaskGP model.
    """
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _build_acquisition(
    model: SingleTaskGP,
    best_f: torch.Tensor,
    q: int,
    qmc_samples: int,
):
    """Construct acquisition function for the current BO model.

    Args:
        model: Trained GP model.
        best_f: Best observed objective value tensor.
        q: Batch size.
        qmc_samples: QMC sample count for q > 1.

    Returns:
        LogExpectedImprovement for q=1 or qLogExpectedImprovement for q>1.
    """
    if q == 1:
        return LogExpectedImprovement(model=model, best_f=best_f)
    return qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([qmc_samples])),
    )


def _resolve_strategy(strategy: str) -> str:
    """Validate and normalize run strategy names.

    Args:
        strategy: Strategy label from inputs.

    Returns:
        Lowercase strategy string.

    Raises:
        ValueError: If strategy is not supported.
    """
    normalized = strategy.strip().lower()
    supported = set(PLAN_AHEAD_STRATEGIES)
    if normalized not in supported:
        raise ValueError(
            f"strategy must be one of {sorted(supported)}, got: {strategy}"
        )
    return normalized


def _resolve_fantasized_operating_mode(
    strategy: str,
    fantasized_operating_mode: str,
) -> str:
    """Validate operating mode against the selected strategy.

    Args:
        strategy: Normalized strategy string.
        fantasized_operating_mode: Requested operating mode string.

    Returns:
        Resolved operating mode.

    Raises:
        ValueError: If strategy and operating mode are inconsistent.
    """
    normalized_mode = fantasized_operating_mode.strip().lower()
    if strategy == "myopic":
        if normalized_mode not in {"", MYOPIC_OPERATING_MODE}:
            raise ValueError(
                "fantasized_operating_mode must be 'myopic' or empty when "
                f"strategy='myopic', got {fantasized_operating_mode!r}"
            )
        return MYOPIC_OPERATING_MODE

    if normalized_mode not in set(FANTASIZED_OPERATING_MODES):
        raise ValueError(
            "fantasized_operating_mode must be one of "
            f"{sorted(FANTASIZED_OPERATING_MODES)} for strategy='fantasized', "
            f"got {fantasized_operating_mode!r}"
        )
    return normalized_mode


def _initialize_source_a_state(
    source_a_cardinality: int,
    initial_available_source_a_count: int,
    seed_initial_available: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize source-A availability and delayed-unlock state.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Return value of _initialize_source_a_state.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Return value of _initialize_source_a_state.
"""
    if initial_available_source_a_count <= 0:
        raise ValueError("initial_available_source_a_count must be positive")
    if initial_available_source_a_count > source_a_cardinality:
        raise ValueError(
            "initial_available_source_a_count cannot exceed source-A cardinality"
        )

    generator = torch.Generator()
    generator.manual_seed(seed_initial_available)
    sampled = torch.randperm(source_a_cardinality, generator=generator)[
        :initial_available_source_a_count
    ]

    available_mask = torch.zeros(source_a_cardinality, dtype=torch.bool)
    available_mask[sampled] = True
    pending_unlock_iteration = torch.full(
        (source_a_cardinality,),
        -1,
        dtype=torch.long,
    )
    return available_mask, pending_unlock_iteration, sampled


def _release_matured_source_a(
    available_source_a_mask: torch.Tensor,
    pending_unlock_iteration: torch.Tensor,
    current_iteration: int,
) -> list[int]:
    """Release source-A entries whose delayed availability has matured.
        Returns:
            list[int]: Return value of _release_matured_source_a.
    
    Returns:
        list[int]: Return value of _release_matured_source_a.
"""
    matured = torch.where(
        (pending_unlock_iteration >= 0)
        & (pending_unlock_iteration <= current_iteration)
    )[0]
    if matured.numel() == 0:
        return []

    available_source_a_mask[matured] = True
    pending_unlock_iteration[matured] = -1
    return [int(idx) for idx in matured.tolist()]


def _row_masks_from_source_a_state(
    source_a_index_per_row: torch.Tensor,
    available_source_a_mask: torch.Tensor,
    pending_unlock_iteration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build row masks for available and unordered X_allowed subsets.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Return value of _row_masks_from_source_a_state.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Return value of _row_masks_from_source_a_state.
"""
    pending_source_a_mask = pending_unlock_iteration >= 0
    unordered_source_a_mask = (~available_source_a_mask) & (~pending_source_a_mask)

    available_row_mask = available_source_a_mask[source_a_index_per_row]
    unordered_row_mask = unordered_source_a_mask[source_a_index_per_row]
    return available_row_mask, unordered_row_mask


def _build_x_ordered_for_fantasizing(
    x_allowed_cpu: torch.Tensor,
    source_a_index_per_row: torch.Tensor,
    pending_unlock_iteration: torch.Tensor,
    current_iteration: int,
    lookahead_horizon: int,
) -> list[torch.Tensor]:
    """Build ordered tensors that become available at each future lookahead step.
        Returns:
            list[torch.Tensor]: Return value of _build_x_ordered_for_fantasizing.
    
    Returns:
        list[torch.Tensor]: Return value of _build_x_ordered_for_fantasizing.
"""
    x_ordered: list[torch.Tensor] = []
    empty = torch.empty((0, x_allowed_cpu.shape[1]), dtype=x_allowed_cpu.dtype)

    for step in range(1, lookahead_horizon + 1):
        unlock_at = current_iteration + step
        source_indices = torch.where(pending_unlock_iteration == unlock_at)[0]
        if source_indices.numel() == 0:
            x_ordered.append(empty.clone())
            continue

        row_mask = (
            source_a_index_per_row.unsqueeze(-1) == source_indices.unsqueeze(0)
        ).any(dim=1)
        x_ordered.append(x_allowed_cpu[row_mask])

    return x_ordered


def _extract_source_a_index_from_candidate(
    candidate: torch.Tensor,
    source_a: torch.Tensor,
) -> int:
    """Map an expansion candidate back to its source-A index.
        Returns:
            int: Return value of _extract_source_a_index_from_candidate.
    
    Returns:
        int: Return value of _extract_source_a_index_from_candidate.
"""
    source_a_width = DEPENDENT_BLOCK_SIZES[0]
    source_a_slice = candidate[
        0, N_CONTINUOUS_DIMS : N_CONTINUOUS_DIMS + source_a_width
    ]
    source_a_slice = source_a_slice.detach().cpu().to(dtype=source_a.dtype)

    matches = torch.where(
        torch.isclose(
            source_a,
            source_a_slice.unsqueeze(0),
            atol=1e-12,
            rtol=0.0,
        ).all(dim=1)
    )[0]
    if matches.numel() != 1:
        raise ValueError("Expected exactly one source-A match for expansion candidate")
    return int(matches.item())


def _scalarize(value: object) -> float:
    """Convert scalar-like values to Python float.
        Returns:
            float: Return value of _scalarize.
    
    Returns:
        float: Return value of _scalarize.
"""
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def run_bo_ga_experiment(
    benchmark_name: str,
    seed_discrete: int,
    seed_starting: int,
    pop_size: int = POP_SIZE,
    ga_ratio: Optional[int] = None,
    n_evals_ga: int = N_EVALS_GA,
    q: int = Q_BATCH_SIZE,
    q_expand: int = Q_EXPAND_SIZE,
    n_eval: int = N_BO_BATCHES,
    n_optimize_acqf: int = N_OPTIMIZE_ACQF_REPETITIONS,
    n_starting: int = N_STARTING_POINTS,
    strategy: str = "fantasized",
    seed_initial_available: int = 1,
    initial_available_source_a_count: int = INITIAL_AVAILABLE_SOURCE_A_COUNT,
    source_a_availability_delay: int = SOURCE_A_AVAILABILITY_DELAY,
    lookahead_horizon: int = LOOKAHEAD_HORIZON,
    num_fantasies: list[int] | None = None,
    batch_constraints: dict[int, int] | None = None,
    fixed_dims: dict[int, float] | None = None,
    qmc_samples: int = QMC_SAMPLES,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
    run_state_path: Path | None = None,
    run_state_combination_index: int | None = None,
    verbose: bool = False,
    progress_label: str | None = None,
    fantasized_operating_mode: str = "layer_averaged",
) -> dict[str, object]:
    """Execute one run_1 scope-extension BO experiment with delayed source-A availability.

    Each iteration releases the source-A entries whose delay has elapsed,
    selects a q-batch from the available rows with the mixed GA optimizer,
    evaluates it and refits the GP, then picks ``q_expand`` expansion
    candidates from the not-yet-requested source-A rows (plan-ahead fantasized
    or myopic) and schedules the chosen source-A entry to unlock
    ``source_a_availability_delay`` iterations later.

    Args:
        benchmark_name: Key from BENCHMARK_CLASS_MAP.
        seed_discrete: Seed for the discrete source tensors.
        seed_starting: Seed for the initial design.
        pop_size: GA population size, used for both the batch and the
            expansion optimizers.
        ga_ratio: Recorded in the result dictionary only; not used in the run.
        n_evals_ga: Acquisition evaluation budget per GA call.
        q: Batch size of the main acquisition step.
        q_expand: Number of expansion candidates requested per iteration.
        n_eval: Number of BO iterations.
        n_optimize_acqf: Number of independent GA restarts per iteration; the
            restart with the highest joint log-EI is kept.
        n_starting: Number of initial design points, drawn from the initially
            available rows.
        strategy: ``"fantasized"`` or ``"myopic"`` expansion strategy.
        seed_initial_available: Seed selecting the initially available
            source-A entries.
        initial_available_source_a_count: Number of source-A entries available
            at iteration 0.
        source_a_availability_delay: Iterations between selecting a source-A
            entry for expansion and it becoming available.
        lookahead_horizon: Lookahead depth of the fantasized strategy.
        num_fantasies: Fantasy counts per lookahead level, of length
            ``lookahead_horizon``. Defaults to NUM_FANTASIES.
        batch_constraints: Mapping from dimension index to the number of
            distinct values allowed within the q-batch. Defaults to
            DEFAULT_BATCH_CONSTRAINTS. Applied to the main step only.
        fixed_dims: Mapping from dimension index to a fixed input value that
            the optimizers do not change.
        qmc_samples: Number of Sobol QMC samples for qLogEI when q > 1.
        source_cardinalities: Tuple (n_a, n_b, n_c) of source row counts.
        run_state_path: Run-state JSON file updated at every phase, or None.
        run_state_combination_index: Combination index written to the run
            state; updates are skipped when None.
        verbose: If True, print a progress bar per phase.
        progress_label: Label for the progress bar; built from the benchmark,
            seeds and strategy when None.
        fantasized_operating_mode: One of ``"only_terminal"``, ``"averaged"``,
            ``"layer_averaged"`` for the fantasized strategy; must be
            ``"myopic"`` or empty for the myopic strategy.

    Returns:
        Dictionary with the run parameters, the source-A cardinality and
        initially available indices, the X_allowed / initial available /
        initial unordered shapes, and per-iteration lists: ``best_fs``,
        ``candidates_list``, ``max_acq_values_list``,
        ``acq_values_repeated_list``, the selected source-A indices, acquisition
        values and candidates, ``released_source_a_by_iteration``, available and
        pending source-A counts before and after selection, and wall times for
        acquisition optimization, model fitting and function evaluation.

    Raises:
        ValueError: If a count or budget argument is out of range, the
            strategy or operating mode is invalid, ``num_fantasies`` does not
            match ``lookahead_horizon``, a batch constraint exceeds ``q``, or
            the selected expansion source-A entry is already available or
            pending.
    """
    if n_eval <= 0:
        raise ValueError("n_eval must be positive")
    if n_evals_ga <= 0:
        raise ValueError("n_evals_ga must be positive")
    if pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if n_optimize_acqf <= 0:
        raise ValueError("n_optimize_acqf must be positive")
    if q_expand <= 0:
        raise ValueError("q_expand must be positive")
    if source_a_availability_delay < 0:
        raise ValueError("source_a_availability_delay must be non-negative")
    if lookahead_horizon <= 0:
        raise ValueError("lookahead_horizon must be positive")

    normalized_strategy = _resolve_strategy(strategy)
    resolved_fantasized_mode = _resolve_fantasized_operating_mode(
        strategy=normalized_strategy,
        fantasized_operating_mode=fantasized_operating_mode,
    )
    active_num_fantasies = (
        list(NUM_FANTASIES) if num_fantasies is None else list(num_fantasies)
    )
    if len(active_num_fantasies) != lookahead_horizon:
        raise ValueError(
            "num_fantasies length must equal lookahead_horizon "
            f"({lookahead_horizon}), got {len(active_num_fantasies)}"
        )

    active_batch_constraints = (
        dict(DEFAULT_BATCH_CONSTRAINTS)
        if batch_constraints is None
        else dict(batch_constraints)
    )
    for dim_idx, k_val in active_batch_constraints.items():
        if k_val > q:
            raise ValueError(
                f"batch constraint k={k_val} for dim {dim_idx} exceeds q={q}."
            )

    display_label = progress_label or (
        f"{benchmark_name} | s1={seed_discrete} | s2={seed_starting} | "
        f"s3={seed_initial_available} | {normalized_strategy}"
    )

    if verbose:
        print_progress(
            label=display_label,
            completed=0,
            total=n_eval,
            extra=f"phase=initializing | best=NA | pop={pop_size} | evals={n_evals_ga}",
        )

    benchmark = build_benchmark(benchmark_name)

    resolved_device = torch.device("cpu")
    print(f"[CPU ONLY] resolved_device is set to: {resolved_device}")
    bounds = benchmark.bounds().to(dtype=torch.double, device=resolved_device)

    source_a, source_b, source_c = generate_dependent_choice_sources(
        bounds=bounds,
        seed_discrete=seed_discrete,
        source_cardinalities=source_cardinalities,
    )
    x_allowed_cpu = build_x_allowed_from_sources(
        source_a=source_a,
        source_b=source_b,
        source_c=source_c,
    )

    n_source_a = int(source_a.shape[0])
    rows_per_source_a = int(source_b.shape[0] * source_c.shape[0])
    source_a_index_per_row = torch.arange(n_source_a).repeat_interleave(
        rows_per_source_a
    )

    available_source_a_mask, pending_unlock_iteration, initial_source_a_indices = (
        _initialize_source_a_state(
            source_a_cardinality=n_source_a,
            initial_available_source_a_count=initial_available_source_a_count,
            seed_initial_available=seed_initial_available,
        )
    )

    initial_available_row_mask, initial_unordered_row_mask = (
        _row_masks_from_source_a_state(
            source_a_index_per_row=source_a_index_per_row,
            available_source_a_mask=available_source_a_mask,
            pending_unlock_iteration=pending_unlock_iteration,
        )
    )
    x_available_initial_cpu = x_allowed_cpu[initial_available_row_mask]
    x_unordered_initial_cpu = x_allowed_cpu[initial_unordered_row_mask]

    train_x = generate_starting_set(
        bounds=bounds,
        x_allowed=x_available_initial_cpu.to(
            device=resolved_device, dtype=bounds.dtype
        ),
        seed_starting=seed_starting,
        n_starting=n_starting,
    )

    train_y = benchmark.value(train_x.detach().cpu()).to(
        dtype=torch.double, device=resolved_device
    )
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)
    train_x = train_x.to(device=resolved_device)

    bounds_for_optimizer = bounds.detach().cpu()

    model = _fit_model(train_x, train_y, bounds)
    acq_function = _build_acquisition(
        model, train_y.max(), q=q, qmc_samples=qmc_samples
    )

    def _emit_phase_status(
        completed: int,
        phase: str,
        best_value: float | None,
    ) -> None:
        safe_completed = max(0, min(completed, n_eval))
        best_text = "NA" if best_value is None else f"{best_value:.6f}"

        if run_state_path is not None and run_state_combination_index is not None:
            update_run_state(
                state_path=run_state_path,
                combination_index=run_state_combination_index,
                total_iterations=n_eval,
                status="running",
                current_iteration=safe_completed,
                best_f=best_value,
                active_phase=phase,
            )

        if verbose:
            print_progress(
                label=display_label,
                completed=safe_completed,
                total=n_eval,
                extra=(
                    f"phase={phase} | best={best_text} "
                    f"| pop={pop_size} | evals={n_evals_ga}"
                ),
            )

    _emit_phase_status(
        completed=0,
        phase="ready_for_iteration_1",
        best_value=float(train_y.max().item()),
    )

    best_fs: list[float] = []
    candidates_list: list[list[list[float]]] = []
    max_acq_values_list: list[float] = []
    acq_values_repeated_list: list[list[float]] = []
    wall_times_acqf_opt: list[float] = []
    wall_times_model_fit: list[float] = []
    wall_times_function_eval: list[float] = []

    selected_source_a_indices: list[Optional[int]] = []
    selected_source_a_acq_values: list[Optional[float]] = []
    selected_source_a_candidates: list[list[list[float]]] = []
    released_source_a_by_iteration: list[list[int]] = []
    available_source_counts_before_selection: list[int] = []
    available_source_counts_after_selection: list[int] = []
    pending_source_counts_before_selection: list[int] = []
    pending_source_counts_after_selection: list[int] = []

    for iteration in range(n_eval):
        _emit_phase_status(
            completed=iteration,
            phase=f"iteration_{iteration + 1}_acq_opt_start",
            best_value=float(train_y.max().item()),
        )

        released_now = _release_matured_source_a(
            available_source_a_mask=available_source_a_mask,
            pending_unlock_iteration=pending_unlock_iteration,
            current_iteration=iteration,
        )
        released_source_a_by_iteration.append(released_now)

        available_row_mask, unordered_row_mask = _row_masks_from_source_a_state(
            source_a_index_per_row=source_a_index_per_row,
            available_source_a_mask=available_source_a_mask,
            pending_unlock_iteration=pending_unlock_iteration,
        )
        x_available_for_optimizer = x_allowed_cpu[available_row_mask]
        x_unordered_for_optimizer = x_allowed_cpu[unordered_row_mask]

        available_source_counts_before_selection.append(
            int(available_source_a_mask.sum().item())
        )
        pending_source_counts_before_selection.append(
            int((pending_unlock_iteration >= 0).sum().item())
        )

        candidates_repeated: list[torch.Tensor] = []
        acq_values_repeated: list[float] = []

        t_start_acq = time.perf_counter()
        for rep_idx in range(n_optimize_acqf):
            _emit_phase_status(
                completed=iteration,
                phase=(
                    f"iteration_{iteration + 1}_acq_opt_repetition_"
                    f"{rep_idx + 1}_start"
                ),
                best_value=float(train_y.max().item()),
            )

            def _main_ga_progress(message: str, rep_num: int = rep_idx + 1) -> None:
                _emit_phase_status(
                    completed=iteration,
                    phase=(
                        f"iteration_{iteration + 1}_acq_opt_repetition_"
                        f"{rep_num}_{message}"
                    ),
                    best_value=float(train_y.max().item()),
                )

            cat_avail, sol_avail, bas_avail = decompose_x_allowed_to_sources(
                x_available_for_optimizer
            )
            candidates, _ = optimize_acqf_mixed_GA(
                acq_function=copy.deepcopy(acq_function),
                bounds=bounds_for_optimizer,
                q=q,
                n_evals_ga=n_evals_ga,
                pop_size=pop_size,
                cat_emb_avail=cat_avail,
                sol_emb_avail=sol_avail,
                bas_emb_avail=bas_avail,
                batch_constraints=active_batch_constraints,
                fixed_dims=fixed_dims,
                progress_callback=_main_ga_progress,
            )
            candidates = candidates.to(dtype=train_x.dtype, device=train_x.device)
            acq_value = get_joint_log_ei(model, train_y.max(), candidates)
            candidates_repeated.append(candidates)
            acq_values_repeated.append(_scalarize(acq_value))

        t_end_acq = time.perf_counter()
        wall_times_acqf_opt.append((t_end_acq - t_start_acq) / n_optimize_acqf)

        best_rep_idx = int(torch.tensor(acq_values_repeated).argmax().item())
        chosen_candidates = candidates_repeated[best_rep_idx]
        chosen_acq_value = acq_values_repeated[best_rep_idx]

        _emit_phase_status(
            completed=iteration,
            phase=f"iteration_{iteration + 1}_objective_evaluation",
            best_value=float(train_y.max().item()),
        )

        t_start_eval = time.perf_counter()
        candidate_values = benchmark.value(chosen_candidates.detach().cpu()).to(
            dtype=torch.double, device=resolved_device
        )
        t_end_eval = time.perf_counter()
        wall_times_function_eval.append(t_end_eval - t_start_eval)

        if candidate_values.ndim == 1:
            candidate_values = candidate_values.unsqueeze(-1)

        train_x = torch.cat([train_x, chosen_candidates], dim=0)
        train_y = torch.cat([train_y, candidate_values], dim=0)

        _emit_phase_status(
            completed=iteration,
            phase=f"iteration_{iteration + 1}_model_refit",
            best_value=float(train_y.max().item()),
        )

        t_start_fit = time.perf_counter()
        model = _fit_model(train_x, train_y, bounds)
        t_end_fit = time.perf_counter()
        wall_times_model_fit.append(t_end_fit - t_start_fit)

        acq_function = _build_acquisition(
            model=model,
            best_f=train_y.max(),
            q=q,
            qmc_samples=qmc_samples,
        )

        selected_source_idx: Optional[int] = None
        selected_source_acq: Optional[float] = None
        selected_source_candidate: list[list[float]] = []

        if x_unordered_for_optimizer.shape[0] > 0:
            _emit_phase_status(
                completed=iteration,
                phase=f"iteration_{iteration + 1}_expand_optimizer_start",
                best_value=float(train_y.max().item()),
            )

            def _expand_ga_progress(message: str) -> None:
                _emit_phase_status(
                    completed=iteration,
                    phase=f"iteration_{iteration + 1}_expand_optimizer_{message}",
                    best_value=float(train_y.max().item()),
                )

            if normalized_strategy == "fantasized":
                x_ordered = _build_x_ordered_for_fantasizing(
                    x_allowed_cpu=x_allowed_cpu,
                    source_a_index_per_row=source_a_index_per_row,
                    pending_unlock_iteration=pending_unlock_iteration,
                    current_iteration=iteration,
                    lookahead_horizon=lookahead_horizon,
                )
                scope_candidates, scope_acq_value = (
                    optimize_plan_ahead_qlogEI_fantasized(
                        model=model,
                        q_batch=q,
                        q_expand=q_expand,
                        x_available=x_available_for_optimizer,
                        x_ordered=x_ordered,
                        x_unordered=x_unordered_for_optimizer,
                        bounds=bounds_for_optimizer,
                        k=lookahead_horizon,
                        num_fantasies=active_num_fantasies,
                        n_samples=n_evals_ga,
                        popsize=pop_size,
                        batch_constraints=None,
                        fixed_dims=fixed_dims,
                        progress_callback=_expand_ga_progress,
                        operating_mode=resolved_fantasized_mode,
                        decompose_allowed_set=decompose_x_allowed_to_sources,
                    )
                )
            else:
                scope_candidates, scope_acq_value = optimize_plan_ahead_qlogEI_myopic(
                    model=model,
                    q_expand=q_expand,
                    x_unordered=x_unordered_for_optimizer,
                    bounds=bounds_for_optimizer,
                    n_samples=n_evals_ga,
                    popsize=pop_size,
                    batch_constraints=None,
                    fixed_dims=fixed_dims,
                    progress_callback=_expand_ga_progress,
                    decompose_allowed_set=decompose_x_allowed_to_sources,
                )

            scope_candidates_cpu = scope_candidates.detach().cpu()
            selected_source_idx = _extract_source_a_index_from_candidate(
                candidate=scope_candidates_cpu,
                source_a=source_a,
            )
            selected_source_acq = _scalarize(scope_acq_value)
            selected_source_candidate = scope_candidates_cpu.tolist()

            if bool(available_source_a_mask[selected_source_idx]):
                raise ValueError("Selected source-A index is already available")
            if int(pending_unlock_iteration[selected_source_idx].item()) >= 0:
                raise ValueError("Selected source-A index is already pending unlock")

            pending_unlock_iteration[selected_source_idx] = (
                iteration + source_a_availability_delay
            )

        selected_source_a_indices.append(selected_source_idx)
        selected_source_a_acq_values.append(selected_source_acq)
        selected_source_a_candidates.append(selected_source_candidate)
        available_source_counts_after_selection.append(
            int(available_source_a_mask.sum().item())
        )
        pending_source_counts_after_selection.append(
            int((pending_unlock_iteration >= 0).sum().item())
        )

        best_fs.append(float(train_y.max().item()))
        candidates_list.append(chosen_candidates.detach().cpu().tolist())
        max_acq_values_list.append(chosen_acq_value)
        acq_values_repeated_list.append(acq_values_repeated)

        _emit_phase_status(
            completed=iteration + 1,
            phase=f"iteration_{iteration + 1}_complete",
            best_value=best_fs[-1],
        )

    return {
        "benchmark": benchmark_name,
        "seed_discrete": seed_discrete,
        "seed_starting": seed_starting,
        "seed_initial_available": seed_initial_available,
        "strategy": normalized_strategy,
        "operating_mode": resolved_fantasized_mode,
        "ga_ratio": ga_ratio,
        "pop_size": pop_size,
        "n_evals_ga": n_evals_ga,
        "q": q,
        "q_expand": q_expand,
        "n_eval": n_eval,
        "n_optimize_acqf": n_optimize_acqf,
        "batch_constraints": active_batch_constraints,
        "fixed_dims": dict(fixed_dims) if fixed_dims is not None else None,
        "n_starting": n_starting,
        "initial_available_source_a_count": initial_available_source_a_count,
        "source_a_availability_delay": source_a_availability_delay,
        "lookahead_horizon": lookahead_horizon,
        "num_fantasies": active_num_fantasies,
        "source_a_cardinality": n_source_a,
        "initial_available_source_a_indices": [
            int(i) for i in initial_source_a_indices.tolist()
        ],
        "x_allowed_shape": [int(x_allowed_cpu.shape[0]), int(x_allowed_cpu.shape[1])],
        "x_available_initial_shape": [
            int(x_available_initial_cpu.shape[0]),
            int(x_available_initial_cpu.shape[1]),
        ],
        "x_unordered_initial_shape": [
            int(x_unordered_initial_cpu.shape[0]),
            int(x_unordered_initial_cpu.shape[1]),
        ],
        "best_fs": best_fs,
        "candidates_list": candidates_list,
        "max_acq_values_list": max_acq_values_list,
        "acq_values_repeated_list": acq_values_repeated_list,
        "selected_source_a_indices": selected_source_a_indices,
        "selected_source_a_acq_values": selected_source_a_acq_values,
        "selected_source_a_candidates": selected_source_a_candidates,
        "released_source_a_by_iteration": released_source_a_by_iteration,
        "available_source_counts_before_selection": available_source_counts_before_selection,
        "available_source_counts_after_selection": available_source_counts_after_selection,
        "pending_source_counts_before_selection": pending_source_counts_before_selection,
        "pending_source_counts_after_selection": pending_source_counts_after_selection,
        "wall_times_acqf_opt": wall_times_acqf_opt,
        "wall_times_model_fit": wall_times_model_fit,
        "wall_times_function_eval": wall_times_function_eval,
    }
