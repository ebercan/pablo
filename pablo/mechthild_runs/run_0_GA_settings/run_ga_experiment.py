"""Core BO execution for the run_0 GA settings benchmark sweep."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
from pablo.optimize import optimize_acqf_mixed_GA
from pablo.utils.acq_utils import get_joint_log_ei
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from pablo.mechthild_runs.run_0_GA_settings.config import (
    ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES,
    BENCHMARK_CLASS_MAP,
    DEFAULT_BATCH_CONSTRAINTS,
    DEPENDENT_BLOCK_SIZES,
    DEPENDENT_SOURCE_CARDINALITIES,
    DEVICE_CHOICES,
    N_BO_BATCHES,
    N_CONTINUOUS_DIMS,
    N_DISCRETE_DIMS,
    N_OPTIMIZE_ACQF_REPETITIONS,
    N_STARTING_POINTS,
    Q_BATCH_SIZE,
    QMC_SAMPLES,
    TOTAL_DIM,
)
from pablo.mechthild_runs.run_0_GA_settings.run_state import update_run_state


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
    """Build a benchmark instance in 33D.

    Args:
        benchmark_name: Key from BENCHMARK_CLASS_MAP.

    Returns:
        A synthetic benchmark object with total dimension 33.

    Raises:
        ValueError: If benchmark_name is unknown.
    """
    if benchmark_name not in BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    benchmark_class = BENCHMARK_CLASS_MAP[benchmark_name]
    return benchmark_class(d=TOTAL_DIM)


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve a user-facing device option into a torch.device.

    Args:
        requested_device: Device selector string: cpu, cuda, or auto.

    Returns:
        Concrete torch.device used by the BO run.

    Raises:
        ValueError: If requested_device is unknown or CUDA is unavailable.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested, but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")

    raise ValueError("device must be one of: cpu, cuda, auto")


def _parse_supported_sm_arches(arch_list: list[str]) -> list[str]:
    """Extract sorted SM architecture identifiers from torch arch list.

    Args:
        arch_list: Raw output from torch.cuda.get_arch_list().

    Returns:
        Sorted unique list of entries like sm_60, sm_70, sm_80.
    """
    sm_arches = {entry.strip() for entry in arch_list if entry.startswith("sm_")}
    return sorted(sm_arches)


def _format_sm_from_capability(capability: tuple[int, int]) -> str:
    """Format a CUDA capability pair into torch SM string form.

    Args:
        capability: CUDA capability tuple (major, minor).

    Returns:
        SM architecture string, e.g. sm_60.
    """
    return f"sm_{int(capability[0])}{int(capability[1])}"


def get_cuda_runtime_metadata() -> dict[str, object]:
    """Collect CUDA compatibility metadata from the active torch runtime.

    Returns:
        Metadata dictionary with availability, device properties, and arch support.
    """
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    arch_list = list(torch.cuda.get_arch_list()) if cuda_available else []
    supported_sm_arches = _parse_supported_sm_arches(arch_list)

    device_index = 0
    device_name = ""
    capability_major = -1
    capability_minor = -1
    device_sm = ""
    arch_supported = False
    compatibility_status = "cuda_unavailable"

    if cuda_available and device_count > 0:
        device_index = int(torch.cuda.current_device())
        device_name = str(torch.cuda.get_device_name(device_index))
        capability = torch.cuda.get_device_capability(device_index)
        capability_major = int(capability[0])
        capability_minor = int(capability[1])
        device_sm = _format_sm_from_capability((capability_major, capability_minor))
        if supported_sm_arches:
            arch_supported = device_sm in supported_sm_arches
        else:
            arch_supported = True
        compatibility_status = (
            "cuda_arch_supported"
            if arch_supported
            else "cuda_arch_not_supported_by_torch_build"
        )

    return {
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_index": device_index,
        "cuda_device_name": device_name,
        "cuda_capability_major": capability_major,
        "cuda_capability_minor": capability_minor,
        "cuda_device_sm": device_sm,
        "torch_cuda_version": str(torch.version.cuda),
        "torch_supported_sm_arches": supported_sm_arches,
        "cuda_arch_supported": arch_supported,
        "compatibility_status": compatibility_status,
    }


def _run_cuda_smoke_test() -> float:
    """Execute a minimal CUDA kernel path and synchronize.

    Returns:
        Deterministic float summary from the probe operation.
    """
    left = torch.ones((64, 64), device="cuda", dtype=torch.float32)
    right = torch.full((64, 64), 2.0, device="cuda", dtype=torch.float32)
    value = float((left @ right).sum().item())
    torch.cuda.synchronize()
    return value


def resolve_execution_device_policy(
    requested_device: str,
    allow_cpu_fallback: bool = ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES,
    perform_cuda_smoke_test: bool = True,
) -> dict[str, object]:
    """Resolve requested device into an execution policy with fallback metadata.

    Args:
        requested_device: Device selector string: cpu, cuda, or auto.
        allow_cpu_fallback: Whether to fall back to CPU for CUDA issues.
        perform_cuda_smoke_test: Whether to run a minimal CUDA compute probe.

    Returns:
        Policy dictionary with requested/actual device and compatibility details.

    Raises:
        ValueError: If requested_device is unknown or fallback is disallowed.
    """
    if requested_device not in DEVICE_CHOICES:
        choices = ", ".join(DEVICE_CHOICES)
        raise ValueError(f"device must be one of: {choices}")

    policy: dict[str, object] = {
        "requested_device": requested_device,
        "actual_device": "cpu",
        "allow_cpu_fallback": bool(allow_cpu_fallback),
        "fallback_used": False,
        "fallback_reason": None,
        "compatibility_status": "cpu_requested",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_index": 0,
        "cuda_device_name": "",
        "cuda_capability_major": -1,
        "cuda_capability_minor": -1,
        "cuda_device_sm": "",
        "torch_cuda_version": str(torch.version.cuda),
        "torch_supported_sm_arches": [],
        "cuda_arch_supported": False,
        "cuda_smoke_test_passed": False,
        "cuda_smoke_test_value": None,
    }

    if requested_device == "cpu":
        return policy

    cuda_meta = get_cuda_runtime_metadata()
    policy.update(cuda_meta)

    wants_cuda = requested_device == "cuda" or (
        requested_device == "auto" and bool(cuda_meta["cuda_available"])
    )

    if not wants_cuda:
        policy["compatibility_status"] = "auto_cpu_no_cuda"
        return policy

    cuda_available = bool(cuda_meta["cuda_available"])
    cuda_count = int(cuda_meta["cuda_device_count"])
    cuda_arch_supported = bool(cuda_meta["cuda_arch_supported"])

    if (not cuda_available) or cuda_count <= 0:
        if allow_cpu_fallback:
            policy["fallback_used"] = True
            policy["fallback_reason"] = "cuda_not_available"
            policy["compatibility_status"] = "fallback_cuda_not_available"
            return policy
        raise ValueError("CUDA requested but no usable CUDA device is available")

    if not cuda_arch_supported:
        if allow_cpu_fallback:
            policy["fallback_used"] = True
            policy["fallback_reason"] = "cuda_arch_not_supported_by_torch_build"
            policy["compatibility_status"] = "fallback_cuda_arch_not_supported"
            return policy
        raise ValueError(
            "CUDA device architecture is not supported by the installed torch build"
        )

    policy["actual_device"] = "cuda"
    policy["compatibility_status"] = "cuda_arch_supported"

    if perform_cuda_smoke_test:
        smoke_value = _run_cuda_smoke_test()
        policy["cuda_smoke_test_passed"] = True
        policy["cuda_smoke_test_value"] = smoke_value
        policy["compatibility_status"] = "cuda_runtime_ready"

    return policy


def generate_dependent_choice_sources(
    bounds: torch.Tensor,
    seed_discrete: int,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dependent source tensors for the three discrete blocks.

    The full search space has 33 dimensions with 3 continuous and 30 discrete.
    The 30 discrete dimensions are split into dependent blocks:
    - block A: 26 dimensions driven by one latent source with 225 choices.
    - block B: 2 dimensions driven by one latent source with 14 choices.
    - block C: 2 dimensions driven by one latent source with 8 choices.

    Args:
        bounds: Tensor with shape (2, 33), row 0 lower and row 1 upper bounds.
        seed_discrete: Seed controlling all random draws for discrete source tensors.
        source_cardinalities: Tuple (n_a, n_b, n_c) for source row counts.

    Returns:
        Tuple (source_a, source_b, source_c) with shapes:
        - source_a: (n_a, 26)
        - source_b: (n_b, 2)
        - source_c: (n_c, 2)

    Raises:
        ValueError: If bounds or cardinalities do not match required shapes.
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
        source_a: Tensor with shape (n_a, 26).
        source_b: Tensor with shape (n_b, 2).
        source_c: Tensor with shape (n_c, 2).

    Returns:
        Tensor with shape (n_a * n_b * n_c, 30).

    Raises:
        ValueError: If source shapes are not aligned with (26, 2, 2).
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


def build_x_allowed_tensor(
    bounds: torch.Tensor,
    seed_discrete: int,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
) -> torch.Tensor:
    """Generate a full X_allowed tensor for one benchmark and seed.

    Args:
        bounds: Tensor with shape (2, 33).
        seed_discrete: Seed for generating dependent source tensors.
        source_cardinalities: Tuple (n_a, n_b, n_c).

    Returns:
        Tensor with shape (n_a * n_b * n_c, 30).
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

    Args:
        bounds: Tensor with shape (2, 33).
        x_allowed: Tensor with shape (n_allowed, 30).
        seed_starting: Seed from seed_list_2.
        n_starting: Number of starting points.

    Returns:
        Tensor with shape (n_starting, 33).

    Raises:
        ValueError: If shapes are inconsistent or n_starting is invalid.
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
        train_x: Tensor with shape (n, 33).
        train_y: Tensor with shape (n, 1).
        bounds: Tensor with shape (2, 33).

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


def run_bo_ga_experiment(
    benchmark_name: str,
    seed_discrete: int,
    seed_starting: int,
    pop_size: int,
    ga_ratio: int,
    n_evals_ga: int | None = None,
    q: int = Q_BATCH_SIZE,
    n_eval: int = N_BO_BATCHES,
    n_optimize_acqf: int = N_OPTIMIZE_ACQF_REPETITIONS,
    n_starting: int = N_STARTING_POINTS,
    batch_constraints: dict[int, int] | None = None,
    qmc_samples: int = QMC_SAMPLES,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
    run_state_path: Path | None = None,
    run_state_combination_index: int | None = None,
    verbose: bool = False,
    progress_label: str | None = None,
    device: str = "cpu",
    allow_cpu_fallback_on_cuda_issues: bool = False,
    prevalidated_device_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    """Execute one BO run for a single benchmark/seed/GA-setting combination.

    Args:
        benchmark_name: Name of benchmark to optimize.
        seed_discrete: Seed for generating dependent discrete sources.
        seed_starting: Seed for selecting starting points.
        pop_size: GA population size.
        ga_ratio: Ratio for computing n_evals_ga = ga_ratio * pop_size.
        n_evals_ga: Optional direct GA eval budget override.
        q: BO batch size.
        n_eval: Number of BO iterations (batches).
        n_optimize_acqf: Number of repeated acquisition optimizations per iteration.
        n_starting: Starting set size.
        batch_constraints: Dimension->k mapping for q-batch distinct-value constraints.
        qmc_samples: QMC sample count for qLogExpectedImprovement.
        source_cardinalities: Tuple (n_a, n_b, n_c) controlling X_allowed size.
        run_state_path: Optional path to run-state JSON file for progress updates.
        run_state_combination_index: Combination index corresponding to run_state_path.
        verbose: If True, print per-iteration progress.
        progress_label: Optional label shown in terminal progress output.
        device: Execution device selector: cpu, cuda, or auto.
        allow_cpu_fallback_on_cuda_issues: Whether CUDA issues may fall back to CPU.
        prevalidated_device_policy: Optional precomputed policy metadata.

    Returns:
        Dictionary with the run settings (``benchmark``, seeds, GA and batch
        settings, ``batch_constraints``, device fields, ``x_allowed_shape`` as
        ``[n_allowed, d]``) and per-iteration traces, each of length
        ``n_eval``:

        - ``best_fs``: best observed value after each iteration.
        - ``candidates_list``: chosen batch per iteration, a ``q x d`` nested list.
        - ``max_acq_values_list``: joint log-EI of the chosen batch.
        - ``acq_values_repeated_list``: joint log-EI of all ``n_optimize_acqf``
          repeats.
        - ``wall_times_acqf_opt`` (mean per repeat), ``wall_times_model_fit``,
          ``wall_times_function_eval``: seconds.

    Raises:
        ValueError: If a batch constraint ``k`` exceeds ``q``, if
            ``benchmark_name`` is unknown, or, when no
            ``prevalidated_device_policy`` is given, if ``device`` is unknown
            or CUDA is requested but unusable without allowed CPU fallback.
    """
    if n_evals_ga is None:
        n_evals_ga = ga_ratio * pop_size

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

    benchmark = build_benchmark(benchmark_name)
    if prevalidated_device_policy is None:
        device_policy = resolve_execution_device_policy(
            requested_device=device,
            allow_cpu_fallback=allow_cpu_fallback_on_cuda_issues,
            perform_cuda_smoke_test=True,
        )
    else:
        device_policy = dict(prevalidated_device_policy)

    actual_device = str(device_policy.get("actual_device", "cpu"))
    resolved_device = torch.device(actual_device)
    bounds = benchmark.bounds().to(dtype=torch.double, device=resolved_device)

    source_a, source_b, source_c = generate_dependent_choice_sources(
        bounds=bounds,
        seed_discrete=seed_discrete,
        source_cardinalities=source_cardinalities,
    )
    x_allowed = build_x_allowed_from_sources(
        source_a=source_a, source_b=source_b, source_c=source_c
    ).to(dtype=bounds.dtype, device=bounds.device)

    train_x = generate_starting_set(
        bounds=bounds,
        x_allowed=x_allowed,
        seed_starting=seed_starting,
        n_starting=n_starting,
    )

    # Evaluate benchmark on CPU: synthetic functions create internal tensors
    # without specifying device, causing cuda/cpu mismatch when x is on GPU.
    train_y = benchmark.value(train_x.detach().cpu()).to(
        dtype=torch.double, device=resolved_device
    )
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)
    train_x = train_x.to(device=resolved_device)

    bounds_for_optimizer = bounds.detach().cpu()
    cat_emb_for_optimizer = source_a.detach().cpu()
    sol_emb_for_optimizer = source_b.detach().cpu()
    bas_emb_for_optimizer = source_c.detach().cpu()

    model = _fit_model(train_x, train_y, bounds)
    acq_function = _build_acquisition(
        model, train_y.max(), q=q, qmc_samples=qmc_samples
    )

    best_fs: list[float] = []
    candidates_list: list[list[list[float]]] = []
    max_acq_values_list: list[float] = []
    acq_values_repeated_list: list[list[float]] = []
    wall_times_acqf_opt: list[float] = []
    wall_times_model_fit: list[float] = []
    wall_times_function_eval: list[float] = []

    display_label = progress_label or (
        f"{benchmark_name} | s1={seed_discrete} | s2={seed_starting}"
    )

    for iteration in range(n_eval):
        candidates_repeated: list[torch.Tensor] = []
        acq_values_repeated: list[float] = []

        t_start_acq = time.perf_counter()
        for _ in range(n_optimize_acqf):
            candidates, _ = optimize_acqf_mixed_GA(
                acq_function=copy.deepcopy(acq_function),
                bounds=bounds_for_optimizer,
                q=q,
                n_evals_ga=n_evals_ga,
                pop_size=pop_size,
                cat_emb_avail=cat_emb_for_optimizer,
                sol_emb_avail=sol_emb_for_optimizer,
                bas_emb_avail=bas_emb_for_optimizer,
                batch_constraints=active_batch_constraints,
            )
            candidates = candidates.to(dtype=train_x.dtype, device=train_x.device)
            acq_value = get_joint_log_ei(model, train_y.max(), candidates)
            candidates_repeated.append(candidates)
            acq_scalar = (
                float(acq_value)
                if isinstance(acq_value, float)
                else float(acq_value.item())
            )
            acq_values_repeated.append(acq_scalar)

        t_end_acq = time.perf_counter()
        wall_times_acqf_opt.append((t_end_acq - t_start_acq) / n_optimize_acqf)

        best_rep_idx = int(torch.tensor(acq_values_repeated).argmax().item())
        chosen_candidates = candidates_repeated[best_rep_idx]
        chosen_acq_value = acq_values_repeated[best_rep_idx]

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

        best_fs.append(float(train_y.max().item()))
        candidates_list.append(chosen_candidates.detach().cpu().tolist())
        max_acq_values_list.append(chosen_acq_value)
        acq_values_repeated_list.append(acq_values_repeated)

        if run_state_path is not None and run_state_combination_index is not None:
            update_run_state(
                state_path=run_state_path,
                combination_index=run_state_combination_index,
                total_iterations=n_eval,
                status="running",
                current_iteration=iteration + 1,
                best_f=best_fs[-1],
            )

        if verbose:
            print_progress(
                label=display_label,
                completed=iteration + 1,
                total=n_eval,
                extra=(
                    f"best={best_fs[-1]:.6f} | pop={pop_size} | " f"evals={n_evals_ga}"
                ),
            )

    return {
        "benchmark": benchmark_name,
        "seed_discrete": seed_discrete,
        "seed_starting": seed_starting,
        "ga_ratio": ga_ratio,
        "pop_size": pop_size,
        "n_evals_ga": n_evals_ga,
        "q": q,
        "n_eval": n_eval,
        "n_optimize_acqf": n_optimize_acqf,
        "batch_constraints": active_batch_constraints,
        "n_starting": n_starting,
        "device": str(resolved_device),
        "requested_device": str(device_policy.get("requested_device", device)),
        "actual_device": str(device_policy.get("actual_device", str(resolved_device))),
        "device_policy": device_policy,
        "x_allowed_shape": [int(x_allowed.shape[0]), int(x_allowed.shape[1])],
        "best_fs": best_fs,
        "candidates_list": candidates_list,
        "max_acq_values_list": max_acq_values_list,
        "acq_values_repeated_list": acq_values_repeated_list,
        "wall_times_acqf_opt": wall_times_acqf_opt,
        "wall_times_model_fit": wall_times_model_fit,
        "wall_times_function_eval": wall_times_function_eval,
    }
