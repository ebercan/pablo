"""Latent-table synthetic setting used by ``inner_optimizer_harness``.

A 26-dimensional mixed space: 4 continuous dims followed by three discrete
blocks (catalyst-like 18 dims × 161 rows, solvent-like 2 × 14, base-like 2 × 8)
whose rows are drawn uniformly within the benchmark bounds from one seed. The
harness measures inner batch optimizers on fantasy nodes built in this setting;
the run_5 sweep itself runs on the NHC space of ``nhc_problem``.
"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from pablo.mechthild_runs.run_5_scope_extension_v2.config import BENCHMARK_CLASS_MAP

N_CONTINUOUS_DIMS = 4
N_DISCRETE_DIMS = 22
TOTAL_DIM = N_CONTINUOUS_DIMS + N_DISCRETE_DIMS
DEPENDENT_BLOCK_SIZES = (18, 2, 2)
DEPENDENT_SOURCE_CARDINALITIES = (161, 14, 8)
INITIAL_AVAILABLE_SOURCE_A_COUNT = 22
N_STARTING_POINTS = 2 * TOTAL_DIM
DEFAULT_BATCH_CONSTRAINTS = {0: (2, 2, 2)}
FIXED_DIM_INDEX = 3
FIXED_DIM_FRACTION = 0.5


def _sample_uniform_block(
    lower: torch.Tensor,
    upper: torch.Tensor,
    n_rows: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``(n_rows, d_block)`` uniformly within per-dimension bounds."""
    span = upper - lower
    return lower.unsqueeze(0) + span.unsqueeze(0) * torch.rand(
        (n_rows, lower.shape[0]), dtype=lower.dtype, generator=generator
    )


def build_benchmark(benchmark_name: str):
    """Build a benchmark instance in ``TOTAL_DIM`` dimensions.

    Args:
        benchmark_name: Key from ``BENCHMARK_CLASS_MAP``.

    Returns:
        A synthetic benchmark object.

    Raises:
        ValueError: If ``benchmark_name`` is unknown.
    """
    if benchmark_name not in BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    return BENCHMARK_CLASS_MAP[benchmark_name](d=TOTAL_DIM)


def generate_dependent_choice_sources(
    bounds: torch.Tensor,
    seed_discrete: int,
    source_cardinalities: tuple[int, int, int] = DEPENDENT_SOURCE_CARDINALITIES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the three latent source tables for the discrete blocks.

    The discrete block of ``N_DISCRETE_DIMS`` dims is split by
    ``DEPENDENT_BLOCK_SIZES`` into a catalyst-like block A, a solvent-like
    block B and a base-like block C, each driven by one latent source with
    the given cardinality.

    Args:
        bounds: ``(2, TOTAL_DIM)`` bounds tensor.
        seed_discrete: Seed controlling all source draws.
        source_cardinalities: ``(n_a, n_b, n_c)`` row counts.

    Returns:
        ``(source_a, source_b, source_c)`` with shapes ``(n_a, 18)``,
        ``(n_b, 2)``, ``(n_c, 2)``.

    Raises:
        ValueError: If bounds or cardinalities do not match required shapes.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] != TOTAL_DIM:
        raise ValueError(f"bounds must have shape (2, {TOTAL_DIM}).")
    if len(source_cardinalities) != 3:
        raise ValueError("source_cardinalities must have length 3.")
    if sum(DEPENDENT_BLOCK_SIZES) != N_DISCRETE_DIMS:
        raise ValueError("Dependent block sizes do not sum to N_DISCRETE_DIMS.")

    n_a, n_b, n_c = source_cardinalities
    if n_a <= 0 or n_b <= 0 or n_c <= 0:
        raise ValueError("All source cardinalities must be positive integers.")

    bounds_cpu = bounds.detach().cpu().to(torch.double)
    discrete_bounds = bounds_cpu[:, N_CONTINUOUS_DIMS:]
    split_a = DEPENDENT_BLOCK_SIZES[0]
    split_b = split_a + DEPENDENT_BLOCK_SIZES[1]

    generator = torch.Generator()
    generator.manual_seed(seed_discrete)
    source_a = _sample_uniform_block(
        discrete_bounds[0, :split_a], discrete_bounds[1, :split_a], n_a, generator
    )
    source_b = _sample_uniform_block(
        discrete_bounds[0, split_a:split_b],
        discrete_bounds[1, split_a:split_b],
        n_b,
        generator,
    )
    source_c = _sample_uniform_block(
        discrete_bounds[0, split_b:], discrete_bounds[1, split_b:], n_c, generator
    )
    return source_a, source_b, source_c


def build_x_allowed_from_sources(
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    source_c: torch.Tensor,
) -> torch.Tensor:
    """Create X_allowed as the cartesian composition of three source tables.

    Rows are ordered with source A slowest and source C fastest, so row ``r``
    belongs to source-A index ``r // (n_b * n_c)``.

    Args:
        source_a: ``(n_a, 18)`` tensor.
        source_b: ``(n_b, 2)`` tensor.
        source_c: ``(n_c, 2)`` tensor.

    Returns:
        ``(n_a * n_b * n_c, N_DISCRETE_DIMS)`` tensor.

    Raises:
        ValueError: If source widths do not match ``DEPENDENT_BLOCK_SIZES``.
    """
    widths = (source_a.shape[1], source_b.shape[1], source_c.shape[1])
    if widths != DEPENDENT_BLOCK_SIZES:
        raise ValueError(
            f"Source widths {widths} must equal DEPENDENT_BLOCK_SIZES {DEPENDENT_BLOCK_SIZES}."
        )
    n_a, n_b, n_c = source_a.shape[0], source_b.shape[0], source_c.shape[0]
    block_a = source_a.repeat_interleave(n_b * n_c, dim=0)
    block_b = source_b.repeat_interleave(n_c, dim=0).repeat(n_a, 1)
    block_c = source_c.repeat(n_b, 1).repeat(n_a, 1)
    return torch.cat([block_a, block_b, block_c], dim=1)


def decompose_x_allowed_to_sources(
    x_subset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an ``(n, N_DISCRETE_DIMS)`` X_allowed subset into unique source rows.

    Args:
        x_subset: ``(n, 22)`` tensor with columns ``[A (18) | B (2) | C (2)]``.

    Returns:
        ``(cat_emb, sol_emb, bas_emb)``: the unique rows of each block, shapes
        ``(n_a, 18)``, ``(n_b, 2)`` and ``(n_c, 2)``, in the sorted order of
        ``torch.unique``.
    """
    split_a = DEPENDENT_BLOCK_SIZES[0]
    split_b = split_a + DEPENDENT_BLOCK_SIZES[1]
    cat_emb = x_subset[:, :split_a].unique(dim=0)
    sol_emb = x_subset[:, split_a:split_b].unique(dim=0)
    bas_emb = x_subset[:, split_b:].unique(dim=0)
    return cat_emb, sol_emb, bas_emb


def generate_starting_set(
    bounds: torch.Tensor,
    x_allowed: torch.Tensor,
    seed_starting: int,
    n_starting: int = N_STARTING_POINTS,
    fixed_dims: dict[int, float] | None = None,
) -> torch.Tensor:
    """Generate the initial design from allowed discrete rows.

    Args:
        bounds: ``(2, TOTAL_DIM)`` tensor.
        x_allowed: ``(n_allowed, N_DISCRETE_DIMS)`` tensor.
        seed_starting: Seed for the row and continuous draws.
        n_starting: Number of starting points.
        fixed_dims: Continuous dims pinned to constants in the design too.

    Returns:
        ``(n_starting, TOTAL_DIM)`` tensor.

    Raises:
        ValueError: If shapes are inconsistent or ``n_starting`` is invalid.
    """
    if bounds.ndim != 2 or bounds.shape != (2, TOTAL_DIM):
        raise ValueError(f"bounds must have shape (2, {TOTAL_DIM}).")
    if x_allowed.ndim != 2 or x_allowed.shape[1] != N_DISCRETE_DIMS:
        raise ValueError(f"x_allowed must have shape (n_allowed, {N_DISCRETE_DIMS}).")
    if n_starting <= 0:
        raise ValueError("n_starting must be positive.")
    if n_starting > x_allowed.shape[0]:
        raise ValueError("n_starting cannot exceed number of allowed discrete rows.")

    generator = torch.Generator()
    generator.manual_seed(seed_starting)
    indices = torch.randperm(x_allowed.shape[0], generator=generator)[:n_starting]
    x_discrete = x_allowed[indices]

    cont_lower = bounds[0, :N_CONTINUOUS_DIMS].detach().cpu()
    cont_upper = bounds[1, :N_CONTINUOUS_DIMS].detach().cpu()
    x_continuous = cont_lower.unsqueeze(0) + (cont_upper - cont_lower).unsqueeze(
        0
    ) * torch.rand((n_starting, N_CONTINUOUS_DIMS), dtype=bounds.dtype, generator=generator)
    for dim_idx, value in (fixed_dims or {}).items():
        x_continuous[:, dim_idx] = value
    x_continuous = x_continuous.to(device=bounds.device, dtype=bounds.dtype)
    return torch.cat([x_continuous, x_discrete.to(x_continuous)], dim=1)


def _fit_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
) -> SingleTaskGP:
    """Fit a SingleTaskGP with Normalize/Standardize transforms.

    Args:
        train_x: ``(n, TOTAL_DIM)`` tensor.
        train_y: ``(n, 1)`` tensor.
        bounds: ``(2, TOTAL_DIM)`` tensor.

    Returns:
        Fitted model.
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



def _initialize_source_a_state(
    source_a_cardinality: int,
    initial_available_source_a_count: int,
    seed_initial_available: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize the available mask, the pending-unlock vector and the initial indices."""
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
    pending_unlock_iteration = torch.full((source_a_cardinality,), -1, dtype=torch.long)
    return available_mask, pending_unlock_iteration, sampled
