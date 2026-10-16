"""Utilities for generating starting sets for mixed-variable Bayesian optimization.

Module: pablo.utils.starting_set_generation
"""

import random

import torch

from benchmark_problems.synthetic import Ackley


def generate_starting_set(
    bounds,
    n_starting,
    dim,
    rand_state: int | None = None,
):
    """Sample continuous points uniformly within bounds.

    Seeds torch's global RNG with ``rand_state``, or with a random integer in
    [1, 1000] drawn from Python's ``random`` when it is ``None``.

    Args:
        bounds: Tensor of shape (2, dim) with lower and upper bounds.
        n_starting: Number of points to sample.
        dim: Problem dimensionality.
        rand_state: Seed for torch's global RNG.

    Returns:
        Tensor of shape (1, n_starting, dim).
    """
    if rand_state is None:
        rand_state = random.randint(1, 1000)
    torch.manual_seed(rand_state)

    starting_set = bounds[0].reshape(1, 1, dim) + (
        bounds[1].reshape(1, 1, dim) - bounds[0].reshape(1, 1, dim)
    ) * torch.rand((n_starting, dim))

    return starting_set


def generate_uniformly_distributed_starting_set(
    bounds: torch.Tensor,
    batch_size: int,
    n_starting: int,
    d: int,
    rand_state: int | None = None,
) -> torch.Tensor:
    """Generate uniformly sampled starting sets for benchmark batches.

    Args:
        bounds: Tensor of shape (2, d) containing lower and upper bounds.
        batch_size: Number of independent starting-set batches.
        n_starting: Number of starting points per batch.
        d: Problem dimensionality.
        rand_state: Optional random seed for deterministic generation.

    Returns:
        Tensor of shape (batch_size, n_starting, d).

    Raises:
        ValueError: If inputs are malformed or dimensions are inconsistent.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(
            "bounds must be a 2D tensor with shape (2, d), containing lower and upper bounds."
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(n_starting, int) or n_starting <= 0:
        raise ValueError("n_starting must be a positive integer.")
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer.")
    if d != int(bounds.shape[1]):
        raise ValueError(
            f"Dimension mismatch: d={d}, but bounds defines dimension {bounds.shape[1]}."
        )

    flat_starting_set = generate_starting_set(
        bounds=bounds,
        n_starting=batch_size * n_starting,
        dim=d,
        rand_state=rand_state,
    )
    return flat_starting_set.reshape(batch_size, n_starting, d)


def generate_discrete_points(
    bounds: torch.Tensor,  # shape 2 x d_discrete, not d!
    n_discrete: int = 8,
    rand_state: int | None = None,
    always_pick: float | None = None,
) -> list[list[float]]:
    """Sample candidate values for each discrete dimension uniformly within bounds.

    The values are drawn from a ``random.Random`` seeded with ``rand_state``,
    so a fixed ``rand_state`` gives the same output. ``rand_state`` also seeds
    torch's global RNG.

    Args:
        bounds: Tensor of shape (2, d_discrete) with lower and upper bounds of
            the discrete dimensions only.
        n_discrete: Number of candidate values per dimension.
        rand_state: Seed for the sampler and for torch's global RNG; a random
            integer in [1, 1000] when ``None``.
        always_pick: If given, replaces the last candidate of every dimension.

    Returns:
        List of length d_discrete; each entry holds n_discrete values. Sampled
        values are 0-d tensors with the dtype of ``bounds``; ``always_pick``
        stays a float.
    """
    if rand_state is None:
        rand_state = random.randint(1, 1000)
    torch.manual_seed(rand_state)
    rng = random.Random(rand_state)
    d_discrete = bounds.shape[-1]
    discrete_points_list = []

    for d in range(d_discrete):
        discrete_points = [
            rng.uniform(bounds[0, d], bounds[1, d]) for _ in range(n_discrete)
        ]
        if always_pick is not None:
            discrete_points[-1] = always_pick
        discrete_points_list.append(discrete_points)
    return discrete_points_list


def generate_starting_set_mixed(
    bounds: torch.Tensor,
    discrete_points: list[list[float]],
    n_starting: int,
    rand_state: int | None = None,
):
    """Sample starting points for a mixed continuous-discrete search space.

    The continuous dimensions come first and are sampled uniformly within
    bounds; the last ``len(discrete_points)`` dimensions are discrete, each
    drawn uniformly from its candidate values. Seeds torch's global RNG with
    ``rand_state``, or with a random integer in [1, 1000] when it is ``None``.

    Args:
        bounds: Tensor of shape (2, d) with lower and upper bounds of all
            dimensions; the discrete columns are not used.
        discrete_points: One list of candidate values per discrete dimension.
        n_starting: Number of points to sample.
        rand_state: Seed for torch's global RNG.

    Returns:
        Tensor of shape (n_starting, d) with the dtype and device of ``bounds``.
    """
    if rand_state is None:
        rand_state = random.randint(1, 1000)
    torch.manual_seed(rand_state)
    d = bounds.shape[1]
    d_discrete = len(discrete_points)
    d_continuous = d - d_discrete

    lower = bounds[0, :d_continuous]
    upper = bounds[1, :d_continuous]

    x_continuous = lower + (upper - lower) * torch.rand(
        n_starting, d_continuous, dtype=bounds.dtype, device=bounds.device
    )

    x_discrete_list = []
    for points in discrete_points:
        vals = torch.tensor(points, dtype=bounds.dtype, device=bounds.device)
        sample_idxs = torch.randint(0, len(vals), (n_starting,), device=bounds.device)
        x_discrete_list.append(vals[sample_idxs].unsqueeze(1))
    x_discrete = torch.cat(x_discrete_list, dim=1)

    return torch.cat([x_continuous, x_discrete], dim=1)


testfun = Ackley()

disc_points = [[1, 2, 3, 5], [-1, 0, -2]]
