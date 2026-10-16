"""NHC space, synthetic yield and featurization-scan surrogate of run_5."""

import pytest
import torch

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    BASE_DIM,
    CAT_DIM,
    N_CONTINUOUS_DIMS,
    SOLVENT_DIM,
    TOTAL_DIM,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.nhc_problem import (
    SyntheticYield,
    fit_surrogate,
    load_nhc_space,
)
from pablo.optimize import GPPoolPosteriorCache, _ExactGPPoolPosterior


@pytest.fixture(scope="module")
def space():
    return load_nhc_space()


@pytest.fixture(scope="module")
def fitted(space):
    """Ackley landscape (seed 0) and the surrogate fitted on the measured start set."""
    landscape = SyntheticYield("Ackley", 0, space)
    y = landscape.observe(space.start_x, torch.Generator().manual_seed(0))
    return landscape, fit_surrogate(space.start_x, y, space.feature_dims)


def test_space_dimensions(space):
    """Blocks have the featurization-scan widths and the base block is zero for non-salt catalysts."""
    dims = space.feature_dims
    assert [len(dims[b]) for b in ("cont", "cat", "solv", "base")] == [N_CONTINUOUS_DIMS, CAT_DIM, SOLVENT_DIM, BASE_DIM]
    assert space.start_x.shape[1] == TOTAL_DIM
    non_salt = ~space.cat_is_salt[space.start_cat_index]
    assert non_salt.any()
    assert (space.start_x[non_salt][:, dims["base"]] == 0).all()
    assert int(space.initial_available.sum()) == len(set(space.start_cat_index.tolist()))


def test_synthetic_yield_scale(space, fitted):
    """The attainable maximum maps to yield 1 and the start set lies in [0, 1]."""
    landscape, _ = fitted
    assert torch.allclose(landscape.yield_true(landscape.x_max), torch.ones(1, 1, dtype=torch.double))
    y = landscape.yield_true(space.start_x)
    assert (y >= 0).all() and (y <= 1).all()


def test_pool_posterior_matches_fantasy_node(space, fitted):
    """The root pool cache reproduces BoTorch's posterior of a fantasy node of the KS1 model."""
    _, model = fitted
    g = torch.Generator().manual_seed(1)
    rows = torch.randint(space.start_x.shape[0], (12,), generator=g)
    pts = space.start_x[rows].clone()
    pts[:, 2] = 30.0 + 60.0 * torch.rand(12, generator=g, dtype=torch.double)
    model.posterior(pts[:6])
    node = model.condition_on_observations(X=pts[:6], Y=torch.randn(6, 1, generator=g, dtype=torch.double))
    node.eval()
    cache = GPPoolPosteriorCache(model)
    assert cache.matches(node)
    posterior = _ExactGPPoolPosterior(node, cache)
    batch, pool = pts[6:8], pts[8:]
    entry = cache.pool(model.transform_inputs(pool), key=None)
    terms = cache.batch_terms(entry, model.transform_inputs(batch))
    mean, cov = posterior.joint(entry, terms)
    with torch.no_grad():
        ref = node.posterior(torch.cat([batch.expand(pool.shape[0], -1, -1), pool.unsqueeze(1)], dim=1))
    assert torch.allclose(mean, ref.mean.squeeze(-1), atol=1e-6)
    assert torch.allclose(cov, ref.covariance_matrix, atol=1e-6)
