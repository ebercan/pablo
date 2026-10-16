"""Tests for the scope-expansion expanders and helpers."""

import functools

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize

from pablo.optimize import optimize_acqf_mixed_pool_greedy
from pablo.scope_expansion import (
    _aggregation_weights,
    _expansion_batch_constraints,
    build_fantasy_tree,
    fantasized_criterion,
    optimize_scope_expansion_fantasized,
    optimize_scope_expansion_myopic,
)


def test_expansion_batch_constraints_partial_partition():
    """Expansion steps get the partial partition for their own point count."""
    assert _expansion_batch_constraints({0: (2, 2, 2)}, 1) is None
    assert _expansion_batch_constraints({0: (2, 2, 2)}, 3) is None
    assert _expansion_batch_constraints({0: (2, 2, 2)}, 4) == {0: (2, 1, 1)}
    assert _expansion_batch_constraints(None, 1) is None
    assert _expansion_batch_constraints({0: 3}, 1) is None


def test_aggregation_weights_modes():
    """Each operating mode weights the tree nodes as its definition says."""
    sizes = [2, 4]
    assert _aggregation_weights(sizes, "only_terminal").tolist() == [0, 0, 0.25, 0.25, 0.25, 0.25]
    assert torch.allclose(_aggregation_weights(sizes, "averaged"), torch.full((6,), 1 / 6, dtype=torch.float64))
    assert _aggregation_weights(sizes, "layer_averaged").tolist() == [0.25, 0.25, 0.125, 0.125, 0.125, 0.125]


@pytest.fixture
def scope_toy():
    g = torch.Generator().manual_seed(0)
    cats = torch.rand(6, 2, generator=g, dtype=torch.double)
    sols = torch.rand(2, 1, generator=g, dtype=torch.double)
    rows = torch.cat([cats.repeat_interleave(2, 0), sols.repeat(6, 1)], dim=-1)  # (12, 3)
    bounds = torch.stack([torch.zeros(5), torch.ones(5)]).double()
    x = torch.cat([torch.rand(20, 2, generator=g, dtype=torch.double), rows[torch.randint(4, (20,), generator=g)]], dim=-1)
    y = (3 * x[:, :1] - x[:, 2:3]).sin()
    model = SingleTaskGP(x, y, input_transform=Normalize(5, bounds=bounds), outcome_transform=Standardize(1))
    model.eval()

    def decompose(xs):
        return xs[:, :2].unique(dim=0), xs[:, 2:].unique(dim=0), torch.zeros(1, 0, dtype=torch.double)

    return {"model": model, "rows": rows, "bounds": bounds, "decompose": decompose}


def test_fantasized_expander_picks_a_locked_row(scope_toy):
    """The fantasized expander returns a point on a locked catalyst."""
    rows = scope_toy["rows"]
    opt = functools.partial(optimize_acqf_mixed_pool_greedy, n_pool=32, n_keep=8, n_mc=32, polish_maxiter=3)
    cand, value = optimize_scope_expansion_fantasized(
        scope_toy["model"], q_batch=4, q_expand=1, x_available=rows[:4], x_ordered=[rows[4:6], rows[6:8]],
        x_unordered=rows[8:], bounds=scope_toy["bounds"], decompose_allowed_set=scope_toy["decompose"],
        k=2, num_fantasies=[2, 1], batch_constraints={0: (2, 2)}, operating_mode="only_terminal",
        node_optimizer=opt, expansion_optimizer=opt,
    )
    assert cand.shape == (1, 5) and torch.isfinite(value)
    assert torch.isclose(cand[:, 2:4].unsqueeze(1), rows[8:, :2].unsqueeze(0)).all(-1).any()


def test_myopic_expander_picks_a_locked_row(scope_toy):
    """The myopic expander returns a point on a locked catalyst."""
    rows = scope_toy["rows"]
    opt = functools.partial(optimize_acqf_mixed_pool_greedy, n_pool=32, n_keep=8, n_mc=32, polish_maxiter=0)
    cand, _ = optimize_scope_expansion_myopic(
        scope_toy["model"], q_expand=1, x_unordered=rows[8:], bounds=scope_toy["bounds"],
        decompose_allowed_set=scope_toy["decompose"], batch_constraints={0: (2, 2)}, expansion_optimizer=opt,
    )
    assert torch.isclose(cand[:, 2:4].unsqueeze(1), rows[8:, :2].unsqueeze(0)).all(-1).any()


def test_fantasized_expander_availability_per_depth(scope_toy):
    """Depth j may use x_ordered[0..j]; the terminal batch may use every ordered row."""
    rows = scope_toy["rows"]
    opt = functools.partial(optimize_acqf_mixed_pool_greedy, n_pool=32, n_keep=8, n_mc=32, polish_maxiter=0)
    seen_by_seed: dict[int, set[int]] = {}

    def spy(**kwargs):
        seen_by_seed.setdefault(kwargs["seed"], set()).add(kwargs["cat_emb_avail"].shape[0])
        return opt(**kwargs)

    optimize_scope_expansion_fantasized(
        scope_toy["model"], q_batch=4, q_expand=1, x_available=rows[:4], x_ordered=[rows[4:6], rows[6:8]],
        x_unordered=rows[8:], bounds=scope_toy["bounds"], decompose_allowed_set=scope_toy["decompose"],
        k=2, num_fantasies=[2, 1], batch_constraints={0: (2, 2)}, operating_mode="averaged",
        node_optimizer=spy, expansion_optimizer=opt, completion_optimizer=spy,
    )
    # rows hold two per catalyst: x_available has 2 catalysts, each x_ordered entry adds 1.
    assert seen_by_seed == {0: {3}, 1: {4}, 3: {4}}


def test_fantasized_expander_rejects_x_ordered_beyond_horizon(scope_toy):
    """Ordered rows that unlock after the terminal batch are a caller error."""
    rows = scope_toy["rows"]
    with pytest.raises(ValueError, match="x_ordered"):
        optimize_scope_expansion_fantasized(
            scope_toy["model"], q_batch=4, q_expand=1, x_available=rows[:4], x_ordered=[rows[4:6], rows[6:8]],
            x_unordered=rows[8:], bounds=scope_toy["bounds"], decompose_allowed_set=scope_toy["decompose"],
            k=1, num_fantasies=[2],
        )


def test_tree_releases_node_caches_and_released_criterion_is_unchanged(scope_toy):
    """Inner tree nodes drop their GP caches, and a criterion that releases member caches keeps its values."""
    rows = scope_toy["rows"]
    cats, sols, bas = scope_toy["decompose"](rows[:4])
    opt = functools.partial(optimize_acqf_mixed_pool_greedy, n_pool=32, n_keep=8, n_mc=32, polish_maxiter=0)
    layers, _ = build_fantasy_tree(
        scope_toy["model"], q_batch=2, tables_per_depth=[(cats, sols, bas, None)] * 2,
        bounds=scope_toy["bounds"], num_fantasies=[2, 1], node_optimizer=opt,
    )
    assert all(m.prediction_strategy is None for m in layers[0])
    g = torch.Generator().manual_seed(1)
    x = torch.cat([torch.rand(3, 2, generator=g, dtype=torch.double), rows[8:11]], dim=-1).unsqueeze(1)  # (3, 1, 5)
    criterion = fantasized_criterion(layers, "averaged", 32)
    with torch.no_grad():
        kept = criterion(x)
        criterion.release_member_caches = True
        released = criterion(x)
    assert torch.allclose(kept, released, rtol=1e-8, atol=1e-10)
    assert all(m.prediction_strategy is None for layer in layers for m in layer)
