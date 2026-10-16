"""Tests for optimize_acqf_mixed_pool_greedy and its pool evaluator."""

import pytest
import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler

from pablo.optimize import (
    GPPoolPosteriorCache,
    WeightedSumAcquisition,
    _PoolAcquisitionEvaluator,
    optimize_acqf_mixed_pool_greedy,
)

D_CONT = 3
FIXED = {2: 0.25}
PARTITION = {0: (2, 2, 2)}


@pytest.fixture
def toy():
    g = torch.Generator().manual_seed(0)
    cat = torch.rand(5, 3, generator=g, dtype=torch.double)
    sol = torch.rand(3, 1, generator=g, dtype=torch.double)
    bas = torch.rand(2, 1, generator=g, dtype=torch.double) + 1.0
    bounds = torch.stack([torch.zeros(D_CONT + 5), torch.ones(D_CONT + 5) * 2]).double()
    x = torch.cat(
        [
            torch.rand(30, D_CONT, generator=g, dtype=torch.double),
            cat[torch.randint(5, (30,), generator=g)],
            sol[torch.randint(3, (30,), generator=g)],
            bas[torch.randint(2, (30,), generator=g)],
        ],
        dim=-1,
    )
    y = (x[:, :2].sum(-1, keepdim=True) - x[:, 3:4]).sin() + 0.05 * torch.randn(30, 1, generator=g, dtype=torch.double)
    model = SingleTaskGP(x, y, input_transform=Normalize(x.shape[-1], bounds=bounds), outcome_transform=Standardize(1))
    model.eval()
    acq = qLogExpectedImprovement(model, y.max(), sampler=SobolQMCNormalSampler(torch.Size([128]), seed=3))
    return {"cat": cat, "sol": sol, "bas": bas, "bounds": bounds, "model": model, "acq": acq, "best_f": y.max()}


def _run(toy, **kwargs):
    base = dict(
        acq_function=toy["acq"], bounds=toy["bounds"], q=6, cat_emb_avail=toy["cat"],
        sol_emb_avail=toy["sol"], bas_emb_avail=toy["bas"], batch_constraints=PARTITION,
        fixed_dims=FIXED, n_pool=64, n_keep=16, seed=4,
    )
    base.update(kwargs)
    return optimize_acqf_mixed_pool_greedy(**base)


def _rows_in(rows, table):
    return torch.isclose(rows.unsqueeze(1), table.unsqueeze(0), atol=1e-12, rtol=0).all(-1).any(-1).all()


def test_fast_path_matches_botorch_qlogei(toy):
    """The cached joint posterior reproduces BoTorch's qLogEI under the same base samples."""
    acq = qLogExpectedImprovement(toy["model"], toy["best_f"], sampler=SobolQMCNormalSampler(torch.Size([64]), seed=9))
    evaluator = _PoolAcquisitionEvaluator(acq, q=4, n_mc=64, seed=9)
    assert evaluator.fast
    g = torch.Generator().manual_seed(1)
    pts = toy["bounds"][0] + (toy["bounds"][1] - toy["bounds"][0]) * torch.rand(23, D_CONT + 5, generator=g, dtype=torch.double)
    batch, pool = pts[:3], pts[3:]
    with torch.no_grad():
        ref = acq(torch.cat([batch.expand(pool.shape[0], -1, -1), pool.unsqueeze(1)], dim=1))
    assert torch.allclose(evaluator.batch_values(batch, pool)[0], ref, atol=1e-8)


def test_batch_is_feasible(toy):
    """Partition, pinned dim, bounds and allowed component rows all hold."""
    salt = torch.tensor([True, False, True, False, True])
    cands, value = _run(toy, cat_is_salt_flags=salt)
    assert cands.shape == (6, D_CONT + 5)
    _, counts = torch.unique(cands[:, 0], return_counts=True)
    assert sorted(counts.tolist()) == [2, 2, 2]
    assert torch.all(cands[:, 2] == FIXED[2])
    assert torch.all(cands >= toy["bounds"][0]) and torch.all(cands <= toy["bounds"][1])
    assert _rows_in(cands[:, 3:6], toy["cat"]) and _rows_in(cands[:, 6:7], toy["sol"])
    cat_idx = torch.isclose(cands[:, 3:6].unsqueeze(1), toy["cat"].unsqueeze(0), atol=1e-12).all(-1).float().argmax(-1)
    non_salt = ~salt[cat_idx]
    assert torch.all(cands[non_salt, 7] == 0) and _rows_in(cands[~non_salt, 7:8], toy["bas"])
    with torch.no_grad():
        assert torch.isclose(value, toy["acq"](cands.unsqueeze(0)).reshape(()))


def test_greedy_value_non_decreasing(toy):
    """Prefix values of the greedy batch never decrease under common base samples."""
    cands, _ = _run(toy, n_swap_rounds=0, polish_maxiter=0)
    evaluator = _PoolAcquisitionEvaluator(toy["acq"], q=6, n_mc=128, seed=4)
    values = [float(evaluator.batch_values(cands[:k], cands[k:k + 1], cache="none")[0]) for k in range(6)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_deterministic_under_seed(toy):
    """The same seed gives the same batch; a different seed may not."""
    a, _ = _run(toy)
    b, _ = _run(toy)
    assert torch.equal(a, b)


def test_pre_batch_counts_toward_partition(toy):
    """``pre_batch`` rows lead the batch and their temperatures join the partition."""
    pre, _ = _run(toy, q=1, batch_constraints=None, polish_maxiter=0)
    cands, _ = _run(toy, pre_batch=pre)
    assert torch.equal(cands[:1], pre)
    _, counts = torch.unique(cands[:, 0], return_counts=True)
    assert sorted(counts.tolist()) == [2, 2, 2]


def test_generic_acquisition_path(toy):
    """A non-qLogEI acquisition function goes through direct evaluation."""
    acq = qExpectedImprovement(toy["model"], toy["best_f"], sampler=SobolQMCNormalSampler(torch.Size([32]), seed=1))
    cands, _ = _run(toy, acq_function=acq, polish_maxiter=5)
    _, counts = torch.unique(cands[:, 0], return_counts=True)
    assert sorted(counts.tolist()) == [2, 2, 2]


def test_rejects_infeasible_pre_batch(toy):
    """A pre_batch whose grouping cannot complete the partition is rejected."""
    pre = torch.zeros(3, D_CONT + 5, dtype=torch.double)
    pre[:, 3:6], pre[:, 6:7], pre[:, 7:8] = toy["cat"][0], toy["sol"][0], toy["bas"][0]
    pre[:, 2] = FIXED[2]
    with pytest.raises(ValueError):
        _run(toy, pre_batch=pre)


def test_root_cache_matches_fantasy_model(toy):
    """Fantasy-node values through the root cache equal BoTorch's qLogEI on the node model."""
    g = torch.Generator().manual_seed(2)
    lo, hi = toy["bounds"]
    pts = lo + (hi - lo) * torch.rand(30, D_CONT + 5, generator=g, dtype=torch.double)
    root = toy["model"]
    root.posterior(pts[:4])
    node = root.condition_on_observations(X=pts[:4], Y=torch.randn(4, 1, generator=g, dtype=torch.double))
    node.eval()
    node.posterior(pts[4:6])
    node = node.condition_on_observations(X=pts[4:6], Y=torch.randn(2, 1, generator=g, dtype=torch.double))
    node.eval()
    cache = GPPoolPosteriorCache(root)
    assert cache.matches(node)
    acq = qLogExpectedImprovement(node, 0.5, sampler=SobolQMCNormalSampler(torch.Size([64]), seed=5))
    evaluator = _PoolAcquisitionEvaluator(acq, q=3, n_mc=64, seed=5, posterior_cache=cache)
    assert evaluator.posteriors[0].cache is cache
    batch, pool = pts[6:8], pts[8:]
    with torch.no_grad():
        ref = acq(torch.cat([batch.expand(pool.shape[0], -1, -1), pool.unsqueeze(1)], dim=1))
    assert torch.allclose(evaluator.batch_values(batch, pool)[0], ref, atol=1e-8)
    weighted = WeightedSumAcquisition([acq, acq], torch.tensor([0.25, 0.75]))
    evaluator = _PoolAcquisitionEvaluator(weighted, q=3, n_mc=64, seed=5, posterior_cache=cache)
    assert torch.allclose(evaluator.batch_values(batch, pool)[0], ref, atol=1e-8)


def test_explicit_incumbents_match_training_lookup(toy):
    """Passing the top training rows as ``incumbents`` reproduces the default local rows."""
    model = toy["model"]
    x_train = model.input_transform.untransform(model.train_inputs[0])
    top = model.train_targets.topk(8).indices
    a, va = _run(toy, n_local=32)
    b, vb = _run(toy, n_local=32, incumbents=x_train[top])
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_rejects_malformed_incumbents(toy):
    """``incumbents`` with the wrong width is rejected."""
    with pytest.raises(ValueError):
        _run(toy, n_local=8, incumbents=torch.zeros(3, D_CONT + 4, dtype=torch.double))
