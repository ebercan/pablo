"""Tests for optimize_acqf_mixed_lookup_alternating (greedy choice-space local search)."""

import pytest
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler

from pablo.optimize import (
    _greedy_group_maps,
    _LookupAcquisitionFunction,
    optimize_acqf_mixed_lookup_alternating,
)

D_CONT, D_CAT, D_SOL, D_BAS = 4, 5, 2, 2
N_CAT, N_SOL, N_BAS = 9, 4, 3
D = D_CONT + D_CAT + D_SOL + D_BAS


@pytest.fixture(scope="module")
def toy():
    """Small component tables, salt flags and a fitted GP with a qLogEI acquisition."""
    torch.manual_seed(0)
    cat = torch.rand(N_CAT, D_CAT)
    sol = torch.rand(N_SOL, D_SOL)
    bas = torch.rand(N_BAS, D_BAS)
    flags = torch.tensor([i % 3 != 0 for i in range(N_CAT)])
    c = torch.randint(N_CAT, (25,))
    s = torch.randint(N_SOL, (25,))
    b = torch.randint(N_BAS, (25,))
    X = torch.cat(
        [torch.rand(25, D_CONT), cat[c], sol[s], bas[b] * flags[c].unsqueeze(-1).double()],
        dim=-1,
    )
    Y = -(X - 0.5).pow(2).sum(-1, keepdim=True)
    model = SingleTaskGP(
        X, Y, input_transform=Normalize(d=D), outcome_transform=Standardize(m=1)
    )
    acq = qLogExpectedImprovement(
        model, best_f=Y.max(), sampler=SobolQMCNormalSampler(torch.Size([32]))
    )
    bounds = torch.stack([torch.zeros(D), torch.ones(D)])
    return dict(cat=cat, sol=sol, bas=bas, flags=flags, acq=acq, bounds=bounds)


def _row_indices(row, toy):
    cat_idx = (toy["cat"] - row[D_CONT : D_CONT + D_CAT]).abs().sum(-1).argmin().item()
    sol_idx = (toy["sol"] - row[D_CONT + D_CAT : D_CONT + D_CAT + D_SOL]).abs().sum(-1).argmin().item()
    return cat_idx, sol_idx


def test_candidates_lie_in_allowed_set_with_salt_rule(toy):
    """Every returned row is an exact table row, and non-salt catalysts get a zero base slice."""
    cands, value = optimize_acqf_mixed_lookup_alternating(
        toy["acq"], toy["bounds"], q=3,
        cat_emb_avail=toy["cat"], sol_emb_avail=toy["sol"], bas_emb_avail=toy["bas"],
        cat_is_salt_flags=toy["flags"], num_restarts=2, raw_samples=32, seed=1,
    )
    assert cands.shape == (3, D)
    assert value.ndim == 0 and torch.isfinite(value)
    for row in cands:
        c_idx, s_idx = _row_indices(row, toy)
        assert torch.allclose(row[D_CONT : D_CONT + D_CAT], toy["cat"][c_idx])
        assert torch.allclose(row[D_CONT + D_CAT : D_CONT + D_CAT + D_SOL], toy["sol"][s_idx])
        base = row[-D_BAS:]
        if toy["flags"][c_idx]:
            assert any(torch.allclose(base, b) for b in toy["bas"])
        else:
            assert torch.all(base == 0)


def test_partition_and_fixed_dims_are_respected(toy):
    """(2, 2, 2) on dim 0 gives three shared temperature values; dim 3 is pinned."""
    cands, _ = optimize_acqf_mixed_lookup_alternating(
        toy["acq"], toy["bounds"], q=6,
        cat_emb_avail=toy["cat"], sol_emb_avail=toy["sol"], bas_emb_avail=toy["bas"],
        batch_constraints={0: (2, 2, 2)}, fixed_dims={3: 0.25},
        num_restarts=2, raw_samples=32, seed=2,
    )
    temps = cands[:, 0]
    assert torch.allclose(temps[0], temps[1])
    assert torch.allclose(temps[2], temps[3])
    assert torch.allclose(temps[4], temps[5])
    assert torch.all(cands[:, 3] == 0.25)


def test_pre_batch_rows_are_kept_and_share_group_temperature(toy):
    """pre_batch rows are returned unchanged and the next candidate joins their temperature group."""
    pre = torch.rand(1, D)
    pre[0, D_CONT:] = torch.cat([toy["cat"][1], toy["sol"][0], toy["bas"][0]])
    cands, _ = optimize_acqf_mixed_lookup_alternating(
        toy["acq"], toy["bounds"], q=4, pre_batch=pre,
        cat_emb_avail=toy["cat"], sol_emb_avail=toy["sol"], bas_emb_avail=toy["bas"],
        batch_constraints={0: (2, 2)}, num_restarts=2, raw_samples=32, seed=3,
    )
    assert torch.allclose(cands[0], pre[0])
    assert torch.allclose(cands[1, 0], pre[0, 0])
    assert torch.allclose(cands[2, 0], cands[3, 0])


def test_greedy_group_maps_partial_partition():
    """A partition larger than q is relaxed following the sequential-GA convention."""
    assert _greedy_group_maps({0: (2, 2, 2)}, q=6, d_continuous=4) == {0: [0, 0, 1, 1, 2, 2]}
    assert _greedy_group_maps({0: (2, 2, 2)}, q=1, d_continuous=4) == {}
    assert _greedy_group_maps({0: (2, 2, 2)}, q=4, d_continuous=4) == {0: [0, 0, 1, 2]}
    assert _greedy_group_maps({0: 1}, q=3, d_continuous=4) == {0: [0, 0, 0]}
    with pytest.raises(ValueError):
        _greedy_group_maps({0: 2}, q=3, d_continuous=4)
    with pytest.raises(ValueError):
        _greedy_group_maps({5: (2, 2, 2)}, q=6, d_continuous=4)


def test_lookup_forward_matches_manual_assembly(toy):
    """The lookup wrapper evaluates the base acquisition on [pre_batch; expanded X]."""
    pre = torch.rand(2, D)
    lookup = _LookupAcquisitionFunction(
        toy["acq"], D_CONT, toy["cat"], toy["sol"], toy["bas"], toy["flags"], pre_batch=pre
    )
    x_idx = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 3.0, 1.0, 2.0]]])
    manual = torch.cat(
        [x_idx[0, 0, :D_CONT], toy["cat"][3], toy["sol"][1], toy["bas"][2] * float(toy["flags"][3])]
    )
    expected = toy["acq"](torch.cat([pre, manual.unsqueeze(0)]).unsqueeze(0))
    assert torch.allclose(lookup(x_idx), expected)


def test_component_tables_required(toy):
    """Choice mode is not supported; the component tables are mandatory."""
    with pytest.raises(ValueError):
        optimize_acqf_mixed_lookup_alternating(toy["acq"], toy["bounds"], q=1)


def test_best_observed_f_undoes_outcome_transform():
    """best_observed_f returns the raw maximum even when the model standardizes targets."""
    from pablo.optimize import best_observed_f

    X = torch.rand(20, 3)
    Y = -20.0 + torch.rand(20, 1)
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    assert abs(best_observed_f(model) - Y.max().item()) < 1e-8
    assert abs(model.train_targets.max().item() - Y.max().item()) > 1.0
