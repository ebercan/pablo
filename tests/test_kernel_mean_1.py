"""The kernel_mean_1 grid, the KS5-KS7 kernel structures and the cond_linear,
cond_trend and form_offset mean functions."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import AdditiveKernel

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import (
    CatalystFloorKernel,
    FormOffsetKernel,
    LibraryCentredKernel,
    LinearTrendKernel,
    ProductPlusMainEffectsKernel,
    SharedDeviationKernel,
    make_kernel_from_config,
)
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    CondLinearMean,
    build_mean_kernel_term,
    build_mean_module,
    library_cat_reference,
)
from pablo.featurization_and_kernel_scan.cross_validation import fit_eval as fe
from pablo.featurization_and_kernel_scan.cross_validation import scheduler
from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import build_run_cells
from pablo.featurization_and_kernel_scan.results_analysis import rebuild_predictive as rb

GRID = config.KERNEL_MEAN_1_GRID
CELLS = build_run_cells(GRID)
INNER = {"base_feat": "grover_pca", "prior_scheme": "P2"}


def _cfg(kernel_structure: str = "KS1", mean_function: str = "constant") -> dict:
    return dict(
        cat_feat="grover_pca", solv_feat="Hansen", kernel_structure=kernel_structure,
        model_class="st_y", outcome_transform="raw", base_kernel="matern52",
        mean_function=mean_function,
    )


@pytest.fixture(scope="module")
def cell():
    """Real grover_pca / Hansen / grover_pca features with the P2 partition."""
    return rb.build_cell(_cfg(), INNER)


def _config_of(c: dict) -> dict[str, str]:
    (inner,) = c["inner_cells"]
    merged = {**c, **inner}
    return {axis: merged[axis] for axis in ("cat_feat", *config.ELIMINATION_GRID.baseline)}


def test_grid_is_every_anchor_under_every_variant():
    """96 cells: 6 anchors x 4 kernel structures x 4 means, ids 0..95 by cost."""
    assert len(CELLS) == 96
    got = {tuple(sorted(_config_of(c).items())) for c in CELLS.values()}
    want = {
        tuple(sorted({**a, "kernel_structure": ks, "mean_function": mf}.items()))
        for a in GRID.anchors
        for ks in ("KS1", "KS2", "KS5", "KS6")
        for mf in ("constant", "cond_linear", "cond_trend", "form_offset")
    }
    assert got == want
    assert sorted(int(k) for k in CELLS) == list(range(96))
    costs = [scheduler._per_cell_seconds(CELLS[str(i)]) for i in range(96)]
    assert costs == sorted(costs, reverse=True)
    assert len(CELLS) * config.FOLD_TOTAL == 13152


def test_anchors_are_the_best_cross_1_configuration_per_level():
    """Each anchor is the highest-gem cross_1 configuration of its level; unimol is not an anchor."""
    scores = pd.read_parquet(paths.run_paths("cross_1").gem_scores_parquet)
    best = scores.sort_values("gem", ascending=False).groupby("cat_feat").head(1)
    best = best[best["cat_feat"] != "unimol"]
    want = {
        tuple(sorted({k: row[k] for k in ("cat_feat", *config.ELIMINATION_GRID.baseline)}.items()))
        for _, row in best.iterrows()
    }
    assert {tuple(sorted(a.items())) for a in GRID.anchors} == want


def test_anchored_grid_rejects_a_bad_anchor_or_variant():
    anchor = dict(GRID.anchors[0])
    with pytest.raises(ValueError, match="keys"):
        config.AnchoredGrid(anchors=({k: v for k, v in anchor.items() if k != "cat_feat"},),
                            variants={"kernel_structure": ("KS1",)}, note="")
    with pytest.raises(ValueError, match="twice"):
        config.AnchoredGrid(anchors=(anchor, dict(anchor)), variants={"kernel_structure": ("KS1",)}, note="")
    with pytest.raises(ValueError, match="not a configuration axis"):
        config.AnchoredGrid(anchors=(anchor,), variants={"cat_feat": ("grover",)}, note="")


def _kernel(cell, ks: str, train_x=None):
    return make_kernel_from_config(
        ks, cell.feature_dims, cell.active_flag_dim, "matern52", partition=cell.partition,
        prior_scheme="P2",
        cat_reference=None if train_x is None else library_cat_reference(cell, train_x),
    )


def _x(cell, n: int = 20) -> torch.Tensor:
    """The first ``n`` rows mapped to the unit box, as the model's input transform does."""
    x = cell.X[:n].clone()
    lo, hi = x.amin(0), x.amax(0)
    span = torch.where(hi - lo < 1e-8, torch.ones_like(hi), hi - lo)
    out = (x - torch.where(hi - lo < 1e-8, torch.zeros_like(lo), lo)) / span
    out[:, cell.form_id_dim] = x[:, cell.form_id_dim]
    return out


def test_ks5_and_ks6_reduce_to_ks1_when_their_scale_is_zero(cell):
    """KS5 at rho = 0 and KS6 at s_+ = 0 give the KS1 covariance."""
    x = _x(cell)
    ks1, ks5, ks6 = _kernel(cell, "KS1"), _kernel(cell, "KS5"), _kernel(cell, "KS6")
    floor = ks5.base_kernel.kernels[0]
    assert isinstance(floor, CatalystFloorKernel)
    floor.initialize(raw_floor=torch.tensor(-40.0, dtype=torch.float64))
    assert isinstance(ks6, ProductPlusMainEffectsKernel)
    ks6.initialize(raw_main_scale=torch.tensor(-40.0, dtype=torch.float64))
    k1 = ks1(x).to_dense()
    np.testing.assert_allclose(ks5(x).to_dense().detach(), k1.detach(), atol=1e-12)
    np.testing.assert_allclose(ks6(x).to_dense().detach(), k1.detach(), atol=1e-12)


def test_ks5_floor_keeps_covariance_between_distant_catalysts(cell):
    """With rho = 0.4 the catalyst factor never falls below 0.4."""
    floor = _kernel(cell, "KS5").base_kernel.kernels[0]
    floor.initialize(raw_floor=floor.raw_floor_constraint.inverse_transform(torch.tensor(0.4, dtype=torch.float64)))
    k = floor(_x(cell)).to_dense().detach()
    assert float(k.min()) >= 0.4 - 1e-12


def test_ks7_deviation_averages_to_zero_over_the_library(cell):
    """The centred catalyst kernel sums to zero against the library rows and is PSD."""
    train_x = cell.X[:51]
    ks7 = _kernel(cell, "KS7", train_x)
    shared = ks7.kernels[0]
    assert isinstance(shared, SharedDeviationKernel)
    centred = shared.centred_kernel
    assert isinstance(centred, LibraryCentredKernel)
    ref = centred.reference
    assert ref.shape == (161, cell.X.shape[1])
    x = _x(cell, 30)
    k_ref = centred(x, ref).to_dense().detach()
    np.testing.assert_allclose(k_ref.mean(dim=-1), 0.0, atol=1e-10)
    eig = torch.linalg.eigvalsh(centred(ref).to_dense().detach())
    assert float(eig.min()) > -1e-8
    np.testing.assert_allclose(
        centred(x, diag=True).detach(), torch.diagonal(centred(x).to_dense().detach()), atol=1e-12
    )


def test_library_reference_matches_the_model_input_transform(cell):
    """The KS7 reference rows equal the library mapped by the fitted Normalize."""
    train_idx = list(range(51))
    model = AcyloinCondensationModel(
        cell, train_idx, "KS7", "matern52", "raw", config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme="P2",
    ).build("st_y", None)
    assert isinstance(model, SingleTaskGP)
    full = torch.zeros(161, cell.X.shape[1], dtype=torch.float64)
    full[:, cell.feature_dims["cat"]] = cell.cat_library
    model.input_transform.eval()
    transformed = model.input_transform(full)
    ref = library_cat_reference(cell, cell.X[train_idx])
    cat = cell.feature_dims["cat"]
    np.testing.assert_allclose(ref[:, cat], transformed[:, cat].detach(), atol=1e-12)


def test_cond_trend_term_is_the_integrated_cond_linear_prior(cell):
    """s² h(x)ᵀ D h(x') with the cond_linear basis and prior variances."""
    term = build_mean_kernel_term("cond_trend", cell)
    assert isinstance(term, LinearTrendKernel)
    x = _x(cell)
    fd = cell.feature_dims
    gate = x[:, cell.active_flag_dim : cell.active_flag_dim + 1]
    h = torch.cat([
        (x[:, fd["solv"]] - 0.5) / len(fd["solv"]) ** 0.5,
        gate * (x[:, fd["base"]] - 0.5) / len(fd["base"]) ** 0.5,
        (x[:, fd["cont"]] - 0.5) / len(fd["cont"]) ** 0.5,
        gate,
    ], dim=-1)
    s2 = term.scale.detach() ** 2
    np.testing.assert_allclose(term(x).to_dense().detach(), s2 * h @ h.T, atol=1e-12)


def test_form_offset_term_correlates_rows_of_one_form(cell):
    term = build_mean_kernel_term("form_offset", cell)
    assert isinstance(term, FormOffsetKernel)
    x = _x(cell)
    forms = x[:, cell.form_id_dim]
    k = term(x).to_dense().detach()
    t2 = float(term.scale.detach() ** 2)
    np.testing.assert_allclose(k, t2 * (forms[:, None] == forms[None, :]).double(), atol=1e-12)


def test_cond_linear_has_no_catalyst_slope_and_gates_the_base(cell):
    mean = build_mean_module("cond_linear", cell)
    assert isinstance(mean, CondLinearMean)
    fd = cell.feature_dims
    assert sum(p.numel() for p in mean.parameters()) == len(fd["solv"]) + len(fd["base"]) + len(fd["cont"]) + 2
    with torch.no_grad():
        mean.b_base.fill_(1.0)
        mean.a_base.fill_(1.0)
    x = _x(cell)
    off = x.clone()
    off[:, cell.active_flag_dim] = 0.0
    other = off.clone()
    other[:, fd["base"]] = 0.3
    other[:, fd["cat"]] = 0.9
    np.testing.assert_allclose(mean(off).detach(), mean(other).detach(), atol=1e-12)


def test_constant_and_linear_means_add_no_kernel_term(cell):
    for mf in ("constant", "block_linear", "type_linear", "cond_linear"):
        assert build_mean_kernel_term(mf, cell) is None
    with pytest.raises(ValueError, match="Unknown mean function"):
        build_mean_kernel_term("quadratic", cell)


@pytest.mark.parametrize(
    "ks, mf",
    [("KS5", "constant"), ("KS6", "constant"), ("KS7", "constant"),
     ("KS1", "cond_linear"), ("KS1", "cond_trend"), ("KS1", "form_offset"), ("KS7", "cond_trend")],
)
def test_one_fold_fits_and_rebuilds(cell, ks, mf):
    """A real fold fits under each new level and its raw predictive is rebuilt exactly."""
    n = cell.X.shape[0]
    torch.manual_seed(0)
    result = fe.fit_eval(
        cell, train_idx=list(range(n - 8)), test_idx=list(range(n - 8, n)), kernel_structure=ks,
        outcome_transform="raw", model_class="st_y", base_kernel="matern52", mean_function=mf,
        n_restarts=1, max_iter=40, prior_scheme="P2",
    )
    assert result.status == "ok"
    fold = dataclasses.asdict(result)
    m, s = rb.rebuild_fold(cell, _cfg(ks, mf), INNER, fold)
    np.testing.assert_allclose(m.numpy(), fold["mu_pred_per_point"], rtol=0, atol=1e-8)
    np.testing.assert_allclose((s ** 2).numpy(), fold["var_pred_per_point"], rtol=1e-8)


def test_mean_kernel_term_is_added_to_the_covariance(cell):
    model = AcyloinCondensationModel(
        cell, list(range(51)), "KS1", "matern52", "raw", config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme="P2", mean_function="form_offset",
    ).build("st_y", None)
    assert isinstance(model.covar_module, AdditiveKernel)
    assert isinstance(model.covar_module.kernels[1], FormOffsetKernel)
