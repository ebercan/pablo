"""Outcome transforms: exact inverses, backward compatibility, and dispatch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import get_y_cs, get_y_yield
from pablo.utils.feature_assembly import LOGIT_EPS
from pablo.utils.outcome_transforms import (
    OUTCOME_TRANSFORM_NAMES,
    forward,
    inverse,
)

Y_GRID = torch.linspace(0.0, 1.0, 401, dtype=torch.float64).unsqueeze(-1)


@pytest.mark.parametrize("name", OUTCOME_TRANSFORM_NAMES)
def test_inverse_is_exact(name: str) -> None:
    """inverse(forward(y)) recovers y to machine precision, boundaries included."""
    recovered = inverse(forward(Y_GRID, name), name)
    assert torch.allclose(recovered, Y_GRID, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("name", OUTCOME_TRANSFORM_NAMES)
def test_forward_is_finite_and_monotone(name: str) -> None:
    """No transform produces a non-finite value at y=0 or y=1, and all are increasing."""
    z = forward(Y_GRID, name)
    assert torch.isfinite(z).all()
    assert (z.diff(dim=0) > 0).all()


@pytest.mark.parametrize("name", OUTCOME_TRANSFORM_NAMES)
def test_shape_and_dtype_preserved(name: str) -> None:
    """Transforms are elementwise: shape and dtype survive a round trip."""
    y = torch.rand(7, 3, 2, dtype=torch.float64)
    z = forward(y, name)
    assert z.shape == y.shape and z.dtype == y.dtype
    assert inverse(z, name).shape == y.shape


def test_logit_matches_the_stored_formula() -> None:
    """forward(y, 'logit') reproduces the y_logit precomputed in feature_assembly."""
    frac = Y_GRID.numpy()
    stored = np.log((frac + LOGIT_EPS) / (1.0 - frac + LOGIT_EPS))
    assert np.allclose(forward(Y_GRID, "logit").numpy(), stored, atol=1e-12, rtol=0.0)


def test_logit_inverse_matches_the_retired_inline_back_transform() -> None:
    """inverse(z, 'logit') is bit-identical to (1 + 2·eps) / (1 + exp(−z)) − eps, eps = LOGIT_EPS."""
    z = forward(Y_GRID, "logit")
    retired = (1.0 + 2.0 * LOGIT_EPS) / (1.0 + torch.exp(-z)) - LOGIT_EPS
    assert torch.equal(inverse(z, "logit"), retired)


def test_arcsin_sqrt_image_is_bounded_by_half_pi() -> None:
    """arcsin_sqrt maps into [0, pi/2], the variance-stabilising range."""
    z = forward(Y_GRID, "arcsin_sqrt")
    assert float(z.min()) >= 0.0
    assert float(z.max()) <= np.pi / 2.0


@pytest.mark.parametrize("bad", ["", "raw ", "arcsin-sqrt", "sqrt", "logit2"])
def test_unknown_transform_raises(bad: str) -> None:
    """An unrecognised name fails loudly in both directions rather than passing through."""
    with pytest.raises(ValueError, match="Unknown outcome_transform"):
        forward(Y_GRID, bad)
    with pytest.raises(ValueError, match="Unknown outcome_transform"):
        inverse(Y_GRID, bad)


def test_config_exposes_every_implemented_transform() -> None:
    """The grid axis and the implementation cannot drift apart."""
    assert config.OUTCOME_TRANSFORMS == list(OUTCOME_TRANSFORM_NAMES)


class _StubCellFeatures:
    """Minimal stand-in carrying only the six target tensors get_y_* reads."""

    def __init__(self) -> None:
        y = torch.tensor([[0.0], [0.31], [0.77], [1.0]], dtype=torch.float64)
        self.y_raw = y
        self.y_logit = forward(y, "logit")
        self.y_conversion_raw = y * 0.9
        self.y_conversion_logit = forward(y * 0.9, "logit")
        self.y_selectivity_raw = y * 0.5
        self.y_selectivity_logit = forward(y * 0.5, "logit")


@pytest.mark.parametrize("name", OUTCOME_TRANSFORM_NAMES)
def test_get_y_dispatch_round_trips_to_raw(name: str) -> None:
    """Every target get_y_* returns back-transforms to the raw tensor it came from."""
    cf = _StubCellFeatures()
    assert torch.allclose(inverse(get_y_yield(cf, name), name), cf.y_raw, atol=1e-12)
    y_c, y_s = get_y_cs(cf, name)
    assert torch.allclose(inverse(y_c, name), cf.y_conversion_raw, atol=1e-12)
    assert torch.allclose(inverse(y_s, name), cf.y_selectivity_raw, atol=1e-12)


def test_get_y_yield_returns_the_stored_tensors_for_raw_and_logit() -> None:
    """Pre-existing grids keep their exact tensors, not recomputed equivalents."""
    cf = _StubCellFeatures()
    assert get_y_yield(cf, "raw") is cf.y_raw
    assert get_y_yield(cf, "logit") is cf.y_logit


@pytest.mark.parametrize("name", OUTCOME_TRANSFORM_NAMES)
def test_end_to_end_single_fold_fit_per_transform(name: str) -> None:
    """Every transform fits a real fold and reports metrics back on the raw scale."""
    from pablo.featurization_and_kernel_scan import paths
    from pablo.featurization_and_kernel_scan.core.kernels import build_partition
    from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import fit_eval
    from pablo.utils.feature_assembly import assemble_features

    cell = assemble_features(
        cat_featurization="interpretable_steric",
        solv_featurization="KT",
        base_featurization="thermo",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )
    cell.partition = build_partition(cell.feature_dims)
    n = cell.X.shape[0]
    result = fit_eval(
        cell,
        train_idx=list(range(n - 8)),
        test_idx=list(range(n - 8, n)),
        kernel_structure="KS1",
        outcome_transform=name,
        model_class="st_y",
        base_kernel="matern52",
        mean_function="type_linear",
        n_restarts=1,
        max_iter=40,
    )
    assert result.status == "ok", result
    assert np.isfinite(result.rmse) and result.rmse > 0
    assert np.isfinite(result.crpss)
    mu = np.asarray(result.mu_pred_per_point, dtype=float)
    assert mu.size == 8 and np.isfinite(mu).all()
    if name != "raw":
        assert mu.min() >= -LOGIT_EPS - 1e-9
        assert mu.max() <= 1.0 + LOGIT_EPS + 1e-9


def test_arcsin_sqrt_inverse_does_not_fold_out_of_range_draws() -> None:
    """Draws below 0 map to the lowest yield and draws above π/2 to the highest, not back inside."""
    lo = inverse(torch.tensor([0.0], dtype=torch.float64), "arcsin_sqrt")
    hi = inverse(torch.tensor([np.pi / 2], dtype=torch.float64), "arcsin_sqrt")
    below = inverse(torch.tensor([-0.4, -1.0], dtype=torch.float64), "arcsin_sqrt")
    above = inverse(torch.tensor([1.8, 2.5], dtype=torch.float64), "arcsin_sqrt")
    assert torch.equal(below, lo.expand(2))
    assert torch.equal(above, hi.expand(2))
