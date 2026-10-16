"""Tests for results_analysis.rebuild_predictive: a fold rebuilt from its stored parameters
reproduces the fitted predictive distribution, and the stored-moment check rejects a mismatch."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from pablo.featurization_and_kernel_scan.cross_validation import fit_eval as fe
from pablo.featurization_and_kernel_scan.results_analysis import rebuild_predictive as rb
from pablo.utils.outcome_transforms import inverse

INNER = {"base_feat": "OHE", "prior_scheme": "P2"}


def _cfg(transform: str) -> dict:
    return dict(
        cat_feat="interpretable_steric", solv_feat="OHE", kernel_structure="KS1", model_class="st_y",
        outcome_transform=transform, base_kernel="rbf", mean_function="constant",
    )


@pytest.fixture(scope="module")
def cell():
    """Real assembled features with the P2 partition."""
    return rb.build_cell(_cfg("raw"), INNER)


def _fold(cell, transform: str) -> dict:
    n = cell.X.shape[0]
    torch.manual_seed(0)
    result = fe.fit_eval(
        cell, train_idx=list(range(n - 8)), test_idx=list(range(n - 8, n)), kernel_structure="KS1",
        outcome_transform=transform, model_class="st_y", base_kernel="rbf", mean_function="constant",
        n_restarts=1, max_iter=40, prior_scheme="P2",
    )
    return dataclasses.asdict(result)


def _frame(m: torch.Tensor, s: torch.Tensor, transform: str) -> pd.DataFrame:
    return pd.DataFrame({"m": m.numpy(), "s": s.numpy(), "transform": transform})


def test_raw_fold_is_reproduced_exactly(cell):
    """Rebuilt raw-scale mean and variance equal what fit_eval reported."""
    fold = _fold(cell, "raw")
    m, s = rb.rebuild_fold(cell, _cfg("raw"), INNER, fold)
    np.testing.assert_allclose(m.numpy(), fold["mu_pred_per_point"], rtol=0, atol=1e-8)
    np.testing.assert_allclose((s ** 2).numpy(), fold["var_pred_per_point"], rtol=1e-8)
    rb.check_against_stored(_frame(m, s, "raw"), np.asarray(fold["mu_pred_per_point"]),
                            np.asarray(fold["var_pred_per_point"]), oid=-1)


@pytest.mark.parametrize("transform", ["probit", "arcsin_sqrt"])
def test_transformed_fold_passes_the_monte_carlo_check(cell, transform):
    """fit_eval's 2000-draw moments of a transformed fold are Monte Carlo draws of the rebuilt distribution."""
    fold = _fold(cell, transform)
    m, s = rb.rebuild_fold(cell, _cfg(transform), INNER, fold)
    rb.check_against_stored(_frame(m, s, transform), np.asarray(fold["mu_pred_per_point"]),
                            np.asarray(fold["var_pred_per_point"]), oid=-1)


def test_shifted_rebuild_fails_the_monte_carlo_check():
    """A rebuild whose mean is off by 20 percent of a predictive SD is rejected."""
    gen = torch.Generator().manual_seed(1)
    m = torch.linspace(-1.5, 0.5, 400, dtype=torch.float64)
    s = torch.full_like(m, 0.4)
    draws = inverse(m + s * torch.randn(fe.N_PRED_SAMPLES, len(m), generator=gen, dtype=torch.float64), "probit")
    mu, var = draws.mean(0).numpy(), draws.var(0).numpy()
    rb.check_against_stored(_frame(m, s, "probit"), mu, var, oid=-1)
    with pytest.raises(RuntimeError, match="Monte Carlo"):
        rb.check_against_stored(_frame(m + 0.2 * s, s, "probit"), mu, var, oid=-1)


def test_raw_mismatch_raises():
    """A raw cell whose rebuilt variance differs from the stored one is rejected."""
    m = torch.tensor([0.1, 0.2], dtype=torch.float64)
    s = torch.tensor([0.05, 0.05], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="rebuilt raw moments differ"):
        rb.check_against_stored(_frame(m, s, "raw"), m.numpy(), (s ** 2).numpy() * 1.01, oid=-1)


@pytest.mark.parametrize("transform", ["probit", "arcsin_sqrt", "logit"])
def test_raw_moments_match_monte_carlo(transform):
    """Quadrature raw-scale moments agree with 400,000 draws, including the clamped arcsine tails."""
    m = np.array([-1.0, 0.3, 1.2])
    s = np.array([0.2, 0.9, 1.5])
    mean, var, _ = rb.raw_moments(m, s, transform)
    gen = torch.Generator().manual_seed(2)
    z = torch.as_tensor(m) + torch.as_tensor(s) * torch.randn(400_000, 3, generator=gen, dtype=torch.float64)
    y = inverse(z, transform).numpy()
    np.testing.assert_allclose(mean, y.mean(0), atol=5 * np.sqrt(var / 400_000).max())
    np.testing.assert_allclose(var, y.var(0), rtol=0.02, atol=1e-8)
