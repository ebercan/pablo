"""Fold fail criteria in fit_eval: each raises its own named error."""

from __future__ import annotations

import warnings

import pytest
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.core import OptimizationResult, OptimizationStatus
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.errors import NanError
from linear_operator.utils.warnings import NumericalWarning

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.cross_validation import fit_eval as fe


@pytest.fixture(scope="module")
def cell():
    """Real assembled features for one small single-task fold."""
    from pablo.featurization_and_kernel_scan.core.kernels import build_partition
    from pablo.utils.feature_assembly import assemble_features

    cell = assemble_features(
        cat_featurization="interpretable_steric",
        solv_featurization="OHE",
        base_featurization="OHE",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )
    cell.partition = build_partition(cell.feature_dims)
    return cell


def _fit(cell):
    n = cell.X.shape[0]
    return fe.fit_eval(
        cell,
        train_idx=list(range(n - 6)),
        test_idx=list(range(n - 6, n)),
        kernel_structure="KS1",
        outcome_transform="raw",
        model_class="st_y",
        base_kernel="rbf",
        mean_function="constant",
        n_restarts=1,
        max_iter=40,
    )


def _warn_jitter(level: float) -> None:
    warnings.warn(f"A not p.d., added jitter of {level:.1e} to the diagonal", NumericalWarning)


def test_unmodified_fold_is_ok(cell):
    """The fixture fold passes every criterion as it stands."""
    assert _fit(cell).status == "ok"


def test_noise_below_ten_times_final_jitter_raises(cell, monkeypatch):
    """Criterion 3: final jitter 1e-4 with fitted noise 2e-4 fails the fold."""
    original = fe._yield_bundle

    def predict_with_jitter(model, *args, **kwargs):
        model.likelihood.noise = torch.tensor(2e-4, dtype=torch.float64)
        _warn_jitter(1e-4)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(fe, "_yield_bundle", predict_with_jitter)
    with pytest.raises(fe.NoiseBelowJitterError, match="below 10 × final jitter 1.0e-04"):
        _fit(cell)


def test_noise_above_ten_times_final_jitter_passes(cell, monkeypatch):
    """Criterion 3 compares against 10 × jitter: 1e-5 jitter allows 2e-4 noise."""
    original = fe._yield_bundle

    def predict_with_jitter(model, *args, **kwargs):
        model.likelihood.noise = torch.tensor(2e-4, dtype=torch.float64)
        _warn_jitter(1e-5)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(fe, "_yield_bundle", predict_with_jitter)
    assert _fit(cell).status == "ok"


def test_jitter_during_fitting_is_not_the_final_jitter(cell, monkeypatch):
    """Criterion 3 reads the jitter at the selected state, not at a fitting probe."""
    original = fe._fit_multistart

    def fit_with_jitter(model, *args, **kwargs):
        original(model, *args, **kwargs)
        _warn_jitter(1e-4)
        model.likelihood.noise = torch.tensor(2e-4, dtype=torch.float64)

    monkeypatch.setattr(fe, "_fit_multistart", fit_with_jitter)
    assert _fit(cell).status == "ok"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mu_pred_per_point", float("nan")),
        ("mu_pred_per_point", float("inf")),
        ("var_pred_per_point", float("nan")),
        ("var_pred_per_point", 0.0),
        ("var_pred_per_point", -1e-3),
    ],
)
def test_non_finite_prediction_raises(cell, monkeypatch, field, value):
    """Criterion 4: a NaN/inf μ or σ, or σ ≤ 0, on any test point fails the fold."""
    original = fe._yield_bundle

    def corrupt(*args, **kwargs):
        bundle = original(*args, **kwargs)
        bundle[field][0] = value
        return bundle

    monkeypatch.setattr(fe, "_yield_bundle", corrupt)
    with pytest.raises(fe.NonFinitePredictionError, match="1/6 yield test points"):
        _fit(cell)


def test_failed_restart_fails_the_fold(cell, monkeypatch):
    """Criterion 1: a restart whose fit raises fails the fold with that error."""

    def always_fails(mll, **kwargs):
        raise ModelFittingError("All attempts to fit the model have failed.")

    monkeypatch.setattr(fe, "fit_gpytorch_mll", always_fails)
    with pytest.raises(ModelFittingError):
        _fit(cell)


def test_non_finite_mll_restart_is_never_selected(cell, monkeypatch):
    """Criterion 1: restarts ending with a non-finite MLL are skipped; none left is a named error."""
    monkeypatch.setattr(fe, "fit_gpytorch_mll", lambda mll, **kwargs: mll)
    monkeypatch.setattr(fe, "_penalized_loss", lambda model, mll, train_X: float("nan"))
    with pytest.raises(fe.NoConvergedRestartError):
        _fit(cell)


def test_nan_parameters_fail_loudly(cell, monkeypatch):
    """A fit that leaves NaN hyperparameters raises instead of being skipped."""

    def poisons_parameters(mll, **kwargs):
        with torch.no_grad():
            for param in mll.parameters():
                param.fill_(float("nan"))

    monkeypatch.setattr(fe, "fit_gpytorch_mll", poisons_parameters)
    with pytest.raises(NanError):
        _fit(cell)


def test_botorch_raises_rather_than_returning_an_abnormal_fit():
    """An L-BFGS exit reported as FAILURE (e.g. ABNORMAL) is retried, then raised.

    This is the botorch behaviour ``_fit_multistart`` relies on: an ABNORMAL
    restart surfaces as ``ModelFittingError`` and is never selected.
    """
    train_X = torch.rand(8, 2, dtype=torch.float64)
    model = SingleTaskGP(train_X, train_X.sum(-1, keepdim=True))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    def abnormal(mll, closure=None, **kwargs):
        warnings.warn(
            "`scipy_minimize` terminated with status OptimizationStatus.FAILURE, "
            "displaying original message from `scipy.optimize.minimize`: "
            "ABNORMAL_TERMINATION_IN_LNSRCH",
            OptimizationWarning,
        )
        return OptimizationResult(fval=0.0, step=1, status=OptimizationStatus.FAILURE)

    with pytest.raises(ModelFittingError):
        fit_gpytorch_mll(mll, optimizer=abnormal)


def test_noise_to_jitter_ratio_is_ten():
    """The criterion-3 multiple is a config constant."""
    assert config.MIN_NOISE_TO_JITTER_RATIO == 10.0
