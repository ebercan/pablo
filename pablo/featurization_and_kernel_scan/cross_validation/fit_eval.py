"""Fit and evaluate a single GP model fold.

Provides ``fit_eval`` to take an assembled ``CellFeatures`` object, split it
into train and test sets, construct the requested model via
``AcyloinCondensationModel``, run multi-start L-BFGS, and compute predictive
metrics on the test set.

The 16 model classes (see ``AcyloinCondensationModel.SPECS``) are evaluated on
both yield and selectivity; both metric sets are always reported on the raw
fraction scale regardless of outcome transform. Yield-only classes (``st_y``,
``mapsaas_y``) emit empty selectivity bundles.
"""

from __future__ import annotations

import copy
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import gpytorch
import torch
from torch import Tensor

from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from pablo.utils.feature_assembly import CellFeatures
from pablo.utils.outcome_transforms import inverse as inverse_outcome
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    InverseGammaPrior,
    ModelSpec,
    assemble_kronecker_targets,
    build_mean_module,
    get_y_cs,
    get_y_yield,
)
from pablo.featurization_and_kernel_scan.core.product_models import CondProdModel, LogSumModel
from pablo.utils.metrics import (
    all_metrics,
    all_metrics_sample_based,
    pearson_spearman,
)


class ExcessiveJitterError(RuntimeError):
    """Raised when Cholesky jitter escalates to ``config.JITTER_THRESHOLD`` or above."""


class NoConvergedRestartError(RuntimeError):
    """Raised when no L-BFGS restart ends converged with a finite MLL.

    A restart whose L-BFGS exits ABNORMAL is never selected: ``fit_gpytorch_mll``
    (botorch 0.17.2) reads the resulting ``OptimizationWarning`` as a failed
    attempt, rolls the parameters back, retries from prior samples, and raises
    ``ModelFittingError`` once its attempts are spent, which propagates out of
    ``_fit_multistart``. A restart whose MLL is NaN or inf is skipped.
    """


class NoiseBelowJitterError(RuntimeError):
    """Raised when the fitted noise variance is below ``config.MIN_NOISE_TO_JITTER_RATIO``
    times the final Cholesky jitter.

    Both quantities are on the standardized scale the GP is fitted on. The final
    jitter is the largest jitter gpytorch added while factorising the training
    covariance at the selected hyperparameters, i.e. during prediction. gpytorch
    adds jitter only after a factorisation fails, so a fold with no jitter warning
    at that step has a final jitter of 0 and cannot fail this check. Checked for
    single-task models and for every GP of ``logsum_cy`` and ``condprod_cy``; for
    ``logsum_cy`` the smallest per-row noise counts.
    """


class NonFinitePredictionError(RuntimeError):
    """Raised when a test point's predictive mean or predictive variance is NaN or
    inf, or its predictive variance is not positive."""


_JITTER_RE = re.compile(r"jitter of ([\d.eE+\-]+)")


def _max_jitter_from_warnings(caught: list[warnings.WarningMessage]) -> float:
    """Parse the maximum Cholesky jitter level from captured warnings."""
    max_val = 0.0
    for w in caught:
        m = _JITTER_RE.search(str(w.message))
        if m is not None:
            max_val = max(max_val, float(m.group(1)))
    return max_val





@dataclass
class FoldResult:
    """Results from one Cross-Validation fold.

    ``status`` is ``"ok"`` for a successful fit.

    Every metric is recorded on the **raw fraction scale** for two quantities:
    yield (the unsuffixed fields) and selectivity (the ``*_sel`` fields). The
    yield-only classes (``st_y``, ``mapsaas_y``) leave the ``*_sel`` per-point
    lists empty and the ``*_sel`` scalars NaN.

    ``mu_pred_per_point`` / ``var_pred_per_point`` are the posterior predictive mean
    and total predictive variance (latent posterior variance + likelihood-noise
    variance) per test point, on the raw fraction scale. The noise term depends on the
    model family: single-task models (``st_y``, ``indep_*``, ``mapsaas_*``) add one
    homoskedastic ``noise * stdv**2``; multitask marginal paths (``icm_*`` / ``lcm_*``)
    use a ``MultitaskGaussianLikelihood`` with a *shared global* noise plus a *per-task*
    noise, so the term for task ``t`` is ``(noise + task_noises[t]) * stdvs[t]**2`` (see
    ``_metrics_marginal``). For sample-based paths (every non-raw outcome transform,
    and the composite target ``ratio_yc``) they are the sample mean and variance of the
    back-transformed samples, where that same global + per-task noise is injected into
    the latent draws before the back-transform (see ``_multitask_predictive_samples``).
    The product ``product_cs`` multiplies noise-free conversion and selectivity draws
    and adds one yield measurement noise (``_product_cs_yield_samples``); the
    ``logsum`` / ``condprod`` classes return their own predictive samples
    (``core/product_models.py``). ``msll`` is the mean standardized log loss against the constant ``N(ȳ_train, σ²_train)``
    baseline (same baseline as CRPSS); ``baseline_nlpd_per_point`` is that baseline's
    per-point NLPD.

    ``posterior_mean_pearson`` / ``posterior_mean_spearman`` are the Pearson and
    (average-rank) Spearman correlation between the raw-scale posterior means of the two
    modeled targets (``spec.targets``) across the test points. They are NaN for
    single-output classes (``st_y``, ``mapsaas_y``) and for folds with ``n_test < 2``
    (e.g. leave-one-out).
    """

    train_idx: list[int]
    test_idx: list[int]
    nlpd_per_point: list[float]
    crps_per_point: list[float]
    baseline_crps_per_point: list[float]
    baseline_nlpd_per_point: list[float]
    mu_pred_per_point: list[float]
    var_pred_per_point: list[float]
    crpss: float
    msll: float
    q2: float
    mae: float
    rmse: float
    calibration_ks: float
    nlpd_per_point_sel: list[float]
    crps_per_point_sel: list[float]
    baseline_crps_per_point_sel: list[float]
    baseline_nlpd_per_point_sel: list[float]
    mu_pred_per_point_sel: list[float]
    var_pred_per_point_sel: list[float]
    crpss_sel: float
    msll_sel: float
    q2_sel: float
    mae_sel: float
    rmse_sel: float
    calibration_ks_sel: float
    posterior_mean_pearson: float
    posterior_mean_spearman: float
    hyperparams: dict[str, float | list[float]]
    wall_time_s: float
    n_train: int
    n_test: int
    status: str = "ok"


def _init_noise_at_prior_mode(model) -> None:
    """Set every likelihood noise with an InverseGamma prior to that prior's mode β/(α+1).

    Noise values live on the standardized scale the prior is stated on.
    """
    for name, module, prior, closure, setting_closure in model.named_priors():
        if "noise" not in name or not isinstance(prior, InverseGammaPrior):
            continue
        if setting_closure is None:
            raise ValueError(f"noise prior {name!r} has no setting closure")
        mode = (prior.rate / (prior.concentration + 1.0)).to(closure(module))
        setting_closure(module, torch.full_like(closure(module), float(mode)))


def _floor_plateau_masks(model) -> dict[str, Tensor]:
    """Raw-parameter elements stuck on a ``GreaterThan`` floor.

    Returns:
        Map from ``state_dict`` key of a raw parameter to a bool mask of the
        elements whose raw value is below ``config.FLOOR_PLATEAU_RAW``, where the
        softplus gradient vanishes. Parameters with no such element are absent.
    """
    masks: dict[str, Tensor] = {}
    for name, param, constraint in model.named_parameters_and_constraints():
        if not isinstance(constraint, GreaterThan):
            continue
        mask = param.detach() < config.FLOOR_PLATEAU_RAW
        if bool(mask.any()):
            masks[name] = mask
    return masks


def _penalized_loss(model, mll, train_X) -> float:
    """Negative penalized MLL per training point, in train mode."""
    model.train()
    model.likelihood.train()
    with torch.no_grad():
        return -mll(model(train_X), model.train_targets).item()


def _fit_multistart(model, train_X, n_restarts, max_iter) -> None:
    """Multi-start L-BFGS fitting in-place, keeping the best state dict.

    Every likelihood noise with an InverseGamma prior first starts at that
    prior's mode, which defines the initial state. Restart 0 starts from the
    initial state; every later restart starts from it plus
    N(0, ``config.RESTART_JITTER_SD``²) on each raw parameter. Restarts are
    scored by the penalized MLL in train mode against ``model.train_targets``,
    the standardized targets the optimizer ran on. A restart ending with a
    non-finite MLL is logged to stderr and skipped.

    If the winner has raw parameters stuck on a ``GreaterThan`` floor (raw below
    ``config.FLOOR_PLATEAU_RAW``, see :func:`_floor_plateau_masks`), those
    elements are reset to their initial values and the model is refitted once;
    the refit replaces the winner when its penalized MLL is higher. The model
    is left in eval mode with the best state loaded.

    Args:
        model: A GPyTorch/BoTorch exact GP with its likelihood.
        train_X: ``n_train x d`` training inputs the model was built on.
        n_restarts: Number of L-BFGS restarts.
        max_iter: Max L-BFGS iterations per restart.

    Raises:
        NoConvergedRestartError: If no restart ends with a finite MLL.
        botorch.exceptions.ModelFittingError: If ``fit_gpytorch_mll`` spends all
            its attempts on a restart.
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _init_noise_at_prior_mode(model)
    init_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_state_dict = None

    for r in range(n_restarts):
        model.load_state_dict(init_state, keep_transforms=False)
        model.train()
        model.likelihood.train()
        if r > 0:
            with torch.no_grad():
                for param in mll.parameters():
                    param.add_(torch.randn_like(param) * config.RESTART_JITTER_SD)

        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": max_iter}})
        loss = _penalized_loss(model, mll, train_X)
        if not math.isfinite(loss):
            print(
                f"[_fit_multistart] restart {r}/{n_restarts} ended with a "
                f"non-finite MLL ({loss})",
                file=sys.stderr, flush=True,
            )
            continue
        if loss < best_loss:
            best_loss = loss
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        raise NoConvergedRestartError(
            f"none of {n_restarts} L-BFGS restarts converged to a finite MLL"
        )
    model.load_state_dict(best_state_dict, keep_transforms=False)

    masks = _floor_plateau_masks(model)
    if masks:
        escaped = copy.deepcopy(best_state_dict)
        for key, mask in masks.items():
            escaped[key] = torch.where(mask, init_state[key], escaped[key])
        model.load_state_dict(escaped, keep_transforms=False)
        model.train()
        model.likelihood.train()
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": max_iter}})
        loss = _penalized_loss(model, mll, train_X)
        if math.isfinite(loss) and loss < best_loss:
            best_state_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state_dict, keep_transforms=False)

    model.eval()
    model.likelihood.eval()


N_PRED_SAMPLES = 2000
"""Posterior-predictive sample count for the sample-based metric paths."""


def _empty_bundle() -> dict:
    """Metric bundle for a quantity a model does not predict (yield-only classes)."""
    nan = float("nan")
    return {
        "nlpd_per_point": [],
        "crps_per_point": [],
        "baseline_crps_per_point": [],
        "baseline_nlpd_per_point": [],
        "crpss": nan,
        "msll": nan,
        "q2": nan,
        "mae": nan,
        "rmse": nan,
        "calibration_ks": nan,
        "mu_pred_per_point": [],
        "var_pred_per_point": [],
    }


def _singletask_predictive_samples(model, test_X, outcome_transform) -> Tensor:
    """Raw-fraction posterior-predictive samples ``(S, n_test, 1)`` from a single GP."""
    posterior = model.posterior(test_X)
    latent = posterior.rsample(sample_shape=torch.Size([N_PRED_SAMPLES]))
    std_noise = model.likelihood.noise.sqrt().squeeze()
    stdvs = model.outcome_transform.stdvs.squeeze()
    predictive = latent + torch.randn_like(latent) * (std_noise * stdvs)
    return inverse_outcome(predictive, outcome_transform)


def _multitask_predictive_samples(model, test_X, outcome_transform) -> Tensor:
    """Raw-fraction joint posterior-predictive samples ``(S, n_test, T)`` from a multitask GP.

    A single joint ``rsample`` preserves the learned cross-task correlation, so the
    sample-based product ``Y = C·S`` and ratio ``S = Y/C`` respect it.
    """
    posterior = model.posterior(test_X)
    latent = posterior.rsample(sample_shape=torch.Size([N_PRED_SAMPLES]))
    stdvs = model.outcome_transform.stdvs.squeeze()
    global_noise = model.likelihood.noise.squeeze()
    task_noises = model.likelihood.task_noises.squeeze()
    per_task_std = (global_noise + task_noises).sqrt() * stdvs
    predictive = latent + torch.randn_like(latent) * per_task_std
    return inverse_outcome(predictive, outcome_transform)


def _metrics_direct(model, test_X, y_true, y_train, outcome_transform) -> dict:
    """Metrics for a directly-modeled target of a single GP.

    Closed-form Gaussian on the raw scale; sample-based under every other
    outcome transform, where the back-transformed posterior is non-Gaussian.
    ``y_true`` / ``y_train`` are on the raw fraction scale.
    """
    if outcome_transform == "raw":
        posterior = model.posterior(test_X)
        noise_var = model.likelihood.noise.squeeze() * (
            model.outcome_transform.stdvs.squeeze() ** 2
        )
        return all_metrics(
            y=y_true, mu=posterior.mean, var_post=posterior.variance,
            noise_var=noise_var, y_train=y_train,
        )
    samples = _singletask_predictive_samples(model, test_X, outcome_transform)
    return all_metrics_sample_based(y=y_true, samples=samples, y_train=y_train)


def _metrics_marginal(model, task_index, test_X, y_true, y_train, outcome_transform) -> dict:
    """Metrics for one task of a multitask GP.

    Closed-form on the raw scale, sample-based under every other transform.
    """
    if outcome_transform == "raw":
        posterior = model.posterior(test_X)
        stdvs = model.outcome_transform.stdvs.squeeze()
        global_noise = model.likelihood.noise.squeeze()
        task_noise = model.likelihood.task_noises.squeeze()[task_index]
        noise_var = (global_noise + task_noise) * (stdvs[task_index] ** 2)
        return all_metrics(
            y=y_true,
            mu=posterior.mean[..., task_index:task_index + 1],
            var_post=posterior.variance[..., task_index:task_index + 1],
            noise_var=noise_var, y_train=y_train,
        )
    samples = _multitask_predictive_samples(model, test_X, outcome_transform)[..., task_index]
    return all_metrics_sample_based(y=y_true, samples=samples, y_train=y_train)


def _modeled_slot_samples(model, spec, test_X, outcome_transform) -> Tensor:
    """Raw-fraction samples ``(S, n_test, 2)`` for the two modeled targets, in column order.

    For ``icm``/``lcm`` one joint sample is split into tasks (correlation preserved);
    for ``independent`` each member is sampled separately (uncorrelated, as designed).
    """
    if spec.architecture in ("icm", "lcm"):
        return _multitask_predictive_samples(model, test_X, outcome_transform)
    s0 = _singletask_predictive_samples(model[0], test_X, outcome_transform)
    s1 = _singletask_predictive_samples(model[1], test_X, outcome_transform)
    return torch.cat([s0, s1], dim=-1)


def _modeled_slot_latent_samples(model, spec, test_X, outcome_transform) -> Tensor:
    """Raw-fraction latent samples ``(S, n_test, 2)`` of the two modeled targets, no noise.

    Joint for ``icm``/``lcm``, member by member for ``independent``.
    """
    shape = torch.Size([N_PRED_SAMPLES])
    if spec.architecture in ("icm", "lcm"):
        return inverse_outcome(model.posterior(test_X).rsample(sample_shape=shape), outcome_transform)
    s0 = inverse_outcome(model[0].posterior(test_X).rsample(sample_shape=shape), outcome_transform)
    s1 = inverse_outcome(model[1].posterior(test_X).rsample(sample_shape=shape), outcome_transform)
    return torch.cat([s0, s1], dim=-1)


def _product_cs_yield_samples(latent_slots: Tensor) -> Tensor:
    """Yield of a new measurement from latent (conversion, selectivity) draws.

    ``Y = C·S + ε`` with ``ε ~ N(0, σ²_Y)``, ``σ²_Y`` the posterior mean of the raw
    yield replicate evidence. Conversion and selectivity carry no noise of their
    own here: the measured ``C`` and ``S = Y / C`` multiply back to the measured
    ``Y``, so a measurement adds the yield noise once.

    Args:
        latent_slots: ``(S, n_test, 2)`` raw-fraction latent conversion and
            selectivity.

    Returns:
        ``(S, n_test, 1)`` yield samples.
    """
    latent_yield = latent_slots[..., 0] * latent_slots[..., 1]
    noise_sd = math.sqrt(config.NOISE_EVIDENCE[("Y", "raw")].prior_mean)
    return (latent_yield + torch.randn_like(latent_yield) * noise_sd).unsqueeze(-1)


_PRODUCT_ARCHITECTURES: tuple[str, ...] = ("logsum", "condprod")
"""Architectures of ``core/product_models.py``; each model draws its own samples."""


def _modeled_slot_means(model, spec, test_X, outcome_transform) -> Tensor:
    """Raw-fraction posterior means ``(n_test, 2)`` of the two modeled targets.

    Column order matches ``spec.targets``. On the raw scale the closed-form posterior
    mean is used; under every other transform the back-transform is non-linear, so the
    sample mean of the back-transformed draws is used (mirroring the metric helpers).
    """
    if spec.architecture in _PRODUCT_ARCHITECTURES:
        samples = model.predictive_samples(test_X, N_PRED_SAMPLES)
        return torch.stack([samples.conversion.mean(dim=0), samples.yield_.mean(dim=0)], dim=-1)
    if outcome_transform != "raw":
        return _modeled_slot_samples(model, spec, test_X, outcome_transform).mean(dim=0)
    if spec.architecture in ("icm", "lcm"):
        return model.posterior(test_X).mean[..., 0:2]
    m0 = model[0].posterior(test_X).mean
    m1 = model[1].posterior(test_X).mean
    return torch.cat([m0, m1], dim=-1)


def _yield_bundle(model, spec, test_X, y_yield, y_train_yield, outcome_transform) -> dict:
    """Yield metric bundle (raw fraction scale) per ``spec.yield_src``."""
    if spec.yield_src == "direct":
        target = model if spec.architecture == "singletask" else model[spec.yield_index]
        return _metrics_direct(target, test_X, y_yield, y_train_yield, outcome_transform)
    if spec.yield_src == "marginal":
        return _metrics_marginal(
            model, spec.yield_index, test_X, y_yield, y_train_yield, outcome_transform
        )
    if spec.yield_src == "product_cs":
        slots = _modeled_slot_latent_samples(model, spec, test_X, outcome_transform)
        yield_samples = _product_cs_yield_samples(slots)
        return all_metrics_sample_based(y=y_yield, samples=yield_samples, y_train=y_train_yield)
    if spec.yield_src in _PRODUCT_ARCHITECTURES:
        samples = model.predictive_samples(test_X, N_PRED_SAMPLES)
        return all_metrics_sample_based(y=y_yield, samples=samples.yield_, y_train=y_train_yield)
    raise ValueError(f"Unknown yield_src {spec.yield_src!r}")


def _sel_bundle(model, spec, test_X, y_sel, y_train_sel, outcome_transform) -> dict:
    """Selectivity metric bundle (raw fraction scale) per ``spec.sel_src``."""
    if spec.sel_src == "none":
        return _empty_bundle()
    if spec.sel_src == "direct":
        return _metrics_direct(
            model[spec.sel_index], test_X, y_sel, y_train_sel, outcome_transform
        )
    if spec.sel_src == "marginal":
        return _metrics_marginal(
            model, spec.sel_index, test_X, y_sel, y_train_sel, outcome_transform
        )
    if spec.sel_src == "ratio_yc":
        slots = _modeled_slot_samples(model, spec, test_X, outcome_transform)
        y_samp, c_samp = slots[..., 0], slots[..., 1]
        c_floored = c_samp.clamp_min(config.CONVERSION_DIVISION_FLOOR)
        sel_samples = (y_samp / c_floored).clamp(0.0, 1.0).unsqueeze(-1)
        return all_metrics_sample_based(y=y_sel, samples=sel_samples, y_train=y_train_sel)
    if spec.sel_src in _PRODUCT_ARCHITECTURES:
        samples = model.predictive_samples(test_X, N_PRED_SAMPLES)
        return all_metrics_sample_based(y=y_sel, samples=samples.selectivity, y_train=y_train_sel)
    raise ValueError(f"Unknown sel_src {spec.sel_src!r}")


def _check_finite_predictions(bundle: dict, target: str) -> None:
    """Raise ``NonFinitePredictionError`` on a non-finite μ or σ, or σ ≤ 0.

    Args:
        bundle: A metric bundle; ``mu_pred_per_point`` and ``var_pred_per_point``
            are ``n_test``-long lists (empty for a target the model does not
            predict).
        target: Name of the target, for the error message.
    """
    mu = torch.tensor(bundle["mu_pred_per_point"], dtype=torch.float64)
    var = torch.tensor(bundle["var_pred_per_point"], dtype=torch.float64)
    bad = ~torch.isfinite(mu) | ~torch.isfinite(var) | (var <= 0.0)
    if bad.any():
        raise NonFinitePredictionError(
            f"{int(bad.sum())}/{bad.numel()} {target} test points have a non-finite "
            f"mean or a non-finite or non-positive predictive variance"
        )


def _fit_single(model, train_X, train_y, n_restarts, max_iter):
    _fit_multistart(model, train_X, n_restarts, max_iter)


def _fit_independent(model_pair, train_X, train_y_pair, n_restarts, max_iter):
    _fit_multistart(model_pair[0], train_X, n_restarts, max_iter)
    _fit_multistart(model_pair[1], train_X, n_restarts, max_iter)


def _fit_logsum(model: LogSumModel, train_X, train_y, n_restarts, max_iter):
    _fit_multistart(model.gp, model.train_x, n_restarts, max_iter)


def _fit_condprod(model: CondProdModel, train_X, train_y, n_restarts, max_iter):
    _fit_multistart(model.conversion_gp, train_X, n_restarts, max_iter)
    _fit_multistart(model.selectivity_gp, model.selectivity_train_x, n_restarts, max_iter)


_FIT_DISPATCH: dict[str, Callable] = {
    "singletask":  _fit_single,
    "independent": _fit_independent,
    "icm":         _fit_single,
    "lcm":         _fit_single,
    "logsum":      _fit_logsum,
    "condprod":    _fit_condprod,
}


def _smallest_fitted_noise(model, spec: ModelSpec) -> float | None:
    """Smallest fitted noise variance on the scale the GP is fitted on, or ``None``.

    ``None`` for the architectures the noise-to-jitter check does not cover
    (``independent``, ``icm``, ``lcm``).
    """
    if spec.architecture == "singletask":
        return model.likelihood.noise.min().item()
    if spec.architecture == "logsum":
        return model.noise_module().training_variances().min().item()
    if spec.architecture == "condprod":
        return min(
            model.conversion_gp.likelihood.noise.min().item(),
            model.selectivity_gp.likelihood.noise.min().item(),
        )
    return None


def _named_hyperparameters(model, spec: ModelSpec) -> dict[str, float | list[float]]:
    """Every fitted parameter, keyed by name, prefixed by member for multi-model classes."""
    if spec.architecture == "independent":
        members = [("task0.", model[0]), ("task1.", model[1])]
    elif spec.architecture == "logsum":
        members = [("", model.gp)]
    elif spec.architecture == "condprod":
        members = [("conversion.", model.conversion_gp), ("selectivity.", model.selectivity_gp)]
    else:
        members = [("", model)]
    return {
        prefix + name: p.detach().cpu().squeeze().tolist()
        for prefix, member in members
        for name, p in member.named_parameters()
    }


def _target_train_tensor(
    cell_features: CellFeatures, target: str, outcome_transform: str, train_idx: list[int]
) -> Tensor:
    """Train-subset tensor for one of ``"Y"`` / ``"C"`` / ``"S"`` on the chosen scale."""
    y_yield = get_y_yield(cell_features, outcome_transform)
    y_c, y_s = get_y_cs(cell_features, outcome_transform)
    return {"Y": y_yield, "C": y_c, "S": y_s}[target][train_idx]


def _build_train_targets(
    cell_features: CellFeatures,
    train_idx: list[int],
    spec: ModelSpec,
    outcome_transform: str,
) -> Tensor | tuple[Tensor, Tensor]:
    """Assemble train_y for the fit phase. Mirrors what the model class uses internally.

    Returned shape matches ``_FIT_DISPATCH[architecture]`` expectations: a tuple of
    ``(y0, y1)`` for ``independent``, a single tensor otherwise (the ``(N, 2)`` stack
    for ``icm``/``lcm``, the yield column for ``singletask``).
    """
    if spec.architecture == "singletask":
        return _target_train_tensor(cell_features, spec.targets[0], outcome_transform, train_idx)
    if spec.architecture == "independent":
        return (
            _target_train_tensor(cell_features, spec.targets[0], outcome_transform, train_idx),
            _target_train_tensor(cell_features, spec.targets[1], outcome_transform, train_idx),
        )
    return assemble_kronecker_targets(
        cell_features, spec.targets, outcome_transform, train_idx
    )


def fit_eval(
    cell_features: CellFeatures,
    train_idx: list[int],
    test_idx: list[int],
    kernel_structure: str,
    outcome_transform: str,
    model_class: str,
    base_kernel: str,
    mean_function: str,
    n_restarts: int = config.N_RESTARTS,
    max_iter: int = config.MAX_ITER,
    prior_scheme: str | None = None,
) -> FoldResult:
    """Fit a GP and evaluate on a test fold.

    Args:
        cell_features: Assembled features for all data points.
        train_idx: Indices of the training points.
        test_idx: Indices of the test points.
        kernel_structure: One of ``config.KERNEL_STRUCTURES``, or ``"Vanilla"``.
        outcome_transform: One of ``config.OUTCOME_TRANSFORMS`` (``"raw"``,
            ``"logit"``, ``"arcsin_sqrt"``, ``"probit"``).
        model_class: One of the IDs in ``config.MODEL_CLASSES`` (``"st_y"``,
            the ``"indep_*"`` / ``"icm_*"`` / ``"lcm_*"`` / ``"mapsaas_*"``
            families, ``"logsum_cy"``, ``"condprod_cy"``); see
            ``AcyloinCondensationModel.SPECS``.
        base_kernel: ``"matern52"``, ``"matern32"``, ``"matern12"``, ``"rbf"``,
            ``"tanimoto"``, or ``"minmax"``.
        mean_function: One of ``config.MEAN_FUNCTIONS``. Vanilla kernels ignore it.
        n_restarts: Number of L-BFGS restarts. Production callers pass
            ``config.N_RESTARTS``; tests can pass a smaller value for speed.
        max_iter: Max L-BFGS iterations per restart. Production callers pass
            ``config.MAX_ITER``; tests can pass a smaller value for speed.
        prior_scheme: A key of ``config.PRIOR_SCHEMES`` ("P1", "P2", "P3"), or
            ``None``. ``cell_features.partition`` must already be built at that
            scheme's granularity.

    Returns:
        FoldResult with predictive metrics and fitted hyperparameters.

    Raises:
        ValueError: If ``model_class`` is not a key of
            ``AcyloinCondensationModel.SPECS``.
        NoConvergedRestartError: If no L-BFGS restart of a fitted model ends
            with a finite MLL.
        ExcessiveJitterError: If Cholesky jitter during fitting or prediction
            reaches ``config.JITTER_THRESHOLD``.
        NoiseBelowJitterError: For single-task, ``logsum_cy`` and ``condprod_cy``
            models, if a fitted noise variance is below ``config.MIN_NOISE_TO_JITTER_RATIO`` times the
            final jitter.
        NonFinitePredictionError: If a yield or selectivity test point has a
            non-finite predictive mean or a non-finite or non-positive
            predictive variance.
    """
    t0 = time.time()
    spec = AcyloinCondensationModel.get_spec(model_class)

    train_X = cell_features.X[train_idx]
    test_X = cell_features.X[test_idx]
    test_y = cell_features.y_raw[test_idx]
    train_y_raw = cell_features.y_raw[train_idx]
    test_sel = cell_features.y_selectivity_raw[test_idx]
    train_sel_raw = cell_features.y_selectivity_raw[train_idx]

    lengthscale_scale = config.LENGTHSCALE_LOGNORMAL_SCALE

    mean_module = build_mean_module(
        mean_function, cell_features, prior_scale=config.MEAN_MODULE_NORMAL_SCALE, train_x=train_X
    )

    model_builder = AcyloinCondensationModel(
        cell_features=cell_features,
        train_idx=train_idx,
        kernel_structure=kernel_structure,
        base_kernel_type=base_kernel,
        outcome_transform=outcome_transform,
        lengthscale_scale=lengthscale_scale,
        prior_scheme=prior_scheme,
        mean_function=mean_function,
    )
    model = model_builder.build(model_class, mean_module)
    train_y = _build_train_targets(cell_features, train_idx, spec, outcome_transform)

    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", message="A not p.d.")
        with gpytorch.settings.cholesky_jitter(double_value=config.CHOLESKY_JITTER_F64):
            _FIT_DISPATCH[spec.architecture](
                model, train_X, train_y, n_restarts, max_iter
            )

            with warnings.catch_warnings(record=True) as caught_final, torch.no_grad():
                warnings.filterwarnings("always", message="A not p.d.")
                res_y = _yield_bundle(
                    model, spec, test_X, test_y, train_y_raw, outcome_transform,
                )
                res_s = _sel_bundle(
                    model, spec, test_X, test_sel, train_sel_raw, outcome_transform,
                )

                pm_pearson, pm_spearman = float("nan"), float("nan")
                if spec.architecture != "singletask" and len(test_idx) >= 2:
                    slot_means = _modeled_slot_means(model, spec, test_X, outcome_transform)
                    pm_pearson, pm_spearman = pearson_spearman(
                        slot_means[:, 0], slot_means[:, 1]
                    )

    final_jitter = _max_jitter_from_warnings(caught_final)
    max_jitter = max(_max_jitter_from_warnings(caught), final_jitter)
    if max_jitter >= config.JITTER_THRESHOLD:
        raise ExcessiveJitterError(
            f"Cholesky jitter reached {max_jitter:.1e} "
            f"(threshold {config.JITTER_THRESHOLD:.1e})"
        )
    noise = _smallest_fitted_noise(model, spec)
    if noise is not None and noise < config.MIN_NOISE_TO_JITTER_RATIO * final_jitter:
        raise NoiseBelowJitterError(
            f"fitted noise variance {noise:.2e} is below "
            f"{config.MIN_NOISE_TO_JITTER_RATIO:g} × final jitter {final_jitter:.1e}"
        )
    _check_finite_predictions(res_y, "yield")
    _check_finite_predictions(res_s, "selectivity")

    hyperparams = _named_hyperparameters(model, spec)

    t1 = time.time()

    return FoldResult(
        train_idx=train_idx,
        test_idx=test_idx,
        nlpd_per_point=res_y["nlpd_per_point"],
        crps_per_point=res_y["crps_per_point"],
        baseline_crps_per_point=res_y["baseline_crps_per_point"],
        baseline_nlpd_per_point=res_y["baseline_nlpd_per_point"],
        mu_pred_per_point=res_y["mu_pred_per_point"],
        var_pred_per_point=res_y["var_pred_per_point"],
        crpss=res_y["crpss"],
        msll=res_y["msll"],
        q2=res_y["q2"],
        mae=res_y["mae"],
        rmse=res_y["rmse"],
        calibration_ks=res_y["calibration_ks"],
        nlpd_per_point_sel=res_s["nlpd_per_point"],
        crps_per_point_sel=res_s["crps_per_point"],
        baseline_crps_per_point_sel=res_s["baseline_crps_per_point"],
        baseline_nlpd_per_point_sel=res_s["baseline_nlpd_per_point"],
        mu_pred_per_point_sel=res_s["mu_pred_per_point"],
        var_pred_per_point_sel=res_s["var_pred_per_point"],
        crpss_sel=res_s["crpss"],
        msll_sel=res_s["msll"],
        q2_sel=res_s["q2"],
        mae_sel=res_s["mae"],
        rmse_sel=res_s["rmse"],
        calibration_ks_sel=res_s["calibration_ks"],
        posterior_mean_pearson=pm_pearson,
        posterior_mean_spearman=pm_spearman,
        hyperparams=hyperparams,
        wall_time_s=t1 - t0,
        n_train=len(train_idx),
        n_test=len(test_idx),
        status="ok",
    )
