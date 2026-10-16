"""Noise structure of the scan's single-task GP: replicate evidence, CV and a batch check.

The scan's GP has homoskedastic observation noise on the scale it is trained on.
This tool measures whether a noise variance that depends on the yield level fits
the data better, predicts held-out experiments better, or changes the batch the
acquisition proposes. The family is a variance function of the raw yield fraction,

    Var(y_i) = sigma^2 * [p_i (1 - p_i)]^theta,     p_i = (y_i + eps) / (1 + 2 eps),

so ``theta = 0`` is homoskedastic on the raw scale and, to first order, ``1`` on the
arcsine scale and ``2`` on the logit scale. A GP trained on ``z = T(y)`` gets, by
the delta method, ``tau_i^2 = sigma^2 * shape_i`` with
``shape_i = [p_i (1 - p_i)]^theta * T'(y_i)^2``, divided by its geometric mean over
the training rows. The variant ``"own"`` is ``shape_i = 1``: the production model.
Replicate members are evaluated at their group mean and test points at the
posterior mean. sigma^2 carries the replicate prior
``InvGamma(nu/2, S_shape/2)`` with ``S_shape = sum_g sum_j (z_gj - zbar_g)^2 / shape_g``,
which is the production prior of ``config.NOISE_EVIDENCE`` when every shape is 1.
Like that prior it pools every replicate group, those of a test fold included.

Tasks (``list`` prints them):

    cv        one fold of ``primary`` or ``leave_one_out`` for one configuration,
              every variant of that configuration fitted on the fold's training
              rows with the fold seed of the scan's scheduler
    profile   one configuration fitted on all rows at every theta of
              ``PROFILE_THETAS`` (and ``"own"``)
    decision  one (configuration, variant, optimizer seed): the q = 6 batch the
              variant's qLogNEI proposes under the production constraints, and
              that batch's value under the ``"own"`` model's qLogNEI

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.cross_validation.noise_structure <subcommand>

Subcommands::

    replicates            replicate-only evidence over theta (no GP)
    list                  print the tasks
    run --task N          run task N, writing results/noise_structure/task_<N>.json
    submit [--tasks L]    sbatch one array task per task (Mechthild)
    summarize             CV, profile and decision tables from the task files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction, qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.operators import DiagLinearOperator
from scipy import stats
from torch import Tensor

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.build_experiments import COLUMN_ORDER, _base_active
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import build_partition
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    InverseGammaPrior,
    build_mean_module,
    get_y_yield,
)
from pablo.featurization_and_kernel_scan.core.replicates import normalize_label, replicate_anchor_values
from pablo.featurization_and_kernel_scan.cross_validation import scheduler
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import (
    N_PRED_SAMPLES,
    _fit_multistart,
    _max_jitter_from_warnings,
    all_metrics,
    all_metrics_sample_based,
)
from pablo.main import GA_SETTINGS, TEMPERATURE_PARTITION, load_available_agents
from pablo.optimize import optimize_acqf_mixed_GA
from pablo.utils.feature_assembly import LOGIT_EPS, CellFeatures, assemble_features
from pablo.utils.outcome_transforms import forward, inverse

CONFIGURATIONS: dict[str, dict] = {
    "probit": dict(
        outer_cell_id=1339, cat_feat="grover_pca", solv_feat="Hansen", base_feat="grover_pca",
        outcome_transform="probit",
    ),
    "arcsin_sqrt": dict(
        outer_cell_id=2235, cat_feat="grover_pca", solv_feat="Hansen", base_feat="steric",
        outcome_transform="arcsin_sqrt",
    ),
    "raw": dict(
        outer_cell_id=1947, cat_feat="grover_pca", solv_feat="Gutmann", base_feat="OHE",
        outcome_transform="raw",
    ),
}
"""The ``cross_1`` cell with the highest gem per outcome transform, by name.

Every one is ``KS1`` / ``matern52`` / ``constant`` / ``st_y`` / ``P2`` with no form
one-hot on the catalyst block (``CONFIG_COMMON``). ``outer_cell_id`` is the cell's
id in ``runs/cross_1/inputs.json``; it seeds every fold exactly as the scheduler
seeded that cell."""

CONFIG_COMMON: dict = dict(
    kernel_structure="KS1", base_kernel="matern52", mean_function="constant",
    model_class="st_y", prior_scheme="P2", inner_id=0,
)
"""Axes shared by every configuration of ``CONFIGURATIONS``."""

OWN_THETA: dict[str, float] = {"raw": 0.0, "arcsin_sqrt": 1.0, "logit": 2.0}
"""theta at which the family is homoskedastic on that outcome scale exactly.

``probit`` has none: ``phi(Phi^-1(p))^2`` is not a power of ``p (1 - p)``, so the
probit configuration carries ``"own"`` as a separate variant."""

CV_THETAS: tuple[float, ...] = (0.0, 1.0, 2.0)
"""theta values every configuration is cross-validated at."""

CV_STUDIES: tuple[str, ...] = ("primary", "leave_one_out")
"""The two studies the gem is the minimum over."""

PROFILE_THETAS: tuple[float, ...] = tuple(float(t) for t in np.arange(-1.0, 3.0001, 0.25))
"""theta grid of the full-data profile."""

DECISION_SEEDS_OWN: tuple[int, ...] = (0, 1, 2)
"""Optimizer seeds of the ``"own"`` batch; their spread is the optimizer's own noise."""

NOISE_SCALE_VARIANTS: dict[str, float] = {"sigma_x3": 9.0, "sigma_div3": 1.0 / 9.0}
"""Homoskedastic variants with sigma^2 frozen at this multiple of the replicate
prior mean: sigma times 3 and sigma divided by 3."""

FIXED_TIME_H: float = 20.0
"""Reaction time the production batch pins (``pablo.main.generate_batch``)."""

Q_BATCH: int = 6
"""Batch size of the production campaign."""

N_AF_SAMPLES: int = 128
"""QMC samples of every qLogNEI; one fixed seed, so two batches are compared on the
same base samples."""

AF_SEED: int = 0
"""Seed of the qLogNEI QMC sampler."""

AVAILABILITY_COLUMN: str = "Next Batch"
"""Availability column of the design sheet the batch is proposed for."""

EXCEL_PATH: Path = paths.PKG_ROOT.parent / "nhc_doe.xlsx"
"""The design sheet; read for availability only."""

N_BOOTSTRAP: int = 2000
"""Fold-bootstrap resamples behind every paired interval in ``summarize``."""

JOB_NAME = "feat_noise_structure"
PARTITION = "short"
TIME_LIMIT = "02:00:00"
MODULE = "pablo.featurization_and_kernel_scan.cross_validation.noise_structure"


def configuration(name: str) -> dict:
    """The full configuration record of one ``CONFIGURATIONS`` entry."""
    return {"name": name, **CONFIGURATIONS[name], **CONFIG_COMMON}


def cv_variants(name: str) -> list[float | str]:
    """Variants cross-validated for one configuration, the production one included."""
    transform = CONFIGURATIONS[name]["outcome_transform"]
    return list(CV_THETAS) + ([] if transform in OWN_THETA else ["own"])


def production_variant(name: str) -> float | str:
    """The variant that is the production model of a configuration."""
    transform = CONFIGURATIONS[name]["outcome_transform"]
    return OWN_THETA.get(transform, "own")


def decision_variants(name: str) -> list[tuple[float | str, int]]:
    """``(variant, optimizer seed)`` of every decision task of one configuration.

    ``"own"`` at every seed of ``DECISION_SEEDS_OWN``; the noise-scale variants and
    the two theta values of ``CV_THETAS`` that are not the production one at seed 0.
    """
    production = production_variant(name)
    alternatives = [t for t in CV_THETAS if t != production]
    if production == "own":
        alternatives = [CV_THETAS[0], CV_THETAS[-1]]
    return (
        [("own", seed) for seed in DECISION_SEEDS_OWN]
        + [(variant, 0) for variant in NOISE_SCALE_VARIANTS]
        + [(theta, 0) for theta in alternatives]
    )


def tasks() -> list[dict]:
    """Every task, ``cv`` first, then ``profile``, then ``decision``.

    Returns:
        One record per task index.
    """
    out: list[dict] = []
    splits = json.loads(paths.CV_SPLITS_JSON.read_text())
    for name in CONFIGURATIONS:
        for block in splits:
            if block["split_type"] not in CV_STUDIES:
                continue
            for fold in range(len(block["folds"])):
                out.append({"kind": "cv", "config": name, "split_type": block["split_type"],
                            "repeat": block["repeat"], "fold": fold})
    for name in CONFIGURATIONS:
        out.append({"kind": "profile", "config": name})
    for name in CONFIGURATIONS:
        for variant, seed in decision_variants(name):
            out.append({"kind": "decision", "config": name, "variant": variant, "seed": seed})
    return out


def fold_seed(cfg: dict, split_type: str, repeat: int, fold: int) -> int:
    """The seed ``scheduler._run_inner_cell`` gives this fold of this cell."""
    type_offset = int(hashlib.sha256(split_type.encode()).hexdigest()[:8], 16)
    return (
        cfg["outer_cell_id"] * 1_000_000_000 + cfg["inner_id"] * 1_000_000
        + type_offset + repeat * 100 + fold
    ) % (2 ** 32)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_cell(cfg: dict) -> CellFeatures:
    """Assemble the experiments of one configuration with its prior-scheme partition."""
    cell = assemble_features(
        cfg["cat_feat"], cfg["solv_feat"], cfg["base_feat"], paths.EXPERIMENTS_PARQUET,
        paths.FEAT_DIR, libraries_root=paths.LIB_DIR,
    )
    cell.partition = build_partition(
        cell.feature_dims, mode=config.PRIOR_SCHEMES[cfg["prior_scheme"]].partition_mode
    )
    return cell


def squash(y: Tensor) -> Tensor:
    """The transforms' boundary guard ``(y + eps) / (1 + 2 eps)``, elementwise."""
    return (y + LOGIT_EPS) / (1.0 + 2.0 * LOGIT_EPS)


def transform_derivative(y: Tensor, transform: str) -> Tensor:
    """``dT/dy`` of an outcome transform at raw fractions ``y``, elementwise, exact.

    Args:
        y: Raw yield fractions in ``[0, 1]``, any shape.
        transform: A name of ``pablo.utils.outcome_transforms``.

    Returns:
        Tensor of the shape of ``y``.
    """
    with torch.enable_grad():
        leaf = y.detach().clone().requires_grad_(True)
        forward(leaf, transform).sum().backward()
    return leaf.grad.detach()


def unnormalized_shape(y: Tensor, transform: str, theta: float | str) -> Tensor:
    """``[p (1 - p)]^theta * T'(y)^2`` at raw fractions ``y``; ones for ``"own"``.

    Args:
        y: Raw yield fractions, any shape; clamped to ``[0, 1]``.
        transform: The model's outcome transform.
        theta: Variance-function exponent, or ``"own"``.

    Returns:
        Tensor of the shape of ``y``.
    """
    if theta == "own":
        return torch.ones_like(y)
    y = y.clamp(0.0, 1.0)
    p = squash(y)
    return (p * (1.0 - p)) ** float(theta) * transform_derivative(y, transform) ** 2


def shaped_replicate_prior(
    z: Tensor, group_ids: np.ndarray, shape: Tensor
) -> tuple[float, float]:
    """``(alpha, beta)`` of the replicate prior on sigma^2 for given row shapes.

    Args:
        z: ``(n,)`` targets on the model scale.
        group_ids: ``(n,)`` replicate-group ids, ``-1`` for singletons.
        shape: ``(n,)`` normalized shapes; constant within a group.

    Returns:
        ``(nu / 2, S_shape / 2)`` in squared model-scale units per unit shape.
    """
    sum_squares, dof = 0.0, 0
    for gid in sorted(set(group_ids.tolist()) - {-1}):
        mask = torch.as_tensor(group_ids == gid)
        members = z[mask]
        sum_squares += float(((members - members.mean()) ** 2).sum() / shape[mask][0])
        dof += int(mask.sum()) - 1
    return dof / 2.0, sum_squares / 2.0


class ShapedNoise(HomoskedasticNoise):
    """Diagonal observation noise ``sigma^2 * shape_i`` on the training rows.

    sigma^2 is the learnable homoskedastic noise of the parent class, or a frozen
    value. Only the training rows have shapes; test-point noise is added by the
    caller.

    Args:
        point_shape: ``(n_train,)`` normalized shapes.
        noise_prior: Prior on sigma^2 (standardized scale).
        fixed_noise: If given, sigma^2 is frozen at this value.
    """

    def __init__(self, point_shape: Tensor, noise_prior, fixed_noise: float | None = None) -> None:
        super().__init__(noise_prior=noise_prior)
        self.register_buffer("point_shape", point_shape)
        self.fixed_noise = fixed_noise
        if fixed_noise is not None:
            self.raw_noise.requires_grad_(False)

    @property
    def noise(self) -> Tensor:
        if self.fixed_noise is not None:
            return self.raw_noise.new_tensor([self.fixed_noise])
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self._set_noise(value)

    def forward(self, *params, shape=None, **kwargs):
        n = shape[-1] if shape is not None else params[0].shape[-2]
        if n != self.point_shape.shape[-1]:
            raise RuntimeError(f"ShapedNoise holds {self.point_shape.shape[-1]} shapes, asked for {n} rows")
        return DiagLinearOperator(self.noise.squeeze(-1) * self.point_shape)


@dataclass
class VariantFit:
    """A fitted variant and what prediction needs from it.

    Attributes:
        model: The fitted ``SingleTaskGP`` in eval mode.
        shape_norm: Geometric mean the unnormalized shapes were divided by.
        sigma2_std: Fitted sigma^2 on the standardized scale.
        var_z: Variance of the training targets on the model scale.
        objective: Penalized log marginal likelihood at the fit, summed over
            training rows (``ExactMarginalLogLikelihood`` times ``n``).
        prior_rate_std: Rate of the sigma^2 prior on the standardized scale.
        jitter: Largest Cholesky jitter added while fitting.
    """

    model: object
    shape_norm: Tensor
    sigma2_std: float
    var_z: float
    objective: float
    prior_rate_std: float
    jitter: float


def build_variant_model(
    cell: CellFeatures, cfg: dict, train_idx: list[int], theta: float | str, seed: int,
    noise_multiple: float | None = None,
) -> tuple[object, Tensor, float, float]:
    """The configuration's st_y GP with noise ``sigma^2 * shape(theta)``, unfitted.

    The model is the production one (``AcyloinCondensationModel``) with its
    likelihood noise replaced by :class:`ShapedNoise`, built after seeding with
    ``seed``.

    Args:
        cell: Assembled features of the configuration.
        cfg: A record of :func:`configuration`.
        train_idx: Training rows.
        theta: Variance-function exponent, or ``"own"``.
        seed: Seed set before the model is built.
        noise_multiple: If given, sigma^2 is frozen at this multiple of its prior
            mean instead of fitted.

    Returns:
        ``(model, shape_norm, var_z, prior_rate_std)``.
    """
    transform = cfg["outcome_transform"]
    group_ids = np.asarray(cell.replicate_id)
    y_anchor = replicate_anchor_values(cell.y_raw.squeeze(-1), group_ids)
    shape_un = unnormalized_shape(y_anchor, transform, theta)
    norm = torch.exp(torch.log(shape_un[train_idx]).mean())
    shape = shape_un / norm

    z = get_y_yield(cell, transform).squeeze(-1)
    var_z = float(z[train_idx].var())
    alpha, beta = shaped_replicate_prior(z, group_ids, shape)
    rate_std = beta / var_z

    _seed_all(seed)
    mean_module = build_mean_module(
        cfg["mean_function"], cell, prior_scale=config.MEAN_MODULE_NORMAL_SCALE,
        train_x=cell.X[train_idx],
    )
    builder = AcyloinCondensationModel(
        cell_features=cell, train_idx=train_idx, kernel_structure=cfg["kernel_structure"],
        base_kernel_type=cfg["base_kernel"], outcome_transform=transform,
        lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE, prior_scheme=cfg["prior_scheme"],
        mean_function=cfg["mean_function"],
    )
    model = builder.build(cfg["model_class"], mean_module)
    prior = InverseGammaPrior(
        concentration=torch.tensor(alpha, dtype=torch.float64),
        rate=torch.tensor(rate_std, dtype=torch.float64),
    )
    fixed = None if noise_multiple is None else noise_multiple * rate_std / (alpha - 1.0)
    model.likelihood.noise_covar = ShapedNoise(shape[train_idx].to(torch.float64), prior, fixed)
    return model, norm, var_z, rate_std


def fit_variant(
    cell: CellFeatures, cfg: dict, train_idx: list[int], theta: float | str, seed: int,
    noise_multiple: float | None = None,
) -> VariantFit:
    """Fit :func:`build_variant_model` by the scan's ``_fit_multistart``.

    Args:
        cell: Assembled features of the configuration.
        cfg: A record of :func:`configuration`.
        train_idx: Training rows.
        theta: Variance-function exponent, or ``"own"``.
        seed: Seed set before the model is built.
        noise_multiple: If given, sigma^2 is frozen at this multiple of its prior
            mean instead of fitted.

    Returns:
        The fitted variant.
    """
    model, norm, var_z, rate_std = build_variant_model(
        cell, cfg, train_idx, theta, seed, noise_multiple
    )
    train_x = cell.X[train_idx]
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", message="A not p.d.")
        with gpytorch.settings.cholesky_jitter(double_value=config.CHOLESKY_JITTER_F64):
            _fit_multistart(model, train_x, config.N_RESTARTS, config.MAX_ITER)
            model.train()
            model.likelihood.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with torch.no_grad():
                objective = float(mll(model(train_x), model.train_targets)) * len(train_idx)
            model.eval()
            model.likelihood.eval()
    return VariantFit(
        model=model, shape_norm=norm, sigma2_std=float(model.likelihood.noise_covar.noise),
        var_z=var_z, objective=objective, prior_rate_std=rate_std,
        jitter=_max_jitter_from_warnings(caught),
    )


def predictive_noise_variance(fit: VariantFit, mu_model: Tensor, transform: str, theta: float | str) -> Tensor:
    """Observation-noise variance on the model scale at test points.

    Args:
        fit: The fitted variant.
        mu_model: ``(n_test,)`` posterior mean on the model scale.
        transform: The model's outcome transform.
        theta: The variant's exponent, or ``"own"``.

    Returns:
        ``(n_test,)`` variances.
    """
    y_hat = inverse(mu_model, transform).clamp(0.0, 1.0)
    shape = unnormalized_shape(y_hat, transform, theta) / fit.shape_norm
    return fit.sigma2_std * fit.var_z * shape


def predict_metrics(
    fit: VariantFit, cell: CellFeatures, cfg: dict, train_idx: list[int],
    test_idx: list[int], theta: float | str,
) -> dict:
    """Raw-scale metric bundle of one variant on one test fold.

    Mirrors ``fit_eval._metrics_direct``: closed-form Gaussian for ``raw``,
    back-transformed predictive samples otherwise, with the variant's own
    test-point noise.
    """
    transform = cfg["outcome_transform"]
    test_x = cell.X[test_idx]
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(double_value=config.CHOLESKY_JITTER_F64):
        posterior = fit.model.posterior(test_x)
        noise = predictive_noise_variance(fit, posterior.mean.squeeze(-1), transform, theta)
        if transform == "raw":
            return all_metrics(
                y=cell.y_raw[test_idx], mu=posterior.mean, var_post=posterior.variance,
                noise_var=noise.unsqueeze(-1), y_train=cell.y_raw[train_idx],
            )
        latent = posterior.rsample(sample_shape=torch.Size([N_PRED_SAMPLES]))
        predictive = latent + torch.randn_like(latent) * noise.sqrt().unsqueeze(-1)
        return all_metrics_sample_based(
            y=cell.y_raw[test_idx], samples=inverse(predictive, transform),
            y_train=cell.y_raw[train_idx],
        )


def run_cv(task: dict) -> dict:
    """Every variant of one configuration on one CV fold."""
    cfg = configuration(task["config"])
    block = next(
        b for b in json.loads(paths.CV_SPLITS_JSON.read_text())
        if b["split_type"] == task["split_type"] and b["repeat"] == task["repeat"]
    )
    fold = block["folds"][task["fold"]]
    seed = fold_seed(cfg, task["split_type"], task["repeat"], task["fold"])
    cell = build_cell(cfg)
    records = []
    for theta in cv_variants(task["config"]):
        t0 = time.time()
        fit = fit_variant(cell, cfg, fold["train"], theta, seed)
        _seed_all(seed + 1)
        metrics = predict_metrics(fit, cell, cfg, fold["train"], fold["test"], theta)
        records.append({
            "theta": theta, "objective": fit.objective, "sigma2_std": fit.sigma2_std,
            "var_z": fit.var_z, "jitter": fit.jitter, "wall_s": time.time() - t0,
            **{k: metrics[k] for k in (
                "crps_per_point", "baseline_crps_per_point", "nlpd_per_point",
                "baseline_nlpd_per_point", "mu_pred_per_point", "var_pred_per_point",
            )},
        })
    return {"test_idx": fold["test"], "variants": records}


def run_profile(task: dict) -> dict:
    """One configuration fitted on all rows at every theta of the profile grid."""
    cfg = configuration(task["config"])
    cell = build_cell(cfg)
    all_rows = list(range(cell.X.shape[0]))
    seed = cfg["outer_cell_id"]
    thetas: list[float | str] = list(PROFILE_THETAS)
    if production_variant(task["config"]) == "own":
        thetas.append("own")
    records = []
    for theta in thetas:
        fit = fit_variant(cell, cfg, all_rows, theta, seed)
        records.append({
            "theta": theta, "objective": fit.objective, "sigma2_std": fit.sigma2_std,
            "var_z": fit.var_z, "prior_rate_std": fit.prior_rate_std, "jitter": fit.jitter,
        })
    return {"variants": records}


@dataclass
class CandidateTables:
    """The production search space in the optimizer's component layout.

    Optimizer rows are ``[cont (4) | cat (d_cat) | form_id | base_active | sol | base]``;
    :class:`ModelLayoutAcquisition` maps them onto the scan's
    ``[cat | solv | base | base_active | cont | form_id]``.

    Attributes:
        cat: ``(n_cat, d_cat + 2)`` catalyst rows with ``form_id`` and ``base_active``.
        sol: ``(n_sol, d_solv)`` solvent rows.
        base: ``(n_base, d_base)`` base rows.
        cat_is_salt: ``(n_cat,)`` bool, the base slice is zeroed where False.
        cat_ids: Catalyst id of each ``cat`` row.
        solvents: Solvent name of each ``sol`` row.
        bases: Base comment of each ``base`` row.
        d_cat, d_solv, d_base: Block widths.
    """

    cat: Tensor
    sol: Tensor
    base: Tensor
    cat_is_salt: Tensor
    cat_ids: list[int]
    solvents: list[str]
    bases: list[str]
    d_cat: int
    d_solv: int
    d_base: int


def _available_components() -> tuple[list[int], list[str], list[str]]:
    """Available catalyst ids, solvent library names and base comments."""
    cat_ids, sol_ids, base_ids = load_available_agents(EXCEL_PATH, AVAILABILITY_COLUMN)
    sol_ref = pd.read_excel(EXCEL_PATH, sheet_name="Solvent References")
    base_ref = pd.read_excel(EXCEL_PATH, sheet_name="Base References")
    sol_lib = pd.read_parquet(paths.SOLVENTS_PARQUET)
    base_lib = pd.read_parquet(paths.BASES_PARQUET)
    by_label = {normalize_label(n): n for n in sol_lib["name"]}
    sheet_name = dict(zip(sol_ref["ID"], sol_ref["Name"]))
    solvents = [by_label[normalize_label(sheet_name[s])] for s in sol_ids]
    base_comment = dict(zip(base_ref["ID"], base_ref["Comment"]))
    bases = [base_comment[b] for b in base_ids]
    unknown = sorted(set(bases) - set(base_lib["comment"]))
    if unknown:
        raise ValueError(f"available bases {unknown} are not in bases.parquet")
    return [int(c) for c in cat_ids], solvents, bases


def candidate_tables(cfg: dict) -> CandidateTables:
    """Feature rows of every available component, assembled by the scan's own code.

    One synthetic experiment per catalyst, per solvent and per base goes through
    ``assemble_features``, so every block is exactly what a training row carries.
    """
    cat_ids, solvents, bases = _available_components()
    salt = _base_active(pd.Series(cat_ids), paths.LIB_DIR).to_numpy()
    salt_cat = cat_ids[int(np.flatnonzero(salt)[0])]
    rows = (
        [(c, solvents[0], bases[0] if s else "no", int(s)) for c, s in zip(cat_ids, salt)]
        + [(salt_cat, sv, bases[0], 1) for sv in solvents]
        + [(salt_cat, solvents[0], b, 1) for b in bases]
    )
    frame = pd.DataFrame({
        "batch": 0, "catalyst_id": [r[0] for r in rows], "solvent": [r[1] for r in rows],
        "base": [r[2] for r in rows], "fa_ratio": 2.0, "loading": 5.0, "temperature": 60.0,
        "time": FIXED_TIME_H, "yield_pct": 0, "conversion": 0, "selectivity": 0.0,
        "replicate_id": -1, "base_active": [r[3] for r in rows],
    })[COLUMN_ORDER]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidates.parquet"
        frame.to_parquet(path, index=False)
        cell = assemble_features(
            cfg["cat_feat"], cfg["solv_feat"], cfg["base_feat"], path, paths.FEAT_DIR,
            libraries_root=paths.LIB_DIR,
        )
    dims = cell.feature_dims
    n_cat, n_sol = len(cat_ids), len(solvents)
    x_cat = cell.X[:n_cat]
    cat = torch.cat([
        x_cat[:, dims["cat"]], x_cat[:, [cell.form_id_dim]], x_cat[:, [cell.active_flag_dim]],
    ], dim=-1)
    return CandidateTables(
        cat=cat, sol=cell.X[n_cat:n_cat + n_sol][:, dims["solv"]],
        base=cell.X[n_cat + n_sol:][:, dims["base"]],
        cat_is_salt=torch.as_tensor(salt > 0), cat_ids=cat_ids, solvents=solvents, bases=bases,
        d_cat=len(dims["cat"]), d_solv=len(dims["solv"]), d_base=len(dims["base"]),
    )


def optimizer_to_model_layout(X: Tensor, d_cat: int, d_solv: int, d_base: int, d_cont: int) -> Tensor:
    """Map optimizer rows onto the scan's design-matrix layout.

    Args:
        X: ``(..., d_cont + d_cat + 2 + d_solv + d_base)`` rows
            ``[cont | cat | form_id | base_active | sol | base]``.
        d_cat: Catalyst block width.
        d_solv: Solvent block width.
        d_base: Base block width.
        d_cont: Continuous block width.

    Returns:
        ``(..., d_cat + d_solv + d_base + 1 + d_cont + 1)`` rows
        ``[cat | solv | base | base_active | cont | form_id]``; gradients flow.
    """
    c, k = d_cont, d_cat
    cont = X[..., :c]
    cat = X[..., c:c + k]
    form_id = X[..., c + k:c + k + 1]
    base_active = X[..., c + k + 1:c + k + 2]
    sol = X[..., c + k + 2:c + k + 2 + d_solv]
    base = X[..., c + k + 2 + d_solv:c + k + 2 + d_solv + d_base]
    return torch.cat([cat, sol, base, base_active, cont, form_id], dim=-1)


class ModelLayoutAcquisition(AcquisitionFunction):
    """Evaluate a scan-layout acquisition on optimizer-layout candidates.

    Args:
        inner: Acquisition function over ``batch x q x d_model`` scan-layout inputs.
        d_cat: Catalyst block width (without ``form_id`` / ``base_active``).
        d_solv: Solvent block width.
        d_base: Base block width.
        d_cont: Continuous block width.
    """

    def __init__(self, inner: AcquisitionFunction, d_cat: int, d_solv: int, d_base: int, d_cont: int) -> None:
        super().__init__(model=inner.model)
        self.inner = inner
        self.d_cat, self.d_solv, self.d_base, self.d_cont = d_cat, d_solv, d_base, d_cont

    def to_model_layout(self, X: Tensor) -> Tensor:
        """``(..., d_opt)`` optimizer rows to ``(..., d_model)`` scan rows."""
        return optimizer_to_model_layout(X, self.d_cat, self.d_solv, self.d_base, self.d_cont)

    def forward(self, X: Tensor) -> Tensor:
        """Acquisition value of ``batch x q x d_opt`` candidates, shape ``batch``."""
        return self.inner(self.to_model_layout(X))


def yield_acquisition(model, cell: CellFeatures, transform: str, tables: CandidateTables) -> ModelLayoutAcquisition:
    """qLogNEI on raw yield over the training rows, in the optimizer layout."""
    objective = GenericMCObjective(lambda samples, X=None: inverse(samples[..., 0], transform))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_AF_SAMPLES]), seed=AF_SEED)
    inner = qLogNoisyExpectedImprovement(
        model=model, X_baseline=cell.X, sampler=sampler, objective=objective,
    )
    return ModelLayoutAcquisition(inner, tables.d_cat, tables.d_solv, tables.d_base, len(cell.feature_dims["cont"]))


def _continuous_bounds() -> dict[str, tuple[float, float]]:
    meta = json.loads((paths.FEAT_DIR / "continuous" / "metadata.json").read_text())
    return {k: (float(v[0]), float(v[1])) for k, v in meta["bounds"].items()}


def _table_row(vec: Tensor, table: Tensor, what: str) -> int:
    """Index of the row of ``table`` equal to ``vec``.

    Raises:
        ValueError: If no row matches to 1e-8.
    """
    dist = (table - vec).abs().amax(dim=-1)
    idx = int(dist.argmin())
    if float(dist[idx]) > 1e-8:
        raise ValueError(f"{what} slice is not a table row (max deviation {float(dist[idx]):.2e})")
    return idx


def decode_batch(batch: Tensor, tables: CandidateTables) -> list[dict]:
    """Conditions of an optimizer-layout batch, matched back to component tables.

    Args:
        batch: ``(q, d_opt)`` optimizer-layout rows.
        tables: The tables the batch was drawn from.

    Returns:
        One record per row: catalyst id, solvent, base and the continuous
        conditions in their raw units.

    Raises:
        ValueError: If a component slice is not a row of its table.
    """
    bounds = _continuous_bounds()
    cont_names = json.loads((paths.FEAT_DIR / "continuous" / "metadata.json").read_text())["raw_columns"]
    k, s = tables.d_cat + 2, tables.d_solv
    out = []
    for row in batch:
        cont, cat, sol, base = row[:4], row[4:4 + k], row[4 + k:4 + k + s], row[4 + k + s:]
        ci = _table_row(cat, tables.cat, "catalyst")
        si = _table_row(sol, tables.sol, "solvent")
        salt = bool(tables.cat_is_salt[ci])
        record = {
            "catalyst_id": tables.cat_ids[ci], "solvent": tables.solvents[si],
            "base": tables.bases[_table_row(base, tables.base, "base")] if salt else "no",
        }
        for j, name in enumerate(cont_names):
            lo, hi = bounds[name]
            record[name] = float(lo + (float(cont[j]) + 1.0) * (hi - lo) / 2.0)
        out.append(record)
    return out


def propose_batch(acquisition: ModelLayoutAcquisition, tables: CandidateTables, seed: int) -> Tensor:
    """The q = 6 batch of the production GA under the production constraints.

    ``optimize_acqf_mixed_GA`` at ``pablo.main.GA_SETTINGS``; temperature is
    grouped by ``pablo.main.TEMPERATURE_PARTITION`` and the reaction time is
    pinned to ``FIXED_TIME_H``.

    Returns:
        ``(q, d_opt)`` batch in the optimizer layout.
    """
    bounds_raw = _continuous_bounds()
    cont_names = json.loads((paths.FEAT_DIR / "continuous" / "metadata.json").read_text())["raw_columns"]
    lo, hi = bounds_raw["time"]
    time_std = 2.0 * (FIXED_TIME_H - lo) / (hi - lo) - 1.0
    d_opt = 4 + tables.cat.shape[1] + tables.d_solv + tables.d_base
    bounds = torch.zeros(2, d_opt, dtype=torch.float64)
    bounds[1] = 1.0
    bounds[0, :4] = -1.0
    n_evals, pop_size = GA_SETTINGS
    batch, _ = optimize_acqf_mixed_GA(
        acquisition, bounds=bounds, q=Q_BATCH, n_evals_ga=n_evals, pop_size=pop_size,
        cat_emb_avail=tables.cat, sol_emb_avail=tables.sol, bas_emb_avail=tables.base,
        cat_is_salt_flags=tables.cat_is_salt,
        batch_constraints={cont_names.index("temperature"): TEMPERATURE_PARTITION},
        fixed_dims={cont_names.index("time"): time_std}, seed=seed,
    )
    return batch.reshape(Q_BATCH, -1)


def run_decision(task: dict) -> dict:
    """One variant's batch and its value under the production model's qLogNEI."""
    name = task["config"]
    cfg = configuration(name)
    transform = cfg["outcome_transform"]
    cell = build_cell(cfg)
    all_rows = list(range(cell.X.shape[0]))
    seed = cfg["outer_cell_id"]
    tables = candidate_tables(cfg)

    own = fit_variant(cell, cfg, all_rows, production_variant(name), seed)
    variant = task["variant"]
    if variant == "own":
        fit = own
    elif variant in NOISE_SCALE_VARIANTS:
        fit = fit_variant(cell, cfg, all_rows, production_variant(name), seed,
                          noise_multiple=NOISE_SCALE_VARIANTS[variant])
    else:
        fit = fit_variant(cell, cfg, all_rows, float(variant), seed)

    with gpytorch.settings.cholesky_jitter(double_value=config.CHOLESKY_JITTER_F64):
        acquisition = yield_acquisition(fit.model, cell, transform, tables)
        t0 = time.time()
        batch = propose_batch(acquisition, tables, task["seed"])
        optimize_s = time.time() - t0
        with torch.no_grad():
            own_acquisition = yield_acquisition(own.model, cell, transform, tables)
            value_own = float(own_acquisition(batch.unsqueeze(0)))
            value_variant = float(acquisition(batch.unsqueeze(0)))
            posterior = own.model.posterior(own_acquisition.to_model_layout(batch))
            median_yield = inverse(posterior.mean.squeeze(-1), transform)
    conditions = decode_batch(batch, tables)
    for record, y in zip(conditions, median_yield.tolist()):
        record["own_median_yield"] = y
    return {
        "batch": batch.tolist(), "conditions": conditions, "value_under_own": value_own,
        "value_under_variant": value_variant, "optimize_s": optimize_s,
        "sigma2_std": fit.sigma2_std, "own_sigma2_std": own.sigma2_std,
    }


def result_path(task_index: int) -> Path:
    """Where task ``task_index`` writes its result."""
    return paths.NOISE_STRUCTURE_DIR / f"task_{task_index}.json"


def run(task_index: int) -> Path:
    """Run one task and write its result JSON, skipping a task already written.

    Returns:
        The result path.
    """
    torch.set_num_threads(1)
    out_path = result_path(task_index)
    if out_path.exists():
        print(f"[noise] task {task_index} already written: {out_path}", flush=True)
        return out_path
    task = tasks()[task_index]
    print(f"[noise] task {task_index}: {task}", flush=True)
    t0 = time.time()
    runner = {"cv": run_cv, "profile": run_profile, "decision": run_decision}[task["kind"]]
    result = runner(task)
    payload = {
        "task": task, "result": result, "wall_s": time.time() - t0,
        "run_meta": {
            "git_commit": scheduler._git_commit(), "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    paths.NOISE_STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)
    print(f"[noise] task {task_index} done in {payload['wall_s']:.0f} s -> {out_path}", flush=True)
    return out_path


def submission_script(task_list: str | None = None) -> str:
    """The sbatch script of one SLURM array over the tasks.

    Args:
        task_list: SLURM ``--array`` index list; every task when omitted.

    Returns:
        The script text.
    """
    logs = paths.NOISE_STRUCTURE_DIR / "logs"
    array = task_list or f"0-{len(tasks()) - 1}%{config.SLURM_MAX_CONCURRENT_WORKERS}"
    return f"""#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH --array={array}
#SBATCH --cpus-per-task={config.SLURM_CPUS_PER_TASK}
#SBATCH --mem={config.SLURM_MEMORY}
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
#SBATCH -o {logs}/task_%A_%a.out
#SBATCH -e {logs}/task_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS={config.SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS={config.SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=

python -m {MODULE} run --task "$SLURM_ARRAY_TASK_ID"
"""


def submit(task_list: str | None = None) -> None:
    """Write :func:`submission_script` next to the results and sbatch it.

    Args:
        task_list: SLURM ``--array`` index list; every task when omitted.
    """
    (paths.NOISE_STRUCTURE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    script_path = paths.NOISE_STRUCTURE_DIR / ".submit_noise_structure.sh"
    script_path.write_text(submission_script(task_list))
    print(f"Submitting {script_path} to {PARTITION}")
    subprocess.run(["sbatch", str(script_path)], check=True)


def replicate_evidence() -> None:
    """Print the replicate-only evidence over theta, with sigma^2 integrated out.

    Within-group sums of squares do not depend on the mean function, so each
    variance function is scored by its marginal likelihood
    ``-sum_g nu_g/2 log g_g - nu/2 log(sum_g S_g / g_g)`` under a Jeffreys prior on
    sigma^2; the shared scale parameter makes the Bayes factors well defined.
    """
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    y = df["yield_pct"].to_numpy() / 100.0
    ids = df["replicate_id"].to_numpy()
    groups = [y[ids == g] for g in sorted(set(ids.tolist()) - {-1})]
    nu = np.array([len(v) - 1 for v in groups], dtype=float)
    ss = np.array([((v - v.mean()) ** 2).sum() for v in groups])
    p = squash(torch.tensor([v.mean() for v in groups], dtype=torch.float64)).numpy()
    g = p * (1.0 - p)

    def log_evidence(shape: np.ndarray) -> float:
        return float(-(nu / 2 * np.log(shape)).sum() - nu.sum() / 2 * np.log((ss / shape).sum()))

    print(f"{len(groups)} replicate groups, nu = {int(nu.sum())}, group means "
          f"{np.round(100 * np.array([v.mean() for v in groups]), 1).tolist()} %, "
          f"group sd {np.round(100 * np.sqrt(ss / np.maximum(nu, 1)), 2).tolist()} pts")
    base = log_evidence(np.ones_like(g))
    named = {
        "theta = 0 (raw)": np.ones_like(g), "theta = 1 (arcsine)": g, "theta = 2 (logit)": g ** 2,
        "probit": stats.norm.pdf(stats.norm.ppf(p)) ** 2,
    }
    for label, shape in named.items():
        delta = log_evidence(shape) - base
        print(f"  homoskedastic on {label:<22} log BF vs raw {delta:+.2f}  (BF {np.exp(delta):.2f})")
    grid = np.linspace(-1.0, 3.0, 801)
    log_ev = np.array([log_evidence(g ** t) for t in grid])
    weights = np.exp(log_ev - log_ev.max())
    cdf = np.cumsum(weights) / weights.sum()
    lo, med, hi = (grid[np.searchsorted(cdf, q)] for q in (0.05, 0.5, 0.95))
    print(f"  theta, flat prior on [-1, 3]: median {med:.2f}, 90% interval [{lo:.2f}, {hi:.2f}], "
          f"maximum at {grid[log_ev.argmax()]:.2f}")


def _load_results() -> tuple[list[dict], list[int]]:
    """Every written task payload, and the indices with no file."""
    found, missing = [], []
    for index in range(len(tasks())):
        path = result_path(index)
        if path.exists():
            found.append(json.loads(path.read_text()))
        else:
            missing.append(index)
    return found, missing


def _variant_label(theta: float | str) -> str:
    return "own" if theta == "own" else f"theta={float(theta):g}"


def _fold_sums(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-fold CRPS sum, baseline CRPS sum and MSLL, each ``(n_folds,)``."""
    crps = np.array([sum(r["crps_per_point"]) for r in rows])
    baseline = np.array([sum(r["baseline_crps_per_point"]) for r in rows])
    msll = np.array([
        np.mean(np.array(r["nlpd_per_point"]) - np.array(r["baseline_nlpd_per_point"])) for r in rows
    ])
    return crps, baseline, msll


def summarize_cv(results: list[dict]) -> None:
    """Pooled CRPSS and median MSLL per variant and study, paired against production.

    ``selected`` takes, fold by fold, the variant with the highest training
    objective: the larger model estimated on the training rows only. Intervals
    resample folds with replacement (``N_BOOTSTRAP``) and pair every variant with
    the production one on the same folds.
    """
    rng = np.random.default_rng(0)
    cv = [r for r in results if r["task"]["kind"] == "cv"]
    for name in CONFIGURATIONS:
        production = _variant_label(production_variant(name))
        print(f"\n[{name}] production variant: {production}")
        print(f"{'study':<15}{'variant':<12}{'folds':>6}{'CRPSS':>9}{'MSLL med':>10}"
              f"{'dCRPSS':>9}{'90% CI':>18}{'dMSLL med':>11}{'90% CI':>18}")
        for study in CV_STUDIES:
            folds = sorted(
                (r for r in cv if r["task"]["config"] == name and r["task"]["split_type"] == study),
                key=lambda r: (r["task"]["repeat"], r["task"]["fold"]),
            )
            if not folds:
                continue
            by_variant: dict[str, list[dict]] = {}
            for r in folds:
                for v in r["result"]["variants"]:
                    by_variant.setdefault(_variant_label(v["theta"]), []).append(v)
            selected = [max(r["result"]["variants"], key=lambda v: v["objective"]) for r in folds]
            by_variant["selected"] = selected
            base_c, base_b, base_m = _fold_sums(by_variant[production])
            n = len(folds)
            idx = rng.integers(0, n, (N_BOOTSTRAP, n))
            for label, rows in by_variant.items():
                c, b, m = _fold_sums(rows)
                crpss = 1.0 - c.sum() / b.sum()
                d_c = crpss - (1.0 - base_c.sum() / base_b.sum())
                d_m = float(np.median(m - base_m))
                boot_c = (base_c[idx].sum(1) / base_b[idx].sum(1)) - (c[idx].sum(1) / b[idx].sum(1))
                boot_m = np.median((m - base_m)[idx], axis=1)
                ci_c = np.quantile(boot_c, [0.05, 0.95])
                ci_m = np.quantile(boot_m, [0.05, 0.95])
                print(f"{study:<15}{label:<12}{n:>6}{crpss:>+9.3f}{float(np.median(m)):>+10.3f}{d_c:>+9.3f}"
                      f"   [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]{d_m:>+11.3f}   [{ci_m[0]:+.3f}, {ci_m[1]:+.3f}]")
            counts = pd.Series([_variant_label(v["theta"]) for v in selected]).value_counts().to_dict()
            print(f"{'':<15}selected by training objective: {counts}")


def summarize_profile(results: list[dict]) -> None:
    """Penalized log marginal likelihood against theta on all rows."""
    for r in [r for r in results if r["task"]["kind"] == "profile"]:
        name = r["task"]["config"]
        variants = r["result"]["variants"]
        family = [v for v in variants if v["theta"] != "own"]
        best = max(v["objective"] for v in family)
        theta_best = max(family, key=lambda v: v["objective"])["theta"]
        inside = [v["theta"] for v in family if best - v["objective"] <= 1.92]
        print(f"\n[{name}] profile (penalized log ML, all rows): maximum at theta={theta_best:g}, "
              f"95% profile interval theta in [{min(inside):g}, {max(inside):g}]")
        for v in variants:
            print(f"  {_variant_label(v['theta']):<12} {v['objective']:>10.2f}  "
                  f"delta {v['objective'] - best:>+7.2f}  sigma^2_std {v['sigma2_std']:.3g}  "
                  f"jitter {v['jitter']:.1e}")


def summarize_decision(results: list[dict]) -> None:
    """Each variant's batch scored by the production model's qLogNEI."""
    for name in CONFIGURATIONS:
        rows = [r for r in results if r["task"]["kind"] == "decision" and r["task"]["config"] == name]
        reference = next((r for r in rows if r["task"]["variant"] == "own" and r["task"]["seed"] == 0), None)
        if reference is None:
            print(f"\n[{name}] decision: the own seed-0 batch is not written yet; {len(rows)} other batches")
            continue
        ref_value = reference["result"]["value_under_own"]
        ref_triples = {(c["catalyst_id"], c["solvent"], c["base"]) for c in reference["result"]["conditions"]}
        print(f"\n[{name}] qLogNEI of the production model, own seed 0 = {ref_value:.3f}")
        print(f"{'variant':<14}{'seed':>5}{'value':>9}{'regret':>9}{'shared':>8}  conditions (cat/solvent/base/T)")
        for r in rows:
            res = r["result"]
            triples = {(c["catalyst_id"], c["solvent"], c["base"]) for c in res["conditions"]}
            cond = "; ".join(
                f"{c['catalyst_id']}/{c['solvent']}/{c['base']}/{c['temperature']:.0f}" for c in res["conditions"]
            )
            label = r["task"]["variant"] if isinstance(r["task"]["variant"], str) else _variant_label(r["task"]["variant"])
            print(f"{label:<14}{r['task']['seed']:>5}{res['value_under_own']:>9.3f}"
                  f"{ref_value - res['value_under_own']:>9.3f}{len(triples & ref_triples):>6}/6  {cond}")


def summarize() -> None:
    """Print the CV, profile and decision tables from every written task."""
    results, missing = _load_results()
    if missing:
        print(f"{len(missing)} of {len(tasks())} tasks have no result file: {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
    summarize_cv(results)
    summarize_profile(results)
    summarize_decision(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replicates", help="Replicate-only evidence over theta.")
    sub.add_parser("list", help="Print the tasks.")
    run_parser = sub.add_parser("run", help="Run one task.")
    run_parser.add_argument("--task", type=int, required=True)
    submit_parser = sub.add_parser("submit", help="sbatch the tasks as one array.")
    submit_parser.add_argument("--tasks", default=None, help="SLURM array index list; all tasks if omitted.")
    sub.add_parser("summarize", help="CV, profile and decision tables.")
    args = parser.parse_args()

    if args.cmd == "replicates":
        replicate_evidence()
    elif args.cmd == "list":
        for index, task in enumerate(tasks()):
            print(f"{index:>4}  {task}")
    elif args.cmd == "run":
        run(args.task)
    elif args.cmd == "submit":
        submit(args.tasks)
    elif args.cmd == "summarize":
        summarize()


if __name__ == "__main__":
    main()
