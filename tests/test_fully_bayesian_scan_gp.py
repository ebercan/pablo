"""Tests for FullyBayesianScanGP: log joint, posterior equivalence, mixtures, NUTS fitting, and diagnostics."""

from __future__ import annotations

import dataclasses

import gpytorch
import pytest
import pyro
import pyro.poutine
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import BLOCKS
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    ChainResult,
    FullyBayesianScanGP,
    MCMCDiagnostics,
    MCMCDiagnosticsError,
    NutsSettings,
    check_mcmc_diagnostics,
)
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import _fit_multistart
from pablo.featurization_and_kernel_scan.results_analysis.rebuild_predictive import build_cell
from pablo.utils.feature_assembly import CellFeatures

CFG: dict[str, str] = {
    "cat_feat": "grover_pca",
    "solv_feat": "Hansen",
    "base_feat": "grover_pca",
    "kernel_structure": "KS1",
    "base_kernel": "matern52",
    "model_class": "st_y",
    "mean_function": "constant",
    "outcome_transform": "probit",
    "prior_scheme": "P2",
}


@pytest.fixture(scope="module")
def cell() -> CellFeatures:
    """Module-scoped fixture assembling features for the rank-1 cross_1 configuration."""
    return build_cell(CFG, CFG)


@pytest.fixture(scope="module")
def loo_split(cell: CellFeatures) -> tuple[int, list[int]]:
    """Leave-one-out fold holding out the experiment with highest raw yield."""
    held_out_idx = int(torch.argmax(cell.y_raw.squeeze(-1)).item())
    train_idx = [i for i in range(cell.X.shape[0]) if i != held_out_idx]
    return held_out_idx, train_idx


def _sample_hyperparameters(fb: FullyBayesianScanGP, gen: torch.Generator) -> dict[str, Tensor]:
    data: dict[str, Tensor] = {}
    for b in fb.prior_spec.blocks:
        g = len(b.groups)
        data[f"lengthscale_{b.name}"] = 0.2 + (5.0 - 0.2) * torch.rand(g, generator=gen, dtype=torch.float64)
    data["outputscale"] = 0.3 + (2.0 - 0.3) * torch.rand((), generator=gen, dtype=torch.float64)
    data["noise"] = 0.01 + (0.3 - 0.01) * torch.rand((), generator=gen, dtype=torch.float64)
    data["mean"] = -0.5 + (0.5 - (-0.5)) * torch.rand((), generator=gen, dtype=torch.float64)
    return data


def _set_template_hyperparameters(template: SingleTaskGP, hyperparams: dict[str, Tensor]) -> None:
    for b_kernel, block_name in zip(template.covar_module.base_kernel.kernels, BLOCKS):
        b_kernel._set_lengthscale(hyperparams[f"lengthscale_{block_name}"].reshape(1, -1))
    template.covar_module.outputscale = hyperparams["outputscale"]
    template.likelihood.noise_covar.noise = hyperparams["noise"]
    template.mean_module.constant = hyperparams["mean"]


def test_log_joint_matches_map(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Pyro log joint evaluates identically to GPyTorch ExactMarginalLogLikelihood under fixed hyperparameters."""
    _, train_idx = loo_split
    template = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure="KS1",
        base_kernel_type="matern52",
        outcome_transform="probit",
        lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme="P2",
        mean_function="constant",
    ).st_y(None)
    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)

    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        hyperparams = _sample_hyperparameters(fb, gen)
        _set_template_hyperparameters(template, hyperparams)

        for b_kernel in template.covar_module.base_kernel.kernels:
            b_kernel.lengthscale_prior.loc = b_kernel.lengthscale_prior.loc.to(torch.float64)
            b_kernel.lengthscale_prior.scale = b_kernel.lengthscale_prior.scale.to(torch.float64)

        template.train()
        template.likelihood.train()
        mll = ExactMarginalLogLikelihood(template.likelihood, template)
        train_inputs = template.train_inputs[0]
        train_dist = template(train_inputs)
        n = template.train_targets.shape[0]
        map_log_joint = mll(train_dist, template.train_targets) * n

        trace = pyro.poutine.trace(
            pyro.poutine.condition(fb.ks1_pyro_model.sample, data=hyperparams)
        ).get_trace()
        pyro_log_joint = trace.log_prob_sum()

        torch.testing.assert_close(pyro_log_joint, map_log_joint, rtol=1e-10, atol=1e-8)


def test_posterior_matches_map_at_fixed_hyperparameters(
    cell: CellFeatures, loo_split: tuple[int, list[int]]
) -> None:
    """Posterior mean and variance from FullyBayesianScanGP match SingleTaskGP under identical hyperparameters."""
    held_out_idx, train_idx = loo_split
    template = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure="KS1",
        base_kernel_type="matern52",
        outcome_transform="probit",
        lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme="P2",
        mean_function="constant",
    ).st_y(None)
    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)

    gen = torch.Generator().manual_seed(0)
    hyperparams = _sample_hyperparameters(fb, gen)
    _set_template_hyperparameters(template, hyperparams)

    mcmc_samples = {site: val.unsqueeze(0) for site, val in hyperparams.items()}
    fb.load_mcmc_samples(mcmc_samples)
    fb.eval()
    template.eval()

    x_eval = torch.cat([cell.X[held_out_idx : held_out_idx + 1], cell.X[train_idx[:3]]], dim=0)
    with torch.no_grad():
        post_fb = fb.posterior(x_eval)
        post_map = template.posterior(x_eval)

    torch.testing.assert_close(post_fb.mean[0], post_map.mean, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(post_fb.variance[0], post_map.variance, rtol=1e-8, atol=1e-8)


def test_mixture_shapes(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Posterior over multiple MCMC draws produces expected mixture dimensions and valid predictive components."""
    _, train_idx = loo_split
    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)

    gen = torch.Generator().manual_seed(10)
    samples_5: dict[str, Tensor] = {}
    for b in fb.prior_spec.blocks:
        samples_5[f"lengthscale_{b.name}"] = 0.2 + 4.8 * torch.rand(
            5, len(b.groups), generator=gen, dtype=torch.float64
        )
    samples_5["outputscale"] = 0.3 + 1.7 * torch.rand(5, generator=gen, dtype=torch.float64)
    samples_5["noise"] = 0.01 + 0.29 * torch.rand(5, generator=gen, dtype=torch.float64)
    samples_5["mean"] = -0.5 + 1.0 * torch.rand(5, generator=gen, dtype=torch.float64)

    fb.load_mcmc_samples(samples_5)
    fb.eval()

    x_eval = cell.X[:4]
    with torch.no_grad():
        post = fb.posterior(x_eval)

    assert isinstance(post, GaussianMixturePosterior)
    assert post.mean.shape == torch.Size([5, 4, 1])

    m, s = fb.predictive_components(x_eval)
    assert m.shape == torch.Size([5, 4])
    assert s.shape == torch.Size([5, 4])
    assert torch.all(s > 0)

    with pytest.raises(ValueError):
        fb.predictive_components(x_eval.unsqueeze(0))


def test_nuts_smoke_synthetic(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Short synthetic NUTS run produces expected draw counts, diagnostics, and repeatable samples across chains."""
    _, train_idx = loo_split
    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)
    settings = NutsSettings(num_chains=2, warmup_steps=10, num_samples=6, thinning=2)
    diag = fb.fit_nuts(settings, seed=0, strict=False)

    assert isinstance(diag, MCMCDiagnostics)
    assert diag.num_draws_kept == 6
    assert len(diag.divergences_per_chain) == 2
    assert diag.potential_evaluations > 0

    with torch.no_grad():
        post = fb.posterior(cell.X[:2])
    assert post.mean.shape == torch.Size([6, 2, 1])

    fb1 = FullyBayesianScanGP.from_config(CFG, cell, train_idx)
    fb2 = FullyBayesianScanGP.from_config(CFG, cell, train_idx)
    chain1 = fb1.run_chain(settings, seed=42)
    chain2 = fb2.run_chain(settings, seed=42)
    for site in chain1.samples:
        torch.testing.assert_close(chain1.samples[site], chain2.samples[site])


def test_nonpd_counts_and_minus_inf(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Non-positive definite covariance yields -inf log-joint, increments failure count, and preserves finite prior gradients."""
    _, train_idx = loo_split
    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)

    outputscale = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
    noise = torch.tensor(1e-12, dtype=torch.float64)
    mean = torch.tensor(0.0, dtype=torch.float64)
    condition_data: dict[str, Tensor] = {
        "outputscale": outputscale,
        "noise": noise,
        "mean": mean,
    }
    for b in fb.prior_spec.blocks:
        condition_data[f"lengthscale_{b.name}"] = torch.full(
            (len(b.groups),), 1e6, dtype=torch.float64
        )

    init_nonpd = fb.ks1_pyro_model.nonpd_evaluations
    trace = pyro.poutine.trace(
        pyro.poutine.condition(fb.ks1_pyro_model.sample, data=condition_data)
    ).get_trace()
    log_joint = trace.log_prob_sum()

    assert torch.isneginf(log_joint)
    assert fb.ks1_pyro_model.nonpd_evaluations == init_nonpd + 1

    grad = torch.autograd.grad(log_joint, outputscale)[0]
    assert torch.isfinite(grad)


def test_check_diagnostics_thresholds() -> None:
    """Convergence diagnostics checker accepts valid runs and rejects individual threshold and NaN violations."""
    good_diag = MCMCDiagnostics(
        divergences_per_chain=(0, 0),
        nonpd_evaluations=0,
        potential_evaluations=100,
        rhat_max=1.00,
        ess_min=150.0,
        num_draws_kept=10,
        wall_time_s=1.0,
    )
    check_mcmc_diagnostics(good_diag)

    bad_div = dataclasses.replace(good_diag, divergences_per_chain=(1, 0))
    with pytest.raises(MCMCDiagnosticsError, match="post-warmup divergences"):
        check_mcmc_diagnostics(bad_div)

    bad_rhat = dataclasses.replace(good_diag, rhat_max=1.05)
    with pytest.raises(MCMCDiagnosticsError, match="max split-R̂"):
        check_mcmc_diagnostics(bad_rhat)

    bad_nan_rhat = dataclasses.replace(good_diag, rhat_max=float("nan"))
    with pytest.raises(MCMCDiagnosticsError, match="max split-R̂ nan"):
        check_mcmc_diagnostics(bad_nan_rhat)

    bad_ess = dataclasses.replace(good_diag, ess_min=50.0)
    with pytest.raises(MCMCDiagnosticsError, match="min bulk ESS"):
        check_mcmc_diagnostics(bad_ess)

    bad_nan_ess = dataclasses.replace(good_diag, ess_min=float("nan"))
    with pytest.raises(MCMCDiagnosticsError, match="min bulk ESS nan"):
        check_mcmc_diagnostics(bad_nan_ess)


def test_out_of_scope_configs_raise(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Unsupported kernel structures, mean functions, model classes, or missing keys raise errors on initialization."""
    _, train_idx = loo_split

    with pytest.raises(ValueError, match="kernel_structure='KS1' only"):
        FullyBayesianScanGP.from_config({**CFG, "kernel_structure": "KS2"}, cell, train_idx)

    with pytest.raises(ValueError, match="mean_function='constant' only"):
        FullyBayesianScanGP.from_config({**CFG, "mean_function": "cond_linear"}, cell, train_idx)

    with pytest.raises(ValueError, match="model_class='st_y' only"):
        FullyBayesianScanGP.from_config({**CFG, "model_class": "icm_cs"}, cell, train_idx)

    missing_cfg = {k: v for k, v in CFG.items() if k != "cat_feat"}
    with pytest.raises(KeyError, match="configuration lacks"):
        FullyBayesianScanGP.from_config(missing_cfg, cell, train_idx)


def test_load_chains_validates(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Loading chains with invalid chain counts or mismatched sample lengths raises validation errors."""
    _, train_idx = loo_split
    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)
    settings = NutsSettings(num_chains=2, warmup_steps=5, num_samples=10, thinning=1)
    dummy_10 = fb.ks1_pyro_model.get_dummy_mcmc_samples(10, dtype=torch.float64)
    dummy_8 = fb.ks1_pyro_model.get_dummy_mcmc_samples(8, dtype=torch.float64)

    c1 = ChainResult(
        seed=0, samples=dummy_10, divergences=0, nonpd_evaluations=0, potential_evaluations=10, wall_time_s=0.1
    )
    c_wrong = ChainResult(
        seed=2, samples=dummy_8, divergences=0, nonpd_evaluations=0, potential_evaluations=8, wall_time_s=0.1
    )

    with pytest.raises(ValueError, match="expected 2 chains"):
        fb.load_chains([c1], settings, strict=False)

    with pytest.raises(ValueError, match="expected 10"):
        fb.load_chains([c1, c_wrong], settings, strict=False)


def test_real_fold_agrees_with_map(cell: CellFeatures, loo_split: tuple[int, list[int]]) -> None:
    """Fully Bayesian NUTS predictive distribution and hyperparameter draws agree with MAP fit on a real LOO fold."""
    held_out_idx, train_idx = loo_split
    template = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure="KS1",
        base_kernel_type="matern52",
        outcome_transform="probit",
        lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme="P2",
        mean_function="constant",
    ).st_y(None)

    with gpytorch.settings.cholesky_jitter(double_value=config.CHOLESKY_JITTER_F64):
        _fit_multistart(template, cell.X[train_idx], n_restarts=2, max_iter=500)

    fb = FullyBayesianScanGP.from_config(CFG, cell, train_idx)
    fb_settings = NutsSettings(num_chains=1, warmup_steps=150, num_samples=150, thinning=1)
    fb.fit_nuts(fb_settings, seed=1, strict=False)

    map_noise = template.likelihood.noise.squeeze()
    noise_draws = fb.likelihood.noise.squeeze()
    q01_noise = torch.quantile(noise_draws, 0.01)
    q99_noise = torch.quantile(noise_draws, 0.99)
    assert q01_noise <= map_noise <= q99_noise

    map_outputscale = template.covar_module.outputscale.squeeze()
    outputscale_draws = fb.covar_module.outputscale.squeeze()
    q01_out = torch.quantile(outputscale_draws, 0.01)
    q99_out = torch.quantile(outputscale_draws, 0.99)
    assert q01_out <= map_outputscale <= q99_out

    x_held_out = cell.X[held_out_idx : held_out_idx + 1]
    template.eval()
    with torch.no_grad():
        map_mean = template.posterior(x_held_out).mean.squeeze()

    m, s = fb.predictive_components(x_held_out)
    fb_mixture_mean = m.mean(dim=0).squeeze()
    mixture_sd = torch.sqrt((s ** 2).mean(dim=0).squeeze() + m.var(dim=0).squeeze())
    assert torch.abs(fb_mixture_mean - map_mean) < 2 * mixture_sd
