"""``_fit_multistart``: noise initialisation, restart starts and floor-plateau escape."""

from __future__ import annotations

import copy

import pytest
import torch
from botorch.fit import fit_gpytorch_mll as real_fit
from gpytorch.kernels import MaternKernel, ScaleKernel

from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import make_singletask_gp
from pablo.featurization_and_kernel_scan.cross_validation import fit_eval
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import (
    _floor_plateau_masks,
    _init_noise_at_prior_mode,
    _fit_multistart,
)

ALPHA, BETA = 4.5, 2e-3


def _toy_model(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(25, 3, generator=g, dtype=torch.float64)
    y = (3 * x[:, :1]).sin() + 0.05 * torch.randn(25, 1, generator=g, dtype=torch.float64)
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
    model = make_singletask_gp(
        x, y, kernel, mean_module=None, form_id_dim=-1, noise_prior_alpha=ALPHA, noise_prior_beta=BETA
    )
    return model.to(torch.float64), x


def _noise_prior_mode(model) -> float:
    for name, _, prior, _, _ in model.likelihood.named_priors():
        if "noise" in name:
            return float(prior.rate / (prior.concentration + 1.0))
    raise AssertionError("no noise prior registered")


def test_noise_starts_at_inverse_gamma_mode():
    """The initial standardized noise is the InverseGamma mode β/(α+1)."""
    model, _ = _toy_model()
    _init_noise_at_prior_mode(model)
    assert model.likelihood.noise.item() == pytest.approx(_noise_prior_mode(model), rel=1e-6)


def test_plateau_mask_flags_raw_parameters_far_below_zero():
    """Only raw values below FLOOR_PLATEAU_RAW under a GreaterThan constraint are flagged."""
    model, _ = _toy_model()
    with torch.no_grad():
        model.likelihood.noise_covar.raw_noise.fill_(-50.0)
    masks = _floor_plateau_masks(model)
    assert set(masks) == {"likelihood.noise_covar.raw_noise"}
    assert bool(masks["likelihood.noise_covar.raw_noise"].all())


def test_restarts_start_from_the_initial_state(monkeypatch):
    """Every restart's first parameters lie within the restart jitter of the initial state, not of the previous optimum."""
    model, x = _toy_model()
    starts = []

    def recording_fit(mll, **kwargs):
        starts.append({k: v.detach().clone() for k, v in mll.model.named_parameters()})
        return real_fit(mll, **kwargs)

    monkeypatch.setattr(fit_eval, "fit_gpytorch_mll", recording_fit)
    _fit_multistart(model, x, n_restarts=4, max_iter=200)
    for start in starts[1:4]:
        for key, value in start.items():
            assert (value - starts[0][key]).abs().max() < 8 * config.RESTART_JITTER_SD


def test_noise_stuck_on_the_floor_is_refitted_from_the_interior(monkeypatch):
    """A restart that ends with raw noise far below zero is escaped and the better fit is kept."""
    model, x = _toy_model()
    calls = {"n": 0}

    def trapping_fit(mll, **kwargs):
        out = real_fit(mll, **kwargs)
        calls["n"] += 1
        if calls["n"] <= 2:
            with torch.no_grad():
                mll.model.likelihood.noise_covar.raw_noise.fill_(-80.0)
        return out

    monkeypatch.setattr(fit_eval, "fit_gpytorch_mll", trapping_fit)
    _fit_multistart(model, x, n_restarts=2, max_iter=200)
    assert calls["n"] == 3
    assert model.likelihood.noise_covar.raw_noise.item() > config.FLOOR_PLATEAU_RAW
    assert model.likelihood.noise.item() > 1.1e-4


def test_returned_state_has_the_best_penalized_mll():
    """The model ends in eval mode with a finite MLL no worse than its initial state."""
    model, x = _toy_model()
    initial = copy.deepcopy(model)
    _fit_multistart(model, x, n_restarts=3, max_iter=200)
    assert not model.training
    mll_fit = fit_eval._penalized_loss(model, fit_eval.ExactMarginalLogLikelihood(model.likelihood, model), x)
    _init_noise_at_prior_mode(initial)
    mll_init = fit_eval._penalized_loss(initial, fit_eval.ExactMarginalLogLikelihood(initial.likelihood, initial), x)
    assert mll_fit < mll_init
