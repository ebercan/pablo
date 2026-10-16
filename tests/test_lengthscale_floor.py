"""Lengthscale-floor infrastructure: the constraint, and the floor-hit reporting.

Two separable things are checked. First, that ``lengthscale_floor`` actually
reaches every base-kernel lengthscale as a ``GreaterThan`` constraint and binds.
Second, that the post-hoc reporting path in
``results_analysis/scan_lengthscale_floor.py`` — which reconstructs the applied
lengthscale as ``softplus(raw) + floor`` from the recorded ``hyperparams`` dict —
sees every lengthscale the live kernel structures actually carry. A parser that
matches nothing reports "no floor hits", which is indistinguishable from a clean
fit unless something asserts the match.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch
from gpytorch.constraints import GreaterThan, Positive

from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import make_kernel_from_config
from pablo.featurization_and_kernel_scan.results_analysis.scan_lengthscale_floor import (
    _floor_from_run_meta,
    _parse_axes,
    _softplus,
)

KS_NAMES = ("KS1", "KS2", "KS3", "KS4")

FEATURE_DIMS = {"cat": [0, 1, 2, 3, 4, 5], "solv": [6, 7], "base": [8, 9, 10], "cont": [12, 13, 14, 15]}
ACTIVE_FLAG_DIM = 11
N_LENGTHSCALE_SCALARS = 1 + 1 + 1 + 4
"""One shared lengthscale per discrete block plus full ARD over ``cont``."""


@contextmanager
def floor_set_to(value: float | None):
    """Run the block with ``config.LENGTHSCALE_FLOOR`` temporarily swapped.

    The floor is one global constant read at kernel-construction time, so this
    is how a test asks for the no-floor behaviour without a second code path.
    """
    original = config.LENGTHSCALE_FLOOR
    config.LENGTHSCALE_FLOOR = value
    try:
        yield
    finally:
        config.LENGTHSCALE_FLOOR = original


def _build(ks: str, floor: float | None):
    with floor_set_to(floor):
        return make_kernel_from_config(
            kernel_structure=ks,
            feature_dims=FEATURE_DIMS,
            active_flag_dim=ACTIVE_FLAG_DIM,
            base_kernel_type="matern52",
        ).to(torch.float64)


def _lengthscale_modules(kernel):
    """Every submodule carrying a ``raw_lengthscale`` parameter."""
    return [m for m in kernel.modules() if hasattr(m, "raw_lengthscale")]


# ── the constraint ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("ks", KS_NAMES)
def test_floor_reaches_every_block_lengthscale(ks):
    """A configured floor becomes a ``GreaterThan`` on every block lengthscale."""
    kernel = _build(ks, 0.05)
    modules = _lengthscale_modules(kernel)
    assert len(modules) == 4, f"{ks} should carry one lengthscale module per block"
    for m in modules:
        constraint = m.raw_lengthscale_constraint
        assert isinstance(constraint, GreaterThan)
        assert float(constraint.lower_bound) == pytest.approx(0.05)


@pytest.mark.parametrize("ks", KS_NAMES)
def test_no_floor_leaves_the_default_positive_constraint(ks):
    """``lengthscale_floor=None`` must not silently install a floor."""
    for m in _lengthscale_modules(_build(ks, None)):
        assert isinstance(m.raw_lengthscale_constraint, Positive)


@pytest.mark.parametrize("ks", KS_NAMES)
def test_floor_binds_at_extreme_raw_values(ks):
    """Driving raw_lengthscale to −inf-ish yields exactly the floor, not zero."""
    kernel = _build(ks, 0.05)
    for m in _lengthscale_modules(kernel):
        with torch.no_grad():
            m.raw_lengthscale.fill_(-60.0)
        assert torch.all(m.lengthscale >= 0.05 - 1e-12)
        torch.testing.assert_close(
            m.lengthscale, torch.full_like(m.lengthscale, 0.05), rtol=0, atol=1e-9
        )


def test_reporting_formula_matches_the_live_constraint():
    """``softplus(raw) + floor`` — the scan's reconstruction — is the real transform.

    The scan never sees the fitted model, only the recorded raw parameters, so
    its arithmetic has to track whatever ``GreaterThan`` does. The tolerance is
    loose because ``GreaterThan`` stores its bound as a float32 tensor, so the
    applied floor differs from the configured float by ~1e-9.
    """
    floor = 0.05
    constraint = GreaterThan(floor)
    raw = torch.linspace(-40.0, 10.0, 50, dtype=torch.float64)
    torch.testing.assert_close(
        torch.as_tensor(_softplus(raw.numpy()) + floor),
        constraint.transform(raw),
        rtol=1e-6,
        atol=1e-8,
    )


# ── the reporting path ───────────────────────────────────────────────────────


@pytest.mark.parametrize("ks", KS_NAMES)
def test_scan_parser_sees_every_lengthscale_of_every_structure(ks):
    """The floor-hit scan must count all 7 lengthscale scalars of each structure.

    ``_parse_axes`` matches parameter names with a regex. If a structure names
    its submodules differently, the parser matches nothing and the fold is
    reported as having no floored dimensions rather than as unparsed.
    """
    kernel = _build(ks, 0.05)
    hyperparams = {
        name: p.detach().squeeze().tolist()
        for name, p in kernel.named_parameters()
        if "raw_lengthscale" in name
    }
    _, _, n_total, _ = _parse_axes(hyperparams, 0.05, model_class="st_y")
    assert n_total == N_LENGTHSCALE_SCALARS


@pytest.mark.parametrize("ks", KS_NAMES)
def test_scan_parser_flags_a_collapsed_fold(ks):
    """A fold with every lengthscale pinned at the floor is reported as floored."""
    kernel = _build(ks, 0.05)
    with torch.no_grad():
        for m in _lengthscale_modules(kernel):
            m.raw_lengthscale.fill_(-60.0)
    hyperparams = {
        name: p.detach().squeeze().tolist()
        for name, p in kernel.named_parameters()
        if "raw_lengthscale" in name
    }
    min_ls, n_floored, n_total, _ = _parse_axes(
        hyperparams, 0.05, model_class="st_y"
    )
    assert n_total == N_LENGTHSCALE_SCALARS
    assert n_floored == N_LENGTHSCALE_SCALARS
    assert min_ls == pytest.approx(0.05)


def test_the_floor_is_one_global_constant_not_a_grid_axis():
    """Nothing on the grid varies the floor; it applies to every fit."""
    assert config.LENGTHSCALE_FLOOR is None or config.LENGTHSCALE_FLOOR > 0.0
    for scheme in config.PRIOR_SCHEMES.values():
        assert not hasattr(scheme, "lengthscale_floor"), (
            "The floor is a constraint, not part of a prior scheme."
        )


def test_scan_reads_the_floor_from_the_run_record():
    """The scan finds the floor in the live record and in archived results.

    Archived runs wrote it nested under the retired per-set record; every set
    that carried a floor carried the same value, so those stay readable.
    """
    assert _floor_from_run_meta({"lengthscale_floor": 0.05}) == 0.05
    assert _floor_from_run_meta({"lengthscale_floor": None}) is None
    assert _floor_from_run_meta({}) is None
    archived = {"hyperprior_sets": {
        "production": {"lengthscale_floor": 0.05},
        "smooth_ood": {"lengthscale_floor": None},
    }}
    assert _floor_from_run_meta(archived) == 0.05
    with pytest.raises(ValueError, match="several different lengthscale floors"):
        _floor_from_run_meta({"hyperprior_sets": {
            "a": {"lengthscale_floor": 0.05}, "b": {"lengthscale_floor": 0.1},
        }})
