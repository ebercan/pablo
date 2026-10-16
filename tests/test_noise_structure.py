"""The variance-function noise family and the batch-check plumbing of noise_structure."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import get_y_yield
from pablo.featurization_and_kernel_scan.cross_validation import noise_structure as ns
from pablo.utils.outcome_transforms import forward


@pytest.fixture(scope="module")
def probit_cfg():
    return ns.configuration("probit")


@pytest.fixture(scope="module")
def cell(probit_cfg):
    return ns.build_cell(probit_cfg)


@pytest.mark.parametrize("transform, theta", [("raw", 0.0), ("arcsin_sqrt", 1.0), ("logit", 2.0)])
def test_shape_is_flat_where_the_family_meets_the_scale(transform, theta):
    """theta = 0 / 1 / 2 is exactly homoskedastic on the raw / arcsine / logit scale."""
    y = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    shape = ns.unnormalized_shape(y, transform, theta)
    assert float(shape.max() / shape.min()) == pytest.approx(1.0, abs=1e-9)
    assert ns.OWN_THETA[transform] == theta


def test_probit_shape_shrinks_toward_zero_yield_under_raw_homoskedastic_noise():
    """Raw-homoskedastic noise mapped to the probit scale is widest at the boundary."""
    y = torch.tensor([0.0, 0.05, 0.5], dtype=torch.float64)
    shape = ns.unnormalized_shape(y, "probit", 0.0)
    assert float(shape[0]) > float(shape[1]) > float(shape[2])


@pytest.mark.parametrize("transform", ["raw", "logit", "arcsin_sqrt", "probit"])
def test_unit_shape_prior_is_the_production_prior(transform):
    """With every shape 1 the shaped replicate prior is config.NOISE_EVIDENCE."""
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    y = torch.tensor(df["yield_pct"].to_numpy() / 100.0, dtype=torch.float64)
    z = forward(y, transform)
    alpha, beta = ns.shaped_replicate_prior(z, df["replicate_id"].to_numpy(), torch.ones_like(z))
    evidence = config.NOISE_EVIDENCE[("Y", transform)]
    assert alpha == pytest.approx(evidence.alpha)
    assert beta == pytest.approx(evidence.beta)


def test_own_variant_carries_the_production_noise_prior(cell, probit_cfg):
    """The unit-shape variant's sigma^2 prior equals the production st_y prior."""
    train_idx = list(range(50))
    model, norm, var_z, rate_std = ns.build_variant_model(cell, probit_cfg, train_idx, "own", seed=0)
    evidence = config.NOISE_EVIDENCE[("Y", "probit")]
    train_y = get_y_yield(cell, "probit")[train_idx].squeeze(-1)
    assert float(norm) == pytest.approx(1.0)
    assert rate_std == pytest.approx(evidence.beta / float(train_y.var()))
    prior = model.likelihood.noise_covar.noise_prior
    assert float(prior.rate) == pytest.approx(rate_std)
    assert float(prior.concentration) == pytest.approx(evidence.alpha)


def test_shaped_noise_serves_training_rows_only():
    """Asking ShapedNoise for a different number of rows is an error, not a broadcast."""
    noise = ns.ShapedNoise(torch.ones(4, dtype=torch.float64), noise_prior=None)
    assert noise(torch.zeros(4, 2, dtype=torch.float64)).shape == (4, 4)
    with pytest.raises(RuntimeError, match="holds 4 shapes"):
        noise(torch.zeros(3, 2, dtype=torch.float64))


def test_frozen_noise_stays_at_its_value():
    """A frozen sigma^2 ignores its raw parameter."""
    noise = ns.ShapedNoise(torch.ones(2, dtype=torch.float64), noise_prior=None, fixed_noise=0.25)
    with torch.no_grad():
        noise.raw_noise.add_(3.0)
    assert float(noise.noise) == pytest.approx(0.25)
    assert not noise.raw_noise.requires_grad


def test_task_list_covers_every_fold_profile_and_decision():
    """Every primary and leave_one_out fold per configuration, then profiles, then decisions."""
    task_list = ns.tasks()
    n_folds = sum(
        len(b["folds"]) for b in json.loads(paths.CV_SPLITS_JSON.read_text())
        if b["split_type"] in ns.CV_STUDIES
    )
    kinds = [t["kind"] for t in task_list]
    assert kinds.count("cv") == n_folds * len(ns.CONFIGURATIONS)
    assert kinds.count("profile") == len(ns.CONFIGURATIONS)
    assert kinds.count("decision") == sum(len(ns.decision_variants(n)) for n in ns.CONFIGURATIONS)
    assert kinds == sorted(kinds, key=["cv", "profile", "decision"].index)


def test_candidate_rows_reproduce_the_training_design(cell, probit_cfg):
    """An experiment rebuilt from the candidate tables is its training row, bit for bit.

    This is what makes the batch check score the same features the GP was fitted
    on: the optimizer assembles a candidate from table rows and the layout map.
    """
    tables = ns.candidate_tables(probit_cfg)
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    dims = cell.feature_dims
    checked = 0
    for i, row in df.iterrows():
        if row["catalyst_id"] not in tables.cat_ids or row["solvent"] not in tables.solvents:
            continue
        if row["base_active"] and row["base"] not in tables.bases:
            continue
        cat = tables.cat[tables.cat_ids.index(row["catalyst_id"])]
        sol = tables.sol[tables.solvents.index(row["solvent"])]
        base = (
            tables.base[tables.bases.index(row["base"])] if row["base_active"]
            else torch.zeros(tables.d_base, dtype=torch.float64)
        )
        cont = cell.X[i, dims["cont"]]
        x_opt = torch.cat([cont, cat, sol, base])
        rebuilt = ns.optimizer_to_model_layout(
            x_opt, tables.d_cat, tables.d_solv, tables.d_base, len(dims["cont"])
        )
        torch.testing.assert_close(rebuilt, cell.X[i], rtol=0.0, atol=1e-12)
        decoded = ns.decode_batch(x_opt.unsqueeze(0), tables)[0]
        assert decoded["catalyst_id"] == row["catalyst_id"]
        assert decoded["solvent"] == row["solvent"]
        assert decoded["base"] == (row["base"] if row["base_active"] else "no")
        assert decoded["temperature"] == pytest.approx(row["temperature"])
        checked += 1
    assert checked >= 60
