"""Replicate-group detection and the noise prior derived from it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.build_experiments import build_experiments_table
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    get_y_cs,
    get_y_yield,
)
from pablo.featurization_and_kernel_scan.core.replicates import (
    MIN_REPLICATE_DOF,
    NOISE_TARGET_COLUMNS,
    detect_replicate_groups,
    estimate_noise_evidence,
    noise_evidence_by_scale,
    normalize_label,
)
from pablo.utils.feature_assembly import assemble_features
from pablo.utils.outcome_transforms import OUTCOME_TRANSFORM_NAMES, forward

EXCEL = paths.PKG_ROOT.parent / "nhc_doe.xlsx"


def _row(**overrides):
    base = dict(
        catalyst_id=4, solvent="THF", base="no",
        fa_ratio=2.0, loading=5.25, temperature=80.0, time=16.0,
    )
    base.update(overrides)
    return base


def test_normalize_label_folds_punctuation_and_case():
    """The two spellings of 1,4-dioxane in the sheet normalize to one label."""
    assert normalize_label("1,4-Dioxane") == normalize_label("1.4-dioxane")
    assert normalize_label(" Me-THF ") == normalize_label("me thf")
    assert normalize_label("THF") != normalize_label("Me-THF")


def test_distinct_conditions_are_not_grouped():
    """Rows differing in any condition beyond tolerance stay singletons."""
    df = pd.DataFrame([
        _row(),
        _row(temperature=100.0),
        _row(catalyst_id=5),
        _row(base="Et3N"),
        _row(loading=8.0),
    ])
    assert detect_replicate_groups(df).tolist() == [-1, -1, -1, -1, -1]


def test_near_equal_conditions_group_transitively():
    """A chain of rows within tolerance forms one group, not two."""
    df = pd.DataFrame([
        _row(fa_ratio=2.990),
        _row(fa_ratio=2.993),
        _row(fa_ratio=2.996),
        _row(fa_ratio=1.500),
    ])
    ids = detect_replicate_groups(df)
    assert ids.tolist() == [0, 0, 0, -1]


def test_time_must_match_exactly():
    """``time`` carries no tolerance: a 16 h run never merges with a 20 h run."""
    df = pd.DataFrame([_row(time=16.0), _row(time=20.0)])
    assert detect_replicate_groups(df).tolist() == [-1, -1]


def test_missing_condition_column_is_an_error():
    """A frame without the condition columns fails rather than grouping nothing."""
    with pytest.raises(ValueError, match="missing condition columns"):
        detect_replicate_groups(pd.DataFrame({"catalyst_id": [1, 1]}))


def test_design_sheet_yields_the_five_known_replicate_groups():
    """The sheet's five annotated repeat sets are all detected, and only those.

    Two of them are the reason the detector cannot match exactly: the H2O triplet
    is logged at 2.993/2.990/2.990 eq, and the dioxane triplet spells its solvent
    two different ways.
    """
    df = build_experiments_table(EXCEL, paths.LIB_DIR)
    groups = {
        gid: sorted(df.loc[df["replicate_id"] == gid, "yield_pct"].tolist())
        for gid in sorted(set(df["replicate_id"]) - {-1})
    }
    assert sorted(groups.values()) == sorted(
        [[19, 22, 22], [43, 44, 44], [50, 51], [15, 16, 17], [5, 6, 6]]
    )


def test_stored_experiments_table_matches_the_design_sheet():
    """``experiments.parquet`` is exactly what the builder produces from the xlsx."""
    stored = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    rebuilt = build_experiments_table(EXCEL, paths.LIB_DIR)
    pd.testing.assert_frame_equal(stored, rebuilt, check_dtype=True)


def test_base_active_tracks_the_catalyst_form_not_the_base_column():
    """A non-salt catalyst is base-inactive even on the run where base was added.

    Catalyst 8 is a CO2 adduct run with Rb2CO3. It self-activates, so the study
    zeroes its base block; deriving ``base_active`` from the base column would
    make it the only non-salt row carrying a live base.
    """
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    row = df[(df["catalyst_id"] == 8) & (df["base"] == "Rb2CO3")]
    assert len(row) == 1
    assert int(row["base_active"].iloc[0]) == 0


def test_noise_prior_is_the_conjugate_posterior_of_the_replicates():
    """alpha = nu/2 and beta = S/2 over the pooled within-group sums of squares."""
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    y = df["yield_pct"].to_numpy() / 100.0
    ids = df["replicate_id"].to_numpy()
    ev = estimate_noise_evidence(y, ids)

    expected_ss, expected_dof = 0.0, 0
    for gid in sorted(set(ids.tolist()) - {-1}):
        values = y[ids == gid]
        expected_ss += float(((values - values.mean()) ** 2).sum())
        expected_dof += len(values) - 1

    assert ev.dof == expected_dof
    assert ev.sum_squares == pytest.approx(expected_ss)
    assert ev.alpha == pytest.approx(expected_dof / 2.0)
    assert ev.beta == pytest.approx(expected_ss / 2.0)
    assert ev.prior_mean == pytest.approx(ev.beta / (ev.alpha - 1.0))
    assert ev.prior_cv == pytest.approx(np.sqrt(2.0 / (ev.dof - 4.0)))


def test_prior_tightens_as_replicate_evidence_grows():
    """More replicate groups at the same spread means a smaller prior CV.

    This is the property the design is for: confidence in the noise prior is a
    function of how much repeatability was measured, never a hand-set constant.
    """
    rng = np.random.default_rng(0)
    spread = rng.normal(0.0, 0.012, size=3)
    few_y, few_ids, many_y, many_ids = [], [], [], []
    for group in range(8):
        values = 0.4 + spread
        many_y.extend(values.tolist())
        many_ids.extend([group] * 3)
        if group < 3:
            few_y.extend(values.tolist())
            few_ids.extend([group] * 3)

    few = estimate_noise_evidence(np.array(few_y), np.array(few_ids))
    many = estimate_noise_evidence(np.array(many_y), np.array(many_ids))
    assert many.dof > few.dof
    assert many.prior_cv < few.prior_cv
    assert many.variance == pytest.approx(few.variance)


def test_thin_replicate_evidence_is_an_error_not_a_guess():
    """Below MIN_REPLICATE_DOF the prior has no finite variance; refuse to build it."""
    y = np.array([0.40, 0.42, 0.51, 0.50])
    ids = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="too thin"):
        estimate_noise_evidence(y, ids)
    assert MIN_REPLICATE_DOF >= 5


def test_config_holds_the_replicate_prior_of_every_target_and_scale():
    """config's noise prior per (target, transform) pools the replicates on that scale.

    The values are the ones a GP of that target and transform is trained on, so
    the evidence for probit yield is the spread of probit-transformed yields, not
    the raw spread rescaled.
    """
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    ids = df["replicate_id"].to_numpy()
    assert set(config.NOISE_EVIDENCE) == {
        (target, name) for target in NOISE_TARGET_COLUMNS for name in OUTCOME_TRANSFORM_NAMES
    }
    for (target, name), evidence in config.NOISE_EVIDENCE.items():
        fraction = torch.tensor(df[NOISE_TARGET_COLUMNS[target]].to_numpy() / 100.0)
        expected = estimate_noise_evidence(forward(fraction, name).numpy(), ids)
        assert evidence.sum_squares == pytest.approx(expected.sum_squares)
        assert evidence.alpha == pytest.approx(evidence.dof / 2.0)
        assert evidence.beta == pytest.approx(evidence.sum_squares / 2.0)


def test_the_noise_prior_sits_on_the_measured_repeatability():
    """The prior mean is S/(nu-2) from the replicates, with nothing layered on.

    That mean sits slightly above the pooled variance S/nu — the gap is the
    posterior's own spread, not a fudge factor.
    """
    evidence = config.NOISE_EVIDENCE[("Y", "raw")]
    mean = evidence.beta / (evidence.alpha - 1.0)
    assert mean == pytest.approx(evidence.prior_mean)
    assert mean == pytest.approx(evidence.sum_squares / (evidence.dof - 2))
    assert mean > evidence.variance


def test_non_finite_replicate_values_are_an_error():
    """A NaN inside a replicate group has no within-group spread; refuse it."""
    values = np.array([0.40, np.nan, 0.41, 0.50, 0.52, 0.51, 0.20, 0.22, 0.21])
    ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError, match="non-finite"):
        estimate_noise_evidence(values, ids)


def test_evidence_by_scale_needs_every_target_column():
    """A table without a target column cannot state that target's noise."""
    df = pd.read_parquet(paths.EXPERIMENTS_PARQUET).drop(columns=["selectivity"])
    with pytest.raises(ValueError, match="selectivity"):
        noise_evidence_by_scale(df)


TRAIN_IDX = list(range(0, 60))


@pytest.fixture(scope="module")
def cell():
    return assemble_features(
        cat_featurization="interpretable",
        solv_featurization="OHE",
        base_featurization="OHE",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )


def _builder(cell, outcome_transform):
    return AcyloinCondensationModel(
        cell_features=cell,
        train_idx=TRAIN_IDX,
        kernel_structure="KS1",
        base_kernel_type="matern52",
        outcome_transform=outcome_transform,
        lengthscale_scale=1.0,
    )


def _prior_mean_on_model_scale(prior, train_var):
    """Prior-mean noise variance mapped from the standardized scale back to the model's."""
    return float(prior.rate / (prior.concentration - 1.0) * train_var)


@pytest.mark.parametrize("outcome_transform", OUTCOME_TRANSFORM_NAMES)
def test_single_task_prior_mean_is_the_replicate_prior_mean_on_its_scale(cell, outcome_transform):
    """Mapped back from the standardized scale, the st_y prior mean is the replicate one.

    That is the property the prior exists for, on every outcome transform: a
    probit-yield GP expects the probit-scale replicate noise.
    """
    model = _builder(cell, outcome_transform).build("st_y", None)
    train_y = get_y_yield(cell, outcome_transform)[TRAIN_IDX].squeeze(-1)
    evidence = config.NOISE_EVIDENCE[("Y", outcome_transform)]
    prior = model.likelihood.noise_covar.noise_prior
    assert float(prior.concentration) == pytest.approx(evidence.alpha)
    assert _prior_mean_on_model_scale(prior, train_y.var()) == pytest.approx(evidence.prior_mean)


@pytest.mark.parametrize("outcome_transform", ["raw", "probit"])
def test_independent_pair_gives_each_target_its_own_prior(cell, outcome_transform):
    """In indep_ys the yield GP gets yield's replicate prior and the selectivity GP its own."""
    model_y, model_s = _builder(cell, outcome_transform).build("indep_ys", None)
    y = get_y_yield(cell, outcome_transform)[TRAIN_IDX].squeeze(-1)
    s = get_y_cs(cell, outcome_transform)[1][TRAIN_IDX].squeeze(-1)
    for model, train_y, target in ((model_y, y, "Y"), (model_s, s, "S")):
        prior = model.likelihood.noise_covar.noise_prior
        expected = config.NOISE_EVIDENCE[(target, outcome_transform)].prior_mean
        assert _prior_mean_on_model_scale(prior, train_y.var()) == pytest.approx(expected)


@pytest.mark.parametrize("model_class", ["icm_cs", "lcm_cs"])
def test_two_task_prior_rate_averages_the_standardized_replicate_rates(cell, model_class):
    """The one prior shared by both task noises averages C's and S's standardized rates."""
    model = _builder(cell, "probit").build(model_class, None)
    conversion, selectivity = (t[TRAIN_IDX].squeeze(-1) for t in get_y_cs(cell, "probit"))
    expected = 0.5 * (
        config.NOISE_EVIDENCE[("C", "probit")].beta / conversion.var()
        + config.NOISE_EVIDENCE[("S", "probit")].beta / selectivity.var()
    )
    assert float(model.likelihood.raw_task_noises_prior.rate) == pytest.approx(float(expected))


def _steric_cell():
    return assemble_features(
        cat_featurization="interpretable_steric",
        solv_featurization="KT",
        base_featurization="thermo",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )


def test_target_scale_puts_the_noise_prior_in_the_targets_units():
    """Scaling raw targets by s and passing target_scale=s leaves the standardized noise prior unchanged."""
    from pablo.featurization_and_kernel_scan.core.kernels import build_partition

    fraction = _steric_cell()
    fraction.partition = build_partition(fraction.feature_dims)
    percent = _steric_cell()
    percent.partition = fraction.partition
    percent.y_raw = percent.y_raw * 100.0
    train_idx = list(range(fraction.X.shape[0]))
    rates = []
    for cell, scale in ((fraction, 1.0), (percent, 100.0)):
        model = AcyloinCondensationModel(
            cell_features=cell, train_idx=train_idx, kernel_structure="KS1", base_kernel_type="matern52",
            outcome_transform="raw", lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE, target_scale=scale,
        ).build("st_y", None)
        prior = [p for name, _, p, _, _ in model.likelihood.named_priors() if "noise" in name][0]
        rates.append(float(prior.rate))
    assert rates[1] == pytest.approx(rates[0], rel=1e-10)


def test_target_scale_needs_the_raw_transform():
    """A target scale with a non-linear outcome transform is rejected."""
    with pytest.raises(ValueError, match="target_scale"):
        AcyloinCondensationModel(
            cell_features=_steric_cell(), train_idx=[0, 1], kernel_structure="KS1", base_kernel_type="matern52",
            outcome_transform="probit", lengthscale_scale=1.0, target_scale=100.0,
        )
