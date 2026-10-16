"""Axis grouping, cross-sweep ranking and the paired fold bootstrap of the gem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan.core.config import EliminationGrid
from pablo.featurization_and_kernel_scan.results_analysis.axis_sensitivity import (
    BASELINE_COLUMN,
    always_top,
    axis_bootstrap,
    binding_summary,
    bootstrap_gems,
    config_positions,
    fold_sums,
    group_summary,
    level_group,
    level_label,
    level_spread,
    missing_cells,
    observed_gems,
    pairwise_win_matrix,
    paired_tests,
    partial_rank_agreement,
    paired_bootstrap,
    pivot,
    pooled_mae,
    rank_table,
    sweep_ranges,
    ranked_cells,
    ranking_matrix,
    row_spearman,
    sweep_table,
    top_k_covering,
    top_k_overlap,
    versus_first,
    winner_counts,
)
from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import (
    SPLIT_FAMILIES,
    compute_gem_scores,
)

BASELINE = {
    "solv_feat": "OHE",
    "base_feat": "OHE",
    "base_kernel": "matern52",
    "kernel_structure": "KS1",
    "mean_function": "constant",
    "outcome_transform": "raw",
    "prior_scheme": "P1",
    "model_class": "st_y",
}
GRID = EliminationGrid(
    cat_feats=("a", "a_pca", "b"),
    baseline=BASELINE,
    sweeps={"kernel_structure": ("KS1", "KS2", "KS3"), "prior_scheme": ("P1", "P2")},
    note="test grid",
)


def _scores(gems: dict[tuple[str, str, str], float]) -> pd.DataFrame:
    """A gem_scores-like frame; keys are (cat_feat, axis, value), the baseline under ("", "")."""
    rows = []
    for (cat_feat, axis, value), gem in gems.items():
        cfg = dict(BASELINE) if not axis else {**BASELINE, axis: value}
        rows.append({
            "cat_feat": cat_feat,
            **cfg,
            "gem": gem,
            "gem_study": "primary",
            "crpss_pooled_primary": gem,
            "crpss_pooled_leave_one_out": gem + 0.1,
        })
    return pd.DataFrame(rows)


def _full_scores() -> pd.DataFrame:
    gems = {}
    for i, c in enumerate(GRID.cat_feats):
        gems[(c, "", "")] = 0.1 * i
        gems[(c, "kernel_structure", "KS2")] = 0.1 * i + 0.05
        gems[(c, "kernel_structure", "KS3")] = 0.3 - 0.1 * i
        gems[(c, "prior_scheme", "P2")] = 0.1 * i - 0.02
    return _scores(gems)


def test_level_label_marks_reduced_levels():
    """A `_pca` level is labelled (pca), any other level (full)."""
    assert level_label("grover_pca") == "grover (pca)"
    assert level_label("grover") == "grover (full)"
    assert level_label("interpretable_dG") == "interpretable_dG (full)"


def test_sweep_table_holds_the_baseline_in_every_sweep():
    """Each sweep has every level × value once, with the baseline row shared between sweeps."""
    scores = _full_scores()
    ks = sweep_table(scores, GRID, "kernel_structure")
    ps = sweep_table(scores, GRID, "prior_scheme")
    assert len(ks) == 3 * 3 and len(ps) == 3 * 2
    assert ks["value"].tolist()[:3] == ["KS1", "KS2", "KS3"]
    assert ks["cat_feat"].unique().tolist() == list(GRID.cat_feats)
    base_ks = ks[ks["value"] == "KS1"].set_index("cat_feat")["gem"]
    base_ps = ps[ps["value"] == "P1"].set_index("cat_feat")["gem"]
    pd.testing.assert_series_equal(base_ks, base_ps)
    assert (ks["prior_scheme"] == "P1").all()


def test_sweep_table_raises_on_a_missing_cell():
    """An absent (level, value) pair is an error, not a shorter sweep."""
    scores = _full_scores()
    scores = scores[~((scores["cat_feat"] == "b") & (scores["kernel_structure"] == "KS3"))]
    with pytest.raises(ValueError, match="incomplete"):
        sweep_table(scores, GRID, "kernel_structure")


def test_win_counts_and_matrix():
    """Winner counts sum to the number of levels; entry [A, B] counts levels where A beats B."""
    wide = pivot(sweep_table(_full_scores(), GRID, "kernel_structure"))
    assert winner_counts(wide).to_dict() == {"KS1": 0, "KS2": 1, "KS3": 2}
    wins = pairwise_win_matrix(wide)
    assert wins.loc["KS2", "KS1"] == 3
    assert wins.loc["KS1", "KS2"] == 0
    assert wins.loc["KS3", "KS1"] == 2
    assert (np.diag(wins) == 0).all()


def test_ranking_matrix_shows_the_baseline_once_and_ranks_descending():
    """The shared baseline is one column; rank 1 is the highest gem."""
    scores = _full_scores()
    wides = {a: pivot(sweep_table(scores, GRID, a)) for a in GRID.sweeps}
    ranking = ranking_matrix(wides, GRID)
    assert ranking.columns.tolist() == [BASELINE_COLUMN, "KS2", "KS3", "P2"]
    ranks = rank_table(ranking)
    assert ranks.loc["b", BASELINE_COLUMN] == 1
    assert ranks.loc["a", "KS3"] == 1
    cells = ranked_cells(ranking)
    assert cells.loc[1, BASELINE_COLUMN] == "b"
    assert cells.loc[1, f"{BASELINE_COLUMN}_gem"] == pytest.approx(0.2)
    assert cells.loc[3, "KS3"] == "b"
    assert top_k_overlap(ranks, 1).loc[BASELINE_COLUMN, "KS3"] == 0
    assert always_top(ranks, 2) == ["a_pca"]


def test_ranking_matrix_rejects_differing_baselines():
    """Two sweeps whose baseline columns disagree cannot share one column."""
    scores = _full_scores()
    wides = {a: pivot(sweep_table(scores, GRID, a)) for a in GRID.sweeps}
    wides["prior_scheme"].loc["a", "P1"] += 1.0
    with pytest.raises(ValueError, match="baseline column"):
        ranking_matrix(wides, GRID)


def _aggregate(model_scale: dict[str, np.ndarray], n_folds: int = 8) -> pd.DataFrame:
    """Gem-study fold rows: baseline CRPS rises with the fold, model = scale × baseline."""
    rows = []
    for cat_feat, scale in model_scale.items():
        for study in ("primary", "leave_one_out"):
            for fold in range(n_folds):
                base = 0.1 * (fold + 1)
                rows.append({
                    "cat_feat": cat_feat,
                    "split_type": study,
                    "repeat": 0,
                    "fold": fold,
                    "status": "ok",
                    "crps_per_point": [scale[fold] * base],
                    "baseline_crps_per_point": [base],
                })
    return pd.DataFrame(rows)


def test_observed_gems_reproduce_gem_scores():
    """Unit fold weights give the pooled gem that compute_gem_scores computes."""
    rng = np.random.default_rng(0)
    agg = _aggregate({"x": rng.uniform(0.3, 1.2, 8), "y": rng.uniform(0.3, 1.2, 8)})
    configs = pd.DataFrame({"cat_feat": ["x", "y"]})
    obs = observed_gems(fold_sums(agg, configs))["gem"][0]

    other = pd.DataFrame([
        {"cat_feat": c, "split_type": s, "repeat": 0, "fold": 0, "status": "ok",
         "crps_per_point": [0.1], "baseline_crps_per_point": [0.1],
         "msll": 0.0, "mae": 0.0, "rmse": 0.0, "q2": 0.0, "calibration_ks": 0.0}
        for c in ("x", "y") for s in SPLIT_FAMILIES if s not in ("primary", "leave_one_out")
    ])
    full = pd.concat([agg.assign(msll=0.0, mae=0.0, rmse=0.0, q2=0.0, calibration_ks=0.0), other])
    expected = full[["split_type", "repeat", "fold"]].drop_duplicates()
    scores = compute_gem_scores(full, expected).set_index("cat_feat")
    assert obs == pytest.approx(scores.loc[["x", "y"], "gem"].to_numpy())


@pytest.mark.parametrize("bayesian", [False, True])
def test_bootstrap_is_noiseless_when_every_fold_has_the_same_skill(bayesian):
    """Model CRPS a fixed fraction of baseline CRPS on every fold pools to that skill in every draw."""
    agg = _aggregate({"x": np.full(8, 0.7)})
    boot = bootstrap_gems(
        fold_sums(agg, pd.DataFrame({"cat_feat": ["x"]})), n_boot=200, seed=1, bayesian=bayesian
    )
    assert np.allclose(boot["gem"], 0.3)


@pytest.mark.parametrize("bayesian", [False, True])
def test_bootstrap_is_paired_across_configurations(bayesian):
    """Two identical configurations have noisy gems but a gem difference of exactly zero in every draw."""
    scale = np.linspace(0.2, 1.4, 8)
    agg = _aggregate({"x": scale, "y": scale, "z": scale[::-1]})
    configs = pd.DataFrame({"cat_feat": ["x", "y", "z"]})
    boot = bootstrap_gems(fold_sums(agg, configs), n_boot=500, seed=2, bayesian=bayesian)["gem"]
    assert boot[:, 0].std() > 0.01
    assert np.array_equal(boot[:, 0], boot[:, 1])
    assert (boot[:, 0] - boot[:, 2]).std() > 0.01


def test_bootstrap_is_reproducible_under_a_seed():
    """The same seed gives the same draws."""
    agg = _aggregate({"x": np.linspace(0.2, 1.4, 8)})
    sums = fold_sums(agg, pd.DataFrame({"cat_feat": ["x"]}))
    assert np.array_equal(bootstrap_gems(sums, 50, 3)["gem"], bootstrap_gems(sums, 50, 3)["gem"])
    assert np.array_equal(
        bootstrap_gems(sums, 50, 3, bayesian=True)["gem"],
        bootstrap_gems(sums, 50, 3, bayesian=True)["gem"],
    )


def test_bayesian_bootstrap_is_continuous_and_matches_the_classical_spread():
    """Dirichlet weights keep every fold in every draw, so the draws are continuous, with the classical bootstrap's mean and spread."""
    agg = _aggregate({"x": np.linspace(0.2, 1.4, 8)})
    sums = fold_sums(agg, pd.DataFrame({"cat_feat": ["x"]}))
    classical = bootstrap_gems(sums, 4000, 4)["gem"][:, 0]
    bayes = bootstrap_gems(sums, 4000, 4, bayesian=True)["gem"][:, 0]
    assert np.unique(bayes).size == bayes.size
    assert np.unique(classical).size < classical.size
    assert np.mean(bayes) == pytest.approx(np.mean(classical), abs=0.01)
    assert bayes.std() == pytest.approx(classical.std(), rel=0.15)


def test_fold_sums_rejects_failed_and_missing_folds():
    """The paired bootstrap needs every fold ok and present for every configuration."""
    agg = _aggregate({"x": np.full(8, 0.5), "y": np.full(8, 0.5)})
    configs = pd.DataFrame({"cat_feat": ["x", "y"]})
    failed = agg.copy()
    failed.loc[0, "status"] = "failed"
    with pytest.raises(ValueError, match="failed folds"):
        fold_sums(failed, configs)
    with pytest.raises(ValueError, match="lack folds"):
        fold_sums(agg.drop(index=0), configs)


def test_paired_bootstrap_verdicts():
    """A difference present on every fold is resolved; identical configurations are indistinguishable."""
    table = pd.DataFrame({
        "cat_feat": ["c", "c", "c"],
        "value": ["v1", "v2", "v3"],
        "gem": [0.5, 0.3, 0.5],
    })
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 0.05, 400)
    boot = np.column_stack([0.5 + noise, 0.3 + noise, 0.5 + noise])
    pairs = paired_bootstrap(table, boot, np.array([0, 1, 2])).set_index(["a", "b"])
    assert pairs.loc[("v1", "v2"), "verdict"] == "v1"
    assert pairs.loc[("v2", "v3"), "verdict"] == "v3"
    assert pairs.loc[("v1", "v3"), "verdict"] == "indistinguishable"
    assert pairs.loc[("v1", "v2"), "diff"] == pytest.approx(0.2)
    assert pairs.loc[("v1", "v2"), "sd"] == pytest.approx(0.0, abs=1e-12)


def test_axis_bootstrap_counts_wins_across_levels():
    """The across-level comparison reports the observed median difference and win count."""
    table = pd.DataFrame({
        "cat_feat": ["c", "c", "d", "d"],
        "value": ["v1", "v2", "v1", "v2"],
        "gem": [0.4, 0.1, 0.2, 0.3],
    })
    boot = np.tile(table["gem"].to_numpy(), (100, 1))
    out = axis_bootstrap(table, boot, np.arange(4)).iloc[0]
    assert out["a_wins"] == 1
    assert out["median_diff"] == pytest.approx(0.1)
    assert out["median_lo"] == pytest.approx(0.1) and out["median_hi"] == pytest.approx(0.1)


def test_config_positions_and_row_spearman():
    """Rows map to their bootstrap columns; row_spearman matches scipy on each row."""
    configs = pd.DataFrame({"cat_feat": ["x", "y", "z"]})
    table = pd.DataFrame({"cat_feat": ["z", "x"]})
    assert config_positions(table, configs).tolist() == [2, 0]
    with pytest.raises(ValueError):
        config_positions(pd.DataFrame({"cat_feat": ["w"]}), configs)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 1.0, 2.0]])
    y = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert row_spearman(x, y) == pytest.approx([0.8, -0.8])


def test_level_group_and_group_summary():
    """Levels split into full, pca and interpretable; per-group wins and medians use only that group."""
    assert level_group("grover") == "full"
    assert level_group("grover_pca") == "pca"
    assert level_group("interpretable_dG") == "interpretable"
    wide = pd.DataFrame(
        {"v1": [0.1, 0.4, 0.2], "v2": [0.3, 0.1, 0.2]},
        index=["grover", "grover_pca", "interpretable"],
    )
    out = group_summary(wide, "v1").set_index(["group", "value"])
    assert out.loc[("full", "v2"), "median_diff"] == pytest.approx(0.2)
    assert out.loc[("full", "v2"), "wins"] == 1
    assert out.loc[("pca", "v1"), "wins"] == 1
    assert out.loc[("interpretable", "v1"), "n_levels"] == 1


def _scores_missing_ks3_on_b() -> pd.DataFrame:
    scores = _full_scores()
    return scores[~((scores["cat_feat"] == "b") & (scores["kernel_structure"] == "KS3"))]


def test_incomplete_sweep_keeps_existing_cells_and_lists_absent_ones():
    """With require_complete=False an absent cell is dropped and reported, not an error."""
    table = sweep_table(_scores_missing_ks3_on_b(), GRID, "kernel_structure", require_complete=False)
    assert len(table) == 3 * 3 - 1
    missing = missing_cells(table, GRID, "kernel_structure")
    assert missing.to_dict("records") == [{"cat_feat": "b", "value": "KS3"}]
    assert missing_cells(sweep_table(_full_scores(), GRID, "prior_scheme"), GRID, "prior_scheme").empty


def test_incomplete_sweep_counts_only_levels_where_values_are_present():
    """Winners use complete levels; pairwise tests and wins use levels with both values."""
    table = sweep_table(_scores_missing_ks3_on_b(), GRID, "kernel_structure", require_complete=False)
    wide = pivot(table)
    assert winner_counts(wide).to_dict() == {"KS1": 0, "KS2": 0, "KS3": 2}
    assert pairwise_win_matrix(wide).loc["KS3", "KS1"] == 2
    tests = paired_tests(wide).set_index(["a", "b"])
    assert tests.loc[("KS1", "KS2"), "n_levels"] == 3
    assert tests.loc[("KS1", "KS3"), "n_levels"] == 2
    binding = binding_summary(table, "KS1").set_index("value")
    assert binding.loc["KS3", "n_levels"] == 2
    assert binding.loc["KS3", "n_switched"] == 0


def test_incomplete_sweep_bootstrap_pairs_skip_absent_cells():
    """Per-level and across-level bootstraps compare only cells that exist."""
    table = sweep_table(_scores_missing_ks3_on_b(), GRID, "kernel_structure", require_complete=False)
    positions = np.arange(len(table))
    boot = np.tile(table["gem"].to_numpy(), (200, 1))
    pairs = paired_bootstrap(table, boot, positions)
    assert len(pairs) == 2 * 3 + 1
    assert not ((pairs["cat_feat"] == "b") & (pairs["b"] == "KS3")).any()
    across = axis_bootstrap(table, boot, positions).set_index(["a", "b"])
    assert across.loc[("KS1", "KS3"), "n_levels"] == 2
    spread = level_spread(pivot(table), pairs).set_index("cat_feat")
    assert spread.loc["b", "n_values"] == 2


def test_partial_rank_agreement_uses_common_levels():
    """A value's column is compared with the baseline column on the levels both have."""
    table = sweep_table(_scores_missing_ks3_on_b(), GRID, "kernel_structure", require_complete=False)
    rng = np.random.default_rng(5)
    boot = table["gem"].to_numpy() + rng.normal(0.0, 0.01, (400, len(table)))
    out = partial_rank_agreement(table, "KS1", boot, np.arange(len(table)), k=1).set_index("value")
    assert out.loc["KS3", "n_levels"] == 2
    assert out.loc["KS2", "n_levels"] == 3
    assert out.loc["KS2", "rho"] == pytest.approx(1.0)
    assert out.loc["KS3", "rho"] == pytest.approx(-1.0)
    assert out.loc["KS2", "top1_overlap"] == 1


def test_top_k_covering_stops_at_the_nth_new_level():
    """K is the smallest count of best configurations that spans n catalyst levels."""
    scores = pd.DataFrame({
        "cat_feat": ["a", "a", "b", "a", "c", "b"],
        "crpss_pooled_leave_one_out": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    }).sample(frac=1.0, random_state=0)
    top = top_k_covering(scores, n_levels=2)
    assert top["rank"].tolist() == [1, 2, 3]
    assert top["cat_feat"].tolist() == ["a", "a", "b"]
    assert top["new_level"].tolist() == [True, False, True]
    assert len(top_k_covering(scores, n_levels=3)) == 5
    with pytest.raises(ValueError):
        top_k_covering(scores, n_levels=4)


def test_versus_first_is_paired_on_the_first_configuration():
    """The first configuration differs from itself by zero; a constant offset has zero spread."""
    rng = np.random.default_rng(6)
    base = rng.normal(0.3, 0.1, 300)
    boot = np.column_stack([base, base - 0.05, base + rng.normal(0.0, 0.2, 300)])
    out = versus_first(boot, np.array([0, 1, 2]))
    assert out.loc[0, "diff_lo"] == 0.0 and out.loc[0, "diff_hi"] == 0.0
    assert out.loc[1, "diff_lo"] == pytest.approx(-0.05) and out.loc[1, "diff_hi"] == pytest.approx(-0.05)
    assert out.loc[1, "p_above_first"] == 0.0
    assert out.loc[2, "diff_lo"] < 0.0 < out.loc[2, "diff_hi"]


def test_pooled_mae_weights_folds_by_test_points():
    """Pooled MAE is the point-weighted mean of the per-fold MAE over ok folds of the study."""
    agg = pd.DataFrame({
        "cat_feat": ["x", "x", "x", "x"],
        "split_type": ["leave_one_out", "leave_one_out", "leave_one_out", "primary"],
        "status": ["ok", "ok", "failed", "ok"],
        "mae": [0.1, 0.4, 9.0, 9.0],
        "n_test": [3, 1, 1, 5],
    })
    out = pooled_mae(agg, ["cat_feat"], "leave_one_out")
    assert out.loc["x"] == pytest.approx((0.3 + 0.4) / 4)


def test_sweep_ranges_per_level_and_axis():
    """Each level gets the min, max and count of a score over each sweep, absent cells skipped."""
    tables = {
        "kernel_structure": sweep_table(_scores_missing_ks3_on_b(), GRID, "kernel_structure", require_complete=False),
        "prior_scheme": sweep_table(_full_scores(), GRID, "prior_scheme"),
    }
    out = sweep_ranges(tables, "gem")
    assert out.loc["a", "kernel_structure_min"] == pytest.approx(0.0)
    assert out.loc["a", "kernel_structure_max"] == pytest.approx(0.3)
    assert out.loc["b", "kernel_structure_n"] == 2
    assert out.loc["b", "kernel_structure_max"] == pytest.approx(0.25)
    assert out.loc["b", "prior_scheme_min"] == pytest.approx(0.18)


def test_top_k_covering_extends_until_an_off_baseline_value_appears():
    """K grows past the level-coverage point until every required axis leaves its baseline."""
    scores = pd.DataFrame({
        "cat_feat": ["a", "b", "a", "c"],
        "kernel_structure": ["KS1", "KS1", "KS1", "KS2"],
        "crpss_pooled_leave_one_out": [0.9, 0.8, 0.7, 0.6],
    })
    assert len(top_k_covering(scores, n_levels=2)) == 2
    assert len(top_k_covering(scores, n_levels=2, off_baseline={"kernel_structure": "KS1"})) == 4
    with pytest.raises(ValueError):
        top_k_covering(scores.iloc[:3], n_levels=2, off_baseline={"kernel_structure": "KS1"})
