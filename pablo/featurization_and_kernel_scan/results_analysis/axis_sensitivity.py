"""Axis sensitivity — how the gem of every catalyst level responds to a one-axis sweep.

The elimination grid (``config.ELIMINATION_GRID``) fits every catalyst level at
one baseline and moves one non-catalyst axis at a time. For each swept axis this
script puts the 50 levels side by side across the axis's values and measures:

* the per-level gem difference of every value against the axis's baseline value;
* how many levels each value wins, and the pairwise win matrix;
* a sign test and a Wilcoxon signed-rank test over the levels, per pair of values;
* which study (``primary`` or ``leave_one_out``) binds the gem, per value;
* the median difference and win count within each level group (full-dimensional,
  PCA-reduced, interpretable);
* the fold-bootstrap noise of the pooled gem, and per level and pair of values
  whether the paired bootstrap interval of the gem difference excludes zero;
* the paired bootstrap interval of the across-level median difference and win
  count, which keeps the correlation the shared folds put between levels;
* over every configuration, the top ``K`` by pooled ``leave_one_out`` CRPSS and by
  the gem (``TOP_K_RANKINGS``), with
  ``K`` the smallest count whose configurations span ``TOP_K_LEVELS`` catalyst
  levels and hold a non-baseline kernel structure and mean function
  (``top_k_covering``), each against the top one in the paired bootstrap
  and with its level's ``leave_one_out`` CRPSS range over every sweep
  (``sweep_ranges``).

**Incomplete sweeps.** A sweep named in ``--allow-incomplete`` (by default
``mean_function``) may lack cells. Its absent cells are listed; winner counts use
the levels on which every value is present, and every pairwise quantity uses the
levels on which both values are present. It stays out of the cross-sweep
ranking, and its columns are compared with the baseline column on their common
levels instead (``partial_rank_agreement``).

The gem is read from ``gem_scores.parquet`` as ``results_analysis/gem_scores.py``
computes it: min(pooled CRPSS on ``primary``, pooled CRPSS on ``leave_one_out``).

**Bootstrap.** The folds of ``primary`` (60) and ``leave_one_out`` (60) are
resampled with replacement, each study independently. One draw of fold counts is
shared by every configuration, so configurations stay paired on
``(split_type, repeat, fold)`` and a gem difference inside one level is
bootstrapped as a paired difference. Per draw the pooled CRPSS of each study is
``1 − Σ c_f · CRPS_model,f / Σ c_f · CRPS_baseline,f`` with ``c_f`` the fold's
draw count, and the gem is the minimum over the two studies. With ``--bayesian``
the counts are replaced by Bayesian-bootstrap weights (Rubin 1981):
``c = n · w`` with ``w ~ Dirichlet(1, …, 1)`` over the ``n`` folds of the study,
so every fold keeps a positive, continuous weight; the pairing is unchanged.

**Ranking.** Across the swept values the 50 levels are ranked by gem within each
column. The baseline configuration is a member of every sweep, so its column
(``KS1`` = ``P1`` = ``raw``) appears once. The Spearman ρ between two columns
is read against the ρ between two independent bootstrap draws of one column.

Outputs (CSV and one Markdown file) go to ``results/axis_sensitivity/``.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.results_analysis.axis_sensitivity
    python -m pablo.featurization_and_kernel_scan.results_analysis.axis_sensitivity --n-boot 4000 --seed 1
    python -m pablo.featurization_and_kernel_scan.results_analysis.axis_sensitivity --bayesian
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.config import EliminationGrid
from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import (
    sweep_membership,
)
from pablo.featurization_and_kernel_scan.features.dim_reduction import (
    is_reduced,
    source_name,
)
from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import (
    GEM_STUDIES,
    fold_status,
)

AXES: tuple[str, ...] = ("kernel_structure", "prior_scheme", "outcome_transform", "mean_function")
"""The sweeps analysed by default."""

ALLOW_INCOMPLETE: tuple[str, ...] = ("mean_function",)
"""Sweeps analysed on the cells that exist when some are absent."""

GEM_THRESHOLD: float = 0.1
"""The gem the elimination rule in ``WORK-PACKAGE.md`` asks a level to clear."""

BASELINE_COLUMN: str = "baseline"
"""Name of the baseline configuration's column in the cross-sweep ranking."""

TOP_K_LEVELS: int = 10
"""Distinct catalyst levels the top-K table of the ``leave_one_out`` ranking must span."""

TOP_K_OFF_BASELINE: tuple[str, ...] = ("kernel_structure", "mean_function")
"""Axes on which the top-K table must hold at least one non-baseline value."""

TOP_K_RANKINGS: dict[str, tuple[str, str]] = {
    "loo": ("crpss_pooled_leave_one_out", "leave_one_out"),
    "gem": ("gem", "gem"),
}
"""Top-K tables written: name → (score column ranked on, its key in the bootstrap output)."""

N_BOOT: int = 2000
SEED: int = 0
TOP_K: tuple[int, ...] = (5, 10)
CI_LEVEL: float = 0.95

DEFAULT_SCORES = paths.RESULTS_DIR / "gem_scores.parquet"
DEFAULT_AGGREGATE = paths.SCAN_AGGREGATE_PARQUET
DEFAULT_OUT_DIR = paths.RESULTS_DIR / "axis_sensitivity"

SWEEP_COLUMNS: tuple[str, ...] = (
    "gem",
    "gem_study",
    "crpss_pooled_primary",
    "crpss_pooled_leave_one_out",
)
"""Per-configuration columns carried from ``gem_scores.parquet`` into a sweep."""


def level_label(cat_feat: str) -> str:
    """Display name of a catalyst level, e.g. ``grover (pca)`` or ``grover (full)``."""
    return f"{source_name(cat_feat)} ({'pca' if is_reduced(cat_feat) else 'full'})"


def level_group(cat_feat: str) -> str:
    """The family of a catalyst level: ``interpretable``, ``pca`` (reduced) or ``full``."""
    if source_name(cat_feat) in {s.name for s in config.CATALYST_SETS["interpretable"]}:
        return "interpretable"
    return "pca" if is_reduced(cat_feat) else "full"


def group_summary(wide: pd.DataFrame, baseline_value: str) -> pd.DataFrame:
    """Median gem, median difference against the baseline value and wins, per level group.

    Args:
        wide: Level × value gem matrix.
        baseline_value: The column the differences are taken against.

    Returns:
        One row per (``group`` of ``level_group``, ``value``): ``n_levels``
        (levels of the group with the value present), ``median_gem``,
        ``median_diff``, ``n_complete`` (levels of the group with every value
        present) and ``wins`` (those of them on which the value has the highest
        gem).
    """
    groups = wide.index.map(level_group)
    diff = wide.sub(wide[baseline_value], axis=0)
    complete = wide.notna().all(axis=1).to_numpy()
    best = wide.idxmax(axis=1)
    rows = []
    for group in ("full", "pca", "interpretable"):
        mask = np.asarray(groups == group)
        for value in wide.columns:
            rows.append({
                "group": group,
                "value": value,
                "n_levels": int(wide.loc[mask, value].notna().sum()),
                "median_gem": wide.loc[mask, value].median(),
                "median_diff": diff.loc[mask, value].median(),
                "n_complete": int((mask & complete).sum()),
                "wins": int((best[mask & complete] == value).sum()),
            })
    return pd.DataFrame(rows)


def sweep_table(
    scores: pd.DataFrame, grid: EliminationGrid, axis: str, require_complete: bool = True
) -> pd.DataFrame:
    """The configurations of one sweep, one row per (catalyst level, axis value).

    Args:
        scores: ``gem_scores.parquet``, carrying ``cat_feat``, every key of
            ``grid.baseline`` and ``SWEEP_COLUMNS``.
        grid: The grid the scores were computed on.
        axis: A key of ``grid.sweeps``.
        require_complete: Whether an absent (level, value) pair is an error.

    Returns:
        The rows of ``scores`` that belong to the ``axis`` sweep (the baseline
        included), with the added column ``value`` holding the axis value,
        ordered by ``grid.cat_feats`` and then ``grid.sweeps[axis]``. With
        ``require_complete``, exactly ``len(grid.cat_feats) · len(grid.sweeps[axis])``
        rows.

    Raises:
        ValueError: If a (level, value) pair is duplicated, absent while
            ``require_complete``, or a row's catalyst level is not in the grid.
    """
    keys = list(grid.baseline)
    in_sweep = scores[keys].apply(
        lambda row: axis in sweep_membership(row.to_dict(), grid), axis=1
    )
    table = scores[in_sweep.to_numpy(dtype=bool)].assign(value=lambda d: d[axis])
    unknown = set(table["cat_feat"]) - set(grid.cat_feats)
    if unknown:
        raise ValueError(f"catalyst levels not in the grid: {sorted(unknown)}")
    counts = table.groupby(["cat_feat", "value"]).size()
    expected = pd.MultiIndex.from_product(
        [grid.cat_feats, grid.sweeps[axis]], names=["cat_feat", "value"]
    )
    counts = counts.reindex(expected, fill_value=0)
    bad = counts[(counts > 1) | ((counts == 0) & require_complete)]
    if len(bad):
        raise ValueError(
            f"sweep {axis!r} is incomplete or duplicated at {len(bad)} (level, value) "
            f"pairs, e.g. {bad.head(5).to_dict()}"
        )
    order = {c: i for i, c in enumerate(grid.cat_feats)}
    value_order = {v: i for i, v in enumerate(grid.sweeps[axis])}
    return table.sort_values(
        ["cat_feat", "value"],
        key=lambda s: s.map(order if s.name == "cat_feat" else value_order),
        ignore_index=True,
    )


def missing_cells(table: pd.DataFrame, grid: EliminationGrid, axis: str) -> pd.DataFrame:
    """The (level, value) pairs of a sweep that have no row in ``table``.

    Args:
        table: Output of ``sweep_table``.
        grid: The grid the sweep comes from.
        axis: The swept axis.

    Returns:
        One row per absent pair, columns ``cat_feat`` and ``value``, in grid order.
    """
    expected = pd.MultiIndex.from_product(
        [grid.cat_feats, grid.sweeps[axis]], names=["cat_feat", "value"]
    )
    present = pd.MultiIndex.from_frame(table[["cat_feat", "value"]])
    return expected[~expected.isin(present)].to_frame(index=False)


def pivot(table: pd.DataFrame, column: str = "gem") -> pd.DataFrame:
    """A sweep as a level × value matrix of one column.

    Args:
        table: Output of ``sweep_table``.
        column: The column to spread.

    Returns:
        Index ``cat_feat`` and columns the axis values, both in the table's order.
    """
    wide = table.pivot(index="cat_feat", columns="value", values=column)
    return wide.loc[table["cat_feat"].unique(), table["value"].unique()]


def baseline_differences(wide: pd.DataFrame, baseline_value: str) -> pd.DataFrame:
    """Per-level gem difference of every other value against the baseline value.

    Args:
        wide: Level × value gem matrix (``pivot``).
        baseline_value: The column the differences are taken against.

    Returns:
        Level × (other values) matrix of ``gem(value) − gem(baseline_value)``.
    """
    others = [v for v in wide.columns if v != baseline_value]
    return wide[others].sub(wide[baseline_value], axis=0)


def summarise_differences(diff: pd.DataFrame) -> pd.DataFrame:
    """Median, spread and the largest moves of each column of ``baseline_differences``.

    Args:
        diff: Level × (other values) difference matrix from ``baseline_differences``.

    Returns:
        One row per value: ``n_levels`` (levels with both values present),
        ``median``, ``q25``, ``q75``, ``n_up``, ``n_down``, the largest drop and
        rise with the levels they occur on.
    """
    rows = []
    for value in diff.columns:
        d = diff[value]
        rows.append({
            "value": value,
            "n_levels": int(d.notna().sum()),
            "median": d.median(),
            "q25": d.quantile(0.25),
            "q75": d.quantile(0.75),
            "n_up": int((d > 0).sum()),
            "n_down": int((d < 0).sum()),
            "largest_drop": d.min(),
            "largest_drop_level": d.idxmin(),
            "largest_rise": d.max(),
            "largest_rise_level": d.idxmax(),
        })
    return pd.DataFrame(rows)


def winner_counts(wide: pd.DataFrame) -> pd.Series:
    """How many levels each value gives the highest gem, over levels with every value present.

    The first column wins ties.

    Args:
        wide: Level × value gem matrix.

    Returns:
        Win count per value, indexed by the columns of ``wide`` (0 for a value
        that wins no level).
    """
    return wide.dropna().idxmax(axis=1).value_counts().reindex(wide.columns, fill_value=0)


def pairwise_win_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Entry ``[A, B]`` is the number of levels on which value A has a higher gem than B.

    A level lacking A or B counts for neither.

    Args:
        wide: Level × value gem matrix.

    Returns:
        Value × value matrix of win counts, rows and columns in the order of
        ``wide.columns``.
    """
    return pd.DataFrame(
        {b: {a: int((wide[a] > wide[b]).sum()) for a in wide.columns} for b in wide.columns}
    ).loc[wide.columns, wide.columns]


def paired_tests(wide: pd.DataFrame) -> pd.DataFrame:
    """Sign test and Wilcoxon signed-rank test over the levels, per pair of values.

    Args:
        wide: Level × value gem matrix.

    Returns:
        One row per unordered pair ``(a, b)`` in column order: ``n_levels``
        (levels with both present), ``n_a_better``, ``n_b_better``,
        ``median_diff`` (a − b), the two-sided sign-test p-value (ties dropped)
        and the two-sided Wilcoxon p-value.
    """
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        d = (wide[a] - wide[b]).dropna().to_numpy(dtype=float)
        n_a, n_b = int((d > 0).sum()), int((d < 0).sum())
        rows.append({
            "a": a,
            "b": b,
            "n_levels": len(d),
            "n_a_better": n_a,
            "n_b_better": n_b,
            "median_diff": float(np.median(d)),
            "sign_test_p": stats.binomtest(n_a, n_a + n_b, 0.5).pvalue,
            "wilcoxon_p": stats.wilcoxon(d).pvalue,
        })
    return pd.DataFrame(rows)


def binding_summary(table: pd.DataFrame, baseline_value: str) -> pd.DataFrame:
    """Which study binds the gem, per axis value.

    Args:
        table: Output of ``sweep_table``.
        baseline_value: The axis's baseline value.

    Returns:
        One row per value: ``n_levels``, the number of levels bound by each gem
        study, the median of the gem and of both component CRPSS, and
        ``n_switched``, the number of levels whose binding study differs from
        the baseline's (over levels where both are present).
    """
    study = pivot(table, "gem_study")
    rows = []
    for value, group in table.groupby("value", sort=False):
        row = {"value": value, "n_levels": len(group)}
        for s in GEM_STUDIES:
            row[f"n_binds_{s}"] = int((group["gem_study"] == s).sum())
        row["median_gem"] = group["gem"].median()
        for s in GEM_STUDIES:
            row[f"median_crpss_{s}"] = group[f"crpss_pooled_{s}"].median()
        both = study[value].notna() & study[baseline_value].notna()
        row["n_switched"] = int((study.loc[both, value] != study.loc[both, baseline_value]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FoldSums:
    """Per-fold CRPS sums of a set of configurations on the gem studies.

    Attributes:
        configs: One row per configuration, the configuration key columns; row
            ``i`` is column ``i`` of the bootstrap output.
        model: Study → ΣCRPS_model per fold, shape ``n_config × n_fold``.
        baseline: Study → ΣCRPS_baseline per fold, shape ``n_config × n_fold``.
    """

    configs: pd.DataFrame
    model: dict[str, np.ndarray]
    baseline: dict[str, np.ndarray]


def fold_sums(
    aggregate: pd.DataFrame, configs: pd.DataFrame, studies: Sequence[str] = GEM_STUDIES
) -> FoldSums:
    """Collect per-fold CRPS sums, aligned on the same fold order for every configuration.

    Args:
        aggregate: Fold rows of ``scan_aggregate.parquet`` with ``split_type``,
            ``repeat``, ``fold``, ``status``, ``crps_per_point``,
            ``baseline_crps_per_point`` and the configuration key columns.
        configs: The configurations to collect, one row each, key columns only.
        studies: The studies to collect.

    Returns:
        The sums, fold columns in ``(repeat, fold)`` order, rows in ``configs`` order.

    Raises:
        ValueError: If a selected fold failed, or a configuration lacks a fold
            another configuration has (the bootstrap pairs on the fold key).
    """
    keys = list(configs.columns)
    target = pd.MultiIndex.from_frame(configs)
    model, baseline = {}, {}
    for study in studies:
        rows = aggregate[aggregate["split_type"] == study]
        rows = rows[pd.MultiIndex.from_frame(rows[keys]).isin(target)]
        if (fold_status(rows) != "ok").any():
            raise ValueError(
                f"{study}: failed folds among the selected configurations; the paired "
                "fold bootstrap needs every fold ok"
            )
        sums = rows[keys + ["repeat", "fold"]].assign(
            m=rows["crps_per_point"].map(np.sum).astype(float),
            b=rows["baseline_crps_per_point"].map(np.sum).astype(float),
        ).set_index(keys + ["repeat", "fold"])
        m = sums["m"].unstack(["repeat", "fold"]).sort_index(axis=1)
        b = sums["b"].unstack(["repeat", "fold"]).sort_index(axis=1)
        m.index = b.index = pd.MultiIndex.from_frame(m.index.to_frame())
        m, b = m.reindex(target), b.reindex(target)
        if m.isna().any().any() or b.isna().any().any():
            raise ValueError(f"{study}: some configurations lack folds that others have")
        model[study] = m.to_numpy(dtype=float)
        baseline[study] = b.to_numpy(dtype=float)
    return FoldSums(configs=configs.reset_index(drop=True), model=model, baseline=baseline)


def pooled_from_counts(sums: FoldSums, counts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pooled CRPSS per study and the gem under per-fold resampling weights.

    Args:
        sums: Per-fold CRPS sums.
        counts: Study → fold weights, shape ``B × n_fold``; all ones reproduces
            the observed pooled CRPSS.

    Returns:
        Study → pooled CRPSS and ``"gem"`` → their minimum, each ``B × n_config``.
    """
    out = {
        s: 1.0 - (counts[s] @ sums.model[s].T) / (counts[s] @ sums.baseline[s].T)
        for s in sums.model
    }
    out["gem"] = np.minimum.reduce([out[s] for s in GEM_STUDIES])
    return out


def bootstrap_gems(
    sums: FoldSums, n_boot: int, seed: int, bayesian: bool = False
) -> dict[str, np.ndarray]:
    """Paired fold bootstrap of the pooled CRPSS and the gem.

    Each study's folds are reweighted independently of the other study; one draw
    is shared by every configuration. The classical bootstrap draws multinomial
    counts (resampling with replacement); the Bayesian bootstrap (Rubin 1981)
    draws ``n · w`` with ``w ~ Dirichlet(1, …, 1)``, which has the same mean and
    keeps every fold at a positive weight.

    Args:
        sums: Per-fold CRPS sums.
        n_boot: Number of draws ``B``.
        seed: Seed of the generator.
        bayesian: Draw Dirichlet weights instead of multinomial counts.

    Returns:
        Study → pooled CRPSS and ``"gem"``, each ``B × n_config``.
    """
    rng = np.random.default_rng(seed)
    counts = {}
    for study, m in sums.model.items():
        n = m.shape[1]
        if bayesian:
            counts[study] = n * rng.dirichlet(np.ones(n), size=n_boot)
        else:
            counts[study] = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    return pooled_from_counts(sums, counts)


def observed_gems(sums: FoldSums) -> dict[str, np.ndarray]:
    """Pooled CRPSS per study and the gem on the full fold set.

    Args:
        sums: Per-fold CRPS sums.

    Returns:
        Study → pooled CRPSS and ``"gem"`` → their minimum, each ``1 × n_config``.
    """
    return pooled_from_counts(
        sums, {s: np.ones((1, m.shape[1])) for s, m in sums.model.items()}
    )


def config_positions(table: pd.DataFrame, configs: pd.DataFrame) -> np.ndarray:
    """Column of every row of ``table`` in the bootstrap output of ``configs``.

    Args:
        table: Rows carrying every configuration key column of ``configs``.
        configs: The configurations the bootstrap was run on, one per column
            of its ``B × n_config`` output.

    Returns:
        Integer array of length ``len(table)``: the position in ``configs`` of
        each row of ``table``.

    Raises:
        ValueError: If a row of ``table`` is not among ``configs``.
    """
    keys = list(configs.columns)
    index = pd.MultiIndex.from_frame(configs)
    positions = index.get_indexer(pd.MultiIndex.from_frame(table[keys]))
    if (positions < 0).any():
        raise ValueError(f"{int((positions < 0).sum())} sweep rows have no fold sums")
    return positions


def bootstrap_sd(boot: dict[str, np.ndarray], configs: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap standard deviation of the gem and of both component CRPSS per configuration.

    Args:
        boot: Study → pooled CRPSS and ``"gem"``, each ``B × n_config``
            (``bootstrap_gems``).
        configs: The configurations, one row per bootstrap column.

    Returns:
        Copy of ``configs`` with ``sd_gem`` and ``sd_<study>`` for each study
        in ``GEM_STUDIES``.
    """
    out = configs.copy()
    for name in ("gem", *GEM_STUDIES):
        out[f"sd_{name}"] = boot[name].std(axis=0, ddof=1)
    return out


def paired_bootstrap(
    table: pd.DataFrame, boot_gem: np.ndarray, positions: np.ndarray, level: float = CI_LEVEL
) -> pd.DataFrame:
    """Paired bootstrap of the gem difference, per level and pair of axis values.

    Args:
        table: Output of ``sweep_table``.
        boot_gem: Bootstrap gems, ``B × n_config``.
        positions: ``config_positions(table, ...)``.
        level: Coverage of the percentile interval.

    Returns:
        One row per (level, a, b) with both values present: observed ``diff`` (a − b), its bootstrap
        ``sd``, the interval ``lo``/``hi``, ``p_a_better`` (share of draws with
        a above b) and ``verdict``: ``a``, ``b`` or ``indistinguishable``,
        according to whether the interval excludes zero.
    """
    alpha = (1.0 - level) / 2.0
    col = pd.Series(positions, index=pd.MultiIndex.from_frame(table[["cat_feat", "value"]]))
    obs = table.set_index(["cat_feat", "value"])["gem"]
    rows = []
    for cat_feat in table["cat_feat"].unique():
        for a, b in itertools.combinations(table["value"].unique(), 2):
            if (cat_feat, a) not in col.index or (cat_feat, b) not in col.index:
                continue
            d = boot_gem[:, col[(cat_feat, a)]] - boot_gem[:, col[(cat_feat, b)]]
            lo, hi = np.quantile(d, [alpha, 1.0 - alpha])
            rows.append({
                "cat_feat": cat_feat,
                "a": a,
                "b": b,
                "diff": obs[(cat_feat, a)] - obs[(cat_feat, b)],
                "sd": d.std(ddof=1),
                "lo": lo,
                "hi": hi,
                "p_a_better": float((d > 0).mean()),
                "verdict": a if lo > 0 else (b if hi < 0 else "indistinguishable"),
            })
    return pd.DataFrame(rows)


def paired_bootstrap_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per pair of values: levels where each is resolved better, and indistinguishable ones.

    Args:
        pairs: Output of ``paired_bootstrap``.

    Returns:
        One row per pair ``(a, b)``: ``n_a_resolved``, ``n_b_resolved``,
        ``n_indistinguishable``, and the median over levels of the bootstrap
        ``sd`` (``median_sd``) and of ``|diff|`` (``median_abs_diff``).
    """
    rows = []
    for (a, b), g in pairs.groupby(["a", "b"], sort=False):
        rows.append({
            "a": a,
            "b": b,
            "n_a_resolved": int((g["verdict"] == a).sum()),
            "n_b_resolved": int((g["verdict"] == b).sum()),
            "n_indistinguishable": int((g["verdict"] == "indistinguishable").sum()),
            "median_sd": g["sd"].median(),
            "median_abs_diff": g["diff"].abs().median(),
        })
    return pd.DataFrame(rows)


def level_spread(wide: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per level: best and worst value, the gem range across values, and whether it is resolved.

    Args:
        wide: Level × value gem matrix.
        pairs: ``paired_bootstrap`` output for the same sweep.

    Returns:
        One row per level with at least two values present: ``n_values``,
        ``best``, ``worst``, ``range`` (best − worst gem), the paired bootstrap
        ``sd`` and ``resolved`` (interval excludes zero) of the best-vs-worst
        difference, and ``n_resolved_pairs``.
    """
    rows = []
    for cat_feat, gems in wide.iterrows():
        gems = gems.dropna()
        if len(gems) < 2:
            continue
        best, worst = gems.idxmax(), gems.idxmin()
        g = pairs[pairs["cat_feat"] == cat_feat]
        bw = g[((g["a"] == best) & (g["b"] == worst)) | ((g["a"] == worst) & (g["b"] == best))]
        rows.append({
            "cat_feat": cat_feat,
            "n_values": len(gems),
            "best": best,
            "worst": worst,
            "range": gems.max() - gems.min(),
            "sd": float(bw["sd"].iloc[0]),
            "resolved": bool((bw["verdict"] != "indistinguishable").iloc[0]),
            "n_resolved_pairs": int((g["verdict"] != "indistinguishable").sum()),
        })
    return pd.DataFrame(rows)


def ranking_matrix(wides: dict[str, pd.DataFrame], grid: EliminationGrid) -> pd.DataFrame:
    """The level × column gem matrix of the cross-sweep ranking.

    Args:
        wides: Axis → level × value gem matrix (``pivot``).
        grid: The grid the sweeps come from.

    Returns:
        Column ``BASELINE_COLUMN`` once, then every non-baseline value of each
        axis in ``wides`` order.

    Raises:
        ValueError: If the baseline columns of two sweeps differ.
    """
    axes = list(wides)
    base = wides[axes[0]][grid.baseline[axes[0]]]
    for axis in axes[1:]:
        if not np.array_equal(wides[axis][grid.baseline[axis]].loc[base.index], base):
            raise ValueError(f"the baseline column of {axis!r} differs from that of {axes[0]!r}")
    columns = {BASELINE_COLUMN: base}
    for axis in axes:
        for value in wides[axis].columns:
            if value != grid.baseline[axis]:
                columns[value] = wides[axis][value].loc[base.index]
    return pd.DataFrame(columns)


def rank_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Rank of every level within each column, 1 for the highest gem (ties share the lower rank).

    Args:
        wide: Level × column gem matrix.

    Returns:
        Integer matrix of the same shape as ``wide`` holding the ranks.
    """
    return wide.rank(ascending=False, method="min").astype(int)


def ranked_cells(wide: pd.DataFrame) -> pd.DataFrame:
    """Each column sorted by gem: row ``r`` names the level at rank ``r`` and its gem.

    Args:
        wide: Level × column gem matrix.

    Returns:
        Index ``rank`` (1 … n); per column of ``wide`` a ``<col>`` column with
        the level and a ``<col>_gem`` column with its gem.
    """
    out = {}
    for col in wide.columns:
        ordered = wide[col].sort_values(ascending=False, kind="stable")
        out[col] = ordered.index.to_list()
        out[f"{col}_gem"] = ordered.to_numpy()
    return pd.DataFrame(out, index=pd.RangeIndex(1, len(wide) + 1, name="rank"))


def rank_correlations(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman and Kendall (τ-b) correlation of the gem between every pair of columns.

    Args:
        wide: Level × column gem matrix.

    Returns:
        ``(rho, tau)``: two column × column matrices with 1 on the diagonal.
    """
    cols = list(wide.columns)
    rho = pd.DataFrame(1.0, index=cols, columns=cols)
    tau = pd.DataFrame(1.0, index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        rho.loc[a, b] = rho.loc[b, a] = stats.spearmanr(wide[a], wide[b]).statistic
        tau.loc[a, b] = tau.loc[b, a] = stats.kendalltau(wide[a], wide[b]).statistic
    return rho, tau


def top_k_overlap(ranks: pd.DataFrame, k: int) -> pd.DataFrame:
    """Entry ``[A, B]`` is the number of levels in the top ``k`` of both column A and column B.

    Args:
        ranks: Level × column rank matrix (``rank_table``).
        k: A level is in a column's top ``k`` when its rank is ≤ ``k``.

    Returns:
        Column × column matrix of overlap counts.
    """
    tops = {c: set(ranks.index[ranks[c] <= k]) for c in ranks.columns}
    return pd.DataFrame(
        {b: {a: len(tops[a] & tops[b]) for a in ranks.columns} for b in ranks.columns}
    ).loc[ranks.columns, ranks.columns]


def rank_ranges(ranks: pd.DataFrame, wide: pd.DataFrame, threshold: float = GEM_THRESHOLD) -> pd.DataFrame:
    """Per level: best, worst and median rank, rank range, and columns clearing the threshold.

    Args:
        ranks: Level × column rank matrix (``rank_table``).
        wide: Level × column gem matrix the ranks come from.
        threshold: Gem a column must reach to count towards ``n_above``.

    Returns:
        One row per level, sorted by ``rank_range`` descending, with the column
        of the best and of the worst rank and ``n_above`` (columns with
        gem ≥ ``threshold``).
    """
    out = pd.DataFrame({
        "best_rank": ranks.min(axis=1),
        "best_in": ranks.idxmin(axis=1),
        "worst_rank": ranks.max(axis=1),
        "worst_in": ranks.idxmax(axis=1),
        "median_rank": ranks.median(axis=1),
        "min_gem": wide.min(axis=1),
        "max_gem": wide.max(axis=1),
        "n_above": (wide >= threshold).sum(axis=1),
    })
    out["rank_range"] = out["worst_rank"] - out["best_rank"]
    return out.sort_values(["rank_range", "median_rank"], ascending=[False, True])


def always_top(ranks: pd.DataFrame, k: int) -> list[str]:
    """Levels in the top ``k`` of every column, best median rank first.

    Args:
        ranks: Level × column rank matrix (``rank_table``).
        k: A level qualifies when its rank is ≤ ``k`` in every column.

    Returns:
        The qualifying levels.
    """
    keep = ranks[(ranks <= k).all(axis=1)]
    return keep.median(axis=1).sort_values(kind="stable").index.to_list()


def row_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman ρ between matching rows of two arrays.

    Args:
        x: Array of shape ``B × n``.
        y: Array of shape ``B × n``.

    Returns:
        Array of shape ``B``: ρ between ``x[i]`` and ``y[i]``, ties given
        average ranks.
    """
    rx = stats.rankdata(x, axis=1)
    ry = stats.rankdata(y, axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    return (rx * ry).sum(axis=1) / np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))


def rank_noise_floor(boot_gem: np.ndarray, positions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Spearman ρ between two independent bootstrap draws of the same column.

    Two draws stand in for two replications of one column whose true ranking
    is identical, so their ρ is the agreement fold noise alone allows. A
    cross-column ρ below it is a ranking change larger than that noise.

    Args:
        boot_gem: Bootstrap gems, ``B × n_config``; draws ``i`` and ``i + B/2``
            are paired.
        positions: Column → bootstrap column of each level.

    Returns:
        One row per column: median and 5th percentile of ρ over the ``B/2`` pairs.
    """
    half = boot_gem.shape[0] // 2
    rows = []
    for col, pos in positions.items():
        rho = row_spearman(boot_gem[:half, pos], boot_gem[half : 2 * half, pos])
        rows.append({"column": col, "median_rho": np.median(rho), "p05_rho": np.quantile(rho, 0.05)})
    return pd.DataFrame(rows)


def rank_correlation_intervals(
    boot_gem: np.ndarray,
    positions: dict[str, np.ndarray],
    observed: pd.DataFrame,
    level: float = CI_LEVEL,
) -> pd.DataFrame:
    """Paired bootstrap interval of the Spearman ρ between every pair of columns.

    Args:
        boot_gem: Bootstrap gems, ``B × n_config``.
        positions: Column → bootstrap column of each level, in ``observed`` row order.
        observed: Level × column gem matrix.
        level: Coverage of the percentile interval.

    Returns:
        One row per unordered pair of columns: observed ``rho``, ``lo`` and ``hi``.
    """
    alpha = (1.0 - level) / 2.0
    rows = []
    for a, b in itertools.combinations(observed.columns, 2):
        draws = row_spearman(boot_gem[:, positions[a]], boot_gem[:, positions[b]])
        lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
        rows.append({
            "a": a,
            "b": b,
            "rho": stats.spearmanr(observed[a], observed[b]).statistic,
            "lo": lo,
            "hi": hi,
        })
    return pd.DataFrame(rows)


def axis_bootstrap(
    table: pd.DataFrame, boot_gem: np.ndarray, positions: np.ndarray, level: float = CI_LEVEL
) -> pd.DataFrame:
    """Paired bootstrap of the across-level comparison of two axis values.

    Every level is resampled on the same fold draw, so the correlation of the
    levels' noise through the shared folds is kept, which the sign and Wilcoxon
    tests over levels assume away.

    Args:
        table: Output of ``sweep_table``.
        boot_gem: Bootstrap gems, ``B × n_config``.
        positions: ``config_positions(table, ...)``.
        level: Coverage of the percentile intervals.

    Returns:
        One row per unordered pair ``(a, b)``, over the ``n_levels`` levels with
        both values present: the observed median over levels of
        gem(a) − gem(b) with its interval, the observed number of levels a wins
        with its interval, and ``p_median_positive``, the share of draws in
        which the median difference is above zero.
    """
    alpha = (1.0 - level) / 2.0
    cat_feats = table["cat_feat"].unique()
    col = pd.Series(positions, index=pd.MultiIndex.from_frame(table[["cat_feat", "value"]]))
    obs = table.set_index(["cat_feat", "value"])["gem"]
    rows = []
    for a, b in itertools.combinations(table["value"].unique(), 2):
        both = [c for c in cat_feats if (c, a) in col.index and (c, b) in col.index]
        pa = col.loc[[(c, a) for c in both]].to_numpy()
        pb = col.loc[[(c, b) for c in both]].to_numpy()
        d = boot_gem[:, pa] - boot_gem[:, pb]
        med = np.median(d, axis=1)
        wins = (d > 0).sum(axis=1)
        d_obs = obs.loc[[(c, a) for c in both]].to_numpy() - obs.loc[[(c, b) for c in both]].to_numpy()
        med_lo, med_hi = np.quantile(med, [alpha, 1.0 - alpha])
        wins_lo, wins_hi = np.quantile(wins, [alpha, 1.0 - alpha])
        rows.append({
            "a": a,
            "b": b,
            "n_levels": len(both),
            "median_diff": float(np.median(d_obs)),
            "median_lo": med_lo,
            "median_hi": med_hi,
            "a_wins": int((d_obs > 0).sum()),
            "a_wins_lo": int(wins_lo),
            "a_wins_hi": int(wins_hi),
            "p_median_positive": float((med > 0).mean()),
        })
    return pd.DataFrame(rows)


def partial_rank_agreement(
    table: pd.DataFrame,
    baseline_value: str,
    boot_gem: np.ndarray,
    positions: np.ndarray,
    k: int = 10,
    level: float = CI_LEVEL,
) -> pd.DataFrame:
    """Ranking agreement of each value's column with the baseline column, on their common levels.

    For a sweep with absent cells, where a value's column covers only some levels.

    Args:
        table: Output of ``sweep_table``.
        baseline_value: The axis's baseline value.
        boot_gem: Bootstrap gems, ``B × n_config``; draws ``i`` and ``i + B/2``
            are paired for the noise floor.
        positions: ``config_positions(table, ...)``.
        k: Size of the top set compared.
        level: Coverage of the percentile interval.

    Returns:
        One row per non-baseline value: ``n_levels`` in common, the Spearman
        ``rho`` and Kendall ``tau`` with the baseline column, the paired
        bootstrap interval ``rho_lo``/``rho_hi``, the median and 5th-percentile
        noise floor of both columns on those levels (``floor_*``), the top-``k``
        overlap when both columns are ranked on the common levels, and
        ``n_above`` (common levels with gem ≥ ``GEM_THRESHOLD``) for both.
    """
    alpha = (1.0 - level) / 2.0
    wide = pivot(table)
    col = pd.Series(positions, index=pd.MultiIndex.from_frame(table[["cat_feat", "value"]]))
    half = boot_gem.shape[0] // 2
    rows = []
    for value in wide.columns:
        if value == baseline_value:
            continue
        common = wide[[baseline_value, value]].dropna()
        pb = col.loc[[(c, baseline_value) for c in common.index]].to_numpy()
        pv = col.loc[[(c, value) for c in common.index]].to_numpy()
        draws = row_spearman(boot_gem[:, pb], boot_gem[:, pv])
        floor_b = row_spearman(boot_gem[:half, pb], boot_gem[half : 2 * half, pb])
        floor_v = row_spearman(boot_gem[:half, pv], boot_gem[half : 2 * half, pv])
        ranks = common.rank(ascending=False, method="min")
        top_b = set(ranks.index[ranks[baseline_value] <= k])
        top_v = set(ranks.index[ranks[value] <= k])
        rows.append({
            "value": value,
            "n_levels": len(common),
            "rho": stats.spearmanr(common[baseline_value], common[value]).statistic,
            "tau": stats.kendalltau(common[baseline_value], common[value]).statistic,
            "rho_lo": np.quantile(draws, alpha),
            "rho_hi": np.quantile(draws, 1.0 - alpha),
            "floor_baseline_median": np.median(floor_b),
            "floor_baseline_p05": np.quantile(floor_b, 0.05),
            "floor_value_median": np.median(floor_v),
            "floor_value_p05": np.quantile(floor_v, 0.05),
            f"top{k}_overlap": len(top_b & top_v),
            "n_above_baseline": int((common[baseline_value] >= GEM_THRESHOLD).sum()),
            "n_above_value": int((common[value] >= GEM_THRESHOLD).sum()),
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    """Render a frame as a GitHub Markdown table, floats to ``digits`` decimals with a sign."""
    frame = df.reset_index() if index else df

    def fmt(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:+.{digits}f}"
        return str(v)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def top_k_covering(
    scores: pd.DataFrame,
    n_levels: int = TOP_K_LEVELS,
    score: str = "crpss_pooled_leave_one_out",
    off_baseline: dict[str, str] | None = None,
) -> pd.DataFrame:
    """The top-K configurations by one score, K set by what they must cover.

    K is the smallest count such that the K best configurations span
    ``n_levels`` distinct catalyst levels and, for every ``axis: value`` of
    ``off_baseline``, include at least one configuration whose ``axis`` is not
    ``value``.

    Args:
        scores: ``gem_scores.parquet``, carrying ``cat_feat``, ``score`` and
            the axes named in ``off_baseline``.
        n_levels: Distinct catalyst levels the top K must span.
        score: The column ranked on, higher is better, e.g. ``gem``.
        off_baseline: Axis → value the top K must depart from at least once,
            e.g. ``{"kernel_structure": "KS1"}``.

    Returns:
        The K best rows, best first, with ``rank`` (1 … K) and ``new_level``
        (whether the row is the first of its catalyst level in the ranking).

    Raises:
        ValueError: If ``scores`` holds fewer than ``n_levels`` catalyst levels,
            or no configuration departs from a required value.
    """
    ranked = scores.sort_values(score, ascending=False, kind="stable", ignore_index=True)
    first = ~ranked["cat_feat"].duplicated()
    covered = first.cumsum().to_numpy()
    if covered[-1] < n_levels:
        raise ValueError(f"only {covered[-1]} catalyst levels, fewer than {n_levels}")
    k = int(np.argmax(covered >= n_levels)) + 1
    for axis, value in (off_baseline or {}).items():
        off = (ranked[axis] != value).to_numpy()
        if not off.any():
            raise ValueError(f"no configuration has {axis} other than {value!r}")
        k = max(k, int(np.argmax(off)) + 1)
    return ranked.head(k).assign(rank=np.arange(1, k + 1), new_level=first.head(k).to_numpy())


def versus_first(
    boot_study: np.ndarray, positions: np.ndarray, level: float = CI_LEVEL
) -> pd.DataFrame:
    """Paired bootstrap of each configuration's score against the first one's.

    Args:
        boot_study: Bootstrap scores, ``B × n_config``.
        positions: Bootstrap column of each ranked configuration, first = the reference.
        level: Coverage of the percentile interval.

    Returns:
        One row per configuration: bootstrap ``sd`` of its score, interval
        ``diff_lo``/``diff_hi`` of its score minus the first's, and
        ``p_above_first``, the share of draws in which it scores higher.
    """
    alpha = (1.0 - level) / 2.0
    d = boot_study[:, positions] - boot_study[:, positions[:1]]
    return pd.DataFrame({
        "sd": boot_study[:, positions].std(axis=0, ddof=1),
        "diff_lo": np.quantile(d, alpha, axis=0),
        "diff_hi": np.quantile(d, 1.0 - alpha, axis=0),
        "p_above_first": (d > 0).mean(axis=0),
    })


def sweep_ranges(tables: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """Per catalyst level, the minimum, maximum and count of one score over each sweep's values.

    Args:
        tables: Axis → ``sweep_table`` output (complete or not).
        column: The score, e.g. ``crpss_pooled_leave_one_out``.

    Returns:
        Index ``cat_feat``; per axis ``<axis>_min``, ``<axis>_max`` and
        ``<axis>_n`` (values present for the level, the baseline included).
    """
    parts = []
    for axis, table in tables.items():
        stats_ = table.groupby("cat_feat")[column].agg(["min", "max", "count"])
        stats_.columns = [f"{axis}_min", f"{axis}_max", f"{axis}_n"]
        parts.append(stats_)
    return pd.concat(parts, axis=1)


def pooled_mae(aggregate: pd.DataFrame, keys: list[str], study: str) -> pd.Series:
    """Mean absolute error over every test point of a study's ok folds, per configuration.

    Args:
        aggregate: Fold rows carrying ``mae`` (per-fold mean) and ``n_test``.
        keys: Configuration key columns.
        study: The study.

    Returns:
        ``Σ mae_f · n_test_f / Σ n_test_f``, indexed by ``keys``.
    """
    rows = aggregate[(aggregate["split_type"] == study) & (fold_status(aggregate) == "ok")]
    weighted = rows.assign(w=rows["mae"] * rows["n_test"])
    sums = weighted.groupby(keys)[["w", "n_test"]].sum()
    return (sums["w"] / sums["n_test"]).rename(f"mae_pooled_{study}")


def _load_aggregate(path: Path, keys: list[str]) -> pd.DataFrame:
    """The gem-study fold rows of the aggregate, with only the columns this script reads."""
    columns = keys + [
        "split_type", "repeat", "fold", "status", "crps_per_point", "baseline_crps_per_point",
        "mae", "n_test",
    ]
    return pd.read_parquet(path, columns=columns, filters=[("split_type", "in", list(GEM_STUDIES))])


def main() -> None:
    """Run the axis-sensitivity analysis and write its tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--axes", nargs="+", default=list(AXES))
    parser.add_argument(
        "--allow-incomplete", nargs="*", default=list(ALLOW_INCOMPLETE),
        help="Sweeps analysed on their existing cells when some are absent.",
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--bayesian", action="store_true",
        help="Bayesian bootstrap: Dirichlet(1, ..., 1) fold weights instead of multinomial counts.",
    )
    args = parser.parse_args()

    grid = config.ELIMINATION_GRID
    keys = ["cat_feat", *grid.baseline]
    scores = pd.read_parquet(args.scores)
    tables = {
        axis: sweep_table(scores, grid, axis, require_complete=axis not in args.allow_incomplete)
        for axis in args.axes
    }
    missing = {axis: missing_cells(table, grid, axis) for axis, table in tables.items()}
    configs = pd.concat([t[keys] for t in tables.values()]).drop_duplicates(ignore_index=True)
    print(
        f"{len(scores)} configurations in {args.scores.name}; "
        + ", ".join(f"{a}: {len(t)} rows" for a, t in tables.items())
        + f"; {len(configs)} distinct configurations; absent cells: "
        + ", ".join(f"{a} {len(m)}" for a, m in missing.items())
    )

    aggregate = _load_aggregate(args.aggregate, keys)
    sums = fold_sums(aggregate, configs)
    observed = observed_gems(sums)
    stored = configs.merge(scores[keys + ["gem"]], on=keys, how="left")["gem"].to_numpy()
    if not np.allclose(observed["gem"][0], stored, rtol=0.0, atol=1e-10):
        raise ValueError("gem recomputed from the aggregate differs from gem_scores.parquet")
    boot = bootstrap_gems(sums, args.n_boot, args.seed, bayesian=args.bayesian)
    kind = "Bayesian (Dirichlet)" if args.bayesian else "multinomial"
    print(f"{args.n_boot} paired {kind} fold-bootstrap draws over {len(configs)} configurations")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# Axis sensitivity of the gem",
        "",
        f"Scores: `{args.scores}` ({len(scores)} rows). Aggregate: `{args.aggregate}` "
        f"({len(aggregate)} gem-study fold rows read). Bootstrap: {args.n_boot} {kind} draws, seed {args.seed}.",
    ]

    sd = bootstrap_sd(boot, sums.configs)
    sd = sd.assign(gem=observed["gem"][0])
    sd.to_csv(out / "bootstrap_sd.csv", index=False)

    wides = {}
    for axis, table in tables.items():
        base = grid.baseline[axis]
        wide = pivot(table)
        wides[axis] = wide
        positions = config_positions(table, sums.configs)
        diff = baseline_differences(wide, base)
        diff_summary = summarise_differences(diff)
        winners = winner_counts(wide).rename("wins").to_frame()
        wins = pairwise_win_matrix(wide)
        tests = paired_tests(wide)
        binding = binding_summary(table, base)
        pairs = paired_bootstrap(table, boot["gem"], positions)
        pair_summary = paired_bootstrap_summary(pairs)
        spread = level_spread(wide, pairs)
        across = axis_bootstrap(table, boot["gem"], positions)
        by_group = group_summary(wide, base)
        groups = table["cat_feat"].map(level_group)
        across_by_group = pd.concat(
            [
                axis_bootstrap(table[groups == g], boot["gem"], positions[(groups == g).to_numpy()])
                .assign(group=g)
                for g in ("full", "pca")
            ],
            ignore_index=True,
        )
        noise = (
            table[["cat_feat", "value"]]
            .assign(sd_gem=boot["gem"][:, positions].std(axis=0, ddof=1))
            .groupby("value", sort=False)["sd_gem"]
            .describe()[["min", "50%", "max"]]
            .rename(columns={"50%": "median"})
        )

        long = table[["cat_feat", "value", *SWEEP_COLUMNS]].assign(
            level=table["cat_feat"].map(level_label),
            sd_gem=boot["gem"][:, positions].std(axis=0, ddof=1),
        )
        long.to_csv(out / f"{axis}_gems.csv", index=False)
        diff.to_csv(out / f"{axis}_differences.csv")
        diff_summary.to_csv(out / f"{axis}_difference_summary.csv", index=False)
        winners.to_csv(out / f"{axis}_winners.csv")
        wins.to_csv(out / f"{axis}_win_matrix.csv")
        tests.to_csv(out / f"{axis}_paired_tests.csv", index=False)
        binding.to_csv(out / f"{axis}_binding.csv", index=False)
        pairs.to_csv(out / f"{axis}_bootstrap_pairs.csv", index=False)
        pair_summary.to_csv(out / f"{axis}_bootstrap_pair_summary.csv", index=False)
        spread.to_csv(out / f"{axis}_level_spread.csv", index=False)
        across.to_csv(out / f"{axis}_axis_bootstrap.csv", index=False)
        by_group.to_csv(out / f"{axis}_by_group.csv", index=False)
        across_by_group.to_csv(out / f"{axis}_axis_bootstrap_by_group.csv", index=False)
        noise.to_csv(out / f"{axis}_noise.csv")
        if len(missing[axis]):
            missing[axis].assign(level=missing[axis]["cat_feat"].map(level_label)).to_csv(
                out / f"{axis}_missing_cells.csv", index=False
            )
            agreement = partial_rank_agreement(table, base, boot["gem"], positions)
            agreement.to_csv(out / f"{axis}_rank_agreement.csv", index=False)
            md += [
                "", f"## {axis}: {len(missing[axis])} absent cells", "",
                markdown_table(missing[axis], index=False),
                "", "### Ranking agreement with the baseline column on common levels", "",
                markdown_table(agreement, index=False),
            ]

        md += [
            "", f"## {axis} (baseline `{base}`)",
            "", "### Differences against the baseline value", "", markdown_table(diff_summary, index=False),
            "", "### Winner counts", "", markdown_table(winners),
            "", "### Pairwise wins (row beats column on n levels)", "", markdown_table(wins),
            "", "### Paired tests over the levels", "", markdown_table(tests, index=False),
            "", "### Paired fold bootstrap of the across-level comparison", "",
            markdown_table(across, index=False),
            "", "### Binding study", "", markdown_table(binding, index=False),
            "", "### By level group", "", markdown_table(by_group, index=False),
            "", "### Paired fold bootstrap of the across-level comparison, per level group", "",
            markdown_table(across_by_group, index=False),
            "", "### Bootstrap SD of the gem per configuration", "", markdown_table(noise),
            "", f"### Paired bootstrap ({CI_LEVEL:.0%} interval of the gem difference)", "",
            markdown_table(pair_summary, index=False),
            "", "### Per-level spread across values", "",
            markdown_table(spread.sort_values("range", ascending=False), index=False),
        ]
        print(f"{axis}: wins {winners['wins'].to_dict()}")

    ranked_axes = [axis for axis in tables if not len(missing[axis])]
    ranking = ranking_matrix({axis: wides[axis] for axis in ranked_axes}, grid)
    ranks = rank_table(ranking)
    cells = ranked_cells(ranking)
    rho, tau = rank_correlations(ranking)
    overlaps = {k: top_k_overlap(ranks, k) for k in TOP_K}
    ranges = rank_ranges(ranks, ranking)
    top10 = always_top(ranks, 10)
    first = tables[ranked_axes[0]]
    table_by_column = {BASELINE_COLUMN: first[first["value"] == grid.baseline[ranked_axes[0]]]}
    for axis in ranked_axes:
        table = tables[axis]
        for value in wides[axis].columns:
            if value != grid.baseline[axis]:
                table_by_column[value] = table[table["value"] == value]
    positions = {
        col: config_positions(t.set_index("cat_feat").loc[ranking.index].reset_index(), sums.configs)
        for col, t in table_by_column.items()
    }
    floor = rank_noise_floor(boot["gem"], positions)
    rho_ci = rank_correlation_intervals(boot["gem"], positions, ranking)

    ranking.to_csv(out / "ranking_gems.csv")
    ranks.to_csv(out / "ranking_ranks.csv")
    cells.to_csv(out / "ranking_table.csv")
    rho.to_csv(out / "ranking_spearman.csv")
    tau.to_csv(out / "ranking_kendall.csv")
    for k, o in overlaps.items():
        o.to_csv(out / f"ranking_top{k}_overlap.csv")
    ranges.to_csv(out / "ranking_rank_ranges.csv")
    floor.to_csv(out / "ranking_noise_floor.csv", index=False)
    rho_ci.to_csv(out / "ranking_spearman_intervals.csv", index=False)

    display = cells.copy()
    for col in ranking.columns:
        mark = np.where(cells[f"{col}_gem"] >= GEM_THRESHOLD, " ✓", "")
        display[col] = [
            f"{level_label(c)} · {g:+.3f}{m}" for c, g, m in zip(cells[col], cells[f"{col}_gem"], mark)
        ]
    display = display[list(ranking.columns)]
    md += [
        "", "## Ranking across sweeps",
        "", f"`{BASELINE_COLUMN}` is KS1 = P1 = raw, one configuration. ✓ marks gem ≥ {GEM_THRESHOLD}.",
        "", markdown_table(display),
        "", "### Spearman ρ", "", markdown_table(rho),
        "", "### Kendall τ", "", markdown_table(tau),
        *itertools.chain.from_iterable(
            ["", f"### Top-{k} overlap", "", markdown_table(o)] for k, o in overlaps.items()
        ),
        "", "### Rank ranges", "", markdown_table(ranges),
        "", "### In the top 10 of every column", "", ", ".join(top10) or "none",
        "", "### Rank noise floor (Spearman ρ between two independent draws of one column)", "",
        markdown_table(floor, index=False),
        "", f"### Spearman ρ between columns, {CI_LEVEL:.0%} paired bootstrap interval", "",
        markdown_table(rho_ci, index=False),
    ]
    maes = pd.concat([pooled_mae(aggregate, keys, study) for study in GEM_STUDIES], axis=1).reset_index()
    for name, (score, boot_key) in TOP_K_RANKINGS.items():
        top = top_k_covering(
            scores, score=score,
            off_baseline={axis: grid.baseline[axis] for axis in TOP_K_OFF_BASELINE},
        )
        top_pos = config_positions(top, sums.configs)
        top = pd.concat([top.reset_index(drop=True), versus_first(boot[boot_key], top_pos)], axis=1)
        top = top.merge(maes, on=keys, how="left")
        ranges = sweep_ranges(tables, score)
        top = top.merge(ranges, left_on="cat_feat", right_index=True, how="left")
        top_columns = [
            "rank", "cat_feat", "kernel_structure", "prior_scheme", "outcome_transform", "mean_function",
            score, "sd", "diff_lo", "diff_hi", "p_above_first",
            *[c for s_ in GEM_STUDIES for c in (f"mae_median_{s_}", f"mae_pooled_{s_}")],
            *[f"crpss_pooled_{s_}" for s_ in GEM_STUDIES if f"crpss_pooled_{s_}" != score],
            *(["gem"] if score != "gem" else []), "gem_study", "new_level",
            *ranges.columns,
        ]
        top[top_columns].to_csv(out / f"{name}_top_k.csv", index=False)
        md += [
            "", f"## Top {len(top)} configurations by {score} "
            f"(the fewest that span {TOP_K_LEVELS} catalyst levels and hold a non-baseline "
            f"value of {', '.join(TOP_K_OFF_BASELINE)})", "",
            markdown_table(top[top_columns], index=False),
        ]
        print(f"{name} top-K: K = {len(top)} spans {top['cat_feat'].nunique()} levels")

    (out / "axis_sensitivity.md").write_text("\n".join(md) + "\n")
    print(f"Wrote tables -> {out}")


if __name__ == "__main__":
    main()
