"""Gem scores — rank configurations by pooled CRPSS on the two interpolation studies.

A study is one split family; a fold is one fold × repeat of it. Per
(configuration, study) the failed folds are dropped and the rest are pooled:

    CRPSS_pooled = 1 − Σ CRPS_model / Σ CRPS_baseline

with both sums over every test point of every successful fold in the study
(``utils.metrics.pooled_crps_skill_score``). Pooling weighs each fold by its
baseline CRPS mass, so one fold whose model CRPS is far above its baseline pulls
the score down by that much instead of being outvoted by a median.

**gem = min(CRPSS_pooled on ``primary``, CRPSS_pooled on ``leave_one_out``).**
The other four studies — type, scaffold, solvent, replicate — are scored the same
way and reported, but never enter the gem. ``gem_study`` names the study that
binds.

Failed folds mark a configuration and never remove it. Per (configuration,
study) with ``n_folds`` folds:

* ``ineligible`` — ``n_failed > floor(0.05 · n_folds)``;
* ``not_production_ready`` — ``n_failed > 0``.

The configuration-level ``ineligible`` / ``not_production_ready`` columns are the
``any`` of these over the six studies. ``n_failed`` is counted from the fold
``status`` column; every expected fold of every configuration must be present as
either ok or failed, and an absent fold is an error. MSLL, MAE, RMSE, Q² and the
calibration KS statistic are reported as per-study medians over the ok folds.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores
    python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --run cross_1
    python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores \
        --aggregate <scan_aggregate.parquet> --cv-splits <cv_splits.json> --out <gem_scores.parquet>

``--run <name>`` (default ``elimination``) reads the run's aggregate and writes
its ``gem_scores.parquet`` (``paths.run_paths``); ``--aggregate`` and ``--out``
override either.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.utils.metrics import pooled_crps_skill_score

SPLIT_FAMILIES: tuple[str, ...] = (
    "primary",
    "leave_one_type_out",
    "leave_one_scaffold_out",
    "leave_one_solvent_out",
    "leave_one_out",
    "leave_one_replicate_out",
)
"""The six studies, each scored and flagged per configuration."""

GEM_STUDIES: tuple[str, ...] = ("primary", "leave_one_out")
"""The studies the gem is the minimum over."""

CONFIG_COLUMNS: tuple[str, ...] = (
    "cat_feat",
    "solv_feat",
    "base_feat",
    "kernel_structure",
    "base_kernel",
    "model_class",
    "mean_function",
    "outcome_transform",
    "prior_scheme",
)
"""Columns that together identify one configuration. Absent ones are skipped.

Archived aggregates also carry ``cat_source`` and ``dim_reduction``; both are
determined by ``cat_feat``, so grouping without them yields the same
configurations."""

MEDIAN_METRICS: tuple[str, ...] = ("msll", "mae", "rmse", "q2", "calibration_ks")
"""Fold-level metrics reported as a per-study median over ok folds."""

FOLD_KEY_COLUMNS: tuple[str, ...] = ("split_type", "repeat", "fold")

INELIGIBLE_FAILED_SHARE: Fraction = Fraction(1, 20)
"""A study is ineligible once ``n_failed`` exceeds ``floor(share · n_folds)``."""

DEFAULT_CV_SPLITS = paths.CV_SPLITS_JSON


def fold_status(df: pd.DataFrame) -> pd.Series:
    """The ``status`` of every fold row.

    A frame without a ``status`` column is read as all ``"ok"``. A failed fold
    missing from such a frame shows up as an absent fold in
    ``compute_gem_scores``, which is an error.

    Args:
        df: Fold-level aggregate.

    Returns:
        ``"ok"`` or ``"failed"`` per row, aligned with ``df``.

    Raises:
        ValueError: If a status other than ok / failed appears.
    """
    if "status" not in df.columns:
        return pd.Series("ok", index=df.index, name="status")
    unknown = set(df["status"].unique()) - {"ok", "failed"}
    if unknown:
        raise ValueError(f"unknown fold status values: {sorted(unknown)}")
    return df["status"]


def ok_folds(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of successful folds.

    Args:
        df: Fold-level aggregate.

    Returns:
        The rows of ``df`` whose ``fold_status`` is ``"ok"``.

    Raises:
        ValueError: If a status other than ok / failed appears.
    """
    return df[fold_status(df) == "ok"]


def load_expected_folds(cv_splits_path: Path) -> pd.DataFrame:
    """Every fold a complete configuration has, from the run's ``cv_splits.json``.

    Args:
        cv_splits_path: The split definitions the run was fitted on.

    Returns:
        One row per ``(split_type, repeat, fold)``.
    """
    blocks = json.loads(Path(cv_splits_path).read_text())
    return pd.DataFrame(
        [
            {"split_type": block["split_type"], "repeat": block["repeat"], "fold": fold}
            for block in blocks
            for fold in range(len(block["folds"]))
        ],
        columns=list(FOLD_KEY_COLUMNS),
    )


def pooled_crpss(folds: pd.DataFrame) -> float:
    """Pooled CRPSS over the test points of a set of ok folds.

    Args:
        folds: Fold rows carrying ``crps_per_point`` and
            ``baseline_crps_per_point``; every row must be an ok fold.

    Returns:
        ``1 − Σ CRPS_model / Σ CRPS_baseline`` over all their test points.

    Raises:
        ValueError: If a row is not an ok fold, or ``folds`` is empty.
    """
    if (fold_status(folds) != "ok").any():
        raise ValueError("pooled CRPSS is defined over ok folds only; filter first")
    return pooled_crps_skill_score(
        folds["crps_per_point"].map(np.sum).to_numpy(dtype=float),
        folds["baseline_crps_per_point"].map(np.sum).to_numpy(dtype=float),
    )


def _check_every_fold_present(
    work: pd.DataFrame, keys: list[str], expected: pd.DataFrame
) -> None:
    """Raise unless every configuration has each expected fold exactly once."""
    fold_keys = list(FOLD_KEY_COLUMNS)
    duplicated = work.duplicated(keys + fold_keys, keep=False)
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} fold rows are duplicated within a configuration"
        )
    unexpected = work[fold_keys].merge(expected, how="left", indicator=True)["_merge"]
    if (unexpected == "left_only").any():
        raise ValueError(
            f"{int((unexpected == 'left_only').sum())} fold rows are not in the "
            "run's cv_splits; the aggregate and the splits come from different runs"
        )
    n_present = work.groupby(keys, dropna=False).size()
    short = n_present[n_present != len(expected)]
    if len(short):
        raise ValueError(
            f"{len(short)} configurations have folds absent (neither ok nor failed); "
            f"expected {len(expected)} each, e.g. {short.head(5).to_dict()}"
        )


def compute_gem_scores(df: pd.DataFrame, expected_folds: pd.DataFrame) -> pd.DataFrame:
    """Reduce fold-level results to one row per configuration.

    Args:
        df: Fold-level aggregate: one row per fold (ok or failed), carrying
            ``split_type``, ``repeat``, ``fold``, the per-point
            ``crps_per_point`` / ``baseline_crps_per_point``, the
            ``MEDIAN_METRICS`` and whichever ``CONFIG_COLUMNS`` are present.
        expected_folds: Every ``(split_type, repeat, fold)`` of the run, from
            ``load_expected_folds``, restricted to ``SPLIT_FAMILIES``.

    Returns:
        One row per configuration, sorted by ``gem`` descending (NaN last). Per
        study ``s``: ``crpss_pooled_<s>``, ``n_ok_<s>``, ``n_failed_<s>``,
        ``ineligible_<s>``, ``not_production_ready_<s>`` and
        ``<metric>_median_<s>``; then ``gem``, ``gem_study``, and the
        configuration-level ``ineligible`` and ``not_production_ready``. A study
        with no ok fold has a NaN pooled CRPSS, and a NaN gem study gives a NaN
        gem.

    Raises:
        ValueError: If an expected study, fold or configuration fold is absent,
            or a fold appears that the splits do not define.
    """
    expected = expected_folds[expected_folds["split_type"].isin(SPLIT_FAMILIES)]
    missing_studies = set(SPLIT_FAMILIES) - set(expected["split_type"])
    if missing_studies:
        raise ValueError(f"expected folds lack the studies {sorted(missing_studies)}")

    keys = [c for c in CONFIG_COLUMNS if c in df.columns]
    work = df[df["split_type"].isin(SPLIT_FAMILIES)].assign(status=fold_status(df))
    _check_every_fold_present(work, keys, expected)
    n_folds = expected.groupby("split_type").size()

    ok = work[work["status"] == "ok"]
    ok = ok.assign(
        _crps_sum=ok["crps_per_point"].map(np.sum).astype(float),
        _baseline_sum=ok["baseline_crps_per_point"].map(np.sum).astype(float),
    )
    counts = (
        work.groupby(keys + ["split_type", "status"], dropna=False)
        .size()
        .unstack("status", fill_value=0)
        .reindex(columns=["ok", "failed"], fill_value=0)
    )
    pooled = ok.groupby(keys + ["split_type"], dropna=False)[["_crps_sum", "_baseline_sum"]].apply(
        lambda g: pooled_crps_skill_score(g["_crps_sum"].to_numpy(), g["_baseline_sum"].to_numpy())
    )
    medians = ok.groupby(keys + ["split_type"], dropna=False)[list(MEDIAN_METRICS)].median()

    per_study = counts.rename(columns={"ok": "n_ok", "failed": "n_failed"})
    per_study["crpss_pooled"] = pooled
    per_study = per_study.join(medians.add_suffix("_median"))
    study_n_folds = per_study.index.get_level_values("split_type").map(n_folds).to_numpy()
    allowed = np.array([math.floor(INELIGIBLE_FAILED_SHARE * int(n)) for n in study_n_folds])
    per_study["ineligible"] = per_study["n_failed"].to_numpy() > allowed
    per_study["not_production_ready"] = per_study["n_failed"] > 0

    fields = (
        ["crpss_pooled", "n_ok", "n_failed", "ineligible", "not_production_ready"]
        + [f"{m}_median" for m in MEDIAN_METRICS]
    )
    wide = per_study[fields].unstack("split_type")
    out = pd.DataFrame(index=wide.index)
    for study in SPLIT_FAMILIES:
        for field in fields:
            out[f"{field}_{study}"] = wide[(field, study)]

    gem_columns = [f"crpss_pooled_{s}" for s in GEM_STUDIES]
    out["gem"] = out[gem_columns].min(axis=1, skipna=False)
    out["gem_study"] = [
        GEM_STUDIES[int(np.argmin(row))] if np.isfinite(row).all() else None
        for row in out[gem_columns].to_numpy(dtype=float)
    ]
    out["ineligible"] = out[[f"ineligible_{s}" for s in SPLIT_FAMILIES]].any(axis=1)
    out["not_production_ready"] = out[
        [f"not_production_ready_{s}" for s in SPLIT_FAMILIES]
    ].any(axis=1)
    out = out.reset_index().sort_values("gem", ascending=False, na_position="last", ignore_index=True)
    out.attrs["n_folds"] = n_folds.to_dict()
    return out


def main() -> None:
    """Compute gem scores from the fold-level aggregate and write them out."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=sorted(config.RUN_GRIDS), default=config.DEFAULT_RUN,
                        help="Run whose aggregate and gem-score paths to use.")
    parser.add_argument("--aggregate", type=Path, default=None,
                        help="Fold-level aggregate; the run's scan_aggregate.parquet by default.")
    parser.add_argument(
        "--cv-splits", type=Path, default=DEFAULT_CV_SPLITS,
        help="cv_splits.json of the run that produced the aggregate.",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Output parquet; the run's gem_scores.parquet by default.")
    parser.add_argument("--top", type=int, default=25, help="Rows to print.")
    args = parser.parse_args()
    run = paths.run_paths(args.run)
    aggregate = args.aggregate or run.aggregate_parquet
    out_path = args.out or run.gem_scores_parquet

    df = pd.read_parquet(aggregate)
    scores = compute_gem_scores(df, load_expected_folds(args.cv_splits))

    label_columns = [
        c for c in ("cat_feat", "solv_feat", "base_kernel", "prior_scheme",
                    "kernel_structure", "model_class", "mean_function")
        if c in scores.columns and scores[c].nunique(dropna=False) > 1
    ]
    n_failed = scores[[f"n_failed_{s}" for s in SPLIT_FAMILIES]].sum(axis=1).rename("n_failed")
    print(
        f"{len(scores)} configurations; folds per study {scores.attrs['n_folds']}; "
        f"{int(n_failed.sum())} failed folds; {int(scores['ineligible'].sum())} ineligible, "
        f"{int(scores['not_production_ready'].sum())} not production-ready"
    )
    printable = scores.head(args.top)[
        label_columns + ["gem", "gem_study"]
        + [f"crpss_pooled_{s}" for s in SPLIT_FAMILIES]
        + ["ineligible", "not_production_ready"]
    ].join(n_failed)
    printable.columns = [c.removeprefix("crpss_pooled_") for c in printable.columns]
    print(printable.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_path, index=False)
    print(f"Wrote {len(scores)} rows -> {out_path}")


if __name__ == "__main__":
    main()
