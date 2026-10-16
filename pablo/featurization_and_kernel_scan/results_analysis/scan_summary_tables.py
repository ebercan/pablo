"""Build summary tables (rows = kernel structure, cols = catalyst featurization).

Reads ``results/scan_aggregate.parquet`` (output of
``cross_validation.concat_results``) and, for each model class (singletask,
multitask) present in it, emits three tables — one per metric (NLPD, CRPS,
MSE). Each cell shows ``mean +/- std`` of the per-fold metric pooled across
all folds, all repeats, all solvent/base/outcome-transform sub-configurations
of that (kernel_structure, catalyst_featurization) pair.

Outputs (under ``--out-dir``, default ``results/``), per model class:
  - Three CSVs per metric: ``summary_<model_class>_<metric>_mean.csv``,
    ``summary_<model_class>_<metric>_std.csv`` and
    ``summary_<model_class>_<metric>_n.csv`` (folds per cell).
  - One Markdown file ``summary_<model_class>_tables.md`` with all three tables.

Per-fold metric definitions:
  - NLPD: ``nlpd_mean`` column (mean of ``nlpd_per_point`` for that fold).
  - CRPS: ``crps_mean`` column.
  - MSE : ``rmse ** 2`` (rmse is on yield-fraction scale).

Failed folds are excluded. By default only the ``primary`` split type (10x
repeated 6-fold catalyst-grouped CV) is used so cells average like-for-like
across the same fold structure. Pass ``--split-type all`` to pool every split
type instead.

Run from the repo root:
    python -m pablo.featurization_and_kernel_scan.results_analysis.scan_summary_tables
        [--agg-path PATH] [--out-dir DIR] [--split-type SPLIT]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import ok_folds

ROOT = Path(__file__).parent.parent
DEFAULT_AGG_PATH = ROOT / "results" / "scan_aggregate.parquet"
DEFAULT_OUT_DIR = ROOT / "results"

CATALYST_ORDER = [
    "map4", "ecfp4", "clamp", "mordred", "mol2vec", "chembert2",
    "molformer_xl_precat", "molformer_xl_carbene", "unimol", "grover",
    "interpretable", "soap",
]
KS_ORDER = ["KS1", "KS2", "KS3", "KS4"]
MODEL_CLASSES = ["singletask", "multitask"]

METRIC_SPECS = [
    ("NLPD", "nlpd_mean", "{:.3f}"),
    ("CRPS", "crps_mean", "{:.4f}"),
    ("MSE",  "mse",       "{:.4f}"),
]


def _load(agg_path: Path, split_type: str) -> pd.DataFrame:
    df = ok_folds(pd.read_parquet(agg_path))
    if split_type != "all":
        df = df[df["split_type"] == split_type].copy()
        assert len(df) > 0, f"No rows with split_type=={split_type!r} in {agg_path}"
    df["mse"] = df["rmse"] ** 2
    return df


def _pivot_mean_std(df: pd.DataFrame, metric_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot one metric into mean, std and count tables.

    Args:
        df: Per-fold rows with ``kernel_structure``, ``cat_feat`` and
            ``metric_col`` columns.
        metric_col: Name of the metric column to summarize.

    Returns:
        Tuple ``(mean, std, n)`` of DataFrames with rows = kernel structure
        (``KS_ORDER``) and columns = catalyst featurization (``CATALYST_ORDER``),
        restricted to the levels present in ``df``. ``std`` uses ddof=1; ``n``
        is the number of folds per cell.
    """
    grouped = df.groupby(["kernel_structure", "cat_feat"])[metric_col]
    mean = grouped.mean().unstack("cat_feat")
    std = grouped.std(ddof=1).unstack("cat_feat")
    n = grouped.size().unstack("cat_feat")

    cats = [c for c in CATALYST_ORDER if c in mean.columns]
    rows = [k for k in KS_ORDER if k in mean.index]
    return mean.loc[rows, cats], std.loc[rows, cats], n.loc[rows, cats]


def _md_table(mean: pd.DataFrame, std: pd.DataFrame, fmt: str) -> str:
    header = "| KS \\ cat | " + " | ".join(mean.columns) + " |"
    sep = "|" + "|".join(["---"] * (len(mean.columns) + 1)) + "|"
    lines = [header, sep]
    for ks in mean.index:
        cells = []
        for c in mean.columns:
            m = mean.loc[ks, c]
            s = std.loc[ks, c]
            if np.isnan(m):
                cells.append("—")
            else:
                cells.append(f"{fmt.format(m)} ± {fmt.format(s)}")
        lines.append(f"| **{ks}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tables(agg_path: Path, out_dir: Path, split_type: str) -> None:
    """Write summary CSVs and a Markdown file for each model class.

    Args:
        agg_path: Aggregate parquet from ``cross_validation.concat_results``.
        out_dir: Directory for the outputs; created if missing.
        split_type: Split type to keep, or ``"all"`` to pool every split type.

    Raises:
        AssertionError: If no row has the requested ``split_type``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = _load(agg_path, split_type)
    print(f"[summary] loaded {len(df_all)} rows from {agg_path} (split_type={split_type})")

    for model_class in MODEL_CLASSES:
        df = df_all[df_all.get("model_class", "singletask") == model_class]
        if df.empty:
            print(f"[summary] no rows for model_class={model_class}; skipping.")
            continue

        print(f"[summary] {model_class}: {len(df)} rows; kernels={sorted(df['kernel_structure'].unique())} "
              f"catalysts={sorted(df['cat_feat'].unique())}")

        md_chunks = [
            f"# Featurization x kernel-structure summary tables ({model_class})\n",
            f"Source: `{agg_path.relative_to(ROOT.parent)}`  ",
            f"Split type: `{split_type}`  ",
            f"Model class: `{model_class}`  ",
            f"Per-cell value: mean ± std of the per-fold metric, pooled across all folds, "
            f"all repeats, and all solvent/base/outcome-transform sub-configurations of "
            f"the (kernel_structure, catalyst_featurization) pair.\n",
        ]

        for label, col, fmt in METRIC_SPECS:
            mean, std, n = _pivot_mean_std(df, col)
            mean.to_csv(out_dir / f"summary_{model_class}_{label}_mean.csv")
            std.to_csv(out_dir / f"summary_{model_class}_{label}_std.csv")
            n.to_csv(out_dir / f"summary_{model_class}_{label}_n.csv")
            md_chunks.append(f"\n## {label}\n")
            sample = int(n.iloc[0, 0]) if not n.empty else 0
            md_chunks.append(f"Sample size per cell: {sample} folds (representative — see "
                             f"`summary_{model_class}_{label}_n.csv` for the full grid).\n")
            md_chunks.append(_md_table(mean, std, fmt))
            md_chunks.append("")

        md_path = out_dir / f"summary_{model_class}_tables.md"
        md_path.write_text("\n".join(md_chunks))
        print(f"[summary] wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agg-path", type=Path, default=DEFAULT_AGG_PATH,
                        help="Aggregate parquet from cross_validation.concat_results.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where to write CSVs and the Markdown summary.")
    parser.add_argument("--split-type", default="primary",
                        choices=["primary", "leave_one_type_out", "leave_one_scaffold_out",
                                 "leave_one_solvent_out", "leave_one_out",
                                 "leave_one_replicate_out", "all"],
                        help="Which split type to summarize. Default: primary "
                             "(the 10x6 stratified catalyst-grouped CV).")
    args = parser.parse_args()
    build_tables(args.agg_path, args.out_dir, args.split_type)


if __name__ == "__main__":
    main()
