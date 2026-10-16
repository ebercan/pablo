"""Render heatmap PNGs for the featurization × kernel scan CV grid results.

For every case study (CV split type), metric and peripheral slice
(solv_feat × base_feat × outcome_transform) one heatmap is written:
  rows = 27 inner configs (KS × model × mean; Vanilla has the constant mean
         only), grouped by kernel structure,
  cols = the catalyst featurizations of ``CATALYST_ORDER`` present in the
         aggregate (up to 14).
Within a fold, any remaining axis (form_onehot, base_kernel) is averaged
before the cell is reduced. Files go to
``results/analysis_pngs/<split_type>_<metric>/<solv>_<base>_<transform>.png``.

Metrics: CRPSS (higher ↑ better), MSLL (lower ↓ better), MAE (lower ↓ better).

A CRPSS cell is the pooled CRPSS over every test point of that cell's ok folds
(``gem_scores.pooled_crpss``). An MSLL or MAE cell is the **median** of the
per-fold metric across the cell's ok folds, not the arithmetic mean: MSLL is
unbounded above, so a handful of catastrophic folds (e.g. an overconfident logit
fold with NLPD in the millions) destroys a mean but leaves the median intact.
Failed folds are excluded explicitly. The per-cell fold count is annotated in the
panel title, and low-fold splits (type/scaffold/replicate/solvent) are flagged as
low-confidence.

Colormap: yellow (best) → deep orange (worst), clipped to the panel's 2nd–98th
percentile.

Usage:
    python -m pablo.featurization_and_kernel_scan.results_analysis.plot_heatmap_tables
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import (
    ok_folds,
    pooled_crpss,
)
from pablo.utils.outcome_transforms import OUTCOME_TRANSFORM_NAMES


_ROOT = Path(__file__).parent.parent
AGG_PATH = _ROOT / "results" / "scan_aggregate.parquet"
OUT_DIR = _ROOT / "results" / "analysis_pngs"

CATALYST_ORDER = [
    "map4", "ecfp4", "clamp", "mordred", "mol2vec", "chembert2",
    "molformer_xl_precat", "molformer_xl_carbene", "unimol", "grover",
    "interpretable_steric", "interpretable_light", "interpretable", "soap",
]

KS_ORDER = ["KS1", "KS2", "KS3", "KS4", "Vanilla"]
MODEL_ORDER = ["st_y", "lcm_ys", "lcm_cs"]
MEAN_ORDER = ["constant", "type_linear"]
FORM_ORDER = [False, True]  # used only in SET2-style breakdowns, not SET1 rows

SOLV_ORDER = ["physical", "onehot"]
BASE_ORDER = ["onehot", "structured"]
TRANSFORM_ORDER = list(OUTCOME_TRANSFORM_NAMES)

METRIC_SPECS = [
    ("CRPSS", "crpss", True),
    ("MSLL", "msll", False),
    ("MAE", "mae", False),
]

CASE_STUDIES = [
    ("primary", "Primary CV"),
    ("leave_one_type_out", "Leave-one-type-out"),
    ("leave_one_scaffold_out", "Leave-one-scaffold-out"),
    ("leave_one_solvent_out", "Leave-one-solvent-out"),
    ("leave_one_out", "Leave-one-out"),
    ("leave_one_replicate_out", "Leave-one-replicate-out"),
]

SET1_ROW_KEYS = ["kernel_structure", "model_class", "mean_function"]
SET2_ROW_KEYS = ["solv_feat", "base_feat", "outcome_transform"]

LOW_N_SPLITS = {
    "leave_one_type_out", "leave_one_scaffold_out",
    "leave_one_replicate_out", "leave_one_solvent_out",
}
"""Split types whose per-cell fold count is small enough that even a median is
noisy (type=2, scaffold=3, replicate=5, solvent=7 folds/cell at the current 69
experiments; see ``core.config.FOLD_TOTAL``). Panels for these
splits are annotated with a low-confidence caution."""

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ylorng", ["#FFE000", "#FFA500", "#E05000"]
)


def _build_set1_rows() -> list[tuple]:
    rows = []
    for ks in KS_ORDER:
        means = ["constant"] if ks == "Vanilla" else MEAN_ORDER
        for model in MODEL_ORDER:
            for mean in means:
                rows.append((ks, model, mean))
    return rows


def _set1_label(ks: str, model: str, mean: str) -> str:
    ks_s = {"Vanilla": "Van"}.get(ks, ks)
    model_s = {"st_y": "st", "lcm_ys": "ys", "lcm_cs": "cs"}.get(model, model)
    mean_s = {"constant": "c", "type_linear": "tl"}.get(mean, mean)
    return f"{ks_s}/{model_s}/{mean_s}"



def _set2_label(solv: str, base: str, transform: str) -> str:
    solv_s = {"physical": "phy", "onehot": "one"}.get(solv, solv)
    base_s = {"structured": "struct", "onehot": "one"}.get(base, base)
    return f"{solv_s} / {base_s} / {transform}"


def _aggregate(
    df: pd.DataFrame,
    row_keys: list[str],
    metric_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate one metric column over the ok folds of each (row × cat_feat) cell.

    CRPSS is pooled over every test point of the cell's folds. Any other metric
    is first collapsed to one value per fold (mean over any axis not in
    ``row_keys``/``cat_feat``/fold-id, e.g. ``base_kernel``), then reduced across
    folds by the **median** — robust to the heavy upper tail of MSLL.

    Args:
        df: Fold-level rows for a single panel slice.
        row_keys: Config columns forming the table rows.
        metric_col: Metric column to aggregate (``crpss`` / ``msll`` / ``mae``).

    Returns:
        Tuple of ``(cell_value, cell_var, cell_n)`` Series, each indexed by
        ``row_keys + ["cat_feat"]``. ``cell_var`` is the sample variance of the
        per-fold metric (spread); ``cell_n`` is the fold count behind each cell.
    """
    df = ok_folds(df)
    fold_keys = ["split_type", "repeat", "fold"]
    group_keys = row_keys + ["cat_feat"] + fold_keys
    per_fold = df.groupby(group_keys, sort=False)[metric_col].mean()
    cell_keys = row_keys + ["cat_feat"]
    if metric_col == "crpss":
        cell_median = df.groupby(cell_keys)[
            ["crps_per_point", "baseline_crps_per_point"]
        ].apply(pooled_crpss)
    else:
        cell_median = per_fold.groupby(level=cell_keys).median()
    cell_var = per_fold.groupby(level=cell_keys).var(ddof=1)
    cell_n = per_fold.groupby(level=cell_keys).size()
    return cell_median, cell_var, cell_n


def _pivot_to_table(
    mean_s: pd.Series,
    var_s: pd.Series,
    row_tuples: list[tuple],
    row_labels: list[str],
    cat_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(row_tuples)
    n_cols = len(cat_order)
    mean_arr = np.full((n_rows, n_cols), np.nan)
    var_arr = np.full((n_rows, n_cols), np.nan)
    for i, row_key in enumerate(row_tuples):
        for j, cat in enumerate(cat_order):
            key = row_key + (cat,)
            if key in mean_s.index:
                mean_arr[i, j] = mean_s[key]
            if key in var_s.index:
                var_arr[i, j] = var_s[key]
    mean_df = pd.DataFrame(mean_arr, index=row_labels, columns=cat_order)
    var_df = pd.DataFrame(var_arr, index=row_labels, columns=cat_order)
    return mean_df, var_df


def _ks_block_boundaries(set1_rows: list[tuple]) -> list[int]:
    boundaries = []
    prev_ks = None
    for i, (ks, *_) in enumerate(set1_rows):
        if prev_ks is not None and ks != prev_ks:
            boundaries.append(i)
        prev_ks = ks
    return boundaries


def _render_heatmap(
    mean_df: pd.DataFrame,
    var_df: pd.DataFrame,
    metric_label: str,
    higher_is_better: bool,
    title: str,
    out_path: Path,
    compact: bool = False,
    block_boundaries: list[int] | None = None,
) -> None:
    n_rows, n_cols = mean_df.shape
    if n_rows == 0 or n_cols == 0:
        return

    if compact:
        cell_w, cell_h = 0.9, 0.38
        label_margin = 2.2
        top_margin = 1.2
        mean_fontsize = 5.5
        row_fontsize = 5.5
        col_fontsize = 6.5
        show_var = False
    else:
        cell_w, cell_h = 1.4, 0.9
        label_margin = 2.8
        top_margin = 1.4
        mean_fontsize = 9.5
        row_fontsize = 8.0
        col_fontsize = 7.5
        show_var = True

    fig_w = cell_w * n_cols + label_margin + 1.5
    fig_h = cell_h * n_rows + top_margin + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    mean_vals = mean_df.values.astype(float)
    var_vals = var_df.values.astype(float)
    flat = mean_vals[~np.isnan(mean_vals)]
    if flat.size == 0:
        plt.close(fig)
        return

    clip_lo = float(np.percentile(flat, 2))
    clip_hi = float(np.percentile(flat, 98))
    if clip_hi - clip_lo < 1e-12:
        clip_lo = float(np.min(flat))
        clip_hi = float(np.max(flat))
    if clip_hi - clip_lo < 1e-12:
        clip_hi = clip_lo + 1e-12
    span = clip_hi - clip_lo

    best_flat_idx = np.nanargmax(mean_vals) if higher_is_better else np.nanargmin(mean_vals)
    best_idx = np.unravel_index(best_flat_idx, mean_vals.shape)

    for r in range(n_rows):
        for c in range(n_cols):
            m = mean_vals[r, c]
            norm_val = (m - clip_lo) / span if not np.isnan(m) else 0.5
            norm_val = float(np.clip(norm_val, 0.0, 1.0))
            if higher_is_better:
                norm_val = 1.0 - norm_val
            bg = CMAP(norm_val) if not np.isnan(m) else (0.95, 0.95, 0.95, 1.0)

            is_best = (r, c) == best_idx
            rect = plt.Rectangle(
                (c, n_rows - r - 1), 1, 1,
                facecolor=bg,
                edgecolor="black" if is_best else "white",
                linewidth=2.5 if is_best else 0.5,
                zorder=3 if is_best else 1,
            )
            ax.add_patch(rect)

            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            txt_color = "white" if lum < 0.45 else "black"
            cx, cy = c + 0.5, n_rows - r - 0.5

            if compact:
                m_str = f"{m:.2f}" if not np.isnan(m) else "—"
                ax.text(cx, cy, m_str, ha="center", va="center",
                        fontsize=mean_fontsize, fontweight="bold", color=txt_color)
            else:
                m_str = f"μ = {m:.4f}" if not np.isnan(m) else "μ = —"
                ax.text(cx, cy + 0.15, m_str, ha="center", va="center",
                        fontsize=mean_fontsize, fontweight="bold", color=txt_color)
                if show_var:
                    v = var_vals[r, c]
                    v_str = f"σ² = {v:.4f}" if not np.isnan(v) else "σ² = —"
                    ax.text(cx, cy - 0.2, v_str, ha="center", va="center",
                            fontsize=mean_fontsize * 0.85, color=txt_color, alpha=0.88)

    if block_boundaries:
        for b in block_boundaries:
            y = n_rows - b
            ax.plot([0, n_cols], [y, y], color="#333333", linewidth=1.5, zorder=5)

    for c, cat in enumerate(mean_df.columns):
        ax.text(c + 0.5, n_rows + 0.08, cat, ha="center", va="bottom",
                fontsize=col_fontsize, fontweight="bold", rotation=35)
    for r, label in enumerate(mean_df.index):
        ax.text(-0.08, n_rows - r - 0.5, label, ha="right", va="center",
                fontsize=row_fontsize)

    direction = "↑ better" if higher_is_better else "↓ better"
    fig.suptitle(f"{title}  ({direction})", fontsize=10, fontweight="bold", y=0.99)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin=clip_lo, vmax=clip_hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_ticks([clip_lo, clip_hi])
    if higher_is_better:
        cbar.set_ticklabels([f"{clip_lo:.3f} (worst)", f"{clip_hi:.3f} (best)"])
    else:
        cbar.set_ticklabels([f"{clip_lo:.3f} (best)", f"{clip_hi:.3f} (worst)"])
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] wrote {out_path}  ({n_rows}×{n_cols})")


def main() -> None:
    df = pd.read_parquet(AGG_PATH)
    print(f"[heatmap] loaded {len(df)} rows from {AGG_PATH.name}")

    cats = [c for c in CATALYST_ORDER if c in df["cat_feat"].unique()]

    set1_rows = _build_set1_rows()
    set1_labels = [_set1_label(*r) for r in set1_rows]
    set1_boundaries = _ks_block_boundaries(set1_rows)

    peripheral_combos = [
        (s, b, t)
        for s in SOLV_ORDER
        for b in BASE_ORDER
        for t in TRANSFORM_ORDER
    ]

    for split_type, case_title in CASE_STUDIES:
        subset = df[df["split_type"] == split_type]
        if subset.empty:
            print(f"[heatmap] WARNING: no rows for {split_type}; skipping.")
            continue
        print(f"[heatmap] {split_type}: {len(subset)} rows")

        for metric_label, metric_col, higher_is_better in METRIC_SPECS:
            if metric_col not in subset.columns:
                print(f"[heatmap] WARNING: column '{metric_col}' missing; skipping.")
                continue

            folder = OUT_DIR / f"{split_type}_{metric_col}"
            folder.mkdir(exist_ok=True, parents=True)

            for solv, base, transform in peripheral_combos:
                slice_df = subset[
                    (subset["solv_feat"] == solv)
                    & (subset["base_feat"] == base)
                    & (subset["outcome_transform"] == transform)
                ]
                if slice_df.empty:
                    continue

                med_s, var_s, n_s = _aggregate(slice_df, SET1_ROW_KEYS, metric_col)
                mean_df, var_df = _pivot_to_table(
                    med_s, var_s, set1_rows, set1_labels, cats,
                )
                periph_label = _set2_label(solv, base, transform)
                n_repr = int(n_s.median()) if len(n_s) else 0
                reduction = "pooled" if metric_col == "crpss" else "median"
                n_note = f"{reduction} over {n_repr} folds/cell"
                if split_type in LOW_N_SPLITS:
                    n_note += "  ⚠ low-n, interpret with caution"
                _render_heatmap(
                    mean_df, var_df, metric_label, higher_is_better,
                    f"{case_title} — {metric_label}\n"
                    f"rows = KS / model / mean   |   cols = catalyst featurization   |   {periph_label}\n"
                    f"{n_note}",
                    folder / f"{solv}_{base}_{transform}.png",
                    compact=True,
                    block_boundaries=set1_boundaries,
                )

    print(f"[heatmap] done — output in {OUT_DIR}")


if __name__ == "__main__":
    main()
