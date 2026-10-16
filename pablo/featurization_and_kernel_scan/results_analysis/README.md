# Results analysis

Plots and summary tables produced from `results/scan_aggregate.parquet`.

Two scripts:

- [plot_heatmap_tables.py](plot_heatmap_tables.py) — renders all heatmap PNGs
  into this folder. **Flat layout** — no per-case-study subfolders. Output
  filenames follow the pattern
  `<model_class>_<case_study>[_<solvent>]_<plot_kind>.png`. Example:
  `singletask_primary_heatmap_all.png`, `multitask_leave_one_solvent_out_THF_heatmap_all.png`.
  The case-study winners chart is `case_study_winners.png`.
- [scan_summary_tables.py](scan_summary_tables.py) — writes per–`model_class`
  CSVs and a Markdown file under `results/` (`summary_singletask_*.csv`,
  `summary_multitask_*.csv`, plus `summary_<model_class>_tables.md`).

## Aggregation rules

All plots use **two-step aggregation**:
1. Within each fold (`split_type`, `repeat`, `fold`), the metric is averaged across
   (`solv_feat`, `base_feat`) sub-configs to produce one per-fold value per
   (KS, cat_feat).
2. Across folds, NLPD and CRPS are simple means; **MSE is `n_test`-weighted**
   because folds have unequal sizes (LOO=1, leave-one-solvent-out ranges 1–47, etc.).

σ² in every cell is the across-fold variance of the per-fold metric. It is `—`
when only one fold is available.

Only `outcome_transform == "raw"` cells are plotted. `logit` cells use a
post-inverse-logit posterior that is no longer Gaussian, so the Gaussian
NLPD/CRPS closed forms in `metrics/predictive.py` do not apply directly. A
sample-based metric in `metrics/predictive.py` is the proper fix.

Color scheme: yellow → orange → deep orange, **per-panel min–max with no
clipping**. The colorbar shows the panel's actual extrema.

## Per-case-study outputs

- `*_primary_heatmap_all.png` / `*_primary_heatmap_per_config.png` — 1×3 grid
  (CRPS / NLPD / MSE), rows = KS1–KS4, cols = 12 catalyst featurizations.
  Averages the (solv × base) sub-configs first, then per-fold across the 120
  primary CV folds (20 repeats × 6 folds). `_per_config` splits by the four
  (solv, base) sub-configs `phy/str`, `phy/one`, `one/str`, `one/one`.
- `*_leave_one_type_out_heatmap_all.png` etc. — same two plots over the 4
  pre-cat-type folds.
- `*_leave_one_solvent_out_<solvent>_heatmap_all.png` — one per held-out
  solvent. Solvents are split rather than averaged because THF accounts for
  47/65 measurements; averaging would let the THF-out fold silently dominate
  the rare-solvent folds.
- `*_leave_one_out_distributions.png` — 1×3 histograms of pooled per-point
  CRPS / NLPD / squared-error for the best (KS, cat_feat) cell (picked by mean
  NLPD).
- `*_leave_one_replicate_group_out_calibration.png` — 4×12 heatmap of
  per-fold `calibration_ks` (lower = better), the KS distance between
  standardized residuals and N(0, 1).

## Metric notes

- CRPS, NLPD, MSE, calibration KS — all lower-is-better.
- NLPD is the Gaussian closed form on the modeling scale of
  `outcome_transform = raw`. It can be negative when predictive variance is
  small enough that `log p(y) > 0`.
- MSE is computed as `rmse**2` per fold and averaged across folds with
  `n_test` weighting.
