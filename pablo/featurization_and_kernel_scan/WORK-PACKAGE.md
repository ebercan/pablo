Date: 26.09.2026

Batch Bayesian optimisation of an NHC-catalysed acyloin condensation. This package
selects the model specification — catalyst, solvent and base representations, kernel
structure, mean function, outcome transform and prior — by cross-validating a grid
of specifications against 69 experiments.

## The grid

One grid. Every cell is a full model specification; every surviving cell is scored
on all six cross-validation studies.

| axis | levels | n |
| --- | --- | --- |
| catalyst level (`cat_feat`) | the 27 featurizations `chembert2` `clamp` `ecfp4` `frad_carbene` `frad_precat` `graphormer_carbene` `graphormer_precat` `grover` `infomax3d_carbene` `infomax3d_precat` `interpretable` `interpretable_dG` `interpretable_light` `interpretable_steric` `map4` `minimol_carbene` `minimol_precat` `mol2vec` `molformer_xl_carbene` `molformer_xl_precat` `mordred` `scage_carbene` `scage_precat` `soap` `transformerm_carbene` `transformerm_precat` `unimol`, plus `<name>_pca` (Kneedle knee, frozen, full-catalog, X-only) for every one except the four `interpretable*` sets | 50 |
| base kernel | `rbf`, `matern52` everywhere; `tanimoto`, `minmax` additionally on full-dimensional `ecfp4` and `map4` | 2 (+2) |
| solvent block | `KT`, `Hansen`, `Gutmann`, `OHE`, `map` | 5 |
| base block | `thermo`, `steric`, `OHE`, `map` | 4 |
| kernel structure | `KS1`, `KS2`, `KS3`, `KS4` | 4 |
| mean function | `constant`, `block_linear`, `type_linear`, `type_quadlog` | 4 |
| outcome transform | `raw`, `logit`, `arcsin_sqrt`, `probit` | 4 |
| prior scheme | `P1`, `P2`, `P3` | 3 |
| model class | `st_y` | 1 |

**2,080 representation combos × 192 model variants = 399,360 cells.**
A *representation* is `(catalyst level, base kernel, solvent block, base block)`.
A *model variant* is `(kernel structure, mean function, outcome transform, prior scheme)`.
At 137 folds that is 54,712,320 fits and 161,647 CPU-h, so the grid is covered in
phases. The first is the elimination phase below; what runs on its surviving
catalyst levels is not yet specified.

Catalyst widths run 10 (`interpretable_steric`) to 7425 (`soap`) full, and 6 to 29
reduced. Measured per-fold costs are under "Runtime".

## Cross-validation studies

Six studies, 137 folds, over 69 experiments. These are fixed.

| # | study | folds | held out per fold | train / test |
| --- | --- | --- | --- | --- |
| 1 | `leave_one_type_out` | 2 | one pre-catalyst form: `bicarbonate`, `free_nhc` | 66 / 3 |
| 2 | `leave_one_scaffold_out` | 3 | one scaffold: `imidazolylidene`, `thiazolylidene`, `triazolinylidene` | 62 / 7 |
| 3 | `leave_one_replicate_out` | 5 | one replicate group, ids 0–4 | 66 / 3 |
| 4 | `leave_one_solvent_out` | 7 | one solvent: `1,4-Dioxane`, `DCM`, `Dimethoxyethane`, `H2O`, `HFIP`, `Me-THF`, `Toluene` | 66 / 3 |
| 5 | `primary` | 60 | 10 repeats × 6 folds, catalyst-grouped | 51 / 18 |
| 6 | `leave_one_out` | 60 | one experiment | 68 / 1 |

Studies 1–4 are **17 folds in total**. They are small because the library contains
only that many types, scaffolds, replicate groups and solvents.

Per fold the run records y-predicted and y-actual, per-point CRPS against the model
and against a constant forecast N(mean(y_train), var(y_train)) (the baseline), MSLL,
MAE, RMSE, Q² and the calibration KS statistic. The baseline depends only on the
fold's data, so it is identical for every configuration.

## Scoring

**Pooled CRPSS.** Per (configuration, study), failed folds are dropped and

    CRPSS_pooled = 1 − ΣCRPS_model / ΣCRPS_baseline

with both sums over every test point of every successful fold in the study
(`pablo/utils/metrics.pooled_crps_skill_score`). 0 ties the baseline, 1 is a perfect
forecast. Pooling weighs a fold by its baseline CRPS mass: three folds with
ΣCRPS_model 0.05, 0.08 and 0.30 against 0.10 each pool to −0.433, where the median
of their fold CRPSS is +0.20.

**Gem score** = min(CRPSS_pooled on `primary`, CRPSS_pooled on `leave_one_out`).
The other four studies are scored the same way and reported; they never enter the
gem. MSLL, MAE, RMSE, Q² and the calibration KS statistic are reported per study as
medians over the successful folds.

**Fold failure.** A fold fails, and is recorded as `status = "failed"` with its
error, when `fit_eval` raises:

| criterion | error |
| --- | --- |
| every L-BFGS restart ends with a non-finite MLL | `NoConvergedRestartError` |
| an L-BFGS restart ends ABNORMAL in all of botorch's fit attempts | `ModelFittingError` |
| Cholesky jitter ≥ 1e-3 | `ExcessiveJitterError` |
| fitted noise variance < 10 × final jitter | `NoiseBelowJitterError` |
| any test-point μ or σ is NaN or inf, or σ ≤ 0 | `NonFinitePredictionError` |
| any other exception, including `NotPSDError` | its own type |

Noise and jitter are both on the standardized scale the GP is fitted on. The final
jitter is the largest jitter added while factorising the training covariance at the
selected hyperparameters; gpytorch adds jitter only after a factorisation fails, so
with no jitter warning it is 0. Jitter escalates 1e-6 → 1e-5 → 1e-4 and then raises
`NotPSDError`, so a factorisation that needs more than 1e-4 fails as `NotPSDError`
and `ExcessiveJitterError` cannot fire at the default three tries.

**Flags.** Per (configuration, study) with n folds: *ineligible* if
n_failed > floor(0.05 · n) (60 folds tolerate 3 failures, 7 folds none), *not
production-ready* if n_failed > 0. Flags mark a configuration; they never remove
it. A fold that is absent — neither ok nor failed — is an error, not a failure.

## Elimination phase

Every catalyst level is fitted at one baseline and in four one-axis sweeps around
it. Solvent block, base block, base kernel and model class stay at the baseline
throughout.

| axis | baseline |
| --- | --- |
| base kernel | `matern52` (every level, full-dimensional `ecfp4` and `map4` included) |
| kernel structure | `KS1` |
| solvent block | `OHE` |
| base block | `OHE` |
| mean function | `constant` |
| outcome transform | `raw` |
| prior scheme | `P1` |
| model class | `st_y` |

| sweep | levels | configurations |
| --- | --- | --- |
| kernel structure | `KS1` `KS2` `KS3` `KS4` | 50 × 4 = 200 |
| prior scheme | `P1` `P2` `P3` | 50 × 3 = 150 |
| outcome transform | `raw` `logit` `arcsin_sqrt` `probit` | 50 × 4 = 200 |
| mean function | `constant` `block_linear` `type_linear` `type_quadlog` | 50 × 4 = 200 |

Each sweep contains the baseline, which is enumerated once: 15 configurations per
catalyst level, 12 of them unique, **600 configurations** over the 50 levels. Every
configuration is one outer cell with one inner cell, so the phase is 600 cells and
82,200 fits. All 600 run as one queue in one submission; there are no rounds
inside the phase. The grid is `config.ELIMINATION_GRID`;
`generate_inputs.configurations` enumerates it and `generate_inputs.sweep_membership`
names the sweep a configuration belongs to (the baseline belongs to all four).

**Elimination rule.** A catalyst level is eliminated if its gem is below 0.1 in any
of its 15 configurations; the survivors clear 0.1 in all 15. Every configuration is
anchored at the baseline, so the survivor set is an intersection over the four
sweeps and does not depend on their order. The rule is applied to
`results/gem_scores.parquet` after the sweep, not inside it, and every configuration
stays on disk, so a different rule is a re-analysis rather than a re-run. Survivors
are reported per sweep and overall, with the per-study pooled CRPSS, `n_ok` /
`n_failed` and the flags.

Open, to be settled before the elimination is applied: how a configuration flagged
*ineligible* counts, and whether a second rule (best level ≥ τ, or within
fold-bootstrap noise of the best) is reported alongside.

## Crossed run `cross_1`

Seven catalyst levels crossed with every other axis that is open at KS1. Every
combination is one configuration.

| axis | levels | n |
| --- | --- | --- |
| catalyst level | `grover_pca` `soap_pca` `interpretable_steric` `unimol` `minimol_precat_pca` `scage_carbene` `transformerm_precat_pca` | 7 |
| solvent block | `KT` `Hansen` `Gutmann` `OHE` `grover_pca` `soap_pca` | 6 |
| base block | `thermo` `steric` `OHE` `grover_pca` `soap_pca` | 5 |
| base kernel | `rbf` `matern52` | 2 |
| outcome transform | `raw` `probit` `arcsin_sqrt` | 3 |
| prior scheme | `P1` `P2` | 2 |
| kernel structure / mean function / model class | `KS1` / `constant` / `st_y` | 1 |

**2,520 configurations**, one outer cell with one inner cell each, 137 folds per
cell against the same `data/grid/cv_splits.json`: 345,240 fits. The grid is
`config.CROSSED_GRID_1`. Assembled design matrices are 21
(`interpretable_steric` × `Gutmann` × `thermo`) to 1,568 (`unimol` × `OHE` ×
`OHE`) columns wide.

**Projected cost: 616 CPU-h, 2 h 07 min on 300 workers.** The slowest cells are
`unimol` × `P2` × `probit` at 30 min; the median cell is 12 min 52 s. Per-fold
cost is the probed baseline time at the cell's width times the probed slowdown of
each axis that differs from the elimination baseline, taken at the same width
(`scheduler._primary_seconds_per_fold`). The solvent block, base block and base
kernel enter only through the width: none of them was probed, and `rbf` is costed
as `matern52`. Widths below 44 columns are costed at 44.

Cell ids are assigned by projected cost, most expensive first, so the queue,
which hands out the lowest pending id, starts the 30-min `unimol` cells at
submission.

## Listed run `monroe_m3`

The Monroe pre-catalyst embedding against `grover_pca` under the ten best
`cross_1` configurations by gem (`runs/cross_1/gem_scores.parquet`, ranks 1-10,
all `grover_pca`). Every configuration is fitted with three catalyst blocks:
`grover_pca` (control), `monroe_precat_pca` (Monroe at its knee, 15 PCs, 87.4%
of the variance) and `monroe_precat_pca8` (the same fit truncated at 8 PCs,
73.6%). The control is refitted, so all three blocks share one code version and
one noise prior.

| rank | solvent | base | base kernel | transform | prior |
| ---: | --- | --- | --- | --- | --- |
| 1 | `Hansen` | `grover_pca` | `matern52` | `probit` | `P2` |
| 2 | `Hansen` | `steric` | `matern52` | `arcsin_sqrt` | `P2` |
| 3 | `Hansen` | `OHE` | `matern52` | `probit` | `P2` |
| 4 | `grover_pca` | `grover_pca` | `matern52` | `probit` | `P2` |
| 5 | `Hansen` | `steric` | `matern52` | `probit` | `P2` |
| 6 | `Gutmann` | `OHE` | `matern52` | `arcsin_sqrt` | `P1` |
| 7 | `Gutmann` | `steric` | `rbf` | `arcsin_sqrt` | `P1` |
| 8 | `Gutmann` | `steric` | `matern52` | `arcsin_sqrt` | `P1` |
| 9 | `Gutmann` | `steric` | `matern52` | `arcsin_sqrt` | `P2` |
| 10 | `Gutmann` | `steric` | `matern52` | `probit` | `P1` |

KS1, `constant` mean and `st_y` throughout. **30 configurations**, one outer
cell each, 137 folds per cell: 4,110 fits, projected at 6 CPU-h and 14 min 32 s
on 30 workers. The grid is `config.MONROE_M3_GRID`.

A Monroe block is promising when its mean paired Δgem over the ten
configurations (Monroe − control, per configuration) is ≥ 0 and it beats the
control in at least 5 of the 10. Each per-configuration Δgem is reported with a
95% paired fold-bootstrap interval over the `primary` and `leave_one_out` folds
(4,000 draws).

## Anchored run `kernel_mean_1`

Two kernel structures and three mean functions, tested at the best `cross_1`
configuration of six of its seven catalyst levels (`runs/cross_1/gem_scores.parquet`;
`unimol` is not an anchor). Their design and the literature behind them are in
`kernel_mean_research.html`. Each anchor is crossed with four kernel structures
and four means; the `KS1` / `constant` cell refits the anchor under the current
code and noise prior and is the paired control of the other 15 cells of that
anchor.

| anchor | solvent | base | base kernel | transform | prior | `cross_1` gem |
| --- | --- | --- | --- | --- | --- | ---: |
| `grover_pca` | `Hansen` | `grover_pca` | `matern52` | `probit` | `P2` | +0.425 |
| `soap_pca` | `Gutmann` | `grover_pca` | `rbf` | `probit` | `P2` | +0.346 |
| `minimol_precat_pca` | `grover_pca` | `OHE` | `rbf` | `arcsin_sqrt` | `P2` | +0.318 |
| `transformerm_precat_pca` | `grover_pca` | `OHE` | `rbf` | `probit` | `P2` | +0.317 |
| `scage_carbene` | `Gutmann` | `thermo` | `rbf` | `probit` | `P1` | +0.297 |
| `interpretable_steric` | `Gutmann` | `thermo` | `rbf` | `probit` | `P2` | +0.278 |

| axis | levels |
| --- | --- |
| kernel structure | `KS1`, `KS2`, `KS5`, `KS6` |
| mean function | `constant`, `cond_linear`, `cond_trend`, `form_offset` |

`st_y` throughout. **96 configurations**, one outer cell each, 137 folds per
cell: 13,152 fits. The grid is `config.KERNEL_MEAN_1_GRID`, a
`config.AnchoredGrid`.

| level | covariance or mean | prior on the new parameter | config |
| --- | --- | --- | --- |
| `KS5` | σ² [ρ + (1 − ρ) k_cat] · k_solv · k_base · k_cont | ρ ~ Beta(1, 3) | `KS5_FLOOR_BETA` |
| `KS6` | s_x² Π_b k_b + s_+² (k_solv + k_base + k_cont) / 3, block kernels shared | s_+ ~ HalfNormal(0.5) | `KS6_MAIN_SCALE_HALFNORMAL` |
| `cond_linear` | c + a_solv·x_solv + a_cont·x_cont + base_active (b_base + a_base·x_base) | slopes N(0, 1/d_block), intercepts N(0, 1) | `MEAN_MODULE_NORMAL_SCALE` |
| `cond_trend` | constant mean; kernel + s² h(x)ᵀ D h(x'), the `cond_linear` coefficients integrated out | s ~ HalfNormal(1) | `COND_TREND_SCALE_HALFNORMAL` |
| `form_offset` | constant mean; kernel + τ² 1[form = form'] | τ ~ HalfNormal(0.5) | `FORM_OFFSET_SCALE_HALFNORMAL` |

`KS5` and `KS6` leave the base block unmasked, as `KS1`, so base-free rows sit at
the all-zero base location. `KS7` (a shared condition response plus a catalyst
deviation centred over the 161 library catalysts) is implemented but not on this
grid: its cost grows with catalyst width. The
`cond_trend` basis is the solvent and continuous columns minus 0.5, the base
columns minus 0.5 times `base_active`, and `base_active` itself.

**Projected cost: 32 CPU-h, longest cell 54 min 15 s** (`scage_carbene` × `KS5` ×
`cond_linear`), median cell 16 min 15 s, so 96 workers finish in under an hour.
None of the five new levels was probed on Mechthild. Their slowdowns
(`scheduler._ESTIMATED_FACTORS`) are ratios to `KS1` / `constant` measured on one
local machine on three folds of the `grover_pca`, `scage_carbene` and `unimol`
cells: `KS5` 1.31–1.42, `KS6` 1.03–1.37, `cond_linear` 2.46–2.75, `cond_trend`
0.89–1.10, `form_offset` about 1.0.

A proposal is accepted when its paired fold-bootstrap gem difference against the
anchor's `KS1` / `constant` cell is resolved positive on at least four of the
six anchors and resolved negative on none, its `leave_one_out` pooled CRPSS is
resolved worse on no anchor, it has no failed fold, and its new scale parameter
is identified (its standard deviation over the 60 `primary` folds is below its
prior standard deviation and its median is away from zero).

## Runtime

Every configuration of the elimination grid was timed on Mechthild with
`timing_probe` (jobs 599692 and 599730): the 12 configurations at three design-matrix
widths, `grover_pca` (44 columns), `grover` (5,028) and `soap` (7,453), two
`primary` folds each, 16 one-CPU tasks per node. All 96 probed folds finished ok.

| level | s/fold at 44 | at 5,028 | at 7,453 |
| --- | --- | --- | --- |
| baseline | 4.5 | 12.6 | 29.4 |
| `KS2` / `KS3` / `KS4` | 3.9 / 4.7 / 4.3 | 12.4 / 10.8 / 8.9 | 27.7 / 23.4 / 18.7 |
| `P2` / `P3` | 3.9 / 1.9 | 26.2 / 7.0 | 26.2 / 7.1 |
| `logit` / `arcsin_sqrt` / `probit` | 4.0 / 4.2 / 5.1 | 11.3 / 13.2 / 11.8 | 19.6 / 21.9 / 19.2 |
| `block_linear` | 14.8 | 129.4 | 249.5 |
| `type_linear` | 20.7 | 80.7 | 124.4 |
| `type_quadlog` | 21.0 | 85.1 | 240.0 |

**The mean function is the only axis that moves the cost.** Kernel structure, prior
scheme and outcome transform stay within 0.24–2.1 times the baseline at every
width; the three non-constant means cost 3.3–4.7 times the baseline at 44 columns
and 4.2–10.3 times at 5,028 and 7,453. A `leave_one_out` fold (68 training points)
costs a median 1.22 times a `primary` fold (51) over 12 configurations probed on
both (range 0.68–2.39). Two folds per level leave the fold-to-fold spread large, up
to 2.3 times between the two folds of one level.

`scheduler estimate-runtime` interpolates these times linearly in width
(`scheduler._PRIMARY_SECONDS_PER_FOLD`), scales each fold by its training-set size
(`_LOO_TO_PRIMARY_COST`, 137 folds weigh as 158 `primary` folds) and adds 67 s of
startup per task:

| sweep | cells | CPU-h | slowest cell |
| --- | --- | --- | --- |
| kernel structure | 200 | 50 | 1 h 18 min |
| prior scheme | 150 | 34 | 1 h 18 min |
| outcome transform | 200 | 51 | 1 h 18 min |
| mean function | 200 | 207 | 10 h 58 min |
| **all, baseline once** | **600** | **302** | **10 h 58 min** |

The baseline cells count toward every sweep row; the median cell is 15 min. The
wall time is set by one cell, `soap` × `block_linear` (cell 201), followed by
`soap` × `type_quadlog` at 10 h 33 min and `grover` × `block_linear` at 5 h 42 min.
All three are round-1 cells and start at submission: **10 h 58 min** when every slot
a finished task frees takes the next cell, **13 h 42 min** (10 h 58 min + 2 h 43 min)
when round 2 is submitted only after round 1 has drained.

## Model, data and priors

* **Outcome transforms.** Four:
  `raw`, `logit`, `arcsin_sqrt`, `probit` (`pablo/utils/outcome_transforms.py`,
  `config.OUTCOME_TRANSFORMS`). Each is a link applied to the boundary-guarded
  squash `p = (y + ε)/(1 + 2ε)`, ε = 1e-3, with an exact algebraic inverse —
  `fit_eval` draws posterior samples on the transform scale and back-transforms
  them before any metric, and every metric and its constant-model baseline is
  defined on the raw fraction scale. `logit` uses the closed form
  `log((y+ε)/(1−y+ε))`, inverted exactly.
* **Features.** All 27 catalyst featurizations are present at 161 rows, giving 50
  `cat_feat` levels (27 full + 23 `_pca`; the 4 interpretable sets have no `_pca`
  level). Full widths run 10 (`interpretable_steric`) to 7425 (`soap`); knees
  run 6 to 29 components and retain 43.7% (`ecfp4`) to 94.4% (`infomax3d_precat`)
  of variance, so the knee is not a constant-information cut — carry a
  fixed-variance variant (d95) if full against `_pca` levels is to be read as a
  controlled comparison. Verify widths with `features/dim_reduction.py`
  metadata, not from this page.
* **Base block on base-free rows.** Every experiment with `base_active = 0` has an
  all-zero base block (`utils/feature_assembly._build_base_block`): the 29 runs
  with no base and experiment 27 (catalyst 8, THF, Rb2CO3, 80 °C), a non-salt
  catalyst with a base added. `base_active` follows the pre-catalyst form, so
  that run sits at the "none" location like every other non-salt row. The
  elimination cells in `results/cells/` were fitted with Rb2CO3 features on that
  row; `cross_1` is fitted with the zero block.
* **Interpretable sets have no `_pca` level.** Their value is that a lengthscale
  maps to a named chemical quantity, and at d=10–30 against 69 experiments there is
  no dimensionality problem to solve. The grid lists its `cat_feat` levels
  directly (`config.CAT_LEVELS`); the source featurization and the form one-hot
  flag are derived from the name.
* **Tanimoto / MinMax only on full-dimensional fingerprints** (ECFP4, MAP4), each
  with a Matérn control in the same cell. The ratio kernels need substructure
  counts to mean anything and are basis-dependent, so PCA'd or dense-embedding
  cells are dropped, not run.
* **Categories are one-hot everywhere.** `cat_form_id` / `ring_type_id` are
  expanded before storage (`interpretable_light` 14→21 dims, `interpretable` and
  `interpretable_dG` 23→30), and the `_carbene` form flag is appended as a
  4-column one-hot rather than an integer.
* **Prior schemes.** `config.PRIOR_SCHEMES` — P1 full ARD at `log√D_total`, P2
  full ARD at `log√D_block`, P3 one lengthscale per block at `log√D_block`.
  A partition built at one granularity with a prior location resolved at another
  is rejected rather than half-applied.
* **Lengthscale floor: 0.05.** One global constant applied to every fit under every
  prior scheme: the prior says where a lengthscale is expected to sit, the floor
  says where the optimizer may not go. Under isotropic blocks `KS1` sits at
  Δ(no-floor − floor) LML = 0, so it is a numerical guard rather than a modelling
  choice. It is calibrated for `P3`; `P1` and `P2` restore per-column ARD on wide
  blocks, so measure the floor-hit rate on the first `P1`/`P2` cells with
  `results_analysis/scan_lengthscale_floor.py` and report it with the run.
* **The prior is Hvarfner throughout.** `P1`, `P2` and `P3` vary only how `log√D`
  is resolved and how many lengthscales it covers. Prior width is not an axis.
* **Noise prior.** `InvGamma(ν/2, S/2)` from the pooled within-replicate sums of
  squares, under a Jeffreys reference prior on σ². `S` is pooled on the scale the
  GP is trained on, one prior per target (yield, conversion, selectivity) and
  outcome transform (`config.NOISE_EVIDENCE`), and the model builders divide β by
  the variance of the training targets on that scale. Location is the measured
  repeatability; spread is `CV = √(2/(ν−4))` and tightens as groups are added.
  A stance that wants to budget for misspecification scales β, which moves the
  mean and leaves the data-derived tightness alone. The two-task models carry one
  prior for both task noises and the global noise, at the mean of the two tasks'
  standardized rates. Results fitted before 3 October 2026 carry the raw-yield β
  on every transform and target: in `cross_1` the prior-mean noise variance of the
  `probit` cells sits 17 times, and of the `arcsin_sqrt` cells 1.7 times, below the
  replicate spread on their own scale.

## Operational

**Runs.** Every CV tool takes `--run <name>`, a key of `config.RUN_GRIDS`;
without it the tool works on `elimination`. `paths.run_paths(name)` gives the
run's files:

| run | `inputs.json` | queue and run state | result cells | logs | aggregates |
| --- | --- | --- | --- | --- | --- |
| `elimination` | `data/grid/inputs.json` | `data/grid/queue/` | `results/cells/` | `results/logs/` | `results/scan_aggregate.parquet`, `results/gem_scores.parquet` |
| `cross_1` | `runs/cross_1/inputs.json` | `runs/cross_1/queue/` | `runs/cross_1/cells/` | `runs/cross_1/logs/` | `runs/cross_1/scan_aggregate.parquet`, `runs/cross_1/gem_scores.parquet` |
| `kernel_mean_1` | `runs/kernel_mean_1/inputs.json` | `runs/kernel_mean_1/queue/` | `runs/kernel_mean_1/cells/` | `runs/kernel_mean_1/logs/` | `runs/kernel_mean_1/scan_aggregate.parquet`, `runs/kernel_mean_1/gem_scores.parquet` |

`data/grid/cv_splits.json` is shared. `runs/` and `results/` are git-ignored, so
`runs/cross_1/inputs.json` is generated on the cluster; `generate_inputs` is
deterministic. The SLURM job name is `feat_kernel_scan` for `elimination` and
`feat_kernel_scan_<run>` otherwise, and the worker count matches the name
exactly, so two runs never share a worker budget.

### `elimination`

**One task per cell, two rounds of 300.** At most 300 tasks are submitted at once
(`config.SLURM_MAX_CONCURRENT_WORKERS`). Each task runs `scheduler worker --single`:
it claims the lowest-id pending cell, runs its 137 folds and exits. Round 1 is
cells 0–299 (the first 25 catalyst levels, `soap` included), round 2 cells
300–599. `autoscale --single` starts a round-2 task in every slot a round-1 task
frees; submitting round 2 only after round 1 has drained adds the second round's
slowest cell to the wall time.

**Partition `medium`, `--time=23:59:00`.** `medium` caps the wall time at 24 h,
and the limit is always explicit (`config.SLURM_SINGLE_CELL_TIME_LIMIT`). The
slowest projected cell is 10 h 58 min, which leaves 2.2 times headroom for the
fold-to-fold spread of the probe.

**Namespace.** Within one run, the result cells and the queue are one flat
namespace keyed by outer-cell id. Re-enumerating a different grid under the same
run name makes `init` match new cell ids against old results, so a new grid gets a
new run name. `init` refuses a stored result whose configuration disagrees with `inputs.json`
(`scheduler.CELL_IDENTITY_KEYS` on the outer record and `INNER_IDENTITY_KEYS` on
the inner one, which is what separates the three prior-scheme cells of a level),
so a collision fails loudly. `data/grid/inputs.json` and `cv_splits.json` are
tracked, so the pull-only sync (`git clean -fd`) leaves them in place, and
`generate_inputs` reproduces the tracked file byte for byte; `data/grid/queue/` and
`results/` are git-ignored.

**Resuming.** `init --stale-after <seconds>` requeues cells whose worker died
before writing a result, using the run-state heartbeat to tell a dead worker from
a live one. A requeued cell resumes from its last finished fold.

`$HOME` on Mechthild is `/mechthild/home/enisbaris`. `/home/enisbaris` does not
exist and a log path there kills every array task instantly with `RaisedSignal:53`
and no output. `MaxArraySize` is 65000, `MaxJobCount` 10000.

**Timing probe.** `cross_validation/timing_probe.py` times the 12 configurations at
three catalyst widths (`grover_pca`, `grover`, `soap`), two folds each, 36 array
tasks on `short`. Each task runs `scheduler._run_outer_cell`, the worker's per-cell
path, and writes `results/timing_probe/task_<i>.json` (`task_<i>_<study>.json` for
a study other than `primary`). `summarize` prints seconds per fold, every level
against the baseline at the same width, the `_PRIMARY_SECONDS_PER_FOLD` table for
`scheduler.py`, and the `leave_one_out` / `primary` ratio where both were probed.

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe list
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe submit --folds 2
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe submit --folds 2 --study leave_one_out --tasks 0,10,12,16,21,22,23,24,28,33,34,35
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe summarize
```

`submit` writes and submits:

```bash
#!/bin/bash
#SBATCH -J feat_kernel_probe
#SBATCH --array=0-35
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH -o /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/results/timing_probe/logs/probe_%A_%a.out
#SBATCH -e /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/results/timing_probe/logs/probe_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=

python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe run --task "$SLURM_ARRAY_TASK_ID" --folds 2 --study primary
```

**Sweep.** From the repo root, with the venv active:

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --dry-run
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler estimate-runtime
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild submit --single --force
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild autoscale --single
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler status
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler submitted
python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results --require-complete
python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --cv-splits pablo/featurization_and_kernel_scan/data/grid/cv_splits.json
```

`autoscale` runs in the foreground until the queue and every worker are done; run
it inside `tmux` or `screen` on the login node. `concat_results` writes every fold
as a row with `status` `ok` or `failed`; `--require-complete` stops on an absent
fold, not on a failed one. `gem_scores` checks every configuration against
`cv_splits.json` and writes one row per configuration: per study
`crpss_pooled_<s>`, `n_ok_<s>`, `n_failed_<s>`, `ineligible_<s>`,
`not_production_ready_<s>` and `<metric>_median_<s>`, then `gem`, `gem_study` and
the configuration-level `ineligible` / `not_production_ready`.

`submit_mechthild submit --single` generates, for 300 tasks:

```bash
#!/bin/bash
#SBATCH -J feat_kernel_scan
#SBATCH --array=1-300%300
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/results/logs/worker_%A_%a.out
#SBATCH -e /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/results/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler worker --single
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

### `cross_1`

From the repo root on Mechthild, with the venv active. The pull-only sync uses
`git clean -fd`, never `git clean -x`: `runs/` and `results/` are git-ignored and
hold the only copy of the results.

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --run cross_1 --dry-run
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler estimate-runtime --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild submit --single --force --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild autoscale --single --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler status --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler submitted --run cross_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init --run cross_1 --stale-after 1800
python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results --run cross_1 --require-complete
python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --run cross_1
```

The dry run prints 2,520 outer cells. `submit --single` fills the 300 slots and
`autoscale --single`, inside `tmux` on the login node, starts one task per freed
slot until the queue is empty: 2,520 tasks, 8.4 per slot. `init --stale-after`
requeues cells whose worker died without writing a result. `concat_results`
writes `runs/cross_1/scan_aggregate.parquet`; `gem_scores` reads it with the
shared `cv_splits.json` and writes `runs/cross_1/gem_scores.parquet`.

`submit_mechthild submit --single --run cross_1` writes
`cross_validation/.submit_array_cross_1.sh` and submits, for 300 tasks:

```bash
#!/bin/bash
#SBATCH -J feat_kernel_scan_cross_1
#SBATCH --array=1-300%300
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/runs/cross_1/logs/worker_%A_%a.out
#SBATCH -e /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/runs/cross_1/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler worker --single --run cross_1
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

`--time=23:59:00` is 48 times the slowest projected cell.

### `monroe_m3`

Requires `data/featurizations/monroe_precat/`, `monroe_precat_pca/` and
`monroe_precat_pca8/` in the checkout. From the repo root on Mechthild, with the
venv active:

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --run monroe_m3
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler estimate-runtime --run monroe_m3
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init --run monroe_m3
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild submit --single --force --run monroe_m3
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler status --run monroe_m3
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init --run monroe_m3 --stale-after 1800
python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results --run monroe_m3 --require-complete
python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --run monroe_m3
```

`submit --single` submits all 30 cells at once; no `autoscale` is needed. It
writes `cross_validation/.submit_array_monroe_m3.sh`:

```bash
#!/bin/bash
#SBATCH -J feat_kernel_scan_monroe_m3
#SBATCH --array=1-30%300
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/runs/monroe_m3/logs/worker_%A_%a.out
#SBATCH -e /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/runs/monroe_m3/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler worker --single --run monroe_m3
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

### `kernel_mean_1`

From the repo root on Mechthild, with the venv active:

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --run kernel_mean_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler estimate-runtime --run kernel_mean_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init --run kernel_mean_1
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild submit --single --force --run kernel_mean_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler status --run kernel_mean_1
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init --run kernel_mean_1 --stale-after 1800
python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results --run kernel_mean_1 --require-complete
python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --run kernel_mean_1
```

`submit --single` submits all 96 cells at once, each its own array task; no
`autoscale` is needed. It writes `cross_validation/.submit_array_kernel_mean_1.sh`:

```bash
#!/bin/bash
#SBATCH -J feat_kernel_scan_kernel_mean_1
#SBATCH --array=1-96%300
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/runs/kernel_mean_1/logs/worker_%A_%a.out
#SBATCH -e /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/runs/kernel_mean_1/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler worker --single --run kernel_mean_1
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

The one-day limit covers the longest projected cell 26 times over.

### Fully Bayesian NUTS settings

`core.models.FullyBayesianScanGP` is the NUTS-fitted counterpart of an `st_y` ·
KS1 · constant configuration: the priors of the MAP model of the same
configuration, without its lengthscale and noise floors, and a flat prior on the
constant mean. The NUTS budget is chosen from a study on the `cross_1` rank-1
configuration (`grover_pca` · `Hansen` · `grover_pca` · `matern52` · `probit` ·
`P2`), leaving out the highest-, median- and lowest-yield experiments (rows 49,
27 and 0) one at a time:

| Setting | Chains | Warmup | Draws per chain | Thinning | Draws kept |
|---|---|---|---|---|---|
| A | 1 | 512 | 256 | 16 | 16 |
| B | 2 | 512 | 256 | 4 | 128 |
| C | 4 | 512 | 256 | 8 | 128 |
| D | 4 | 1000 | 1000 | 8 | 500 |

Each array task runs one chain, 3 × 11 = 33 tasks, and writes
`runs/nuts_settings/chains/chain_<task>.pt`; a task whose file exists is skipped.
From the repo root on Mechthild, with the venv active:

```bash
python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study generate
python -m pablo.featurization_and_kernel_scan.fully_bayesian.submit_nuts_settings
squeue -u $USER -n fb_nuts
python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study aggregate
```

`aggregate` merges the chains of each (point, setting), reports cost,
divergences, non-positive-definite evaluations, split-R̂, bulk ESS, the mixture
predictive on the probit scale and its raw-yield CRPS next to the MAP fit of the
same fold, and writes `runs/nuts_settings/table.md` and `table.parquet`. A local
check of one shortened chain:

```bash
python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study run --task 0 --warmup 10 --samples 10
python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study aggregate --smoke
```

`submit_nuts_settings` must run from the repo root, since the log paths are
relative to it. It writes `runs/nuts_settings/submit_nuts_settings.sh`:

```bash
#!/bin/bash
#SBATCH -J fb_nuts
#SBATCH --array=0-32%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o pablo/featurization_and_kernel_scan/runs/nuts_settings/logs/task_%A_%a.out
#SBATCH -e pablo/featurization_and_kernel_scan/runs/nuts_settings/logs/task_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[task] id=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study run --task $SLURM_ARRAY_TASK_ID
echo "[task] id=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

A local 150 + 150 chain on this cell takes at most 138 s (the figure includes a
two-restart MAP fit), at most 0.46 s per iteration, so the longest task (a D chain, 2,000 iterations) projects to about
15 minutes against the one-hour limit.

### Noise structure

`cross_validation/noise_structure.py` tests whether a noise variance that depends
on the yield level describes the data better than the homoskedastic noise every
scan GP carries. The family is `Var(y) = σ²·[p(1−p)]^θ` on the raw yield
fraction: `θ = 0` is homoskedastic on the raw scale and, to first order, `θ = 1`
on the arcsine scale and `θ = 2` on the logit scale. A GP on another outcome
scale receives it through the delta method, and σ² keeps the replicate prior,
pooled with the same shapes. `own` is the production model of a scale. The
configurations are the best `cross_1` cell of each outcome transform (`probit`,
`arcsin_sqrt`, `raw`; `noise_structure.CONFIGURATIONS`).

| task | n | fits | output |
| --- | --- | --- | --- |
| `cv` | 360 | every `primary` and `leave_one_out` fold of each configuration at every θ, with the scheduler's fold seed | per-point CRPS and NLPD on the raw scale, training objective |
| `profile` | 3 | all 69 experiments at θ = −1 … 3 in steps of 0.25 | penalized log marginal likelihood against θ |
| `decision` | 21 | the q = 6 qLogNEI batch of each variant from the production GA (`pablo.main.GA_SETTINGS`): `own` at three optimizer seeds, σ × 3, σ ÷ 3, two θ | the batch and its value under the production model's qLogNEI |

The batches follow the production constraints: temperatures in three pairs
(`pablo.main.TEMPERATURE_PARTITION`), time pinned at 20 h, and only the
components marked available in the design sheet's `Next Batch` column.
`replicates` prints the replicate-only evidence over θ and needs no GP.

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.noise_structure replicates
python -m pablo.featurization_and_kernel_scan.cross_validation.noise_structure list
python -m pablo.featurization_and_kernel_scan.cross_validation.noise_structure submit
squeue -u enisbaris -n feat_noise_structure
python -m pablo.featurization_and_kernel_scan.cross_validation.noise_structure summarize
```

A task whose `results/noise_structure/task_<i>.json` exists returns at once, so
resubmitting the array, or `submit --tasks <ids>`, runs only the missing tasks;
`summarize` names them. `results/` is git-ignored, so the task files exist only
on the cluster until they are copied back. `submit` writes
`results/noise_structure/.submit_noise_structure.sh` and submits:

```bash
#!/bin/bash
#SBATCH -J feat_noise_structure
#SBATCH --array=0-383%300
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH -o /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/results/noise_structure/logs/task_%A_%a.out
#SBATCH -e /mechthild/home/enisbaris/pablo/pablo/featurization_and_kernel_scan/results/noise_structure/logs/task_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=

python -m pablo.featurization_and_kernel_scan.cross_validation.noise_structure run --task "$SLURM_ARRAY_TASK_ID"
```

## Descriptor fixes, 18.09.2026

* `xtb_homo_eV`, `xtb_lumo_eV` and `xtb_gap_eV` were read at the unrelaxed
  ETKDG/MMFF94 geometry for all 161 catalysts. `_parse_xtb_output` took the
  first regex match over the `--opt` log, and xTB prints a full property block
  for the input geometry before the optimisation starts. `xtb_dipole_D` and
  `xtb_polarizability_angst3` were unaffected — xTB prints them only after
  convergence. Over the library: HOMO MAE 0.23 eV, LUMO 0.65 eV, gap 0.87 eV
  (max 3.0 eV); Spearman(old, corrected) for the gap 0.45; gap range
  [0.85, 2.89] → [1.54, 4.66] eV. The parser now reads only the segment after
  the final `GEOMETRY OPTIMIZATION CONVERGED`.
* `data/xtb/results_checkpoint.parquet` and the three assembled tables
  (`interpretable`, `interpretable_light`, `interpretable_dG`) were patched in
  place from the saved logs by `features/interpretable/repair_xtb_scalars.py`
  — no xTB re-run, no change to `data/xtb/inputs/`. The Stage-2 JSON caches for
  160 catalysts were never committed, so `interpretable`/`interpretable_dG`
  cannot be rebuilt through the assemblers; `interpretable_light` was rebuilt
  through `assemble_light` as a cross-check and reproduced the patch exactly.
  `tests/test_xtb_log_parsing.py` pins the parser to the converged block and
  the checkpoint to a fresh re-parse of the logs.
* Computed on the pre-fix columns: `archive/26.08.2026_pre_fingerprint/`
  (28 of 336 cells), `archive/03.06.2026/` (2 of 14 featurizations) and
  `mechthild_runs/run_3_hyperprior_scan/` (4 of 8 outer cells), each annotated in
  its MANIFEST. Production BO (`pablo/main.py`: `grover` + `form_*` one-hots) and
  `run_4` read only the `form_*` columns and are clean. The `tighter_ls` verdict
  holds on `grover` alone (median CRPSS 0.388 vs ≤ 0.171) and is unaffected.
* **`data/xtb/outputs/` is git-ignored and was the only copy of the logs the
  fix was recovered from.** A backup now exists outside the repo. 15 MB is not
  too large for history; consider tracking it.

## Descriptor fixes, 19.09.2026

* The 161 library entries are 113 distinct free carbenes; the rest are the same
  carbene as another salt, CO2 adduct or bicarbonate. Descriptors computed on
  the free carbene were nonetheless entry-specific, because each entry was
  embedded and optimised independently: six of the twenty-six duplicate groups
  disagreed, worst case `vbur_q4` by 0.81 pp (7% of that column's library
  spread), `xtb_deltaG_deprot` by 0.52 kcal/mol, `lp_B1` by 0.053 A. Each group
  now carries the values of its lowest-xTB-energy member
  (`features/interpretable/canonicalize_duplicates.py`, idempotent, dry-run by
  default). Nothing was recomputed — re-deriving `lp_*`/`vbur_q*` from the
  canonical geometry reproduces that member's stored values exactly. Patched in
  the two intermediate parquets and the xTB checkpoint as well as the three
  assembled tables, so a re-assembly reproduces it; re-run after any rebuild
  that regenerates the per-catalyst JSON caches.
* `form_*`, `mw` and `heavy_atom_count` still vary within a group, asserted in
  both the script and `tests/test_xtb_log_parsing.py`. C83 is an acyclic
  diaminocarbene (no ring, `ring_other`); it is handled correctly everywhere
  because the pipeline reads `carbene_idx` from `info.json` rather than
  re-deriving it — see gotcha 7.
## To build

* **Elimination script** in `results_analysis/`: the rule above applied to
  `gem_scores.parquet`, with sweep membership from
  `generate_inputs.sweep_membership`, writing the survivor table per sweep and
  overall.
* **Per-study fold ordering in the worker**, so a cell runs one study to completion
  before the next and can be abandoned between studies rather than after all 137
  folds.
* **Featurization rebuild check.** The four `interpretable*` sets carry the xTB
  scalar repair of 18.09 and the duplicate-group canonicalisation of 19.09.
  Confirm every catalyst featurization is current before `generate_inputs` runs.
