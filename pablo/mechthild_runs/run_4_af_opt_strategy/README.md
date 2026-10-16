# run_5 — AF-optimizer comparison: models × batch-schemes × tiers

Compares two GP setups against four q=6 batch decompositions, each at six GA-budget
tiers, repeated over 25 reps. Measures how batch-decomposition and GA budget affect the
qLogNEI batch the optimizer selects, for both a single-task yield model and a multitask
yield/selectivity model.

Earlier batch-vs-sequential result sets and reports are archived under
[`data/_archive_run5_ladder/`](data/_archive_run5_ladder/) and
[`data/_archive_run6_tiers/`](data/_archive_run6_tiers/).

## Setups

| | `st_y` (Setup 1) | `lcm_ys` (Setup 2) |
|---|---|---|
| Architecture | `SingleTaskGP` on yield only | rank-2 LCM `KroneckerMultiTaskGP` on (Yield, Selectivity) |
| Kernel / mean / transform | Vanilla / constant / raw | KS1 / type_linear / raw |
| AF objective | yield output directly (`yield_method="direct"`) | **Yield task only** — `samples[..., 0]`, no C·S product (`yield_method="marginal"`) |
| Acquisition | qLogNEI | qLogNEI |
| Featurization | grover / map / map (shared) | grover / map / map (shared) |

`st_y` reproduces the validated `REPRODUCE.py` / `pablo.main.train_gp_model` recipe.

**Percent-scale targets.** The featscan loader returns yield/selectivity as fractions in
`[0, 1]`. qLogEI scales linearly with the outcome units, so fractions push log-EI
strongly negative (a constant `log(100)` offset) without changing the argmax. Targets are
rescaled to percent before fitting (`main_variant._scale_cell_targets`,
`config.TARGET_PERCENT_SCALE = 100`) so AF values are positive and match `REPRODUCE.py`.

## Batch-schemes

A q=6 batch is built in sub-batches of the sizes in `steps`:

| scheme | steps | GA calls |
|---|---|---|
| `full_batch` | `[6]` | 1 |
| `split_3_3` | `[3, 3]` | 2 |
| `split_2_2_2` | `[2, 2, 2]` | 3 |
| `sequential` | `[1,1,1,1,1,1]` | 6 |

At sub-batch `i` (cumulative `k = sum(steps[:i+1])`), the GA frees `steps[i]` new
candidates with the already-chosen ones held fixed (`pre_batch`), then the continuous
L-BFGS refine re-optimizes the continuous dims of all `k` candidates under the partial
temperature partition for `k`. The `(2,2,2)` temperature partition is enforced on the
full q=6 batch (at `k=6`) regardless of decomposition. `sequential` reproduces the
original greedy method; `full_batch` is a single GA call.

## Grid & packaging

- Cells = `25 reps × 2 models × 4 schemes × 6 tiers = 1200` optimizations.
- **One worker = one `(rep, model, scheme)` cell**; the six tiers are an inner loop.
  `N_WORKERS = 25 × 2 × 4 = 200`. Flat index `idx ∈ [0,200)` decodes via
  `main_variant._decode` as `rep` (slowest) → `model` → `scheme` (fastest).
- Per-optimization seed via `main_variant._opt_seed(rep, model, scheme, tier)`, unique
  over the 1200-cell grid (1000-stride leaves room for the sub-batch `seed+i` bumps).
- All models fit once into a single shared bundle (`data/prepared_bundle.pt`: one state
  dict per model + the shared AF search space).

## Result schema

`data/results/result_<idx>.json`:

```
{ worker_id, rep_idx, model_idx, model, scheme_idx, scheme, meta,
  results: [ <one record per tier> ] }
```

Each tier record: `{ tier_idx, model, scheme, scheme_steps, rep_idx, seed,
config_snapshot, result }`, where `result` is the uniform scheme output:

```
result: { steps: [ { sub_batch, increment, k,
                      post_ga:         { candidates, acq_value, wall_time_s },
                      post_refinement: { candidates, acq_value, wall_time_s } }, ... ],
          candidates,          # final (q=6, d_total) batch
          final_acq_value, total_wall_time_s }
```

The unified `steps` list replaces the former method-A / method-B split; `full_batch` has
a single-element `steps`, `sequential` six.

## Tiers & calibration (gated)

Six GA-budget tiers per scheme. Per-eval cost differs by scheme, so each scheme has its
own ladder (`config.TIER_SETTINGS_BY_SCHEME`). **The ladders shipped in `config.py` are
placeholders.** Fill them from `calibrate`, which times each `(model, scheme)` and
recommends `ga_n_evals` per tier against `config.TIER_TARGET_HOURS`:

```bash
python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant calibrate
```

`ga_n_evals` is the GA budget **per sub-batch GA call**. Each worker runs all six tiers,
so its wall time ≈ `sum(TIER_TARGET_HOURS)` h. Set `config.SLURM_TIME_LIMIT` to cover
that sum (+ margin) before submitting. Run `calibrate` on the sweep's node type.

## Files

- `config.py` — `MODELS`, `BATCH_SCHEMES`, `TIER_SETTINGS_BY_SCHEME`, `TIER_TARGET_HOURS`,
  `TARGET_PERCENT_SCALE`, `N_REPS`, `N_TIERS`, packaging, SLURM.
- `main_variant.py` — `prepare` (fit all models + serialize), `calibrate`, `smoke`,
  `worker` (one cell, all tiers), `cell` (single-opt local gate); `_run_scheme` carries
  the generalized sub-batch algorithm.
- `scheduler.py` — file queue (`init`/`worker`/`status`/`submitted`).
- `submit_mechthild.py` — SLURM array submitter (`submit`/`status`/`autoscale`).
- `run_state.py` — per-worker heartbeat (model / scheme / tier / sub-batch).

## Usage (from repository root)

### 1) Prepare (fit both models)

```bash
python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant prepare
```

### 2) Smoke / local single-cell gate

```bash
python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant smoke --model st_y
python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant smoke --model lcm_ys
python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant cell --model st_y --scheme full_batch --tier 0
```

`smoke` runs a tiny full-batch GA and prints the AF (st_y lands ~0.4–0.6 at a full
budget). `cell` runs one real (model, scheme, tier, rep) opt at the ladder settings.

### 3) Calibrate, then fill `config.py` and set `SLURM_TIME_LIMIT` (gated)

### 4) Initialize queue

```bash
python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler init
```

Creates 200 pending locks (skips any with an existing `result_<i>.json`).

### 5) Submit to Mechthild

```bash
python -m pablo.mechthild_runs.run_4_af_opt_strategy.submit_mechthild submit
```

`submit` arrays only uncovered pending jobs (`autoscale` resubmits as slots free). Full
sbatch script generated by `submit_mechthild.py` (time limit from `config.py`):

```bash
#!/bin/bash
#SBATCH -J af_opt_cmp
#SBATCH --array=1-200%200
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o pablo/mechthild_runs/run_4_af_opt_strategy/data/logs/worker_%A_%a.out
#SBATCH -e pablo/mechthild_runs/run_4_af_opt_strategy/data/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TORCH_NUM_THREADS=2

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler worker --single
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

### 6) Monitor

```bash
python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler status
python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler submitted
python -m pablo.mechthild_runs.run_4_af_opt_strategy.submit_mechthild status
```

### 7) Re-submit after failures

`scheduler init` + `submit` are idempotent over completed work: `init` marks every worker
with a `result_<idx>.json` as done and only re-pends the rest.
