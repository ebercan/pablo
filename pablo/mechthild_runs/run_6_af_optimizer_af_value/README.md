# run_6 — pool-greedy vs GA by acquisition value at matched walltime

Compares `optimize_acqf_mixed_pool_greedy` with the production GA pipeline (`optimize_acqf_mixed_GA` + `optimize_acqf_continuous`) on one acquisition function: seven walltime tiers per optimizer (2 min to 2 h), three repetitions each, 42 SLURM tasks.

## Setting

| | |
|---|---|
| Data | all 69 rows of the featscan `experiments.parquet` |
| Model | `st_y`, catalyst `grover_pca`, solvent `Hansen`, base `grover_pca`, `matern52`, `probit`, prior `P2`, `KS1`, constant mean |
| Acquisition | qLogNEI on the probit-scale GP output, `X_baseline` = training inputs, 128 Sobol samples seeded by the rep |
| Batch | q = 6, temperature partition (2, 2, 2), time pinned at 20 h |
| Search space | catalysts, solvents and bases marked `y` in the `Next Batch` column of `pablo/nhc_doe.xlsx`; base slice zeroed for non-salt catalysts |
| Score | a separate qLogNEI with 4,096 QMC samples (seed 12345) and the unpruned baseline, applied to every final batch |

The model is fitted once by `prepare` and shared by all 42 tasks, so the runs differ only in optimizer, budget and seed.

## Grid

`N_WORKERS = 2 optimizers × 7 tiers × 3 reps = 42`. The tiers form two blocks (`config.TIER_BLOCKS`): the flat index runs through the blocks in order and inside a block decodes optimizer (slowest), tier, rep (fastest). Block 0 (tiers 0–4) is indices 0–29: 0–14 GA, 15–29 pool-greedy. Block 1 (tiers 5–6) is indices 30–41: 30–35 GA, 36–41 pool-greedy. The seed of a run is its rep (0, 1, 2) for both optimizers.

| tier | target | GA evals | GA refine restarts | pool-greedy scale | `n_pool` | partition | time | mem |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 min | 49,500 | 1 | 74.7 | 76,493 | short | 00:45:00 | 16G |
| 1 | 5 min | 149,500 | 2 | 188.3 | 192,819 | short | 00:45:00 | 16G |
| 2 | 10 min | 298,000 | 4 | 377.5 | 386,560 | short | 00:45:00 | 16G |
| 3 | 15 min | 446,500 | 6 | 566.8 | 580,403 | short | 00:45:00 | 16G |
| 4 | 25 min | 743,000 | 10 | 945.3 | 967,987 | short | 00:45:00 | 16G |
| 5 | 60 min | 1,875,000 | 20 | 1800 | 1,843,200 | short | 01:45:00 | 24G |
| 6 | 120 min | 4,687,500 | 20 | 3600 | 3,686,400 | medium | 03:30:00 | 40G |

GA: population 500 and 1,024 raw samples (the production values). One refine restart costs about 65 s on a worker node, so the production 20 restarts (~22 min) do not fit the short tiers; the restart count grows with the tier and the GA evaluations fill the rest of the target. Pool-greedy: a scale multiplies `n_pool` (1,024), `n_override` (256), `n_local` (256) and `n_refine` (32); `refine_maxiter` 30, one swap round and `polish_maxiter` 50 stay fixed. Its local pool rows are drawn around the 8 best training points, passed as `incumbents` in the optimizer layout.

Tiers 0–4 come from `calibrate` on node005 (1.149 ms per GA evaluation, 64.7 s per refine restart, pool-greedy 160 s at scale 100 and 636 s at scale 400). Tiers 5–6 come from the measured tier-4 runs: 1.28 ms per GA evaluation and 58 s per restart, so 20 restarts (~20 min) leave 2,400 s and 6,000 s of GA; pool-greedy ran at 2.0 s per scale unit on a shared node with ~6.4 MB of peak memory per unit (~11.5 GB at 1800, ~23 GB at 3600). `short` caps the wall time at 2 h, so tier 6 runs on `medium`.

## Result file

`data/results/result_<idx>.json`, written last by the task:

| field | content |
|---|---|
| `optimizer`, `tier`, `rep`, `seed`, `target_minutes`, `settings` | the unit and its optimizer arguments |
| `score` | common-scorer qLogNEI of the final batch, the comparison quantity |
| `own_acq_value` | value on the optimizer's own 128-sample qLogNEI |
| `ga_acq_value` | GA only: value before the refine |
| `wall_time_s` | optimizer wall time (GA: `ga_wall_time_s` + `refine_wall_time_s`) |
| `candidates` | `6 × d` batch in the optimizer layout `[cont(4) \| cat \| form_id \| is_salt \| sol \| base]` |
| `decoded` | per row: workbook catalyst / solvent / base ids, `fa_ratio`, `loading`, `temperature`, `time` in raw units |
| `meta` | host, SLURM ids, torch threads and versions |

`data/` (bundle, results, queue, logs, `calibration.json`) is git-ignored.

## Commands (repo root, Mechthild)

```bash
source .venv/bin/activate
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant prepare
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant smoke
```

### 1. Calibrate

```bash
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.submit_mechthild calibrate
```

Generated script:

```bash
#!/bin/bash
#SBATCH -J afopt_r6_cal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o pablo/mechthild_runs/run_6_af_optimizer_af_value/data/logs/calibrate_%j.out
#SBATCH -e pablo/mechthild_runs/run_6_af_optimizer_af_value/data/logs/calibrate_%j.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
source .venv/bin/activate

echo "[calibrate] node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant calibrate
```

The job ends with a `GA_N_EVALS` / `PG_TIERS` block for `config.py` (also in `data/calibration.json`).

### 2. Queue and submit

One array per group of tiers that share partition, time limit and memory (`config.SLURM_BY_TIER`); `submit` raises on a mix. Each task runs `scheduler worker --single --tiers …` and claims only pending jobs of its tiers. `init` marks every job with a result file as done and pends the rest.

```bash
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler init
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.submit_mechthild submit --tiers 5
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.submit_mechthild submit --tiers 6
```

Tiers 0–4 go out together with `--tiers 0 1 2 3 4`. Generated script for tier 6 (`--dry-run` prints it without submitting; tier 5 differs in `-J afopt_r6_t5`, `--mem=24G`, `--time=01:45:00`, `--partition=short` and `--tiers 5`):

```bash
#!/bin/bash
#SBATCH -J afopt_r6_t6
#SBATCH --array=1-6%30
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=03:30:00
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o pablo/mechthild_runs/run_6_af_optimizer_af_value/data/logs/worker_%A_%a.out
#SBATCH -e pablo/mechthild_runs/run_6_af_optimizer_af_value/data/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
source .venv/bin/activate

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler worker --single --tiers 6
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
```

### 3. Monitor

```bash
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler status
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler submitted
squeue -u $USER -n afopt_r6_t5,afopt_r6_t6
```

### 4. Failed tasks

A crashed task leaves its lock in `running/` and no result file; its traceback is in `data/logs/worker_<job>_<task>.err`. Once `squeue` shows no `afopt_r6_*` task, requeue and resubmit the tiers that failed:

```bash
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler requeue
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.submit_mechthild submit --tiers 6
```

### 5. Verify and plot

```bash
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.verify_results [--partial]
python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.plots.plot_results
```

`verify_results` checks the unit, q, the (2, 2, 2) partition, the pinned time, bounds, that every row decodes to available agents with a zero base slice for non-salt catalysts, finite values, and wall time within 30 percent of the tier target; `--partial` accepts missing results. `plot_results` writes `plots/af_vs_walltime.png` (common score against wall time, every run plus per-tier medians) and prints the per-tier median table with the pool-greedy minus GA difference.
