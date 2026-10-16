# run_5 — Scope expansion on the NHC space: fantasized expander against random and myopic baselines

Benchmarks which locked catalyst to order next when an order takes four batches
to arrive. The search space, the starting data and the surrogate are those of the
NHC campaign; only the yield is synthetic, so the best catalyst is known.

## Setting

| item | value |
|---|---|
| point layout | `[cont (4) │ cat (16) │ solv (4) │ base (4)]`, d = 28 (`nhc_problem.py`) |
| continuous | formaldehyde eq [1, 3], loading [0.5, 10], temperature [25, 100] °C, time [1, 24] h; time pinned at 20 h |
| catalyst | `grover_pca`, 16 dims × 161 library catalysts |
| solvent | `Hansen` (δD, δP, δH, logP), 4 dims × 14 |
| base | `grover_pca`, 4 dims × 8; the block is zero for the 10 catalysts that need no external base |
| starting set | every recorded row of `nhc_doe.xlsx` (69, replicates included), read through `build_experiments_table` |
| available at start | the catalysts `nhc_doe.xlsx` marks available for the next batch (22); the other 139 are locked |
| batch | q = 6, temperature partition (2, 2, 2) |
| BO iterations | 100 (69 + 600 rows at the end) |
| expander | orders 1 locked catalyst per iteration; it is dead for the next 4 batches and usable in the 5th |
| fantasy tree | horizon 4 (depth j = the batch j + 1 ahead), fantasies [12, 4, 2, 1] |
| benchmarks | Ackley, Gallagher101, LunacekBiRastrigin (`benchmark_problems/synthetic.py`), d = 28 |
| seeds | 20 paired seeds per benchmark; 3 × 20 × 5 modes = 300 jobs |

### Surrogate

The featurization-scan model `grover_pca / Hansen / grover_pca / matern52 / probit / P2 / KS1 / constant`,
built from the scan's own code (`core.kernels.make_kernel_from_config`,
`core.models.make_singletask_gp`) on the 28-column layout and refit every
iteration with the scan's 5-restart L-BFGS (`fit_eval._fit_multistart`):

- KS1: one output scale over the product of Matérn-5/2 block kernels (catalyst, solvent, base, continuous);
- P2: full ARD (28 lengthscales), LogNormal lengthscale priors located at `log √D_block`;
- the measured yield enters on the probit scale; constant mean; `Normalize` inputs, `Standardize` outcomes;
- noise prior InverseGamma from the replicate evidence of yield on the probit scale (`NOISE_EVIDENCE[("Y", "probit")]`).

Every acquisition function runs on that probit scale.

### Synthetic yield

Per (benchmark, seed), `nhc_problem.SyntheticYield`:

1. column j of a point is mapped from its range in the space (`u ∈ [0, 1]`) to the
   benchmark coordinate `b_lo + (b_hi − b_lo)·(o_j + 0.5·u)`, with `u` reversed for a random
   half of the columns and `o_j ~ U(0.125, 0.375)`, so Ackley's optimum lies in the middle half
   of every column's range; Gallagher101 and LunacekBiRastrigin also take the seed for their
   rotations;
2. yield = `(f − f_min)/(f_max − f_min)`, the extremes over the attainable space (all
   17,052 library (cat, solv, base) rows × 128 Sobol conditions at 20 h, best and worst 16
   refined by L-BFGS-B), so the best attainable point has yield 1;
3. measured yield = yield + N(0, 0.0105²), censored to [0, 1]. 0.0105 is the pooled replicate
   SD of the raw yield fraction (5 replicate groups, 9 dof).

`best_fs` and the regret are computed on the noise-free yield; the model sees only measured yields.
The same seed gives the same landscape, start-set measurements and random-number stream in every
mode: the noise, the random expander's draw and the pool seeds of the batch and the picks depend on
the seed, the benchmark and the iteration only. The modes differ only through the catalysts they order.

## Expanders and the cross-scored record

Every iteration, in every mode, after the batch is measured and the model refit:

1. a fantasy tree is built on the current model: depth j simulates batch t + j + 1 over the
   catalysts usable by then (available plus those ordered earlier and unlocked by then);
2. four criteria each pick the best locked point (catalyst, solvent, base, conditions):
   - `myopic` — qLogEI of the current model;
   - `only_terminal` — mean qLogEI over the 96 leaf models;
   - `averaged` — mean over all 252 tree models;
   - `layer_averaged` — mean over depths of the per-depth means;
3. `random` picks a locked catalyst uniformly; its point is the best one for that catalyst under `myopic`;
4. each of the five picked points is scored under each of the four criteria
   (`expansion_scores[t][pick][criterion]`, the criterion's forward value: Monte Carlo qLogEI per node);
5. the operating mode's pick is ordered.

The three fantasized criteria share one tree; every pick uses `optimize_acqf_mixed_pool_greedy`.
An order placed in the last 5 iterations (t ≥ 95) is never usable within the 100 batches; it is
still recorded.

## Inner optimizer

`optimize_acqf_mixed_pool_greedy` (`pablo/optimize.py`) builds every batch from a candidate pool
(random rows with Sobol conditions, the parent node's batch, perturbations of the best training
points), refines the best single points, adds rows greedily under the (2, 2, 2) partition and
polishes the free conditions. Inside the tree every node is the root conditioned on its path, so
one root cache (`GPPoolPosteriorCache`) serves all 157 node batches.

| role (`config.py`) | pool | refine | swaps | polish | QMC |
|---|---|---|---|---|---|
| batch | `BATCH_POOL_SIZE` = 4096 | 32 | 1 | 50 | 1024 |
| tree node | `TREE_POOL_SIZE` = 4096 | 32 | 0 | 20 | `TREE_QMC_SAMPLES` = 128 |
| expansion pick | `EXPANSION_POOL_SIZE` = 4096 | `EXPANSION_REFINE` = 0 | 0 | 0 | 128 (tree) / 512 (myopic) |

The tree releases the GPyTorch prediction caches of a non-root node once its children exist, and the
three tree criteria release each member's caches after evaluating it
(`fantasized_criterion(..., release_member_caches=True)`), so at most one tree model holds its dense
`n × n` caches at a time. Without this a 252-model tree at 669 rows would hold about 20 GB.

`inner_optimizer_harness.py` measures inner optimizers against the GA on fantasy nodes of the
latent-table setting in `latent_benchmark.py` (26 dims, random source tables).
`real_data_optimizer_benchmark.py` compares pool-greedy with the production GA on the NHC dataset.

## Runtime

One iteration on a single CPU, local (Ryzen 5 4600H, one thread), Ackley, at four training-set sizes;
a job's iterations run from 75 to 669 rows:

| rows | GP fit | batch | fantasy tree | five picks | cross-scores | total | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 69 | 2.7 s | 2.1 s | 172 s | 23 s | 6 s | 206 s | 1.0 GB |
| 243 | 8.4 s | 2.8 s | 235 s | 31 s | 10 s | 287 s | 1.3 GB |
| 453 | 27.7 s | 3.7 s | 348 s | 49 s | 27 s | 455 s | 2.3 GB |
| 663 | 37.6 s | 4.0 s | 443 s | 76 s | 53 s | 614 s | 3.5 GB |

Every mode builds the tree, so every job costs the same: interpolated over the 100 iterations,
≈ 11.0 h locally, plus up to 2 min for the attainable range of Gallagher101. On Mechthild, with 16
workers per node, the first iteration took 253–424 s (mean 355 s over 68 jobs), 1.2–2.1× the local
time, so a job takes ≈ 13.5–23 h and the sweep ≈ 5,700 CPU-h. The tail is close to the 1-day cap of
`medium`, so the workers run on `long` with a 2-day limit.
`SLURM_MEMORY` = 8 GB leaves 2.3× headroom over the peak and lets a 192 GB node run 16 workers;
the non-GPU nodes hold all 300 workers at once, which finishes the sweep in ≈ 23 h.

## Result fields worth knowing

Per iteration: `best_fs` (best noise-free yield so far), `best_observed`, `candidates_list`,
`y_observed_list`, `y_true_list`, `selected_source_a_indices` (`cat_table` row; ids in
`catalyst_ids`), `expansion_picks` (pick → catalyst index, id and point), `expansion_scores`
(pick → criterion → value), `released_source_a_by_iteration`, `n_candidates_using_released`,
`wall_times_acqf_opt`, `wall_times_model_fit`, `wall_times_expander`,
`wall_times_expander_parts` (tree / picks / scores). Per run: `landscape` (`f_min`, `f_max`,
`x_max`, `noise_sd`, flips, offsets), `start_y_observed`, `start_y_true`, `global_max` = 1.

## Commands (repo root, venv active)

Generate combinations (writes `inputs.json`, 300 jobs):

```bash
python -m pablo.mechthild_runs.run_5_scope_extension_v2.generate_combinations
```

Run one job locally, shortened (the result is a smoke output; `init` and `run_single_job` reject it as
a finished job):

```bash
python -m pablo.mechthild_runs.run_5_scope_extension_v2.run_single_job 4 --n_eval_override 1 --num_fantasies_override 2 1 1 1
```

Initialize one queue per mode, then submit one array per mode. `init` raises on a result file whose
stored combination is not the current `inputs.json` entry or that comes from a shortened run, and on
a `done/` lock without a result file; move such files and queue folders out first.

```bash
for m in random myopic only_terminal averaged layer_averaged; do
  python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu init --operating_mode $m
  python -m pablo.mechthild_runs.run_5_scope_extension_v2.submit_mechthild_cpu --operating_mode $m --force
done
```

Each array task runs one job and exits. A failed job returns to `pending/` with status
`pending_failed`; a worker killed by the time limit or its node leaves the job in `running/`.
Monitor, return stale jobs to pending, resubmit for the jobs left without a worker:

```bash
python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu status --operating_mode myopic
python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu submitted --operating_mode myopic
python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu failed-pending --operating_mode myopic
python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu requeue-stale --operating_mode myopic
python -m pablo.mechthild_runs.run_5_scope_extension_v2.submit_mechthild_cpu --operating_mode myopic --force
```

Verify (all modes by default, exit status 1 on any failed file) and plot:

```bash
python -m pablo.mechthild_runs.run_5_scope_extension_v2.verify_results
python -m pablo.mechthild_runs.run_5_scope_extension_v2.plots.plot_results
```

The submitter writes this SLURM array script (`.submit_dynamic.sh`; `--dry_run` prints it without
submitting), with one task per job not covered by a queued worker:

```bash
#!/bin/bash
#SBATCH -J r5_<mode>
#SBATCH --array=1-<tasks>%512
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --partition=long
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o pablo/mechthild_runs/run_5_scope_extension_v2/logs/worker_%A_%a.out
#SBATCH -e pablo/mechthild_runs/run_5_scope_extension_v2/logs/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Array task $SLURM_ARRAY_TASK_ID (job $SLURM_JOB_ID) on $SLURMD_NODENAME"
python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu worker \
    --operating_mode <mode> --queue_dir "<run folder>/queue_state_<mode>"
```

Results land in `results_<mode>/result_<idx>.json`, written through a temporary file and a rename;
a job is skipped when its result file exists. Each job logs one line per iteration (best yield, fit,
batch and expander seconds) to its worker's `.out` file.
