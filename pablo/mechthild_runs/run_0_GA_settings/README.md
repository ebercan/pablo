# Run 1: GA Settings Benchmark - HPC Submission

This folder contains the run_1 benchmark sweep for GA settings on a mixed 33D search space.

## Overview

This run evaluates:

- 6 benchmarks:
  - LunacekBiRastrigin
  - Gallagher101
  - Michalewicz
  - Ackley
  - Levy
  - Rastrigin
- 12 GA settings:
  - ratios = [100, 125, 150, 175]
  - pop_size = [35, 70, 100]
  - n_evals_ga = ratio * pop_size
- 16 seed pairings:
  - seed_list_1 = [42, 69, 420, 49]
  - seed_list_2 = [19, 31, 41, 7]
  - full cross-product: 4 x 4

Total: 12 x 6 x 16 = 1152 BO runs.

## Search Space Design

- Total dimension: 33
- Continuous dimensions: 3
- Discrete dimensions: 30
- q batch size: 6
- BO iterations: 100
- Starting points: 47
- AF-opt repeats per iteration: 1
- Batch constraint: continuous dim 0 has at most 3 distinct values per q-batch

Dependent discrete construction:

- Source A: 225 choices expanded to 26 dims
- Source B: 14 choices expanded to 2 dims
- Source C: 8 choices expanded to 2 dims
- X_allowed shape: (225 * 14 * 8, 30) = (25200, 30)

## Prerequisites

Generate combinations before running workers:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.generate_combinations
```

This creates [pablo/mechthild_runs/run_0_GA_settings/inputs.json](pablo/mechthild_runs/run_0_GA_settings/inputs.json).

## Files

- [pablo/mechthild_runs/run_0_GA_settings/generate_combinations.py](pablo/mechthild_runs/run_0_GA_settings/generate_combinations.py): build all 1152 combinations
- [pablo/mechthild_runs/run_0_GA_settings/run_single_job.py](pablo/mechthild_runs/run_0_GA_settings/run_single_job.py): execute one indexed job
- [pablo/mechthild_runs/run_0_GA_settings/run_ga_experiment.py](pablo/mechthild_runs/run_0_GA_settings/run_ga_experiment.py): core BO loop and X_allowed construction
- [pablo/mechthild_runs/run_0_GA_settings/mechthild_scheduler.py](pablo/mechthild_runs/run_0_GA_settings/mechthild_scheduler.py): queue init, worker, status
- [pablo/mechthild_runs/run_0_GA_settings/submit_mechthild_dynamic.py](pablo/mechthild_runs/run_0_GA_settings/submit_mechthild_dynamic.py): dynamic SLURM submission

Runtime note:

- X_allowed is regenerated in-memory per run from the benchmark bounds and seed_discrete.
- No shared X_allowed cache files are written.

## Quick Start

```bash
# 1) Generate combinations
python -m pablo.mechthild_runs.run_0_GA_settings.generate_combinations

# 2) Initialize queue (all 1152 jobs)
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler init

# 3) Submit workers based on current cluster capacity (one-shot)
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic

# 3b) Optional: keep polling every 30s and auto-top-up workers
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale

# 3c) Optional: explicit CPU/GPU partitions
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic --partition medium --gpu_partition gpu

# 4) Monitor progress
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler status
squeue -u $USER
```

## Operating Modes

The submitter now supports two explicit operating modes.

1. GPU-first mode (default): fills GPU slots first, then CPU slots.
2. CPU-only mode: ignores GPU availability checks and submits CPU workers only.

You can activate CPU-only mode in two ways.

Alias commands:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic status_cpu_only --partition medium
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic submit_cpu_only --force --partition medium
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale_cpu_only --partition medium --poll_interval_seconds 30
```

Flag with standard commands:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic status --cpu_only --partition medium
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic submit --cpu_only --force --partition medium
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale --cpu_only --partition medium --poll_interval_seconds 30
```

Immediate CPU-only launch command (30s polling):

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale_cpu_only --partition medium --poll_interval_seconds 30
```

## Dynamic Submitter Commands

Use the submitter module for one-shot submit, live autoscaling, and quick capacity checks.

One-shot submission (default mode):

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic
```

Single command to print current usage and available worker slots:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic status
```

This prints compact lines for:

- CPU partition workers in use / available now
- GPU partition workers in use / available now
- CPU/GPU workers pending launch (already queued in SLURM)
- CPU/GPU effective submit slots now (free minus in-flight pending)
- total workers in use / available now
- pending jobs

Detailed parser diagnostics (raw node parsing + per-type free GPU counts):

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic diagnostics
```

This prints:

- parsed per-node GPU totals used by scheduler logic
- free GPU counts by type (for typed gres clusters)
- raw `scontrol`, `sinfo`, and `squeue` context for mismatch debugging

Live autoscaling mode (poll every 30s by default):

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale
```

CPU-only autoscaling alias:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale_cpu_only --partition medium --poll_interval_seconds 30
```

CPU-only autoscaling with flag:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale --cpu_only --partition medium --poll_interval_seconds 30
```

Autoscale behavior:

- polls every 30 seconds while process is alive
- GPU-first mode: computes current free slots from live SLURM snapshots for both GPU and CPU pools
- CPU free-slot estimate is memory-aware and uses per-node `CfgTRES`/`AllocTRES` (CPU + mem) when available
- GPU-first mode: submits to GPU first, then submits remaining demand to CPU (fallback)
- GPU query failure behavior: if GPU slot query fails or returns zero, CPU submissions still happen in the same poll cycle
- in-flight behavior: already pending workers for each pool are subtracted from new submit headroom
- throttle behavior: pending reasons like `AssocMax*`/`QOSMax*` force effective submit slots to zero for that pool
- CPU-only mode: skips GPU checks and submits only CPU workers
- exits automatically when pending queue becomes empty

Adjust poll interval:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic autoscale --poll_interval_seconds 15
```

Skip one-shot confirmation prompt:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic --force
```

## Single Job Testing

Use this before launching large arrays.

Run one combination directly:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.run_single_job 0
```

Run one combination on GPU with a short 2-iteration override:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.run_single_job 42 --device cuda --n_eval_override 2
```

Force strict behavior (no CPU fallback) for debugging:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.run_single_job 42 --device cuda --allow_cpu_fallback 0
```

Or test a one-job queue path:

```bash
# Create only one pending lock
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler init --start 0 --end 1

# Process it with one local worker process
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler worker
```

## Progress in Terminal

Four progress views are available:

1) Per-job BO progress bar (inside each worker)

- run_single_job now prints a single-line dynamic progress bar for the 100 BO iterations.
- Format example:

```text
[123] Ackley (s1=42, s2=19) [##########............] 42/100 | best=0.123456 | pop=70 | evals=8750
```

2) Queue-level progress bar

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler status
```

This prints:

- Progress bar based on done/total lock files
- pending/running/done/total counts

Live monitor (recommended):

```bash
watch -n 5 'python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler status; echo; squeue -u $USER'
```

3) Submitted-run lines (running jobs only)

Print one line per running job with run index, benchmark, GA settings, and current BO iteration:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler submitted
```

Line format:

```text
<idx> | <benchmark> | ratio=<ga_ratio> pop=<pop_size> evals=<n_evals_ga> | dev=<requested->actual> | iter=<current>/<total> [ERROR]
```

- ERROR appears when the run has recorded failure metadata from a prior failed attempt.
- `dev` reports requested and actual device path, including fallback reason when applicable.

4) Failed-pending lines only

Print only runs that are currently in pending because they failed and were requeued:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler failed-pending
```

- One line per failed-pending run.
- If pending jobs are only waiting their turn, this command prints nothing.

## Canceling Jobs

Cancel all your jobs:

```bash
scancel -u $USER
```

Cancel by job id:

```bash
scancel <job_id>
```

Cancel by run_1 job name prefix:

```bash
scancel -n ga_set2_cpu
scancel -n ga_set2_gpu
```

Verify cancellation:

```bash
squeue -u $USER
```

## Troubleshooting

If jobs fail and return to pending:

```bash
# Check queue state
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler status
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler submitted
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler failed-pending
ls pablo/mechthild_runs/run_0_GA_settings/queue_state/pending/

# Run one failing job manually
python -m pablo.mechthild_runs.run_0_GA_settings.run_single_job <idx>

# Check SLURM logs
cat pablo/mechthild_runs/run_0_GA_settings/logs/worker_*.err | tail -50

# Check accounting status
sacct --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed
```

Known issues (now fixed):

- **Device mismatch crash**: benchmark evaluation functions create intermediate tensors
  on CPU without specifying device. The BO experiment now evaluates benchmarks on CPU
  and moves results to the resolved device. This was the root cause of intermittent
  ERROR terminations when GPU jobs fell back to CPU.
- **Lock race noise**: the atomic lock claiming (`mv -n`) produces expected stderr
  when another worker claims first. This stderr is now suppressed.
- **P100 incompatibility**: torch `cu128` requires sm_70+. P100 nodes (sm_60) are
  excluded from GPU job scripts via `#SBATCH --exclude=gpu[01-05]`.
- **venv resolution**: SLURM scripts now use `venv/` only (no `.venv` fallback).

Status note:

- scheduler status reads lock-file counts from queue_state
- squeue reads active SLURM jobs
- these can differ briefly during worker startup/shutdown/cancel windows
- scheduler `status/submitted/failed-pending` are read-only by default (no automatic stale requeue)
- to explicitly reconcile stale running locks, use `--auto_requeue_stale`

Explicit stale-reconcile examples:

```bash
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler status --auto_requeue_stale
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler submitted --auto_requeue_stale
python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler failed-pending --auto_requeue_stale
```

## Configuration

Edit [pablo/mechthild_runs/run_0_GA_settings/config.py](pablo/mechthild_runs/run_0_GA_settings/config.py):

```python
DEVICE_CHOICES = ("cpu", "cuda", "auto")
DEFAULT_DEVICE_REQUEST = "cpu"
ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES = True

# P100 nodes (gpu01-gpu05) have compute capability sm_60, below the
# minimum sm_70 supported by torch cu128. GPU jobs are excluded from
# these nodes and route to A100 (gpua*, sm_80) and H100 (gpuh*, sm_90).
GPU_EXCLUDE_NODES = "gpu[01-05]"
```

Edit [pablo/mechthild_runs/run_0_GA_settings/submit_mechthild_dynamic.py](pablo/mechthild_runs/run_0_GA_settings/submit_mechthild_dynamic.py):

```python
CPU_PARTITION = "medium"
GPU_PARTITION = "gpu"
CPUS_PER_TASK = 1
GPUS_PER_TASK = 1
GPU_GRES_REQUEST = "gpu:1"
MEMORY = "16G"
TIME_LIMIT = "24:00:00"
JOB_NAME_PREFIX = "ga_set2"
```

GPU scripts automatically include `#SBATCH --exclude=gpu[01-05]`
to avoid P100 nodes with incompatible CUDA compute capability.

## Test Suite

Two test tiers validate the pipeline before HPC submission:

1. **CPU-only tests**: verify the full pipeline works on CPU.
2. **GPU-prioritized tests**: verify GPU-first with CPU fallback behavior.

Run all tests in verbose mode:

```bash
python -m pytest tests/test_run_1_cpu_only.py tests/test_run_1_gpu_prioritized.py -v --tb=short
```

Run all tests in short (summary) mode:

```bash
python -m pytest tests/test_run_1_cpu_only.py tests/test_run_1_gpu_prioritized.py --run1-mode=short
```

Short mode output:

```text
run_1 Test Suite Summary
========================================
CPU-only:        8/8  PASSED ✅
GPU-prioritized: 6/6  PASSED ✅
Overall:         SAFE TO SUBMIT ✅
```

Run only one tier:

```bash
# CPU-only
python -m pytest tests/test_run_1_cpu_only.py -v

# GPU-prioritized (uses CPU fallback when CUDA unavailable)
python -m pytest tests/test_run_1_gpu_prioritized.py -v
```

Run the full regression suite (all run_1 tests):

```bash
python -m pytest tests/test_run_1_*.py -v --tb=short
```

### Test files

- `tests/test_run_1_cpu_only.py` — device policy, tensor consistency, BO smoke, lock claiming, worker pipeline
- `tests/test_run_1_gpu_prioritized.py` — cuda fallback, device consistency, BO smoke, SLURM script generation
- `tests/test_run_1_ga_settings.py` — tensor shapes, combination counts, BO experiment contracts
- `tests/test_run_1_gpu_scheduler_logic.py` — scheduler command forwarding, device routing
- `tests/test_run_1_run_single_job.py` — run_single_job execution wiring
- `tests/test_run_1_scheduler_reporting.py` — state file parsing, reporting lines
- `tests/test_run_1_scheduler_stale_reconcile.py` — stale lock detection and requeue
- `tests/test_run_1_submit_dynamic.py` — capacity computation, autoscale, script generation

## Production Go/No-Go Gate

Before starting large submission/autoscale runs:

1. Run the test suite in short mode — all tiers must show PASSED.
2. Verify GPU jobs land on A100/H100 nodes (not P100).
3. Run one single job manually to confirm end-to-end execution.

## Output Layout

Generated files (runtime only, not tracked):

```text
run_0_GA_settings/
├── inputs.json
├── queue_state/
│   ├── pending/job_*.lock
│   ├── running/job_*.lock
│   ├── done/job_*.lock
│   └── run_state/job_*.json
├── results/
│   ├── result_0.json
│   └── ...
└── logs/
    ├── worker_*.out
    └── worker_*.err
```

## Cleanup

After completion:

```bash
# 1) Cancel workers first
scancel -u $USER

# 2) Confirm no jobs remain
squeue -u $USER

# 3) Remove queue locks
rm -rf pablo/mechthild_runs/run_0_GA_settings/queue_state/

# 4) Remove logs/results when done collecting
rm -rf pablo/mechthild_runs/run_0_GA_settings/logs/
rm -rf pablo/mechthild_runs/run_0_GA_settings/results/

# 5) Remove generated combinations file if desired
rm pablo/mechthild_runs/run_0_GA_settings/inputs.json
```
