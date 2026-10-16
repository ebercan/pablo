# Run 3: Acquisition Function Optimization Robustness

This benchmark tests batch generation robustness of `pablo/main.py` by sweeping GA optimizer settings and recording the generated batch plus acquisition function value across independent runs.

**State**: with 65 total results (initial + 2 batches)

## Overview

Each job runs `pablo.main.generate_batch()` once with patched GA settings. The generated 6-candidate batch and its AF value are saved to a JSON result file.

**GA settings grid:**

| Parameter    | Values                          |
|--------------|---------------------------------|
| `n_evals_ga` | 800000, 1300000, 1700000, 2000000 |
| `pop_size`   | 1000, 2000, 3000, 4000          |
| Repeats      | 50 per combination              |

**Total: 4 × 4 × 50 = 800 jobs.**

All other `main.py` settings are left at defaults except `GA_OPTIMIZATION_REPEATS`:
- `q_batch = 6`
- `GA_OPTIMIZATION_REPEATS = 1` (external repeats in this benchmark already cover repetition)
- `TEMPERATURE_PARTITION = (2, 2, 2)`
- `BATCH_GENERATION_OPTIMIZER = optimize_acqf_mixed_GA` is forced inside run scripts

## Files

| File | Description |
|------|-------------|
| `config.py` | Sweep grid constants |
| `generate_combinations.py` | Builds `inputs.json` with 800 entries |
| `run_single_job.py` | Runs one combination: patches GA settings, forces GA optimizer, calls `generate_batch()`, saves JSON |
| `submit_mechthild.py` | Submits SLURM array job (0-(num_jobs-1)) to Mechthild |
| `test_case.py` | Local test case with minimal GA settings |

## Quick Start (HPC)

```bash
# 1) Generate combinations
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.generate_combinations

# 2) Submit all 800 jobs as a SLURM array
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.submit_mechthild

# 2b) Skip confirmation prompt
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.submit_mechthild --force
```

## Local Test

Run a quick sanity check with minimal GA settings (`n_evals_ga=1000`, `pop_size=50`) before submitting to HPC:

```bash
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.test_case
```

This will:
1. Patch `GA_N_EVALS=1000`, `GA_POP_SIZE=50`, `GA_OPTIMIZATION_REPEATS=1` in `pablo.main`.
2. Force `BATCH_GENERATION_OPTIMIZER=optimize_acqf_mixed_GA` in `pablo.main`.
3. Run `generate_batch()` to produce a 6-candidate batch.
4. Print the generated batch table.
5. Print `TEST PASSED` if 6 candidates were produced, `TEST FAILED` otherwise.

## JSON Result Structure

Each result file is saved as `results/result_{i}.json`. Example:

```json
{
  "combination_index": 0,
  "n_evals_ga": 800000,
  "pop_size": 1000,
  "repeat_index": 0,
  "best_acq_value": 1.2345,
  "acq_values_per_repeat": [1.2345],
  "decoded_batch": [
    {
      "Temperature": 75.0,
      "Formaldehyde_Ratio": 2.500,
      "Catalyst_Loading": 5.000,
      "Time": 20.0,
      "Catalyst_ID": 12,
      "Catalyst_BLD": "1234",
      "Solvent_ID": 3,
      "Solvent_Name": "THF",
      "Base_ID": -1,
      "Base_ID_Workbook": 0,
      "Base_Name": "no base",
      "Posterior_Mean": 45.20,
      "Posterior_Var": 12.300
    },
    {
      "Temperature": 60.0,
      "Formaldehyde_Ratio": 1.800,
      "Catalyst_Loading": 3.200,
      "Time": 20.0,
      "Catalyst_ID": 5,
      "Catalyst_BLD": "5678",
      "Solvent_ID": 1,
      "Solvent_Name": "DCM",
      "Base_ID": 2,
      "Base_ID_Workbook": 2,
      "Base_Name": "K2CO3",
      "Posterior_Mean": 38.50,
      "Posterior_Var": 8.100
    }
  ]
}
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `combination_index` | Job index (0-(num_jobs-1)) |
| `n_evals_ga` | Number of GA evaluations used |
| `pop_size` | GA population size used |
| `repeat_index` | Repeat index within this (n_evals_ga, pop_size) pair (0-(N_REPEATS-1)) |
| `best_acq_value` | Acquisition function value from the single GA optimization run |
| `acq_values_per_repeat` | AF value list (length 1 since `GA_OPTIMIZATION_REPEATS=1`) |
| `decoded_batch` | List of 6 decoded candidate dicts with experimental parameters, IDs, and posterior stats |

## Single Job Testing

Run one specific combination by index:

```bash
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.run_single_job 0
```

## Monitoring

```bash
# Check SLURM job status
squeue -u $USER

# Count completed results
ls results/ | wc -l

# Check for errors
cat logs/worker_*.err | tail -50
```

## Canceling Jobs

```bash
# Cancel all run_3 jobs
scancel -n run3_acq_robustness

# Cancel all your jobs
scancel -u $USER
```

## Output Layout

```text
run_2_acq_opt_robustness/
├── inputs.json          (generated, 800 entries)
├── results/
│   ├── result_0.json
│   ├── result_1.json
│   └── ...              (800 files total)
└── logs/
    ├── worker_*.out
    └── worker_*.err
```
