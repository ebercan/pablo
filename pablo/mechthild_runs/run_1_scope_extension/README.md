# run_2 Scope Extension (CPU only)

This folder runs the run_2 PlanAhead benchmark sweep with four operating modes:

- `myopic`
- `only_terminal`
- `averaged`
- `layer_averaged`

GPU submission is not used in this run. All workers run on CPU partitions.

## Core behavior

- Combination generation includes all four modes.
- Queue and results are separated by operating mode.
- A run is skipped if its corresponding per-mode result file already exists.
- A single unified submitter is used, selected by `--operating_mode`.

## Mode to folder mapping

Per `config.py`, each operating mode writes to its own queue/results directories:

- `myopic`:
  - queue: `queue_state_myopic/`
  - results: `results_myopic/`
  - partition: `medium`
- `only_terminal`:
  - queue: `queue_state_only_terminal/`
  - results: `results_only_terminal/`
  - partition: `long`
- `averaged`:
  - queue: `queue_state_averaged/`
  - results: `results_averaged/`
  - partition: `long`
- `layer_averaged`:
  - queue: `queue_state_layer_averaged/`
  - results: `results_layer_averaged/`
  - partition: `long`

## 1) Generate combinations

Run from repository root:

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.generate_combinations
```

This writes `pablo/mechthild_runs/run_1_scope_extension/inputs.json`.

## 2) Initialize queues

Initialize each mode once before submission:

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu init --operating_mode myopic
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu init --operating_mode only_terminal
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu init --operating_mode averaged
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu init --operating_mode layer_averaged
```

`init` creates pending lock files only for that mode and marks pre-existing results as done for that mode.

## 3) Submit workers (unified interface)

Status for one mode:

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu status --operating_mode myopic
```

Submit one batch for one mode:

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu submit --operating_mode myopic
python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu submit --operating_mode only_terminal
python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu submit --operating_mode averaged
python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu submit --operating_mode layer_averaged
```

Autoscale one mode:

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu autoscale --operating_mode myopic
```

## 4) Monitor queue state

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu status --operating_mode myopic
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu submitted --operating_mode myopic
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu failed-pending --operating_mode myopic
```

Repeat with any other mode value.

## 5) Run a single combination

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.run_single_job 0
python -m pablo.mechthild_runs.run_1_scope_extension.run_single_job 0 --n_eval_override 2
```

`run_single_job` resolves the mode from `inputs.json` and writes into that mode's results folder.

## Operating modes

There is one scheduler and one submitter; the mode is an argument, not a module:

```bash
python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu worker \
    --queue_dir <dir> --operating_mode myopic
```

The former `myopic_cpu/` and `fantasized_cpu/` wrapper modules did nothing but append
`--operating_mode <const>` to `sys.argv` and have been removed. `myopic_cpu/queue_state/`
is kept — it holds that run's queue artifacts.