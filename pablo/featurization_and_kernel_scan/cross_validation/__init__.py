"""The active cross-validation grid: enumeration, execution, and aggregation.

    cv_splits.py         the six split blocks, hashed into every result
    generate_inputs.py   enumerate one run's grid (--run) into its inputs.json:
                         data/grid/ for elimination, runs/<name>/ otherwise
    scheduler.py         file-based queue worker and run-state reporting
    submit_mechthild.py  SLURM array submission and autoscaling
    fit_eval.py          fit one GP on one fold and score it
    run_state.py         per-worker progress files
    concat_results.py    merge a run's per-cell result JSONs into its scan_aggregate.parquet
    timing_probe.py      time each grid level on real folds for the scheduler's cost model

Everything here reads its paths from ``paths`` and its axes from
``core.config``; nothing here builds featurizations.
"""
