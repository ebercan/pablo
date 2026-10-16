"""Featurization x kernel x model cross-validation study for the NHC reaction.

    paths.py           canonical filesystem anchors — import paths from here
    core/              config, priors, kernels (KS1-KS4), model factory
    features/          featurization code + the registry and its single CLI
    cross_validation/  the active grid: splits, queue, fit/eval, aggregation
    results_analysis/  reports and plots over scan_aggregate.parquet
    data/              libraries, featurizations, xtb, weights, grid
    results/           the live run's output (gitignored)
    archive/           frozen runs, each with a MANIFEST.json (gitignored)

Predictive metrics live in ``pablo.utils.metrics``. See README.md for the
operator manual.
"""
