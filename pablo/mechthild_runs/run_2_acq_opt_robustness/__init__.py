"""Run 2: Acquisition function optimization robustness benchmark.

Tests batch generation robustness of pablo/main.py by sweeping GA settings
(n_evals_ga × pop_size × repeat, 300 independent HPC jobs with the values in
``config.py``).
"""
