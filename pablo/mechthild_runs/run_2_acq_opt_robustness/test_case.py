"""Local test case for run_2 acquisition-function optimization robustness.

Runs generate_batch() with minimal GA settings (n_evals_ga=1000, pop_size=50)
to verify the full pipeline works before submitting to HPC.

Usage::

    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.test_case
"""

from __future__ import annotations

import pablo.main as main_module



def run_test() -> None:
    """Run a minimal batch generation test and print pass/fail."""
    n_evals_ga = 1000
    pop_size = 50

    print("=" * 70)
    print("RUN 3 LOCAL TEST")
    print(f"  n_evals_ga = {n_evals_ga}")
    print(f"  pop_size   = {pop_size}")
    print("=" * 70)

    # main.py reads the pair GA_SETTINGS = (n_evals_ga, pop_size). Assigning
    # GA_N_EVALS / GA_POP_SIZE attached two attributes nothing reads, so every cell
    # silently ran at the 20000/500 default and the swept values were ignored.
    main_module.GA_SETTINGS = (n_evals_ga, pop_size)
    main_module.GA_OPTIMIZATION_REPEATS = 1
    print(
        f"Patched GA_SETTINGS=({n_evals_ga}, {pop_size}), "
        "GA_OPTIMIZATION_REPEATS=1"
    )

    candidates, decoded = main_module.generate_batch(
        q_batch=6,
        max_temperature_values=3,
    )
    main_module.print_candidates(decoded)

    batch_size = candidates.shape[0]
    n_decoded = len(decoded)

    print("\n" + "=" * 70)
    if batch_size == 6 and n_decoded == 6:
        print("TEST PASSED")
        print(f"  Generated batch of {batch_size} candidates.")
        print(f"  Decoded {n_decoded} candidates successfully.")
    else:
        print("TEST FAILED")
        print(
            f"  Expected 6 candidates, got {batch_size} (tensor), {n_decoded} (decoded)."
        )
    print("=" * 70)


if __name__ == "__main__":
    run_test()
