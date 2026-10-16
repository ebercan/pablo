"""Execute one indexed run_2 AF optimization robustness job.

Usage:
    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.run_single_job <combination_index>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pablo.main as main_module


from pablo.mechthild_runs.run_2_acq_opt_robustness.config import (
    INPUTS_FILENAME,
    RESULTS_DIRNAME,
)

_last_acq_results: dict = {}


def _make_acq_interceptor():
    """Wrap ``pablo.main._optimize_with_repeats`` to record its AF values.

    Returns:
        A drop-in replacement for ``_optimize_with_repeats`` that returns the
        original result unchanged and stores ``best_acq_value`` (float) and
        ``acq_values_per_repeat`` (list of float) in ``_last_acq_results``.
    """
    original_fn = main_module._optimize_with_repeats

    def patched_optimize(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        _last_acq_results["best_acq_value"] = float(result[1])
        _last_acq_results["acq_values_per_repeat"] = [float(v) for v in result[2]]
        return result

    return patched_optimize


def main(combination_index: int) -> None:
    """Run one GA setting combination and persist its result as JSON.

    Sets ``pablo.main.GA_SETTINGS = (n_evals_ga, pop_size)`` from the combination
    entry and ``pablo.main.GA_OPTIMIZATION_REPEATS = 1``, intercepts AF values from
    ``_optimize_with_repeats``, then calls ``generate_batch(q_batch=6,
    max_temperature_values=3)`` and writes the decoded batch with the AF values to
    ``result_<combination_index>.json``. Exits with status 0 without running if that
    file already exists.

    Args:
        combination_index: Index in inputs.json.

    Raises:
        ValueError: If ``combination_index`` is not a key in inputs.json.
    """
    base_dir = Path(__file__).parent
    inputs_path = base_dir / INPUTS_FILENAME
    results_dir = base_dir / RESULTS_DIRNAME

    output_path = results_dir / f"result_{combination_index}.json"
    if output_path.exists():
        print(f"Result {output_path.name} already exists. Skipping execution.")
        sys.exit(0)

    with open(inputs_path, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)

    key = str(combination_index)
    if key not in inputs:
        raise ValueError(
            f"Combination index {combination_index} not found in {inputs_path}"
        )

    comb = inputs[key]
    n_evals_ga = int(comb["n_evals_ga"])
    pop_size = int(comb["pop_size"])
    repeat_index = int(comb["repeat_index"])

    print(f"Running combination {combination_index}: {comb['label']}")

    # main.py reads the pair GA_SETTINGS = (n_evals_ga, pop_size). Assigning
    # GA_N_EVALS / GA_POP_SIZE attached two attributes nothing reads, so every cell
    # silently ran at the 20000/500 default and the swept values were ignored.
    main_module.GA_SETTINGS = (n_evals_ga, pop_size)
    main_module.GA_OPTIMIZATION_REPEATS = 1
    print(
        f"  Patched GA_SETTINGS=({n_evals_ga}, {pop_size}), "
        "GA_OPTIMIZATION_REPEATS=1"
    )

    _last_acq_results.clear()
    main_module._optimize_with_repeats = _make_acq_interceptor()

    candidates, decoded = main_module.generate_batch(
        q_batch=6,
        max_temperature_values=3,
    )
    main_module.print_candidates(decoded)

    results_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "combination_index": combination_index,
        "n_evals_ga": n_evals_ga,
        "pop_size": pop_size,
        "repeat_index": repeat_index,
        "best_acq_value": _last_acq_results.get("best_acq_value"),
        "acq_values_per_repeat": _last_acq_results.get("acq_values_per_repeat"),
        "decoded_batch": decoded,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)

    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one run_2 AF optimization robustness combination"
    )
    parser.add_argument("combination_index", type=int, help="Index from inputs.json")
    args = parser.parse_args()
    main(args.combination_index)
