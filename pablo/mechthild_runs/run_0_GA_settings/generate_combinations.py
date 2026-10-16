"""Generate benchmark combinations for the run_0 GA settings experiments.

Writes the indexed combinations to ``inputs.json`` next to this file.

Usage:
    python -m pablo.mechthild_runs.run_0_GA_settings.generate_combinations
"""

from __future__ import annotations

import json
from pathlib import Path

from pablo.mechthild_runs.run_0_GA_settings.config import (
    BENCHMARK_CLASS_MAP,
    DEFAULT_BATCH_CONSTRAINTS,
    GA_RATIOS,
    INPUTS_FILENAME,
    N_BO_BATCHES,
    N_OPTIMIZE_ACQF_REPETITIONS,
    N_STARTING_POINTS,
    POP_SIZES,
    Q_BATCH_SIZE,
    SEED_LIST_1,
    SEED_LIST_2,
)


def generate_ga_settings() -> list[dict[str, int]]:
    """Create the 12 GA settings from pop size and ratio grids.

    Returns:
        List of setting dictionaries with fields:
        setting_idx, pop_size, ga_ratio, n_evals_ga.
    """
    settings: list[dict[str, int]] = []
    setting_idx = 0

    for pop_size in POP_SIZES:
        for ga_ratio in GA_RATIOS:
            settings.append(
                {
                    "setting_idx": setting_idx,
                    "pop_size": pop_size,
                    "ga_ratio": ga_ratio,
                    "n_evals_ga": pop_size * ga_ratio,
                }
            )
            setting_idx += 1

    return settings


def generate_all_combinations() -> list[dict[str, object]]:
    """Generate all benchmark x seed x GA-setting combinations.

    Returns:
        List of combination dictionaries.
    """
    settings = generate_ga_settings()
    combinations: list[dict[str, object]] = []

    for setting in settings:
        for benchmark in BENCHMARK_CLASS_MAP:
            for seed_discrete in SEED_LIST_1:
                for seed_starting in SEED_LIST_2:
                    combinations.append(
                        {
                            "optimizer": "optimize_acqf_mixed_GA",
                            "benchmark": benchmark,
                            "seed_discrete": seed_discrete,
                            "seed_starting": seed_starting,
                            "setting_idx": setting["setting_idx"],
                            "ga_ratio": setting["ga_ratio"],
                            "pop_size": setting["pop_size"],
                            "n_evals_ga": setting["n_evals_ga"],
                            "q": Q_BATCH_SIZE,
                            "n_eval": N_BO_BATCHES,
                            "n_optimize_acqf": N_OPTIMIZE_ACQF_REPETITIONS,
                            "n_starting": N_STARTING_POINTS,
                            "batch_constraints": dict(DEFAULT_BATCH_CONSTRAINTS),
                            "label": (
                                f"{benchmark}|s1={seed_discrete}|s2={seed_starting}|"
                                f"ratio={setting['ga_ratio']}|pop={setting['pop_size']}"
                            ),
                        }
                    )

    return combinations


def write_inputs_json(combinations: list[dict[str, object]], output_path: Path) -> None:
    """Write indexed combinations to inputs.json.

    Args:
        combinations: Combination list.
        output_path: JSON target path.
    """
    indexed = {str(i): combinations[i] for i in range(len(combinations))}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(indexed, handle, indent=2)


if __name__ == "__main__":
    combinations = generate_all_combinations()

    output_path = Path(__file__).parent / INPUTS_FILENAME
    write_inputs_json(combinations, output_path)

    print(f"Generated {len(combinations)} combinations")
    print(f"  GA settings: {len(generate_ga_settings())}")
    print(f"  Benchmarks: {len(BENCHMARK_CLASS_MAP)}")
    print(f"  seed_list_1 x seed_list_2: {len(SEED_LIST_1)} x {len(SEED_LIST_2)}")
    print(f"  Output: {output_path}")
