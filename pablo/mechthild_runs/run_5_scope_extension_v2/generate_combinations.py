"""Generate benchmark × seed × operating-mode combinations for run_5.

Run as a script, it writes every combination, keyed by its string index, to
``inputs.json`` next to this file and prints the count per operating mode.

Usage:
    python -m pablo.mechthild_runs.run_5_scope_extension_v2.generate_combinations
"""

from __future__ import annotations

import json
from pathlib import Path

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    ALL_OPERATING_MODES,
    BENCHMARK_CLASS_MAP,
    DEFAULT_BATCH_CONSTRAINTS,
    INPUTS_PATH,
    get_strategy_for_operating_mode,
    LOOKAHEAD_HORIZON,
    N_BO_BATCHES,
    NUM_FANTASIES,
    Q_BATCH_SIZE,
    Q_EXPAND_SIZE,
    SEEDS,
    SOURCE_A_AVAILABILITY_DELAY,
)


def generate_all_combinations() -> list[dict[str, object]]:
    """Generate all benchmark x seed x operating-mode combinations.

    Returns:
        List of combination dictionaries, one per job.
    """
    combinations: list[dict[str, object]] = []
    for benchmark in BENCHMARK_CLASS_MAP:
        for seed in SEEDS:
            for operating_mode in ALL_OPERATING_MODES:
                combinations.append(
                    {
                        "optimizer": "pool_greedy",
                        "space": "nhc",
                        "strategy": get_strategy_for_operating_mode(operating_mode),
                        "operating_mode": operating_mode,
                        "benchmark": benchmark,
                        "seed": seed,
                        "source_a_availability_delay": SOURCE_A_AVAILABILITY_DELAY,
                        "lookahead_horizon": LOOKAHEAD_HORIZON,
                        "num_fantasies": list(NUM_FANTASIES),
                        "q": Q_BATCH_SIZE,
                        "q_expand": Q_EXPAND_SIZE,
                        "n_eval": N_BO_BATCHES,
                        "batch_constraints": {
                            str(k): list(v) for k, v in DEFAULT_BATCH_CONSTRAINTS.items()
                        },
                        "label": f"{benchmark}|seed={seed}|mode={operating_mode}",
                    }
                )
    return combinations


def write_inputs_json(combinations: list[dict[str, object]], output_path: Path) -> None:
    """Write combinations to a JSON file as ``{"0": combo_0, "1": combo_1, ...}``.

    Args:
        combinations: Combination dicts from ``generate_all_combinations``.
        output_path: Destination JSON path; overwritten if it exists.
    """
    indexed = {str(i): combinations[i] for i in range(len(combinations))}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(indexed, handle, indent=2)


if __name__ == "__main__":
    combos = generate_all_combinations()
    by_mode = {mode: 0 for mode in ALL_OPERATING_MODES}
    for combo in combos:
        by_mode[str(combo["operating_mode"])] += 1
    write_inputs_json(combos, INPUTS_PATH)
    print(f"Wrote {len(combos)} combinations to {INPUTS_PATH}")
    for mode, count in by_mode.items():
        print(f"  {mode}: {count}")
