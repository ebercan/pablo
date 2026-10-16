"""Generate benchmark combinations for the run_1 scope-extension PlanAhead experiments.

Writes ``inputs.json`` next to this script. Run from the repo root:

    python -m pablo.mechthild_runs.run_1_scope_extension.generate_combinations
"""

from __future__ import annotations

import json
from pathlib import Path

from pablo.mechthild_runs.run_1_scope_extension.config import (
    ALL_OPERATING_MODES,
    BENCHMARK_CLASS_MAP,
    DEFAULT_BATCH_CONSTRAINTS,
    FANTASIZED_OPERATING_MODES,
    get_strategy_for_operating_mode,
    INITIAL_AVAILABLE_SOURCE_A_COUNT,
    INITIAL_AVAILABLE_SOURCE_A_SEEDS,
    INPUTS_FILENAME,
    LOOKAHEAD_HORIZON,
    MYOPIC_OPERATING_MODE,
    N_BO_BATCHES,
    N_EVALS_GA,
    N_OPTIMIZE_ACQF_REPETITIONS,
    N_STARTING_POINTS,
    NUM_FANTASIES,
    POP_SIZE,
    Q_BATCH_SIZE,
    Q_EXPAND_SIZE,
    SEED_LIST_1,
    SEED_LIST_2,
    SOURCE_A_AVAILABILITY_DELAY,
)


def _validate_mode_strategy_pair(strategy: str, operating_mode: str) -> None:
    """Validate strategy and operating_mode consistency.

    Args:
        strategy: Strategy assigned to one combination.
        operating_mode: Operating mode assigned to one combination.

    Raises:
        ValueError: If strategy and operating_mode are inconsistent.
    """
    expected_strategy = get_strategy_for_operating_mode(operating_mode)
    if strategy != expected_strategy:
        raise ValueError(
            "Inconsistent strategy/operating_mode pair: "
            f"strategy={strategy!r}, operating_mode={operating_mode!r}, "
            f"expected_strategy={expected_strategy!r}"
        )


def generate_all_combinations() -> list[dict[str, object]]:
    """Generate all benchmark x seed x operating-mode combinations.

    Returns:
        List of combination dictionaries.
    """
    combinations: list[dict[str, object]] = []

    for benchmark in BENCHMARK_CLASS_MAP:
        for seed_discrete in SEED_LIST_1:
            for seed_starting in SEED_LIST_2:
                for seed_initial_available in INITIAL_AVAILABLE_SOURCE_A_SEEDS:
                    for operating_mode in ALL_OPERATING_MODES:
                        strategy = get_strategy_for_operating_mode(operating_mode)
                        _validate_mode_strategy_pair(
                            strategy=strategy,
                            operating_mode=operating_mode,
                        )
                        combinations.append(
                            {
                                "optimizer": "plan_ahead",
                                "strategy": strategy,
                                "operating_mode": operating_mode,
                                "benchmark": benchmark,
                                "seed_discrete": seed_discrete,
                                "seed_starting": seed_starting,
                                "seed_initial_available": seed_initial_available,
                                "initial_available_source_a_count": INITIAL_AVAILABLE_SOURCE_A_COUNT,
                                "source_a_availability_delay": SOURCE_A_AVAILABILITY_DELAY,
                                "lookahead_horizon": LOOKAHEAD_HORIZON,
                                "num_fantasies": list(NUM_FANTASIES),
                                "pop_size": POP_SIZE,
                                "n_evals_ga": N_EVALS_GA,
                                "q": Q_BATCH_SIZE,
                                "q_expand": Q_EXPAND_SIZE,
                                "n_eval": N_BO_BATCHES,
                                "n_optimize_acqf": N_OPTIMIZE_ACQF_REPETITIONS,
                                "n_starting": N_STARTING_POINTS,
                                "batch_constraints": dict(DEFAULT_BATCH_CONSTRAINTS),
                                "label": (
                                    f"{benchmark}|s1={seed_discrete}|s2={seed_starting}|"
                                    f"s3={seed_initial_available}|strategy={strategy}|"
                                    f"mode={operating_mode}|"
                                    f"evals={N_EVALS_GA}|pop={POP_SIZE}"
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

    by_mode: dict[str, int] = {mode: 0 for mode in ALL_OPERATING_MODES}
    for combo in combinations:
        mode = str(combo["operating_mode"])
        by_mode[mode] += 1

    output_path = Path(__file__).parent / INPUTS_FILENAME
    write_inputs_json(combinations, output_path)

    print(f"Generated {len(combinations)} combinations")
    print(f"  Benchmarks: {len(BENCHMARK_CLASS_MAP)}")
    print(f"  seed_list_1 x seed_list_2: {len(SEED_LIST_1)} x {len(SEED_LIST_2)}")
    print(f"  initial-availability seeds: {len(INITIAL_AVAILABLE_SOURCE_A_SEEDS)}")
    print(f"  operating modes: {len(ALL_OPERATING_MODES)}")
    print(
        "  myopic mode count: "
        f"{by_mode[MYOPIC_OPERATING_MODE]} ({MYOPIC_OPERATING_MODE})"
    )
    for mode in FANTASIZED_OPERATING_MODES:
        print(f"  fantasized mode count: {by_mode[mode]} ({mode})")
    print(f"  Output: {output_path}")
