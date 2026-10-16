"""Generate all GA setting combinations for the run_2 robustness sweep.

Writes one job per ``(n_evals_ga, pop_size, repeat)`` combination to
``inputs.json`` next to this file. Run from the repo root:

    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.generate_combinations
"""

from __future__ import annotations

import json
from pathlib import Path

from pablo.mechthild_runs.run_2_acq_opt_robustness.config import (
    INPUTS_FILENAME,
    N_EVALS_GA_VALUES,
    N_REPEATS,
    POP_SIZES,
)


def generate_all_combinations() -> list[dict[str, object]]:
    """Generate all (n_evals_ga, pop_size, repeat) combinations.

    Returns:
        List of combination dictionaries, one per job.
    """
    combinations: list[dict[str, object]] = []

    for n_evals_ga in N_EVALS_GA_VALUES:
        for pop_size in POP_SIZES:
            for repeat_index in range(N_REPEATS):
                combinations.append(
                    {
                        "n_evals_ga": n_evals_ga,
                        "pop_size": pop_size,
                        "repeat_index": repeat_index,
                        "label": (
                            f"n_evals={n_evals_ga}|pop={pop_size}|rep={repeat_index}"
                        ),
                    }
                )

    return combinations


def write_inputs_json(
    combinations: list[dict[str, object]], output_path: Path
) -> None:
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
    print(f"  n_evals_ga values: {N_EVALS_GA_VALUES}")
    print(f"  pop_size values:   {POP_SIZES}")
    print(f"  repeats per combo: {N_REPEATS}")
    print(
        f"  Total: {len(N_EVALS_GA_VALUES)} x {len(POP_SIZES)} x {N_REPEATS} "
        f"= {len(combinations)}"
    )
    print(f"  Output: {output_path}")
