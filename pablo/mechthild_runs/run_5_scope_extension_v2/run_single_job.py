"""Execute one indexed run_5 combination from inputs.json.

Run from the repo root:

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.run_single_job <combination_index> \
        [--queue_dir DIR] [--n_eval_override N] [--num_fantasies_override F1 F2 F3 F4]

The overrides shorten a local smoke run; ``init`` and this script reject such a
result as a finished job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    INPUTS_PATH,
    combination_operating_mode,
    queue_dir as default_queue_dir,
    results_dir,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.run_bo_experiment import run_bo_experiment
from pablo.mechthild_runs.run_5_scope_extension_v2.run_state import (
    check_existing_result,
    get_run_state_path,
    update_run_state,
)


def _parse_batch_constraints(raw: dict[str, list[int]]) -> dict[int, tuple[int, ...]]:
    """``{"0": [2, 2, 2]}`` → ``{0: (2, 2, 2)}``."""
    return {int(dim): tuple(int(v) for v in spec) for dim, spec in raw.items()}


def main(
    combination_index: int,
    queue_dir: str | Path | None = None,
    n_eval_override: int | None = None,
    num_fantasies_override: list[int] | None = None,
) -> None:
    """Run one combination and write ``result_<combination_index>.json``.

    The result goes to the operating mode's results folder, through a temporary
    file and a rename, so an interrupted write leaves no result file. If the
    result file exists and is a full run of this combination, the process
    exits with status 0 without running anything.

    Args:
        combination_index: Index in ``inputs.json``.
        queue_dir: Queue directory holding the run state; the mode's default
            queue folder when ``None``.
        n_eval_override: BO iterations of a shortened run.
        num_fantasies_override: Fantasies per depth of a shortened run.

    Raises:
        ValueError: If ``n_eval_override`` is not positive, the index is not in
            ``inputs.json``, the combination's mode and strategy disagree, or an
            existing result file is not a full run of this combination.
    """
    if n_eval_override is not None and n_eval_override <= 0:
        raise ValueError("n_eval_override must be greater than zero")

    with open(INPUTS_PATH, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)
    key = str(combination_index)
    if key not in inputs:
        raise ValueError(f"Combination index {combination_index} not found in {INPUTS_PATH.name}")
    comb = inputs[key]
    operating_mode = combination_operating_mode(comb)

    queue_path = default_queue_dir(operating_mode) if queue_dir is None else Path(queue_dir)
    output_path = results_dir(operating_mode) / f"result_{combination_index}.json"
    if output_path.exists():
        check_existing_result(output_path, comb)
        print(f"Result {output_path.name} already exists. Skipping execution.")
        sys.exit(0)

    print(f"Running combination {combination_index}: {comb['label']}", flush=True)
    state_path = get_run_state_path(queue_path, combination_index)
    total_iterations = int(comb["n_eval"]) if n_eval_override is None else int(n_eval_override)
    num_fantasies = [int(v) for v in (comb["num_fantasies"] if num_fantasies_override is None else num_fantasies_override)]
    update_run_state(state_path, combination_index, total_iterations, status="running", current_iteration=0)

    run_result = run_bo_experiment(
        benchmark_name=str(comb["benchmark"]),
        seed=int(comb["seed"]),
        operating_mode=operating_mode,
        q=int(comb["q"]),
        q_expand=int(comb["q_expand"]),
        n_eval=total_iterations,
        source_a_availability_delay=int(comb["source_a_availability_delay"]),
        lookahead_horizon=int(comb["lookahead_horizon"]),
        num_fantasies=num_fantasies,
        batch_constraints=_parse_batch_constraints(comb["batch_constraints"]),
        run_state_path=state_path,
        run_state_combination_index=combination_index,
        log_label=f"[{combination_index}]",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "combination_index": combination_index,
        "combination": comb,
        "execution_options": {
            "n_eval_override": n_eval_override,
            "num_fantasies_override": num_fantasies_override,
        },
        "result": run_result,
    }
    temp_path = output_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle)
    temp_path.replace(output_path)

    update_run_state(
        state_path,
        combination_index,
        total_iterations,
        status="done",
        current_iteration=total_iterations,
        last_error=None,
        last_return_code=0,
    )
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one run_5 combination")
    parser.add_argument("combination_index", type=int, help="Index from inputs.json")
    parser.add_argument("--queue_dir", type=str, default=None)
    parser.add_argument("--n_eval_override", type=int, default=None)
    parser.add_argument(
        "--num_fantasies_override",
        type=int,
        nargs="+",
        default=None,
        help="Fantasies per depth, e.g. --num_fantasies_override 2 1 1 1",
    )
    args = parser.parse_args()
    main(
        args.combination_index,
        queue_dir=args.queue_dir,
        n_eval_override=args.n_eval_override,
        num_fantasies_override=args.num_fantasies_override,
    )
