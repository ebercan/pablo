"""Execute one indexed run_1 scope-extension PlanAhead combination from inputs.json.

Usage:
    python -m pablo.mechthild_runs.run_1_scope_extension.run_single_job <combination_index> [--queue_dir DIR] [--n_eval_override N] [--n_evals_ga_override N] [--pop_size_override N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pablo.mechthild_runs.run_1_scope_extension.config import (
    ALL_OPERATING_MODES,
    get_queue_dirname_for_operating_mode,
    get_results_dirname_for_operating_mode,
    get_strategy_for_operating_mode,
    INPUTS_FILENAME,
    QUEUE_DIRNAME,
    RESULTS_DIRNAME,
)
from pablo.mechthild_runs.run_1_scope_extension.run_ga_experiment import (
    run_bo_ga_experiment,
)
from pablo.mechthild_runs.run_1_scope_extension.run_state import (
    get_run_state_path,
    update_run_state,
)


def _resolve_operating_mode(combination: dict[str, object]) -> str:
    """Resolve operating_mode from one combination payload.

    Args:
        combination: One inputs.json entry.

    Returns:
        Operating mode string.

    Raises:
        ValueError: If strategy/mode fields are inconsistent.
    """
    strategy = str(combination.get("strategy", "")).strip().lower()
    mode = str(combination.get("operating_mode", "")).strip().lower()

    if not mode:
        if strategy == "myopic":
            mode = "myopic"
        elif strategy == "fantasized":
            mode = "layer_averaged"
        else:
            raise ValueError(
                "Combination is missing operating_mode and contains unknown "
                f"strategy {strategy!r}"
            )

    if mode not in ALL_OPERATING_MODES:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {mode!r}"
        )

    expected_strategy = get_strategy_for_operating_mode(mode)
    if strategy != expected_strategy:
        raise ValueError(
            "Combination has inconsistent strategy/operating_mode fields: "
            f"strategy={strategy!r}, operating_mode={mode!r}, "
            f"expected_strategy={expected_strategy!r}"
        )

    return mode


def main(
    combination_index: int,
    queue_dir: str | Path | None = None,
    n_eval_override: int | None = None,
    n_evals_ga_override: int | None = None,
    pop_size_override: int | None = None,
) -> None:
    """Run one benchmark configuration and persist its result as JSON.

    Args:
        combination_index: Index in this run's inputs.json.
        queue_dir: Optional path to queue_state directory.
        n_eval_override: Optional BO iteration override for short runs/tests.
        n_evals_ga_override: Optional GA evaluation override for short runs/tests.
        pop_size_override: Optional GA population override for short runs/tests.

    Raises:
        ValueError: If the index is not found in inputs.json.
    """
    if n_eval_override is not None and n_eval_override <= 0:
        raise ValueError("n_eval_override must be greater than zero")
    if n_evals_ga_override is not None and n_evals_ga_override <= 0:
        raise ValueError("n_evals_ga_override must be greater than zero")
    if pop_size_override is not None and pop_size_override <= 0:
        raise ValueError("pop_size_override must be greater than zero")

    base_dir = Path(__file__).parent
    inputs_path = base_dir / INPUTS_FILENAME

    with open(inputs_path, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)

    key = str(combination_index)
    if key not in inputs:
        raise ValueError(
            f"Combination index {combination_index} not found in {inputs_path}"
        )

    comb = inputs[key]
    operating_mode = _resolve_operating_mode(comb)

    if queue_dir is not None:
        queue_path = Path(queue_dir)
    else:
        if QUEUE_DIRNAME != "queue_state":
            queue_path = base_dir / QUEUE_DIRNAME
        else:
            queue_path = base_dir / get_queue_dirname_for_operating_mode(operating_mode)

    if RESULTS_DIRNAME != "results":
        results_dir = base_dir / RESULTS_DIRNAME
    else:
        results_dir = base_dir / get_results_dirname_for_operating_mode(operating_mode)

    output_path = results_dir / f"result_{combination_index}.json"
    if output_path.exists():
        print(f"Result {output_path.name} already exists. Skipping execution.")
        sys.exit(0)

    batch_constraints_raw = comb["batch_constraints"]
    batch_constraints = {
        int(dim): int(k_val) for dim, k_val in batch_constraints_raw.items()
    }

    print(f"Running combination {combination_index}: {comb['label']}")

    state_path = get_run_state_path(queue_path, combination_index)
    total_iterations = (
        int(comb["n_eval"]) if n_eval_override is None else int(n_eval_override)
    )
    n_evals_ga = (
        int(comb["n_evals_ga"])
        if n_evals_ga_override is None
        else int(n_evals_ga_override)
    )
    pop_size = (
        int(comb["pop_size"]) if pop_size_override is None else int(pop_size_override)
    )
    update_run_state(
        state_path=state_path,
        combination_index=combination_index,
        total_iterations=total_iterations,
        status="running",
        current_iteration=0,
    )

    progress_label = (
        f"[{combination_index}] {comb['benchmark']} "
        "("
        f"s1={comb['seed_discrete']}, "
        f"s2={comb['seed_starting']}, "
        f"s3={comb['seed_initial_available']}, "
        f"strategy={comb['strategy']}, "
        f"mode={operating_mode}"
        ""
        ")"
    )

    run_result = run_bo_ga_experiment(
        benchmark_name=comb["benchmark"],
        seed_discrete=int(comb["seed_discrete"]),
        seed_starting=int(comb["seed_starting"]),
        seed_initial_available=int(comb["seed_initial_available"]),
        strategy=str(comb["strategy"]),
        pop_size=pop_size,
        ga_ratio=int(comb["ga_ratio"]) if "ga_ratio" in comb else None,
        n_evals_ga=n_evals_ga,
        q=int(comb["q"]),
        q_expand=int(comb["q_expand"]),
        n_eval=total_iterations,
        n_optimize_acqf=int(comb["n_optimize_acqf"]),
        n_starting=int(comb["n_starting"]),
        initial_available_source_a_count=int(comb["initial_available_source_a_count"]),
        source_a_availability_delay=int(comb["source_a_availability_delay"]),
        lookahead_horizon=int(comb["lookahead_horizon"]),
        num_fantasies=[int(v) for v in comb["num_fantasies"]],
        batch_constraints=batch_constraints,
        run_state_path=state_path,
        run_state_combination_index=combination_index,
        verbose=True,
        progress_label=progress_label,
        fantasized_operating_mode=operating_mode,
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "combination_index": combination_index,
        "combination": comb,
        "execution_options": {
            "n_eval_override": n_eval_override,
            "n_evals_ga_override": n_evals_ga_override,
            "pop_size_override": pop_size_override,
            "operating_mode": operating_mode,
        },
        "device_policy": {
            "requested_device": "cpu",
            "actual_device": "cpu",
        },
        "result": run_result,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle)

    update_run_state(
        state_path=state_path,
        combination_index=combination_index,
        total_iterations=total_iterations,
        status="done",
        current_iteration=total_iterations,
        last_error=None,
        last_return_code=0,
    )

    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one run_1 PlanAhead combination")
    parser.add_argument("combination_index", type=int, help="Index from inputs.json")
    parser.add_argument(
        "--queue_dir",
        type=str,
        default=None,
        help="Optional queue_state path for run-state metadata",
    )
    parser.add_argument(
        "--n_eval_override",
        type=int,
        default=None,
        help="Optional override for BO iteration count",
    )
    parser.add_argument(
        "--n_evals_ga_override",
        type=int,
        default=None,
        help="Optional override for GA evaluations per optimization",
    )
    parser.add_argument(
        "--pop_size_override",
        type=int,
        default=None,
        help="Optional override for GA population size",
    )
    args = parser.parse_args()
    main(
        args.combination_index,
        queue_dir=args.queue_dir,
        n_eval_override=args.n_eval_override,
        n_evals_ga_override=args.n_evals_ga_override,
        pop_size_override=args.pop_size_override,
    )
