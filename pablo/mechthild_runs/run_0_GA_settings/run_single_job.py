"""Execute one indexed run_0 GA-settings combination from inputs.json.

Writes ``results/result_<combination_index>.json`` and updates the worker's
run-state file under the queue directory.

Usage:
    python -m pablo.mechthild_runs.run_0_GA_settings.run_single_job <combination_index> \
        [--queue_dir DIR] [--device {cpu,cuda,auto}] [--allow_cpu_fallback {0,1}] \
        [--n_eval_override N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pablo.mechthild_runs.run_0_GA_settings.config import (
    ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES,
    DEVICE_CHOICES,
    INPUTS_FILENAME,
    QUEUE_DIRNAME,
    RESULTS_DIRNAME,
)
from pablo.mechthild_runs.run_0_GA_settings.run_ga_experiment import (
    resolve_execution_device_policy,
    run_bo_ga_experiment,
)
from pablo.mechthild_runs.run_0_GA_settings.run_state import get_run_state_path, update_run_state


def main(
    combination_index: int,
    queue_dir: str | Path | None = None,
    device: str = "cpu",
    n_eval_override: int | None = None,
    allow_cpu_fallback: bool = ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES,
) -> None:
    """Run one benchmark configuration and persist its result as JSON.

    Args:
        combination_index: Index in this sweep's inputs.json.
        queue_dir: Optional path to queue_state directory.
        device: Execution device. Use cpu, cuda, or auto.
        n_eval_override: Optional BO iteration override for short runs/tests.
        allow_cpu_fallback: Whether CUDA issues may fall back to CPU.

    Raises:
        ValueError: If ``device`` is not in ``DEVICE_CHOICES``, if
            ``n_eval_override`` is not positive, or if the index is not found
            in inputs.json.
    """
    if device not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of: {', '.join(DEVICE_CHOICES)}")
    if n_eval_override is not None and n_eval_override <= 0:
        raise ValueError("n_eval_override must be greater than zero")

    base_dir = Path(__file__).parent
    inputs_path = base_dir / INPUTS_FILENAME
    queue_path = Path(queue_dir) if queue_dir is not None else base_dir / QUEUE_DIRNAME

    with open(inputs_path, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)

    key = str(combination_index)
    if key not in inputs:
        raise ValueError(
            f"Combination index {combination_index} not found in {inputs_path}"
        )

    comb = inputs[key]
    batch_constraints_raw = comb["batch_constraints"]
    batch_constraints = {
        int(dim): int(k_val) for dim, k_val in batch_constraints_raw.items()
    }

    print(f"Running combination {combination_index}: {comb['label']}")

    state_path = get_run_state_path(queue_path, combination_index)
    total_iterations = (
        int(comb["n_eval"]) if n_eval_override is None else int(n_eval_override)
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
        f"(s1={comb['seed_discrete']}, s2={comb['seed_starting']}, req_dev={device})"
    )

    execution_policy = resolve_execution_device_policy(
        requested_device=device,
        allow_cpu_fallback=allow_cpu_fallback,
        perform_cuda_smoke_test=True,
    )
    actual_device = str(execution_policy["actual_device"])

    update_run_state(
        state_path=state_path,
        combination_index=combination_index,
        total_iterations=total_iterations,
        requested_device=str(execution_policy["requested_device"]),
        actual_device=actual_device,
        fallback_used=bool(execution_policy["fallback_used"]),
        fallback_reason=execution_policy["fallback_reason"],
        compatibility_status=str(execution_policy["compatibility_status"]),
    )

    run_result = run_bo_ga_experiment(
        benchmark_name=comb["benchmark"],
        seed_discrete=int(comb["seed_discrete"]),
        seed_starting=int(comb["seed_starting"]),
        pop_size=int(comb["pop_size"]),
        ga_ratio=int(comb["ga_ratio"]),
        n_evals_ga=int(comb["n_evals_ga"]),
        q=int(comb["q"]),
        n_eval=total_iterations,
        n_optimize_acqf=int(comb["n_optimize_acqf"]),
        n_starting=int(comb["n_starting"]),
        batch_constraints=batch_constraints,
        device=actual_device,
        allow_cpu_fallback_on_cuda_issues=allow_cpu_fallback,
        prevalidated_device_policy=execution_policy,
        run_state_path=state_path,
        run_state_combination_index=combination_index,
        verbose=True,
        progress_label=progress_label,
    )

    results_dir = base_dir / RESULTS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"result_{combination_index}.json"

    output_data = {
        "combination_index": combination_index,
        "combination": comb,
        "execution_options": {
            "requested_device": str(execution_policy["requested_device"]),
            "actual_device": actual_device,
            "allow_cpu_fallback": bool(allow_cpu_fallback),
            "fallback_used": bool(execution_policy["fallback_used"]),
            "fallback_reason": execution_policy["fallback_reason"],
            "compatibility_status": str(execution_policy["compatibility_status"]),
            "device": actual_device,
            "n_eval_override": n_eval_override,
        },
        "device_policy": execution_policy,
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
        requested_device=str(execution_policy["requested_device"]),
        actual_device=actual_device,
        fallback_used=bool(execution_policy["fallback_used"]),
        fallback_reason=execution_policy["fallback_reason"],
        compatibility_status=str(execution_policy["compatibility_status"]),
        last_error=None,
        last_return_code=0,
    )

    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one run_0 GA settings combination"
    )
    parser.add_argument("combination_index", type=int, help="Index from inputs.json")
    parser.add_argument(
        "--queue_dir",
        type=str,
        default=None,
        help="Optional queue_state path for run-state metadata",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=list(DEVICE_CHOICES),
        help="Execution device for BO run",
    )
    parser.add_argument(
        "--allow_cpu_fallback",
        type=int,
        choices=[0, 1],
        default=int(ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES),
        help="Allow fallback to CPU when requested CUDA is unavailable/incompatible",
    )
    parser.add_argument(
        "--n_eval_override",
        type=int,
        default=None,
        help="Optional override for BO iteration count",
    )
    args = parser.parse_args()
    main(
        args.combination_index,
        queue_dir=args.queue_dir,
        device=args.device,
        n_eval_override=args.n_eval_override,
        allow_cpu_fallback=bool(args.allow_cpu_fallback),
    )
