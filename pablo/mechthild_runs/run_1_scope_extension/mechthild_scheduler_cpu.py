"""Mode-aware SLURM queue scheduler for the run_1 scope-extension CPU workflows.

Usage::

    python -m pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu \
        {init,worker,status,submitted,failed-pending} [--operating_mode MODE] [--queue_dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pablo.mechthild_runs.run_1_scope_extension.config import (
    ALL_OPERATING_MODES,
    get_job_name_prefix_for_operating_mode,
    get_queue_dirname_for_operating_mode,
    get_results_dirname_for_operating_mode,
    get_strategy_for_operating_mode,
    INPUTS_FILENAME,
    MYOPIC_OPERATING_MODE,
)
from pablo.mechthild_runs.run_1_scope_extension.run_state import (
    get_run_state_path,
    load_run_state,
    load_run_state_for_reporting,
    update_run_state,
)


def _resolve_combination_operating_mode(combination: dict[str, Any]) -> str:
    """Resolve operating mode from one inputs combination.

    Args:
        combination: Combination payload loaded from inputs.json.

    Returns:
        Operating mode string.

    Raises:
        ValueError: If strategy/mode values are inconsistent.
    """
    strategy = str(combination.get("strategy", "")).strip().lower()
    operating_mode = str(combination.get("operating_mode", "")).strip().lower()

    if not operating_mode:
        if strategy == "myopic":
            operating_mode = "myopic"
        elif strategy == "fantasized":
            operating_mode = "layer_averaged"
        else:
            raise ValueError(
                "Combination is missing operating_mode and has unknown strategy "
                f"{strategy!r}"
            )

    if operating_mode not in ALL_OPERATING_MODES:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )

    expected_strategy = get_strategy_for_operating_mode(operating_mode)
    if strategy != expected_strategy:
        raise ValueError(
            "Inconsistent combination fields: "
            f"strategy={strategy!r}, operating_mode={operating_mode!r}, "
            f"expected_strategy={expected_strategy!r}"
        )

    return operating_mode


def _combination_matches_mode(combination: dict[str, Any], operating_mode: str) -> bool:
    """Check whether one combination belongs to the target operating mode.
        Returns:
            bool: Return value of _combination_matches_mode.
    
    Returns:
        bool: Return value of _combination_matches_mode.
"""
    return _resolve_combination_operating_mode(combination) == operating_mode


def _default_queue_dir(operating_mode: str) -> Path:
    """Return the default queue directory for one operating mode.
        Returns:
            Path: Return value of _default_queue_dir.
    
    Returns:
        Path: Return value of _default_queue_dir.
"""
    return Path(__file__).parent / get_queue_dirname_for_operating_mode(operating_mode)


def _results_dir(operating_mode: str) -> Path:
    """Return the per-mode results directory path.
        Returns:
            Path: Return value of _results_dir.
    
    Returns:
        Path: Return value of _results_dir.
"""
    return Path(__file__).parent / get_results_dirname_for_operating_mode(
        operating_mode
    )


def load_inputs(inputs_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load combinations from the run_1 inputs.json.

    Args:
        inputs_path: Optional explicit inputs.json path. Defaults to
            ``INPUTS_FILENAME`` next to this module.

    Returns:
        Mapping from string index to combination payload; empty if the file
        does not exist.
    """
    resolved_path = (
        inputs_path
        if inputs_path is not None
        else Path(__file__).parent / INPUTS_FILENAME
    )
    if not resolved_path.exists():
        return {}
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def count_mode_combinations(
    operating_mode: str,
    inputs: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Count how many combinations belong to one operating mode.

    Args:
        operating_mode: Target operating mode name.
        inputs: Pre-loaded combinations; ``load_inputs()`` is called if None.

    Returns:
        Number of combinations that match ``operating_mode``.
    """
    local_inputs = load_inputs() if inputs is None else inputs
    return sum(
        1
        for _, combination in local_inputs.items()
        if _combination_matches_mode(combination, operating_mode)
    )


def format_progress_bar(done: int, total: int, width: int = 30) -> str:
    """Format one textual progress bar line.

    Args:
        done: Number of completed units.
        total: Total number of units; 0 is treated as 1 for the ratio.
        width: Number of characters in the bar.

    Returns:
        String ``"[###...] done/total (pct%)"``.
    """
    safe_total = max(total, 1)
    ratio = done / safe_total
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    percent = 100.0 * ratio
    return f"[{bar}] {done}/{total} ({percent:.1f}%)"


def _extract_job_idx(lock_name: str) -> int:
    """Extract job index from queue lock filename.
        Returns:
            int: Return value of _extract_job_idx.
    
    Returns:
        int: Return value of _extract_job_idx.
"""
    return int(lock_name.replace("job_", "").replace(".lock", ""))


def _filtered_lock_count(
    directory: Path,
    operating_mode: str,
    inputs: dict[str, dict[str, Any]],
) -> int:
    """Count lock files whose combination belongs to one mode.
        Returns:
            int: Return value of _filtered_lock_count.
    
    Returns:
        int: Return value of _filtered_lock_count.
"""
    if not directory.exists():
        return 0

    count = 0
    for lock_file in directory.glob("job_*.lock"):
        idx = _extract_job_idx(lock_file.name)
        key = str(idx)
        if key not in inputs:
            continue
        if _combination_matches_mode(inputs[key], operating_mode):
            count += 1
    return count


def _count_live_workers(operating_mode: str) -> tuple[int, bool, str]:
    """Count live SLURM workers for one mode by job-name prefix.
        Returns:
            tuple[int, bool, str]: Return value of _count_live_workers.
    
    Returns:
        tuple[int, bool, str]: Return value of _count_live_workers.
"""
    user = os.environ.get("USER", "").strip()
    if not user:
        return 0, False, "missing_user_env"
    if shutil.which("squeue") is None:
        return 0, False, "squeue_not_found"

    job_prefix = get_job_name_prefix_for_operating_mode(operating_mode)
    result = subprocess.run(
        ["squeue", "-u", user, "-t", "R,CG", "-h", "--format=%j"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_detail = (
            result.stderr.strip() or f"squeue_return_code_{result.returncode}"
        )
        return 0, False, error_detail

    live = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(job_prefix):
            live += 1

    return live, True, ""


def get_queue_status(
    queue_dir: str | Path,
    operating_mode: str,
) -> dict[str, object]:
    """Collect queue lock counts and worker telemetry for one mode.

    Args:
        queue_dir: Queue directory holding ``pending/``, ``running/`` and
            ``done/`` lock folders.
        operating_mode: Operating mode whose combinations are counted.

    Returns:
        Dictionary with keys ``pending``, ``running``, ``done``, ``total``,
        ``stale_detected`` and ``stale_requeued`` (both always 0),
        ``live_workers``, ``live_query_reliable`` and ``live_query_error``
        (from ``squeue``).
    """
    queue_path = Path(queue_dir)
    local_inputs = load_inputs()

    pending = _filtered_lock_count(queue_path / "pending", operating_mode, local_inputs)
    running = _filtered_lock_count(queue_path / "running", operating_mode, local_inputs)
    done = _filtered_lock_count(queue_path / "done", operating_mode, local_inputs)
    live_workers, reliable, error = _count_live_workers(operating_mode)

    return {
        "pending": pending,
        "running": running,
        "done": done,
        "total": pending + running + done,
        "stale_detected": 0,
        "stale_requeued": 0,
        "live_workers": live_workers,
        "live_query_reliable": reliable,
        "live_query_error": error,
    }


def initialize_queue(
    queue_dir: str | Path,
    operating_mode: str,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    """Initialize queue locks for one operating mode.

    Args:
        queue_dir: Path to queue_state directory.
        operating_mode: Mode this queue manages.
        start_idx: Start index (inclusive).
        end_idx: End index (exclusive).
    """
    queue_path = Path(queue_dir)
    pending_dir = queue_path / "pending"
    running_dir = queue_path / "running"
    done_dir = queue_path / "done"

    pending_dir.mkdir(parents=True, exist_ok=True)
    running_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    total_combinations = len(inputs)
    if end_idx is None:
        end_idx = total_combinations
    end_idx = min(end_idx, total_combinations)

    results_dir = _results_dir(operating_mode)

    created = 0
    skipped_completed = 0
    skipped_other_mode = 0

    for idx in range(start_idx, end_idx):
        key = str(idx)
        if key not in inputs:
            continue
        combination = inputs[key]
        if not _combination_matches_mode(combination, operating_mode):
            skipped_other_mode += 1
            continue

        lock_name = f"job_{idx}.lock"
        pending_lock = pending_dir / lock_name
        running_lock = running_dir / lock_name
        done_lock = done_dir / lock_name
        result_file = results_dir / f"result_{idx}.json"

        if result_file.exists():
            skipped_completed += 1
            if not done_lock.exists():
                done_lock.touch()
            if pending_lock.exists():
                pending_lock.unlink()
            if running_lock.exists():
                running_lock.unlink()
            created_or_existing_pending = False
        elif pending_lock.exists() or running_lock.exists() or done_lock.exists():
            created_or_existing_pending = pending_lock.exists()
        else:
            pending_lock.touch()
            created += 1
            created_or_existing_pending = True

        total_iterations = int(combination["n_eval"])
        state_path = get_run_state_path(queue_path, idx)
        if state_path.exists():
            continue

        status = "pending_waiting"
        if done_lock.exists():
            status = "done"
        elif running_lock.exists():
            status = "running"
        elif created_or_existing_pending:
            status = "pending_waiting"

        update_run_state(
            state_path=state_path,
            combination_index=idx,
            total_iterations=total_iterations,
            status=status,
            current_iteration=0,
        )

    print(
        f"[{operating_mode}] Initialized {created} jobs in pending queue "
        f"({queue_path})"
    )
    print(f"  Skipped {skipped_completed} already completed in {results_dir.name}/")
    print(f"  Skipped {skipped_other_mode} jobs belonging to other operating modes")
    print(f"  Combination range: {start_idx} to {end_idx - 1}")
    print(f"  Total combinations available: {total_combinations}")
    print(
        f"  Total combinations for mode '{operating_mode}': "
        f"{count_mode_combinations(operating_mode, inputs=inputs)}"
    )


def _resolve_slurm_worker_identity() -> tuple[str | None, str | None, str | None]:
    """Resolve SLURM identity from environment variables.
        Returns:
            tuple[str | None, str | None, str | None]: Return value of _resolve_slurm_worker_identity.
    
    Returns:
        tuple[str | None, str | None, str | None]: Return value of _resolve_slurm_worker_identity.
"""
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip() or None
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip() or None

    if slurm_array_task_id in {"", "n/a", "na", "none", "4294967294"}:
        slurm_array_task_id = None

    slurm_worker_id = None
    if slurm_job_id and slurm_array_task_id:
        slurm_worker_id = f"{slurm_job_id}_{slurm_array_task_id}"
    elif slurm_job_id:
        slurm_worker_id = slurm_job_id

    return slurm_job_id, slurm_array_task_id, slurm_worker_id


def _claim_next_mode_job(
    pending_dir: Path,
    running_dir: Path,
    operating_mode: str,
    inputs: dict[str, dict[str, Any]],
) -> int | None:
    """Atomically claim one pending job lock for the requested mode.
        Returns:
            int | None: Return value of _claim_next_mode_job.
    
    Returns:
        int | None: Return value of _claim_next_mode_job.
"""
    for pending_file in sorted(pending_dir.glob("job_*.lock")):
        idx = _extract_job_idx(pending_file.name)
        key = str(idx)
        if key not in inputs:
            continue
        if not _combination_matches_mode(inputs[key], operating_mode):
            continue

        target_file = running_dir / pending_file.name
        claim = subprocess.run(
            ["mv", "-n", str(pending_file), str(target_file)],
            check=False,
            stderr=subprocess.DEVNULL,
        )
        if claim.returncode == 0 and target_file.exists():
            return idx

    return None


def run_single_combination(combination_idx: int, queue_dir: str | Path) -> int:
    """Run one combination through ``run_single_job`` in a subprocess.

    The interpreter is taken from ``RUN2_WORKER_PYTHON`` (default: the current
    one); a positive integer in ``RUN2_WORKER_N_EVAL_OVERRIDE`` is passed as
    ``--n_eval_override``.

    Args:
        combination_idx: Index of the combination in inputs.json.
        queue_dir: Queue directory passed to ``run_single_job``.

    Returns:
        Exit code of the ``run_single_job`` subprocess.
    """
    worker_python = os.environ.get("RUN2_WORKER_PYTHON", "").strip()
    if not worker_python:
        worker_python = sys.executable or "python"

    n_eval_override_raw = os.environ.get("RUN2_WORKER_N_EVAL_OVERRIDE", "").strip()

    command = [
        worker_python,
        "-m",
        "pablo.mechthild_runs.run_1_scope_extension.run_single_job",
        str(combination_idx),
        "--queue_dir",
        str(queue_dir),
    ]

    if n_eval_override_raw.isdigit() and int(n_eval_override_raw) > 0:
        command.extend(["--n_eval_override", n_eval_override_raw])

    result = subprocess.run(command, check=False)
    return int(result.returncode)


def worker_loop(queue_dir: str | Path, operating_mode: str) -> None:
    """Process pending jobs for one operating mode until the queue drains.

    Each claimed job runs through :func:`run_single_combination`. On exit
    code 0 its lock moves to ``done/`` and the run state becomes ``done``;
    otherwise the lock returns to ``pending/`` with status ``pending_failed``.

    Args:
        queue_dir: Queue directory holding the lock folders.
        operating_mode: Operating mode whose combinations this worker claims.
    """
    queue_path = Path(queue_dir)
    pending_dir = queue_path / "pending"
    running_dir = queue_path / "running"
    done_dir = queue_path / "done"

    pending_dir.mkdir(parents=True, exist_ok=True)
    running_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{operating_mode}] Worker started. Queue directory: {queue_path}")
    inputs = load_inputs()
    slurm_job_id, slurm_array_task_id, slurm_worker_id = (
        _resolve_slurm_worker_identity()
    )

    while True:
        mode_pending = []
        for lock in pending_dir.glob("job_*.lock"):
            idx = _extract_job_idx(lock.name)
            key = str(idx)
            if key not in inputs:
                continue
            if _combination_matches_mode(inputs[key], operating_mode):
                mode_pending.append(lock)

        if not mode_pending:
            print(f"[{operating_mode}] No pending jobs left. Worker exiting.")
            break

        job_idx = _claim_next_mode_job(
            pending_dir=pending_dir,
            running_dir=running_dir,
            operating_mode=operating_mode,
            inputs=inputs,
        )
        if job_idx is None:
            continue

        key = str(job_idx)
        combination = inputs[key]
        total_iterations = int(combination["n_eval"])
        state_path = get_run_state_path(queue_path, job_idx)
        state_before = load_run_state(
            state_path=state_path,
            combination_index=job_idx,
            total_iterations=total_iterations,
        )
        attempts = int(state_before.get("attempts", 0)) + 1

        update_run_state(
            state_path=state_path,
            combination_index=job_idx,
            total_iterations=total_iterations,
            status="running",
            attempts=attempts,
            current_iteration=0,
            slurm_job_id=slurm_job_id,
            slurm_array_task_id=slurm_array_task_id,
            slurm_worker_id=slurm_worker_id,
            last_error=None,
            last_return_code=None,
        )

        lock_name = f"job_{job_idx}.lock"
        running_lock = running_dir / lock_name
        done_lock = done_dir / lock_name
        pending_lock = pending_dir / lock_name

        print(f"[{operating_mode}] Processing combination {job_idx}")
        return_code = run_single_combination(job_idx, queue_path)

        if return_code == 0:
            subprocess.run(
                ["mv", "-f", str(running_lock), str(done_lock)],
                check=False,
                stderr=subprocess.DEVNULL,
            )
            update_run_state(
                state_path=state_path,
                combination_index=job_idx,
                total_iterations=total_iterations,
                status="done",
                current_iteration=total_iterations,
                last_error=None,
                last_return_code=0,
            )
            print(f"[{operating_mode}] Completed combination {job_idx}")
        else:
            subprocess.run(
                ["mv", "-f", str(running_lock), str(pending_lock)],
                check=False,
                stderr=subprocess.DEVNULL,
            )
            update_run_state(
                state_path=state_path,
                combination_index=job_idx,
                total_iterations=total_iterations,
                status="pending_failed",
                last_error=f"exit_code={return_code}",
                last_return_code=return_code,
            )
            print(
                f"[{operating_mode}] Combination {job_idx} failed with code "
                f"{return_code}, returned to pending"
            )


def _format_submitted_line(
    combination_idx: int,
    combination: dict[str, Any],
    state: dict[str, Any],
) -> str:
    """Render one line for submitted/running view.
        Returns:
            str: Return value of _format_submitted_line.
    
    Returns:
        str: Return value of _format_submitted_line.
"""
    error_suffix = " ERROR" if state.get("last_error") else ""
    state_warning = str(state.get("state_file_warning", "")).strip()
    warning_suffix = f" WARN={state_warning}" if state_warning else ""
    current_iteration = int(state.get("current_iteration", 0))
    total_iterations = int(state.get("total_iterations", int(combination["n_eval"])))
    strategy = str(combination.get("strategy", "unknown"))
    mode = str(combination.get("operating_mode", "unknown"))
    seed_initial = combination.get("seed_initial_available", "na")
    return (
        f"{combination_idx} | {combination['benchmark']} | "
        f"s3={seed_initial} strategy={strategy} mode={mode} "
        f"pop={combination['pop_size']} evals={combination['n_evals_ga']} | "
        f"iter={current_iteration}/{total_iterations}{error_suffix}{warning_suffix}"
    )


def _format_failed_pending_line(
    combination_idx: int,
    combination: dict[str, Any],
    state: dict[str, Any],
) -> str:
    """Render one line for failed-pending view.
        Returns:
            str: Return value of _format_failed_pending_line.
    
    Returns:
        str: Return value of _format_failed_pending_line.
"""
    error_message = str(state.get("last_error", "unknown_error"))
    state_warning = str(state.get("state_file_warning", "")).strip()
    warning_suffix = f" | WARN={state_warning}" if state_warning else ""
    strategy = str(combination.get("strategy", "unknown"))
    mode = str(combination.get("operating_mode", "unknown"))
    seed_initial = combination.get("seed_initial_available", "na")
    return (
        f"{combination_idx} | {combination['benchmark']} | "
        f"s3={seed_initial} strategy={strategy} mode={mode} "
        f"pop={combination['pop_size']} evals={combination['n_evals_ga']} | "
        f"ERROR | {error_message}{warning_suffix}"
    )


def get_submitted_lines(
    queue_dir: str | Path,
    operating_mode: str,
    inputs: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Build one status line per running job for the selected mode.

    Args:
        queue_dir: Queue directory holding the lock folders.
        operating_mode: Operating mode to filter by.
        inputs: Pre-loaded combinations; ``load_inputs()`` is called if None.

    Returns:
        One formatted line per lock in ``running/``, sorted by lock filename,
        with iteration progress from the run state.
    """
    queue_path = Path(queue_dir)
    local_inputs = load_inputs() if inputs is None else inputs
    running_dir = queue_path / "running"
    if not running_dir.exists():
        return []

    lines: list[str] = []
    for lock_file in sorted(running_dir.glob("job_*.lock")):
        combination_idx = _extract_job_idx(lock_file.name)
        key = str(combination_idx)
        if key not in local_inputs:
            continue
        combination = local_inputs[key]
        if not _combination_matches_mode(combination, operating_mode):
            continue

        total_iterations = int(combination["n_eval"])
        state_path = get_run_state_path(queue_path, combination_idx)
        state = load_run_state_for_reporting(
            state_path=state_path,
            combination_index=combination_idx,
            total_iterations=total_iterations,
        )
        lines.append(
            _format_submitted_line(
                combination_idx=combination_idx,
                combination=combination,
                state=state,
            )
        )
    return lines


def get_failed_pending_lines(
    queue_dir: str | Path,
    operating_mode: str,
    inputs: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Build lines for ``pending_failed`` jobs in one operating mode.

    Args:
        queue_dir: Queue directory holding the lock folders.
        operating_mode: Operating mode to filter by.
        inputs: Pre-loaded combinations; ``load_inputs()`` is called if None.

    Returns:
        One formatted error line per lock in ``pending/`` whose run state has
        status ``pending_failed``.
    """
    queue_path = Path(queue_dir)
    local_inputs = load_inputs() if inputs is None else inputs
    pending_dir = queue_path / "pending"
    if not pending_dir.exists():
        return []

    lines: list[str] = []
    for lock_file in sorted(pending_dir.glob("job_*.lock")):
        combination_idx = _extract_job_idx(lock_file.name)
        key = str(combination_idx)
        if key not in local_inputs:
            continue

        combination = local_inputs[key]
        if not _combination_matches_mode(combination, operating_mode):
            continue

        total_iterations = int(combination["n_eval"])
        state_path = get_run_state_path(queue_path, combination_idx)
        if not state_path.exists():
            continue

        state = load_run_state_for_reporting(
            state_path=state_path,
            combination_index=combination_idx,
            total_iterations=total_iterations,
        )
        if state.get("status") != "pending_failed":
            continue

        lines.append(
            _format_failed_pending_line(
                combination_idx=combination_idx,
                combination=combination,
                state=state,
            )
        )

    return lines


def _resolve_queue_dir_argument(
    queue_dir: str | None,
    operating_mode: str,
) -> Path:
    """Resolve queue directory from CLI args and mode defaults.
        Returns:
            Path: Return value of _resolve_queue_dir_argument.
    
    Returns:
        Path: Return value of _resolve_queue_dir_argument.
"""
    if queue_dir is None:
        return _default_queue_dir(operating_mode)
    return Path(queue_dir)


def main() -> None:
    """CLI entrypoint for mode-aware run_1 queue operations."""
    parser = argparse.ArgumentParser(
        description="Manage run_1 PlanAhead CPU queue state by operating mode"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser(
        "init", help="Initialize queue_state locks for one operating mode"
    )
    init_parser.add_argument(
        "--operating_mode",
        choices=ALL_OPERATING_MODES,
        default=MYOPIC_OPERATING_MODE,
    )
    init_parser.add_argument("--queue_dir", default=None)
    init_parser.add_argument("--start", type=int, default=0)
    init_parser.add_argument("--end", type=int, default=None)

    worker_parser = subparsers.add_parser("worker", help="Run worker loop")
    worker_parser.add_argument(
        "--operating_mode",
        choices=ALL_OPERATING_MODES,
        default=MYOPIC_OPERATING_MODE,
    )
    worker_parser.add_argument("--queue_dir", default=None)

    status_parser = subparsers.add_parser("status", help="Show queue status")
    status_parser.add_argument(
        "--operating_mode",
        choices=ALL_OPERATING_MODES,
        default=MYOPIC_OPERATING_MODE,
    )
    status_parser.add_argument("--queue_dir", default=None)
    status_parser.add_argument("--auto_requeue_stale", action="store_true")
    status_parser.add_argument("--no_auto_requeue_stale", action="store_true")

    submitted_parser = subparsers.add_parser(
        "submitted", help="Show running jobs with iteration progress"
    )
    submitted_parser.add_argument(
        "--operating_mode",
        choices=ALL_OPERATING_MODES,
        default=MYOPIC_OPERATING_MODE,
    )
    submitted_parser.add_argument("--queue_dir", default=None)
    submitted_parser.add_argument("--auto_requeue_stale", action="store_true")
    submitted_parser.add_argument("--no_auto_requeue_stale", action="store_true")

    failed_pending_parser = subparsers.add_parser(
        "failed-pending", help="Show failed jobs currently in pending"
    )
    failed_pending_parser.add_argument(
        "--operating_mode",
        choices=ALL_OPERATING_MODES,
        default=MYOPIC_OPERATING_MODE,
    )
    failed_pending_parser.add_argument("--queue_dir", default=None)
    failed_pending_parser.add_argument("--auto_requeue_stale", action="store_true")
    failed_pending_parser.add_argument("--no_auto_requeue_stale", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    operating_mode = str(args.operating_mode)
    queue_path = _resolve_queue_dir_argument(args.queue_dir, operating_mode)

    if args.command == "init":
        initialize_queue(
            queue_dir=queue_path,
            operating_mode=operating_mode,
            start_idx=int(args.start),
            end_idx=args.end,
        )
        return

    if args.command == "worker":
        worker_loop(queue_dir=queue_path, operating_mode=operating_mode)
        return

    if args.command == "status":
        status = get_queue_status(queue_dir=queue_path, operating_mode=operating_mode)
        print(f"[{operating_mode}] Queue Status:")
        print(f"  Progress: {format_progress_bar(status['done'], status['total'])}")
        print(f"  Pending: {status['pending']}")
        print(f"  Running: {status['running']}")
        print(f"  Done: {status['done']}")
        print(f"  Total: {status['total']}")
        print(f"  Live workers in squeue: {status['live_workers']}")
        if not bool(status.get("live_query_reliable", True)):
            print(
                "  Live worker query warning: "
                f"{str(status.get('live_query_error', 'unknown'))}"
            )
        print(f"  Stale running detected: {status['stale_detected']}")
        print(f"  Stale running auto-requeued: {status['stale_requeued']}")
        return

    if args.command == "submitted":
        lines = get_submitted_lines(
            queue_dir=queue_path,
            operating_mode=operating_mode,
        )
        for line in lines:
            print(line)
        return

    if args.command == "failed-pending":
        lines = get_failed_pending_lines(
            queue_dir=queue_path,
            operating_mode=operating_mode,
        )
        for line in lines:
            print(line)
        return


if __name__ == "__main__":
    main()
