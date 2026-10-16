"""File-based SLURM queue scheduler for the run_0 GA-settings sweep.

Usage:
    python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler init [--start N] [--end M]
    python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler worker
    python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler status [--auto_requeue_stale]
    python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler submitted [--auto_requeue_stale]
    python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler failed-pending [--auto_requeue_stale]

Every command takes ``--queue_dir`` (default: ``queue_state/`` next to this file).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pablo.mechthild_runs.run_0_GA_settings.config import (
    ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES,
    INPUTS_FILENAME,
    QUEUE_DIRNAME,
)
from pablo.mechthild_runs.run_0_GA_settings.run_state import (
    get_run_state_path,
    load_run_state,
    load_run_state_for_reporting,
    update_run_state,
)


RUN2_WORKER_JOB_PREFIX = "ga_set2"
STALE_LOCK_MIN_AGE_SECONDS = 300


def format_progress_bar(done: int, total: int, width: int = 30) -> str:
    """Format a text progress bar for queue completion.

    Args:
        done: Number of completed jobs.
        total: Total number of jobs.
        width: Character width of the rendered bar.

    Returns:
        Progress bar string with percentage.
    """
    safe_total = max(total, 1)
    ratio = done / safe_total
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    percent = 100.0 * ratio
    return f"[{bar}] {done}/{total} ({percent:.1f}%)"


def count_combinations() -> int:
    """Count combinations listed in inputs.json.

    Returns:
        Number of indexed combinations.
    """
    inputs_path = Path(__file__).parent / INPUTS_FILENAME
    if not inputs_path.exists():
        return 0

    with open(inputs_path, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)
    return len(inputs)


def load_inputs(inputs_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load run combinations from inputs.json.

    Args:
        inputs_path: Optional custom path to inputs.json.

    Returns:
        Mapping from string index to combination metadata.
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


def _looks_like_run1_worker_name(job_name: str) -> bool:
    """Check whether SLURM job name belongs to run_0 worker pools.

    Args:
        job_name: SLURM job name from squeue.

    Returns:
        True when name matches run_0 worker CPU/GPU prefixes.
    """
    return job_name.startswith(f"{RUN2_WORKER_JOB_PREFIX}_cpu") or job_name.startswith(
        f"{RUN2_WORKER_JOB_PREFIX}_gpu"
    )


def _normalize_array_task_id(raw_value: str | None) -> str | None:
    """Normalize SLURM array-task identifiers.

    Args:
        raw_value: Raw array task identifier string.

    Returns:
        Normalized array task identifier, or None for non-array sentinel values.
    """
    if raw_value is None:
        return None

    cleaned = str(raw_value).strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"n/a", "na", "none"}:
        return None

    # SLURM may expose this sentinel value for non-array jobs.
    if cleaned == "4294967294":
        return None

    return cleaned


def _normalize_worker_id_candidates(worker_id: str) -> set[str]:
    """Build robust identity candidates for one worker/job identifier.

    Args:
        worker_id: Raw worker identifier string.

    Returns:
        Candidate identifiers that should be considered equivalent.
    """
    cleaned = worker_id.strip()
    if not cleaned:
        return set()

    candidates = {cleaned}
    if "." in cleaned:
        candidates.add(cleaned.split(".", 1)[0])

    if "_" in cleaned:
        base, suffix = cleaned.split("_", 1)
        if base:
            candidates.add(base)
        if "." in suffix and base:
            candidates.add(f"{base}_{suffix.split('.', 1)[0]}")

    return {candidate for candidate in candidates if candidate}


def _get_live_run1_worker_ids() -> tuple[set[str], bool, str]:
    """Get live run_0 worker IDs currently visible in squeue.

    Returns:
        Tuple of (worker identifiers, query reliability flag, error detail).
    """
    user = os.environ.get("USER", "").strip()
    if not user:
        return set(), False, "missing_user_env"
    if shutil.which("squeue") is None:
        return set(), False, "squeue_not_found"

    result = subprocess.run(
        ["squeue", "-u", user, "-t", "R,CG", "-h", "--format=%A|%a|%i|%j"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_detail = (
            result.stderr.strip() or f"squeue_return_code_{result.returncode}"
        )
        return set(), False, error_detail

    live_ids: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        pieces = stripped.split("|", 3)
        if len(pieces) != 4:
            continue

        slurm_job_id = pieces[0].strip()
        slurm_array_task_id = _normalize_array_task_id(pieces[1])
        worker_id = pieces[2].strip()
        job_name = pieces[3].strip()
        if not _looks_like_run1_worker_name(job_name):
            continue

        live_ids.update(_normalize_worker_id_candidates(worker_id))
        live_ids.update(_normalize_worker_id_candidates(slurm_job_id))
        if slurm_job_id and slurm_array_task_id:
            live_ids.add(f"{slurm_job_id}_{slurm_array_task_id}")

    return live_ids, True, ""


def reconcile_stale_running_locks(
    queue_dir: str | Path,
    inputs: dict[str, dict[str, Any]] | None = None,
    auto_requeue: bool = False,
) -> dict[str, object]:
    """Detect stale running locks and optionally auto-requeue them.

    Args:
        queue_dir: Path to queue_state.
        inputs: Optional preloaded input mapping.
        auto_requeue: If True, move stale running locks back to pending.

    Returns:
        Dictionary with stale detection/requeue counters and live worker count.
    """
    queue_path = Path(queue_dir)
    running_dir = queue_path / "running"
    pending_dir = queue_path / "pending"

    if not running_dir.exists():
        return {
            "stale_detected": 0,
            "stale_requeued": 0,
            "live_workers": 0,
            "live_query_reliable": True,
            "live_query_error": "",
        }

    pending_dir.mkdir(parents=True, exist_ok=True)
    local_inputs = load_inputs() if inputs is None else inputs
    live_worker_ids, live_query_reliable, live_query_error = _get_live_run1_worker_ids()

    if not live_query_reliable:
        return {
            "stale_detected": 0,
            "stale_requeued": 0,
            "live_workers": int(len(live_worker_ids)),
            "live_query_reliable": False,
            "live_query_error": str(live_query_error),
        }

    stale_detected = 0
    stale_requeued = 0

    for running_lock in sorted(running_dir.glob("job_*.lock")):
        combination_idx = _extract_job_idx(running_lock.name)
        key = str(combination_idx)
        total_iterations = (
            int(local_inputs[key]["n_eval"]) if key in local_inputs else 1
        )
        state_path = get_run_state_path(queue_path, combination_idx)
        state = load_run_state_for_reporting(
            state_path=state_path,
            combination_index=combination_idx,
            total_iterations=total_iterations,
        )

        worker_id = str(state.get("slurm_worker_id", "") or "").strip()
        stale_reason = ""
        if worker_id:
            worker_candidates = _normalize_worker_id_candidates(worker_id)
            if worker_candidates and worker_candidates.isdisjoint(live_worker_ids):
                stale_reason = f"stale_worker_not_in_squeue:{worker_id}"
        else:
            lock_age_seconds = time.time() - running_lock.stat().st_mtime
            if not live_worker_ids and lock_age_seconds >= STALE_LOCK_MIN_AGE_SECONDS:
                stale_reason = "stale_worker_missing_identity_and_no_live_workers"

        if not stale_reason:
            continue

        stale_detected += 1
        if not auto_requeue:
            continue

        pending_lock = pending_dir / running_lock.name
        if pending_lock.exists():
            running_lock.unlink()
        else:
            running_lock.replace(pending_lock)

        update_run_state(
            state_path=state_path,
            combination_index=combination_idx,
            total_iterations=total_iterations,
            status="pending_failed",
            stale_recovered_count=int(state.get("stale_recovered_count", 0)) + 1,
            last_error=stale_reason,
            last_return_code=None,
        )
        stale_requeued += 1

    return {
        "stale_detected": stale_detected,
        "stale_requeued": stale_requeued,
        "live_workers": len(live_worker_ids),
        "live_query_reliable": True,
        "live_query_error": "",
    }


def get_queue_status(
    queue_dir: str | Path,
    auto_requeue_stale: bool = False,
) -> dict[str, object]:
    """Return current queue_state counts.

    Args:
        queue_dir: Path to queue_state.
        auto_requeue_stale: If True, move stale running locks back to pending
            (criteria in ``reconcile_stale_running_locks``). Stale locks are
            counted either way.

    Returns:
        Dict with ``pending``, ``running``, ``done`` and ``total`` lock counts,
        ``stale_detected``, ``stale_requeued``, ``live_workers``,
        ``live_query_reliable`` and ``live_query_error``.
    """
    queue_path = Path(queue_dir)
    pending_dir = queue_path / "pending"
    running_dir = queue_path / "running"
    done_dir = queue_path / "done"

    reconcile_summary = reconcile_stale_running_locks(
        queue_dir=queue_path,
        auto_requeue=auto_requeue_stale,
    )

    pending = len(list(pending_dir.glob("*.lock"))) if pending_dir.exists() else 0
    running = len(list(running_dir.glob("*.lock"))) if running_dir.exists() else 0
    done = len(list(done_dir.glob("*.lock"))) if done_dir.exists() else 0

    return {
        "pending": pending,
        "running": running,
        "done": done,
        "total": pending + running + done,
        "stale_detected": int(reconcile_summary["stale_detected"]),
        "stale_requeued": int(reconcile_summary["stale_requeued"]),
        "live_workers": int(reconcile_summary["live_workers"]),
        "live_query_reliable": bool(reconcile_summary.get("live_query_reliable", True)),
        "live_query_error": str(reconcile_summary.get("live_query_error", "")),
    }


def initialize_queue(
    queue_dir: str | Path,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    """Create pending lock files and run-state records for a combination range.

    Combinations that already have ``results/result_<idx>.json`` get a done
    lock and lose any pending or running lock. Combinations with an existing
    lock are left as they are. A run-state file is written for every
    combination in ``inputs.json`` that has none yet.

    Args:
        queue_dir: Path to queue_state.
        start_idx: First combination index (inclusive).
        end_idx: Last combination index (exclusive); None or a value beyond
            the number of combinations means up to the last one.
    """
    queue_path = Path(queue_dir)
    pending_dir = queue_path / "pending"
    running_dir = queue_path / "running"
    done_dir = queue_path / "done"

    results_dir = queue_path.parent / "results"

    pending_dir.mkdir(parents=True, exist_ok=True)
    running_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    total_combinations = len(inputs)
    if end_idx is None:
        end_idx = total_combinations
    end_idx = min(end_idx, total_combinations)

    created = 0
    skipped_completed = 0
    for idx in range(start_idx, end_idx):
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

        if str(idx) not in inputs:
            continue

        total_iterations = int(inputs[str(idx)]["n_eval"])
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
        f"Initialized {created} jobs in pending queue (skipped {skipped_completed} already completed in results/)"
    )
    print(f"Combination range: {start_idx} to {end_idx - 1}")
    print(f"Total combinations available: {total_combinations}")


def _extract_job_idx(lock_name: str) -> int:
    """Parse numeric job index from lock filename.
        Returns:
            int: Return value of _extract_job_idx.
    
    Returns:
        int: Return value of _extract_job_idx.
"""
    return int(lock_name.replace("job_", "").replace(".lock", ""))


def run_single_combination(combination_idx: int, queue_dir: str | Path) -> int:
    """Execute one job index via run_single_job module.

    Args:
        combination_idx: Combination index.
        queue_dir: Path to queue_state directory.

    Returns:
        Process return code.
    """
    worker_device = os.environ.get("RUN2_WORKER_DEVICE", "cpu").strip().lower()
    if worker_device not in {"cpu", "cuda"}:
        worker_device = "cpu"

    fallback_raw = os.environ.get("RUN2_WORKER_ALLOW_CPU_FALLBACK", "").strip().lower()
    if fallback_raw in {"0", "false", "no"}:
        allow_cpu_fallback = False
    elif fallback_raw in {"1", "true", "yes"}:
        allow_cpu_fallback = True
    else:
        allow_cpu_fallback = ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES

    n_eval_override_raw = os.environ.get("RUN2_WORKER_N_EVAL_OVERRIDE", "").strip()

    worker_python = os.environ.get("RUN2_WORKER_PYTHON", "").strip()
    if not worker_python:
        worker_python = sys.executable or "python"

    command = [
        worker_python,
        "-m",
        "pablo.mechthild_runs.run_0_GA_settings.run_single_job",
        str(combination_idx),
        "--queue_dir",
        str(queue_dir),
        "--device",
        worker_device,
        "--allow_cpu_fallback",
        "1" if allow_cpu_fallback else "0",
    ]

    if n_eval_override_raw.isdigit() and int(n_eval_override_raw) > 0:
        command.extend(["--n_eval_override", n_eval_override_raw])

    result = subprocess.run(command, check=False)
    return int(result.returncode)


def _format_submitted_line(
    combination_idx: int,
    combination: dict[str, Any],
    state: dict[str, Any],
) -> str:
    """Render one submitted line for running job monitoring.

    Args:
        combination_idx: Combination index.
        combination: Combination metadata from inputs.json.
        state: Run-state metadata.

    Returns:
        Formatted single-line status string.
    """
    requested_device = str(state.get("requested_device", "unknown"))
    actual_device = str(state.get("actual_device", requested_device))
    fallback_used = bool(state.get("fallback_used", False))
    fallback_reason = state.get("fallback_reason", None)
    device_path = (
        f"dev={requested_device}->{actual_device}"
        if requested_device != actual_device
        else f"dev={actual_device}"
    )
    if fallback_used and fallback_reason:
        device_path += f" ({fallback_reason})"

    error_suffix = " ERROR" if state.get("last_error") else ""
    state_warning = str(state.get("state_file_warning", "")).strip()
    warning_suffix = f" WARN={state_warning}" if state_warning else ""
    current_iteration = int(state.get("current_iteration", 0))
    total_iterations = int(state.get("total_iterations", int(combination["n_eval"])))
    return (
        f"{combination_idx} | {combination['benchmark']} | "
        f"ratio={combination['ga_ratio']} pop={combination['pop_size']} "
        f"evals={combination['n_evals_ga']} | {device_path} | "
        f"iter={current_iteration}/{total_iterations}{error_suffix}{warning_suffix}"
    )


def _format_failed_pending_line(
    combination_idx: int,
    combination: dict[str, Any],
    state: dict[str, Any],
) -> str:
    """Render one failed-pending line.

    Args:
        combination_idx: Combination index.
        combination: Combination metadata from inputs.json.
        state: Run-state metadata.

    Returns:
        Formatted single-line status string.
    """
    requested_device = str(state.get("requested_device", "unknown"))
    actual_device = str(state.get("actual_device", requested_device))
    fallback_used = bool(state.get("fallback_used", False))
    fallback_reason = state.get("fallback_reason", None)
    device_path = (
        f"dev={requested_device}->{actual_device}"
        if requested_device != actual_device
        else f"dev={actual_device}"
    )
    if fallback_used and fallback_reason:
        device_path += f" ({fallback_reason})"

    error_message = str(state.get("last_error", "unknown_error"))
    state_warning = str(state.get("state_file_warning", "")).strip()
    warning_suffix = f" | WARN={state_warning}" if state_warning else ""
    return (
        f"{combination_idx} | {combination['benchmark']} | "
        f"ratio={combination['ga_ratio']} pop={combination['pop_size']} "
        f"evals={combination['n_evals_ga']} | {device_path} | ERROR | "
        f"{error_message}{warning_suffix}"
    )


def get_submitted_lines(
    queue_dir: str | Path,
    inputs: dict[str, dict[str, Any]] | None = None,
    auto_requeue_stale: bool = False,
) -> list[str]:
    """Build per-running-job lines for the ``submitted`` command.

    Args:
        queue_dir: Path to queue_state.
        inputs: Optional preloaded input mapping; loaded from inputs.json if None.
        auto_requeue_stale: If True, move stale running locks back to pending
            (criteria in ``reconcile_stale_running_locks``). Stale locks are
            counted either way.

    Returns:
        List of formatted lines, one per running lock file, preceded by INFO or
        WARN lines about stale-lock reconciliation when relevant.
    """
    queue_path = Path(queue_dir)
    local_inputs = load_inputs() if inputs is None else inputs
    reconcile_summary = reconcile_stale_running_locks(
        queue_dir=queue_path,
        inputs=local_inputs,
        auto_requeue=auto_requeue_stale,
    )

    running_dir = queue_path / "running"
    if not running_dir.exists():
        lines: list[str] = []
        if int(reconcile_summary["stale_requeued"]) > 0:
            lines.append(
                "INFO | auto_requeued_stale_running_locks="
                f"{int(reconcile_summary['stale_requeued'])}"
            )
        return lines

    lines: list[str] = []
    if auto_requeue_stale and not bool(
        reconcile_summary.get("live_query_reliable", True)
    ):
        lines.append(
            "WARN | stale_reconcile_skipped="
            f"{str(reconcile_summary.get('live_query_error', 'unknown'))}"
        )

    if int(reconcile_summary["stale_requeued"]) > 0:
        lines.append(
            "INFO | auto_requeued_stale_running_locks="
            f"{int(reconcile_summary['stale_requeued'])}"
        )

    for lock_file in sorted(running_dir.glob("job_*.lock")):
        combination_idx = _extract_job_idx(lock_file.name)
        key = str(combination_idx)
        if key not in local_inputs:
            continue

        combination = local_inputs[key]
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
    inputs: dict[str, dict[str, Any]] | None = None,
    auto_requeue_stale: bool = False,
) -> list[str]:
    """Build lines for failed runs currently in pending.

    Args:
        queue_dir: Path to queue_state.
        inputs: Optional preloaded input mapping; loaded from inputs.json if None.
        auto_requeue_stale: If True, move stale running locks back to pending
            (criteria in ``reconcile_stale_running_locks``). Stale locks are
            counted either way.

    Returns:
        List of formatted lines for pending_failed jobs only.
    """
    queue_path = Path(queue_dir)
    local_inputs = load_inputs() if inputs is None else inputs
    reconcile_stale_running_locks(
        queue_dir=queue_path,
        inputs=local_inputs,
        auto_requeue=auto_requeue_stale,
    )

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


def _claim_next_job(pending_dir: Path, running_dir: Path) -> int | None:
    """Atomically claim one pending lock via shell mv -n.

    Race-condition failures (another worker claimed first) are expected and
    their stderr is suppressed to keep SLURM .err logs clean.

    Args:
        pending_dir: queue_state/pending path.
        running_dir: queue_state/running path.

    Returns:
        Claimed job index, or None when no claim succeeds.
    """
    pending_files = sorted(pending_dir.glob("*.lock"))
    for pending_file in pending_files:
        target_file = running_dir / pending_file.name
        claim = subprocess.run(
            ["mv", "-n", str(pending_file), str(target_file)],
            check=False,
            stderr=subprocess.DEVNULL,
        )
        if claim.returncode == 0 and target_file.exists():
            return _extract_job_idx(target_file.name)
    return None


def _resolve_slurm_worker_identity() -> tuple[str | None, str | None, str | None]:
    """Resolve SLURM worker identity from environment variables.

    Returns:
        Tuple (slurm_job_id, slurm_array_task_id, slurm_worker_id).
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip() or None
    slurm_array_task_id = _normalize_array_task_id(
        os.environ.get("SLURM_ARRAY_TASK_ID", "")
    )

    slurm_worker_id = None
    if slurm_job_id and slurm_array_task_id:
        slurm_worker_id = f"{slurm_job_id}_{slurm_array_task_id}"
    elif slurm_job_id:
        slurm_worker_id = slurm_job_id

    return slurm_job_id, slurm_array_task_id, slurm_worker_id


def worker_loop(queue_dir: str | Path) -> None:
    """Poll queue_state and execute jobs until pending is empty.

    Args:
        queue_dir: Path to queue_state.
    """
    queue_path = Path(queue_dir)
    pending_dir = queue_path / "pending"
    running_dir = queue_path / "running"
    done_dir = queue_path / "done"

    pending_dir.mkdir(parents=True, exist_ok=True)
    running_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    print(f"Worker started. Queue directory: {queue_path}")
    inputs = load_inputs()
    slurm_job_id, slurm_array_task_id, slurm_worker_id = (
        _resolve_slurm_worker_identity()
    )

    while True:
        if not any(pending_dir.glob("*.lock")):
            print("No pending jobs left. Worker exiting.")
            break

        job_idx = _claim_next_job(pending_dir, running_dir)
        if job_idx is None:
            continue

        key = str(job_idx)
        if key not in inputs:
            raise ValueError(f"Combination index {job_idx} not found in inputs.json")
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

        print(f"Processing combination {job_idx}")
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
            print(f"Completed combination {job_idx}")
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
                f"Combination {job_idx} failed with code {return_code}, returned to pending"
            )


def main() -> None:
    """CLI for queue initialization, worker execution, and status checks."""
    default_queue_dir = str(Path(__file__).parent / QUEUE_DIRNAME)

    parser = argparse.ArgumentParser(
        description="Manage run_0 GA settings SLURM queue state"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser("init", help="Initialize queue_state locks")
    init_parser.add_argument(
        "--queue_dir",
        default=default_queue_dir,
        help="Path to queue_state directory",
    )
    init_parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive)",
    )
    init_parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive)",
    )

    worker_parser = subparsers.add_parser("worker", help="Run worker loop")
    worker_parser.add_argument(
        "--queue_dir",
        default=default_queue_dir,
        help="Path to queue_state directory",
    )

    status_parser = subparsers.add_parser("status", help="Show queue_state counts")
    status_parser.add_argument(
        "--queue_dir",
        default=default_queue_dir,
        help="Path to queue_state directory",
    )
    status_requeue_group = status_parser.add_mutually_exclusive_group()
    status_requeue_group.add_argument(
        "--auto_requeue_stale",
        action="store_true",
        help="Enable automatic stale-running reconciliation",
    )
    status_requeue_group.add_argument(
        "--no_auto_requeue_stale",
        action="store_true",
        help="Compatibility flag (default behavior is already no auto-requeue)",
    )

    submitted_parser = subparsers.add_parser(
        "submitted",
        help="Show one line per currently running run",
    )
    submitted_parser.add_argument(
        "--queue_dir",
        default=default_queue_dir,
        help="Path to queue_state directory",
    )
    submitted_requeue_group = submitted_parser.add_mutually_exclusive_group()
    submitted_requeue_group.add_argument(
        "--auto_requeue_stale",
        action="store_true",
        help="Enable automatic stale-running reconciliation",
    )
    submitted_requeue_group.add_argument(
        "--no_auto_requeue_stale",
        action="store_true",
        help="Compatibility flag (default behavior is already no auto-requeue)",
    )

    failed_pending_parser = subparsers.add_parser(
        "failed-pending",
        help="Show only failed runs currently in pending",
    )
    failed_pending_parser.add_argument(
        "--queue_dir",
        default=default_queue_dir,
        help="Path to queue_state directory",
    )
    failed_pending_requeue_group = failed_pending_parser.add_mutually_exclusive_group()
    failed_pending_requeue_group.add_argument(
        "--auto_requeue_stale",
        action="store_true",
        help="Enable automatic stale-running reconciliation",
    )
    failed_pending_requeue_group.add_argument(
        "--no_auto_requeue_stale",
        action="store_true",
        help="Compatibility flag (default behavior is already no auto-requeue)",
    )

    args = parser.parse_args()

    if args.command == "init":
        initialize_queue(args.queue_dir, start_idx=args.start, end_idx=args.end)
        return

    if args.command == "worker":
        worker_loop(args.queue_dir)
        return

    if args.command == "status":
        status = get_queue_status(
            args.queue_dir,
            auto_requeue_stale=bool(args.auto_requeue_stale),
        )
        print("Queue Status:")
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
            args.queue_dir,
            auto_requeue_stale=bool(args.auto_requeue_stale),
        )
        for line in lines:
            print(line)
        return

    if args.command == "failed-pending":
        lines = get_failed_pending_lines(
            args.queue_dir,
            auto_requeue_stale=bool(args.auto_requeue_stale),
        )
        for line in lines:
            print(line)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
