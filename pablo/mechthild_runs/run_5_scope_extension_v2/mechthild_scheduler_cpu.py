"""File-lock queue of run_5, one queue per operating mode.

A mode's queue directory holds ``pending/``, ``running/`` and ``done/`` with one
``job_<idx>.lock`` per combination of that mode in ``inputs.json``, and the
per-job state files of ``run_state.py``. A worker claims one pending lock by
renaming it into ``running/`` (only one of several concurrent renames of a file
succeeds), runs the job through ``run_single_job`` in a subprocess, moves the
lock to ``done/`` on success or back to ``pending/`` (status ``pending_failed``)
on failure, and exits: one SLURM array task runs one job.

Run from the repo root:

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu <command> \
        --operating_mode MODE [--queue_dir DIR]

Commands:
    init: Create pending locks and states for the mode's jobs (``--start``, ``--end``).
    worker: Claim and run one pending job of the mode.
    status: Lock counts, a progress bar and the mode's live SLURM workers.
    submitted: One line per running job with its iteration progress.
    failed-pending: One line per failed job that was returned to pending.
    requeue-stale: Return running jobs whose SLURM worker is gone to pending.

``--queue_dir`` defaults to the mode's queue folder next to this file.
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

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    ALL_OPERATING_MODES,
    INPUTS_PATH,
    combination_operating_mode,
    job_name,
    queue_dir,
    results_dir,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.run_state import (
    check_existing_result,
    get_run_state_path,
    load_run_state,
    update_run_state,
)

QUEUE_SUBDIRS = ("pending", "running", "done")


def load_inputs() -> dict[str, dict[str, Any]]:
    """Combinations of ``inputs.json`` keyed by their string index."""
    with open(INPUTS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def lock_indices(directory: Path, operating_mode: str, inputs: dict[str, dict[str, Any]]) -> list[int]:
    """Sorted indices of the ``job_<idx>.lock`` files in ``directory`` that belong to ``operating_mode``.

    Raises:
        ValueError: If a lock names an index that is not in ``inputs.json``.
    """
    indices = []
    for lock in directory.glob("job_*.lock"):
        idx = int(lock.stem.removeprefix("job_"))
        if str(idx) not in inputs:
            raise ValueError(f"{lock} names combination {idx}, which is not in {INPUTS_PATH.name}")
        if combination_operating_mode(inputs[str(idx)]) == operating_mode:
            indices.append(idx)
    return sorted(indices)


def _queue_dirs(queue_path: Path) -> dict[str, Path]:
    """``pending``/``running``/``done`` folders of a queue, created if missing."""
    dirs = {name: queue_path / name for name in QUEUE_SUBDIRS}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def format_progress_bar(done: int, total: int, width: int = 30) -> str:
    """``[###...] done/total (pp.p%)``; an empty queue renders as an empty bar."""
    ratio = done / total if total else 0.0
    filled = int(width * ratio)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total} ({100.0 * ratio:.1f}%)"


def initialize_queue(
    queue_path: Path,
    operating_mode: str,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    """Create a pending lock and a state file for every unfinished job of one mode.

    A job with a result file gets a ``done`` lock (after ``check_existing_result``);
    a job that already has a lock keeps it.

    Args:
        queue_path: Queue directory of the mode.
        operating_mode: Mode this queue manages.
        start_idx: First combination index considered (inclusive).
        end_idx: Last combination index considered (exclusive); all when ``None``.

    Raises:
        ValueError: If an existing result file is not a full run of its
            combination, or a ``done`` lock has no result file.
    """
    dirs = _queue_dirs(queue_path)
    inputs = load_inputs()
    end_idx = len(inputs) if end_idx is None else min(end_idx, len(inputs))
    mode_results = results_dir(operating_mode)

    created = completed = kept = 0
    for idx in range(start_idx, end_idx):
        combination = inputs[str(idx)]
        if combination_operating_mode(combination) != operating_mode:
            continue
        locks = {name: directory / f"job_{idx}.lock" for name, directory in dirs.items()}
        result_file = mode_results / f"result_{idx}.json"
        if result_file.exists():
            check_existing_result(result_file, combination)
            locks["done"].touch()
            locks["pending"].unlink(missing_ok=True)
            locks["running"].unlink(missing_ok=True)
            status = "done"
            completed += 1
        elif locks["done"].exists():
            raise ValueError(
                f"{locks['done']} has no result file {result_file}; move the old queue "
                f"folder {queue_path} out before init"
            )
        elif locks["running"].exists():
            status = "running"
            kept += 1
        else:
            locks["pending"].touch()
            status = "pending_waiting"
            created += 1

        state_path = get_run_state_path(queue_path, idx)
        if not state_path.exists():
            update_run_state(state_path, idx, int(combination["n_eval"]), status=status)

    print(f"[{operating_mode}] queue {queue_path}, combinations {start_idx}..{end_idx - 1}")
    print(f"  new pending locks: {created}, pending in total: {len(lock_indices(dirs['pending'], operating_mode, inputs))}")
    print(f"  completed in {mode_results.name}/: {completed}")
    print(f"  running locks kept: {kept}")


def _slurm_ids() -> tuple[str | None, str | None]:
    """``SLURM_JOB_ID`` (unique per array task) and ``SLURM_ARRAY_TASK_ID`` of this process."""
    return os.environ.get("SLURM_JOB_ID") or None, os.environ.get("SLURM_ARRAY_TASK_ID") or None


def _claim_next_job(dirs: dict[str, Path], operating_mode: str, inputs: dict[str, dict[str, Any]]) -> int | None:
    """Move the first claimable pending lock of the mode into ``running/``.

    Returns:
        The claimed combination index, or ``None`` if no pending lock is left.
    """
    for idx in lock_indices(dirs["pending"], operating_mode, inputs):
        name = f"job_{idx}.lock"
        claim = subprocess.run(
            ["mv", str(dirs["pending"] / name), str(dirs["running"] / name)],
            check=False,
            stderr=subprocess.DEVNULL,
        )
        if claim.returncode == 0:
            return idx
    return None


def run_worker(queue_path: Path, operating_mode: str) -> None:
    """Claim one pending job of the mode, run it, and record the outcome.

    On success the lock moves to ``done/``; on failure it returns to
    ``pending/`` with status ``pending_failed`` and the subprocess exit code.

    Args:
        queue_path: Queue directory of the mode.
        operating_mode: Mode whose pending jobs are claimed.
    """
    dirs = _queue_dirs(queue_path)
    inputs = load_inputs()
    job_idx = _claim_next_job(dirs, operating_mode, inputs)
    if job_idx is None:
        print(f"[{operating_mode}] No pending job left.")
        return

    total_iterations = int(inputs[str(job_idx)]["n_eval"])
    state_path = get_run_state_path(queue_path, job_idx)
    attempts = int(load_run_state(state_path, job_idx, total_iterations)["attempts"]) + 1
    slurm_job_id, slurm_array_task_id = _slurm_ids()
    update_run_state(
        state_path,
        job_idx,
        total_iterations,
        status="running",
        attempts=attempts,
        current_iteration=0,
        slurm_job_id=slurm_job_id,
        slurm_array_task_id=slurm_array_task_id,
        last_error=None,
        last_return_code=None,
    )

    print(f"[{operating_mode}] Running combination {job_idx} (attempt {attempts})", flush=True)
    return_code = subprocess.run(
        [
            sys.executable,
            "-m",
            "pablo.mechthild_runs.run_5_scope_extension_v2.run_single_job",
            str(job_idx),
            "--queue_dir",
            str(queue_path),
        ],
        check=False,
    ).returncode

    name = f"job_{job_idx}.lock"
    if return_code == 0:
        (dirs["running"] / name).replace(dirs["done"] / name)
        update_run_state(state_path, job_idx, total_iterations, status="done", last_return_code=0)
        print(f"[{operating_mode}] Completed combination {job_idx}")
    else:
        (dirs["running"] / name).replace(dirs["pending"] / name)
        update_run_state(
            state_path,
            job_idx,
            total_iterations,
            status="pending_failed",
            last_error=f"exit_code={return_code}",
            last_return_code=return_code,
        )
        print(f"[{operating_mode}] Combination {job_idx} failed with code {return_code}, returned to pending")


def squeue_lines(fields: str) -> list[str] | None:
    """``squeue -u $USER -h -r -o <fields>`` lines, or ``None`` where ``squeue`` is not installed.

    Raises:
        subprocess.CalledProcessError: If ``squeue`` fails.
    """
    if shutil.which("squeue") is None:
        return None
    result = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h", "-r", "-o", fields],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def requeue_stale(queue_path: Path, operating_mode: str) -> None:
    """Return running jobs whose recorded SLURM job is no longer queued to ``pending/``.

    A worker killed by the time limit or the node leaves its lock in
    ``running/`` with status ``running``; such a job restarts from iteration 0
    when claimed again. A lock whose state is not ``running`` yet (a claim in
    progress) is left alone.

    Args:
        queue_path: Queue directory of the mode.
        operating_mode: Mode whose running jobs are checked.

    Raises:
        RuntimeError: If ``squeue`` is not available.
    """
    lines = squeue_lines("%A")
    if lines is None:
        raise RuntimeError("requeue-stale needs squeue")
    live = {line.strip() for line in lines}
    dirs = _queue_dirs(queue_path)
    inputs = load_inputs()
    requeued = []
    for idx in lock_indices(dirs["running"], operating_mode, inputs):
        state_path = get_run_state_path(queue_path, idx)
        state = load_run_state(state_path, idx, int(inputs[str(idx)]["n_eval"]))
        if state["status"] != "running" or state["slurm_job_id"] in live:
            continue
        name = f"job_{idx}.lock"
        (dirs["running"] / name).replace(dirs["pending"] / name)
        update_run_state(
            state_path,
            idx,
            int(inputs[str(idx)]["n_eval"]),
            status="pending_failed",
            last_error=f"stale: SLURM job {state['slurm_job_id']} gone at iteration {state['current_iteration']}",
        )
        requeued.append(idx)
    print(f"[{operating_mode}] Requeued {len(requeued)} stale jobs: {requeued}")


def print_status(queue_path: Path, operating_mode: str) -> None:
    """Print lock counts, a progress bar and the mode's live SLURM workers."""
    dirs = _queue_dirs(queue_path)
    inputs = load_inputs()
    counts = {name: len(lock_indices(d, operating_mode, inputs)) for name, d in dirs.items()}
    total = sum(counts.values())
    print(f"[{operating_mode}] {queue_path}")
    print(f"  Progress: {format_progress_bar(counts['done'], total)}")
    print(f"  Pending: {counts['pending']}  Running: {counts['running']}  Done: {counts['done']}")
    lines = squeue_lines("%j %T")
    if lines is None:
        print("  SLURM workers: squeue not available")
        return
    states = [line.split()[1] for line in lines if line.split()[0] == job_name(operating_mode)]
    print(f"  SLURM workers: {states.count('RUNNING')} running, {states.count('PENDING')} pending")


def _job_lines(queue_path: Path, operating_mode: str, subdir: str, status: str | None) -> list[str]:
    """One line per lock in ``subdir`` of the mode, optionally only jobs with run-state ``status``."""
    dirs = _queue_dirs(queue_path)
    inputs = load_inputs()
    lines = []
    for idx in lock_indices(dirs[subdir], operating_mode, inputs):
        combination = inputs[str(idx)]
        state = load_run_state(get_run_state_path(queue_path, idx), idx, int(combination["n_eval"]))
        if status is not None and state["status"] != status:
            continue
        line = (
            f"{idx} | {combination['label']} | iter={state['current_iteration']}/{state['total_iterations']}"
            f" | attempts={state['attempts']} | slurm_job={state['slurm_job_id']}"
        )
        if state["last_error"]:
            line += f" | ERROR {state['last_error']}"
        lines.append(line)
    return lines


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="run_5 file-lock queue, one per operating mode")
    parser.add_argument(
        "command", choices=["init", "worker", "status", "submitted", "failed-pending", "requeue-stale"]
    )
    parser.add_argument("--operating_mode", choices=ALL_OPERATING_MODES, required=True)
    parser.add_argument("--queue_dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0, help="init: first combination index")
    parser.add_argument("--end", type=int, default=None, help="init: last combination index (exclusive)")
    args = parser.parse_args()

    mode = str(args.operating_mode)
    queue_path = queue_dir(mode) if args.queue_dir is None else Path(args.queue_dir)
    if args.command == "init":
        initialize_queue(queue_path, mode, args.start, args.end)
    elif args.command == "worker":
        run_worker(queue_path, mode)
    elif args.command == "status":
        print_status(queue_path, mode)
    elif args.command == "submitted":
        print("\n".join(_job_lines(queue_path, mode, "running", None)))
    elif args.command == "failed-pending":
        print("\n".join(_job_lines(queue_path, mode, "pending", "pending_failed")))
    else:
        requeue_stale(queue_path, mode)


if __name__ == "__main__":
    main()
