"""CPU-only SLURM submitter for the run_1 scope-extension sweep, one operating mode per call.

Usage:
    python -m pablo.mechthild_runs.run_1_scope_extension.submit_mechthild_cpu [submit|status|autoscale] \
        [--operating_mode MODE] [--queue_dir DIR] [--partition NAME] [--force] \
        [--poll_interval_seconds N]

``submit`` (default) submits one worker array after a confirmation prompt, ``status``
prints the capacity snapshot, ``autoscale`` keeps submitting until the queue drains.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from pablo.mechthild_runs.run_1_scope_extension.config import (
    ALL_OPERATING_MODES,
    LOGS_DIRNAME,
    MYOPIC_OPERATING_MODE,
    get_cpu_partition_for_operating_mode,
    get_job_name_prefix_for_operating_mode,
    get_queue_dirname_for_operating_mode,
    get_time_limit_for_operating_mode,
)

CPUS_PER_TASK = 1
MEMORY = "12G"
EMAIL = "enisbaris@mpi-magdeburg.mpg.de"
MAX_CONCURRENT_WORKERS = 512
DEFAULT_POLL_INTERVAL_SECONDS = 30


def _run_capture(command: list[str]) -> subprocess.CompletedProcess:
    """Run one command and capture output without raising.
        Returns:
            subprocess.CompletedProcess: Return value of _run_capture.
    
    Returns:
        subprocess.CompletedProcess: Return value of _run_capture.
"""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _resolve_queue_dir(queue_dir: str | None, operating_mode: str) -> Path:
    """Resolve queue path from explicit arg or per-mode default.
        Returns:
            Path: Return value of _resolve_queue_dir.
    
    Returns:
        Path: Return value of _resolve_queue_dir.
"""
    if queue_dir is None:
        return Path(__file__).parent / get_queue_dirname_for_operating_mode(operating_mode)
    return Path(queue_dir)


def _resolve_partition(partition: str | None, operating_mode: str) -> str:
    """Resolve SLURM partition from explicit arg or per-mode default.
        Returns:
            str: Return value of _resolve_partition.
    
    Returns:
        str: Return value of _resolve_partition.
"""
    if partition is None:
        return get_cpu_partition_for_operating_mode(operating_mode)
    return partition


def get_capacity_snapshot(
    queue_dir: str | Path,
    operating_mode: str,
    partition: str,
) -> dict[str, int | bool]:
    """Collect queue and worker capacity metrics for one operating mode.

    Jobs are the ``*.lock`` files in ``pending/`` and ``running/``. Workers are the
    user's ``squeue`` entries whose job name contains the mode's job-name prefix. If any
    pending worker waits on an ASSOCMAX, QOSMAX, MAXJOBS or MAXSUBMIT reason, no new
    slots are offered.

    Args:
        queue_dir: The mode's queue directory.
        operating_mode: One of ``ALL_OPERATING_MODES``.
        partition: SLURM partition name; not used in the computation.

    Returns:
        Dict with keys ``pending_jobs``, ``running_jobs``, ``uncovered_jobs`` (jobs
        not matched by a SLURM worker), ``active_workers``, ``pending_workers``,
        ``submission_throttled``, ``effective_slots`` (``MAX_CONCURRENT_WORKERS``
        minus current workers, 0 when throttled) and ``workers_to_submit``
        (min of uncovered jobs and effective slots).
    """
    queue_path = Path(queue_dir)
    pending_dir = queue_path / "pending"
    running_dir = queue_path / "running"

    pending_jobs = len(list(pending_dir.glob("*.lock"))) if pending_dir.exists() else 0
    running_jobs = len(list(running_dir.glob("*.lock"))) if running_dir.exists() else 0

    active_workers = 0
    pending_workers = 0
    submission_throttled = False

    user = os.environ.get("USER", "")
    job_prefix = get_job_name_prefix_for_operating_mode(operating_mode)
    squeue_res = _run_capture(["squeue", "-u", user, "-h", "-o", "%j %T %r"])
    if squeue_res.returncode == 0:
        for line in squeue_res.stdout.splitlines():
            if job_prefix not in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            state = parts[1]
            if state in ("RUNNING", "COMPLETING", "CONFIGURING"):
                active_workers += 1
            elif state == "PENDING":
                pending_workers += 1
                if len(parts) >= 3:
                    reason = parts[2]
                    if any(
                        token in reason
                        for token in ("ASSOCMAX", "QOSMAX", "MAXJOBS", "MAXSUBMIT")
                    ):
                        submission_throttled = True

    total_slurm_workers = active_workers + pending_workers
    uncovered_jobs = max(0, pending_jobs + running_jobs - total_slurm_workers)
    available_slots = max(0, MAX_CONCURRENT_WORKERS - total_slurm_workers)

    if submission_throttled:
        available_slots = 0

    workers_to_submit = min(uncovered_jobs, available_slots)

    return {
        "pending_jobs": pending_jobs,
        "running_jobs": running_jobs,
        "uncovered_jobs": uncovered_jobs,
        "active_workers": active_workers,
        "pending_workers": pending_workers,
        "submission_throttled": submission_throttled,
        "effective_slots": available_slots,
        "workers_to_submit": workers_to_submit,
    }


def print_capacity_status(
    snapshot: dict[str, int | bool],
    operating_mode: str,
    partition: str,
) -> None:
    """Print the capacity snapshot as a readable table.

    Args:
        snapshot: Dict returned by ``get_capacity_snapshot``.
        operating_mode: One of ``ALL_OPERATING_MODES``; selects the job-name prefix shown.
        partition: SLURM partition name shown in the header.
    """
    job_prefix = get_job_name_prefix_for_operating_mode(operating_mode)
    print(f"--- Queue Status ({job_prefix} / {partition}) ---")
    print(f"Pending experiment jobs : {snapshot['pending_jobs']}")
    print(f"Running experiment jobs : {snapshot['running_jobs']}")
    print(f"Active SLURM workers    : {snapshot['active_workers']}")
    print(f"Pending SLURM workers   : {snapshot['pending_workers']}")
    print(f"Max Concurrent Allowed  : {MAX_CONCURRENT_WORKERS}")
    print(f"Uncovered jobs          : {snapshot['uncovered_jobs']}")
    print(f"Throttle active         : {snapshot['submission_throttled']}")
    print(f"New workers to submit   : {snapshot['workers_to_submit']}")
    print("-------------------------------------------------")


def submit_array_job(
    num_workers: int,
    queue_dir: str | Path,
    operating_mode: str,
    partition: str,
) -> None:
    """Write one SLURM array script and submit it with sbatch.

    The script is written to ``.submit_dynamic.sh`` next to this file; each array task
    runs one ``mechthild_scheduler_cpu worker`` on the queue. Logs go to
    ``LOGS_DIRNAME``, which is created if missing.

    Args:
        num_workers: Number of array tasks.
        queue_dir: The mode's queue directory.
        operating_mode: One of ``ALL_OPERATING_MODES``; sets job name and time limit.
        partition: SLURM partition to submit to.

    Raises:
        subprocess.CalledProcessError: If ``sbatch`` fails.
    """
    queue_path = Path(queue_dir).resolve()
    script_dir = Path(__file__).parent.resolve()
    logs_dir = script_dir / LOGS_DIRNAME
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_prefix = get_job_name_prefix_for_operating_mode(operating_mode)
    time_limit = get_time_limit_for_operating_mode(operating_mode)
    scheduler_module = "pablo.mechthild_runs.run_1_scope_extension.mechthild_scheduler_cpu"

    slurm_script = f"""#!/bin/bash
#SBATCH -J {job_prefix}
#SBATCH --array=1-{num_workers}%{MAX_CONCURRENT_WORKERS}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={EMAIL}
#SBATCH -o {LOGS_DIRNAME}/worker_%A_%a.out
#SBATCH -e {LOGS_DIRNAME}/worker_%A_%a.err

export PYTHONPATH=\"$(pwd):$PYTHONPATH\"
echo \"Starting array task $SLURM_ARRAY_TASK_ID on node $SLURM_NODENAME\"
python -m {scheduler_module} worker --queue_dir \"{queue_path}\" --operating_mode {operating_mode}
"""

    submit_script_path = script_dir / ".submit_dynamic.sh"
    submit_script_path.write_text(slurm_script)

    print(
        f"Submitting {num_workers} workers for mode {operating_mode} "
        f"to partition {partition} (max concurrency: {MAX_CONCURRENT_WORKERS})"
    )
    subprocess.run(["sbatch", str(submit_script_path)], check=True)


def run_submit_once(
    queue_dir: str | Path,
    operating_mode: str,
    partition: str,
    force: bool = False,
) -> None:
    """Print the capacity snapshot and submit one worker array for the uncovered jobs.

    Args:
        queue_dir: The mode's queue directory.
        operating_mode: One of ``ALL_OPERATING_MODES``.
        partition: SLURM partition to submit to.
        force: Skip the ``[y/N]`` confirmation prompt.
    """
    snapshot = get_capacity_snapshot(queue_dir, operating_mode, partition)
    print_capacity_status(snapshot, operating_mode, partition)

    workers = int(snapshot["workers_to_submit"])
    if workers <= 0:
        print("Nothing to submit.")
        return

    if not force:
        choice = input(f"Submit {workers} workers? [y/N]: ")
        if choice.strip().lower() != "y":
            print("Aborting.")
            return

    submit_array_job(
        num_workers=workers,
        queue_dir=queue_dir,
        operating_mode=operating_mode,
        partition=partition,
    )


def run_autoscale(
    queue_dir: str | Path,
    operating_mode: str,
    partition: str,
    poll_interval_seconds: int,
) -> None:
    """Poll capacity and submit worker arrays until no jobs or workers remain.

    Exits when there are no pending jobs and no active or pending SLURM workers.

    Args:
        queue_dir: The mode's queue directory.
        operating_mode: One of ``ALL_OPERATING_MODES``.
        partition: SLURM partition to submit to.
        poll_interval_seconds: Sleep between polls, in seconds.
    """
    print(
        f"Starting autoscaler for mode={operating_mode} partition={partition} "
        "(Ctrl+C to stop)"
    )
    while True:
        snapshot = get_capacity_snapshot(queue_dir, operating_mode, partition)
        workers = int(snapshot["workers_to_submit"])

        if (
            int(snapshot["pending_jobs"]) == 0
            and int(snapshot["active_workers"]) == 0
            and int(snapshot["pending_workers"]) == 0
        ):
            print("All jobs finished. Exiting autoscaler.")
            break

        if workers > 0:
            print(f"Submitting {workers} new SLURM workers...")
            submit_array_job(
                num_workers=workers,
                queue_dir=queue_dir,
                operating_mode=operating_mode,
                partition=partition,
            )
        else:
            print(
                "[Autoscaler] No new slots or jobs. "
                f"(Pending jobs: {snapshot['pending_jobs']})"
            )

        time.sleep(poll_interval_seconds)


def main() -> None:
    """CLI entry point: parse arguments and run ``submit``, ``status`` or ``autoscale``.

    Queue directory and partition default to the per-mode values from ``config``.
    """
    parser = argparse.ArgumentParser(description="Submit run_1 CPU workers by mode")
    parser.add_argument(
        "command",
        nargs="?",
        default="submit",
        choices=["submit", "status", "autoscale"],
    )
    parser.add_argument(
        "--operating_mode",
        choices=ALL_OPERATING_MODES,
        default=MYOPIC_OPERATING_MODE,
    )
    parser.add_argument("--queue_dir", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )

    args = parser.parse_args()
    operating_mode = str(args.operating_mode)
    queue_dir = _resolve_queue_dir(args.queue_dir, operating_mode)
    partition = _resolve_partition(args.partition, operating_mode)

    if args.command == "status":
        snapshot = get_capacity_snapshot(queue_dir, operating_mode, partition)
        print_capacity_status(snapshot, operating_mode, partition)
        return

    if args.command == "autoscale":
        run_autoscale(
            queue_dir=queue_dir,
            operating_mode=operating_mode,
            partition=partition,
            poll_interval_seconds=int(args.poll_interval_seconds),
        )
        return

    run_submit_once(
        queue_dir=queue_dir,
        operating_mode=operating_mode,
        partition=partition,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
