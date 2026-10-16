"""SLURM submitter for the run_4 AF-optimizer comparison.

One SLURM array task = one worker = one ``scheduler worker --single`` call, which
claims one queued ``(rep, model, scheme)`` cell and runs all N_TIERS tiers for it
(six optimizations). The array spans all uncovered cells (N_WORKERS = 200), capped at
SLURM_MAX_CONCURRENT_WORKERS. ``SLURM_TIME_LIMIT`` (config) must cover the summed
six-tier wall time reported by
``python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant calibrate``
— set it before submitting.

Subcommands:
  status     Print queue + SLURM worker status.
  submit     Submit one SLURM array covering uncovered pending jobs.
  autoscale  Loop, submitting new arrays as slots open.

Usage:
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.submit_mechthild status
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.submit_mechthild submit [--force]
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.submit_mechthild autoscale
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from pablo.mechthild_runs.run_4_af_opt_strategy.config import (
    SLURM_CPUS_PER_TASK,
    SLURM_EMAIL,
    SLURM_MAX_CONCURRENT_WORKERS,
    SLURM_MEMORY,
    SLURM_PARTITION,
    SLURM_POLL_INTERVAL_SECONDS,
    SLURM_TIME_LIMIT,
)

JOB_NAME_PREFIX = "af_opt_cmp"

ROOT = Path(__file__).parent
QUEUE_DIR = ROOT / "data" / "queue"
LOGS_DIR = ROOT / "data" / "logs"
SCHEDULER_MODULE = "pablo.mechthild_runs.run_4_af_opt_strategy.scheduler"


def _capacity_snapshot() -> dict[str, int | bool]:
    """Compute queue and SLURM worker counts and how many workers to submit.

    Returns:
        A dict with pending/running job counts, active/pending SLURM worker
        counts, a throttle flag, the number of uncovered jobs, and the number
        of workers that can be submitted now.
    """
    pending = (
        len(list((QUEUE_DIR / "pending").glob("job_*.lock")))
        if (QUEUE_DIR / "pending").exists()
        else 0
    )
    running = (
        len(list((QUEUE_DIR / "running").glob("job_*.lock")))
        if (QUEUE_DIR / "running").exists()
        else 0
    )

    active = pending_w = 0
    throttled = False
    user = os.environ.get("USER", "")
    res = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%j %T %r"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode == 0:
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2 or JOB_NAME_PREFIX not in parts[0]:
                continue
            if parts[1] in ("RUNNING", "COMPLETING", "CONFIGURING"):
                active += 1
            elif parts[1] == "PENDING":
                pending_w += 1
                if len(parts) >= 3 and any(
                    t in parts[2] for t in ("ASSOCMAX", "QOSMAX", "MAXJOBS", "MAXSUBMIT")
                ):
                    throttled = True

    total_workers = active + pending_w
    uncovered = max(0, pending + running - total_workers)
    available = 0 if throttled else max(0, SLURM_MAX_CONCURRENT_WORKERS - total_workers)
    return {
        "pending_jobs": pending,
        "running_jobs": running,
        "uncovered_jobs": uncovered,
        "active_workers": active,
        "pending_workers": pending_w,
        "throttled": throttled,
        "workers_to_submit": min(uncovered, available),
    }


def _print_status(snap: dict) -> None:
    """Print a capacity snapshot.

    Args:
        snap: Dict returned by :func:`_capacity_snapshot`.
    """
    print(f"--- {JOB_NAME_PREFIX} / {SLURM_PARTITION} ---")
    print(f"Pending jobs        : {snap['pending_jobs']}")
    print(f"Running jobs        : {snap['running_jobs']}")
    print(f"Active SLURM workers: {snap['active_workers']}")
    print(f"Pending SLURM wrkrs : {snap['pending_workers']}")
    print(f"Max concurrent      : {SLURM_MAX_CONCURRENT_WORKERS}")
    print(f"Uncovered jobs      : {snap['uncovered_jobs']}")
    print(f"Workers to submit   : {snap['workers_to_submit']}")


def _submit_array(num_workers: int) -> None:
    """Write and ``sbatch`` an array job covering ``num_workers`` workers.

    Args:
        num_workers: Number of array tasks to submit.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/bash
#SBATCH -J {JOB_NAME_PREFIX}
#SBATCH --array=1-{num_workers}%{SLURM_MAX_CONCURRENT_WORKERS}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --mem={SLURM_MEMORY}
#SBATCH --time={SLURM_TIME_LIMIT}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={SLURM_EMAIL}
#SBATCH -o {LOGS_DIR}/worker_%A_%a.out
#SBATCH -e {LOGS_DIR}/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS={SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS={SLURM_CPUS_PER_TASK}
export TORCH_NUM_THREADS={SLURM_CPUS_PER_TASK}

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m {SCHEDULER_MODULE} worker --single
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
"""
    script_path = ROOT / ".submit_array.sh"
    script_path.write_text(script)
    print(f"Submitting {num_workers} workers to {SLURM_PARTITION} (cap {SLURM_MAX_CONCURRENT_WORKERS})")
    subprocess.run(["sbatch", str(script_path)], check=True)


def submit_once(force: bool) -> None:
    """Submit one array covering currently uncovered pending jobs.

    Args:
        force: Skip the interactive confirmation when True.
    """
    snap = _capacity_snapshot()
    _print_status(snap)
    n = int(snap["workers_to_submit"])
    if n <= 0:
        print("Nothing to submit.")
        return
    if not force:
        choice = input(f"Submit {n} workers? [y/N]: ")
        if choice.strip().lower() != "y":
            print("Aborting.")
            return
    _submit_array(n)


def autoscale(poll_interval: int) -> None:
    """Repeatedly submit arrays as SLURM slots free up until the queue drains.

    Args:
        poll_interval: Seconds to sleep between capacity checks.
    """
    print(f"Autoscaler: partition={SLURM_PARTITION}, poll={poll_interval}s.")
    while True:
        snap = _capacity_snapshot()
        if all(
            snap[k] == 0
            for k in ("pending_jobs", "running_jobs", "active_workers", "pending_workers")
        ):
            print("All done. Exiting autoscaler.")
            return
        n = int(snap["workers_to_submit"])
        if n > 0:
            print(f"[autoscale] submitting {n} workers")
            _submit_array(n)
        else:
            print(
                f"[autoscale] no slots (pending={snap['pending_jobs']}, "
                f"running={snap['running_jobs']})"
            )
        time.sleep(poll_interval)


def main() -> None:
    """CLI entrypoint: ``submit [--force]`` | ``status`` | ``autoscale``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="submit",
        choices=["submit", "status", "autoscale"],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--poll_interval_seconds", type=int, default=SLURM_POLL_INTERVAL_SECONDS
    )
    args = parser.parse_args()

    if args.command == "status":
        _print_status(_capacity_snapshot())
    elif args.command == "submit":
        submit_once(force=args.force)
    elif args.command == "autoscale":
        autoscale(poll_interval=args.poll_interval_seconds)


if __name__ == "__main__":
    main()
