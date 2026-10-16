"""SLURM submitter for the featurization × kernel CV scan.

Each SLURM array task runs ``scheduler worker``: with ``--single`` it runs one
outer cell (one model configuration) and exits, otherwise it drains the queue.
Active plus pending workers never exceed ``config.SLURM_MAX_CONCURRENT_WORKERS``.

``--run <name>`` (default ``elimination``) selects the run: its queue and log
folders come from ``paths.run_paths(name)``, the worker line carries the same
``--run``, and the array is named ``slurm_job_name(name)`` so the worker count of
one run never includes another's.

CLI::

    python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild \
        [status|submit|autoscale] [--run NAME] [--single] [--time-limit T] [--force]

Subcommands:
  status     Print queue lock counts, live SLURM worker counts and the number of
             workers a submission would add.
  submit     (default) Submit one SLURM array of N workers covering uncovered
             pending jobs.
  autoscale  Loop, sleeping `--poll_interval_seconds` between checks, submitting
             new array jobs whenever pending work exceeds the live worker pool.
             Exits when no pending/running jobs and no live SLURM workers remain.

Pattern follows pablo/mechthild_runs/run_1_scope_extension/submit_mechthild_cpu.py.
Each SLURM array task runs `python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler worker`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config

SLURM_JOB_NAME_BASE: str = "feat_kernel_scan"


def slurm_job_name(run: str) -> str:
    """SLURM job name of one run's worker arrays.

    Args:
        run: A key of ``config.RUN_GRIDS``.

    Returns:
        ``feat_kernel_scan`` for the elimination run, ``feat_kernel_scan_<run>``
        otherwise. Worker counting matches the name exactly.
    """
    return SLURM_JOB_NAME_BASE if run == paths.ELIMINATION_RUN else f"{SLURM_JOB_NAME_BASE}_{run}"


PARTITION = config.SLURM_PARTITION
RUN_NAME = config.DEFAULT_RUN
JOB_NAME_PREFIX = slurm_job_name(RUN_NAME)
CPUS_PER_TASK = config.SLURM_CPUS_PER_TASK
MEMORY = config.SLURM_MEMORY
TIME_LIMIT = config.SLURM_TIME_LIMIT
EMAIL = config.SLURM_EMAIL
MAX_CONCURRENT_WORKERS = config.SLURM_MAX_CONCURRENT_WORKERS
DEFAULT_POLL_INTERVAL_SECONDS = config.SLURM_POLL_INTERVAL_SECONDS

ROOT = paths.PKG_ROOT
DATA_DIR = paths.DATA_DIR
QUEUE_DIR = paths.QUEUE_DIR
LOGS_DIR = paths.RESULTS_LOGS_DIR
SCHEDULER_MODULE = "pablo.featurization_and_kernel_scan.cross_validation.scheduler"
SINGLE_CELL_TIME_LIMIT = config.SLURM_SINGLE_CELL_TIME_LIMIT


def select_run(run: str) -> None:
    """Point the queue and log folders and the job name at one run.

    Args:
        run: A key of ``config.RUN_GRIDS``.

    Raises:
        KeyError: If ``run`` is not a known run.
    """
    global RUN_NAME, JOB_NAME_PREFIX, QUEUE_DIR, LOGS_DIR
    if run not in config.RUN_GRIDS:
        raise KeyError(f"unknown run {run!r}; known runs: {sorted(config.RUN_GRIDS)}")
    run_paths = paths.run_paths(run)
    RUN_NAME = run
    JOB_NAME_PREFIX = slurm_job_name(run)
    QUEUE_DIR = run_paths.queue_dir
    LOGS_DIR = run_paths.logs_dir


def _capacity_snapshot() -> dict[str, int | bool]:
    """Compute pending/running locks plus live SLURM worker counts.
        Returns:
            dict[str, int | bool]: Return value of _capacity_snapshot.
    
    Returns:
        dict[str, int | bool]: Return value of _capacity_snapshot.
"""
    pending = len(list((QUEUE_DIR / "pending").glob("job_*.lock"))) if (QUEUE_DIR / "pending").exists() else 0
    running = len(list((QUEUE_DIR / "running").glob("job_*.lock"))) if (QUEUE_DIR / "running").exists() else 0

    active = pending_w = 0
    throttled = False
    user = os.environ.get("USER", "")
    res = subprocess.run(
        ["squeue", "-u", user, "-h", "-r", "-o", "%j %T %r"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode == 0:
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[0] != JOB_NAME_PREFIX:
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
    available = 0 if throttled else max(0, MAX_CONCURRENT_WORKERS - total_workers)
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
    print(f"--- {JOB_NAME_PREFIX} / {PARTITION} ---")
    print(f"Pending jobs        : {snap['pending_jobs']}")
    print(f"Running jobs        : {snap['running_jobs']}")
    print(f"Active SLURM workers: {snap['active_workers']}")
    print(f"Pending SLURM workers: {snap['pending_workers']}")
    print(f"Max concurrent      : {MAX_CONCURRENT_WORKERS}")
    print(f"Uncovered jobs      : {snap['uncovered_jobs']}")
    print(f"Throttle active     : {snap['throttled']}")
    print(f"Workers to submit   : {snap['workers_to_submit']}")
    print("------------------------------------")


def array_script(num_workers: int, single: bool = False, time_limit: str | None = None) -> str:
    """The SLURM array script for the selected run.

    Args:
        num_workers: Number of array tasks to submit.
        single: If True each array task runs exactly one outer cell and exits
            (``worker --single``), so no task is ever killed part-way through a
            cell by the wall clock. The array size must then equal the number
            of uncovered cells. If False each task drains pending cells until
            the queue is empty.
        time_limit: SLURM ``--time`` override. Defaults to
            ``SINGLE_CELL_TIME_LIMIT`` when ``single`` is set, else the
            configured ``config.SLURM_TIME_LIMIT``.

    Returns:
        The script text.
    """
    if time_limit is None:
        time_limit = SINGLE_CELL_TIME_LIMIT if single else TIME_LIMIT
    worker_args = "worker --single" if single else "worker"
    if RUN_NAME != paths.ELIMINATION_RUN:
        worker_args += f" --run {RUN_NAME}"
    return f"""#!/bin/bash
#SBATCH -J {JOB_NAME_PREFIX}
#SBATCH --array=1-{num_workers}%{MAX_CONCURRENT_WORKERS}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --time={time_limit}
#SBATCH --partition={PARTITION}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={EMAIL}
#SBATCH -o {LOGS_DIR}/worker_%A_%a.out
#SBATCH -e {LOGS_DIR}/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS={CPUS_PER_TASK}
export MKL_NUM_THREADS={CPUS_PER_TASK}

echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m {SCHEDULER_MODULE} {worker_args}
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
"""


def script_path() -> Path:
    """Where the selected run's array script is written.

    Returns:
        ``.submit_array.sh`` next to this module for the elimination run,
        ``.submit_array_<run>.sh`` otherwise.
    """
    suffix = "" if RUN_NAME == paths.ELIMINATION_RUN else f"_{RUN_NAME}"
    return Path(__file__).parent / f".submit_array{suffix}.sh"


def _submit_array(num_workers: int, single: bool = False, time_limit: str | None = None) -> None:
    """Write and sbatch one SLURM array script for the selected run.

    Args:
        num_workers: Number of array tasks to submit.
        single: One outer cell per task (``worker --single``); see ``array_script``.
        time_limit: SLURM ``--time`` override; see ``array_script``.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if time_limit is None:
        time_limit = SINGLE_CELL_TIME_LIMIT if single else TIME_LIMIT
    path = script_path()
    path.write_text(array_script(num_workers, single=single, time_limit=time_limit))
    mode = "one cell per task" if single else "drain queue"
    print(
        f"Submitting {num_workers} {JOB_NAME_PREFIX} workers to {PARTITION} "
        f"(cap {MAX_CONCURRENT_WORKERS}, {mode}, --time={time_limit})"
    )
    subprocess.run(["sbatch", str(path)], check=True)


def submit_once(force: bool, single: bool = False, time_limit: str | None = None) -> None:
    """Print the capacity snapshot and submit one array covering uncovered work.

    The array size is ``workers_to_submit`` from the snapshot: uncovered
    pending plus running jobs, capped by the free slots under
    ``SLURM_MAX_CONCURRENT_WORKERS`` and 0 while SLURM throttles the user.

    Args:
        force: Skip the y/N confirmation prompt.
        single: One outer cell per array task (``worker --single``).
        time_limit: SLURM ``--time`` override; see ``array_script``.
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
    _submit_array(n, single=single, time_limit=time_limit)


def autoscale(poll_interval_seconds: int, single: bool = False,
              time_limit: str | None = None) -> None:
    """Submit worker arrays in a loop until the selected run's queue is finished.

    Each poll submits ``workers_to_submit`` workers if positive. Returns once no
    pending or running locks and no active or pending SLURM workers remain.

    Args:
        poll_interval_seconds: Sleep between polls, in seconds.
        single: One outer cell per array task (``worker --single``).
        time_limit: SLURM ``--time`` override; see ``array_script``.
    """
    print(f"Autoscaler: run={RUN_NAME}, partition={PARTITION}, poll={poll_interval_seconds}s. Ctrl+C to stop.")
    while True:
        snap = _capacity_snapshot()
        if (snap["pending_jobs"] == 0 and snap["running_jobs"] == 0
                and snap["active_workers"] == 0 and snap["pending_workers"] == 0):
            print("All jobs and workers finished. Exiting autoscaler.")
            return
        n = int(snap["workers_to_submit"])
        if n > 0:
            print(f"[autoscale] submitting {n} workers")
            _submit_array(n, single=single, time_limit=time_limit)
        else:
            print(f"[autoscale] no slots / no work (pending={snap['pending_jobs']}, "
                  f"running={snap['running_jobs']}, workers={snap['active_workers']}+"
                  f"{snap['pending_workers']})")
        time.sleep(poll_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="submit",
                        choices=["submit", "status", "autoscale"])
    parser.add_argument("--force", action="store_true",
                        help="Skip the y/N prompt for `submit`.")
    parser.add_argument("--poll_interval_seconds", type=int,
                        default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--single", action="store_true",
                        help="One outer cell per array task (`worker --single`); "
                             "nothing is killed mid-cell by the wall clock.")
    parser.add_argument("--time-limit", dest="time_limit", default=None,
                        help=f"SLURM --time override (default {SINGLE_CELL_TIME_LIMIT} "
                             f"with --single, else {TIME_LIMIT}).")
    parser.add_argument("--run", choices=sorted(config.RUN_GRIDS), default=config.DEFAULT_RUN,
                        help="Run whose queue to cover (paths.run_paths).")
    args = parser.parse_args()
    select_run(args.run)

    if args.command == "status":
        _print_status(_capacity_snapshot())
    elif args.command == "submit":
        submit_once(force=args.force, single=args.single, time_limit=args.time_limit)
    elif args.command == "autoscale":
        autoscale(poll_interval_seconds=args.poll_interval_seconds,
                  single=args.single, time_limit=args.time_limit)


if __name__ == "__main__":
    main()
