"""SLURM submitter of run_5 workers, one array per operating mode.

Every array task runs one ``mechthild_scheduler_cpu worker``, which runs one
job. ``submit`` counts the mode's pending and running jobs, subtracts the
mode's queued SLURM workers (running or pending), and submits one array task
per job left uncovered. Run from the repo root; the log paths in the script are
relative to it:

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.submit_mechthild_cpu \
        --operating_mode MODE [--force] [--dry_run]

``--dry_run`` writes ``.submit_dynamic.sh`` and prints it without calling ``sbatch``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    ALL_OPERATING_MODES,
    LOGS_DIRNAME,
    RUN_DIR,
    SLURM_CPUS_PER_TASK,
    SLURM_EMAIL,
    SLURM_MAX_CONCURRENT_TASKS,
    SLURM_MEMORY,
    SLURM_PARTITION,
    SLURM_TIME_LIMIT,
    job_name,
    queue_dir,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu import (
    lock_indices,
    squeue_lines,
    load_inputs,
)

SCRIPT_PATH = RUN_DIR / ".submit_dynamic.sh"


def uncovered_jobs(queue_path: Path, operating_mode: str, count_workers: bool = True) -> int:
    """Pending plus running jobs of the mode minus its queued SLURM workers, floored at 0.

    Args:
        queue_path: Queue directory of the mode.
        operating_mode: Mode whose jobs and workers are counted.
        count_workers: Subtract the workers listed by ``squeue``; ``False``
            counts every pending and running job as uncovered.

    Returns:
        Number of array tasks to submit.

    Raises:
        RuntimeError: If ``count_workers`` is set and ``squeue`` is not available.
    """
    inputs = load_inputs()
    jobs = sum(len(lock_indices(queue_path / sub, operating_mode, inputs)) for sub in ("pending", "running"))
    workers = 0
    if count_workers:
        lines = squeue_lines("%j %T")
        if lines is None:
            raise RuntimeError("submit needs squeue; use --dry_run off the cluster")
        workers = sum(
            1 for line in lines
            if line.split()[0] == job_name(operating_mode) and line.split()[1] in ("RUNNING", "PENDING", "CONFIGURING")
        )
    print(f"[{operating_mode}] pending + running jobs: {jobs}, queued SLURM workers: {workers}")
    return max(0, jobs - workers)


def write_array_script(num_tasks: int, queue_path: Path, operating_mode: str) -> Path:
    """Write the SLURM array script of ``num_tasks`` workers to ``SCRIPT_PATH``.

    The ``-o``/``-e`` paths are relative to the submit directory, which must be
    the repo root; the logs folder is created there.

    Args:
        num_tasks: Number of array tasks.
        queue_path: Queue directory passed to each worker.
        operating_mode: Mode of the workers.

    Returns:
        ``SCRIPT_PATH``.

    Raises:
        ValueError: If the current directory does not contain this folder.
    """
    logs_rel = RUN_DIR.resolve().relative_to(Path.cwd().resolve()) / LOGS_DIRNAME
    logs_rel.mkdir(parents=True, exist_ok=True)
    SCRIPT_PATH.write_text(f"""#!/bin/bash
#SBATCH -J {job_name(operating_mode)}
#SBATCH --array=1-{num_tasks}%{SLURM_MAX_CONCURRENT_TASKS}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --mem={SLURM_MEMORY}
#SBATCH --time={SLURM_TIME_LIMIT}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={SLURM_EMAIL}
#SBATCH -o {logs_rel}/worker_%A_%a.out
#SBATCH -e {logs_rel}/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Array task $SLURM_ARRAY_TASK_ID (job $SLURM_JOB_ID) on $SLURMD_NODENAME"
python -m pablo.mechthild_runs.run_5_scope_extension_v2.mechthild_scheduler_cpu worker \\
    --operating_mode {operating_mode} --queue_dir "{queue_path.resolve()}"
""")
    return SCRIPT_PATH


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Submit run_5 workers for one operating mode")
    parser.add_argument("--operating_mode", choices=ALL_OPERATING_MODES, required=True)
    parser.add_argument("--queue_dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="submit without asking")
    parser.add_argument("--dry_run", action="store_true", help="write and print the script only")
    args = parser.parse_args()

    mode = str(args.operating_mode)
    queue_path = queue_dir(mode) if args.queue_dir is None else Path(args.queue_dir)
    num_tasks = uncovered_jobs(queue_path, mode, count_workers=not args.dry_run)
    if num_tasks == 0:
        print("Nothing to submit.")
        return
    script = write_array_script(num_tasks, queue_path, mode)
    if args.dry_run:
        print(script.read_text())
        return
    if not args.force and input(f"Submit {num_tasks} workers? [y/N]: ").strip().lower() != "y":
        print("Aborted.")
        return
    subprocess.run(["sbatch", str(script)], check=True)


if __name__ == "__main__":
    main()
