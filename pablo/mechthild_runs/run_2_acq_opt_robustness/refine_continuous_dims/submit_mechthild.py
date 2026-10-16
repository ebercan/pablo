"""SLURM array submission for refine-continuous-dims sweep.

Submits 244 CPU workers as a single SLURM array job.  Each array task runs
``run_worker.py`` with ``$SLURM_ARRAY_TASK_ID`` as the worker index.

Usage (from repository root)::

    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.submit_mechthild
    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.submit_mechthild --force
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.config import (
    CPUS_PER_TASK,
    JOB_NAME,
    MEMORY,
    PARTITION,
    TIME_LIMIT,
    WORKERS,
)

EMAIL = "enisbaris@mpi-magdeburg.mpg.de"
LOGS_DIRNAME = "logs"


def submit(force: bool = False) -> None:
    """Write the refine-continuous-dims SLURM array script and submit it with sbatch.

    Creates ``logs/`` and ``submit_script.sh`` next to this file, then runs
    ``sbatch`` on the script: one array task per worker (``0 .. WORKERS - 1``).

    Args:
        force: If True, skip the interactive ``[y/N]`` confirmation prompt.

    Raises:
        subprocess.CalledProcessError: If ``sbatch`` exits non-zero.
    """
    base_dir = Path(__file__).parent
    max_index = WORKERS - 1

    print(f"Workers: {WORKERS} (array 0-{max_index})")
    print(
        f"Partition: {PARTITION}  |  cpus-per-task: {CPUS_PER_TASK}  |  mem: {MEMORY}  |  time: {TIME_LIMIT}"
    )
    print(f"Each worker optimises 2 batches via optimize_acqf_continuous.")

    if not force:
        choice = input(f"Submit {WORKERS} array tasks? [y/N]: ")
        if choice.strip().lower() != "y":
            print("Aborting.")
            return

    logs_dir = base_dir / LOGS_DIRNAME
    logs_dir.mkdir(parents=True, exist_ok=True)

    slurm_script = f"""#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH --array=0-{max_index}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={EMAIL}
#SBATCH -o {logs_dir}/worker_%A_%a.out
#SBATCH -e {logs_dir}/worker_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "Starting worker $SLURM_ARRAY_TASK_ID on node $SLURM_NODENAME"

python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.run_worker $SLURM_ARRAY_TASK_ID
"""

    submit_script_path = base_dir / "submit_script.sh"
    submit_script_path.write_text(slurm_script)

    print(f"Submitting {WORKERS} array tasks to partition {PARTITION} ...")
    subprocess.run(["sbatch", str(submit_script_path)], check=True)
    print("Submitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit refine-continuous-dims workers to Mechthild"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    submit(force=args.force)
