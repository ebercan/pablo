"""SLURM array submission for the run_2 AF optimization robustness benchmark.

Submits every job in ``inputs.json`` as one SLURM array job on the medium
partition. Each array task runs ``run_single_job`` with $SLURM_ARRAY_TASK_ID
as the job index. The generated script is written to ``submit_script.sh``
next to this file.

Usage:
    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.submit_mechthild [--force]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pablo.mechthild_runs.run_2_acq_opt_robustness.config import (
    INPUTS_FILENAME,
    LOGS_DIRNAME,
)

JOB_NAME = "run3_acq_robustness"
PARTITION = "medium"
CPUS_PER_TASK = 1
MEMORY = "16G"
TIME_LIMIT = "24:00:00"
EMAIL = "enisbaris@mpi-magdeburg.mpg.de"


def submit(force: bool = False) -> None:
    """Generate and submit the SLURM array job script.

    Args:
        force: Skip confirmation prompt when True.
    """
    base_dir = Path(__file__).parent
    inputs_path = base_dir / INPUTS_FILENAME

    with open(inputs_path, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)

    n_jobs = len(inputs)
    max_index = n_jobs - 1

    print(f"Jobs to submit: {n_jobs} (array 0-{max_index})")
    print(f"Partition: {PARTITION}")
    print(f"Memory: {MEMORY}")
    print(f"Time limit: {TIME_LIMIT}")

    if not force:
        choice = input(f"Submit {n_jobs} array tasks? [y/N]: ")
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
echo "Starting array task $SLURM_ARRAY_TASK_ID on node $SLURM_NODENAME"

python -m pablo.mechthild_runs.run_2_acq_opt_robustness.run_single_job $SLURM_ARRAY_TASK_ID
"""

    submit_script_path = base_dir / "submit_script.sh"
    submit_script_path.write_text(slurm_script)

    print(f"Submitting {n_jobs} array tasks to partition {PARTITION}...")
    subprocess.run(["sbatch", str(submit_script_path)], check=True)
    print("Submitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit run_2 AF robustness jobs to Mechthild"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    submit(force=args.force)
