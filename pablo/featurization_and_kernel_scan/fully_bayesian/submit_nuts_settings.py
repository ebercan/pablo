"""SLURM submitter for the NUTS-settings study on Mechthild.

Generates an sbatch array job script and submits it to the cluster scheduler
using partition 'short'. Run it from the repo root with the venv active; the
job inherits that environment and resolves its relative log paths there.

CLI::

    python -m pablo.featurization_and_kernel_scan.fully_bayesian.submit_nuts_settings [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study import STUDY_DIR

PARTITION: str = "short"
"""Target SLURM partition."""

TIME_LIMIT: str = "01:00:00"
"""Wall time limit per task on the short partition."""

MAX_CONCURRENT: int = 50
"""Maximum concurrent array worker tasks."""

CPUS: int = 1
"""Number of CPU cores per array task."""

MEMORY: str = "2G"
"""Memory allocated per task."""

JOB_NAME: str = "fb_nuts"
"""SLURM job identifier."""

SCRIPT_PATH: Path = STUDY_DIR / "submit_nuts_settings.sh"
"""Destination path for the generated sbatch script."""


def generate_sbatch_script(n_tasks: int) -> str:
    """Generate the SLURM sbatch script content for the NUTS study array.

    Args:
        n_tasks: Total number of tasks in inputs.json.

    Returns:
        The generated bash script text.

    Raises:
        ValueError: If n_tasks is less than 1.
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")

    logs_rel = (STUDY_DIR / "logs").relative_to(paths.REPO_ROOT)
    array_spec = f"0-{n_tasks - 1}%{MAX_CONCURRENT}"
    email = config.SLURM_EMAIL

    return f"""#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH --array={array_spec}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEMORY}
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={email}
#SBATCH -o {logs_rel}/task_%A_%a.out
#SBATCH -e {logs_rel}/task_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS={CPUS}
export MKL_NUM_THREADS={CPUS}

echo "[task] id=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study run --task $SLURM_ARRAY_TASK_ID
echo "[task] id=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
"""


def main() -> None:
    """CLI entry point for submitting the NUTS settings study."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated sbatch script without writing or submitting.",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != paths.REPO_ROOT:
        raise RuntimeError(
            f"submit from the repo root {paths.REPO_ROOT}: the sbatch log paths are relative to it"
        )
    inputs_path = STUDY_DIR / "inputs.json"
    tasks = json.loads(inputs_path.read_text())
    n_tasks = len(tasks)

    script_text = generate_sbatch_script(n_tasks)

    if args.dry_run:
        print(script_text)
        return

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = STUDY_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    SCRIPT_PATH.write_text(script_text)
    print(f"Wrote submit script to {SCRIPT_PATH}")
    subprocess.run(["sbatch", str(SCRIPT_PATH)], check=True)


if __name__ == "__main__":
    main()
