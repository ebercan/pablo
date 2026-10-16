"""SLURM submission for run_6.

``submit --tiers T ...`` writes and submits one array with one task per pending job of
those tiers; each task runs ``scheduler worker --single --tiers T ...`` and so claims
only jobs of those tiers. The tiers of one array must share their (partition, time
limit, memory) in ``config.SLURM_BY_TIER``. ``calibrate`` submits the single
calibration job. ``--dry-run`` prints the script instead of calling ``sbatch``. Run
from the repo root; log paths are relative to it.

Usage:
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.submit_mechthild calibrate [--dry-run]
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.submit_mechthild submit --tiers T [T ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pablo.mechthild_runs.run_6_af_optimizer_af_value import config
from pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler import pending_ids

PACKAGE = "pablo.mechthild_runs.run_6_af_optimizer_af_value"
RUN_DIR_REL = Path("pablo/mechthild_runs/run_6_af_optimizer_af_value")
LOGS_REL = RUN_DIR_REL / "data" / "logs"
SCRIPT_REL = RUN_DIR_REL / ".submit_array.sh"


def _header(job_name: str, partition: str, time_limit: str, memory: str, log_stem: str, array: str | None) -> str:
    """``#SBATCH`` block and environment shared by both job kinds."""
    array_line = f"#SBATCH --array={array}\n" if array else ""
    return f"""#!/bin/bash
#SBATCH -J {job_name}
{array_line}#SBATCH --cpus-per-task={config.SLURM_CPUS_PER_TASK}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={config.SLURM_EMAIL}
#SBATCH -o {LOGS_REL}/{log_stem}.out
#SBATCH -e {LOGS_REL}/{log_stem}.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS={config.SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS={config.SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=""
source .venv/bin/activate
"""


def slurm_settings(tiers: list[int]) -> tuple[str, str, str]:
    """The shared (partition, time limit, memory) of ``tiers``; raises when they differ."""
    settings = {config.SLURM_BY_TIER[t] for t in tiers}
    if len(settings) != 1:
        raise ValueError(f"tiers {tiers} have different SLURM settings {sorted(settings)}; submit them separately")
    return settings.pop()


def worker_script(tiers: list[int], n_tasks: int) -> str:
    """Array script with ``n_tasks`` tasks, each running one queued job of ``tiers``."""
    partition, time_limit, memory = slurm_settings(tiers)
    tier_args = " ".join(str(t) for t in tiers)
    name = f"{config.SLURM_JOB_NAME}_t{'-'.join(str(t) for t in tiers)}"
    return _header(
        name, partition, time_limit, memory, "worker_%A_%a", f"1-{n_tasks}%{config.SLURM_MAX_CONCURRENT_WORKERS}",
    ) + f"""
echo "[worker] task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME $(date -Iseconds)"
python -m {PACKAGE}.scheduler worker --single --tiers {tier_args}
echo "[worker] task=$SLURM_ARRAY_TASK_ID done $(date -Iseconds)"
"""


def calibrate_script() -> str:
    """Single-job script running ``main_variant calibrate``."""
    return _header(
        f"{config.SLURM_JOB_NAME}_cal", config.SLURM_CALIBRATE_PARTITION, config.SLURM_CALIBRATE_TIME_LIMIT,
        config.SLURM_CALIBRATE_MEMORY, "calibrate_%j", None,
    ) + f"""
echo "[calibrate] node=$SLURMD_NODENAME $(date -Iseconds)"
python -m {PACKAGE}.main_variant calibrate
"""


def _sbatch(script: str, dry_run: bool) -> None:
    """Print ``script`` (dry run) or write it to :data:`SCRIPT_REL` and submit it."""
    if dry_run:
        print(script)
        return
    LOGS_REL.mkdir(parents=True, exist_ok=True)
    SCRIPT_REL.write_text(script)
    subprocess.run(["sbatch", str(SCRIPT_REL)], check=True)


def main() -> None:
    """CLI: ``calibrate`` | ``submit --tiers T ...``, each with ``--dry-run``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["calibrate", "submit"])
    parser.add_argument("--tiers", type=int, nargs="+", choices=range(config.N_TIERS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not Path("pablo").is_dir():
        raise RuntimeError("run from the repo root: the log paths are relative to it")
    if args.command == "calibrate":
        _sbatch(calibrate_script(), args.dry_run)
        return
    if not args.tiers:
        raise ValueError("submit needs --tiers")
    tiers = sorted(set(args.tiers))
    n_pending = len(pending_ids(set(tiers)))
    if n_pending == 0:
        raise RuntimeError(f"no pending jobs in tiers {tiers}; run `scheduler init` first")
    _sbatch(worker_script(tiers, n_pending), args.dry_run)


if __name__ == "__main__":
    main()
