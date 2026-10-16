"""Time every level of the elimination grid on a few real folds.

The probe runs each of the elimination grid's configurations — the baseline plus
every non-baseline level of every sweep — at three catalyst widths, for a few
primary folds each, through ``scheduler._run_outer_cell``: the function a worker
runs for one cell. ``summarize`` divides each level's seconds per fold by the
baseline's at the same width and prints the raw seconds per fold in the form of
``_PRIMARY_SECONDS_PER_FOLD`` in ``scheduler.py``, which the scheduler's cost
model interpolates over catalyst width.

Tasks are width-major: task ``i`` is configuration ``i % 12`` of
``generate_inputs.configurations`` at width ``PROBE_WIDTHS[i // 12]``. Each task
writes ``results/timing_probe/task_<i>.json`` in the result-JSON format of the
scan, plus a ``probe`` block.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe <subcommand> [options]

Subcommands::

    list                        print the tasks
    run --task N [--folds K] [--study S]      run task N on the first K folds of study S
    submit [--folds K] [--study S] [--tasks L] sbatch one array task per probe task (Mechthild)
    summarize                   seconds per fold and multipliers vs baseline
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.cross_validation import scheduler
from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import (
    configurations,
    enumerate_cells,
    sweep_membership,
)

PROBE_WIDTHS: tuple[str, ...] = ("grover_pca", "grover", "soap")
"""Catalyst levels the probe times at: narrow, wide and the widest on the grid."""

DEFAULT_FOLDS: int = 2
"""Folds per probe task."""

DEFAULT_STUDY: str = "primary"
"""Study the probe folds are drawn from. ``summarize`` reads this study's results;
a probe on ``leave_one_out`` (68 training points against 51) measures how per-fold
cost grows with the training-set size."""

JOB_NAME = "feat_kernel_probe"
PARTITION = "short"
TIME_LIMIT = "02:00:00"
MODULE = "pablo.featurization_and_kernel_scan.cross_validation.timing_probe"


def probe_tasks() -> list[dict]:
    """Every probe task as an ``inputs.json`` outer-cell record.

    Returns:
        One cell per (width, configuration), width-major, with
        ``outer_cell_id`` equal to the task index.
    """
    grid = config.ELIMINATION_GRID
    cells = list(enumerate_cells(PROBE_WIDTHS, grid).values())
    for task, cell in enumerate(cells):
        cell["outer_cell_id"] = task
    return cells


def level_label(cell: dict) -> str:
    """``axis=level`` for a non-baseline cell, ``baseline`` for the baseline.

    Args:
        cell: A probe cell.

    Returns:
        The label.
    """
    grid = config.ELIMINATION_GRID
    (inner,) = cell["inner_cells"]
    sweeps = sweep_membership({**cell, **inner}, grid)
    if len(sweeps) > 1:
        return "baseline"
    axis = sweeps[0]
    return f"{axis}={({**cell, **inner})[axis]}"


def _truncated_splits(n_folds: int, study: str) -> list[dict]:
    """The first ``n_folds`` folds of the first repeat of one study.

    Args:
        n_folds: Number of folds to keep.
        study: A ``split_type`` of ``cv_splits.json``.

    Returns:
        A splits list with one repeat block.

    Raises:
        ValueError: If the study is absent or has fewer than ``n_folds`` folds.
    """
    blocks = [b for b in json.loads(paths.CV_SPLITS_JSON.read_text()) if b["split_type"] == study]
    if not blocks or len(blocks[0]["folds"]) < n_folds:
        raise ValueError(f"cannot take {n_folds} folds from study {study!r}")
    return [{**blocks[0], "folds": blocks[0]["folds"][:n_folds]}]


def result_path(task: int, study: str) -> Path:
    """Where a probe task writes its result.

    Args:
        task: Probe task index.
        study: The study its folds were drawn from.

    Returns:
        ``task_<i>.json`` for ``primary``, ``task_<i>_<study>.json`` otherwise.
    """
    suffix = "" if study == DEFAULT_STUDY else f"_{study}"
    return paths.TIMING_PROBE_DIR / f"task_{task}{suffix}.json"


def run(task: int, n_folds: int, study: str = DEFAULT_STUDY) -> Path:
    """Run one probe task and write its result JSON.

    Args:
        task: Probe task index.
        n_folds: Folds to run.
        study: The study the folds are drawn from.

    Returns:
        The written result path.

    Raises:
        RuntimeError: If the probe task fails before completing its first fold.
    """
    cell = probe_tasks()[task]
    paths.TIMING_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = result_path(task, study)
    run_meta = {
        "git_commit": scheduler._git_commit(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[probe] task {task}: {cell['cat_feat']} {level_label(cell)}, {n_folds} {study} folds", flush=True)
    payload = scheduler._run_outer_cell(task, cell, _truncated_splits(n_folds, study), run_meta)
    (inner,) = payload["inner_cells"]
    if inner.get("status") == "failed":
        raise RuntimeError(f"probe task {task} failed before its first fold: {inner['error']}")
    payload["probe"] = {
        "label": level_label(cell),
        "width_level": cell["cat_feat"],
        "study": study,
        "n_folds": inner["n_folds"],
        "fold_status": [fold.get("status", "ok") for fold in inner["folds"]],
        "seconds_per_fold": inner["wall_time_s"] / inner["n_folds"],
        "design_matrix_width": inner.get("feature_dims", {}).get("X_total"),
    }
    out_path.write_text(json.dumps(payload))
    print(
        f"[probe] task {task} done: {payload['probe']['seconds_per_fold']:.1f} s/fold, "
        f"status {payload['probe']['fold_status']} -> {out_path}",
        flush=True,
    )
    return out_path


def submit(n_folds: int, study: str = DEFAULT_STUDY, tasks: str | None = None) -> None:
    """Write and sbatch one SLURM array over the probe tasks.

    Args:
        n_folds: Folds per task.
        study: The study the folds are drawn from.
        tasks: SLURM ``--array`` index list, e.g. ``"12,21,24,33"``; every task
            when omitted.
    """
    logs = paths.TIMING_PROBE_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_tasks = len(probe_tasks())
    script = f"""#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH --array={tasks or f"0-{n_tasks - 1}"}
#SBATCH --cpus-per-task={config.SLURM_CPUS_PER_TASK}
#SBATCH --mem={config.SLURM_MEMORY}
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
#SBATCH -o {logs}/probe_%A_%a.out
#SBATCH -e {logs}/probe_%A_%a.err

export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS={config.SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS={config.SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=

python -m {MODULE} run --task "$SLURM_ARRAY_TASK_ID" --folds {n_folds} --study {study}
"""
    script_path = Path(__file__).parent / ".submit_timing_probe.sh"
    script_path.write_text(script)
    print(f"Submitting probe tasks {tasks or f'0-{n_tasks - 1}'} to {PARTITION} ({n_folds} {study} folds each)")
    subprocess.run(["sbatch", str(script_path)], check=True)


def summarize() -> None:
    """Print seconds per fold and multipliers vs baseline for every level.

    Raises:
        FileNotFoundError: If a probe task has no result file.
        ValueError: If a baseline task has a failed fold.
    """
    tasks = probe_tasks()
    probes = []
    for task, cell in enumerate(tasks):
        path = result_path(task, DEFAULT_STUDY)
        if not path.exists():
            raise FileNotFoundError(f"probe task {task} ({cell['cat_feat']} {level_label(cell)}) has no {path}")
        probes.append(json.loads(path.read_text())["probe"])

    baseline = {p["width_level"]: p for p in probes if p["label"] == "baseline"}
    for width, p in baseline.items():
        if any(status != "ok" for status in p["fold_status"]):
            raise ValueError(f"baseline at {width} has a failed fold: {p['fold_status']}")

    labels = list(dict.fromkeys(p["label"] for p in probes))
    widths = list(PROBE_WIDTHS)
    header = "".join(f"{w + ' s/fold':>20}{'×':>7}" for w in widths)
    print(f"{'level':<32}{header}{'median ×':>10}  status")
    multipliers: dict[str, float] = {}
    for label in labels:
        row = {p["width_level"]: p for p in probes if p["label"] == label}
        ratios = [row[w]["seconds_per_fold"] / baseline[w]["seconds_per_fold"] for w in widths]
        multipliers[label] = statistics.median(ratios)
        cols = "".join(
            f"{row[w]['seconds_per_fold']:>20.1f}{r:>7.2f}" for w, r in zip(widths, ratios)
        )
        failed = sum(s != "ok" for w in widths for s in row[w]["fold_status"])
        print(f"{label:<32}{cols}{multipliers[label]:>10.2f}  {failed} failed folds")

    print()
    print("_PRIMARY_SECONDS_PER_FOLD for scheduler.py:")
    for label in labels:
        row = {p["width_level"]: p for p in probes if p["label"] == label}
        values = ", ".join(f"{row[w]['seconds_per_fold']:.2f}" for w in widths)
        print(f'    "{label}": ({values}),')
    print("widths:", tuple(baseline[w]["design_matrix_width"] for w in widths))

    ratios = []
    for task in range(len(tasks)):
        other = result_path(task, "leave_one_out")
        if other.exists():
            loo = json.loads(other.read_text())["probe"]["seconds_per_fold"]
            ratios.append(loo / probes[task]["seconds_per_fold"])
    if ratios:
        print(
            f"leave_one_out / primary seconds per fold over {len(ratios)} tasks: "
            f"median {statistics.median(ratios):.2f} (range {min(ratios):.2f}-{max(ratios):.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Print the probe tasks.")
    run_parser = sub.add_parser("run", help="Run one probe task.")
    run_parser.add_argument("--task", type=int, required=True)
    run_parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    run_parser.add_argument("--study", default=DEFAULT_STUDY)
    submit_parser = sub.add_parser("submit", help="sbatch the probe tasks as one array.")
    submit_parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    submit_parser.add_argument("--study", default=DEFAULT_STUDY)
    submit_parser.add_argument("--tasks", default=None, help="SLURM array index list; all tasks if omitted.")
    sub.add_parser("summarize", help="Seconds per fold and multipliers vs baseline.")
    args = parser.parse_args()

    if args.cmd == "list":
        for task, cell in enumerate(probe_tasks()):
            print(f"{task:>3}  {cell['cat_feat']:<12} {level_label(cell)}")
    elif args.cmd == "run":
        run(args.task, args.folds, args.study)
    elif args.cmd == "submit":
        submit(args.folds, args.study, args.tasks)
    elif args.cmd == "summarize":
        summarize()


if __name__ == "__main__":
    main()
