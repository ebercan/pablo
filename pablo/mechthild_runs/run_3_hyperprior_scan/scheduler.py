"""File-based queue scheduler for the run_3 hyperprior scan.

Outer cells are keyed by ``(hyperprior_config × cat_feat)``, 8 total.
Inner axis: ``(outcome_transform × model_class × mean_function)`` = 8 configs.
Each inner config runs 135 CV folds (reuses cv_splits.json from the main scan).

Before each outer cell, the relevant config-module attributes are monkey-patched
to apply the requested hyper-prior (lengthscale σ, floor, Dirichlet conc).

Subcommands:
  init       Touch pending locks for cells missing a result file.
  worker     Loop, atomically claim one pending lock, run its inner configs,
             write result_<outer_id>.json. ``--single`` exits after one cell.
  status     Print queue status + ETA.
  submitted  Per-running-worker progress from heartbeat files.

Usage:
    python -m pablo.mechthild_runs.run_3_hyperprior_scan.scheduler init
    python -m pablo.mechthild_runs.run_3_hyperprior_scan.scheduler worker [--single]
    python -m pablo.mechthild_runs.run_3_hyperprior_scan.scheduler status
    python -m pablo.mechthild_runs.run_3_hyperprior_scan.scheduler submitted
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from pablo.utils.feature_assembly import assemble_features
from pablo.featurization_and_kernel_scan.core import config as scan_config
from pablo.featurization_and_kernel_scan import paths as scan_paths
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import fit_eval
from pablo.featurization_and_kernel_scan.core.models import build_mean_module
from pablo.featurization_and_kernel_scan.cross_validation.run_state import (
    get_run_state_path,
    load_run_state_for_reporting,
    update_run_state,
)
from pablo.mechthild_runs.run_3_hyperprior_scan.config import (
    FOLD_TOTAL,
    HYPERPRIOR_BY_NAME,
    HyperPriorConfig,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INPUTS_PATH = DATA_DIR / "inputs.json"
QUEUE_DIR = DATA_DIR / "queue"
RUN_STATE_DIR = QUEUE_DIR / "run_state"
RESULTS_DIR = DATA_DIR / "results" / "cells"

SCAN_DATA_DIR = scan_paths.DATA_DIR
SCAN_LIB_DIR = scan_paths.LIB_DIR
SPLITS_PATH = scan_paths.CV_SPLITS_JSON
SCAN_CONFIG_PATH = Path(scan_config.__file__)

JOB_NAME_PREFIX = "hp_prior_scan"

_MEAN_ABBREV = {"constant": "M1", "type_linear": "M3"}


def _load_inputs() -> dict[str, dict]:
    return json.loads(INPUTS_PATH.read_text())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_commit() -> str | None:
    if shutil.which("git") is None:
        return None
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return res.stdout.strip() or None if res.returncode == 0 else None


def _run_meta(splits_hash: str, config_hash: str) -> dict:
    return {
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "splits_sha256": splits_hash,
        "config_sha256": config_hash,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _split_type_seed_offset(split_type: str) -> int:
    return int(hashlib.sha256(split_type.encode()).hexdigest()[:8], 16)


def _ensure_dirs() -> None:
    for sub in ("pending", "running", "done"):
        (QUEUE_DIR / sub).mkdir(parents=True, exist_ok=True)
    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_hyperprior(hp: HyperPriorConfig) -> None:
    """Monkey-patch the scan config module with the given hyper-prior values."""
    scan_config.LENGTHSCALE_LOGNORMAL_SCALE = hp.ls_scale
    if hp.ls_floor is not None:
        scan_config.LENGTHSCALE_FLOOR = hp.ls_floor
    else:
        if hasattr(scan_config, "LENGTHSCALE_FLOOR"):
            delattr(scan_config, "LENGTHSCALE_FLOOR")
    scan_config.CONCENTRATION_DIRICHLET = hp.dirichlet_conc


def init_queue() -> None:
    """Create the queue directories and a pending lock for every unfinished outer cell.

    A cell with an existing ``result_<idx>.json`` gets a done lock and its
    pending/running locks are removed. A cell that already has a lock in any
    queue directory is left as it is.
    """
    _ensure_dirs()
    inputs = _load_inputs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    created = 0
    skipped_done = 0
    for key in sorted(inputs.keys(), key=int):
        idx = int(key)
        lock_name = f"job_{idx}.lock"
        result_file = RESULTS_DIR / f"result_{idx}.json"

        if result_file.exists():
            skipped_done += 1
            (done_dir / lock_name).touch(exist_ok=True)
            for stale in (pending_dir / lock_name, running_dir / lock_name):
                if stale.exists():
                    stale.unlink()
            continue

        if any((d / lock_name).exists() for d in (pending_dir, running_dir, done_dir)):
            continue

        (pending_dir / lock_name).touch()
        created += 1

    print(
        f"[init] created {created} pending locks, skipped {skipped_done} done "
        f"({len(inputs)} total outer cells)"
    )


def _claim_next(pending_dir: Path, running_dir: Path) -> int | None:
    for pf in sorted(pending_dir.glob("job_*.lock"), key=lambda p: int(p.stem.split("_")[1])):
        idx = int(pf.stem.split("_")[1])
        target = running_dir / pf.name
        try:
            os.rename(pf, target)
            return idx
        except (FileNotFoundError, OSError):
            continue
    return None


def _run_inner_cell(
    outer_cfg: dict,
    inner_cfg: dict,
    cell_features,
    splits: list,
    on_fold_done=None,
) -> dict:
    kernel_structure = outer_cfg["kernel_structure"]
    outcome_transform = inner_cfg["outcome_transform"]
    mean_function = inner_cfg["mean_function"]

    fold_records: list[dict] = []
    folds_done = 0
    t_cell = time.time()
    for repeat_block in splits:
        split_type = repeat_block.get("split_type", "primary")
        repeat_idx = repeat_block["repeat"]
        type_offset = _split_type_seed_offset(split_type)

        for fold_idx, fold in enumerate(repeat_block["folds"]):
            seed = (
                outer_cfg["outer_cell_id"] * 1_000_000_000
                + inner_cfg["inner_id"] * 1_000_000
                + type_offset
                + repeat_idx * 100
                + fold_idx
            ) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            try:
                fold_res = fit_eval(
                    cell_features=cell_features,
                    train_idx=fold["train"],
                    test_idx=fold["test"],
                    kernel_structure=kernel_structure,
                    outcome_transform=outcome_transform,
                    model_class=inner_cfg["model_class"],
                    base_kernel=inner_cfg["base_kernel"],
                    mean_function=mean_function,
                )
                rec = dataclasses.asdict(fold_res)
            except Exception as e:
                print(
                    f"[FOLD_FAILED] outer={outer_cfg['outer_cell_id']} "
                    f"hp={outer_cfg['hyperprior']} cat={outer_cfg['cat_feat']} "
                    f"transform={outcome_transform} | "
                    f"{kernel_structure}/{inner_cfg['model_class']}/"
                    f"{inner_cfg['base_kernel']}/{mean_function} | "
                    f"{split_type}/r{repeat_idx}/f{fold_idx} | "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )
                rec = {
                    "train_idx": fold["train"],
                    "test_idx": fold["test"],
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "n_train": len(fold["train"]),
                    "n_test": len(fold["test"]),
                    "wall_time_s": 0.0,
                    "nlpd_per_point": [],
                    "crps_per_point": [],
                    "baseline_crps_per_point": [],
                    "crpss": float("nan"),
                    "q2": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "calibration_ks": float("nan"),
                    "hyperparams": {},
                    "n_params": 0,
                }

            rec["split_type"] = split_type
            rec["repeat"] = repeat_idx
            rec["fold"] = fold_idx
            rec["seed"] = seed
            for k in (
                "left_out_catalysts", "left_out_type", "left_out_solvent",
                "left_out_idx", "left_out_replicate_id", "left_out_scaffold",
            ):
                if k in fold:
                    rec[k] = fold[k]
            fold_records.append(rec)
            folds_done += 1
            if on_fold_done is not None:
                on_fold_done(folds_done)

    return {
        "inner_id": inner_cfg["inner_id"],
        "inner_config": inner_cfg,
        "n_folds": len(fold_records),
        "wall_time_s": time.time() - t_cell,
        "folds": fold_records,
    }


def _run_outer_cell(
    idx: int,
    outer_cfg: dict,
    splits: list,
    run_meta: dict,
    state_path: Path | None = None,
    started_at_epoch: float | None = None,
) -> dict:
    t0 = time.time()
    started = started_at_epoch if started_at_epoch is not None else t0
    inner_total = len(outer_cfg["inner_cells"])

    hp = HYPERPRIOR_BY_NAME[outer_cfg["hyperprior"]]
    _apply_hyperprior(hp)

    cell_features = assemble_features(
        cat_featurization=outer_cfg["cat_feat"],
        solv_featurization=outer_cfg["solv_feat"],
        base_featurization=outer_cfg["base_feat"],
        experiments_path=SCAN_LIB_DIR / "experiments.parquet",
        featurizations_root=scan_paths.FEAT_DIR,
        libraries_root=SCAN_LIB_DIR,
        append_form_onehot_to_cat=outer_cfg.get("append_form_onehot_to_cat", False),
    )
    feature_dims_snapshot = {
        "cat": len(cell_features.feature_dims["cat"]),
        "solv": len(cell_features.feature_dims["solv"]),
        "base": len(cell_features.feature_dims["base"]),
        "cont": len(cell_features.feature_dims["cont"]),
        "active_flag_dim": cell_features.active_flag_dim,
        "form_id_dim": cell_features.form_id_dim,
        "X_total": cell_features.X.shape[1],
        "n_observations": cell_features.X.shape[0],
    }
    feature_names = cell_features.feature_names

    inner_results = []
    for k, inner_cfg in enumerate(outer_cfg["inner_cells"]):
        t_inner = time.time()
        try:
            if state_path is not None:
                label = (
                    f"{inner_cfg['outcome_transform']}/"
                    f"{inner_cfg['model_class']}/"
                    f"{_MEAN_ABBREV.get(inner_cfg.get('mean_function', 'constant'), '?')}"
                )
                update_run_state(
                    state_path, idx, inner_total, FOLD_TOTAL,
                    inner_done=k,
                    inner_id_current=inner_cfg["inner_id"],
                    inner_label_current=label,
                    fold_done_in_inner=0,
                    wall_time_so_far_s=time.time() - started,
                )

            def _on_fold(folds_done, _idx=idx, _started=started, _inner_total=inner_total):
                if state_path is None:
                    return
                update_run_state(
                    state_path, _idx, _inner_total, FOLD_TOTAL,
                    fold_done_in_inner=folds_done,
                    wall_time_so_far_s=time.time() - _started,
                )

            inner_res = _run_inner_cell(
                outer_cfg, inner_cfg, cell_features, splits, on_fold_done=_on_fold,
            )

        except Exception as e:
            inner_res = {
                "inner_id": inner_cfg["inner_id"],
                "inner_config": inner_cfg,
                "n_folds": 0,
                "wall_time_s": time.time() - t_inner,
                "folds": [],
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }

        inner_results.append(inner_res)

    return {
        "outer_cell_id": idx,
        "outer_config": {k: v for k, v in outer_cfg.items() if k != "inner_cells"},
        "run_meta": run_meta,
        "feature_dims": feature_dims_snapshot,
        "feature_names": feature_names,
        "n_inner_cells": len(inner_results),
        "total_wall_time_s": time.time() - t0,
        "inner_cells": inner_results,
    }


def worker_loop(single: bool = False) -> None:
    """Claim pending outer cells one at a time and run them until the queue is empty.

    For each claimed cell, runs all inner configs over every CV fold, writes
    ``result_<idx>.json``, marks the run state done and moves the lock to
    ``done/``. If a cell raises, its run state is set to ``pending_failed`` with
    the error message and the exception propagates, ending the worker.

    Args:
        single: If True, exit after completing one outer cell.
    """
    warnings.filterwarnings("ignore", message="A not p.d.", module="linear_operator")
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    inputs = _load_inputs()
    splits = json.loads(SPLITS_PATH.read_text())
    splits_hash = _file_sha256(SPLITS_PATH)
    config_hash = _file_sha256(SCAN_CONFIG_PATH)
    run_meta = _run_meta(splits_hash, config_hash)

    hostname = socket.gethostname()
    slurm_job = os.environ.get("SLURM_JOB_ID")
    slurm_array = os.environ.get("SLURM_ARRAY_TASK_ID")

    print(f"[worker] started on {hostname}, queue={QUEUE_DIR}")
    while True:
        idx = _claim_next(pending_dir, running_dir)
        if idx is None:
            print("[worker] no pending jobs left, exiting.")
            return

        lock_name = f"job_{idx}.lock"
        cfg = inputs[str(idx)]
        inner_total = len(cfg["inner_cells"])
        print(
            f"[worker] outer cell {idx}: hp={cfg['hyperprior']} "
            f"cat={cfg['cat_feat']} ({inner_total} inner configs)"
        )

        t0 = time.time()
        state_path = get_run_state_path(QUEUE_DIR, idx)
        update_run_state(
            state_path, idx, inner_total, FOLD_TOTAL,
            status="running",
            cat_feat=cfg["cat_feat"],
            solv_feat=cfg["solv_feat"],
            base_feat=cfg["base_feat"],
            kernel_structure=cfg["kernel_structure"],
            inner_done=0, inner_id_current=None, inner_label_current=None,
            fold_done_in_inner=0,
            started_at_epoch=t0, wall_time_so_far_s=0.0,
            hostname=hostname, slurm_job_id=slurm_job, slurm_array_task_id=slurm_array,
            last_error=None,
        )

        try:
            payload = _run_outer_cell(
                idx, cfg, splits, run_meta,
                state_path=state_path, started_at_epoch=t0,
            )
        except Exception as e:
            update_run_state(
                state_path, idx, inner_total, FOLD_TOTAL,
                status="pending_failed",
                last_error=f"{type(e).__name__}: {e}",
                wall_time_so_far_s=time.time() - t0,
            )
            raise

        result_path = RESULTS_DIR / f"result_{idx}.json"
        result_path.write_text(json.dumps(payload))
        update_run_state(
            state_path, idx, inner_total, FOLD_TOTAL,
            status="done",
            inner_done=inner_total,
            inner_id_current=None, inner_label_current=None,
            fold_done_in_inner=FOLD_TOTAL,
            wall_time_so_far_s=time.time() - t0,
        )
        os.rename(running_dir / lock_name, done_dir / lock_name)
        print(f"[worker] outer cell {idx} done in {time.time() - t0:.1f}s -> {result_path.name}")

        if single:
            print("[worker] --single set, exiting after one cell.")
            return


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "?"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _live_squeue_counts() -> tuple[int | None, int | None]:
    if shutil.which("squeue") is None:
        return None, None
    user = os.environ.get("USER", "")
    res = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%j %T"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        return None, None
    active = pending_w = 0
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or JOB_NAME_PREFIX not in parts[0]:
            continue
        if parts[1] in ("RUNNING", "COMPLETING", "CONFIGURING"):
            active += 1
        elif parts[1] == "PENDING":
            pending_w += 1
    return active, pending_w


def _mean_done_cell_wall_time(done_dir: Path) -> tuple[float | None, int]:
    walls: list[float] = []
    for lock in done_dir.glob("job_*.lock"):
        idx = int(lock.stem.split("_")[1])
        result = RESULTS_DIR / f"result_{idx}.json"
        if not result.exists():
            continue
        try:
            payload = json.loads(result.read_text())
            wall = payload.get("total_wall_time_s")
            if isinstance(wall, (int, float)) and wall > 0:
                walls.append(float(wall))
        except (json.JSONDecodeError, OSError):
            continue
    if not walls:
        return None, 0
    return sum(walls) / len(walls), len(walls)


def status() -> None:
    """Print pending/running/done lock counts, SLURM worker counts and an ETA.

    The ETA is (pending + running cells) / concurrency × mean wall time of
    finished cells, where concurrency is the active SLURM worker count, or the
    running lock count when squeue reports none. It is omitted when no cell
    has finished.
    """
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"
    pending = len(list(pending_dir.glob("job_*.lock")))
    running = len(list(running_dir.glob("job_*.lock")))
    done = len(list(done_dir.glob("job_*.lock")))
    inputs = _load_inputs()
    print(f"--- Queue status ({QUEUE_DIR}) ---")
    print(f"Total outer cells     : {len(inputs)}")
    print(f"Pending               : {pending}")
    print(f"Running               : {running}")
    print(f"Done                  : {done}")

    active, pending_w = _live_squeue_counts()
    if active is not None:
        print(f"SLURM active workers  : {active}")
        print(f"SLURM pending workers : {pending_w}")

    mean_wall, n_done = _mean_done_cell_wall_time(done_dir)
    if mean_wall is None:
        print("Mean cell wall_time   : n/a")
        return
    print(f"Mean cell wall_time   : {_fmt_dur(mean_wall)} (n={n_done})")

    remaining = pending + running
    concurrency = max(1, active or running or 1)
    eta_s = (remaining / concurrency) * mean_wall
    print(f"Estimated remaining   : {_fmt_dur(eta_s)} @ {concurrency} workers")


def submitted() -> None:
    """Print one progress line per running outer cell from its run-state file.

    Each line shows the hyper-prior, catalyst featurization, inner-config and
    fold progress, elapsed time, a per-cell ETA and the host.
    """
    _ensure_dirs()
    running_dir = QUEUE_DIR / "running"
    inputs = _load_inputs()
    now = time.time()

    for lock in sorted(running_dir.glob("job_*.lock"), key=lambda p: int(p.stem.split("_")[1])):
        idx = int(lock.stem.split("_")[1])
        key = str(idx)
        if key not in inputs:
            continue
        cfg = inputs[key]
        inner_total = len(cfg["inner_cells"])
        state = load_run_state_for_reporting(
            get_run_state_path(QUEUE_DIR, idx), idx, inner_total, FOLD_TOTAL,
        )
        inner_done = int(state["inner_done"])
        fold_done = int(state["fold_done_in_inner"])
        elapsed_s = (
            now - float(state["started_at_epoch"])
            if state.get("started_at_epoch") is not None
            else float(state.get("wall_time_so_far_s", 0.0))
        )
        total_units = inner_total * FOLD_TOTAL
        units_done = inner_done * FOLD_TOTAL + fold_done
        frac = units_done / total_units if total_units > 0 else 0.0
        eta_cell = (elapsed_s * (1 - frac) / frac) if frac > 0 else None
        inner_label = state.get("inner_label_current") or "?"
        host = state.get("hostname") or "?"
        print(
            f"{idx:>3} | hp={cfg['hyperprior']:<10} cat={cfg['cat_feat']:<16} | "
            f"inner={inner_done:>2}/{inner_total} ({inner_label}) "
            f"fold={fold_done:>3}/{FOLD_TOTAL} | "
            f"elapsed={_fmt_dur(elapsed_s):>6} | ETA={_fmt_dur(eta_cell):>6} | {host}"
        )


def main() -> None:
    """Parse the CLI subcommand and dispatch to init, worker, status or submitted."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    wp = sub.add_parser("worker")
    wp.add_argument("--single", action="store_true")
    sub.add_parser("status")
    sub.add_parser("submitted")
    args = parser.parse_args()

    if args.cmd == "init":
        init_queue()
    elif args.cmd == "worker":
        worker_loop(single=args.single)
    elif args.cmd == "status":
        status()
    elif args.cmd == "submitted":
        submitted()


if __name__ == "__main__":
    main()
