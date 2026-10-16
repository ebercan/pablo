"""File-based queue scheduler for the featurization × kernel CV scan.

One outer cell is one model configuration ``(cat_feat, solv_feat,
kernel_structure, model_class, outcome_transform, base_kernel, mean_function)``
plus its ``inner_cells`` (``base_feat``, ``prior_scheme``). On every grid of
``config.RUN_GRIDS`` each cell has exactly one inner cell, so a cell is one
configuration run over all ``config.FOLD_TOTAL`` folds.

``--run <name>`` (default ``elimination``) selects the run: its ``inputs.json``,
queue, run state and result cells come from ``paths.run_paths(name)`` and its
SLURM job name from ``submit_mechthild.slurm_job_name(name)``.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler <subcommand> [--run <name>] [options]

Subcommands:
  init             Touch one ``pending/job_<outer_id>.lock`` per outer cell
                   missing a result file. Idempotent.
  worker           Loop, atomically claim the lowest-id pending lock, run its
                   inner cells × 137 folds, write
                   ``<cells_dir>/result_<outer_id>.json``,
                   move the lock to ``done/``. Repeat until empty.
                   ``--single`` exits after one cell (one-cell SLURM tasks).
  status           Print pending / running / done counts plus live SLURM
                   worker count, mean completed-cell wall-time, and a global ETA.
  submitted        Per-running-worker view: ``inner=k/n fold=f/137``, elapsed,
                   per-cell ETA, host. Reads heartbeat JSONs maintained by
                   the workers under ``<queue_dir>/run_state/``.
  estimate-runtime Pre-flight projection from the Mechthild cost model:
                   slowest cells, CPU-h per elimination sweep (per axis level
                   on a crossed grid), and full-scan wall time at several SLURM
                   concurrencies.

Pattern mirrors pablo/mechthild_runs/run_1_scope_extension/mechthild_scheduler_cpu.py
without the operating-mode multiplexing.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import heapq
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
from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import build_partition
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import fit_eval
from pablo.featurization_and_kernel_scan.cross_validation.run_state import (
    load_run_state_for_reporting,
    update_run_state,
)
from pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild import (
    slurm_job_name,
)

ROOT = paths.PKG_ROOT
DATA_DIR = paths.DATA_DIR
RUN_NAME = config.DEFAULT_RUN
INPUTS_PATH = paths.INPUTS_JSON
QUEUE_DIR = paths.QUEUE_DIR
RUN_STATE_DIR = paths.RUN_STATE_DIR
RESULTS_DIR = paths.RESULTS_CELLS_DIR
SPLITS_PATH = paths.CV_SPLITS_JSON
CONFIG_PATH = Path(config.__file__)
JOB_NAME_PREFIX = slurm_job_name(RUN_NAME)
FOLD_TOTAL = config.FOLD_TOTAL
STALE_HEARTBEAT_SECONDS = 1800.0


def select_run(run: str) -> None:
    """Point every module-level path and the SLURM job name at one run.

    Args:
        run: A key of ``config.RUN_GRIDS``.

    Raises:
        KeyError: If ``run`` is not a known run.
    """
    global RUN_NAME, INPUTS_PATH, QUEUE_DIR, RUN_STATE_DIR, RESULTS_DIR, JOB_NAME_PREFIX
    if run not in config.RUN_GRIDS:
        raise KeyError(f"unknown run {run!r}; known runs: {sorted(config.RUN_GRIDS)}")
    run_paths = paths.run_paths(run)
    RUN_NAME = run
    INPUTS_PATH = run_paths.inputs_json
    QUEUE_DIR = run_paths.queue_dir
    RUN_STATE_DIR = run_paths.run_state_dir
    RESULTS_DIR = run_paths.cells_dir
    JOB_NAME_PREFIX = slurm_job_name(run)





def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_commit() -> str | None:
    if shutil.which("git") is None:
        return None
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None








def _ensure_dirs() -> None:
    for sub in ("pending", "running", "done"):
        (QUEUE_DIR / sub).mkdir(parents=True, exist_ok=True)
    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _load_completed_folds(result_path: Path) -> tuple[dict | None, set[tuple]]:
    """Load an existing result JSON and return (payload, completed_fold_keys).

    A fold key is ``(inner_id, split_type, repeat, fold)``.
    """
    if not result_path.exists():
        return None, set()
    payload = json.loads(result_path.read_text())
    completed = set()
    for inner_block in payload.get("inner_cells", []):
        inner_id = inner_block["inner_id"]
        for fold in inner_block.get("folds", []):
            if fold.get("status", "ok") == "ok":
                completed.add((inner_id, fold["split_type"], fold["repeat"], fold["fold"]))
    return payload, completed


def _existing_fold_records_map(payload: dict) -> dict[tuple, dict]:
    records: dict[tuple, dict] = {}
    for inner_block in payload.get("inner_cells", []):
        inner_id = inner_block["inner_id"]
        for fold in inner_block.get("folds", []):
            key = (inner_id, fold["split_type"], fold["repeat"], fold["fold"])
            records[key] = fold
    return records


def _verify_resume_hashes(existing_payload: dict, current_splits_hash: str) -> None:
    existing_splits = existing_payload.get("run_meta", {}).get("splits_sha256")
    if existing_splits is not None and existing_splits != current_splits_hash:
        raise RuntimeError(
            f"CV splits hash mismatch: existing result has {existing_splits[:16]}..., "
            f"current is {current_splits_hash[:16]}.... Cannot resume with different splits."
        )
    existing_config = existing_payload.get("run_meta", {}).get("config_sha256")
    current_config = _file_sha256(CONFIG_PATH)
    if existing_config is not None and existing_config != current_config:
        print(
            f"[resume] WARNING: config.py hash changed "
            f"({existing_config[:12]}... -> {current_config[:12]}...). "
            f"Bug fixes in models.py/fit_eval.py are fine; prior/grid changes are not.",
            file=sys.stderr, flush=True,
        )





def _heartbeat_age_seconds(idx: int, inner_total: int) -> float | None:
    """Seconds since the worker on outer cell ``idx`` last wrote its heartbeat.

    Args:
        idx: Outer cell id.
        inner_total: Number of inner cells for that outer cell.

    Returns:
        Age in seconds, or ``None`` if the cell has never written a heartbeat.
    """
    state_path = RUN_STATE_DIR / f"job_{idx}.json"
    if not state_path.exists():
        return None
    state = load_run_state_for_reporting(state_path, idx, inner_total, FOLD_TOTAL)
    return max(0.0, time.time() - float(state["updated_at_epoch"]))


def _worker_is_live(idx: int, inner_total: int, stale_after: float) -> bool:
    """Whether a worker still appears to be running outer cell ``idx``.

    A cell is live only if it has a heartbeat younger than ``stale_after``.
    A cell with no heartbeat at all has never been picked up, or its worker
    died before writing one; either way it is not live.

    Args:
        idx: Outer cell id.
        inner_total: Number of inner cells for that outer cell.
        stale_after: Heartbeat age in seconds beyond which a worker is
            considered dead.

    Returns:
        True if a worker is presumed alive on this cell.
    """
    age = _heartbeat_age_seconds(idx, inner_total)
    return age is not None and age < stale_after


CELL_IDENTITY_KEYS: tuple[str, ...] = (
    "cat_feat",
    "solv_feat",
    "kernel_structure",
    "model_class",
    "outcome_transform",
    "base_kernel",
    "mean_function",
)
"""Outer-config keys that must agree before a stored result is credited to a cell.

``results/cells/`` and ``data/grid/`` are one flat namespace keyed by integer
outer-cell id, so enumerating a *different* grid into the same namespace makes
cell 0 of the new grid collide with cell 0 of the old one. Without this check
``init`` reads the old result, counts 137 completed folds, and marks the new
cell done — the new configuration is never fitted and the old numbers are
reported under the new labels. Finished runs belong in ``archive/<date>/``.
"""


INNER_IDENTITY_KEYS: tuple[str, ...] = ("base_feat", "prior_scheme")
"""Inner-config keys checked the same way. Cells of the prior-scheme sweep share
every outer key and differ only here."""


def _assert_result_matches_cell(result_payload: dict, cell_cfg: dict, idx: int) -> None:
    """Fail loudly if a stored result was produced by a different configuration.

    Args:
        result_payload: Parsed ``results/cells/result_<idx>.json``.
        cell_cfg: The ``inputs.json`` record for the same outer cell id.
        idx: Outer cell id, for the error message.

    Raises:
        ValueError: If any identifying axis differs between the two.
    """
    stored = result_payload.get("outer_config", {})
    mismatched = {
        key: (stored.get(key), cell_cfg.get(key))
        for key in CELL_IDENTITY_KEYS
        if key in cell_cfg and stored.get(key) != cell_cfg.get(key)
    }
    inputs_inner = {inner["inner_id"]: inner for inner in cell_cfg["inner_cells"]}
    for block in result_payload.get("inner_cells", []):
        stored_inner = block["inner_config"]
        now_inner = inputs_inner.get(stored_inner["inner_id"], {})
        for key in INNER_IDENTITY_KEYS:
            if stored_inner.get(key) != now_inner.get(key):
                mismatched[f"inner {stored_inner['inner_id']} {key}"] = (
                    stored_inner.get(key), now_inner.get(key),
                )
    if mismatched:
        detail = ", ".join(
            f"{key}: stored={was!r} != inputs={now!r}" for key, (was, now) in mismatched.items()
        )
        raise ValueError(
            f"{RESULTS_DIR / f'result_{idx}.json'} was produced by a different "
            f"configuration than inputs.json now names for cell {idx} ({detail}). "
            "The results and queue directories are one namespace keyed by cell id, "
            "so a previous grid's results are sitting where this grid's belong. "
            "Archive the finished run under archive/<date>/ (cells, logs, inputs.json "
            "and its aggregates) before enumerating a new grid."
        )


def init_queue(stale_after: float = STALE_HEARTBEAT_SECONDS) -> None:
    """Create pending locks for outer cells that need work.

    A cell needs work if it has no result JSON, or if its result JSON has
    incomplete or failed folds.  Cells with all folds ``status=="ok"`` get
    a done lock and are skipped.

    A cell whose worker died before finishing its first inner cell has no
    result JSON but still holds a lock in ``running/``.  Such a cell is
    requeued rather than skipped, which is what a lock-only existence check
    would do.  Liveness is decided from the run-state heartbeat, so a cell a
    worker is actively running is never requeued underneath it.

    Args:
        stale_after: Heartbeat age in seconds beyond which the worker holding
            a ``running/`` lock is presumed dead and its cell is requeued.
    """
    _ensure_dirs()
    inputs = json.loads(INPUTS_PATH.read_text())
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    created = 0
    skipped_done = 0
    resume_partial = 0
    requeued_stranded = 0
    left_live = 0
    for key in sorted(inputs.keys(), key=int):
        idx = int(key)
        lock_name = f"job_{idx}.lock"
        result_file = RESULTS_DIR / f"result_{idx}.json"
        inner_total = len(inputs[key]["inner_cells"])
        live = _worker_is_live(idx, inner_total, stale_after)

        if result_file.exists():
            payload, completed = _load_completed_folds(result_file)
            _assert_result_matches_cell(payload, inputs[key], idx)
            expected = inner_total * FOLD_TOTAL
            if len(completed) >= expected:
                skipped_done += 1
                (done_dir / lock_name).touch(exist_ok=True)
                for stale in (pending_dir / lock_name, running_dir / lock_name):
                    if stale.exists():
                        stale.unlink()
                continue
            if live:
                left_live += 1
                continue
            resume_partial += 1
            (pending_dir / lock_name).touch(exist_ok=True)
            for stale in (running_dir / lock_name, done_dir / lock_name):
                if stale.exists():
                    stale.unlink()
            continue

        if live:
            left_live += 1
            continue

        if (pending_dir / lock_name).exists():
            continue

        stranded = [d for d in (running_dir, done_dir) if (d / lock_name).exists()]
        for stale in stranded:
            (stale / lock_name).unlink()
        (pending_dir / lock_name).touch()
        if stranded:
            requeued_stranded += 1
        else:
            created += 1

    print(
        f"[init] {skipped_done} complete, {resume_partial} partial (queued for resume), "
        f"{requeued_stranded} stranded (no result, dead worker - requeued), "
        f"{left_live} left to live workers, "
        f"{created} new pending ({len(inputs)} total outer cells)"
    )


def _claim_next(pending_dir: Path, running_dir: Path) -> int | None:
    """Atomically claim one pending lock; return its idx or None if empty.
        Returns:
            int | None: Return value of _claim_next.
    
    Returns:
        int | None: Return value of _claim_next.
"""
    for pending_file in sorted(
        pending_dir.glob("job_*.lock"),
        key=lambda p: int(p.stem.split("_")[1]),
    ):
        idx = int(pending_file.stem.split("_")[1])
        target = running_dir / pending_file.name
        try:
            os.rename(pending_file, target)
            return idx
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def _run_inner_cell(
    outer_cfg: dict,
    inner_cfg: dict,
    cell_features,
    splits: list,
    on_fold_done=None,
    completed_folds: set[tuple] | None = None,
    existing_fold_records: dict[tuple, dict] | None = None,
) -> dict:
    """Run all CV folds for one inner (base_feat, prior_scheme) configuration.

    Args:
        outer_cfg: Dictionary containing outer cell configuration; supplies
            the model configuration (``kernel_structure``, ``outcome_transform``,
            ``model_class``, ``base_kernel``, ``mean_function``) plus the
            ``cat_feat`` / ``solv_feat`` keys.
        inner_cfg: Dictionary containing inner cell configuration:
            ``inner_id``, ``base_feat`` (the base featurization this cell
            assembled features for) and optional ``prior_scheme``.
        cell_features: A ``CellFeatures`` instance pre-assembled for the
            given ``(cat, solv, inner base_feat)``.
        splits: List of repeat blocks, each containing a list of CV folds.
        on_fold_done: Optional callback invoked after each fold finishes,
            used by the scheduler to bump the worker's heartbeat file.
        completed_folds: Set of ``(inner_id, split_type, repeat, fold)``
            keys already completed with ``status=="ok"``.  Folds in this
            set are skipped and their existing records reused.
        existing_fold_records: Map from fold key to the existing fold
            record dict, used to reuse results for skipped folds.

    Returns:
        A dictionary containing the inner config, total folds executed,
        wall time, and a list of fold records.
    """
    kernel_structure = outer_cfg["kernel_structure"]
    outcome_transform = outer_cfg["outcome_transform"]
    prior_scheme = inner_cfg.get("prior_scheme")

    fold_records: list[dict] = []
    folds_done = 0
    folds_skipped = 0
    t_cell = time.time()
    for repeat_block in splits:
        split_type = repeat_block.get("split_type", "primary")
        repeat_idx = repeat_block["repeat"]
        type_offset = int(hashlib.sha256(split_type.encode()).hexdigest()[:8], 16)

        for fold_idx, fold in enumerate(repeat_block["folds"]):
            fold_key = (inner_cfg["inner_id"], split_type, repeat_idx, fold_idx)

            if completed_folds and fold_key in completed_folds:
                rec = existing_fold_records[fold_key]
                fold_records.append(rec)
                folds_done += 1
                folds_skipped += 1
                if on_fold_done is not None:
                    on_fold_done(folds_done)
                continue

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
                    model_class=outer_cfg["model_class"],
                    base_kernel=outer_cfg["base_kernel"],
                    mean_function=outer_cfg["mean_function"],
                    prior_scheme=prior_scheme,
                )
                rec = dataclasses.asdict(fold_res)
            except Exception as e:
                print(
                    f"[FOLD_FAILED] outer={outer_cfg['outer_cell_id']} "
                    f"cat={outer_cfg['cat_feat']} solv={outer_cfg['solv_feat']} "
                    f"base={inner_cfg['base_feat']} transform={outcome_transform} | "
                    f"{kernel_structure}/{outer_cfg['model_class']}/"
                    f"{outer_cfg['base_kernel']}/"
                    f"{outer_cfg.get('mean_function', 'constant')} | "
                    f"{split_type}/r{repeat_idx}/f{fold_idx} | "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )
                nan = float("nan")
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
                    "baseline_nlpd_per_point": [],
                    "mu_pred_per_point": [],
                    "var_pred_per_point": [],
                    "crpss": nan,
                    "msll": nan,
                    "q2": nan,
                    "mae": nan,
                    "rmse": nan,
                    "calibration_ks": nan,
                    "nlpd_per_point_sel": [],
                    "crps_per_point_sel": [],
                    "baseline_crps_per_point_sel": [],
                    "baseline_nlpd_per_point_sel": [],
                    "mu_pred_per_point_sel": [],
                    "var_pred_per_point_sel": [],
                    "crpss_sel": nan,
                    "msll_sel": nan,
                    "q2_sel": nan,
                    "mae_sel": nan,
                    "rmse_sel": nan,
                    "calibration_ks_sel": nan,
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

    if folds_skipped > 0:
        print(
            f"[resume] inner {inner_cfg['inner_id']}: skipped {folds_skipped}/{folds_done} "
            f"already-ok folds",
            flush=True,
        )

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
    result_path: Path | None = None,
    splits_hash: str | None = None,
) -> dict:
    """Run all inner configs for one outer cell, resuming from existing results.

    Args:
        idx: The ID of the outer cell.
        outer_cfg: Configuration dictionary for the outer cell (supplies the
            model configuration — ``kernel_structure``, ``model_class``,
            ``outcome_transform``, ``base_kernel``, ``mean_function`` — plus the
            ``cat_feat`` / ``solv_feat`` keys; ``base_feat`` is on the inner axis
            and drives a per-inner feature re-assembly).
        splits: List of repeat blocks for the cross-validation splits.
        run_meta: Metadata describing the current run (e.g. host, commit).
        state_path: Optional path to the worker's heartbeat JSON.
        started_at_epoch: Optional start time of the worker.
        result_path: Path to ``result_<idx>.json``.  If the file exists,
            completed folds are loaded and skipped.  Updated atomically
            after each inner cell completes.
        splits_hash: SHA-256 of ``cv_splits.json``, checked against any
            existing result for consistency.

    Returns:
        A dictionary containing the outer configuration, run metadata,
        feature dimensions, execution time, and results of all inner cells.
    """
    t0 = time.time()
    started = started_at_epoch if started_at_epoch is not None else t0
    inner_total = len(outer_cfg["inner_cells"])

    existing_payload, completed_folds = _load_completed_folds(
        result_path
    ) if result_path is not None else (None, set())

    if existing_payload is not None:
        if splits_hash is not None:
            _verify_resume_hashes(existing_payload, splits_hash)
        existing_records = _existing_fold_records_map(existing_payload)
        n_resumable = len(completed_folds)
        n_expected = len(outer_cfg["inner_cells"]) * FOLD_TOTAL
        print(
            f"[resume] outer cell {idx}: {n_resumable}/{n_expected} folds already ok, "
            f"running remaining {n_expected - n_resumable}",
            flush=True,
        )
    else:
        existing_records = {}

    inner_results = []
    for k, inner_cfg in enumerate(outer_cfg["inner_cells"]):
        t_inner = time.time()
        base_feat = inner_cfg["base_feat"]
        try:
            if state_path is not None:
                update_run_state(
                    state_path, idx, inner_total, FOLD_TOTAL,
                    inner_done=k,
                    inner_id_current=inner_cfg["inner_id"],
                    inner_label_current=(
                        f"base={base_feat} "
                        f"P={inner_cfg.get('prior_scheme', 'legacy')}"
                    ),
                    fold_done_in_inner=0,
                    wall_time_so_far_s=time.time() - started,
                )

            cell_features = assemble_features(
                cat_featurization=outer_cfg["cat_feat"],
                solv_featurization=outer_cfg["solv_feat"],
                base_featurization=base_feat,
                experiments_path=paths.EXPERIMENTS_PARQUET,
                featurizations_root=paths.FEAT_DIR,
                libraries_root=paths.LIB_DIR,
                append_form_onehot_to_cat=outer_cfg.get("append_form_onehot_to_cat", False),
            )
            prior_scheme = inner_cfg.get("prior_scheme")
            cell_features.partition = build_partition(
                cell_features.feature_dims,
                mode=(
                    "legacy" if prior_scheme is None
                    else config.PRIOR_SCHEMES[prior_scheme].partition_mode
                ),
            )
            feature_dims_snapshot = {
                "cat": len(cell_features.feature_dims["cat"]),
                "solv": len(cell_features.feature_dims["solv"]),
                "base": len(cell_features.feature_dims["base"]),
                "cont": len(cell_features.feature_dims["cont"]),
                "partition_group_sizes": {
                    block: [len(g) for g in groups]
                    for block, groups in cell_features.partition.items()
                },
                "active_flag_dim": cell_features.active_flag_dim,
                "form_id_dim": cell_features.form_id_dim,
                "X_total": cell_features.X.shape[1],
                "n_observations": cell_features.X.shape[0],
            }

            def _on_fold(folds_done, _idx=idx, _started=started, _inner_total=inner_total):
                if state_path is None:
                    return
                update_run_state(
                    state_path, _idx, _inner_total, FOLD_TOTAL,
                    fold_done_in_inner=folds_done,
                    wall_time_so_far_s=time.time() - _started,
                )

            inner_res = _run_inner_cell(
                outer_cfg, inner_cfg, cell_features, splits,
                on_fold_done=_on_fold,
                completed_folds=completed_folds,
                existing_fold_records=existing_records,
            )
            inner_res["feature_dims"] = feature_dims_snapshot
            inner_res["feature_names"] = cell_features.feature_names

        except Exception as e:
            inner_res = {
                "inner_id": inner_cfg["inner_id"],
                "inner_config": inner_cfg,
                "prior_scheme": inner_cfg.get("prior_scheme"),
                "n_folds": 0,
                "wall_time_s": time.time() - t_inner,
                "folds": [],
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }

        inner_results.append(inner_res)

        if result_path is not None:
            _atomic_write_json(result_path, {
                "outer_cell_id": idx,
                "outer_config": {k: v for k, v in outer_cfg.items() if k != "inner_cells"},
                "run_meta": run_meta,
                "n_inner_cells": len(inner_results),
                "total_wall_time_s": time.time() - t0,
                "inner_cells": inner_results,
            })

    return {
        "outer_cell_id": idx,
        "outer_config": {k: v for k, v in outer_cfg.items() if k != "inner_cells"},
        "run_meta": run_meta,
        "n_inner_cells": len(inner_results),
        "total_wall_time_s": time.time() - t0,
        "inner_cells": inner_results,
    }


def worker_loop(single: bool = False) -> None:
    """Claim and run pending outer cells until the queue drains.

    Args:
        single: If True, claim and process exactly one outer cell, then return.
            Use for local one-shot smoke tests that exercise the same code
            path the HPC worker runs.
    """
    warnings.filterwarnings("ignore", message="A not p.d.", module="linear_operator")
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    inputs = json.loads(INPUTS_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())
    splits_hash = _file_sha256(SPLITS_PATH)
    config_hash = _file_sha256(CONFIG_PATH)
    run_meta = {
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "run": RUN_NAME,
        "splits_sha256": splits_hash,
        "config_sha256": config_hash,
        "lengthscale_floor": config.LENGTHSCALE_FLOOR,
        "lengthscale_lognormal_scale": config.LENGTHSCALE_LOGNORMAL_SCALE,
        "lengthscale_loc_shift": config.LENGTHSCALE_LOGNORMAL_LOC_SHIFT,
        "noise_inverse_gamma": {
            f"{target}/{transform}": {"alpha": evidence.alpha, "beta": evidence.beta}
            for (target, transform), evidence in config.NOISE_EVIDENCE.items()
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

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
            f"[worker] running outer cell {idx}: {cfg['cat_feat']}/"
            f"{cfg['solv_feat']}/{cfg['kernel_structure']}/{cfg['model_class']} "
            f"({inner_total} base cells)"
        )

        t0 = time.time()
        state_path = RUN_STATE_DIR / f"job_{idx}.json"
        update_run_state(
            state_path, idx, inner_total, FOLD_TOTAL,
            status="running",
            cat_feat=cfg["cat_feat"], base_feat=None,
            solv_feat=cfg["solv_feat"], kernel_structure=cfg["kernel_structure"],
            model_class=cfg["model_class"],
            inner_done=0, inner_id_current=None, inner_label_current=None,
            fold_done_in_inner=0,
            started_at_epoch=t0, wall_time_so_far_s=0.0,
            hostname=hostname, slurm_job_id=slurm_job, slurm_array_task_id=slurm_array,
            last_error=None,
        )

        result_path = RESULTS_DIR / f"result_{idx}.json"

        try:
            payload = _run_outer_cell(
                idx, cfg, splits, run_meta,
                state_path=state_path, started_at_epoch=t0,
                result_path=result_path, splits_hash=splits_hash,
            )
        except Exception as e:
            update_run_state(
                state_path, idx, inner_total, FOLD_TOTAL,
                status="pending_failed",
                last_error=f"{type(e).__name__}: {e}",
                wall_time_so_far_s=time.time() - t0,
            )
            raise
        update_run_state(
            state_path, idx, inner_total, FOLD_TOTAL,
            status="done",
            inner_done=inner_total,
            inner_id_current=None, inner_label_current=None,
            fold_done_in_inner=FOLD_TOTAL,
            wall_time_so_far_s=time.time() - t0,
        )
        try:
            os.rename(running_dir / lock_name, done_dir / lock_name)
        except (FileNotFoundError, OSError):
            pass
        print(f"[worker] outer cell {idx} done in {time.time() - t0:.1f}s -> {result_path.name}")

        if single:
            print("[worker] --single set, exiting after one cell.")
            return


def _fmt_dur(seconds: float | None) -> str:
    """Render a duration in the smallest human-readable form.
        Returns:
            str: Return value of _fmt_dur.
    
    Returns:
        str: Return value of _fmt_dur.
"""
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN-safe
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
    """Return (active_workers, pending_workers) from ``squeue``, or (None, None).
        Returns:
            tuple[int | None, int | None]: Return value of _live_squeue_counts.
    
    Returns:
        tuple[int | None, int | None]: Return value of _live_squeue_counts.
"""
    if shutil.which("squeue") is None:
        return None, None
    user = os.environ.get("USER", "")
    res = subprocess.run(
        ["squeue", "-u", user, "-h", "-r", "-o", "%j %T"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        return None, None
    active = pending_w = 0
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != JOB_NAME_PREFIX:
            continue
        if parts[1] in ("RUNNING", "COMPLETING", "CONFIGURING"):
            active += 1
        elif parts[1] == "PENDING":
            pending_w += 1
    return active, pending_w


def _mean_done_cell_wall_time(done_dir: Path) -> tuple[float | None, int]:
    """Read ``total_wall_time_s`` from completed result files; return (mean, n).
        Returns:
            tuple[float | None, int]: Return value of _mean_done_cell_wall_time.
    
    Returns:
        tuple[float | None, int]: Return value of _mean_done_cell_wall_time.
"""
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
    """Print the run's queue status.

    Prints pending, running and done cell counts, live SLURM worker counts
    (when ``squeue`` is available), the mean wall time of completed cells,
    and the estimated remaining time and completion clock time at the current
    worker concurrency.
    """
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"
    pending = len(list(pending_dir.glob("job_*.lock")))
    running = len(list(running_dir.glob("job_*.lock")))
    done = len(list(done_dir.glob("job_*.lock")))
    inputs = json.loads(INPUTS_PATH.read_text())
    print(f"--- Queue status ({QUEUE_DIR}) ---")
    print(f"Total outer cells in inputs : {len(inputs)}")
    print(f"Pending                     : {pending}")
    print(f"Running                     : {running}")
    print(f"Done                        : {done}")

    active, pending_w = _live_squeue_counts()
    if active is not None:
        print(f"SLURM workers (active)      : {active}")
        print(f"SLURM workers (pending)     : {pending_w}")

    mean_wall, n_done = _mean_done_cell_wall_time(done_dir)
    if mean_wall is None:
        print(
            "Mean cell wall_time         : n/a (no completed results yet "
            "— run 'estimate-runtime' for a pre-flight projection)"
        )
        return
    print(f"Mean cell wall_time         : {_fmt_dur(mean_wall)} (n={n_done} completed)")

    remaining = pending + running  # running cells are partially done, but treat as full units for the upper bound
    concurrency = max(1, active or running or 1)
    eta_global_s = (remaining / concurrency) * mean_wall
    print(f"Estimated remaining         : {_fmt_dur(eta_global_s)} @ {concurrency} live workers")
    eta_completion = time.time() + eta_global_s
    print(
        f"ETA scan completion         : "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(eta_completion))} (local)"
    )


def submitted() -> None:
    """List every currently-running worker with per-cell progress + ETA."""
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    inputs = json.loads(INPUTS_PATH.read_text())
    lines: list[str] = []
    now = time.time()
    for lock in sorted(running_dir.glob("job_*.lock"), key=lambda p: int(p.stem.split("_")[1])):
        idx = int(lock.stem.split("_")[1])
        key = str(idx)
        if key not in inputs:
            continue
        cfg = inputs[key]
        inner_total = len(cfg["inner_cells"])
        state = load_run_state_for_reporting(
            RUN_STATE_DIR / f"job_{idx}.json", idx, inner_total, FOLD_TOTAL,
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
        warn = state.get("state_file_warning")
        warn_suffix = f"  [WARN={warn}]" if warn else ""
        err = state.get("last_error")
        err_suffix = f"  [ERR={err}]" if err else ""
        lines.append(
            f"{idx:>3} | {cfg['cat_feat']}/{cfg['solv_feat']}/"
            f"{cfg['kernel_structure']}/{cfg['model_class']} | "
            f"inner={inner_done:>2}/{inner_total} ({inner_label}) fold={fold_done:>3}/{FOLD_TOTAL} | "
            f"elapsed={_fmt_dur(elapsed_s):>6} | ETA-cell={_fmt_dur(eta_cell):>6} | "
            f"host={host}{warn_suffix}{err_suffix}"
        )
    for line in lines:
        print(line)

    pending = len(list(pending_dir.glob("job_*.lock")))
    running = len(list(running_dir.glob("job_*.lock")))
    done = len(list(done_dir.glob("job_*.lock")))
    mean_wall, n_done = _mean_done_cell_wall_time(done_dir)
    summary = f"[summary] {done} done | {running} running | {pending} pending"
    if mean_wall is not None:
        active, _ = _live_squeue_counts()
        concurrency = max(1, active or running or 1)
        remaining = pending + running
        eta_global = (remaining / concurrency) * mean_wall
        summary += f" | mean_cell={_fmt_dur(mean_wall)} (n={n_done}) | ETA-global={_fmt_dur(eta_global)} @ {concurrency}cc"
    print(summary)


# Cost model, measured on Mechthild by ``timing_probe`` (jobs 599692 and 599730,
# 2026-09-26): every configuration of the elimination grid at three catalyst
# widths, two folds each, one array task per configuration on the short partition
# at --cpus-per-task=1 --mem=4G, 16 tasks per node. All 96 folds finished ok.
_PROBE_WIDTHS: tuple[int, ...] = (44, 5028, 7453)
"""Design-matrix widths of the probed levels ``grover_pca``, ``grover`` and ``soap``."""

_PRIMARY_SECONDS_PER_FOLD: dict[str, tuple[float, float, float]] = {
    "baseline": (4.46, 12.59, 29.39),
    "kernel_structure=KS2": (3.91, 12.44, 27.74),
    "kernel_structure=KS3": (4.65, 10.80, 23.37),
    "kernel_structure=KS4": (4.25, 8.91, 18.74),
    "prior_scheme=P2": (3.85, 26.18, 26.24),
    "prior_scheme=P3": (1.88, 6.95, 7.14),
    "outcome_transform=logit": (4.00, 11.28, 19.61),
    "outcome_transform=arcsin_sqrt": (4.21, 13.17, 21.91),
    "outcome_transform=probit": (5.09, 11.81, 19.18),
    "mean_function=block_linear": (14.78, 129.45, 249.55),
    "mean_function=type_linear": (20.70, 80.74, 124.41),
    "mean_function=type_quadlog": (20.95, 85.10, 240.03),
}
"""Seconds per ``primary`` fold (51 training points) at ``_PROBE_WIDTHS``, keyed by
``timing_probe.level_label``.

A cell's width is interpolated linearly between the probed widths and held flat
outside them. The mean functions carry the cost: at ``soap`` width
``block_linear`` and ``type_quadlog`` are 8x the baseline, and kernel structure,
prior scheme and outcome transform stay within 0.24x-2.1x of it."""

_ESTIMATED_FACTORS: dict[str, tuple[float, float, float]] = {
    "kernel_structure=KS5": (1.35, 1.35, 1.35),
    "kernel_structure=KS6": (1.2, 1.2, 1.2),
    "kernel_structure=KS7": (1.5, 10.7, 10.7),
    "mean_function=cond_linear": (2.6, 2.6, 2.6),
    "mean_function=cond_trend": (1.1, 1.1, 1.1),
    "mean_function=form_offset": (1.05, 1.05, 1.05),
    "model_class=lcm_cs": (10.2, 10.2, 10.2),
    "model_class=logsum_cy": (7.7, 7.7, 7.7),
    "model_class=condprod_cy": (2.65, 2.65, 2.65),
}
"""Slowdowns against the baseline at ``_PROBE_WIDTHS`` for the levels ``timing_probe``
has not measured.

Ratios of seconds per fold to KS1 / constant on the same anchor, measured on one
local machine on three folds (two ``primary``, one ``leave_one_out``) of the
``kernel_mean_1`` cells of the ``grover_pca`` (29 columns), ``scage_carbene``
(about 530) and ``unimol`` (1,559) anchors. KS5 1.31-1.42, KS6 1.03-1.37,
``cond_linear`` 2.46-2.75, ``cond_trend`` 0.89-1.10 and ``form_offset`` about 1.0
do not trend with width and are held flat. KS7 evaluates the catalyst kernel
against the 161 library rows and grows with catalyst width: 1.5, 3.6 and 4.3 at
the three anchors; its 5,028- and 7,453-column values are set so that the linear
interpolation reproduces 4.3 at 1,559 columns and are not measurements. Levels
compose multiplicatively, as for the probed ones: KS7 with ``cond_linear`` on
``unimol`` measured 9.2 against the product 10.7.

The model classes are timed the same way on the ``product_1`` setting only
(``grover_pca``, 44 columns, P2, 5 restarts): seconds per fold against ``st_y`` on
the same transform, summed over three folds. ``lcm_cs`` 9.1 (raw) and 11.2
(probit), ``logsum_cy`` 7.7 (raw), ``condprod_cy`` 2.65 (probit). They are held
flat in width, untested beyond 44 columns."""

_PRIMARY_SECONDS_PER_FOLD.update({
    label: tuple(f * s for f, s in zip(factors, _PRIMARY_SECONDS_PER_FOLD["baseline"]))
    for label, factors in _ESTIMATED_FACTORS.items()
})

_LOO_TO_PRIMARY_COST: float = 1.22
"""Cost of a ``leave_one_out`` fold (68 training points) relative to a ``primary``
fold (51), the median over 12 configurations probed on both (range 0.68-2.39).
Folds of the other studies scale linearly in training-set size between the two."""

STARTUP_SECONDS: float = 67.0
"""One-off cost before a task's first fold: imports and ``assemble_features``.
Measured as task wall time minus fold time over the 36 primary probe tasks,
median 67 s (15-72 s)."""


@functools.lru_cache(maxsize=None)
def _stored_width(name: str) -> int:
    """Column count of ``data/featurizations/<name>/descriptors.parquet``.

    Args:
        name: Featurization directory name.

    Returns:
        Number of columns.
    """
    import pandas as pd

    return pd.read_parquet(paths.FEAT_DIR / name / "descriptors.parquet").shape[1]


def _design_matrix_width(cell: dict, inner: dict) -> int:
    """Assembled design-matrix width for one inner cell.

    Args:
        cell: An ``inputs.json`` outer-cell record.
        inner: One of its ``inner_cells``.

    Returns:
        Total columns: catalyst block (plus the form one-hot where appended),
        solvent, base, the ``base_active`` flag, the continuous block and
        ``form_id``.
    """
    d_cat = _stored_width(cell["cat_feat"])
    if cell.get("append_form_onehot_to_cat"):
        d_cat += 4
    d_solv = _stored_width(f"solvent_{cell['solv_feat']}")
    d_base = _stored_width(f"base_{inner['base_feat']}")
    return d_cat + d_solv + d_base + 1 + 4 + 1


@functools.lru_cache(maxsize=None)
def _fold_cost_factors() -> tuple[float, ...]:
    """Per-fold cost relative to a ``primary`` fold, one entry per fold of ``cv_splits.json``.

    Returns:
        ``1 + (_LOO_TO_PRIMARY_COST - 1) · (n_train - 51) / (68 - 51)`` per fold.
    """
    splits = json.loads(SPLITS_PATH.read_text())
    n_primary = len(splits[0]["folds"][0]["train"])
    n_loo = max(len(fold["train"]) for block in splits for fold in block["folds"])
    slope = (_LOO_TO_PRIMARY_COST - 1.0) / (n_loo - n_primary)
    return tuple(
        1.0 + slope * (len(fold["train"]) - n_primary)
        for block in splits for fold in block["folds"]
    )


COST_NEUTRAL_AXES: tuple[str, ...] = ("solv_feat", "base_feat", "base_kernel")
"""Axes whose level changes the projected cost only through the design-matrix
width. None was probed: the solvent and base blocks are 2-14 columns, and ``rbf``
is costed as ``matern52``."""


def _primary_seconds_per_fold(cell: dict, inner: dict) -> float:
    """Projected Mechthild seconds for one ``primary`` fold of one inner cell.

    The baseline time at the cell's width, times the probed slowdown of every
    axis on which the configuration differs from ``ELIMINATION_GRID.baseline``,
    each slowdown taken at that width. A one-axis configuration of the
    elimination grid therefore gets exactly its probed time; a crossed
    configuration composes its axes multiplicatively.

    Args:
        cell: An ``inputs.json`` outer-cell record.
        inner: One of its ``inner_cells``.

    Returns:
        Seconds per fold, interpolated in design-matrix width.

    Raises:
        KeyError: If the configuration moves an axis outside
            ``COST_NEUTRAL_AXES`` to a level that was never probed.
    """
    cfg = {**cell, **inner}
    width = _design_matrix_width(cell, inner)
    seconds = float(np.interp(width, _PROBE_WIDTHS, _PRIMARY_SECONDS_PER_FOLD["baseline"]))
    for axis, baseline_level in config.ELIMINATION_GRID.baseline.items():
        if axis in COST_NEUTRAL_AXES or cfg[axis] == baseline_level:
            continue
        probed = _PRIMARY_SECONDS_PER_FOLD[f"{axis}={cfg[axis]}"]
        seconds *= float(
            np.interp(width, _PROBE_WIDTHS, probed)
            / np.interp(width, _PROBE_WIDTHS, _PRIMARY_SECONDS_PER_FOLD["baseline"])
        )
    return seconds


def _per_cell_seconds(cell: dict) -> float:
    """Total wall time for one outer cell run by one task.

    Startup is paid once; every fold of every inner cell costs the projected
    ``primary`` time scaled by its training-set size (``_fold_cost_factors``).

    Args:
        cell: An ``inputs.json`` outer-cell record.

    Returns:
        Projected seconds for the whole cell.
    """
    folds = sum(_fold_cost_factors())
    return STARTUP_SECONDS + sum(
        _primary_seconds_per_fold(cell, inner) * folds for inner in cell["inner_cells"]
    )


def _cell_config(cell: dict) -> dict[str, str]:
    """The configuration of a one-inner-cell outer cell, outer and inner axes merged.

    Args:
        cell: An ``inputs.json`` outer-cell record with one inner cell.

    Returns:
        Every key of ``ELIMINATION_GRID.baseline`` mapped to the cell's level.
    """
    (inner,) = cell["inner_cells"]
    merged = {**cell, **inner}
    return {axis: merged[axis] for axis in config.ELIMINATION_GRID.baseline}


def _cost_breakdown(
    costed: list[tuple[float, str, dict]],
    grid: config.EliminationGrid | config.CrossedGrid | config.ListedGrid | config.AnchoredGrid,
) -> list[tuple[str, list[tuple[float, str, dict]]]]:
    """Group costed cells for the per-sweep or per-level CPU-time table.

    Args:
        costed: ``(seconds, cell_id, cell)`` per outer cell.
        grid: The grid the cells were enumerated from.

    Returns:
        ``(label, cells)`` pairs: one per sweep of an elimination grid (the
        baseline counts toward every sweep), one per level of every multi-level
        axis of a crossed or listed grid, the catalyst axis included.
    """
    from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import (
        sweep_membership,
    )

    if isinstance(grid, config.EliminationGrid):
        return [
            (axis, [c for c in costed if axis in sweep_membership(_cell_config(c[2]), grid)])
            for axis in grid.sweeps
        ]
    groups = []
    for axis, levels in {"cat_feat": grid.cat_feats, **grid.axes}.items():
        if len(levels) < 2:
            continue
        for level in levels:
            groups.append((
                f"{axis}={level}",
                [c for c in costed if {**_cell_config(c[2]), "cat_feat": c[2]["cat_feat"]}[axis] == level],
            ))
    return groups


def _list_schedule_wall(seconds_in_order: list[float], concurrency: int) -> float:
    """Wall time of cells run in order on ``concurrency`` slots.

    Each cell starts on whichever slot frees first, which is how both draining
    workers and ``autoscale --single`` consume the id-ordered queue.

    Args:
        seconds_in_order: Per-cell seconds, in claim (ascending id) order.
        concurrency: Number of slots.

    Returns:
        Time at which the last cell finishes.
    """
    slots = [0.0] * min(concurrency, len(seconds_in_order))
    heapq.heapify(slots)
    for seconds in seconds_in_order:
        heapq.heappush(slots, heapq.heappop(slots) + seconds)
    return max(slots)


def estimate_runtime(
    inputs: dict[str, dict] | None = None,
    grid: config.EliminationGrid | config.CrossedGrid | config.ListedGrid | config.AnchoredGrid | None = None,
) -> None:
    """Instant projection: probed per-fold times × current grid.

    Per-cell wall time is the projected seconds per fold of the cell's
    configuration, interpolated in design-matrix width
    (``_primary_seconds_per_fold``), times the training-size-weighted fold count,
    plus startup. Full-scan wall time is projected at several SLURM concurrencies
    by list-scheduling the cells in claim order (``_list_schedule_wall``). Once
    cells finish, ``scheduler status`` reports the true mean cell wall time from
    their result files.

    Args:
        inputs: Outer-cell map to project, as ``generate_inputs.enumerate_cells``
            returns it; the selected run's ``inputs.json`` when omitted.
        grid: The grid ``inputs`` was enumerated from; the selected run's grid
            when omitted.
    """
    from pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild import (
        MAX_CONCURRENT_WORKERS,
    )

    if inputs is None:
        inputs = json.loads(INPUTS_PATH.read_text())
    if grid is None:
        grid = config.RUN_GRIDS[RUN_NAME]
    n_cells = len(inputs)

    costed = []
    for cid, cell in inputs.items():
        costed.append((_per_cell_seconds(cell), cid, cell))
    costed.sort(key=lambda c: c[0], reverse=True)
    total_work_s = sum(c[0] for c in costed)
    max_cell_s = costed[0][0]

    print(
        f"[estimate-runtime] run={RUN_NAME}; cost model probed on Mechthild; "
        f"{STARTUP_SECONDS:.0f}s startup + per-level seconds per fold interpolated in width"
    )
    print(f"[grid] {n_cells} outer cells (= workers) × {FOLD_TOTAL} folds per inner cell")
    print()
    print("Most expensive cells:")
    print(
        f"  {'cell':>5}  {'cat_feat':<24}  {'solv':<10}  {'base':<10}  {'kernel':<9}  "
        f"{'transform':<11}  {'prior':<5}  {'per_cell':>10}"
    )
    for cell_s, cid, cell in costed[:8]:
        (inner,) = cell["inner_cells"]
        print(
            f"  {cid:>5}  {cell['cat_feat']:<24}  {cell['solv_feat']:<10}  "
            f"{inner['base_feat']:<10}  {cell['base_kernel']:<9}  "
            f"{cell['outcome_transform']:<11}  {inner['prior_scheme']:<5}  "
            f"{_fmt_dur(cell_s):>10}"
        )
    print(f"  ... median cell {_fmt_dur(costed[len(costed) // 2][0])}")
    print()
    if isinstance(grid, config.EliminationGrid):
        print("CPU time per sweep (the baseline cell counts toward all four):")
    else:
        print("CPU time per level of every multi-level axis:")
    for label, group in _cost_breakdown(costed, grid):
        print(
            f"  {label:<32} {len(group):>5} cells  "
            f"{sum(c[0] for c in group) / 3600:>7.1f} CPU-h  "
            f"slowest {_fmt_dur(max(c[0] for c in group))}"
        )
    print()
    print(
        f"Total work across all cells = {_fmt_dur(total_work_s)} "
        f"({total_work_s / 3600:.0f} CPU-h); longest single cell = {_fmt_dur(max_cell_s)}."
    )
    print()
    in_id_order = [inputs[k] for k in sorted(inputs, key=int)]
    seconds_in_order = [_per_cell_seconds(cell) for cell in in_id_order]
    print(
        "Full-scan wall time; workers claim cells in ascending id, each freed slot "
        "takes the next cell (draining workers, or `autoscale --single`):"
    )
    print(f"  {'concurrent':>10}  {'wall_time':>10}")
    for cc in sorted({50, 100, 200, MAX_CONCURRENT_WORKERS, n_cells}):
        print(f"  {cc:>10}  {_fmt_dur(_list_schedule_wall(seconds_in_order, cc)):>10}")
    cc = MAX_CONCURRENT_WORKERS
    rounds = [seconds_in_order[i:i + cc] for i in range(0, n_cells, cc)]
    print(
        f"Back-to-back rounds of {cc} (`submit --single` again only once the "
        f"previous round has drained): "
        + " + ".join(_fmt_dur(max(r)) for r in rounds)
        + f" = {_fmt_dur(sum(max(r) for r in rounds))}"
    )
    print()
    print(
        "[note] If the scan is already running, `scheduler status` reports the "
        "true mean cell wall-time from completed result_*.json files — "
        "use that over this table once any cells finish."
    )


def main() -> None:
    run_parent = argparse.ArgumentParser(add_help=False)
    run_parent.add_argument(
        "--run", choices=sorted(config.RUN_GRIDS), default=config.DEFAULT_RUN,
        help="Run whose inputs.json, queue and result cells to use (paths.run_paths).",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    init_parser = sub.add_parser(
        "init", parents=[run_parent], help="Initialize pending locks from inputs.json."
    )
    init_parser.add_argument(
        "--stale-after",
        type=float,
        default=STALE_HEARTBEAT_SECONDS,
        help="Heartbeat age (s) beyond which a running/ lock is treated as dead and requeued.",
    )
    worker_parser = sub.add_parser(
        "worker", parents=[run_parent], help="Run a worker loop until the queue drains."
    )
    worker_parser.add_argument(
        "--single", action="store_true",
        help="Claim and run exactly one outer cell, then exit. For local smoke tests.",
    )
    sub.add_parser(
        "status", parents=[run_parent],
        help="Print queue status, mean cell wall-time, and global ETA.",
    )
    sub.add_parser(
        "submitted", parents=[run_parent],
        help="Per-running-worker progress + ETA from heartbeat files.",
    )
    sub.add_parser(
        "estimate-runtime", parents=[run_parent],
        help="Project CPU-h and full-scan wall time from the probed cost model.",
    )
    args = parser.parse_args()
    select_run(args.run)

    if args.cmd == "init":
        init_queue(stale_after=args.stale_after)
    elif args.cmd == "worker":
        worker_loop(single=args.single)
    elif args.cmd == "status":
        status()
    elif args.cmd == "submitted":
        submitted()
    elif args.cmd == "estimate-runtime":
        estimate_runtime()


if __name__ == "__main__":
    main()
