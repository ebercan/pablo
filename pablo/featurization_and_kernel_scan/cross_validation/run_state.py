"""Per-worker heartbeat state for the featurization & kernel scan.

Each running worker maintains a JSON file at
``<queue_dir>/run_state/job_<outer_cell_id>.json`` describing what inner
config and fold it is currently on, the elapsed wall-time, the SLURM/host
identifiers, and any error. The scheduler's ``submitted`` subcommand reads
these files to produce a live view of every running worker.

Writes are atomic (temporary file, then ``replace``). Reads tolerate the
truncated or trailing bytes that concurrent writes over NFS can leave behind.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


RUN_STATE_DIRNAME = "run_state"


def get_run_state_dir(queue_dir: str | Path) -> Path:
    """Return ``queue_dir/run_state/``, creating it if missing.

    Args:
        queue_dir: The run's queue directory.

    Returns:
        The ``run_state/`` directory inside it.
    """
    run_state_dir = Path(queue_dir) / RUN_STATE_DIRNAME
    run_state_dir.mkdir(parents=True, exist_ok=True)
    return run_state_dir


def get_run_state_path(queue_dir: str | Path, outer_cell_id: int) -> Path:
    """Return the heartbeat JSON path for one outer cell.

    Args:
        queue_dir: The run's queue directory.
        outer_cell_id: Index of the outer cell (one worker per cell).

    Returns:
        ``queue_dir/run_state/job_<outer_cell_id>.json``.
    """
    return get_run_state_dir(queue_dir) / f"job_{outer_cell_id}.json"





def _normalize(
    state: dict[str, Any], outer_cell_id: int, inner_total: int, fold_total: int
) -> dict[str, Any]:
    """Fill missing fields with defaults and coerce field types.

    Args:
        state: Raw heartbeat state as read from disk.
        outer_cell_id: Default for ``outer_cell_id``.
        inner_total: Default for ``inner_total``.
        fold_total: Default for ``fold_total_in_inner``.

    Returns:
        A new dict holding every heartbeat field, with ``state``'s values
        where present and defaults elsewhere.
    """
    out = dict(state)
    out["outer_cell_id"] = int(out.get("outer_cell_id", outer_cell_id))
    out["status"] = str(out.get("status", "pending_waiting"))
    for k in ("cat_feat", "solv_feat", "base_feat", "kernel_structure"):
        out[k] = out.get(k, None)
    out["inner_done"] = int(out.get("inner_done", 0))
    out["inner_total"] = int(out.get("inner_total", inner_total))
    out["inner_id_current"] = (
        int(out["inner_id_current"]) if out.get("inner_id_current") is not None else None
    )
    out["inner_label_current"] = out.get("inner_label_current", None)
    out["fold_done_in_inner"] = int(out.get("fold_done_in_inner", 0))
    out["fold_total_in_inner"] = int(out.get("fold_total_in_inner", fold_total))
    out["started_at_epoch"] = (
        float(out["started_at_epoch"]) if out.get("started_at_epoch") is not None else None
    )
    out["updated_at_epoch"] = float(out.get("updated_at_epoch", time.time()))
    out["wall_time_so_far_s"] = float(out.get("wall_time_so_far_s", 0.0))
    out["hostname"] = out.get("hostname", None)
    for k in ("slurm_job_id", "slurm_array_task_id"):
        v = out.get(k, None)
        out[k] = str(v) if v is not None else None
    out["last_error"] = out.get("last_error", None)
    if "state_file_warning" in out:
        out["state_file_warning"] = str(out["state_file_warning"])
    return out


def _extract_first_json_object(raw_text: str) -> tuple[str, bool]:
    """Pull out the first top-level JSON object from a possibly-truncated payload.

    Tolerant of concurrent-write trailing bytes that NFS can leave behind.

    Args:
        raw_text: File contents.

    Returns:
        ``(blob, has_trailing_content)``: the text of the first complete
        top-level JSON object (``""`` if there is none or it is unbalanced),
        and whether non-whitespace text follows it.
    """
    text = raw_text.lstrip()
    if not text:
        return "", False
    start = text.find("{")
    if start < 0:
        return "", False
    depth, in_string, escaped, end = 0, False, False, -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
            if depth < 0:
                return "", False
    if depth != 0 or end < 0:
        return "", False
    return text[start : end + 1], bool(text[end + 1 :].strip())


def load_run_state(
    state_path: Path, outer_cell_id: int, inner_total: int, fold_total: int
) -> dict[str, Any]:
    """Load a worker's heartbeat state, using defaults if the file is missing or empty.

    Args:
        state_path: Heartbeat JSON path.
        outer_cell_id: Outer cell the worker runs.
        inner_total: Number of inner cells in that outer cell.
        fold_total: Number of CV folds per inner cell.

    Returns:
        The normalized heartbeat state dict (see ``_normalize``).
    """
    raw: dict[str, Any] | None = None
    try:
        text = state_path.read_text(encoding="utf-8")
        if text:
            blob, _ = _extract_first_json_object(text)
            if blob:
                raw = json.loads(blob)
    except FileNotFoundError:
        pass
    if raw is None:
        raw = {
            "outer_cell_id": int(outer_cell_id),
            "status": "pending_waiting",
            "cat_feat": None,
            "solv_feat": None,
            "base_feat": None,
            "kernel_structure": None,
            "inner_done": 0,
            "inner_total": int(inner_total),
            "inner_id_current": None,
            "inner_label_current": None,
            "fold_done_in_inner": 0,
            "fold_total_in_inner": int(fold_total),
            "started_at_epoch": None,
            "updated_at_epoch": time.time(),
            "wall_time_so_far_s": 0.0,
            "hostname": None,
            "slurm_job_id": None,
            "slurm_array_task_id": None,
            "last_error": None,
        }
    return _normalize(raw, outer_cell_id, inner_total, fold_total)


def load_run_state_for_reporting(
    state_path: Path, outer_cell_id: int, inner_total: int, fold_total: int
) -> dict[str, Any]:
    """Load a worker's heartbeat state for the scheduler's reporting views.

    A missing file gives the default state. A file that vanishes between the
    existence check and the read, an unreadable file, or a file with trailing
    bytes after the first JSON object gives a state annotated with
    ``state_file_warning`` (and ``last_error`` for the first two).

    Args:
        state_path: Heartbeat JSON path.
        outer_cell_id: Outer cell the worker runs.
        inner_total: Number of inner cells in that outer cell.
        fold_total: Number of CV folds per inner cell.

    Returns:
        The normalized heartbeat state dict (see ``_normalize``).
    """
    if not state_path.exists():
        return _normalize(
            {
                "outer_cell_id": int(outer_cell_id),
                "status": "pending_waiting",
                "cat_feat": None,
                "solv_feat": None,
                "base_feat": None,
                "kernel_structure": None,
                "inner_done": 0,
                "inner_total": int(inner_total),
                "inner_id_current": None,
                "inner_label_current": None,
                "fold_done_in_inner": 0,
                "fold_total_in_inner": int(fold_total),
                "started_at_epoch": None,
                "updated_at_epoch": time.time(),
                "wall_time_so_far_s": 0.0,
                "hostname": None,
                "slurm_job_id": None,
                "slurm_array_task_id": None,
                "last_error": None,
            },
            outer_cell_id, inner_total, fold_total,
        )
    try:
        text = state_path.read_text(encoding="utf-8")
        blob, has_trailing = _extract_first_json_object(text)
    except FileNotFoundError:
        s = {
            "outer_cell_id": int(outer_cell_id),
            "status": "pending_waiting",
            "cat_feat": None,
            "solv_feat": None,
            "base_feat": None,
            "kernel_structure": None,
            "inner_done": 0,
            "inner_total": int(inner_total),
            "inner_id_current": None,
            "inner_label_current": None,
            "fold_done_in_inner": 0,
            "fold_total_in_inner": int(fold_total),
            "started_at_epoch": None,
            "updated_at_epoch": time.time(),
            "wall_time_so_far_s": 0.0,
            "hostname": None,
            "slurm_job_id": None,
            "slurm_array_task_id": None,
            "last_error": None,
        }
        s["state_file_warning"] = "missing_after_exists_race"
        s["last_error"] = "state_file_not_found"
        return _normalize(s, outer_cell_id, inner_total, fold_total)
    if not blob:
        s = {
            "outer_cell_id": int(outer_cell_id),
            "status": "pending_waiting",
            "cat_feat": None,
            "solv_feat": None,
            "base_feat": None,
            "kernel_structure": None,
            "inner_done": 0,
            "inner_total": int(inner_total),
            "inner_id_current": None,
            "inner_label_current": None,
            "fold_done_in_inner": 0,
            "fold_total_in_inner": int(fold_total),
            "started_at_epoch": None,
            "updated_at_epoch": time.time(),
            "wall_time_so_far_s": 0.0,
            "hostname": None,
            "slurm_job_id": None,
            "slurm_array_task_id": None,
            "last_error": None,
        }
        s["state_file_warning"] = "state_json_unreadable_using_defaults"
        s["last_error"] = "state_file_corrupt"
        return _normalize(s, outer_cell_id, inner_total, fold_total)
    raw = json.loads(blob)
    out = _normalize(raw, outer_cell_id, inner_total, fold_total)
    if has_trailing:
        out["state_file_warning"] = "state_json_extra_trailing_data_ignored"
    return out


def write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    """Write the heartbeat state as JSON atomically.

    Writes and fsyncs a uniquely named temporary file next to ``state_path``,
    then replaces ``state_path`` with it. Creates the parent directory if
    missing.

    Args:
        state_path: Destination heartbeat JSON path.
        state: JSON-serializable state dict.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_name(f"{state_path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(state_path)


def update_run_state(
    state_path: Path,
    outer_cell_id: int,
    inner_total: int,
    fold_total: int,
    **updates: Any,
) -> dict[str, Any]:
    """Load the heartbeat state, apply updates and write it back atomically.

    ``updated_at_epoch`` is set to the current time.

    Args:
        state_path: Heartbeat JSON path.
        outer_cell_id: Outer cell the worker runs.
        inner_total: Number of inner cells in that outer cell.
        fold_total: Number of CV folds per inner cell.
        **updates: Fields to set on the state.

    Returns:
        The state dict as written.
    """
    state = load_run_state(state_path, outer_cell_id, inner_total, fold_total)
    for k, v in updates.items():
        state[k] = v
    state["updated_at_epoch"] = time.time()
    write_run_state(state_path, state)
    return state
