"""Per-worker heartbeat state for the run_4 AF-optimizer comparison.

Each running worker maintains a JSON file at
``data/queue/run_state/job_<worker_id>.json`` describing its progress: which model /
scheme it is running, which tier of N_TIERS, the current sub-batch step, elapsed
wall-time, and SLURM/host identifiers.  The ``submitted`` CLI subcommand in
:mod:`scheduler` reads these files to produce a live view of every running worker.

Atomic-write + NFS-tolerant read patterns are mirrored from
:mod:`pablo.featurization_and_kernel_scan.cross_validation.run_state`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


RUN_STATE_DIRNAME = "run_state"


def get_run_state_dir(queue_dir: str | Path) -> Path:
    """Return ``<queue_dir>/run_state``, creating it if absent.

    Args:
        queue_dir: Root directory of the file-based queue.

    Returns:
        Path to the heartbeat directory.
    """
    d = Path(queue_dir) / RUN_STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_run_state_path(queue_dir: str | Path, worker_id: int) -> Path:
    """Return the path of one worker's heartbeat file, creating its directory.

    Args:
        queue_dir: Root directory of the file-based queue.
        worker_id: Worker index.

    Returns:
        ``<queue_dir>/run_state/job_<worker_id>.json``.
    """
    return get_run_state_dir(queue_dir) / f"job_{worker_id}.json"


def default_run_state(worker_id: int, opt_total: int) -> dict[str, Any]:
    """Build the heartbeat state of a worker that has not started.

    Args:
        worker_id: Worker index.
        opt_total: Total number of acquisition optimizations the worker will run.

    Returns:
        State dict with status ``"pending_waiting"``, ``opt_done = 0``,
        ``updated_at_epoch`` set to the current time and every progress, timing
        and SLURM field ``None``.
    """
    return {
        "worker_id": int(worker_id),
        "status": "pending_waiting",
        "rep_idx": None,
        "model_name": None,
        "current_scheme": None,
        "current_tier": None,
        "opt_done": 0,
        "opt_total": int(opt_total),
        "stage": None,
        "sequential_step": None,
        "sequential_step_total": None,
        "started_at_epoch": None,
        "updated_at_epoch": time.time(),
        "wall_time_so_far_s": 0.0,
        "hostname": None,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
    }


def _normalize(
    state: dict[str, Any], worker_id: int, opt_total: int,
) -> dict[str, Any]:
    out = dict(state)
    out["worker_id"] = int(out.get("worker_id", worker_id))
    out["status"] = str(out.get("status", "pending_waiting"))
    out["rep_idx"] = (
        int(out["rep_idx"]) if out.get("rep_idx") is not None else None
    )
    out["model_name"] = out.get("model_name", None)
    out["current_scheme"] = out.get("current_scheme", None)
    out["current_tier"] = (
        int(out["current_tier"]) if out.get("current_tier") is not None else None
    )
    out["opt_done"] = int(out.get("opt_done", 0))
    out["opt_total"] = int(out.get("opt_total", opt_total))
    out["stage"] = out.get("stage", None)
    out["sequential_step"] = (
        int(out["sequential_step"])
        if out.get("sequential_step") is not None
        else None
    )
    out["sequential_step_total"] = (
        int(out["sequential_step_total"])
        if out.get("sequential_step_total") is not None
        else None
    )
    out["started_at_epoch"] = (
        float(out["started_at_epoch"])
        if out.get("started_at_epoch") is not None
        else None
    )
    out["updated_at_epoch"] = float(out.get("updated_at_epoch", time.time()))
    out["wall_time_so_far_s"] = float(out.get("wall_time_so_far_s", 0.0))
    out["hostname"] = out.get("hostname", None)
    for k in ("slurm_job_id", "slurm_array_task_id"):
        v = out.get(k, None)
        out[k] = str(v) if v is not None else None
    if "state_file_warning" in out:
        out["state_file_warning"] = str(out["state_file_warning"])
    return out


def _extract_first_json_object(raw_text: str) -> tuple[str, bool]:
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
    state_path: Path, worker_id: int, opt_total: int,
) -> dict[str, Any]:
    """Read and normalize a worker's heartbeat file.

    The first complete JSON object in the file is used; trailing data after it
    is ignored. A missing or empty file, or one without a complete JSON object,
    yields ``default_run_state``.

    Args:
        state_path: Path to the heartbeat JSON file.
        worker_id: Worker index, used for defaults and missing fields.
        opt_total: Total optimization count, used for defaults and missing fields.

    Returns:
        Normalized state dict.

    Raises:
        json.JSONDecodeError: If the extracted JSON object does not parse.
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
        raw = default_run_state(worker_id, opt_total)
    return _normalize(raw, worker_id, opt_total)


def load_run_state_for_reporting(
    state_path: Path, worker_id: int, opt_total: int,
) -> dict[str, Any]:
    """Read a worker's heartbeat file for status reporting, flagging read problems.

    Like ``load_run_state``, but a fallback to defaults or ignored trailing data
    is recorded in ``state_file_warning``: ``"missing_after_exists_race"`` when
    the file vanishes between the existence check and the read,
    ``"state_json_unreadable_using_defaults"`` when it holds no complete JSON
    object, and ``"state_json_extra_trailing_data_ignored"`` when data follows
    the first object. A file that does not exist gives the defaults with no
    warning.

    Args:
        state_path: Path to the heartbeat JSON file.
        worker_id: Worker index, used for defaults and missing fields.
        opt_total: Total optimization count, used for defaults and missing fields.

    Returns:
        Normalized state dict, with ``state_file_warning`` when one applies.

    Raises:
        json.JSONDecodeError: If the extracted JSON object does not parse.
    """
    if not state_path.exists():
        return _normalize(
            default_run_state(worker_id, opt_total), worker_id, opt_total,
        )
    try:
        text = state_path.read_text(encoding="utf-8")
        blob, has_trailing = _extract_first_json_object(text)
    except FileNotFoundError:
        s = default_run_state(worker_id, opt_total)
        s["state_file_warning"] = "missing_after_exists_race"
        return _normalize(s, worker_id, opt_total)
    if not blob:
        s = default_run_state(worker_id, opt_total)
        s["state_file_warning"] = "state_json_unreadable_using_defaults"
        return _normalize(s, worker_id, opt_total)
    raw = json.loads(blob)
    out = _normalize(raw, worker_id, opt_total)
    if has_trailing:
        out["state_file_warning"] = "state_json_extra_trailing_data_ignored"
    return out


def write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    """Write a heartbeat state atomically.

    The JSON goes to a uniquely named temporary file in the same directory,
    is fsynced, and is then renamed onto ``state_path``, so readers never see a
    partial file.

    Args:
        state_path: Destination JSON file; its parent directory is created.
        state: JSON-serializable state dict.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_name(
        f"{state_path.stem}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(state_path)


def update_run_state(
    state_path: Path, worker_id: int, opt_total: int, **updates: Any,
) -> dict[str, Any]:
    """Merge field updates into a worker's heartbeat file and write it atomically.

    ``updated_at_epoch`` is set to the current time after the updates are applied.

    Args:
        state_path: Path to the heartbeat JSON file.
        worker_id: Worker index, used if the file is missing or empty.
        opt_total: Total optimization count, used if the file is missing or empty.
        **updates: State fields to overwrite.

    Returns:
        The state dict as written.
    """
    state = load_run_state(state_path, worker_id, opt_total)
    for k, v in updates.items():
        state[k] = v
    state["updated_at_epoch"] = time.time()
    write_run_state(state_path, state)
    return state
