"""Helpers for per-run state files in the run_1 scope-extension queue workflow."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pablo.mechthild_runs.run_1_scope_extension.config import RUN_STATE_DIRNAME


def get_run_state_dir(queue_dir: str | Path) -> Path:
    """Return run-state directory inside queue_state.

    Args:
        queue_dir: Path to queue_state root directory.

    Returns:
        Existing run-state directory path.
    """
    queue_path = Path(queue_dir)
    run_state_dir = queue_path / RUN_STATE_DIRNAME
    run_state_dir.mkdir(parents=True, exist_ok=True)
    return run_state_dir


def get_run_state_path(queue_dir: str | Path, combination_index: int) -> Path:
    """Build state file path for one combination index.

    Args:
        queue_dir: Path to queue_state root directory.
        combination_index: Combination index from inputs.json.

    Returns:
        Path to state JSON file.
    """
    run_state_dir = get_run_state_dir(queue_dir)
    return run_state_dir / f"job_{combination_index}.json"


def default_run_state(combination_index: int, total_iterations: int) -> dict[str, Any]:
    """Create default in-memory state payload.

    Args:
        combination_index: Combination index from inputs.json.
        total_iterations: Planned BO iteration count.

    Returns:
        Default run-state dictionary.
    """
    return {
        "combination_index": int(combination_index),
        "status": "pending_waiting",
        "attempts": 0,
        "current_iteration": 0,
        "total_iterations": int(total_iterations),
        "best_f": None,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "slurm_worker_id": None,
        "stale_recovered_count": 0,
        "last_error": None,
        "last_return_code": None,
        "updated_at_epoch": time.time(),
    }


def _normalize_run_state(
    state: dict[str, Any],
    combination_index: int,
    total_iterations: int,
) -> dict[str, Any]:
    """Normalize state fields and fill missing defaults.

    Args:
        state: Raw state dictionary loaded from disk.
        combination_index: Combination index from inputs.json.
        total_iterations: Planned BO iteration count.

    Returns:
        Normalized run-state dictionary.
    """
    normalized = dict(state)
    normalized["combination_index"] = int(
        normalized.get("combination_index", combination_index)
    )
    normalized["status"] = str(normalized.get("status", "pending_waiting"))
    normalized["attempts"] = int(normalized.get("attempts", 0))
    normalized["current_iteration"] = int(normalized.get("current_iteration", 0))
    normalized["total_iterations"] = int(
        normalized.get("total_iterations", total_iterations)
    )
    normalized["best_f"] = normalized.get("best_f", None)
    normalized["slurm_job_id"] = normalized.get("slurm_job_id", None)
    if normalized["slurm_job_id"] is not None:
        normalized["slurm_job_id"] = str(normalized["slurm_job_id"])

    normalized["slurm_array_task_id"] = normalized.get("slurm_array_task_id", None)
    if normalized["slurm_array_task_id"] is not None:
        normalized["slurm_array_task_id"] = str(normalized["slurm_array_task_id"])

    normalized["slurm_worker_id"] = normalized.get("slurm_worker_id", None)
    if normalized["slurm_worker_id"] is not None:
        normalized["slurm_worker_id"] = str(normalized["slurm_worker_id"])

    normalized["stale_recovered_count"] = int(
        normalized.get("stale_recovered_count", 0)
    )
    normalized["last_error"] = normalized.get("last_error", None)
    normalized["last_return_code"] = normalized.get("last_return_code", None)
    normalized["updated_at_epoch"] = float(
        normalized.get("updated_at_epoch", time.time())
    )
    if "state_file_warning" in normalized:
        normalized["state_file_warning"] = str(normalized["state_file_warning"])
    return normalized


def _extract_first_json_object_blob(raw_text: str) -> tuple[str, bool]:
    """Extract first top-level JSON object from text payload.

    Args:
        raw_text: Raw file content.

    Returns:
        Tuple (json_blob, has_trailing_content).
    """
    text = raw_text.lstrip()
    if not text:
        return "", False

    start_idx = text.find("{")
    if start_idx < 0:
        return "", False

    depth = 0
    in_string = False
    escaped = False
    end_idx = -1

    for idx in range(start_idx, len(text)):
        char = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
            continue

        if char == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
            if depth < 0:
                return "", False

    if depth != 0 or end_idx < 0:
        return "", False

    json_blob = text[start_idx : end_idx + 1]
    trailing = text[end_idx + 1 :].strip()
    return json_blob, bool(trailing)


def load_run_state(
    state_path: Path,
    combination_index: int,
    total_iterations: int,
) -> dict[str, Any]:
    """Load an existing run-state file or return defaults.

    The file is read directly, without an existence check, so a file removed
    by a concurrent writer is handled like a missing one. A missing or empty
    file, or one without a JSON object, yields ``default_run_state``. Only the
    first top-level JSON object is parsed; trailing content is ignored.

    Args:
        state_path: Path to state JSON file.
        combination_index: Combination index from inputs.json.
        total_iterations: Planned BO iteration count.

    Returns:
        Normalized run-state dictionary.
    """
    state: dict[str, Any] | None = None
    try:
        raw_text = state_path.read_text(encoding="utf-8")
        if raw_text:
            first_blob, _ = _extract_first_json_object_blob(raw_text)
            if first_blob:
                state = json.loads(first_blob)
    except FileNotFoundError:
        pass

    if state is None:
        state = default_run_state(
            combination_index=combination_index,
            total_iterations=total_iterations,
        )

    return _normalize_run_state(
        state=state,
        combination_index=combination_index,
        total_iterations=total_iterations,
    )


def load_run_state_for_reporting(
    state_path: Path,
    combination_index: int,
    total_iterations: int,
) -> dict[str, Any]:
    """Load state robustly for status/report commands.

    This loader tolerates trailing data by extracting the first JSON object
    from the file and annotating the returned state with a warning field.

    Args:
        state_path: Path to state JSON file.
        combination_index: Combination index from inputs.json.
        total_iterations: Planned BO iteration count.

    Returns:
        Normalized run-state dictionary with optional state_file_warning.
    """
    if not state_path.exists():
        return _normalize_run_state(
            state=default_run_state(
                combination_index=combination_index,
                total_iterations=total_iterations,
            ),
            combination_index=combination_index,
            total_iterations=total_iterations,
        )

    try:
        raw_text = state_path.read_text(encoding="utf-8")
        first_blob, has_trailing_content = _extract_first_json_object_blob(raw_text)
    except FileNotFoundError:
        state = default_run_state(
            combination_index=combination_index,
            total_iterations=total_iterations,
        )
        state["state_file_warning"] = "state_json_unreadable_missing_race"
        state["last_error"] = "state_file_not_found"
        return _normalize_run_state(
            state=state,
            combination_index=combination_index,
            total_iterations=total_iterations,
        )

    if not first_blob:
        state = default_run_state(
            combination_index=combination_index,
            total_iterations=total_iterations,
        )
        state["state_file_warning"] = "state_json_unreadable_using_defaults"
        state["last_error"] = "state_file_corrupt"
        return _normalize_run_state(
            state=state,
            combination_index=combination_index,
            total_iterations=total_iterations,
        )

    state = json.loads(first_blob)
    normalized = _normalize_run_state(
        state=state,
        combination_index=combination_index,
        total_iterations=total_iterations,
    )
    if has_trailing_content:
        normalized["state_file_warning"] = "state_json_extra_trailing_data_ignored"
    return normalized


def write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    """Write run-state JSON atomically.

    Args:
        state_path: Path to state JSON file.
        state: State payload to persist.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_name(
        f"{state_path.stem}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(state_path)


def update_run_state(
    state_path: Path,
    combination_index: int,
    total_iterations: int,
    **updates: Any,
) -> dict[str, Any]:
    """Update run-state fields and persist to disk.

    Args:
        state_path: Path to state JSON file.
        combination_index: Combination index from inputs.json.
        total_iterations: Planned BO iteration count.
        **updates: Field updates to apply.

    Returns:
        Updated state dictionary.
    """
    state = load_run_state(
        state_path=state_path,
        combination_index=combination_index,
        total_iterations=total_iterations,
    )
    for key, value in updates.items():
        state[key] = value
    state["updated_at_epoch"] = time.time()
    write_run_state(state_path=state_path, state=state)
    return state
