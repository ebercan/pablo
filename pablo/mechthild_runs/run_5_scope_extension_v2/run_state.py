"""Per-job run-state files of the run_5 queue.

One JSON file per combination, ``<queue_dir>/run_state/job_<idx>.json``. The
worker and the job write it, always through a temporary file and a rename, so a
reader sees either the previous or the new state; the scheduler's ``status``,
``submitted`` and ``failed-pending`` commands read it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pablo.mechthild_runs.run_5_scope_extension_v2.config import RUN_STATE_DIRNAME


def get_run_state_path(queue_dir: str | Path, combination_index: int) -> Path:
    """State file of one combination; creates ``<queue_dir>/run_state/`` if needed.

    Args:
        queue_dir: Queue directory of the operating mode.
        combination_index: Index in ``inputs.json``.

    Returns:
        Path of ``job_<combination_index>.json``.
    """
    run_state_dir = Path(queue_dir) / RUN_STATE_DIRNAME
    run_state_dir.mkdir(parents=True, exist_ok=True)
    return run_state_dir / f"job_{combination_index}.json"


def default_run_state(combination_index: int, total_iterations: int) -> dict[str, Any]:
    """State of a job that has not been claimed yet.

    Args:
        combination_index: Index in ``inputs.json``.
        total_iterations: Planned BO iteration count.

    Returns:
        State dict with status ``pending_waiting``.
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
        "last_error": None,
        "last_return_code": None,
        "updated_at_epoch": time.time(),
    }


def load_run_state(state_path: Path, combination_index: int, total_iterations: int) -> dict[str, Any]:
    """State stored at ``state_path``, or ``default_run_state`` if there is no file.

    Args:
        state_path: Path from ``get_run_state_path``.
        combination_index: Index in ``inputs.json``.
        total_iterations: Planned BO iteration count.

    Returns:
        State dict.
    """
    if not state_path.exists():
        return default_run_state(combination_index, total_iterations)
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    """Write ``state`` to ``state_path`` through a temporary file and a rename.

    Args:
        state_path: Path from ``get_run_state_path``.
        state: State dict.
    """
    temp_path = state_path.with_name(f"{state_path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
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
    """Apply ``updates`` to the stored state and write it back.

    Args:
        state_path: Path from ``get_run_state_path``.
        combination_index: Index in ``inputs.json``.
        total_iterations: Planned BO iteration count.
        **updates: Fields to set.

    Returns:
        The updated state dict.
    """
    state = load_run_state(state_path, combination_index, total_iterations)
    state.update(updates)
    state["updated_at_epoch"] = time.time()
    write_run_state(state_path, state)
    return state


def check_existing_result(path: Path, combination: dict[str, object]) -> None:
    """Raise unless the result file at ``path`` is a full run of ``combination``.

    A result left over from an earlier ``inputs.json`` (another index layout or
    setting), or a shortened smoke run of this combination, would otherwise be
    skipped as finished.

    Args:
        path: An existing ``result_<idx>.json``.
        combination: The ``inputs.json`` entry of that index.

    Raises:
        ValueError: If the file's stored combination differs from
            ``combination`` or the run used ``--n_eval_override`` or
            ``--num_fantasies_override``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stored = payload.get("combination")
    if stored != combination:
        raise ValueError(
            f"{path} holds a result for {stored!r}, not for the current combination "
            f"{combination!r}; move it out of the results directory before running"
        )
    options = payload.get("execution_options", {})
    overrides = {k: options.get(k) for k in ("n_eval_override", "num_fantasies_override")}
    if any(v is not None for v in overrides.values()):
        raise ValueError(
            f"{path} is a shortened run ({overrides}); move it out of the results "
            "directory before running"
        )
