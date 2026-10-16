"""Per-worker heartbeat files for run_6.

A running worker keeps ``data/queue/run_state/job_<id>.json`` up to date with its
stage, start time and host; ``scheduler submitted`` reads them. Writes are atomic
(temporary file + rename) so a reader never sees a partial file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def run_state_path(queue_dir: Path, worker_id: int) -> Path:
    """``<queue_dir>/run_state/job_<worker_id>.json``."""
    return queue_dir / "run_state" / f"job_{worker_id}.json"


def write_run_state(path: Path, state: dict[str, Any]) -> None:
    """Write ``state`` to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def read_run_state(path: Path) -> dict[str, Any] | None:
    """The heartbeat at ``path``, or ``None`` when the worker has not written one."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_run_state(path: Path, **updates: Any) -> dict[str, Any]:
    """Merge ``updates`` into the heartbeat at ``path``, stamp ``updated_at_epoch`` and write it."""
    state = read_run_state(path) or {}
    state.update(updates)
    state["updated_at_epoch"] = time.time()
    write_run_state(path, state)
    return state
