"""File-based queue for run_6.

``N_WORKERS`` jobs, one per ``(optimizer, tier, rep)``. Lock files move
``pending/ → running/ → done/`` under ``data/queue/``; a job is claimed by an atomic
rename. A worker started with ``--tiers`` claims only jobs of those tiers, so an array
submitted with one tier's partition, time limit and memory never runs another tier's
job. A job whose worker crashes keeps its lock in ``running/`` and has no result
file; ``requeue`` moves such locks back to ``pending/`` once no worker is alive.

Usage (from the repo root):
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler init
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler worker [--single] [--tiers T ...]
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler status
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler submitted
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.scheduler requeue
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

from pablo.mechthild_runs.run_6_af_optimizer_af_value import config
from pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant import (
    DATA_DIR,
    RESULTS_DIR,
    decode_worker,
    run_worker,
)
from pablo.mechthild_runs.run_6_af_optimizer_af_value.run_state import (
    read_run_state,
    run_state_path,
    update_run_state,
)

QUEUE_DIR = DATA_DIR / "queue"
PENDING, RUNNING, DONE = QUEUE_DIR / "pending", QUEUE_DIR / "running", QUEUE_DIR / "done"


def _ensure_dirs() -> None:
    """Create the queue and results directories."""
    for d in (PENDING, RUNNING, DONE, QUEUE_DIR / "run_state", RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _lock_ids(d: Path) -> list[int]:
    """Sorted job ids with a lock in ``d``."""
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("job_*.lock"))


def init_queue() -> None:
    """Pend every job without a result file; jobs with one are marked done."""
    _ensure_dirs()
    created = done = 0
    for idx in range(config.N_WORKERS):
        lock = f"job_{idx}.lock"
        if (RESULTS_DIR / f"result_{idx}.json").exists():
            for stale in (PENDING / lock, RUNNING / lock):
                if stale.exists():
                    stale.unlink()
            (DONE / lock).touch(exist_ok=True)
            done += 1
            continue
        if any((d / lock).exists() for d in (PENDING, RUNNING, DONE)):
            continue
        (PENDING / lock).touch()
        created += 1
    print(f"[init] {created} pending locks created, {done} jobs already done, {config.N_WORKERS} total")


def pending_ids(tiers: set[int] | None = None) -> list[int]:
    """Pending job ids, restricted to ``tiers`` when given."""
    return [i for i in _lock_ids(PENDING) if tiers is None or decode_worker(i)[1] in tiers]


def _claim_next(tiers: set[int] | None) -> int | None:
    """Claim the lowest pending job of ``tiers`` (any tier when ``None``) by renaming its
    lock into ``running/``; ``None`` when none is left."""
    for idx in pending_ids(tiers):
        src = PENDING / f"job_{idx}.lock"
        try:
            os.rename(src, RUNNING / src.name)
        except FileNotFoundError:
            continue
        return idx
    return None


def worker_loop(single: bool, tiers: set[int] | None) -> None:
    """Run pending jobs of ``tiers`` (all when ``None``) until none is left; ``single`` stops after one."""
    _ensure_dirs()
    print(f"[worker] {socket.gethostname()}, queue {QUEUE_DIR}, tiers {sorted(tiers) if tiers else 'all'}", flush=True)
    while (idx := _claim_next(tiers)) is not None:
        state = run_state_path(QUEUE_DIR, idx)
        update_run_state(
            state,
            status="running",
            started_at_epoch=time.time(),
            hostname=socket.gethostname(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            stage="loading",
        )
        run_worker(idx, heartbeat=lambda **kw: update_run_state(state, **kw))
        update_run_state(state, status="done", stage=None)
        os.rename(RUNNING / f"job_{idx}.lock", DONE / f"job_{idx}.lock")
        if single:
            return
    print("[worker] no pending jobs left", flush=True)


def status() -> None:
    """Print the lock counts and the number of result files, in total and per tier."""
    _ensure_dirs()
    n_results = len(list(RESULTS_DIR.glob("result_*.json")))
    print(
        f"pending {len(_lock_ids(PENDING))} | running {len(_lock_ids(RUNNING))} | "
        f"done {len(_lock_ids(DONE))} | results {n_results} / {config.N_WORKERS}"
    )
    for tier in range(config.N_TIERS):
        counts = [sum(decode_worker(i)[1] == tier for i in _lock_ids(d)) for d in (PENDING, RUNNING, DONE)]
        print(f"  tier {tier} ({config.TIER_TARGET_MINUTES[tier]:g} min): pending {counts[0]} running {counts[1]} done {counts[2]}")


def submitted() -> None:
    """One line per running job: unit, stage, elapsed time, host."""
    _ensure_dirs()
    now = time.time()
    ids = _lock_ids(RUNNING)
    if not ids:
        print("no running jobs")
        return
    for idx in ids:
        optimizer, tier, rep = decode_worker(idx)
        state = read_run_state(run_state_path(QUEUE_DIR, idx))
        if state is None:
            print(f"{idx:>3} | {optimizer:>11} tier {tier} rep {rep} | no heartbeat")
            continue
        elapsed = now - state["started_at_epoch"]
        since = now - state["updated_at_epoch"]
        print(
            f"{idx:>3} | {optimizer:>11} tier {tier} rep {rep} | {state.get('stage')} | "
            f"elapsed {elapsed / 60:5.1f} min (target {config.TIER_TARGET_MINUTES[tier]:g}) | "
            f"last update {since:4.0f} s ago | {state.get('hostname')}"
        )


def requeue() -> None:
    """Move every running lock without a result file back to ``pending/``.

    Run only when ``squeue`` shows no live worker of this run; a live worker's lock
    would be pended twice.
    """
    _ensure_dirs()
    moved = []
    for idx in _lock_ids(RUNNING):
        if not (RESULTS_DIR / f"result_{idx}.json").exists():
            os.rename(RUNNING / f"job_{idx}.lock", PENDING / f"job_{idx}.lock")
            moved.append(idx)
    print(f"[requeue] {len(moved)} jobs back to pending: {moved}")


def main() -> None:
    """CLI: ``init`` | ``worker [--single]`` | ``status`` | ``submitted`` | ``requeue``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    wp = sub.add_parser("worker")
    wp.add_argument("--single", action="store_true")
    wp.add_argument("--tiers", type=int, nargs="+", choices=range(config.N_TIERS))
    sub.add_parser("status")
    sub.add_parser("submitted")
    sub.add_parser("requeue")
    args = parser.parse_args()
    if args.cmd == "init":
        init_queue()
    elif args.cmd == "worker":
        worker_loop(args.single, set(args.tiers) if args.tiers else None)
    elif args.cmd == "status":
        status()
    elif args.cmd == "submitted":
        submitted()
    elif args.cmd == "requeue":
        requeue()


if __name__ == "__main__":
    main()
