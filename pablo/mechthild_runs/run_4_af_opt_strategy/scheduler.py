"""File-based queue scheduler for the run_4 AF-optimizer comparison.

Flat queue of ``N_WORKERS`` jobs; one job = one ``run_worker(worker_id)`` call
that runs all ``config.N_TIERS`` GA-budget tiers of one ``(rep, model, scheme)``
cell and writes ``result_<id>.json``. Lock
files move through ``pending/`` -> ``running/`` -> ``done/`` under ``data/queue/``.

Subcommands:
  init       Touch ``pending/job_<i>.lock`` for every worker without a result file.
  worker     Loop: atomically claim one pending lock, run it, move it to done.
             ``--single`` exits after one job.
  status     Print pending/running/done counts.
  submitted  Per-running-worker progress from heartbeat files.

Usage:
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler init
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler worker [--single]
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler status
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.scheduler submitted
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

from pablo.mechthild_runs.run_4_af_opt_strategy import config
from pablo.mechthild_runs.run_4_af_opt_strategy.main_variant import _decode, run_worker
from pablo.mechthild_runs.run_4_af_opt_strategy.run_state import (
    get_run_state_path,
    load_run_state_for_reporting,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUEUE_DIR = DATA_DIR / "queue"
RESULTS_DIR = DATA_DIR / "results"

# One per-cell worker runs all N_TIERS tiers (one optimization per tier).
OPT_TOTAL = config.N_TIERS


def _ensure_dirs() -> None:
    """Create the queue subdirectories and the results directory if missing."""
    for sub in ("pending", "running", "done", "run_state"):
        (QUEUE_DIR / sub).mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def init_queue() -> None:
    """Touch a pending lock for every worker that has no result yet.

    Workers whose ``result_<i>.json`` already exists are marked done (and any
    stale pending/running lock for them is removed). Workers already represented
    by a lock in any queue directory are left untouched.
    """
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    created = 0
    skipped_done = 0
    for idx in range(config.N_WORKERS):
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
        f"({config.N_WORKERS} total workers)"
    )


def _claim_next(pending_dir: Path, running_dir: Path) -> int | None:
    """Atomically claim the lowest-indexed pending job by renaming its lock.

    Args:
        pending_dir: ``pending/`` directory.
        running_dir: ``running/`` directory.

    Returns:
        The claimed worker index, or ``None`` if no pending job remains.
    """
    for pf in sorted(
        pending_dir.glob("job_*.lock"), key=lambda p: int(p.stem.split("_")[1])
    ):
        idx = int(pf.stem.split("_")[1])
        target = running_dir / pf.name
        try:
            os.rename(pf, target)
            return idx
        except (FileNotFoundError, OSError):
            continue
    return None


def worker_loop(single: bool = False) -> None:
    """Claim and run pending jobs until the queue drains.

    Args:
        single: When True, exit after running exactly one job (the SLURM-array
            mode where each array task handles one worker).
    """
    _ensure_dirs()
    pending_dir = QUEUE_DIR / "pending"
    running_dir = QUEUE_DIR / "running"
    done_dir = QUEUE_DIR / "done"

    print(f"[worker] started on {socket.gethostname()}, queue={QUEUE_DIR}")
    while True:
        idx = _claim_next(pending_dir, running_dir)
        if idx is None:
            print("[worker] no pending jobs left, exiting.")
            return

        lock_name = f"job_{idx}.lock"
        print(f"[worker] running job {idx}")
        state_path = get_run_state_path(QUEUE_DIR, idx)
        run_worker(idx, state_path=state_path)
        os.rename(running_dir / lock_name, done_dir / lock_name)
        print(f"[worker] job {idx} done -> result_{idx}.json")

        if single:
            print("[worker] --single set, exiting after one job.")
            return


def status() -> None:
    """Print pending/running/done lock counts for the queue."""
    _ensure_dirs()
    pending = len(list((QUEUE_DIR / "pending").glob("job_*.lock")))
    running = len(list((QUEUE_DIR / "running").glob("job_*.lock")))
    done = len(list((QUEUE_DIR / "done").glob("job_*.lock")))
    print(f"--- Queue status ({QUEUE_DIR}) ---")
    print(f"Total workers : {config.N_WORKERS}")
    print(f"Pending       : {pending}")
    print(f"Running       : {running}")
    print(f"Done          : {done}")


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


def _stage_label(state: dict) -> str:
    """Build a human-readable label for the current tier / sub-batch stage."""
    scheme = state.get("current_scheme")
    tier = state.get("current_tier")
    stage = state.get("stage")
    seq_step = state.get("sequential_step")
    seq_total = state.get("sequential_step_total")

    if scheme is None or tier is None:
        return "loading"

    ladder = config.TIER_SETTINGS_BY_SCHEME.get(scheme)
    if ladder and 0 <= tier < len(ladder):
        s = ladder[tier]
        budget = f"evals={s['ga_n_evals']} pop={s['ga_pop_size']}"
    else:
        budget = "?"

    stage_str = stage or "?"
    sub = f" sub {seq_step}/{seq_total}" if seq_step is not None else ""
    return f"t{tier}/{config.N_TIERS} {scheme}{sub} {stage_str} ({budget})"


def submitted() -> None:
    """Print one progress line per running worker from its heartbeat state file.

    Each line gives the worker id, its ``(rep, model, scheme)`` cell, the
    current tier and stage, tiers done out of ``config.N_TIERS``, elapsed time
    and host.
    """
    _ensure_dirs()
    running_dir = QUEUE_DIR / "running"
    now = time.time()

    locks = sorted(
        running_dir.glob("job_*.lock"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not locks:
        print("No running workers.")
        return

    for lock in locks:
        idx = int(lock.stem.split("_")[1])
        rep_idx, model_idx, scheme_idx = _decode(idx)
        model_name = config.MODELS[model_idx]["name"]
        scheme_name = config.BATCH_SCHEMES[scheme_idx]["name"]
        state = load_run_state_for_reporting(
            get_run_state_path(QUEUE_DIR, idx), idx, OPT_TOTAL,
        )
        elapsed_s = (
            now - float(state["started_at_epoch"])
            if state.get("started_at_epoch") is not None
            else float(state.get("wall_time_so_far_s", 0.0))
        )
        host = state.get("hostname") or "?"
        label = _stage_label(state)
        done = state.get("opt_done", 0)
        print(
            f"{idx:>3} | rep={rep_idx:>2} {model_name:>6} {scheme_name:>12} | "
            f"{label} | done={done}/{OPT_TOTAL} | elapsed={_fmt_dur(elapsed_s):>6} | {host}"
        )


def main() -> None:
    """CLI entrypoint: ``init`` | ``worker [--single]`` | ``status`` | ``submitted``."""
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
