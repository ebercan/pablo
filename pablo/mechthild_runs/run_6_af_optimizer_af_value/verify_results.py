"""Sanity gate over run_6 result files.

Checks, per result: the unit matches its index, the batch has q rows, the temperature
dim splits into the configured partition, the time dim equals the pinned value, every
row decodes to available agents (``decode_batch`` raises otherwise, and checks the zero
base slice of non-salt catalysts), ``score`` and ``own_acq_value`` are finite, and the
wall time is within ``--time-tolerance`` of the tier target (skipped for smoke results).
Exits non-zero on any failure, and when results are missing unless ``--partial``.

Usage:
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.verify_results [--results-dir DIR] [--partial]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

from pablo.main import DTYPE
from pablo.mechthild_runs.run_6_af_optimizer_af_value import config
from pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant import (
    RESULTS_DIR,
    _load_context,
    decode_batch,
    decode_worker,
)


def check_result(ctx: dict, res: dict, idx: int, time_tolerance: float) -> list[str]:
    """Failure messages of one result (empty when it passes)."""
    errors = []
    if (res["optimizer"], res["tier"], res["rep"]) != decode_worker(idx):
        errors.append(f"unit {(res['optimizer'], res['tier'], res['rep'])} != decode {decode_worker(idx)}")
    cands = torch.tensor(res["candidates"], dtype=DTYPE)
    if cands.shape[0] != config.Q_BATCH:
        errors.append(f"{cands.shape[0]} rows, expected {config.Q_BATCH}")
    counts = sorted(Counter(round(v, 9) for v in cands[:, config.TEMPERATURE_DIM].tolist()).values(), reverse=True)
    if tuple(counts) != tuple(sorted(config.TEMPERATURE_PARTITION, reverse=True)):
        errors.append(f"temperature groups {counts}, expected {config.TEMPERATURE_PARTITION}")
    pinned = ctx["fixed_dims"][config.TIME_DIM]
    if not torch.allclose(cands[:, config.TIME_DIM], torch.full((cands.shape[0],), pinned, dtype=DTYPE), atol=1e-9):
        errors.append(f"time dim not pinned at {pinned}")
    if bool((cands < ctx["bounds"][0] - 1e-9).any() or (cands > ctx["bounds"][1] + 1e-9).any()):
        errors.append("candidates outside bounds")
    decode_batch(ctx, cands)
    for key in ("score", "own_acq_value"):
        if not math.isfinite(res[key]):
            errors.append(f"{key} is {res[key]}")
    if not res["smoke_settings"]:
        target = config.TIER_TARGET_MINUTES[res["tier"]] * 60.0
        if abs(res["wall_time_s"] - target) > time_tolerance * target:
            errors.append(f"wall time {res['wall_time_s']:.0f} s off target {target:.0f} s by > {time_tolerance:.0%}")
    return errors


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--time-tolerance", type=float, default=0.3)
    args = parser.parse_args()
    ctx = _load_context()
    present = {int(p.stem.split("_")[1]): p for p in args.results_dir.glob("result_*.json")}
    failed = 0
    for idx in sorted(present):
        errors = check_result(ctx, json.loads(present[idx].read_text()), idx, args.time_tolerance)
        if errors:
            failed += 1
            print(f"FAIL result_{idx}: " + "; ".join(errors))
    missing = sorted(set(range(config.N_WORKERS)) - set(present))
    print(f"{len(present)} results checked, {failed} failed, {len(missing)} missing {missing if missing else ''}")
    if failed or (missing and not args.partial):
        sys.exit(1)


if __name__ == "__main__":
    main()
