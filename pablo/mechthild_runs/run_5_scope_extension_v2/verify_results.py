"""Sanity gate over finished run_5 result files.

Usage::

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.verify_results \\
        [--operating_mode MODE ...] [--max_files N]

Checks every ``result_*.json`` of the selected modes (all modes by default;
``--max_files`` caps the files per mode, in sorted order) and exits with status
1 if any file fails. Per file:

1. ``best_fs`` has ``N_BO_BATCHES`` entries.
2. ``best_fs`` is non-decreasing.
3. Every iteration with an expansion records all five picks scored under all
   four criteria, and the ordered catalyst is the operating mode's pick.
4. Every batch has ``Q_BATCH_SIZE`` points whose distinct temperature levels
   come from the (2, 2, 2) groups (two groups may share a level, e.g. both at
   a bound), and time sits at ``FIXED_DIM_VALUE`` in every point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    ALL_OPERATING_MODES,
    DEFAULT_BATCH_CONSTRAINTS,
    EXPANSION_CRITERIA,
    FIXED_DIM_INDEX,
    FIXED_DIM_VALUE,
    N_BO_BATCHES,
    Q_BATCH_SIZE,
    RANDOM_OPERATING_MODE,
    TEMPERATURE_DIM,
    results_dir,
)


def _temperature_level_counts(values: list[float], tol: float = 1e-4) -> list[int]:
    """Sorted multiplicities of the distinct values (within ``tol``)."""
    levels: list[float] = []
    counts: list[int] = []
    for value in sorted(values):
        if levels and abs(value - levels[-1]) < tol:
            counts[-1] += 1
        else:
            levels.append(value)
            counts.append(1)
    return sorted(counts)


def _is_coarsening(counts: list[int], partition: list[int]) -> bool:
    """Whether the partition's parts can be grouped so the group sums equal ``counts``."""
    if not counts:
        return not partition
    first, rest = counts[0], counts[1:]
    for mask in range(1, 1 << len(partition)):
        picked = [p for i, p in enumerate(partition) if mask >> i & 1]
        if sum(picked) == first:
            left = [p for i, p in enumerate(partition) if not mask >> i & 1]
            if _is_coarsening(rest, left):
                return True
    return False


def verify_result_file(path: Path, mode: str) -> bool:
    """Run every check on one result file and print one line per check.

    Args:
        path: A ``result_<idx>.json``.
        mode: Operating mode of the file's results folder.

    Returns:
        True if every check passes.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    res = payload["result"]
    best_fs = [float(v) for v in res["best_fs"]]
    batches = res["candidates_list"]
    partition = sorted(DEFAULT_BATCH_CONSTRAINTS[TEMPERATURE_DIM])
    pick_names = {RANDOM_OPERATING_MODE, *EXPANSION_CRITERIA}

    checks = {
        f"{N_BO_BATCHES} iterations (got {len(best_fs)})": len(best_fs) == N_BO_BATCHES,
        f"best_fs non-decreasing ({best_fs[0]:.4f} -> {best_fs[-1]:.4f})": all(
            a <= b + 1e-9 for a, b in zip(best_fs, best_fs[1:])
        ),
        "picks cross-scored, mode's pick ordered": all(
            selected is None if picks is None else (
                set(picks) == pick_names == set(scores)
                and all(set(row) == set(EXPANSION_CRITERIA) for row in scores.values())
                and picks[mode]["catalyst_index"] == selected
            )
            for picks, scores, selected in zip(
                res["expansion_picks"], res["expansion_scores"], res["selected_source_a_indices"]
            )
        ),
        f"temperature partition {tuple(partition)} in all {len(batches)} batches": all(
            len(batch) == Q_BATCH_SIZE
            and _is_coarsening(_temperature_level_counts([p[TEMPERATURE_DIM] for p in batch]), partition)
            for batch in batches
        ),
        f"time pinned at {FIXED_DIM_VALUE} h": all(
            abs(p[FIXED_DIM_INDEX] - FIXED_DIM_VALUE) < 1e-9 for batch in batches for p in batch
        ),
    }
    passed = all(checks.values())
    print(f"{mode}/{path.name} ({payload['combination']['label']}): {'PASSED' if passed else 'FAILED'}")
    for name, ok in checks.items():
        if not ok:
            print(f"  failed: {name}")
    return passed


def main() -> None:
    """Check the selected result files; exit with status 1 if any fails."""
    parser = argparse.ArgumentParser(description="Sanity gate over run_5 result files")
    parser.add_argument("--operating_mode", nargs="+", choices=ALL_OPERATING_MODES, default=list(ALL_OPERATING_MODES))
    parser.add_argument("--max_files", type=int, default=None, help="files per mode (default: all)")
    args = parser.parse_args()

    n_files = n_failed = 0
    for mode in args.operating_mode:
        files = sorted(results_dir(mode).glob("result_*.json"))[: args.max_files]
        for path in files:
            n_files += 1
            n_failed += not verify_result_file(path, mode)
    print(f"{n_files - n_failed} of {n_files} result files passed")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
