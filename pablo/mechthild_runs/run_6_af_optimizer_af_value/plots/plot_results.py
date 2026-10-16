"""Common-scorer qLogNEI against wall time for both optimizers.

Reads every ``result_*.json``, plots one point per run and the per-tier median of each
optimizer, writes ``af_vs_walltime.png`` next to this script and prints the per-tier
table (median score and wall time per optimizer, median score difference).

Usage:
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.plots.plot_results [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant import RESULTS_DIR

COLORS = {"ga": "#4c6a92", "pool_greedy": "#b5653a"}
LABELS = {"ga": "GA + L-BFGS refine", "pool_greedy": "pool-greedy"}


def load_results(results_dir: Path) -> pd.DataFrame:
    """One row per result file: optimizer, tier, rep, score, own value, wall time."""
    rows = []
    for p in sorted(results_dir.glob("result_*.json")):
        r = json.loads(p.read_text())
        rows.append(
            {k: r[k] for k in ("worker_id", "optimizer", "tier", "rep", "score", "own_acq_value", "wall_time_s")}
        )
    if not rows:
        raise FileNotFoundError(f"no result_*.json in {results_dir}")
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "af_vs_walltime.png")
    args = parser.parse_args()
    df = load_results(args.results_dir)
    df["wall_min"] = df.wall_time_s / 60.0

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for opt, g in df.groupby("optimizer"):
        ax.scatter(g.wall_min, g.score, s=18, alpha=0.5, color=COLORS[opt])
        med = g.groupby("tier")[["wall_min", "score"]].median()
        ax.plot(med.wall_min, med.score, marker="o", color=COLORS[opt], label=LABELS[opt])
    ax.set_xscale("log")
    ax.set_xlabel("wall time per run (min)")
    ax.set_ylabel("qLogNEI, common scorer")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"figure -> {args.out}")

    table = df.pivot_table(index="tier", columns="optimizer", values=["score", "wall_min"], aggfunc="median")
    if {"ga", "pool_greedy"} <= set(df.optimizer):
        table[("delta", "pg-ga")] = table[("score", "pool_greedy")] - table[("score", "ga")]
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
