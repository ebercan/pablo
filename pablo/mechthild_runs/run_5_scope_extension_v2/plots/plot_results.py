"""Plot run_5 results: relative-improvement curves, regret, expander usage, wall times.

Run from the repo root:

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.plots.plot_results

Only (benchmark, seed) pairs present in every operating mode are used, so all
curves are paired. Undefined values (a relative improvement with a zero
denominator, a non-positive regret on the log axis) are NaN and left out of the
means. Outputs are written next to this script.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    ALL_OPERATING_MODES,
    BENCHMARK_CLASS_MAP,
    results_dir,
)

BENCHMARK_NAMES = list(BENCHMARK_CLASS_MAP)
COLORS = {
    "random": "gray",
    "myopic": "red",
    "only_terminal": "purple",
    "averaged": "blue",
    "layer_averaged": "green",
}


def load_rows() -> list[dict]:
    """Load every result JSON and keep only seed tuples complete across all modes.

    Returns:
        One dict per result file whose ``run_key`` (benchmark, seed) has a
        result in every operating mode. Keys: ``run_key``, ``benchmark``,
        ``mode``, ``best_fs`` (best noise-free yield measured so far per
        iteration), ``global_max`` (the best attainable yield, 1),
        ``n_using_released``, ``wall_expander``, ``wall_acq``, ``wall_fit``.
    """
    rows = []
    for mode in ALL_OPERATING_MODES:
        mode_dir = results_dir(mode)
        if not mode_dir.exists():
            continue
        for path in sorted(mode_dir.glob("result_*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            comb, res = payload["combination"], payload["result"]
            rows.append(
                {
                    "run_key": (comb["benchmark"], int(comb["seed"])),
                    "benchmark": comb["benchmark"],
                    "mode": mode,
                    "best_fs": [float(v) for v in res["best_fs"]],
                    "global_max": res["global_max"],
                    "n_using_released": res.get("n_candidates_using_released", []),
                    "wall_expander": res.get("wall_times_expander", []),
                    "wall_acq": res.get("wall_times_acqf_opt", []),
                    "wall_fit": res.get("wall_times_model_fit", []),
                }
            )
    modes_by_key = defaultdict(set)
    for row in rows:
        modes_by_key[row["run_key"]].add(row["mode"])
    complete = {k for k, modes in modes_by_key.items() if len(modes) == len(ALL_OPERATING_MODES)}
    return [row for row in rows if row["run_key"] in complete]


def _curve_stats(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray, int]:
    """NaN-aware mean and 95% half-width per iteration; the half-width is NaN for one curve."""
    length = min(len(c) for c in curves)
    arr = np.array([c[:length] for c in curves])
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    margin = 1.96 * np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if len(arr) > 1 else np.full_like(mean, np.nan)
    return mean, margin, len(arr)


def plot_relative_improvement(rows: list[dict], output: Path) -> None:
    """Plot mean relative improvement per benchmark and mode with a 95% CI band.

    Relative improvement is (best so far − first value) / (per-seed best across
    modes − first value); a curve whose denominator is below 1e-12 is NaN.

    Args:
        rows: Paired result rows from ``load_rows``.
        output: Destination PNG path.
    """
    best_by_seed: dict[tuple, float] = {}
    for row in rows:
        best_by_seed[row["run_key"]] = max(best_by_seed.get(row["run_key"], -np.inf), max(row["best_fs"]))
    grouped = defaultdict(list)
    for row in rows:
        start, top = row["best_fs"][0], best_by_seed[row["run_key"]]
        denom = top - start
        curve = [(v - start) / denom if abs(denom) > 1e-12 else np.nan for v in row["best_fs"]]
        grouped[(row["benchmark"], row["mode"])].append(curve)
    fig, axes = plt.subplots(len(BENCHMARK_NAMES), 1, figsize=(10, 5 * len(BENCHMARK_NAMES)), sharex=True)
    for ax, benchmark in zip(np.atleast_1d(axes), BENCHMARK_NAMES):
        for mode in ALL_OPERATING_MODES:
            if (benchmark, mode) not in grouped:
                continue
            mean, margin, n = _curve_stats(grouped[(benchmark, mode)])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=COLORS[mode], label=f"{mode} (n={n})")
            ax.fill_between(x, mean - margin, mean + margin, color=COLORS[mode], alpha=0.15)
        ax.set_title(f"{benchmark}: relative improvement")
        ax.set_ylabel("relative improvement")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    np.atleast_1d(axes)[-1].set_xlabel("BO iteration")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regret(rows: list[dict], output: Path) -> None:
    """Plot mean simple regret (best attainable − best so far, yield scale) per benchmark on a log axis.

    A non-positive regret (a point above the estimated attainable maximum) is NaN.

    Args:
        rows: Paired result rows from ``load_rows``.
        output: Destination PNG path.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["mode"])].append(
            [row["global_max"] - v if row["global_max"] - v > 0 else np.nan for v in row["best_fs"]]
        )
    fig, axes = plt.subplots(len(BENCHMARK_NAMES), 1, figsize=(10, 5 * len(BENCHMARK_NAMES)), sharex=True)
    for ax, benchmark in zip(np.atleast_1d(axes), BENCHMARK_NAMES):
        for mode in ALL_OPERATING_MODES:
            if (benchmark, mode) not in grouped:
                continue
            mean, margin, n = _curve_stats(grouped[(benchmark, mode)])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=COLORS[mode], label=f"{mode} (n={n})")
            ax.fill_between(x, mean - margin, mean + margin, color=COLORS[mode], alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(f"{benchmark}: simple regret")
        ax.set_ylabel("1 - best noise-free yield")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("BO iteration")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_expander_usage_and_walltime(rows: list[dict], output: Path) -> None:
    """Bar-plot the share of iterations using a just-released catalyst and expander wall time.

    Left panel: fraction of iterations whose batch uses a just-released
    catalyst. Right panel: mean expander wall time per iteration (log axis).

    Args:
        rows: Paired result rows from ``load_rows``.
        output: Destination PNG path.
    """
    usage = defaultdict(list)
    walls = defaultdict(list)
    for row in rows:
        key = (row["benchmark"], row["mode"])
        if row["n_using_released"]:
            usage[key].append(np.mean([v > 0 for v in row["n_using_released"]]))
        if row["wall_expander"]:
            walls[key].append(np.mean(row["wall_expander"]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    width = 0.8 / len(ALL_OPERATING_MODES)
    x = np.arange(len(BENCHMARK_NAMES))
    for i, mode in enumerate(ALL_OPERATING_MODES):
        u = [np.mean(usage[(b, mode)]) if usage[(b, mode)] else np.nan for b in BENCHMARK_NAMES]
        w = [np.mean(walls[(b, mode)]) if walls[(b, mode)] else np.nan for b in BENCHMARK_NAMES]
        axes[0].bar(x + i * width, u, width, color=COLORS[mode], label=mode)
        axes[1].bar(x + i * width, w, width, color=COLORS[mode], label=mode)
    for ax, title, ylabel in (
        (axes[0], "iterations using a just-released catalyst", "fraction of iterations"),
        (axes[1], "expander wall time per iteration", "seconds"),
    ):
        ax.set_xticks(x + width * (len(ALL_OPERATING_MODES) - 1) / 2)
        ax.set_xticklabels(BENCHMARK_NAMES)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_yscale("log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load paired results and write the three summary PNGs next to this script."""
    rows = load_rows()
    out_dir = Path(__file__).resolve().parent
    print(f"{len(rows)} paired result rows")
    if not rows:
        return
    plot_relative_improvement(rows, out_dir / "relative_improvement.png")
    plot_regret(rows, out_dir / "simple_regret.png")
    plot_expander_usage_and_walltime(rows, out_dir / "expander_usage_walltime.png")
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    main()
