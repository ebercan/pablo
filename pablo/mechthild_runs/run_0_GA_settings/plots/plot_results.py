"""Relative-improvement plots comparing the run_0 GA settings, saved as PNGs.

Reads ``run_0_GA_settings/results/result_*.json`` and writes two images into
``plots/``: ``relative_improvement_6_benchmarks.png`` (mean relative improvement
with 95% intervals per GA setting, one panel per benchmark) and
``relative_improvement_summary_table.png`` (per-setting summary table).

Usage:
    python -m pablo.mechthild_runs.run_0_GA_settings.plots.plot_results
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from pablo.mechthild_runs.run_0_GA_settings.config import (
    BENCHMARK_CLASS_MAP,
    SEED_LIST_1,
    SEED_LIST_2,
)

BENCHMARK_NAMES = list(BENCHMARK_CLASS_MAP)
from pablo.mechthild_runs.run_0_GA_settings.generate_combinations import generate_ga_settings


def _result_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_rows(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    result_paths = sorted(results_dir.glob("result_*.json"), key=_result_index)

    for result_path in result_paths:
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        combination = payload["combination"]
        result = payload["result"]
        best_fs = [float(value) for value in result["best_fs"]]
        if len(best_fs) == 0:
            raise ValueError(f"Empty best_fs in {result_path}")

        rows.append(
            {
                "benchmark": str(combination["benchmark"]),
                "seed_discrete": int(combination["seed_discrete"]),
                "seed_starting": int(combination["seed_starting"]),
                "setting_idx": int(combination["setting_idx"]),
                "best_fs": best_fs,
            }
        )

    return rows


def _validate_rows(rows: list[dict[str, object]]) -> int:
    n_settings = len(generate_ga_settings())
    expected_total = (
        len(BENCHMARK_NAMES) * len(SEED_LIST_1) * len(SEED_LIST_2) * n_settings
    )
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} runs, found {len(rows)}")

    cell_to_settings: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for row in rows:
        key = (
            str(row["benchmark"]),
            int(row["seed_discrete"]),
            int(row["seed_starting"]),
        )
        cell_to_settings[key].add(int(row["setting_idx"]))

    expected_cells = len(BENCHMARK_NAMES) * len(SEED_LIST_1) * len(SEED_LIST_2)
    if len(cell_to_settings) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} benchmark/seed cells, found {len(cell_to_settings)}"
        )

    for key, setting_ids in cell_to_settings.items():
        if len(setting_ids) != n_settings:
            raise ValueError(
                f"Cell {key} has {len(setting_ids)} settings, expected {n_settings}"
            )

    curve_lengths = {len(list(row["best_fs"])) for row in rows}
    if len(curve_lengths) != 1:
        raise ValueError(f"Inconsistent best_fs lengths: {sorted(curve_lengths)}")

    curve_len = curve_lengths.pop()
    if curve_len < 100:
        raise ValueError(f"Expected at least 100 iterations, found {curve_len}")

    return curve_len


def _setting_meta_by_idx() -> dict[int, dict[str, int]]:
    return {
        int(item["setting_idx"]): {
            "ga_ratio": int(item["ga_ratio"]),
            "pop_size": int(item["pop_size"]),
            "n_evals_ga": int(item["n_evals_ga"]),
        }
        for item in generate_ga_settings()
    }


def _best_for_discrete_choice(
    rows: list[dict[str, object]],
) -> dict[tuple[str, int], float]:
    best_by_discrete: dict[tuple[str, int], float] = {}

    for benchmark in BENCHMARK_NAMES:
        for seed_discrete in SEED_LIST_1:
            key = (benchmark, seed_discrete)
            values: list[float] = []
            for row in rows:
                if (
                    str(row["benchmark"]) == benchmark
                    and int(row["seed_discrete"]) == seed_discrete
                ):
                    values.extend(list(row["best_fs"]))

            if len(values) == 0:
                raise ValueError(f"No values for benchmark/discrete pair {key}")
            best_by_discrete[key] = max(values)

    return best_by_discrete


def _relative_improvement_curve(
    best_fs: list[float],
    best_f_in_starting_set: float,
    best_f_for_discrete_choices: float,
) -> list[float]:
    denominator = best_f_for_discrete_choices - best_f_in_starting_set
    if abs(denominator) < 1e-12:
        return [0.0 for _ in best_fs]

    return [(value - best_f_in_starting_set) / denominator for value in best_fs]


def _curve_auc(curve: np.ndarray | list[float]) -> float:
    # Trapezoidal rule: half of the first and last points, others unchanged
    arr = np.array(curve)
    return float(np.sum(arr) - 0.5 * arr[0] - 0.5 * arr[-1])


def _build_mean_ri_curves_and_ci(
    rows: list[dict[str, object]],
    curve_len: int,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    best_by_discrete = _best_for_discrete_choice(rows)
    grouped_curves: dict[tuple[str, int], list[list[float]]] = defaultdict(list)

    for row in rows:
        benchmark = str(row["benchmark"])
        seed_discrete = int(row["seed_discrete"])
        setting_idx = int(row["setting_idx"])
        best_fs = list(row["best_fs"])[:curve_len]

        best_f_in_starting_set = float(best_fs[0])
        # Sometimes best_fs[0] can act weird, but it's identical to old code
        best_f_for_discrete_choices = best_by_discrete[(benchmark, seed_discrete)]
        ri_curve = _relative_improvement_curve(
            best_fs=best_fs,
            best_f_in_starting_set=best_f_in_starting_set,
            best_f_for_discrete_choices=best_f_for_discrete_choices,
        )

        grouped_curves[(benchmark, setting_idx)].append(ri_curve)

    stats: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for benchmark in BENCHMARK_NAMES:
        for setting_idx in range(len(generate_ga_settings())):
            key = (benchmark, setting_idx)
            curves = np.array(grouped_curves[key])
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0, ddof=1)
            # 16 runs per setting -> 1.96 * SE
            se_curve = std_curve / np.sqrt(len(curves))
            margin_curve = 1.96 * se_curve
            stats[key] = (mean_curve, margin_curve)

    return stats


def _plot_benchmark_panels(
    stats: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    curve_len: int,
) -> None:
    setting_meta = _setting_meta_by_idx()
    x_values = list(range(1, curve_len + 1))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#a6cee3",
        "#1f78b4",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(22, 16), sharex=True, sharey=True)
    axes_flat = list(axes.flatten())

    for benchmark_idx, benchmark in enumerate(BENCHMARK_NAMES):
        axis = axes_flat[benchmark_idx]

        lines_data = []
        for setting_idx in range(len(generate_ga_settings())):
            mean_curve, margin_curve = stats[(benchmark, setting_idx)]
            final_ri = mean_curve[-1]
            auc = _curve_auc(mean_curve)

            upper_bound = mean_curve + margin_curve
            lower_bound = mean_curve - margin_curve

            # Trapezoidal AUC for CI, then get average by dividing by 100
            diff = upper_bound - lower_bound
            ci_auc = _curve_auc(diff)
            avg_ci = ci_auc / 100.0

            meta = setting_meta[setting_idx]
            label = (
                f"#{setting_idx + 1} ga={meta['ga_ratio']} "
                f"pop={meta['pop_size']} neval={meta['n_evals_ga']} | "
                f"RI*={final_ri:.4f} AUC={auc:.1f} <|CI|>={avg_ci:.3f}"
            )
            lines_data.append(
                (final_ri, mean_curve, upper_bound, lower_bound, label, setting_idx)
            )

        lines_data.sort(key=lambda x: x[0], reverse=True)

        for _, mean_curve, upper_bound, lower_bound, label, setting_idx in lines_data:
            color = colors[setting_idx % len(colors)]

            # Plot mean curve
            axis.plot(x_values, mean_curve, linewidth=2, color=color, label=label)

            # Plot confidence intervals
            axis.plot(
                x_values,
                upper_bound,
                linewidth=1,
                color=color,
                linestyle="--",
                alpha=0.7,
            )
            axis.plot(
                x_values,
                lower_bound,
                linewidth=1,
                color=color,
                linestyle="--",
                alpha=0.7,
            )

        axis.set_title(f"{benchmark}: Relative Improvement", fontsize=14)
        axis.set_xlim(1, curve_len)
        axis.set_ylim(0.0, 1.05)
        if benchmark_idx >= 4:
            axis.set_xlabel("Iterations", fontsize=12)
        if benchmark_idx % 2 == 0:
            axis.set_ylabel("Relative Improvement", fontsize=12)
        axis.grid(True, alpha=0.25)
        # Add legend
        axis.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_table(
    stats: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    num_rows = len(generate_ga_settings())
    num_cols = len(BENCHMARK_NAMES)
    setting_meta = _setting_meta_by_idx()

    fig, ax = plt.subplots(figsize=(num_cols * 3.0, num_rows * 1.2))
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)

    # Hide spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # Draw grid
    for c in range(num_cols + 1):
        ax.axvline(c, color="black", lw=1.5)
    for r in range(num_rows + 1):
        ax.axhline(r, color="black", lw=1.5)

    for c, benchmark in enumerate(BENCHMARK_NAMES):
        for r in range(num_rows):
            # To have Setting #1 at the top, if r=0 is bottom, then setting_idx mapping:
            # num_rows is 12. If r=11 (top row), setting_idx = 0.
            setting_idx = num_rows - 1 - r
            mean_curve, _ = stats[(benchmark, setting_idx)]
            ri = mean_curve[-1]
            auc = _curve_auc(mean_curve)

            # Diagonal line splitting the cell from top-left (c, r+1) to bottom-right (c+1, r)
            ax.plot([c, c + 1], [r + 1, r], color="gray", lw=1.0)

            # Top-right text
            ax.text(
                c + 0.65,
                r + 0.65,
                f"{ri:.4f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="darkblue",
            )
            # Bottom-left text
            ax.text(
                c + 0.35,
                r + 0.35,
                f"{auc:.1f}",
                ha="center",
                va="center",
                fontsize=12,
                color="darkred",
            )

    # Set labels
    ax.set_xticks(np.arange(num_cols) + 0.5)
    # Using small names or splitting if needed, but original BENCHMARK_NAMES work
    ax.set_xticklabels(
        BENCHMARK_NAMES, rotation=0, ha="center", fontsize=12, fontweight="bold"
    )
    ax.xaxis.tick_top()  # Put column labels at the top

    ylabels = []
    for r in range(num_rows):
        s_idx = num_rows - 1 - r
        m = setting_meta[s_idx]
        ylabels.append(f"#{s_idx+1} [ga={m['ga_ratio']} pop={m['pop_size']}]")

    ax.set_yticks(np.arange(num_rows) + 0.5)
    ax.set_yticklabels(ylabels, fontsize=12, fontweight="bold")

    ax.set_title(
        "Performance Summary Grid (Top-Right: Final RI*  |  Bottom-Left: RI-AUC)",
        pad=40,
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load run_0 results, compute relative-improvement statistics and save both PNGs."""
    plots_dir = Path(__file__).resolve().parent
    run_1_dir = plots_dir.parent
    results_dir = run_1_dir / "results"

    rows = _load_rows(results_dir)
    curve_len = _validate_rows(rows)
    stats = _build_mean_ri_curves_and_ci(rows, curve_len=curve_len)

    output_path = plots_dir / "relative_improvement_6_benchmarks.png"
    _plot_benchmark_panels(
        stats=stats,
        output_path=output_path,
        curve_len=curve_len,
    )

    table_path = plots_dir / "relative_improvement_summary_table.png"
    _plot_summary_table(
        stats=stats,
        output_path=table_path,
    )


if __name__ == "__main__":
    main()
