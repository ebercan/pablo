"""Plot per-iteration wall times of the run_1 operating modes.

Reads ``result_*.json`` from ``results_averaged/`` and
``results_layer_averaged/`` and draws three stacked panels (acquisition
optimization, model fitting, function evaluation), each showing the mean wall
time per BO batch iteration with a 95% confidence band (1.96 × standard
error) per mode. Curves are truncated to the shortest run. The figure is saved
as ``plot_wall_times.png`` next to this script.

Usage:
    python -m pablo.mechthild_runs.run_1_scope_extension.plots.plot_wall_times
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

OPERATING_MODES = ["averaged", "layer_averaged"]
WALL_TIME_TYPES = [
    "wall_times_acqf_opt",
    "wall_times_model_fit",
    "wall_times_function_eval",
]
WALL_TIME_LABELS = {
    "wall_times_acqf_opt": "Acquisition Function Optimization",
    "wall_times_model_fit": "Model Fitting",
    "wall_times_function_eval": "Function Evaluation",
}


def _result_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_rows(results_dir: Path, operating_mode: str) -> list[dict]:
    """Load the wall-time series of every result file in one mode's folder.

    Args:
        results_dir: Directory containing ``result_*.json`` files.
        operating_mode: Mode label stored on every returned row.

    Returns:
        One dict per result file, in result-index order, with keys
        ``operating_mode``, ``wall_times_acqf_opt``, ``wall_times_model_fit``
        and ``wall_times_function_eval`` (lists of seconds per iteration).
    """
    rows = []
    for result_path in sorted(results_dir.glob("result_*.json"), key=_result_index):
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        result = payload["result"]
        rows.append(
            {
                "operating_mode": operating_mode,
                "wall_times_acqf_opt": [
                    float(x) for x in result.get("wall_times_acqf_opt", [])
                ],
                "wall_times_model_fit": [
                    float(x) for x in result.get("wall_times_model_fit", [])
                ],
                "wall_times_function_eval": [
                    float(x) for x in result.get("wall_times_function_eval", [])
                ],
            }
        )
    return rows


def build_wall_time_curves(rows, curve_len, wall_time_key):
    """Aggregate one wall-time series across runs, per operating mode.

    Args:
        rows: Rows returned by ``load_rows``.
        curve_len: Number of iterations to keep from each run.
        wall_time_key: Row key of the series, e.g. ``"wall_times_acqf_opt"``.

    Returns:
        Dict mapping each mode with data to ``(mean_curve, margin_curve,
        n_runs)``, where both curves have length ``curve_len`` and
        ``margin_curve`` is 1.96 × the standard error (zero for a single run).
    """
    grouped = defaultdict(list)
    for row in rows:
        operating_mode = row["operating_mode"]
        wall_times = row[wall_time_key][:curve_len]
        if len(wall_times) > 0:
            grouped[operating_mode].append(wall_times)

    stats = {}
    for operating_mode in OPERATING_MODES:
        curves = np.array(grouped[operating_mode])
        if len(curves) > 0:
            mean_curve = np.mean(curves, axis=0)
            std_curve = (
                np.std(curves, axis=0, ddof=1)
                if len(curves) > 1
                else np.zeros_like(mean_curve)
            )
            se_curve = std_curve / np.sqrt(len(curves))
            margin_curve = 1.96 * se_curve
            stats[operating_mode] = (mean_curve, margin_curve, len(curves))
    return stats


def plot_all_wall_times(all_stats, output_path, curve_len):
    """Draw the three-panel wall-time figure and save it.

    Args:
        all_stats: Dict mapping each key of ``WALL_TIME_TYPES`` to the output
            of ``build_wall_time_curves``.
        output_path: Destination path of the PNG.
        curve_len: Number of BO iterations on the x-axis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    x_values = list(range(1, curve_len + 1))
    colors = {"averaged": "blue", "layer_averaged": "green"}

    for idx, wall_time_type in enumerate(WALL_TIME_TYPES):
        ax = axes[idx]
        stats = all_stats[wall_time_type]

        for operating_mode in OPERATING_MODES:
            if operating_mode not in stats:
                continue
            mean_curve, margin_curve, n_runs = stats[operating_mode]

            upper = mean_curve + margin_curve
            lower = mean_curve - margin_curve

            label = f"{operating_mode} (n={n_runs})"
            color = colors[operating_mode]

            ax.plot(x_values, mean_curve, label=label, color=color, linewidth=1.5)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.2)

        ax.set_title(WALL_TIME_LABELS[wall_time_type])
        ax.set_ylabel("Wall Time (s)")
        ax.set_xlim(1, curve_len)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("BO Batch Iteration")
    axes[-1].set_xticks(range(1, curve_len + 1, 10))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_results_dir(plots_dir: Path, mode_dirname: str) -> Path:
    """Locate a mode's results folder next to the plots folder.

    Args:
        plots_dir: Directory of this script.
        mode_dirname: Results folder name, e.g. ``"results_averaged"``.

    Returns:
        ``plots_dir.parent / mode_dirname`` if it exists, otherwise
        ``plots_dir.parent / "results" / mode_dirname`` (which may not exist).
    """
    candidate = plots_dir.parent / mode_dirname
    if candidate.exists():
        return candidate
    return plots_dir.parent / "results" / mode_dirname


def main():
    plots_dir = Path(__file__).resolve().parent

    rows = []
    for operating_mode in OPERATING_MODES:
        results_dir = resolve_results_dir(plots_dir, f"results_{operating_mode}")
        if not results_dir.exists():
            print(f"No results directory for {operating_mode}: {results_dir}")
            continue
        rows.extend(load_rows(results_dir, operating_mode))

    curve_lengths = {
        len(r["wall_times_acqf_opt"]) for r in rows if len(r["wall_times_acqf_opt"]) > 0
    }
    if not curve_lengths:
        print("No wall time data found.")
        return
    curve_len = min(curve_lengths)

    all_stats = {}
    for wall_time_type in WALL_TIME_TYPES:
        all_stats[wall_time_type] = build_wall_time_curves(
            rows, curve_len, wall_time_type
        )

    output_path = plots_dir / "plot_wall_times.png"
    plot_all_wall_times(all_stats, output_path, curve_len)
    print(f"Saved wall times plot to {output_path}")


if __name__ == "__main__":
    main()
