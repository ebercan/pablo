"""Compare the averaged and layer_averaged operating modes of run_1.

Reads ``results_averaged/`` and ``results_layer_averaged/`` (next to ``plots/`` or
under ``results/``) and writes two PNGs into ``plots/``:
``plot_averaged_vs_layer_averaged.png`` (relative improvement per benchmark) and
``scope_extension_accuracy_averaged_vs_layer.png`` (how often newly released
block-A choices are picked).

Usage:
    python -m pablo.mechthild_runs.run_1_scope_extension.plots.plot_averaged_vs_layer_averaged
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

from pablo.mechthild_runs.run_1_scope_extension.config import (
    BENCHMARK_CLASS_MAP,
    DEPENDENT_BLOCK_SIZES,
    N_CONTINUOUS_DIMS,
    TOTAL_DIM,
)
from pablo.mechthild_runs.run_1_scope_extension.run_ga_experiment import (
    generate_dependent_choice_sources,
)

BENCHMARK_NAMES = ["LunacekBiRastrigin", "Gallagher101", "Rastrigin"]
OPERATING_MODES = ["averaged", "layer_averaged"]


def _result_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_rows(results_dir: Path, operating_mode: str) -> list[dict]:
    """Load the result JSONs of one operating mode into flat row dicts.

    Args:
        results_dir: Directory holding ``result_<i>.json`` files, read in order of ``i``.
        operating_mode: Mode label used when a file's ``combination`` has no
            ``operating_mode`` entry.

    Returns:
        One dict per file with keys ``benchmark``, ``strategy``, ``operating_mode``,
        ``seed_discrete``, ``best_fs`` (best observed value per iteration),
        ``candidates_list`` and ``released_source_a_by_iteration`` (empty lists
        when absent from the file).
    """
    rows = []
    for result_path in sorted(results_dir.glob("result_*.json"), key=_result_index):
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        combination = payload["combination"]
        result = payload["result"]
        rows.append(
            {
                "benchmark": str(combination["benchmark"]),
                "strategy": str(combination["strategy"]),
                "operating_mode": str(
                    combination.get("operating_mode", operating_mode)
                ),
                "seed_discrete": int(combination["seed_discrete"]),
                "best_fs": [float(x) for x in result["best_fs"]],
                "candidates_list": result.get("candidates_list", []),
                "released_source_a_by_iteration": result.get(
                    "released_source_a_by_iteration", []
                ),
            }
        )
    return rows


def get_best_by_discrete(rows: list[dict]) -> dict:
    """Best value reached by any run, per benchmark and discrete seed.

    Args:
        rows: Result rows from ``load_rows``, pooled over operating modes.

    Returns:
        Dict mapping ``(benchmark, seed_discrete)`` to the maximum of ``best_fs``
        over all rows with that key. Benchmarks outside ``BENCHMARK_NAMES`` are
        ignored.
    """
    best_by_discrete = {}
    for benchmark in BENCHMARK_NAMES:
        for row in rows:
            if row["benchmark"] == benchmark:
                key = (benchmark, row["seed_discrete"])
                values = row["best_fs"]
                if key not in best_by_discrete:
                    best_by_discrete[key] = max(values)
                else:
                    best_by_discrete[key] = max(best_by_discrete[key], max(values))
    return best_by_discrete


def compute_relative_improvement(best_fs, start_f, max_f):
    """Relative improvement curve ``(f - start_f) / (max_f - start_f)``.

    Args:
        best_fs: Best observed value per iteration.
        start_f: Reference value mapped to 0, the run's first entry.
        max_f: Reference value mapped to 1, the best value reached for this
            benchmark and discrete seed.

    Returns:
        List with one value per entry of ``best_fs``; all zeros when
        ``|max_f - start_f| < 1e-12``.
    """
    denominator = max_f - start_f
    if abs(denominator) < 1e-12:
        return [0.0 for _ in best_fs]
    return [(val - start_f) / denominator for val in best_fs]


def build_curves(rows, curve_len, seed_discretes):
    """Mean relative improvement curves with 95% intervals per benchmark and mode.

    Each run is normalised by its own first value and by the best value any run
    reached for the same benchmark and discrete seed.

    Args:
        rows: Result rows from ``load_rows``.
        curve_len: Number of iterations kept from each ``best_fs``.
        seed_discretes: Discrete seeds in the data; not used in the computation.

    Returns:
        Dict mapping ``(benchmark, operating_mode)`` to ``(mean_curve,
        margin_curve, n_runs)``, where both curves are arrays of length
        ``curve_len`` and ``margin_curve`` is 1.96 standard errors (zero for a
        single run). Pairs without runs are left out.
    """
    best_by_discrete = get_best_by_discrete(rows)
    grouped = defaultdict(list)
    for row in rows:
        benchmark = row["benchmark"]
        operating_mode = row["operating_mode"]
        sd = row["seed_discrete"]
        best_fs = row["best_fs"][:curve_len]

        start_f = best_fs[0]
        max_f = best_by_discrete[(benchmark, sd)]

        ri_curve = compute_relative_improvement(best_fs, start_f, max_f)
        grouped[(benchmark, operating_mode)].append(ri_curve)

    stats = {}
    for benchmark in BENCHMARK_NAMES:
        for operating_mode in OPERATING_MODES:
            curves = np.array(grouped[(benchmark, operating_mode)])
            if len(curves) > 0:
                mean_curve = np.mean(curves, axis=0)
                std_curve = (
                    np.std(curves, axis=0, ddof=1)
                    if len(curves) > 1
                    else np.zeros_like(mean_curve)
                )
                se_curve = std_curve / np.sqrt(len(curves))
                margin_curve = 1.96 * se_curve
                stats[(benchmark, operating_mode)] = (
                    mean_curve,
                    margin_curve,
                    len(curves),
                )
    return stats


def plot_averaged_vs_layer_averaged(stats, output_path, curve_len, seed_discretes):
    """Plot mean relative improvement with 95% bands, one panel per benchmark.

    Three stacked panels (one per entry of ``BENCHMARK_NAMES``), each with one
    curve per operating mode.

    Args:
        stats: Output of ``build_curves``.
        output_path: Path of the PNG to write.
        curve_len: Number of iterations on the x-axis.
        seed_discretes: Discrete seeds in the data; not used in the plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    axes_flat = list(axes.flatten())
    x_values = list(range(1, curve_len + 1))

    colors = {"averaged": "blue", "layer_averaged": "green"}

    for idx, benchmark in enumerate(BENCHMARK_NAMES):
        ax = axes_flat[idx]
        for operating_mode in OPERATING_MODES:
            if (benchmark, operating_mode) not in stats:
                continue
            mean_curve, margin_curve, n_runs = stats[(benchmark, operating_mode)]

            upper = mean_curve + margin_curve
            lower = mean_curve - margin_curve

            label = f"{operating_mode} (n={n_runs})"
            color = colors[operating_mode]

            ax.plot(x_values, mean_curve, label=label, color=color, linewidth=1.5)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.2)

        ax.set_title(f"{benchmark}: Average Relative Improvement")
        if idx == 2:
            ax.set_xlabel("Iterations")
        ax.set_ylabel("Relative Improvement")
        ax.set_xlim(1, curve_len)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_source_a_cache(seed_discretes):
    """Block-A choice table for every benchmark and discrete seed.

    Args:
        seed_discretes: Discrete seeds to generate tables for.

    Returns:
        Dict mapping ``(benchmark, seed_discrete)`` to a numpy array of shape
        ``n_a x DEPENDENT_BLOCK_SIZES[0]``, the ``source_a`` rows produced by
        ``generate_dependent_choice_sources``.
    """
    cache = {}
    for bm_name in BENCHMARK_NAMES:
        benchmark_class = BENCHMARK_CLASS_MAP[bm_name]
        benchmark = benchmark_class(d=TOTAL_DIM)
        bounds = benchmark.bounds()
        for sd in seed_discretes:
            source_a, _, _ = generate_dependent_choice_sources(bounds, sd)
            cache[(bm_name, sd)] = source_a.detach().cpu().numpy()
    return cache


def extract_source_a_index(candidate, source_a):
    """Index of the block-A choice a candidate uses.

    Args:
        candidate: Full candidate vector of length ``TOTAL_DIM``; block A is the
            slice ``[N_CONTINUOUS_DIMS, N_CONTINUOUS_DIMS + DEPENDENT_BLOCK_SIZES[0])``.
        source_a: Block-A choice table, shape ``n_a x DEPENDENT_BLOCK_SIZES[0]``.

    Returns:
        Index of the first row of ``source_a`` whose entries all lie within
        1e-5 of the candidate's block-A slice, or -1 if none does.
    """
    slice_a = np.array(candidate)[
        N_CONTINUOUS_DIMS : N_CONTINUOUS_DIMS + DEPENDENT_BLOCK_SIZES[0]
    ]
    diff = np.abs(source_a - slice_a)
    match_idx = np.where(np.all(diff < 1e-5, axis=1))[0]
    if len(match_idx) > 0:
        return match_idx[0]
    return -1


def plot_scope_extension(rows, output_path, seed_discretes):
    """Bar chart of how often each mode picks newly released block-A choices.

    Only iterations where at least one block-A choice was released count. Per
    benchmark and mode, the left bar is the percentage of those iterations whose
    batch contains at least one released choice; the right bar (second y-axis)
    is the mean number of such candidates in the iterations that contain any.
    Rows without ``candidates_list`` or ``released_source_a_by_iteration`` are
    skipped.

    Args:
        rows: Result rows from ``load_rows``.
        output_path: Path of the PNG to write.
        seed_discretes: Discrete seeds whose block-A tables are generated.
    """
    source_a_cache = get_source_a_cache(seed_discretes)
    stats = defaultdict(lambda: {"valid_iters": 0, "hits": 0, "chosen_counts": []})

    for row in rows:
        benchmark = row["benchmark"]
        operating_mode = row["operating_mode"]
        sd = row["seed_discrete"]

        candidates_list = row["candidates_list"]
        released_list = row["released_source_a_by_iteration"]

        if not candidates_list or not released_list:
            continue

        source_a = source_a_cache[(benchmark, sd)]

        for iter_idx in range(len(candidates_list)):
            if iter_idx >= len(released_list):
                break

            released_now = released_list[iter_idx]
            if len(released_now) == 0:
                continue

            stats[(benchmark, operating_mode)]["valid_iters"] += 1
            chosen_counts_now = 0
            for cand in candidates_list[iter_idx]:
                source_idx = extract_source_a_index(cand, source_a)
                if source_idx in released_now:
                    chosen_counts_now += 1

            if chosen_counts_now > 0:
                stats[(benchmark, operating_mode)]["hits"] += 1
                stats[(benchmark, operating_mode)]["chosen_counts"].append(
                    chosen_counts_now
                )

    labels = []
    percentages = []
    avg_chosen = []

    for bm in BENCHMARK_NAMES:
        for mode in OPERATING_MODES:
            labels.append(f"{bm}\n({mode})")

            st = stats[(bm, mode)]
            valid = st["valid_iters"]
            hits = st["hits"]
            counts = st["chosen_counts"]

            pct = (hits / valid * 100.0) if valid > 0 else 0.0
            avg = np.mean(counts) if len(counts) > 0 else 0.0

            percentages.append(pct)
            avg_chosen.append(avg)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))
    rects1 = ax1.bar(
        x - width / 2,
        percentages,
        width,
        label="% of iterations picking >= 1 new point",
        color="royalblue",
    )

    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + width / 2,
        avg_chosen,
        width,
        label="Avg. new points chosen (when > 0)",
        color="darkorange",
    )

    ax1.set_ylabel("Percentage (%)", color="royalblue", fontsize=12)
    ax2.set_ylabel("Average Count Chosen", color="darkorange", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)

    def autolabel(rects, ax, is_pct=False):
        for rect in rects:
            height = rect.get_height()
            label_str = f"{height:.1f}%" if is_pct else f"{height:.2f}"
            ax.annotate(
                label_str,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1, ax1, is_pct=True)
    autolabel(rects2, ax2, is_pct=False)

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")

    plt.title(
        "Scope Extension Accuracy: Selection of Newly Available Points", fontsize=14
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_results_dir(plots_dir: Path, mode_dirname: str) -> Path:
    """Locate a mode's results folder next to ``plots/`` or under ``results/``.

    Args:
        plots_dir: The ``plots`` directory of run_1.
        mode_dirname: Folder name, e.g. ``results_averaged``.

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

    curve_lengths = {len(r["best_fs"]) for r in rows}
    if not curve_lengths:
        print("No results found.")
        return
    curve_len = min(curve_lengths)

    seed_discretes = sorted(list({r["seed_discrete"] for r in rows}))

    stats = build_curves(rows, curve_len, seed_discretes)
    output_path_ri = plots_dir / "plot_averaged_vs_layer_averaged.png"
    plot_averaged_vs_layer_averaged(stats, output_path_ri, curve_len, seed_discretes)
    print(f"Saved RI plot to {output_path_ri}")

    output_path_se = plots_dir / "scope_extension_accuracy_averaged_vs_layer.png"
    plot_scope_extension(rows, output_path_se, seed_discretes)
    print(f"Saved scope extension plot to {output_path_se}")


if __name__ == "__main__":
    main()
