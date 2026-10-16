"""Plot run_1 scope-extension results for the four operating modes.

Reads ``results/results_<mode>/result_*.json`` for ``averaged``,
``layer_averaged``, ``only_terminal`` and ``myopic``, keeps only runs present
in all four modes, and writes three PNGs next to this script:

- ``plot_low_load.png``: mean relative improvement per benchmark with 95% CI.
- ``scope_extension_accuracy.png``: share of iterations that pick at least one
  newly released source-A choice, and the mean number picked when any is.
- ``newly_available_picking_rate.png``: per-iteration picking rate of newly
  released source-A choices, with hit rate, used/released counts and mean
  adoption delay in the legend.

Usage:
    python -m pablo.mechthild_runs.run_1_scope_extension.plots.plot_low_load
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
OPERATING_MODES = ["averaged", "layer_averaged", "only_terminal", "myopic"]


def _result_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_rows(results_dir: Path) -> list[dict]:
    """Load result rows and keep only runs present in all four operating modes.

    A run is identified by ``(benchmark, seed_discrete, seed_starting,
    seed_initial_available)``. Missing mode folders are skipped.

    Args:
        results_dir: Folder holding one ``results_<mode>/`` subfolder per mode.

    Returns:
        One dict per result file with keys ``run_key``, ``benchmark``,
        ``operating_mode``, ``seed_discrete``, ``best_fs``, ``candidates_list``
        and ``released_source_a_by_iteration``.
    """
    all_rows = []
    
    for mode in OPERATING_MODES:
        mode_dir = results_dir / f"results_{mode}"
        if not mode_dir.exists():
            continue
            
        for result_path in sorted(mode_dir.glob("result_*.json"), key=_result_index):
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            combination = payload["combination"]
            result = payload["result"]
            
            # The unique key to identify the identical run across different modes
            run_key = (
                str(combination["benchmark"]),
                int(combination.get("seed_discrete", -1)),
                int(combination.get("seed_starting", -1)),
                int(combination.get("seed_initial_available", -1)),
            )

            all_rows.append(
                {
                    "run_key": run_key,
                    "benchmark": run_key[0],
                    "operating_mode": mode,
                    "seed_discrete": run_key[1],
                    "best_fs": [float(x) for x in result["best_fs"]],
                    "candidates_list": result.get("candidates_list", []),
                    "released_source_a_by_iteration": result.get(
                        "released_source_a_by_iteration", []
                    ),
                }
            )

    # Filter to ensure we only keep runs that exist across ALL operating modes
    run_modes = defaultdict(set)
    for row in all_rows:
        run_modes[row["run_key"]].add(row["operating_mode"])
    
    complete_keys = {
        key for key, modes in run_modes.items() if len(modes) == len(OPERATING_MODES)
    }
    
    filtered_rows = [row for row in all_rows if row["run_key"] in complete_keys]
    
    return filtered_rows


def get_best_by_discrete(rows: list[dict]) -> dict:
    """Find the best objective value reached per benchmark and discrete seed.

    The maximum is taken over all runs and all operating modes.

    Args:
        rows: Result rows from ``load_rows``.

    Returns:
        Dict mapping ``(benchmark, seed_discrete)`` to the maximum ``best_fs``
        value.
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
    """Rescale a best-so-far curve to relative improvement.

    Each value becomes ``(f - start_f) / (max_f - start_f)``, so 0 is the
    starting value and 1 is ``max_f``. If ``max_f - start_f`` is below 1e-12
    the whole curve is returned as zeros.

    Args:
        best_fs: Best-so-far objective values per iteration.
        start_f: Value mapped to 0 (the first entry of the curve).
        max_f: Value mapped to 1 (best value for this benchmark and seed).

    Returns:
        List of relative improvements, same length as ``best_fs``.
    """
    denominator = max_f - start_f
    if abs(denominator) < 1e-12:
        return [0.0 for _ in best_fs]
    return [(val - start_f) / denominator for val in best_fs]


def build_curves(rows, curve_len, seed_discretes):
    """Average relative-improvement curves per benchmark and operating mode.

    Each run is truncated to ``curve_len`` iterations and normalised with
    ``compute_relative_improvement`` against its first value and the best value
    for its ``(benchmark, seed_discrete)``.

    Args:
        rows: Result rows from ``load_rows``.
        curve_len: Number of iterations kept per run.
        seed_discretes: Discrete seeds in the data (unused).

    Returns:
        Dict mapping ``(benchmark, mode)`` to ``(mean_curve, margin_curve,
        n_runs)``, where both curves are arrays of length ``curve_len`` and
        ``margin_curve`` is the 95% CI half-width, 1.96 · SE (zeros for a
        single run). Pairs without runs are omitted.
    """
    best_by_discrete = get_best_by_discrete(rows)
    grouped = defaultdict(list)
    for row in rows:
        benchmark = row["benchmark"]
        mode = row["operating_mode"]
        sd = row["seed_discrete"]
        best_fs = row["best_fs"][:curve_len]

        start_f = best_fs[0]
        max_f = best_by_discrete[(benchmark, sd)]

        ri_curve = compute_relative_improvement(best_fs, start_f, max_f)
        grouped[(benchmark, mode)].append(ri_curve)

    stats = {}
    for benchmark in BENCHMARK_NAMES:
        for mode in OPERATING_MODES:
            curves = np.array(grouped[(benchmark, mode)])
            if len(curves) > 0:
                mean_curve = np.mean(curves, axis=0)
                std_curve = (
                    np.std(curves, axis=0, ddof=1)
                    if len(curves) > 1
                    else np.zeros_like(mean_curve)
                )
                se_curve = std_curve / np.sqrt(len(curves))
                margin_curve = 1.96 * se_curve
                stats[(benchmark, mode)] = (
                    mean_curve,
                    margin_curve,
                    len(curves),
                )
    return stats


def plot_low_load(stats, output_path, curve_len, seed_discretes):
    """Plot mean relative improvement with 95% CI bands, one panel per benchmark.

    Args:
        stats: Output of ``build_curves``.
        output_path: Path of the PNG to write.
        curve_len: Number of iterations on the x-axis.
        seed_discretes: Discrete seeds in the data (unused).
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    axes_flat = list(axes.flatten())
    x_values = list(range(1, curve_len + 1))

    colors = {
        "averaged": "blue", 
        "layer_averaged": "green", 
        "only_terminal": "purple", 
        "myopic": "red"
    }

    for idx, benchmark in enumerate(BENCHMARK_NAMES):
        ax = axes_flat[idx]
        for mode in OPERATING_MODES:
            if (benchmark, mode) not in stats:
                continue
            mean_curve, margin_curve, n_runs = stats[(benchmark, mode)]

            upper = mean_curve + margin_curve
            lower = mean_curve - margin_curve

            label = f"{mode} (n={n_runs})"
            color = colors[mode]

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
    """Regenerate the source-A choice table for every benchmark and discrete seed.

    Args:
        seed_discretes: Discrete seeds to regenerate.

    Returns:
        Dict mapping ``(benchmark, seed_discrete)`` to a numpy array of shape
        ``n_a x d_a`` (225 x 18 with the run_1 config).
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
    """Find which source-A choice a candidate uses.

    Compares the candidate's block-A slice (the ``d_a`` entries after the
    continuous dimensions) to every row of ``source_a``.

    Args:
        candidate: One candidate point of length ``d`` (continuous + discrete).
        source_a: Source-A choice table of shape ``n_a x d_a``.

    Returns:
        Index of the first row whose entries all differ by less than 1e-5,
        or -1 if no row matches.
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
    """Plot how often each mode picks newly released source-A choices.

    Only iterations in which at least one source-A choice was released count.
    For each benchmark and mode the bar chart shows the percentage of those
    iterations whose batch contains at least one just-released choice, and the
    mean number of such choices in the batches that contain any.

    Args:
        rows: Result rows from ``load_rows``.
        output_path: Path of the PNG to write.
        seed_discretes: Discrete seeds whose source-A tables are regenerated.
    """
    source_a_cache = get_source_a_cache(seed_discretes)
    stats = defaultdict(lambda: {"valid_iters": 0, "hits": 0, "chosen_counts": []})

    for row in rows:
        benchmark = row["benchmark"]
        mode = row["operating_mode"]
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

            stats[(benchmark, mode)]["valid_iters"] += 1
            chosen_counts_now = 0
            for cand in candidates_list[iter_idx]:
                source_idx = extract_source_a_index(cand, source_a)
                if source_idx in released_now:
                    chosen_counts_now += 1

            if chosen_counts_now > 0:
                stats[(benchmark, mode)]["hits"] += 1
                stats[(benchmark, mode)]["chosen_counts"].append(chosen_counts_now)

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

    # Increased figsize to prevent x-labels from overlapping
    fig, ax1 = plt.subplots(figsize=(18, 6))
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
    ax1.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')

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


def plot_newly_available_picking_rate(rows, output_path, seed_discretes):
    """Plot the per-iteration picking rate of newly released source-A choices.

    For each iteration with a non-empty batch, the rate is the percentage of
    batch candidates whose source-A choice was released in that same
    iteration (0 when nothing was released). Curves are averaged over runs,
    truncated to the shortest run, with 95% CI bands. The legend reports, per
    mode, the mean share of iterations with at least one such pick (Hits), the
    mean number of released choices ever picked versus released (Used), and
    the mean delay in iterations from release to first pick (Delay).

    Args:
        rows: Result rows from ``load_rows``.
        output_path: Path of the PNG to write.
        seed_discretes: Discrete seeds whose source-A tables are regenerated.
    """
    source_a_cache = get_source_a_cache(seed_discretes)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False, sharey=True)
    axes_flat = list(axes.flatten())
    
    colors = {
        "averaged": "blue", 
        "layer_averaged": "green", 
        "only_terminal": "purple", 
        "myopic": "red"
    }

    for idx, benchmark in enumerate(BENCHMARK_NAMES):
        ax = axes_flat[idx]
        
        for mode in OPERATING_MODES:
            rates_by_iter = []
            binary_rates_for_run = []
            
            total_released_all_runs = []
            total_picked_all_runs = []
            all_delays_across_all_runs = []
            
            for row in rows:
                if row["benchmark"] != benchmark or row["operating_mode"] != mode:
                    continue
                    
                sd = row["seed_discrete"]
                candidates_list = row["candidates_list"]
                released_list = row["released_source_a_by_iteration"]
                
                if not candidates_list or not released_list:
                    continue
                    
                source_a = source_a_cache[(benchmark, sd)]
                
                run_rates = []
                binary_hits = 0
                num_valid_iters = 0
                
                release_iters = {}
                for iter_idx in range(min(len(candidates_list), len(released_list))):
                    for src_idx in released_list[iter_idx]:
                        release_iters[src_idx] = iter_idx
                        
                picked_sources = set()
                run_delays = []
                
                for iter_idx in range(min(len(candidates_list), len(released_list))):
                    released_now = released_list[iter_idx]
                    cands = candidates_list[iter_idx]
                    
                    for cand in cands:
                        src_idx = extract_source_a_index(cand, source_a)
                        if src_idx in release_iters and src_idx not in picked_sources:
                            delay = iter_idx - release_iters[src_idx]
                            if delay >= 0:
                                picked_sources.add(src_idx)
                                run_delays.append(delay)
                    
                    if len(cands) == 0:
                        continue
                        
                    num_valid_iters += 1
                    
                    if len(released_now) == 0:
                        run_rates.append(0.0)
                        continue
                        
                    chosen_counts_now = 0
                    for cand in cands:
                        source_idx = extract_source_a_index(cand, source_a)
                        if source_idx in released_now:
                            chosen_counts_now += 1
                            
                    rate = (chosen_counts_now / len(cands)) * 100.0
                    run_rates.append(rate)
                    
                    if chosen_counts_now > 0:
                        binary_hits += 1
                    
                if run_rates:
                    rates_by_iter.append(run_rates)
                if num_valid_iters > 0:
                    binary_rates_for_run.append((binary_hits / num_valid_iters) * 100.0)
                    
                total_released_all_runs.append(len(release_iters))
                total_picked_all_runs.append(len(picked_sources))
                all_delays_across_all_runs.extend(run_delays)
            
            if not rates_by_iter:
                continue
                
            min_iters = min(len(r) for r in rates_by_iter)
            rates_array = np.array([r[:min_iters] for r in rates_by_iter])
            
            mean_curve = np.mean(rates_array, axis=0)
            overall_avg_binary = np.mean(binary_rates_for_run) if binary_rates_for_run else 0.0
            
            avg_released = np.mean(total_released_all_runs) if total_released_all_runs else 0.0
            avg_picked = np.mean(total_picked_all_runs) if total_picked_all_runs else 0.0
            avg_delay = np.mean(all_delays_across_all_runs) if all_delays_across_all_runs else 0.0
            
            if len(rates_array) > 1:
                std_curve = np.std(rates_array, axis=0, ddof=1)
                se_curve = std_curve / np.sqrt(len(rates_array))
                margin_curve = 1.96 * se_curve
            else:
                margin_curve = np.zeros_like(mean_curve)
                
            upper = mean_curve + margin_curve
            lower = np.clip(mean_curve - margin_curve, 0, 100)
            
            x_values = list(range(1, min_iters + 1))
            
            label = f"{mode} (Hits: {overall_avg_binary:.1f}%, Used: {avg_picked:.1f}/{avg_released:.1f}, Delay: {avg_delay:.1f})"
            color = colors[mode]
            
            ax.plot(x_values, mean_curve, label=label, color=color, linewidth=1.5)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.2)
            
        ax.set_title(f"{benchmark}: Newly Available Picking Rate")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Picking Rate (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    plots_dir = Path(__file__).resolve().parent
    results_dir = plots_dir.parent / "results"
    
    # Check if results folder exists directly
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found.")
        return

    rows = load_rows(results_dir)
    if not rows:
        print("No valid complete rows found across all 4 modes.")
        return
        
    curve_lengths = {len(r["best_fs"]) for r in rows}
    curve_len = min(curve_lengths)

    seed_discretes = sorted(list({r["seed_discrete"] for r in rows}))

    stats = build_curves(rows, curve_len, seed_discretes)
    output_path_ri = plots_dir / "plot_low_load.png"
    plot_low_load(stats, output_path_ri, curve_len, seed_discretes)
    print(f"Saved RI plot to {output_path_ri}")

    output_path_se = plots_dir / "scope_extension_accuracy.png"
    plot_scope_extension(rows, output_path_se, seed_discretes)
    print(f"Saved scope extension plot to {output_path_se}")

    output_path_picking = plots_dir / "newly_available_picking_rate.png"
    plot_newly_available_picking_rate(rows, output_path_picking, seed_discretes)
    print(f"Saved newly available picking rate plot to {output_path_picking}")


if __name__ == "__main__":
    main()
