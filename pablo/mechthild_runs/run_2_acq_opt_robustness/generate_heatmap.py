"""Generate robustness visualizations from run_2 GA optimization results.

Reads ``results_mixed/`` and writes ``af_robustness_heatmap.png`` and
``af_value_distribution_fits.png`` next to this script. Run from the repo root::

    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.generate_heatmap
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def collect_grouped_af_values(results_dir: Path) -> dict[tuple[int, int], list[float]]:
    """Collect AF values grouped by GA settings for heatmap generation.

    Args:
        results_dir: Directory containing result JSON files with
            ``n_evals_ga``, ``pop_size``, and ``best_acq_value`` keys.

    Returns:
        Mapping from ``(n_evals_ga, pop_size)`` to a list of AF values.
    """
    data: dict[tuple[int, int], list[float]] = {}
    for file_path in results_dir.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        n_eval = d["n_evals_ga"]
        pop = d["pop_size"]
        af = d["best_acq_value"]
        key = (n_eval, pop)
        if key not in data:
            data[key] = []
        data[key].append(af)
    return data


def collect_all_converged_af_values(base_dir: Path) -> np.ndarray:
    """Collect all converged AF values across all JSON result files.

    This function scans all JSON files below ``base_dir`` and aggregates
    converged AF values from ``acq_values_per_repeat`` if present, otherwise
    falls back to ``best_acq_value``.

    Args:
        base_dir: Run directory containing JSON files.

    Returns:
        One-dimensional array of shape ``(n_values,)`` with all converged AF
        values concatenated across files and GA settings.
    """
    af_values: list[float] = []
    for file_path in base_dir.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("acq_values_per_repeat"), list):
            af_values.extend(
                float(v)
                for v in d["acq_values_per_repeat"]
                if isinstance(v, (int, float))
            )
        elif isinstance(d, dict) and isinstance(d.get("best_acq_value"), (int, float)):
            af_values.append(float(d["best_acq_value"]))
    return np.asarray(af_values, dtype=float)


def _compute_fit_metrics(
    values: np.ndarray, dist_name: str, params: tuple[float, ...], n_params: int
) -> dict[str, float]:
    """Compute AIC and KS goodness-of-fit metrics for one distribution.

    Args:
        values: AF values of shape ``(n_values,)``.
        dist_name: Scipy distribution name.
        params: Distribution parameters passed to scipy ``pdf``/``cdf``.
        n_params: Number of free parameters used in fitting.

    Returns:
        Dictionary with fit metrics ``aic``, ``ks_stat``, and ``ks_pvalue``.
    """
    logpdf = stats.__dict__[dist_name].logpdf(values, *params)
    log_likelihood = float(np.sum(logpdf))
    aic = 2 * n_params - 2 * log_likelihood
    ks_stat, ks_pvalue = stats.kstest(values, dist_name, args=params)
    return {"aic": float(aic), "ks_stat": float(ks_stat), "ks_pvalue": float(ks_pvalue)}


def fit_af_distributions(all_values: np.ndarray) -> dict[str, dict[str, Any]]:
    """Fit Normal, Lognormal, and Gamma distributions to AF values.

    Args:
        all_values: Concatenated AF values with shape ``(n_values,)``.

    Returns:
        Mapping from distribution name to fit details and metrics. The
        ``plot_values`` array in each entry has shape ``(n_values,)`` and is
        the exact data domain used for fitting and plotting.
    """
    min_value = float(np.min(all_values))
    positive_shift = max(0.0, 1e-9 - min_value)
    positive_values = all_values + positive_shift

    mu, sigma = stats.norm.fit(all_values)
    normal_params = (mu, sigma)
    normal_metrics = _compute_fit_metrics(all_values, "norm", normal_params, n_params=2)

    lognorm_shape, _, lognorm_scale = stats.lognorm.fit(positive_values, floc=0)
    lognorm_params = (lognorm_shape, 0.0, lognorm_scale)
    lognorm_metrics = _compute_fit_metrics(
        positive_values, "lognorm", lognorm_params, n_params=2
    )

    gamma_shape, _, gamma_scale = stats.gamma.fit(positive_values, floc=0)
    gamma_params = (gamma_shape, 0.0, gamma_scale)
    gamma_metrics = _compute_fit_metrics(
        positive_values, "gamma", gamma_params, n_params=2
    )

    return {
        "Normal": {
            "dist_key": "norm",
            "params": normal_params,
            "metrics": normal_metrics,
            "plot_values": all_values,
            "shift": 0.0,
        },
        "Lognormal": {
            "dist_key": "lognorm",
            "params": lognorm_params,
            "metrics": lognorm_metrics,
            "plot_values": positive_values,
            "shift": positive_shift,
        },
        "Gamma": {
            "dist_key": "gamma",
            "params": gamma_params,
            "metrics": gamma_metrics,
            "plot_values": positive_values,
            "shift": positive_shift,
        },
    }


def generate_distribution_fit_plot(
    all_values: np.ndarray, fit_results: dict[str, dict[str, Any]], output_path: Path
) -> None:
    """Generate a 3x1 distribution-fit PNG for concatenated AF values.

    Args:
        all_values: AF value array of shape ``(n_values,)``.
        fit_results: Fitting outputs from :func:`fit_af_distributions`.
        output_path: File path for the output PNG.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    x = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 600)

    for ax, (name, payload) in zip(axes, fit_results.items(), strict=True):
        dist = stats.__dict__[payload["dist_key"]]
        params = payload["params"]
        shift = payload["shift"]
        pdf = dist.pdf(x + shift, *params)

        ax.hist(all_values, bins=40, density=True, alpha=0.6, color="steelblue")
        ax.plot(x, pdf, color="darkred", linewidth=2)

        m = payload["metrics"]
        ax.set_title(
            f"{name} fit | AIC={m['aic']:.2f}, KS={m['ks_stat']:.4f}, p={m['ks_pvalue']:.3g}"
        )
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Converged AF values (all GA settings concatenated)")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Distribution fit plot saved to {output_path}")


def generate_heatmap() -> None:
    """Generate AF heatmap and concatenated distribution-fit visualizations.

    Produces:
        1. ``af_robustness_heatmap.png`` from grouped AF values per GA setting.
        2. ``af_value_distribution_fits.png`` with a 3x1 grid of fitted
           Normal, Lognormal, and Gamma distributions over all converged AF
           values concatenated across all JSON result files.
    """
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results_mixed"

    data = collect_grouped_af_values(results_dir)

    n_eval_list = sorted(list(set(k[0] for k in data.keys())))
    pop_list = sorted(list(set(k[1] for k in data.keys())))

    mean_matrix = np.zeros((len(n_eval_list), len(pop_list)))
    var_matrix = np.zeros((len(n_eval_list), len(pop_list)))

    for i, n_eval in enumerate(n_eval_list):
        for j, pop in enumerate(pop_list):
            key = (n_eval, pop)
            if key in data:
                vals = data[key]
                mean_matrix[i, j] = np.mean(vals)
                var_matrix[i, j] = np.var(vals)
            else:
                mean_matrix[i, j] = np.nan
                var_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(var_matrix, cmap="YlOrRd", aspect="auto", origin="lower")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Variance of Best AF Value", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(pop_list)), labels=pop_list)
    ax.set_yticks(np.arange(len(n_eval_list)), labels=n_eval_list)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(n_eval_list)):
        for j in range(len(pop_list)):
            key = (n_eval_list[i], pop_list[j])
            cell_vals = data.get(key, [])

            formatted_vals = [f"{v:.3f}" for v in cell_vals]
            vals_lines = [
                " ".join(formatted_vals[k : k + 3])
                for k in range(0, len(formatted_vals), 3)
            ]
            vals_text = "\n".join(vals_lines) if vals_lines else "-"

            ax.text(
                j,
                i + 0.45,
                f"mean: {mean_matrix[i, j]:.4f}",
                ha="center",
                va="top",
                color="black",
                fontsize=8,
                fontweight="bold",
            )

            ax.text(
                j,
                i + 0.08,
                vals_text,
                ha="center",
                va="center",
                color="black",
                fontsize=6.6,
                linespacing=1.12,
                fontfamily="monospace",
            )

    ax.set_title(
        "AF Value Robustness (Color = Variance, Top = Mean AF, Bottom = All AF Repeats)"
    )
    ax.set_xlabel("Population Size (pop_size)")
    ax.set_ylabel("Number of GA Evaluations (n_evals_ga)")
    fig.tight_layout()

    output_path = base_dir / "af_robustness_heatmap.png"
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")

    all_values = collect_all_converged_af_values(base_dir)
    fit_results = fit_af_distributions(all_values)
    distribution_plot_path = base_dir / "af_value_distribution_fits.png"
    generate_distribution_fit_plot(all_values, fit_results, distribution_plot_path)

    ranked = sorted(
        fit_results.items(),
        key=lambda item: (item[1]["metrics"]["aic"], item[1]["metrics"]["ks_stat"]),
    )
    print("\nDistribution fit ranking (best to worst):")
    for name, payload in ranked:
        m = payload["metrics"]
        print(
            f"- {name}: AIC={m['aic']:.2f}, KS={m['ks_stat']:.4f}, p={m['ks_pvalue']:.3g}"
        )


if __name__ == "__main__":
    generate_heatmap()
