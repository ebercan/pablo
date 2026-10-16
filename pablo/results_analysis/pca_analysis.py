"""Run PCA sensitivity analysis and generate two diagnostic figures.

This script computes MAP fingerprints for each species family, fits PCA once per
family to obtain explained-variance ratios, and then evaluates the Kneedle
suggested dimensionality across a configurable set of sensitivity values.

Run from the repo root::

    python -m pablo.results_analysis.pca_analysis

The outputs, written next to this script, are:
    1) ``kneedle_sensitivity.png``:
        - x-axis: Kneedle sensitivity
        - y-axis: selected PCA dimension
        - 3 solid curves: catalyst, solvent, base
        - 3 dashed curves: forward-difference slopes for each species
    2) ``variance_explained.png``:
        - x-axis: number of dimensions
        - y-axis: cumulative fraction of variance explained
        - 3 solid curves: catalyst, solvent, base
        - 3 dashed curves: forward-difference slopes for each species
"""

from pathlib import Path

import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from skfp.fingerprints import MAPFingerprint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Edit this list to change which sensitivities are evaluated.
KNEEDLE_SENSITIVITIES = list(range(1, 15, 1))
TARGET_X_TICKS = 12
SMOOTH_CURVE_POINTS = 500
SMOOTHING_WINDOW = 21

EXCEL_FILE = Path(__file__).parent.parent / "nhc_doe.xlsx"
OUTPUT_PLOT = Path(__file__).parent / "kneedle_sensitivity.png"
VARIANCE_EXPLAINED_PLOT = Path(__file__).parent / "variance_explained.png"


def compute_variance_list(fingerprints: np.ndarray) -> list[float]:
    """Compute sorted explained-variance ratios from standardized fingerprints.

    Args:
        fingerprints: Array with shape (n_samples, n_fingerprint_features).

    Returns:
        Descending list of explained-variance ratios.
    """
    x_scaled = StandardScaler().fit_transform(fingerprints)
    pca_full = PCA(random_state=42)
    pca_full.fit(x_scaled)
    return sorted(pca_full.explained_variance_ratio_.tolist(), reverse=True)


def select_dimension_with_kneedle(variance_list: list[float], sensitivity: int) -> int:
    """Return Kneedle-selected dimension for one explained-variance curve.

    Uses the same KneeLocator settings as :class:`pablo.utils.pca_utils.PCAPipeline`. If no knee is
    detected, falls back to the first dimension reaching 80% cumulative variance.

    Args:
        variance_list: Descending explained-variance ratios.
        sensitivity: Kneedle sensitivity parameter ``S``.

    Returns:
        Selected PCA dimensionality.
    """
    kneedle = kneed.KneeLocator(
        range(1, len(variance_list) + 1),
        variance_list,
        S=int(sensitivity),
        curve="convex",
        direction="decreasing",
    )

    if kneedle.knee is not None:
        return int(kneedle.knee)

    cumulative = np.cumsum(variance_list)
    return int(np.argmax(cumulative >= 0.8)) + 1



def load_fingerprints(excel_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load catalyst/solvent/base references and compute MAP fingerprints.

    Args:
        excel_file: Path to the NHC dataset workbook.

    Returns:
        Tuple ``(cat_fp, sol_fp, bas_fp)`` with each array of shape
        ``(n_species, n_bits)``.
    """
    cat_ref = pd.read_excel(excel_file, sheet_name="Catalyst References Achiral")
    sol_ref = pd.read_excel(excel_file, sheet_name="Solvent References")
    bas_ref = pd.read_excel(excel_file, sheet_name="Base References")

    cat_ref = cat_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    sol_ref = sol_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    bas_ref = bas_ref.dropna(subset=["smiles"]).reset_index(drop=True)

    cat_fp = np.asarray(
        MAPFingerprint(radius=2).transform(cat_ref["smiles"].tolist()),
        dtype=np.float64,
    )
    sol_fp = np.asarray(
        MAPFingerprint(radius=1).transform(sol_ref["smiles"].tolist()),
        dtype=np.float64,
    )
    bas_fp = np.asarray(
        MAPFingerprint(radius=1).transform(bas_ref["smiles"].tolist()),
        dtype=np.float64,
    )

    return cat_fp, sol_fp, bas_fp


def build_smoothed_curve(
    x_values: list[int], y_values: list[int], num_points: int = SMOOTH_CURVE_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Create a visually smooth curve from discrete samples.

    Uses dense linear interpolation followed by moving-average smoothing.

    Args:
        x_values: Discrete x values.
        y_values: Discrete y values.
        num_points: Number of dense points used for the smooth curve.

    Returns:
        Tuple of dense x and smoothed y arrays.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    if len(x_unique) < 3:
        return x_unique, y_unique

    x_dense = np.linspace(x_unique.min(), x_unique.max(), num_points)
    y_dense = np.interp(x_dense, x_unique, y_unique)

    window = min(SMOOTHING_WINDOW, len(y_dense))
    if window % 2 == 0:
        window -= 1

    if window < 3:
        return x_dense, y_dense

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    y_padded = np.pad(y_dense, (pad, pad), mode="edge")
    y_smoothed = np.convolve(y_padded, kernel, mode="valid")

    return x_dense, y_smoothed


def forward_difference(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Compute first-order forward numerical derivative dy/dx.

    Args:
        x_values: X values with shape (n,).
        y_values: Y values with shape (n,).

    Returns:
        Forward-difference slope array with shape (n,), where the final point
        repeats the previous slope.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if len(x) < 2:
        return np.zeros_like(y, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    slopes = np.empty_like(y, dtype=float)

    valid_dx = np.where(dx == 0.0, 1.0, dx)
    slopes[:-1] = dy / valid_dx
    slopes[-1] = slopes[-2]

    return slopes


def normalize_curve_start_to_end(values: np.ndarray) -> np.ndarray:
    """Rescale a curve so first point is 1.0 and last point is 0.0.

    Args:
        values: Input curve with shape (n,).

    Returns:
        Rescaled curve with the same shape. If start and end are equal, returns
        a linear fallback from 1.0 to 0.0.
    """
    v = np.asarray(values, dtype=float)

    if len(v) == 0:
        return v
    if len(v) == 1:
        return np.asarray([1.0], dtype=float)

    start = float(v[0])
    end = float(v[-1])

    if np.isclose(start, end):
        return np.linspace(1.0, 0.0, len(v), dtype=float)

    return (v - end) / (start - end)


def plot_kneedle_sensitivity(
    sensitivities: list[int],
    cat_dims: list[int],
    sol_dims: list[int],
    bas_dims: list[int],
    output_path: Path,
) -> None:
    """Create and save the single-figure sensitivity plot.

    Args:
        sensitivities: Kneedle sensitivity values on x-axis.
        cat_dims: Selected catalyst dimensions per sensitivity.
        sol_dims: Selected solvent dimensions per sensitivity.
        bas_dims: Selected base dimensions per sensitivity.
        output_path: Destination image path.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cat_x_smooth, cat_y_smooth = build_smoothed_curve(sensitivities, cat_dims)
    sol_x_smooth, sol_y_smooth = build_smoothed_curve(sensitivities, sol_dims)
    bas_x_smooth, bas_y_smooth = build_smoothed_curve(sensitivities, bas_dims)

    cat_line = ax.plot(cat_x_smooth, cat_y_smooth, linewidth=2.2, label="Catalyst")[0]
    sol_line = ax.plot(sol_x_smooth, sol_y_smooth, linewidth=2.2, label="Solvent")[0]
    bas_line = ax.plot(bas_x_smooth, bas_y_smooth, linewidth=2.2, label="Base")[0]

    cat_slope = normalize_curve_start_to_end(
        forward_difference(cat_x_smooth, cat_y_smooth)
    )
    sol_slope = normalize_curve_start_to_end(
        forward_difference(sol_x_smooth, sol_y_smooth)
    )
    bas_slope = normalize_curve_start_to_end(
        forward_difference(bas_x_smooth, bas_y_smooth)
    )

    slope_ax = ax.twinx()
    slope_ax.set_ylim(0.0, 1.0)
    slope_ax.set_ylabel("Normalized slope shape")

    slope_ax.plot(
        cat_x_smooth,
        cat_slope,
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.15,
        color=cat_line.get_color(),
        label="Catalyst slope",
    )
    slope_ax.plot(
        sol_x_smooth,
        sol_slope,
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.15,
        color=sol_line.get_color(),
        label="Solvent slope",
    )
    slope_ax.plot(
        bas_x_smooth,
        bas_slope,
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.15,
        color=bas_line.get_color(),
        label="Base slope",
    )
    slope_ax.grid(False)

    # Keep source samples visible while using smooth trend curves.
    ax.scatter(sensitivities, cat_dims, marker="o", s=14, alpha=0.45)
    ax.scatter(sensitivities, sol_dims, marker="s", s=14, alpha=0.45)
    ax.scatter(sensitivities, bas_dims, marker="^", s=14, alpha=0.45)

    ax.set_xlabel("Kneedle sensitivity")
    ax.set_ylabel("Converged PCA dimension")
    ax.set_title("Kneedle sensitivity vs selected PCA dimensionality")
    ax.set_xlim(min(sensitivities), max(sensitivities))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TARGET_X_TICKS, integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    lines_main, labels_main = ax.get_legend_handles_labels()
    lines_slope, labels_slope = slope_ax.get_legend_handles_labels()
    ax.legend(lines_main + lines_slope, labels_main + labels_slope)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")
    plt.show()


def plot_variance_explained(
    cat_variance: list[float],
    sol_variance: list[float],
    bas_variance: list[float],
    output_path: Path,
) -> None:
    """Create and save cumulative variance-explained figure.

    Args:
        cat_variance: Catalyst explained-variance ratios.
        sol_variance: Solvent explained-variance ratios.
        bas_variance: Base explained-variance ratios.
        output_path: Destination image path.
    """
    cat_cum = np.cumsum(np.asarray(cat_variance, dtype=float))
    sol_cum = np.cumsum(np.asarray(sol_variance, dtype=float))
    bas_cum = np.cumsum(np.asarray(bas_variance, dtype=float))

    x_cat = np.arange(1, len(cat_cum) + 1)
    x_sol = np.arange(1, len(sol_cum) + 1)
    x_bas = np.arange(1, len(bas_cum) + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cat_line = ax.plot(
        x_cat,
        cat_cum,
        marker="o",
        markersize=3.2,
        linewidth=2.0,
        label="Catalyst",
    )[0]
    sol_line = ax.plot(
        x_sol,
        sol_cum,
        marker="s",
        markersize=3.2,
        linewidth=2.0,
        label="Solvent",
    )[0]
    bas_line = ax.plot(
        x_bas,
        bas_cum,
        marker="^",
        markersize=3.2,
        linewidth=2.0,
        label="Base",
    )[0]

    cat_slope = normalize_curve_start_to_end(forward_difference(x_cat, cat_cum))
    sol_slope = normalize_curve_start_to_end(forward_difference(x_sol, sol_cum))
    bas_slope = normalize_curve_start_to_end(forward_difference(x_bas, bas_cum))

    slope_ax = ax.twinx()
    slope_ax.set_ylim(0.0, 1.0)
    slope_ax.set_ylabel("Normalized slope shape")

    slope_ax.plot(
        x_cat,
        cat_slope,
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.15,
        color=cat_line.get_color(),
        label="Catalyst slope",
    )
    slope_ax.plot(
        x_sol,
        sol_slope,
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.15,
        color=sol_line.get_color(),
        label="Solvent slope",
    )
    slope_ax.plot(
        x_bas,
        bas_slope,
        linestyle=(0, (2.0, 2.0)),
        linewidth=1.15,
        color=bas_line.get_color(),
        label="Base slope",
    )
    slope_ax.grid(False)

    x_max = max(len(cat_cum), len(sol_cum), len(bas_cum))
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TARGET_X_TICKS, integer=True))

    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Cumulative fraction of variance explained")
    ax.set_title("Variance explained vs PCA dimensions")
    ax.grid(True, linestyle="--", alpha=0.35)
    lines_main, labels_main = ax.get_legend_handles_labels()
    lines_slope, labels_slope = slope_ax.get_legend_handles_labels()
    ax.legend(lines_main + lines_slope, labels_main + labels_slope)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")
    plt.show()


def main() -> None:
    """Run sensitivity analysis and generate both figures."""
    print(f"Loading dataset from: {EXCEL_FILE}")
    cat_fp, sol_fp, bas_fp = load_fingerprints(EXCEL_FILE)

    cat_variance = compute_variance_list(cat_fp)
    sol_variance = compute_variance_list(sol_fp)
    bas_variance = compute_variance_list(bas_fp)

    cat_dims = [select_dimension_with_kneedle(cat_variance, s) for s in KNEEDLE_SENSITIVITIES]
    sol_dims = [select_dimension_with_kneedle(sol_variance, s) for s in KNEEDLE_SENSITIVITIES]
    bas_dims = [select_dimension_with_kneedle(bas_variance, s) for s in KNEEDLE_SENSITIVITIES]

    print("Sensitivity values:", KNEEDLE_SENSITIVITIES)
    print("Catalyst dims:", cat_dims)
    print("Solvent dims:", sol_dims)
    print("Base dims:", bas_dims)

    plot_kneedle_sensitivity(
        sensitivities=KNEEDLE_SENSITIVITIES,
        cat_dims=cat_dims,
        sol_dims=sol_dims,
        bas_dims=bas_dims,
        output_path=OUTPUT_PLOT,
    )

    plot_variance_explained(
        cat_variance=cat_variance,
        sol_variance=sol_variance,
        bas_variance=bas_variance,
        output_path=VARIANCE_EXPLAINED_PLOT,
    )


if __name__ == "__main__":
    main()
