"""
Delta-delta scatter plots: pairwise molecular distance vs pairwise functional distance.

Reproduces the style of Figures S35-S40 from Science doi:10.1126/science.adk9227.

By default, three 4-panel figures, one per catalyst featurization (MAP4, GROVER,
Mordred), with KT solvent and thermo base blocks, plus a MAP4 Tanimoto figure.
``--cat``/``--solvent``/``--base`` instead write one figure for that featurization
triple. Each figure:
    top-left:     Manhattan distance vs Δ yield
    top-right:    Euclidean distance vs Δ yield
    bottom-left:  Manhattan distance vs Δ selectivity
    bottom-right: Euclidean distance vs Δ selectivity

Distance is computed on the full concatenated vector of the stored descriptors:
catalyst + solvent + base (zeros where ``base_active`` is 0).

Every experiment row with a yield and a selectivity is used. Replicate targets
(replicate_id >= 0) are replaced by the within-group mean so replicates are not
overrepresented.

CLI, from the repo root::

    python -m pablo.results_analysis.delta_delta_plot
    python -m pablo.results_analysis.delta_delta_plot \
        --cat grover_pca --solvent Hansen --base grover_pca --out <png>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from pablo.featurization_and_kernel_scan import paths as fks_paths

DATA = fks_paths.DATA_DIR
LIB_DIR = fks_paths.LIB_DIR
FEAT_DIR = fks_paths.FEAT_DIR

# Solvent and base are split into competing descriptor blocks; this script needs
# one of each. These names are the block suffixes, i.e. solvent_<X> / base_<Y>.
SOLVENT_FEATURIZATION = "KT"
BASE_FEATURIZATION = "thermo"


def _load_experiment_rows() -> pd.DataFrame:
    """Return experiment rows with replicate targets replaced by group mean.
        Returns:
            pd.DataFrame: Return value of _load_experiment_rows.
    
    Returns:
        pd.DataFrame: Return value of _load_experiment_rows.
"""
    exp = pd.read_parquet(LIB_DIR / "experiments.parquet")
    exp = exp.dropna(subset=["yield_pct", "selectivity"]).astype(
        {"yield_pct": float, "selectivity": float}
    )
    rep_mask = exp.replicate_id >= 0
    for col in ["yield_pct", "selectivity"]:
        group_means = (
            exp[rep_mask]
            .groupby(["catalyst_id", "replicate_id"])[col]
            .transform("mean")
        )
        exp.loc[rep_mask, col] = group_means
    return exp.reset_index(drop=True)


def _build_full_embedding(
    rows: pd.DataFrame,
    cat_featurization: str,
    solvent_featurization: str = SOLVENT_FEATURIZATION,
    base_featurization: str = BASE_FEATURIZATION,
) -> np.ndarray:
    """Build the full (cat + solv + base) embedding matrix for all experiment rows.

    Args:
        rows: Experiment DataFrame (output of _load_experiment_rows).
        cat_featurization: Name of the catalyst featurization folder under FEAT_DIR.
        solvent_featurization: Solvent block, resolving to ``solvent_<name>/``.
        base_featurization: Base block, resolving to ``base_<name>/``.

    Returns:
        (N, D_cat + D_solv + D_base) float64 array, one row per experiment.
    """
    cat_emb = pd.read_parquet(FEAT_DIR / cat_featurization / "descriptors.parquet").select_dtypes(include="number")
    solv_desc = pd.read_parquet(
        FEAT_DIR / f"solvent_{solvent_featurization}" / "descriptors.parquet"
    ).select_dtypes(include="number")
    base_desc = pd.read_parquet(
        FEAT_DIR / f"base_{base_featurization}" / "descriptors.parquet"
    ).select_dtypes(include="number")

    sl = pd.read_parquet(LIB_DIR / "solvents.parquet")
    bl = pd.read_parquet(LIB_DIR / "bases.parquet")
    solv_name_to_id = dict(zip(sl["name"], sl["id"]))
    base_comment_to_id = dict(zip(bl["comment"], bl["id"]))

    d_base = base_desc.shape[1]
    base_zeros = np.zeros(d_base, dtype=np.float64)

    cat_block = cat_emb.loc[rows["catalyst_id"].to_numpy()].to_numpy(dtype=np.float64)

    solv_ids = rows["solvent"].map(solv_name_to_id).to_numpy()
    solv_block = solv_desc.loc[solv_ids].to_numpy(dtype=np.float64)

    base_block = np.zeros((len(rows), d_base), dtype=np.float64)
    for i, (base_name, active) in enumerate(
        zip(rows["base"].to_numpy(), rows["base_active"].to_numpy())
    ):
        if active:
            base_block[i] = base_desc.loc[base_comment_to_id[base_name]].to_numpy(
                dtype=np.float64
            )

    return np.concatenate([cat_block, solv_block, base_block], axis=1)


def _tanimoto_distance(fps: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto distances from a binary (n, d) fingerprint array.
        Returns:
            np.ndarray: Return value of _tanimoto_distance.
    
    Returns:
        np.ndarray: Return value of _tanimoto_distance.
"""
    dot = fps @ fps.T
    norms = fps.sum(axis=1)
    union = norms[:, None] + norms[None, :] - dot
    sim = np.where(union == 0, 1.0, dot / union)
    return 1.0 - sim


def _euclidean_distance(emb: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean (L2) distances from an (n, d) array.
        Returns:
            np.ndarray: Return value of _euclidean_distance.
    
    Returns:
        np.ndarray: Return value of _euclidean_distance.
"""
    diff = emb[:, None, :] - emb[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _upper_triangle_flat(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def _scatter_with_marginals(
    ax_scatter,
    ax_top,
    ax_right,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    r_value: float,
) -> None:
    ax_scatter.scatter(x, y, s=18, alpha=0.35, color="steelblue", linewidths=0)
    ax_scatter.set_xlabel(x_label, fontsize=9)
    ax_scatter.set_ylabel(y_label, fontsize=9)
    ax_scatter.tick_params(labelsize=8)
    ax_scatter.annotate(
        f"$R = {r_value:.2f}$",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right", va="top",
        fontsize=9,
    )

    ax_top.hist(x, bins=40, color="steelblue", edgecolor="none")
    ax_top.set_xlim(ax_scatter.get_xlim())
    ax_top.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    ax_right.hist(y, bins=40, color="steelblue", edgecolor="none", orientation="horizontal")
    ax_right.set_ylim(ax_scatter.get_ylim())
    ax_right.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    for spine in ax_right.spines.values():
        spine.set_visible(False)


def _make_figure(
    rows: pd.DataFrame,
    cat_featurization: str,
    label: str,
    out_path: Path,
    solvent_featurization: str = SOLVENT_FEATURIZATION,
    base_featurization: str = BASE_FEATURIZATION,
) -> None:
    """Build and save a 4-panel delta-delta figure for one featurization.

    Args:
        rows: Experiment rows from _load_experiment_rows.
        cat_featurization: Folder name under FEAT_DIR for the catalyst embedding.
        label: Human-readable featurization name for axis labels.
        out_path: File path for the saved figure.
        solvent_featurization: Solvent block, resolving to ``solvent_<name>/``.
        base_featurization: Base block, resolving to ``base_<name>/``.
    """
    emb = _build_full_embedding(rows, cat_featurization, solvent_featurization, base_featurization)

    man_mat = np.abs(emb[:, None, :] - emb[None, :, :]).sum(axis=-1)
    euc_mat = _euclidean_distance(emb)
    yield_mat = np.abs(
        rows["yield_pct"].to_numpy()[:, None] - rows["yield_pct"].to_numpy()[None, :]
    )
    sel_mat = np.abs(
        rows["selectivity"].to_numpy()[:, None] - rows["selectivity"].to_numpy()[None, :]
    )

    man_flat = _upper_triangle_flat(man_mat)
    euc_flat = _upper_triangle_flat(euc_mat)
    yield_flat = _upper_triangle_flat(yield_mat)
    sel_flat = _upper_triangle_flat(sel_mat)

    configs = [
        (man_flat, yield_flat,  f"Δ {label} (Manhattan)",  "Δ Yield [%]"),
        (euc_flat, yield_flat,  f"Δ {label} (Euclidean)",  "Δ Yield [%]"),
        (man_flat, sel_flat,    f"Δ {label} (Manhattan)",  "Δ Selectivity [%]"),
        (euc_flat, sel_flat,    f"Δ {label} (Euclidean)",  "Δ Selectivity [%]"),
    ]

    fig = plt.figure(figsize=(11, 10))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    for i, (x, y, x_label, y_label) in enumerate(configs):
        r_value, _ = pearsonr(x, y)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[i],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_scatter = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        fig.add_subplot(inner[0, 1]).set_visible(False)

        _scatter_with_marginals(
            ax_scatter, ax_top, ax_right, x, y, x_label, y_label, r_value
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_delta_delta_plots(out_dir: Path = None) -> None:
    """Generate and save four 4-panel delta-delta figures.

    Three figures plot Manhattan and Euclidean distances on the full
    (catalyst + solvent + base) embedding with the MAP4, GROVER or Mordred
    catalyst featurization; the fourth compares catalyst-only MAP4 Tanimoto
    and Euclidean distances (``delta_delta_map4_tanimoto.png``).

    Args:
        out_dir: Output directory. Defaults to the directory of this script.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent

    rows = _load_experiment_rows()

    featurizations = [
        ("map4",    "MAP4",    "delta_delta_map4.png"),
        ("grover",  "GROVER",  "delta_delta_grover.png"),
        ("mordred", "Mordred", "delta_delta_mordred.png"),
    ]

    for cat_feat, label, fname in featurizations:
        _make_figure(rows, cat_feat, label, out_dir / fname)

    _make_tanimoto_figure(rows, out_dir / "delta_delta_map4_tanimoto.png")


def _make_tanimoto_figure(rows: pd.DataFrame, out_path: Path) -> None:
    """Build the Tanimoto vs Euclidean figure using raw MAP4 binary fingerprints.

    Catalyst-only distances (no solv/base), matching the original paper style.
    Left column: Tanimoto, right column: Euclidean on the raw descriptors.
    """
    raw_fps = pd.read_parquet(FEAT_DIR / "map4" / "raw_model_output.parquet")
    fps = raw_fps.loc[rows["catalyst_id"].to_numpy()].to_numpy(dtype=np.float64)

    desc = pd.read_parquet(FEAT_DIR / "map4" / "descriptors.parquet")
    emb = desc.loc[rows["catalyst_id"].to_numpy()].to_numpy(dtype=np.float64)

    tan_mat = _tanimoto_distance(fps)
    euc_mat = _euclidean_distance(emb)
    yield_mat = np.abs(
        rows["yield_pct"].to_numpy()[:, None] - rows["yield_pct"].to_numpy()[None, :]
    )
    sel_mat = np.abs(
        rows["selectivity"].to_numpy()[:, None] - rows["selectivity"].to_numpy()[None, :]
    )

    tan_flat = _upper_triangle_flat(tan_mat)
    euc_flat = _upper_triangle_flat(euc_mat)
    yield_flat = _upper_triangle_flat(yield_mat)
    sel_flat = _upper_triangle_flat(sel_mat)

    configs = [
        (tan_flat, yield_flat, "Δ MAP4 (Tanimoto)",      "Δ Yield [%]"),
        (euc_flat, yield_flat, "Δ MAP4 (Euclidean)", "Δ Yield [%]"),
        (tan_flat, sel_flat,   "Δ MAP4 (Tanimoto)",      "Δ Selectivity [%]"),
        (euc_flat, sel_flat,   "Δ MAP4 (Euclidean)", "Δ Selectivity [%]"),
    ]

    fig = plt.figure(figsize=(11, 10))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    for i, (x, y, x_label, y_label) in enumerate(configs):
        r_value, _ = pearsonr(x, y)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[i],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_scatter = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        fig.add_subplot(inner[0, 1]).set_visible(False)

        _scatter_with_marginals(
            ax_scatter, ax_top, ax_right, x, y, x_label, y_label, r_value
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    """Write the default figures, or one figure for a featurization triple."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cat", default=None, help="Catalyst featurization directory.")
    parser.add_argument("--solvent", default=SOLVENT_FEATURIZATION, help="Solvent block (solvent_<name>/).")
    parser.add_argument("--base", default=BASE_FEATURIZATION, help="Base block (base_<name>/).")
    parser.add_argument("--out", type=Path, default=None, help="PNG path for the --cat figure.")
    args = parser.parse_args()
    if args.cat is None:
        make_delta_delta_plots()
        return
    out = args.out or Path(__file__).parent / f"delta_delta_{args.cat}_{args.solvent}_{args.base}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _make_figure(
        _load_experiment_rows(), args.cat, f"{args.cat} + {args.solvent} + {args.base}",
        out, args.solvent, args.base,
    )


if __name__ == "__main__":
    main()
