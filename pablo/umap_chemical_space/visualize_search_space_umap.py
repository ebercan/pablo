"""Generate interactive UMAP visualizations for the NHC discrete search space.

This script embeds the discrete space with ``pablo.main.load_reference_data``
(catalyst, solvent and base featurizations selectable with ``--cat``,
``--solvent`` and ``--base``, the ``pablo.main`` production featurizations by
default) and produces nine plots, each as an interactive HTML file and, in 2D, a
PNG of the same name:

1. UMAP of all discrete combinations colored by catalyst group.
2. UMAP of all discrete combinations colored by base.
3. UMAP of all discrete combinations colored by solvent.
4. UMAP of all discrete combinations colored by k-means clusters.
5. UMAP of catalyst-only embeddings colored by catalyst group.
6. UMAP of base-only embeddings colored by base.
7. UMAP of solvent-only embeddings colored by solvent.
8. UMAP of catalyst-only embeddings colored by catalyst ID buckets.
9. UMAP of all discrete combinations colored by catalyst ID buckets.

All UMAPs use ``TARGET_UMAP_DIMENSIONALITY`` (default 2), so dimensionality can
be switched from a single global variable.

CLI, from the repo root::

    python -m pablo.umap_chemical_space.visualize_search_space_umap
    python -m pablo.umap_chemical_space.visualize_search_space_umap \
        --cat grover_pca --solvent Hansen --base grover_pca --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import umap

from pablo.main import (
    BASE_FEATURES,
    CAT_FEATURES,
    DEFAULT_EXCEL_FILE,
    SOLVENT_FEATURES,
    build_discrete_space,
    load_reference_data,
)
from rdkit import Chem
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.1
MAX_CLUSTERS = 15
TARGET_UMAP_DIMENSIONALITY = 2
MARKER_SIZE = 10.0
MARKER_OPACITY = 0.82

if TARGET_UMAP_DIMENSIONALITY not in (2, 3):
    raise ValueError("TARGET_UMAP_DIMENSIONALITY must be 2 or 3.")

ALL_DISCRETE_CATALYST_PLOT = "umap_all_discrete_catalyst_groups.html"
ALL_DISCRETE_BASE_PLOT = "umap_all_discrete_bases.html"
ALL_DISCRETE_SOLVENT_PLOT = "umap_all_discrete_solvents.html"
ALL_DISCRETE_CLUSTER_PLOT = "umap_all_discrete_clusters.html"
CATALYST_ONLY_PLOT = "umap_catalysts_only_groups.html"
BASE_ONLY_PLOT = "umap_bases_only.html"
SOLVENT_ONLY_PLOT = "umap_solvents_only.html"
CATALYST_ONLY_ID_BUCKET_PLOT = "umap_catalysts_only_id_buckets.html"
ALL_DISCRETE_ID_BUCKET_PLOT = "umap_all_discrete_catalyst_id_buckets.html"

NO_BASE_LABEL = "No base (N/A)"
CURRENT_CATALYST_BUCKET_LABEL = "Current cats (1-22)"
PRICING_CATALYST_BUCKET_LABEL = "Pricing cats (23-135)"
OTHER_CATALYST_BUCKET_LABEL = "Other cats (136-161)"

# Requested primary sequence: black, red, yellow, blue, purple, green, pink.
PRIMARY_PLOT_COLORS = [
    "#000000",  # black
    "#ff1f1f",  # red
    "#ffd400",  # yellow
    "#0057ff",  # blue
    "#7b1fff",  # purple
    "#00a63e",  # green
    "#ff4fa3",  # pink
]

# Extra colors are used only when categories exceed seven labels.
EXTENDED_PLOT_COLORS = [
    "#ff7a00",  # orange
    "#00c8ff",  # cyan
    "#8b5e3c",  # brown
    "#7f8c00",  # olive
    "#1f3b8a",  # navy
    "#009688",  # teal
    "#c2185b",  # magenta-rose
    "#6b7280",  # slate gray
    "#6f2da8",  # dark orchid
    "#2e8b57",  # sea green
    "#b35c00",  # dark orange
    "#3f51b5",  # indigo
    "#4caf50",  # medium green
    "#d81b60",  # deep pink
]

CATALYST_ID_BUCKET_COLOR_MAP: Dict[str, str] = {
    CURRENT_CATALYST_BUCKET_LABEL: PRIMARY_PLOT_COLORS[5],
    PRICING_CATALYST_BUCKET_LABEL: PRIMARY_PLOT_COLORS[3],
    OTHER_CATALYST_BUCKET_LABEL: PRIMARY_PLOT_COLORS[0],
}

_BICARBONATE_SMARTS = Chem.MolFromSmarts("[OH1]C(=O)[O-]")
_CARBOXYLATE_SMARTS = Chem.MolFromSmarts("[CX3](=O)[O-]")
_HALIDE_Z = {9, 17, 35, 53}


def fit_umap(
    features: np.ndarray,
    n_components: int = TARGET_UMAP_DIMENSIONALITY,
) -> np.ndarray:
    """Fit a UMAP embedding.

    Args:
        features: Feature matrix with shape (n_samples, n_features).
        n_components: Number of UMAP components.

    Returns:
        Array of shape (n_samples, n_components) containing UMAP coordinates.

    Raises:
        ValueError: If ``features`` has fewer than 3 samples.
    """
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("UMAP requires at least 3 samples for this workflow.")

    n_neighbors = min(UMAP_N_NEIGHBORS, n_samples - 1)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        n_components=n_components,
        metric="euclidean",
        init="spectral",
    )
    return reducer.fit_transform(features)


def choose_k_silhouette(
    features: np.ndarray,
    max_k: int = MAX_CLUSTERS,
) -> Tuple[int, np.ndarray, List[Tuple[int, float]]]:
    """Select k for k-means using silhouette score on high-dimensional features.

    Args:
        features: Matrix of shape (n_samples, n_features) used for clustering.
        max_k: Maximum candidate k.

    Returns:
        Tuple ``(best_k, labels, scored_candidates)`` where:
            - best_k is the selected cluster count.
            - labels has shape (n_samples,).
            - scored_candidates is ``[(k, silhouette_score), ...]``.

    Raises:
        ValueError: If fewer than 3 samples are available.
    """
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Silhouette-based clustering requires at least 3 samples.")

    upper_k = min(max_k, n_samples - 1)
    candidate_scores: List[Tuple[int, float]] = []
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, n_init=10)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels, metric="euclidean")
        candidate_scores.append((k, float(score)))
        labels_by_k[k] = labels

    # Tie-breaker prefers smaller k for interpretability.
    best_k = max(candidate_scores, key=lambda item: (item[1], -item[0]))[0]
    return best_k, labels_by_k[best_k], candidate_scores


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest molecular fragment by atom count.

    Args:
        mol: RDKit molecule.

    Returns:
        Largest fragment molecule.
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 0:
        return mol
    return max(frags, key=lambda frag: frag.GetNumAtoms())


def classify_activation_group(smiles: str) -> str:
    """Classify catalyst activation class from SMILES.

    Args:
        smiles: Catalyst SMILES string.

    Returns:
        One of: ``Ionic salt``, ``Bicarbonate salt``, ``CO2 adduct``,
        ``Free NHC``.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse catalyst SMILES: {smiles!r}")

    frags = Chem.GetMolFrags(mol, asMols=True)

    if len(frags) > 1:
        if any(frag.HasSubstructMatch(_BICARBONATE_SMARTS) for frag in frags):
            return "Bicarbonate salt"
        return "Ionic salt"

    has_carboxylate = mol.HasSubstructMatch(_CARBOXYLATE_SMARTS)
    has_positive = any(atom.GetFormalCharge() > 0 for atom in mol.GetAtoms())
    has_negative = any(atom.GetFormalCharge() < 0 for atom in mol.GetAtoms())
    if has_carboxylate and has_positive and has_negative:
        return "CO2 adduct"
    return "Free NHC"


def classify_ring_type(smiles: str) -> str:
    """Classify ring family using nitrogen count in five-membered rings.

    Args:
        smiles: Catalyst SMILES string.

    Returns:
        One of: ``Triazolium-like``, ``Imidazolium-like``, ``Other ring``.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse catalyst SMILES: {smiles!r}")

    has_two_n_ring = False
    for ring_atom_indices in mol.GetRingInfo().AtomRings():
        if len(ring_atom_indices) != 5:
            continue
        n_count = sum(
            1
            for idx in ring_atom_indices
            if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
        )
        if n_count >= 3:
            return "Triazolium-like"
        if n_count == 2:
            has_two_n_ring = True

    if has_two_n_ring:
        return "Imidazolium-like"
    return "Other ring"


def classify_counterion_family(smiles: str) -> str:
    """Classify anion family for multi-fragment catalyst salts.

    Args:
        smiles: Catalyst SMILES string.

    Returns:
        Counterion class label. For single-fragment molecules returns the
        string ``"None"``.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse catalyst SMILES: {smiles!r}")

    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) <= 1:
        return "None"

    if any(frag.HasSubstructMatch(_BICARBONATE_SMARTS) for frag in frags):
        return "Bicarbonate"

    charges = [
        sum(atom.GetFormalCharge() for atom in frag.GetAtoms()) for frag in frags
    ]
    anion_frags = [frag for frag, charge in zip(frags, charges) if charge < 0]

    if len(anion_frags) == 0:
        cation_idx = int(np.argmax(np.asarray(charges, dtype=float)))
        anion_frags = [frag for idx, frag in enumerate(frags) if idx != cation_idx]

    atomic_numbers: List[int] = []
    for frag in anion_frags:
        atomic_numbers.extend(atom.GetAtomicNum() for atom in frag.GetAtoms())
    atoms = set(atomic_numbers)
    fluorine_count = atomic_numbers.count(9)

    if len(atoms.intersection(_HALIDE_Z)) > 0 and fluorine_count < 3:
        return "Halide"
    if 5 in atoms and fluorine_count >= 4:
        return "BF4-like"
    if 15 in atoms and fluorine_count >= 5:
        return "PF6-like"
    if 80 in atoms:
        return "Heavy-halide"
    return "Other anion"


def substituent_atomic_sum(smiles: str) -> float:
    """Compute substituent-size proxy from atomic-number sum.

    Uses the largest molecular fragment to avoid counterion dominance in salts.

    Args:
        smiles: Catalyst SMILES string.

    Returns:
        Atomic-number sum for the largest fragment.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse catalyst SMILES: {smiles!r}")
    core = _largest_fragment(mol)
    return float(sum(atom.GetAtomicNum() for atom in core.GetAtoms()))


def assign_size_tertiles(size_values: Dict[int, float]) -> Dict[int, str]:
    """Assign substituent-size classes (S/M/L) from tertiles.

    Args:
        size_values: Mapping from catalyst ID to atomic-number sum.

    Returns:
        Mapping from catalyst ID to size class label.
    """
    ids = list(size_values.keys())
    values = np.asarray([size_values[i] for i in ids], dtype=float)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])

    out: Dict[int, str] = {}
    for cat_id in ids:
        value = size_values[cat_id]
        if value <= q1:
            out[cat_id] = "Size S"
        elif value <= q2:
            out[cat_id] = "Size M"
        else:
            out[cat_id] = "Size L"
    return out


def resolve_catalyst_group(
    activation_group: str,
    ring_group: str,
    size_group: str,
    counterion_group: str,
) -> str:
    """Resolve final catalyst group using broad chemistry-aware buckets.

    Priority order is Activation > Counterion family > Ring type. Salts are
    intentionally grouped into broad anion families so the total catalyst
    group count stays manageable in downstream plots.

    Args:
        activation_group: Activation class label.
        ring_group: Ring class label.
        size_group: Size class label.
        counterion_group: Counterion family label.

    Returns:
        Final group label.
    """
    if activation_group == "Ionic salt":
        dominant_salt_families = {"Halide", "BF4-like", "PF6-like"}
        if counterion_group in dominant_salt_families:
            return f"Ionic salt | {counterion_group}"
        return "Ionic salt | Other anion"
    if activation_group == "Bicarbonate salt":
        return "Bicarbonate salt"
    if activation_group == "CO2 adduct":
        return "CO2 adduct"
    if activation_group == "Free NHC":
        return f"Free NHC | {ring_group}"
    if ring_group != "":
        return f"Other catalyst | {ring_group}"
    return f"Other catalyst | {size_group}"


def load_reference_tables(
    excel_file: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load catalyst/solvent/base reference tables.

    Args:
        excel_file: Path to workbook.

    Returns:
        Tuple of cleaned dataframes ``(cat_ref, sol_ref, bas_ref)``.
    """
    cat_ref = pd.read_excel(excel_file, sheet_name="Catalyst References Achiral")
    sol_ref = pd.read_excel(excel_file, sheet_name="Solvent References")
    bas_ref = pd.read_excel(excel_file, sheet_name="Base References")

    cat_ref = cat_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    sol_ref = sol_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    bas_ref = bas_ref.dropna(subset=["smiles"]).reset_index(drop=True)

    return cat_ref, sol_ref, bas_ref


def build_catalyst_metadata(
    cat_ids: Sequence[int],
    cat_smiles_by_id: Dict[int, str],
) -> Tuple[
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
    Dict[int, float],
]:
    """Build catalyst group metadata used for plotting.

    Args:
        cat_ids: Catalyst IDs in canonical order.
        cat_smiles_by_id: Mapping catalyst ID -> SMILES.

    Returns:
        Tuple ``(cat_group, activation_group, ring_group, counterion_group,
        atomic_sums)`` keyed by catalyst ID.
    """
    activation_by_id: Dict[int, str] = {}
    ring_by_id: Dict[int, str] = {}
    counterion_by_id: Dict[int, str] = {}
    atomic_sum_by_id: Dict[int, float] = {}

    for cat_id in cat_ids:
        smiles = cat_smiles_by_id[cat_id]
        activation_by_id[cat_id] = classify_activation_group(smiles)
        ring_by_id[cat_id] = classify_ring_type(smiles)
        counterion_by_id[cat_id] = classify_counterion_family(smiles)
        atomic_sum_by_id[cat_id] = substituent_atomic_sum(smiles)

    size_by_id = assign_size_tertiles(atomic_sum_by_id)

    cat_group_by_id: Dict[int, str] = {}
    for cat_id in cat_ids:
        cat_group_by_id[cat_id] = resolve_catalyst_group(
            activation_group=activation_by_id[cat_id],
            ring_group=ring_by_id[cat_id],
            size_group=size_by_id[cat_id],
            counterion_group=counterion_by_id[cat_id],
        )

    return (
        cat_group_by_id,
        activation_by_id,
        ring_by_id,
        counterion_by_id,
        atomic_sum_by_id,
    )


def build_requested_palette(
    n_colors: int,
    start_index: int = 0,
) -> List[str]:
    """Generate deterministic high-contrast colors.

    Args:
        n_colors: Number of colors required.
        start_index: Palette offset for reserving early colors.

    Returns:
        List of Plotly-compatible hex color strings.
    """
    if n_colors <= 0:
        return []

    ordered = PRIMARY_PLOT_COLORS + EXTENDED_PLOT_COLORS
    shifted = ordered[start_index:] + ordered[:start_index]

    if n_colors <= len(shifted):
        return shifted[:n_colors]

    # For unusually large label counts, continue cyclically.
    return [shifted[i % len(shifted)] for i in range(n_colors)]


def build_color_map(labels: Iterable[str], palette: Sequence[str]) -> Dict[str, str]:
    """Build deterministic label-to-color map.

    Args:
        labels: Category labels.
        palette: Color sequence.

    Returns:
        Mapping label -> hex/rgb color string.
    """
    unique_labels = sorted(set(labels))
    return {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}


def assign_catalyst_id_bucket(cat_id: int) -> str:
    """Map catalyst ID to the requested sourcing bucket.

    Args:
        cat_id: Catalyst ID from the catalyst reference sheet.

    Returns:
        One of ``Current cats (1-22)``, ``Pricing cats (23-135)``, or
        ``Other cats (136-161)``.

    Raises:
        ValueError: If catalyst ID is outside the expected 1..161 range.
    """
    if 1 <= cat_id <= 22:
        return CURRENT_CATALYST_BUCKET_LABEL
    if 23 <= cat_id <= 135:
        return PRICING_CATALYST_BUCKET_LABEL
    if 136 <= cat_id <= 161:
        return OTHER_CATALYST_BUCKET_LABEL
    raise ValueError(
        f"Catalyst ID {cat_id} is outside expected ranges 1-22, 23-135, 136-161."
    )


def _write_png(
    embedding: np.ndarray,
    labels: np.ndarray,
    color_map: Dict[str, str],
    title: str,
    legend_title: str,
    output_path: Path,
) -> None:
    """Write a static 2D UMAP scatter plot.

    Args:
        embedding: UMAP coordinates with shape (n_samples, 2).
        labels: Category label per sample, shape (n_samples,).
        color_map: Label -> color mapping; also the legend order.
        title: Plot title.
        legend_title: Legend title.
        output_path: PNG output path.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    point_size = 8.0 if len(labels) > 1000 else 40.0
    for label, color in color_map.items():
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], s=point_size, c=color,
            alpha=MARKER_OPACITY, linewidths=0, label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, markerscale=2.0 if point_size < 20 else 1.0, frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {output_path}")


def make_umap_plot(
    embedding: np.ndarray,
    labels: Sequence[str],
    hover_text: Sequence[str],
    color_map: Dict[str, str],
    title: str,
    legend_title: str,
    output_path: Path,
) -> None:
    """Write an interactive UMAP scatter plot, plus a static PNG in 2D.

    The 2D or 3D layout follows ``TARGET_UMAP_DIMENSIONALITY``. Only labels
    present in ``color_map`` are drawn.

    Args:
        embedding: UMAP coordinates with shape (n_samples, d) where d is 2 or 3.
        labels: Category label per sample.
        hover_text: Hover text per sample.
        color_map: Label -> color mapping.
        title: Plot title.
        legend_title: Legend title.
        output_path: HTML output path. When ``TARGET_UMAP_DIMENSIONALITY`` is 2,
            a static PNG is also written to ``output_path.with_suffix(".png")``.
    """
    fig = go.Figure()
    labels_array = np.asarray(labels)

    if TARGET_UMAP_DIMENSIONALITY == 2:
        for label in color_map:
            mask = labels_array == label
            if not np.any(mask):
                continue

            fig.add_trace(
                go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode="markers",
                    name=label,
                    marker={
                        "size": MARKER_SIZE,
                        "opacity": MARKER_OPACITY,
                        "color": color_map[label],
                        "line": {"width": 0},
                    },
                    text=[hover_text[i] for i in np.where(mask)[0]],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            legend={"title": {"text": legend_title}},
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
        )
        fig.write_html(output_path)
        print(f"Saved interactive plot: {output_path}")
        _write_png(embedding, labels_array, color_map, title, legend_title, output_path.with_suffix(".png"))
        return

    for label in color_map:
        mask = labels_array == label
        if not np.any(mask):
            continue

        fig.add_trace(
            go.Scatter3d(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                z=embedding[mask, 2],
                mode="markers",
                name=label,
                marker={
                    "size": MARKER_SIZE,
                    "opacity": MARKER_OPACITY,
                    "color": color_map[label],
                    "line": {"width": 0},
                },
                text=[hover_text[i] for i in np.where(mask)[0]],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    x_span = max(float(np.ptp(embedding[:, 0])), 1e-9)
    y_span = max(float(np.ptp(embedding[:, 1])), 1e-9)
    z_span = max(float(np.ptp(embedding[:, 2])), 1e-9)
    max_span = max(x_span, y_span, z_span)

    fig.update_layout(
        title=title,
        legend={"title": {"text": legend_title}},
        scene={
            "xaxis_title": "UMAP 1",
            "yaxis_title": "UMAP 2",
            "zaxis_title": "UMAP 3",
            "aspectmode": "manual",
            "aspectratio": {
                "x": x_span / max_span,
                "y": y_span / max_span,
                "z": z_span / max_span,
            },
            "camera": {"eye": {"x": 1.65, "y": 1.65, "z": 0.95}},
        },
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    fig.write_html(output_path)
    print(f"Saved interactive plot: {output_path}")


def main() -> None:
    """Generate nine UMAP plots for the selected featurizations."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cat", default=CAT_FEATURES, help="Catalyst featurization directory.")
    parser.add_argument("--solvent", default=SOLVENT_FEATURES, help="Solvent featurization (solvent_<name>/).")
    parser.add_argument("--base", default=BASE_FEATURES, help="Base featurization (base_<name>/).")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Output folder.")
    args = parser.parse_args()

    local_excel_file = Path(__file__).parent.parent / "nhc_doe.xlsx"
    excel_file = local_excel_file if local_excel_file.exists() else DEFAULT_EXCEL_FILE
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading references from {excel_file}; featurizations "
        f"cat={args.cat} solvent={args.solvent} base={args.base}"
    )
    cat_map, sol_map, bas_map, cat_is_salt, _, cat_ids, sol_ids, bas_ids = (
        load_reference_data(excel_file, args.cat, args.solvent, args.base)
    )

    cat_ref, _, _ = load_reference_tables(excel_file)
    cat_smiles_by_id = {
        int(row["ID"]): str(row["smiles"])
        for _, row in cat_ref[["ID", "smiles"]].iterrows()
    }

    (
        cat_group_by_id,
        activation_by_id,
        ring_by_id,
        counterion_by_id,
        atomic_sum_by_id,
    ) = build_catalyst_metadata(
        cat_ids=cat_ids,
        cat_smiles_by_id=cat_smiles_by_id,
    )

    # All discrete combinations in high-dimensional PCA space.
    x_discrete, combo_ids = build_discrete_space(
        cat_map=cat_map,
        sol_map=sol_map,
        bas_map=bas_map,
        cat_is_salt=cat_is_salt,
        available_catalysts=cat_ids,
        available_solvents=sol_ids,
        available_bases=bas_ids,
    )
    x_discrete_np = x_discrete.cpu().numpy()
    combo_np = combo_ids.cpu().numpy().astype(int)
    embedding_all = fit_umap(
        x_discrete_np,
        n_components=TARGET_UMAP_DIMENSIONALITY,
    )

    best_k, cluster_labels, scored_candidates = choose_k_silhouette(x_discrete_np)
    print("Silhouette scores by k:")
    for k, score in scored_candidates:
        print(f"  k={k:>2}: {score:.4f}")
    print(f"Selected k={best_k} for clustering.")

    # Consistent color maps reused across paired plots.
    cat_palette = build_requested_palette(
        n_colors=len(set(cat_group_by_id.values())),
        start_index=0,
    )
    cat_color_map = build_color_map(cat_group_by_id.values(), cat_palette)

    base_labels_all = [f"Base {int(b_id)}" for b_id in bas_ids]
    base_palette = build_requested_palette(
        n_colors=len(set(base_labels_all)),
        start_index=1,
    )
    base_color_map = build_color_map(base_labels_all, base_palette)
    base_color_map[NO_BASE_LABEL] = PRIMARY_PLOT_COLORS[0]

    solvent_labels_all = [f"Solvent {int(s_id)}" for s_id in sol_ids]
    solvent_palette = build_requested_palette(
        n_colors=len(set(solvent_labels_all)),
        start_index=0,
    )
    solvent_color_map = build_color_map(solvent_labels_all, solvent_palette)

    cluster_label_names = [f"Cluster {int(c) + 1}" for c in cluster_labels]
    cluster_palette = build_requested_palette(
        n_colors=len(set(cluster_label_names)),
        start_index=0,
    )
    cluster_color_map = build_color_map(cluster_label_names, cluster_palette)

    # Plot 1: All discrete colored by catalyst groups.
    all_cat_labels: List[str] = []
    all_cat_hover: List[str] = []
    for cat_id, sol_id, bas_id in combo_np:
        base_display = bas_id if cat_is_salt[cat_id] else "N/A"
        all_cat_labels.append(cat_group_by_id[cat_id])
        all_cat_hover.append(
            "<br>".join(
                [
                    f"Catalyst ID: {cat_id}",
                    f"Catalyst group: {cat_group_by_id[cat_id]}",
                    f"Activation: {activation_by_id[cat_id]}",
                    f"Ring type: {ring_by_id[cat_id]}",
                    f"Counterion family: {counterion_by_id[cat_id]}",
                    f"Atomic-number sum: {atomic_sum_by_id[cat_id]:.1f}",
                    f"Solvent ID: {sol_id}",
                    f"Base ID: {base_display}",
                ]
            )
        )

    make_umap_plot(
        embedding=embedding_all,
        labels=all_cat_labels,
        hover_text=all_cat_hover,
        color_map=cat_color_map,
        title="UMAP (all discrete) colored by catalyst group",
        legend_title="Catalyst group",
        output_path=output_dir / ALL_DISCRETE_CATALYST_PLOT,
    )

    # Plot 2: All discrete colored by base labels.
    all_base_labels: List[str] = []
    all_base_hover: List[str] = []
    for cat_id, sol_id, bas_id in combo_np:
        if cat_is_salt[cat_id]:
            base_label = f"Base {bas_id}"
        else:
            base_label = NO_BASE_LABEL
        all_base_labels.append(base_label)
        all_base_hover.append(
            "<br>".join(
                [
                    f"Catalyst ID: {cat_id}",
                    f"Solvent ID: {sol_id}",
                    f"Base label: {base_label}",
                    f"Salt catalyst: {cat_is_salt[cat_id]}",
                ]
            )
        )

    make_umap_plot(
        embedding=embedding_all,
        labels=all_base_labels,
        hover_text=all_base_hover,
        color_map=base_color_map,
        title="UMAP (all discrete) colored by base",
        legend_title="Base",
        output_path=output_dir / ALL_DISCRETE_BASE_PLOT,
    )

    # Plot 3: All discrete colored by solvent labels.
    all_sol_labels: List[str] = []
    all_sol_hover: List[str] = []
    for cat_id, sol_id, bas_id in combo_np:
        solvent_label = f"Solvent {sol_id}"
        all_sol_labels.append(solvent_label)
        all_sol_hover.append(
            "<br>".join(
                [
                    f"Catalyst ID: {cat_id}",
                    f"Solvent label: {solvent_label}",
                    f"Base ID: {bas_id if cat_is_salt[cat_id] else 'N/A'}",
                ]
            )
        )

    make_umap_plot(
        embedding=embedding_all,
        labels=all_sol_labels,
        hover_text=all_sol_hover,
        color_map=solvent_color_map,
        title="UMAP (all discrete) colored by solvent",
        legend_title="Solvent",
        output_path=output_dir / ALL_DISCRETE_SOLVENT_PLOT,
    )

    # Plot 4: All discrete colored by high-dimensional k-means clusters.
    cluster_hover = [
        "<br>".join(
            [
                f"Cluster: Cluster {int(cluster_labels[i]) + 1}",
                f"Catalyst ID: {combo_np[i, 0]}",
                f"Solvent ID: {combo_np[i, 1]}",
                f"Base ID: {combo_np[i, 2] if cat_is_salt[int(combo_np[i, 0])] else 'N/A'}",
            ]
        )
        for i in range(len(cluster_labels))
    ]
    make_umap_plot(
        embedding=embedding_all,
        labels=cluster_label_names,
        hover_text=cluster_hover,
        color_map=cluster_color_map,
        title=f"UMAP (all discrete) colored by k-means clusters (k={best_k})",
        legend_title="Cluster",
        output_path=output_dir / ALL_DISCRETE_CLUSTER_PLOT,
    )

    # Plot 5: Catalyst-only UMAP with same catalyst group colors as plot 1.
    catalyst_matrix = torch.stack([cat_map[idx] for idx in cat_ids], dim=0).cpu().numpy()
    catalyst_embedding = fit_umap(
        catalyst_matrix,
        n_components=TARGET_UMAP_DIMENSIONALITY,
    )
    catalyst_labels = [cat_group_by_id[c_id] for c_id in cat_ids]
    catalyst_hover = [
        "<br>".join(
            [
                f"Catalyst ID: {c_id}",
                f"Catalyst group: {cat_group_by_id[c_id]}",
                f"Activation: {activation_by_id[c_id]}",
                f"Ring type: {ring_by_id[c_id]}",
                f"Counterion family: {counterion_by_id[c_id]}",
                f"Atomic-number sum: {atomic_sum_by_id[c_id]:.1f}",
            ]
        )
        for c_id in cat_ids
    ]
    make_umap_plot(
        embedding=catalyst_embedding,
        labels=catalyst_labels,
        hover_text=catalyst_hover,
        color_map=cat_color_map,
        title="UMAP (catalysts only) colored by catalyst group",
        legend_title="Catalyst group",
        output_path=output_dir / CATALYST_ONLY_PLOT,
    )

    # Plot 6: Base-only UMAP with same base colors as plot 2.
    base_matrix = torch.stack([bas_map[idx] for idx in bas_ids], dim=0).cpu().numpy()
    base_embedding = fit_umap(
        base_matrix,
        n_components=TARGET_UMAP_DIMENSIONALITY,
    )
    base_labels = [f"Base {b_id}" for b_id in bas_ids]
    base_hover = [f"Base ID: {b_id}" for b_id in bas_ids]
    make_umap_plot(
        embedding=base_embedding,
        labels=base_labels,
        hover_text=base_hover,
        color_map=base_color_map,
        title="UMAP (bases only) colored by base",
        legend_title="Base",
        output_path=output_dir / BASE_ONLY_PLOT,
    )

    # Plot 7: Solvent-only UMAP with same solvent colors as plot 3.
    solvent_matrix = torch.stack([sol_map[idx] for idx in sol_ids], dim=0).cpu().numpy()
    solvent_embedding = fit_umap(
        solvent_matrix,
        n_components=TARGET_UMAP_DIMENSIONALITY,
    )
    solvent_labels = [f"Solvent {s_id}" for s_id in sol_ids]
    solvent_hover = [f"Solvent ID: {s_id}" for s_id in sol_ids]
    make_umap_plot(
        embedding=solvent_embedding,
        labels=solvent_labels,
        hover_text=solvent_hover,
        color_map=solvent_color_map,
        title="UMAP (solvents only) colored by solvent",
        legend_title="Solvent",
        output_path=output_dir / SOLVENT_ONLY_PLOT,
    )

    # Plot 8: Catalyst-only UMAP colored by catalyst ID buckets.
    catalyst_bucket_labels: List[str] = []
    catalyst_bucket_hover: List[str] = []
    for c_id in cat_ids:
        bucket = assign_catalyst_id_bucket(int(c_id))
        catalyst_bucket_labels.append(bucket)
        catalyst_bucket_hover.append(
            "<br>".join(
                [
                    f"Catalyst ID: {int(c_id)}",
                    f"Catalyst bucket: {bucket}",
                    f"Catalyst group: {cat_group_by_id[int(c_id)]}",
                    f"Activation: {activation_by_id[int(c_id)]}",
                    f"Ring type: {ring_by_id[int(c_id)]}",
                ]
            )
        )

    make_umap_plot(
        embedding=catalyst_embedding,
        labels=catalyst_bucket_labels,
        hover_text=catalyst_bucket_hover,
        color_map=CATALYST_ID_BUCKET_COLOR_MAP,
        title="UMAP (catalysts only) colored by catalyst ID bucket",
        legend_title="Catalyst bucket",
        output_path=output_dir / CATALYST_ONLY_ID_BUCKET_PLOT,
    )

    # Plot 9: All discrete UMAP colored by catalyst ID buckets.
    all_discrete_bucket_labels: List[str] = []
    all_discrete_bucket_hover: List[str] = []
    for cat_id, sol_id, bas_id in combo_np:
        bucket = assign_catalyst_id_bucket(int(cat_id))
        base_display = bas_id if cat_is_salt[int(cat_id)] else "N/A"
        all_discrete_bucket_labels.append(bucket)
        all_discrete_bucket_hover.append(
            "<br>".join(
                [
                    f"Catalyst ID: {int(cat_id)}",
                    f"Catalyst bucket: {bucket}",
                    f"Solvent ID: {int(sol_id)}",
                    f"Base ID: {base_display}",
                ]
            )
        )

    make_umap_plot(
        embedding=embedding_all,
        labels=all_discrete_bucket_labels,
        hover_text=all_discrete_bucket_hover,
        color_map=CATALYST_ID_BUCKET_COLOR_MAP,
        title="UMAP (all discrete) colored by catalyst ID bucket",
        legend_title="Catalyst bucket",
        output_path=output_dir / ALL_DISCRETE_ID_BUCKET_PLOT,
    )

    print(f"Done. Generated 9 UMAP plots in {output_dir}.")


if __name__ == "__main__":
    main()
