"""Shared I/O core for every featurizer family.

Library loading, free-carbene conversion, counter-ion stripping, row normalisation, column-prefix
slugging and parquet/metadata writing, imported by every featurizer family.

Output contract. A featurizer writes into ``paths.FEAT_DIR / <name>/``:

    descriptors.parquet       delivered matrix, index ``catalyst_id``,
                              ``solvent_id`` or ``base_id``
    raw_model_output.parquet  pre-normalisation matrix, when one exists
    metadata.json             shape, label, and family-specific keys

No dimensionality reduction happens at any stage — the delivered matrix is the
raw, full-width featurization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from pablo.featurization_and_kernel_scan import paths
from pablo.utils.catalyst_form_classifier import classify_library, to_free_carbene


def load_catalyst_library() -> tuple[list[int], list[str]]:
    """Load catalyst IDs and pre-catalyst SMILES.

    Returns:
        ``(catalyst_ids, smiles)``, row-aligned, in library order.
    """
    df = pd.read_parquet(paths.CATALYSTS_PARQUET)
    return df["id"].tolist(), df["smiles"].tolist()


def load_catalyst_library_with_forms() -> tuple[list[int], list[str], dict[int, str]]:
    """Load catalyst IDs, SMILES, and the pre-catalyst form classification.

    Returns:
        ``(catalyst_ids, smiles, form_map)`` where ``form_map`` maps catalyst id
        to one of ``salt`` / ``free_nhc`` / ``co2_adduct`` / ``bicarbonate``.
    """
    cat_ids, smiles = load_catalyst_library()
    return cat_ids, smiles, classify_library(cat_ids, smiles)


def to_free_carbene_smiles(pre_cat_smiles: str, form: str) -> str:
    """Return only the free-carbene SMILES for a pre-catalyst.

    Args:
        pre_cat_smiles: The pre-catalyst SMILES as stored in the library.
        form: Its form classification, from ``load_catalyst_library_with_forms``.

    Returns:
        The free-carbene SMILES string. If the input cannot be parsed or the
        result fails validation, ``to_free_carbene`` warns and returns the
        input (or its largest fragment) instead.

    Raises:
        ValueError: If ``form`` is not one of ``salt``, ``free_nhc``,
            ``co2_adduct``, ``bicarbonate`` or ``unknown``.
    """
    free_smiles, _ = to_free_carbene(pre_cat_smiles, form)
    return free_smiles


def strip_counterion_smiles(pre_cat_smiles: str) -> str:
    """Return the pre-catalyst SMILES without its counter-ion.

    Keeps the ``.``-separated fragment with the most heavy atoms, written as it
    appears in the stored SMILES. A salt or bicarbonate keeps its azolium
    cation; a single-fragment CO₂ adduct or free carbene is returned unchanged.

    Args:
        pre_cat_smiles: The pre-catalyst SMILES as stored in the library.

    Returns:
        The SMILES of the largest fragment.

    Raises:
        ValueError: If a fragment does not parse, or two fragments tie for the
            most heavy atoms.
    """
    fragments = pre_cat_smiles.split(".")
    sizes = []
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            raise ValueError(f"Fragment {frag!r} of {pre_cat_smiles!r} does not parse")
        sizes.append(mol.GetNumHeavyAtoms())
    largest = max(sizes)
    if sizes.count(largest) > 1:
        raise ValueError(f"{pre_cat_smiles!r} has two largest fragments ({largest} heavy atoms)")
    return fragments[sizes.index(largest)]


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, leaving all-zero rows untouched.

    Args:
        X: ``(n, d)`` float array.

    Returns:
        ``(n, d)`` array with unit-norm rows; all-zero input rows stay zero.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def slug(name: str) -> str:
    """Convert a label into a lowercase, alphanumeric column prefix.

    Args:
        name: Arbitrary label, e.g. ``"MolFormer-XL"``.

    Returns:
        The slug, e.g. ``"molformer_xl"``.
    """
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def save_feature_matrix(
    row_ids: list[int],
    X: np.ndarray,
    out_path: Path,
    col_prefix: str,
    col_names: list[str] | None = None,
    index_name: str = "catalyst_id",
) -> None:
    """Write a feature matrix to parquet, indexed by component id.

    Args:
        row_ids: Component ids aligned with the rows of ``X``.
        X: ``(n_rows, d_features)`` matrix; cast to float64 on write.
        out_path: Destination parquet path; parents are created.
        col_prefix: Used to name columns ``<prefix>_0 ... <prefix>_{d-1}`` when
            ``col_names`` is not given.
        col_names: Explicit column names, length ``d_features``.
        index_name: Index label — ``catalyst_id``, ``solvent_id`` or ``base_id``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if col_names is None:
        col_names = [f"{col_prefix}_{i}" for i in range(X.shape[1])]

    df = pd.DataFrame(
        X.astype(np.float64),
        index=pd.Index(row_ids, name=index_name),
        columns=col_names,
    )
    df.to_parquet(out_path)


def save_embeddings(
    cat_ids: list[int],
    raw_model_output: np.ndarray,
    descriptors: np.ndarray,
    out_dir: Path,
    model_name: str,
    extra_meta: dict | None = None,
    index_name: str = "catalyst_id",
) -> None:
    """Write the raw output, the delivered matrix, and ``metadata.json``.

    Args:
        cat_ids: Component ids aligned with both matrices — catalyst ids
            unless ``index_name`` says otherwise.
        raw_model_output: ``(n, d_raw)`` output before normalisation.
        descriptors: ``(n, d)`` delivered matrix, already normalised.
        out_dir: Featurization directory to write into.
        model_name: Label recorded in the metadata and printed on completion.
        extra_meta: Additional ``metadata.json`` keys.
        index_name: Index label — ``catalyst_id``, ``solvent_id`` or ``base_id``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = slug(model_name)

    save_feature_matrix(
        cat_ids, raw_model_output, out_dir / "raw_model_output.parquet", f"{prefix}_raw",
        index_name=index_name,
    )
    save_feature_matrix(
        cat_ids, descriptors, out_dir / "descriptors.parquet", prefix, index_name=index_name
    )

    meta = {
        "n_molecules": int(descriptors.shape[0]),
        "n_features": int(descriptors.shape[1]),
        "label": model_name,
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
        "raw_model_dim": int(raw_model_output.shape[1]),
        **(extra_meta or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{model_name}] Saved → {out_dir.relative_to(paths.PKG_ROOT)}")
