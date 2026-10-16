"""Fingerprint and embedding featurizations computed locally.

    map4     MAP4 fingerprints (radius-2, skfp)
    ecfp4    ECFP4 count fingerprints (Morgan radius-2, RDKit)
    mordred  2D Mordred descriptors with the 5-step preprocessing
    mol2vec  pretrained Word2Vec embeddings, L2-normalised
    soap     SOAP power spectrum on xTB free-carbene geometries, L2-normalised

All five run on CPU with no HPC involvement. ``soap`` is the one exception to
"no prerequisites": it reads the xTB-optimised geometries under
``paths.XTB_INPUTS_DIR``, so the interpretable pipeline's ``xtb`` step must have
run first.

No dimensionality reduction is applied: each featurizer ships its raw,
full-width matrix.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from skfp.fingerprints import MAPFingerprint

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features import common

MAP4_RADIUS = 2
ECFP4_RADIUS = 2
ECFP4_NBITS = 2048
_ECFP4_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP4_RADIUS,
    fpSize=ECFP4_NBITS,
)

# ── MAP4 ──────────────────────────────────────────────────────────────────────


def run_map4() -> None:
    """Generate raw MAP4 fingerprints for the 161-molecule library."""
    out_dir = paths.FEAT_DIR / "map4"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_ids, smiles = common.load_catalyst_library()
    print(f"\n[MAP4] Computing fingerprints for {len(smiles)} molecules ...")

    fp_transformer = MAPFingerprint(radius=MAP4_RADIUS)
    X_raw = np.asarray(fp_transformer.transform(smiles), dtype=np.float64)
    print(f"[MAP4] Raw fingerprint shape: {X_raw.shape}")

    common.save_feature_matrix(
        cat_ids, X_raw, out_dir / "raw_model_output.parquet", "map4_raw"
    )
    common.save_feature_matrix(cat_ids, X_raw, out_dir / "descriptors.parquet", "map4")

    meta = {
        "n_molecules": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "label": "MAP4",
        "fp_radius": MAP4_RADIUS,
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[MAP4] Done → {out_dir.relative_to(paths.PKG_ROOT)}")


# ── ECFP4 ─────────────────────────────────────────────────────────────────────


def _ecfp4_fingerprint(smiles: str) -> list[float]:
    """Compute a 2048-dim ECFP4 count fingerprint for one molecule.

    Uses RDKit's MorganGenerator API to avoid deprecated Morgan calls.

    Args:
        smiles: SMILES string.

    Returns:
        List of 2048 counts (float64).
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit failed to parse SMILES: {smiles}"
    fp = _ECFP4_GENERATOR.GetCountFingerprint(mol)
    arr = np.zeros(ECFP4_NBITS, dtype=np.float64)
    for bit, count in fp.GetNonzeroElements().items():
        arr[bit] = float(count)
    return arr.tolist()


def run_ecfp4() -> None:
    """Generate raw ECFP4 count fingerprints for the 161-molecule library."""
    out_dir = paths.FEAT_DIR / "ecfp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_ids, smiles = common.load_catalyst_library()
    print(f"\n[ECFP4] Computing fingerprints for {len(smiles)} molecules ...")

    X_raw = np.array([_ecfp4_fingerprint(smi) for smi in smiles], dtype=np.float64)
    print(f"[ECFP4] Raw fingerprint shape: {X_raw.shape}")

    common.save_feature_matrix(
        cat_ids, X_raw, out_dir / "raw_model_output.parquet", "ecfp4_raw"
    )
    common.save_feature_matrix(cat_ids, X_raw, out_dir / "descriptors.parquet", "ecfp4")

    meta = {
        "n_molecules": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "label": "ECFP4",
        "fp_radius": ECFP4_RADIUS,
        "n_bits": ECFP4_NBITS,
        "count_fingerprint": True,
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[ECFP4] Done → {out_dir.relative_to(paths.PKG_ROOT)}")


# ── Mordred ────────────────────────────────────────────────────────────────────


def _preprocess_mordred(
    X_raw: np.ndarray, smiles: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply the 5-step Mordred preprocessing pipeline.

    Steps:
        1. Drop columns with any NaN or Inf.
        2. Drop zero-variance columns.
        3. Drop columns with variance < 0.01 after z-score standardization.
        4. Drop one of each pair with |correlation| > 0.95 (keep higher-variance one).
        5. Standardize (z-score) on the library.

    Step 5 leaves the delivered matrix z-scored on the library.

    Args:
        X_raw: (161, d_mordred) raw descriptor matrix.
        smiles: SMILES strings (for error reporting only).

    Returns:
        (X_processed, kept_descriptor_names) where X_processed is (161, d_kept).
    """
    col_names = [str(i) for i in range(X_raw.shape[1])]
    df = pd.DataFrame(X_raw, columns=col_names)

    # Step 1: drop any NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1)
    print(f"  After NaN/Inf drop:           {df.shape[1]} descriptors")

    # Step 2: drop zero-variance
    df = df.loc[:, df.var() > 0]
    print(f"  After zero-variance drop:     {df.shape[1]} descriptors")

    # Step 3: drop variance < 0.01 after standardization
    from sklearn.preprocessing import StandardScaler as SS

    X_std = SS().fit_transform(df.values)
    var_mask = X_std.var(axis=0) >= 0.01
    df = df.loc[:, var_mask]
    print(f"  After low-variance (std) drop:{df.shape[1]} descriptors")

    # Step 4: drop high-correlation duplicates (|corr| > 0.95, keep higher variance)
    X_vals = df.values.astype(np.float64)
    corr = np.corrcoef(X_vals.T)
    variances = X_vals.var(axis=0)
    n = corr.shape[0]
    drop_indices: set[int] = set()
    for i in range(n):
        if i in drop_indices:
            continue
        for j in range(i + 1, n):
            if j in drop_indices:
                continue
            if abs(corr[i, j]) > 0.95:
                if variances[i] >= variances[j]:
                    drop_indices.add(j)
                else:
                    drop_indices.add(i)
                    break
    keep_indices = [i for i in range(n) if i not in drop_indices]
    df = df.iloc[:, keep_indices]
    print(f"  After correlation dedup:      {df.shape[1]} descriptors")

    # Step 5: final standardization
    X_final = SS().fit_transform(df.values).astype(np.float64)
    return X_final, df.columns.tolist()


def run_mordred() -> None:
    """Generate preprocessed Mordred descriptors for the 161-molecule library."""
    from mordred import Calculator, descriptors as mordred_desc

    out_dir = paths.FEAT_DIR / "mordred"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_ids, smiles = common.load_catalyst_library()
    print(f"\n[Mordred] Computing 2D descriptors for {len(smiles)} molecules ...")

    calc = Calculator(mordred_desc, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols), "Some SMILES failed RDKit parsing"

    result = calc.pandas(mols)
    result_numeric = result.apply(pd.to_numeric, errors="coerce")
    raw_col_names = [str(c) for c in result_numeric.columns]
    X_raw = result_numeric.values.astype(np.float64)
    print(f"[Mordred] Raw descriptor shape: {X_raw.shape}")

    common.save_feature_matrix(
        cat_ids,
        X_raw,
        out_dir / "raw_model_output.parquet",
        "mordred_raw",
        col_names=raw_col_names,
    )

    X_processed, kept_cols = _preprocess_mordred(X_raw, smiles)

    common.save_feature_matrix(
        cat_ids,
        X_processed,
        out_dir / "descriptors.parquet",
        "mordred",
        col_names=kept_cols,
    )

    meta = {
        "n_molecules": int(X_processed.shape[0]),
        "n_features": int(X_processed.shape[1]),
        "label": "Mordred",
        "n_raw_descriptors": int(X_raw.shape[1]),
        "n_kept_descriptors": int(X_processed.shape[1]),
        "kept_descriptor_indices": kept_cols[:20],
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[Mordred] Done → {out_dir.relative_to(paths.PKG_ROOT)}")


# ── Mol2vec ────────────────────────────────────────────────────────────────────


def run_mol2vec(model_path: Path) -> None:
    """Generate L2-normalized Mol2vec embeddings for the 161-molecule library.

    Requires pretrained weights.  The canonical mol2vec_weights.pkl is available at:
    https://github.com/samoturk/mol2vec/blob/master/examples/models/mol2vec_weights.pkl
    Download it into ``data/weights/`` (``paths.WEIGHTS_DIR``); the registry
    passes ``paths.WEIGHTS_DIR / "mol2vec_weights.pkl"`` as ``model_path``.

    Args:
        model_path: Path to the pretrained mol2vec Word2Vec .pkl model.
    """
    from rdkit import RDLogger
    from gensim.models import Word2Vec
    from mol2vec.features import DfVec, mol2alt_sentence, MolSentence

    out_dir = paths.FEAT_DIR / "mol2vec"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert model_path.exists(), (
        f"Mol2vec model not found at {model_path}.  "
        "Download from: https://github.com/samoturk/mol2vec/tree/master/examples/models"
    )

    cat_ids, smiles = common.load_catalyst_library()
    print(f"\n[Mol2vec] Generating embeddings for {len(smiles)} molecules ...")

    w2v_model = Word2Vec.load(str(model_path))

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols), "Some SMILES failed RDKit parsing"

    RDLogger.DisableLog("rdApp.warning")
    sentences = [MolSentence(mol2alt_sentence(m, 1)) for m in mols]
    RDLogger.EnableLog("rdApp.warning")

    def _sentences2vec(sentences, model, unseen="UNK"):
        keys = set(model.wv.key_to_index.keys())
        vec = []
        unseen_vec = model.wv.get_vector(unseen) if unseen else None
        for sentence in sentences:
            if unseen:
                v = sum(
                    [
                        model.wv.get_vector(y) if y in keys else unseen_vec
                        for y in sentence
                    ]
                )
            else:
                v = sum([model.wv.get_vector(y) for y in sentence if y in keys])
            vec.append(v)
        return np.array(vec)

    embeddings_raw = [
        DfVec(s) for s in _sentences2vec(sentences, w2v_model, unseen="UNK")
    ]
    X_raw = np.array([e.vec for e in embeddings_raw], dtype=np.float64)
    print(f"[Mol2vec] Raw embedding shape: {X_raw.shape}")

    common.save_feature_matrix(
        cat_ids, X_raw, out_dir / "raw_model_output.parquet", "mol2vec_raw"
    )

    X_norm = common.l2_normalize(X_raw)

    common.save_feature_matrix(cat_ids, X_norm, out_dir / "descriptors.parquet", "mol2vec")

    meta = {
        "n_molecules": int(X_norm.shape[0]),
        "n_features": int(X_norm.shape[1]),
        "label": "Mol2vec",
        "model_path": str(model_path),
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[Mol2vec] Done → {out_dir.relative_to(paths.PKG_ROOT)}")



# ── SOAP ──────────────────────────────────────────────────────────────────────


SOAP_R_CUT: float = 6.0
"""SOAP cutoff radius in Å, shared by the catalyst and the component SOAP."""

SOAP_N_MAX: int = 6
"""Number of SOAP radial basis functions."""

SOAP_L_MAX: int = 4
"""Maximum degree of the SOAP spherical harmonics."""


def atoms_from_xyz(xyz_path: Path):
    """Build an ASE Atoms object from an xyz file.

    Args:
        xyz_path: Path to the ``.xyz`` geometry file.

    Returns:
        ASE ``Atoms`` with the file's element symbols and Cartesian
        coordinates (Å).
    """
    from ase import Atoms

    from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
        read_xyz,
    )

    elements, coords = read_xyz(xyz_path)
    return Atoms(symbols=elements, positions=coords)


def soap_matrix(
    geometries: dict,
    rcut: float = SOAP_R_CUT,
    nmax: int = SOAP_N_MAX,
    lmax: int = SOAP_L_MAX,
) -> tuple[list[int], np.ndarray, list[str]]:
    """Atom-averaged SOAP power spectrum over a set of geometries.

    The species list is the union over ``geometries``, so the descriptor width
    depends on which elements the set contains.

    Args:
        geometries: Mapping of integer id to ASE ``Atoms``.
        rcut: Cutoff radius in Å.
        nmax: Number of radial basis functions.
        lmax: Maximum spherical-harmonic degree.

    Returns:
        ``(ids, raw, species)``: the sorted ids, the ``(n, d)`` unnormalised
        float64 matrix row-aligned with them, and the sorted species list.
    """
    from dscribe.descriptors import SOAP

    species = sorted(
        {sym for atoms in geometries.values() for sym in atoms.get_chemical_symbols()}
    )
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        average="inner",
    )
    ids = sorted(geometries.keys())
    raw = np.stack([soap.create(geometries[i]) for i in ids], axis=0).astype(np.float64)
    return ids, raw, species


def generate_soap(
    rcut: float = SOAP_R_CUT, nmax: int = SOAP_N_MAX, lmax: int = SOAP_L_MAX
) -> None:
    """Compute the L2-normalized, atom-averaged SOAP power spectrum for every
    xTB-optimized geometry in the 161-molecule library.

    Reads ``data/xtb_inputs/C*/xtbopt.xyz`` and writes the ``soap`` featurizer
    outputs under ``data/featurizations/soap/``. Exits with status 1 if no
    geometry is found.

    Args:
        rcut: Cutoff radius in Å.
        nmax: Number of radial basis functions.
        lmax: Maximum spherical-harmonic degree.
    """
    import sys

    geometries: dict[int, "Atoms"] = {}
    for xtbopt in sorted(paths.XTB_INPUTS_DIR.glob("C*/xtbopt.xyz")):
        cid = int(xtbopt.parent.name[1:])
        geometries[cid] = atoms_from_xyz(xtbopt)

    if not geometries:
        print(
            "[SOAP] No xTB-optimized geometries found under data/xtb_inputs/C*/xtbopt.xyz."
        )
        print("       Run generate_interpretable_features --xtb first.")
        sys.exit(1)

    cat_ids, raw, species = soap_matrix(geometries, rcut, nmax, lmax)

    normed = common.l2_normalize(raw)

    out_dir = paths.FEAT_DIR / "soap"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Written directly rather than through common.save_feature_matrix: SOAP's stored
    # columns are integer labels (0, 1, ... 7424), not "soap_N" strings. Routing it
    # through the shared writer would rename all 7425 columns and change the
    # feature_names recorded in every run's diagnostics.
    raw_df = pd.DataFrame(raw, index=pd.Index(cat_ids, name="catalyst_id"))
    raw_df.to_parquet(out_dir / "raw_model_output.parquet")
    normed_df = pd.DataFrame(normed, index=pd.Index(cat_ids, name="catalyst_id"))
    normed_df.to_parquet(out_dir / "descriptors.parquet")

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_molecules": len(cat_ids),
                "n_features": int(normed.shape[1]),
                "descriptors_file": "descriptors.parquet",
                "soap": {
                    "r_cut": rcut,
                    "n_max": nmax,
                    "l_max": lmax,
                    "average": "inner",
                    "species": species,
                },
                "geometry_source": "xTB GFN2 / ALPB(toluene) optimized free-carbene geometry",
            },
            indent=2,
        )
    )
    print(
        f"✓ SOAP ({normed.shape[1]} dims) → {out_dir.relative_to(paths.PKG_ROOT)}"
    )


