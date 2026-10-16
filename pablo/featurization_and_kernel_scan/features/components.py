"""Solvent, base, and continuous-variable featurizations.

These are the three non-catalyst blocks of the design matrix:

    solvent_KT, solvent_Hansen, solvent_Gutmann, solvent_OHE
    base_thermo, base_steric, base_OHE
    solvent_map, base_map       MAP2 fingerprints over each library
    solvent_soap, base_soap     SOAP on xTB geometries of each library
    continuous                  continuous design variables on [-1, 1]
    <source>_pca                frozen PCA-knee reductions of solvent_grover,
                                solvent_soap, base_grover and base_soap

Scaling applied here: solvent physical descriptors and the continuous base
descriptors (proton affinity, Sterimol) are z-scored over the full 14-solvent
or 8-base library; continuous variables are mapped from their design bounds to
[-1, 1]; SOAP vectors are row-wise L2-normalized. One-hot blocks and MAP
fingerprints are left as computed. Solvent and base matrices are indexed by
``solvent_id`` / ``base_id`` rather than ``catalyst_id``.

Note on ``component_slug``: it collapses each run of non-alphanumeric characters
to a single underscore and preserves case, which is what the one-hot column
names (``is_Me_THF``) depend on. It is deliberately *not*
``common.slug``, which lowercases and maps every character individually.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import re
from rdkit import Chem
from skfp.fingerprints import MAPFingerprint

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features import common
from pablo.featurization_and_kernel_scan.features import fingerprints as fp

# ── Solvent / Base / Continuous ───────────────────────────────────────────────

# Solvent physical descriptor table.
# Columns: kt_alpha, kt_beta, kt_pi_star, epsilon, gutmann_dn,
#          hansen_dD, hansen_dP, hansen_dH, logP
#
# Primary references:
# - Kamlet-Taft: Marcus (1993) Chem. Soc. Rev. 22, 409
# - Gutmann DN: Gutmann (1978) "The Donor-Acceptor Approach to Molecular Interactions"
# - Hansen: Hansen (2007) "Hansen Solubility Parameters: A User's Handbook"
# - logP: PubChem / Hansch-Leo fragment method
_SOLVENT_PHYSICAL = {
    "THF":               (0.00,  0.55,  0.58,  7.58,  20.0,  16.8,  5.7,  8.0,  0.46),
    "Me-THF":            (0.00,  0.52,  0.53,  6.97,  18.0,  16.9,  5.0,  4.3,  0.95),
    "Propylenecarbonate":(0.00,  0.40,  0.83, 64.92,  15.1,  20.0, 18.0,  4.1, -0.41),
    "Dimethoxyethane":   (0.00,  0.41,  0.41,  7.20,  24.0,  15.2,  6.0,  7.2,  0.73),
    "Toluene":           (0.00,  0.11,  0.54,  2.38,   0.0,  18.0,  1.4,  2.0,  2.73),
    "Ethyl-Acetate":     (0.00,  0.45,  0.55,  6.02,  17.1,  15.8,  5.3,  7.2,  0.73),
    "Heptane":           (0.00,  0.00, -0.08,  1.92,   0.0,  15.3,  0.0,  0.0,  4.66),
    "m-xylene":          (0.00,  0.12,  0.47,  2.37,   0.0,  17.8,  1.0,  3.1,  3.20),
    "H2O":               (1.17,  0.47,  1.09, 78.36,  18.0,  15.5, 16.0, 42.3, -1.38),
    "EtOH":              (0.86,  0.75,  0.54, 24.55,  32.0,  15.8,  8.8, 19.4, -0.31),
    "DMF":               (0.00,  0.69,  0.88, 36.71,  26.6,  17.4, 13.7, 11.3, -1.01),
    "DCM":               (0.13,  0.10,  0.82,  8.93,   0.0,  18.2,  6.3,  6.1,  1.25),
    "1,4-Dioxane":       (0.00,  0.37,  0.55,  2.22,  14.8,  19.0,  1.8,  7.4, -0.27),
    "HFIP":              (1.96,  0.00,  0.65, 16.70,   0.0,  17.2,  4.5, 14.8,  1.50),
}

_SOLVENT_RAW_COLS = [
    "kt_alpha", "kt_beta", "kt_pi_star", "epsilon", "gutmann_dn",
    "hansen_dD", "hansen_dP", "hansen_dH", "logP",
]

SOLVENT_BLOCKS: dict[str, list[str]] = {
    "KT": ["kt_alpha", "kt_beta", "kt_pi_star"],
    "Hansen": ["hansen_dD", "hansen_dP", "hansen_dH", "logP"],
    "Gutmann": ["gutmann_dn", "epsilon"],
}
"""Solvent descriptor blocks written to ``solvent_<key>/``.

Each block is one theory of what makes a solvent matter, kept separate so the
CV grid can ask which theory the GP actually needs rather than handing it a
pooled table and hoping ARD sorts it out. ``KT`` is the Kamlet-Taft
solvatochromic triple (hydrogen-bond donation, acceptance, dipolarity/
polarizability); ``Hansen`` is the cohesion-energy decomposition plus ``logP``,
which shares its solubility/partition character; ``Gutmann`` is Lewis basicity
towards a cation (donor number) alongside bulk permittivity. ``OHE`` is built
separately — it carries no theory at all and exists as the no-descriptor
control.

Every column here is continuous and is z-scored across the full 14-solvent
library before being split into blocks, so a block's values do not depend on
which other blocks exist."""

# Base metadata: proton affinity + is_organic + Sterimol fragment SMILES and
# (atom1, atom2) 1-based indices.
# Sterimol convention: atom1 = reference/origin (e.g. C or Si bonded to basic site),
#   atom2 = basic atom (N or O); measured along atom1→atom2 axis.
_BASE_META = {
    #                                  PA     org  smiles_fragment                     a1  a2
    "Triethylamine":                 (981.8,  1, "CCN(CC)CC",                            1,  3),
    "Potassium tert-butoxide":       (1575.0, 0, "[O-]C(C)(C)C",                         2,  1),
    "Rubidium carbonate":            (1500.0, 0, "[O-]C([O-])=O",                        2,  1),
    "N-Ethyl-N-isopropylpropan-2-amine (Hünig's base)": (994.3, 1, "CCN(C(C)C)C(C)C", 1,  3),
    "Cesium carbonate":              (1500.0, 0, "[O-]C([O-])=O",                        2,  1),
    "1,8-Diazabicyclo[5.4.0]undec-7-ene": (1028.0, 1, "C1CCN2CCCCC2=N1",              9, 10),
    "Potassium bis(trimethylsilyl)amide": (1475.0, 0, "[N-]([Si](C)(C)C)[Si](C)(C)C", 2,  1),
    "Sodium tert-butoxide":          (1575.0, 0, "[O-]C(C)(C)C",                         2,  1),
}

BASE_BLOCKS: dict[str, list[str]] = {
    "thermo": ["pa_kJ_mol", "is_inorganic", "is_organic"],
    "steric": ["sterimol_L", "sterimol_B1", "sterimol_B5"],
}
"""Base descriptor blocks written to ``base_<key>/``.

``thermo`` is what the base does thermodynamically — proton affinity, plus a
two-column one-hot for the organic/inorganic split that decides whether the
conjugate acid is a soluble ammonium or a lattice-bound alkali salt.
``steric`` is the Sterimol shadow around the basic atom. ``OHE`` is the
no-descriptor control.

The organic flag is one-hot rather than an integer code because the assembled
experiment tensor zero-fills the base block for catalysts that need no external
base (`requires_external_base` is False). An all-zero one-hot is then exactly
"no base" and cannot be confused with either real class, which an integer code
can only fake by reserving a magic index."""

_BASE_CONTINUOUS_COLS = [
    "pa_kJ_mol", "sterimol_L", "sterimol_B1", "sterimol_B5",
]

_CONTINUOUS_BOUNDS = {
    "fa_ratio":    (1.0,  3.0),
    "loading":     (0.5, 10.0),
    "temperature": (25.0, 100.0),
    "time":        (1.0, 24.0),
}


def component_slug(name: str) -> str:
    """Reduce a library member name to a safe, readable column suffix.

    Args:
        name: Solvent or base name as it appears in the reference sheet.

    Returns:
        The name with every run of non-alphanumeric characters collapsed to a
        single underscore, stripped at both ends.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def _build_identity_onehot(
    ids: list[int], names: list[str], index_name: str
) -> pd.DataFrame:
    """Build a one-hot identity encoding over a small library.

    This is the no-descriptor control: it asserts only that the members are
    distinct, so any featurization that fails to beat it is not earning its
    chemistry.

    Args:
        ids: Integer IDs from the reference sheet.
        names: Corresponding names, used for the column suffixes.
        index_name: Index name for the returned frame (``solvent_id`` / ``base_id``).

    Returns:
        DataFrame indexed by ID with one 0/1 column per library member, in the
        order the IDs were supplied.
    """
    cols = [f"is_{component_slug(n)}" for n in names]
    assert len(set(cols)) == len(cols), f"Duplicate one-hot column names: {cols}"
    block = np.eye(len(ids), dtype=np.float64)
    return pd.DataFrame(block, index=pd.Index(ids, name=index_name), columns=cols)


def _build_solvent_raw(solvent_ids: list[int], solvent_names: list[str]) -> pd.DataFrame:
    """Build the z-scored solvent descriptor table that the blocks are cut from.

    The scaler is always fit on the full library supplied here, so the caller
    must pass all 14 solvents. Standardization is per column, which is
    why splitting the result into blocks afterwards leaves every value untouched.

    Args:
        solvent_ids: Integer IDs from the solvent reference sheet.
        solvent_names: Corresponding solvent name strings (all 14 expected).

    Returns:
        DataFrame indexed by solvent_id with ``_SOLVENT_RAW_COLS`` plus ``name``.
    """
    from sklearn.preprocessing import StandardScaler

    rows = []
    for sid, name in zip(solvent_ids, solvent_names):
        assert name in _SOLVENT_PHYSICAL, (
            f"Solvent '{name}' has no entry in the physical descriptor table.  "
            "Add it to _SOLVENT_PHYSICAL before proceeding."
        )
        rows.append({**dict(zip(_SOLVENT_RAW_COLS, _SOLVENT_PHYSICAL[name])),
                     "solvent_id": sid, "name": name})

    df = pd.DataFrame(rows).set_index("solvent_id")
    df[_SOLVENT_RAW_COLS] = StandardScaler().fit_transform(
        df[_SOLVENT_RAW_COLS].values
    )
    return df


def _compute_base_sterimol(smiles: str, atom1: int, atom2: int) -> tuple[float, float, float]:
    """Compute Sterimol L, B1, B5 for the basic site of a base fragment.

    Generates one seeded ETKDG+MMFF94 conformer.  If MMFF setup fails (e.g. for
    highly charged anions like carbonate), the raw ETKDG geometry is used —
    conformational variation in Sterimol for bases is small relative to the
    spread across the library, so ETKDG alone is acceptable.

    Args:
        smiles: SMILES of the base fragment (neutral or anionic as appropriate).
        atom1: 1-based index of the reference/origin atom (e.g. C or Si
            bonded to the basic site); defines the measurement axis.
        atom2: 1-based index of the basic atom (N or O) whose substituents
            are measured.

    Returns:
        (L, B1, B5) in Ångströms.
    """
    from morfeus import Sterimol
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # unseeded embedding drifts the column on every rerun
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)  # rc=-1 leaves ETKDG geometry intact
    conf = mol.GetConformer()
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array(conf.GetPositions())
    s = Sterimol(elements, coords, atom1, atom2)
    return s.L_value, s.B_1_value, s.B_5_value


def _build_base_raw(base_ids: list[int], base_names: list[str]) -> pd.DataFrame:
    """Build the z-scored base descriptor table that the blocks are cut from.

    Args:
        base_ids: Integer IDs from the base reference sheet.
        base_names: Corresponding name strings (all 8 expected, so that the
            standardization uses library-wide statistics).

    Returns:
        DataFrame indexed by base_id with ``pa_kJ_mol``, the ``is_inorganic`` /
        ``is_organic`` one-hot pair, the three Sterimol columns, and ``name``.
    """
    from sklearn.preprocessing import StandardScaler

    rows = []
    for bid, name in zip(base_ids, base_names):
        assert name in _BASE_META, (
            f"Base '{name}' has no entry in _BASE_META.  "
            "Add it before proceeding."
        )
        pa, is_org, smi, a1, a2 = _BASE_META[name]
        L, B1, B5 = _compute_base_sterimol(smi, a1, a2)
        rows.append({
            "base_id": bid,
            "name": name,
            "pa_kJ_mol": pa,
            "is_inorganic": float(is_org == 0),
            "is_organic": float(is_org == 1),
            "sterimol_L": L,
            "sterimol_B1": B1,
            "sterimol_B5": B5,
        })

    df = pd.DataFrame(rows).set_index("base_id")
    df[_BASE_CONTINUOUS_COLS] = StandardScaler().fit_transform(
        df[_BASE_CONTINUOUS_COLS].values
    )
    return df


def _standardize_from_bounds(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Standardize to [−1, 1] using explicit design bounds.

    z = 2 * (x − lo) / (hi − lo) − 1

    Args:
        values: 1-D array of raw values.
        lo: Lower design bound.
        hi: Upper design bound.

    Returns:
        Standardized 1-D array.
    """
    return 2.0 * (values - lo) / (hi - lo) - 1.0


_SOLVENT_BLOCK_METHODS = {
    "KT": (
        "Kamlet-Taft solvatochromic parameters: alpha (hydrogen-bond donation), "
        "beta (hydrogen-bond acceptance), pi* (dipolarity/polarizability). "
        "Z-scored across the 14-solvent library. Marcus (1993) Chem. Soc. Rev. 22, 409."
    ),
    "Hansen": (
        "Hansen solubility parameters dD/dP/dH (dispersion, polar, hydrogen "
        "bonding) plus logP, which shares their solubility/partition character. "
        "Z-scored across the 14-solvent library. Hansen (2007), Handbook; logP "
        "from PubChem / Hansch-Leo."
    ),
    "Gutmann": (
        "Gutmann donor number (Lewis basicity towards a cation) and bulk "
        "relative permittivity. Z-scored across the 14-solvent library. "
        "Gutmann (1978), The Donor-Acceptor Approach to Molecular Interactions."
    ),
}

_BASE_BLOCK_METHODS = {
    "thermo": (
        "Gas-phase proton affinity (kJ/mol, z-scored across the 8-base library) "
        "plus a two-column organic/inorganic one-hot. The one-hot is zero in "
        "both columns for the 'no base' rows the assembler zero-fills."
    ),
    "steric": (
        "Sterimol L/B1/B5 around the basic atom, z-scored across the 8-base "
        "library. Computed via morfeus on one ETKDG+MMFF94 conformer of each "
        "base fragment SMILES (anion form for ionic bases). Measurement axis: "
        "reference_atom→basic_atom (N for amines/DBU/KHMDS, O for "
        "alkoxides/carbonates)."
    ),
}


def _write_component_block(
    df: pd.DataFrame,
    out_dir: Path,
    feature_cols: list[str] | None,
    label: str,
    method: str,
) -> None:
    """Write one solvent or base featurization directory.

    Args:
        df: Table indexed by ``solvent_id`` / ``base_id``. A ``name`` column may
            ride along for readability; it is non-numeric and the feature
            assembler drops it.
        out_dir: Destination directory, ``solvent_<block>`` or ``base_<block>``.
        feature_cols: The numeric columns that make up the block, or ``None`` to
            take every column of ``df``.
        label: Short human label for the progress line.
        method: One-paragraph description written into ``metadata.json``.
    """
    cols = feature_cols if feature_cols is not None else list(df.columns)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(json.dumps({
        "n_molecules": int(len(df)),
        "n_features": len(cols),
        "feature_cols": cols,
        "descriptors_file": "descriptors.parquet",
        "method": method,
    }, indent=2))
    print(f"  ✓ {label} ({len(cols)} dims × {len(df)} rows) → {out_dir.relative_to(paths.PKG_ROOT)}")


def run_solvent_base() -> None:
    """Write every solvent block, every base block, and the continuous block.

    Solvent blocks land in ``solvent_<block>/`` and base blocks in
    ``base_<block>/``, matching the ``solvent_``/``base_`` prefix that
    ``pablo.utils.feature_assembly`` and ``pablo.main`` prepend to the names
    listed in ``config.SOLVENT_FEATURIZATIONS`` / ``config.BASE_FEATURIZATIONS``.
    """
    paths.FEAT_DIR.mkdir(parents=True, exist_ok=True)

    solv_df = pd.read_parquet(paths.SOLVENTS_PARQUET)
    base_df = pd.read_parquet(paths.BASES_PARQUET)
    exp_df  = pd.read_parquet(paths.EXPERIMENTS_PARQUET)

    solvent_ids   = solv_df["id"].tolist()
    solvent_names = solv_df["name"].tolist()
    base_ids      = base_df["id"].tolist()
    base_names    = base_df["name"].tolist()

    print("\n[Solvent/Base] Generating solvent, base, and continuous features ...")

    solv_raw = _build_solvent_raw(solvent_ids, solvent_names)
    for block, cols in SOLVENT_BLOCKS.items():
        _write_component_block(
            df=solv_raw[cols + ["name"]],
            out_dir=paths.FEAT_DIR / f"solvent_{block}",
            feature_cols=cols,
            label=f"Solvent {block}",
            method=_SOLVENT_BLOCK_METHODS[block],
        )
    _write_component_block(
        df=_build_identity_onehot(solvent_ids, solvent_names, "solvent_id"),
        out_dir=paths.FEAT_DIR / "solvent_OHE",
        feature_cols=None,
        label="Solvent OHE",
        method=(
            "One-hot identity over the 14-solvent library. No chemistry: the "
            "no-descriptor control every other solvent block has to beat."
        ),
    )

    base_raw = _build_base_raw(base_ids, base_names)
    for block, cols in BASE_BLOCKS.items():
        _write_component_block(
            df=base_raw[cols + ["name"]],
            out_dir=paths.FEAT_DIR / f"base_{block}",
            feature_cols=cols,
            label=f"Base {block}",
            method=_BASE_BLOCK_METHODS[block],
        )
    _write_component_block(
        df=_build_identity_onehot(base_ids, base_names, "base_id"),
        out_dir=paths.FEAT_DIR / "base_OHE",
        feature_cols=None,
        label="Base OHE",
        method=(
            "One-hot identity over the 8-base library. Rows for catalysts that "
            "need no external base are zero-filled at assembly time, so the "
            "all-zero vector is the 'no base' category."
        ),
    )

    out_dir = paths.FEAT_DIR / "continuous"
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_valid = exp_df[exp_df["yield_pct"].notna()].copy()
    cont_df = pd.DataFrame(index=exp_valid.index)
    for col, (lo, hi) in _CONTINUOUS_BOUNDS.items():
        cont_df[f"{col}_std"] = _standardize_from_bounds(
            exp_valid[col].to_numpy(dtype=np.float64), lo, hi
        )
    cont_df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(json.dumps({
        "columns": [f"{c}_std" for c in _CONTINUOUS_BOUNDS],
        "raw_columns": list(_CONTINUOUS_BOUNDS.keys()),
        "bounds": _CONTINUOUS_BOUNDS,
        "standardization": "z = 2*(x - lo)/(hi - lo) - 1  (maps [lo,hi] → [-1,1])",
        "n_experiments": len(cont_df),
    }, indent=2))
    print(f"  ✓ Continuous features (standardized by bounds) → {out_dir.relative_to(paths.PKG_ROOT)}")
    print("[Solvent/Base] Done.")


# ── MAP2 fingerprints for solvent / base ─────────────────────────────────────
# Mirrors pablo/main.py: radius=1 MAP fingerprint on the library.

SOLVENT_BASE_MAP_RADIUS = 1


def _run_map_for_library(
    ids: list[int],
    names: list[str],
    smiles: list[str],
    out_dir: Path,
    index_name: str,
    label: str,
) -> None:
    """Compute raw MAP fingerprints on a small library (solvent or base).

    Writes ``raw_model_output.parquet``, ``descriptors.parquet``, and
    ``metadata.json`` under ``out_dir``, indexed by ``index_name``
    (``solvent_id`` or ``base_id``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{label}] Computing MAP(r={SOLVENT_BASE_MAP_RADIUS}) for {len(smiles)} molecules ...")

    fp_transformer = MAPFingerprint(radius=SOLVENT_BASE_MAP_RADIUS)
    X_raw = np.asarray(fp_transformer.transform(smiles), dtype=np.float64)
    print(f"[{label}] Raw fingerprint shape: {X_raw.shape}")

    raw_cols = [f"map2_raw_{i}" for i in range(X_raw.shape[1])]
    pd.DataFrame(X_raw, index=pd.Index(ids, name=index_name), columns=raw_cols).to_parquet(
        out_dir / "raw_model_output.parquet"
    )
    map_cols = [f"map2_{i}" for i in range(X_raw.shape[1])]
    pd.DataFrame(X_raw, index=pd.Index(ids, name=index_name), columns=map_cols).to_parquet(
        out_dir / "descriptors.parquet"
    )

    meta = {
        "n_molecules": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "label": label,
        "fp_radius": SOLVENT_BASE_MAP_RADIUS,
        "names": names,
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{label}] Done → {out_dir.relative_to(paths.PKG_ROOT)}")


def run_solvent_map() -> None:
    """Raw solvent MAP2 fingerprints on the 14-solvent library."""
    df = pd.read_parquet(paths.SOLVENTS_PARQUET)
    _run_map_for_library(
        ids=df["id"].tolist(),
        names=df["name"].tolist(),
        smiles=df["smiles"].tolist(),
        out_dir=paths.FEAT_DIR / "solvent_map",
        index_name="solvent_id",
        label="Solvent-MAP2",
    )


def run_base_map() -> None:
    """Raw base MAP2 fingerprints on the 8-base library."""
    df = pd.read_parquet(paths.BASES_PARQUET)
    _run_map_for_library(
        ids=df["id"].tolist(),
        names=df["name"].tolist(),
        smiles=df["smiles"].tolist(),
        out_dir=paths.FEAT_DIR / "base_map",
        index_name="base_id",
        label="Base-MAP2",
    )




# ── SOAP for solvent / base ──────────────────────────────────────────────────

COMPONENT_N_CONFORMERS: int = 10
"""ETKDG conformers embedded per solvent or base before the force-field pick."""

ION_CONTACT_MAX_A: float = 4.0
"""Largest closest-approach distance, in Å, allowed between two fragments of a
salt after the xTB optimisation. A larger gap means the ions drifted apart and
the SOAP environment describes free ions, not the ion pair."""

ION_START_DISTANCE_A: float = 2.7
"""Distance, in Å, at which a monatomic counter-ion is placed from its charged
site in the start geometry. xTB relaxes it; the value only has to be a sane
contact distance."""


def _start_geometry(smiles: str) -> tuple[list[str], np.ndarray, list[list[int]], str, int]:
    """Build the pre-xTB geometry of a solvent, a base, or a salt.

    The molecular fragment is embedded with ``COMPONENT_N_CONFORMERS`` seeded
    ETKDGv3 conformers and the lowest force-field energy one is kept (MMFF94
    where it has parameters, UFF otherwise). Each monatomic ion is then placed
    ``ION_START_DISTANCE_A`` beyond its own oppositely charged site, along the
    site's bond direction. Embedding the ions with the molecule instead puts
    unbonded ions on top of each other, since neither ETKDG nor UFF pulls them
    apart.

    Args:
        smiles: Library SMILES, counter-ions included.

    Returns:
        ``(elements, coords, fragments, ff_used, n_conformers)``: symbols and an
        ``(n_atoms, 3)`` array with hydrogens, the atom indices of each fragment,
        the force field used, and the number of conformers embedded.

    Raises:
        AssertionError: If there is not exactly one molecular fragment, it
            cannot be embedded, or an ion has no charged site to sit on.
    """
    from rdkit.Chem import AllChem

    frags = Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True)
    ions = [f for f in frags if f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetFormalCharge() != 0]
    bodies = [f for f in frags if not (f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetFormalCharge() != 0)]
    assert len(bodies) == 1, f"{smiles}: expected one molecular fragment, got {len(bodies)}"

    body = Chem.AddHs(bodies[0])
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = list(AllChem.EmbedMultipleConfs(body, numConfs=COMPONENT_N_CONFORMERS, params=params))
    assert conf_ids, f"ETKDG embedding failed for {smiles}"
    if AllChem.MMFFHasAllMoleculeParams(body):
        ff_used = "mmff94"
        ff_results = AllChem.MMFFOptimizeMoleculeConfs(body, maxIters=2000)
    else:
        ff_used = "uff"
        ff_results = AllChem.UFFOptimizeMoleculeConfs(body, maxIters=2000)
    best = conf_ids[int(np.argmin([energy for _, energy in ff_results]))]

    elements = [atom.GetSymbol() for atom in body.GetAtoms()]
    coords = [row for row in np.asarray(body.GetConformer(best).GetPositions())]
    fragments = [list(range(len(elements)))]

    used_sites: list[int] = []
    for ion in ions:
        atom = ion.GetAtomWithIdx(0)
        sites = [
            a.GetIdx() for a in body.GetAtoms()
            if a.GetFormalCharge() * atom.GetFormalCharge() < 0 and a.GetIdx() not in used_sites
        ]
        assert sites, f"{smiles}: no free oppositely charged site for {atom.GetSymbol()}"
        site = sites[0]
        used_sites.append(site)
        nbrs = [n.GetIdx() for n in body.GetAtomWithIdx(site).GetNeighbors()]
        direction = coords[site] - np.mean([coords[n] for n in nbrs], axis=0)
        direction = direction / np.linalg.norm(direction)
        fragments.append([len(elements)])
        elements.append(atom.GetSymbol())
        coords.append(coords[site] + ION_START_DISTANCE_A * direction)

    return elements, np.asarray(coords), fragments, ff_used, len(conf_ids)


def _component_geometry(smiles: str, out_dir: Path):
    """Optimise one solvent or base with xTB and return its geometry.

    The start geometry comes from ``_start_geometry``. It is optimised with
    GFN2-xTB in ALPB(toluene), the same level as the catalyst geometries. Salts
    are optimised as the full ion pair at net formal charge. ``mol.xyz``,
    ``xtbopt.xyz``, ``xtb.log`` and ``info.json`` stay in ``out_dir``.

    Args:
        smiles: Library SMILES, counter-ions included.
        out_dir: Per-component working directory under
            ``paths.XTB_COMPONENTS_DIR``.

    Returns:
        ASE ``Atoms`` of the optimised geometry, hydrogens included.

    Raises:
        AssertionError: If the xTB optimisation fails, or the fragments of a
            salt end further apart than ``ION_CONTACT_MAX_A``.
    """
    from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
        xyz_text_from_elements_coords,
    )

    elements, coords, fragments, ff_used, n_conformers = _start_geometry(smiles)
    charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mol.xyz").write_text(
        xyz_text_from_elements_coords(elements, coords, f"{smiles} ETKDG+{ff_used.upper()}")
    )
    result = subprocess.run(
        ["xtb", "mol.xyz", "--gfn2", "--opt", "normal", "--chrg", str(charge),
         "--alpb", "toluene", "--norestart"],
        capture_output=True, text=True, timeout=3300, cwd=str(out_dir),
    )
    (out_dir / "xtb.log").write_text(result.stdout + "\nSTDERR:\n" + result.stderr)
    assert result.returncode == 0 and "GEOMETRY OPTIMIZATION CONVERGED" in result.stdout, (
        f"xTB optimisation failed for {smiles}; see {out_dir / 'xtb.log'}"
    )

    atoms = fp.atoms_from_xyz(out_dir / "xtbopt.xyz")
    assert atoms.get_chemical_symbols() == elements, f"xTB reordered the atoms of {smiles}"

    contacts = []
    if len(fragments) > 1:
        pos = atoms.get_positions()
        for frag in fragments:
            others = [i for i in range(len(elements)) if i not in frag]
            gap = np.linalg.norm(pos[frag][:, None, :] - pos[others][None, :, :], axis=-1).min()
            contacts.append(float(gap))
        assert max(contacts) <= ION_CONTACT_MAX_A, (
            f"{smiles}: fragments {max(contacts):.2f} Å apart after optimisation "
            f"(limit {ION_CONTACT_MAX_A} Å)"
        )

    (out_dir / "info.json").write_text(json.dumps({
        "smiles": smiles,
        "net_charge": charge,
        "ff_used": ff_used,
        "n_conformers": n_conformers,
        "n_fragments": len(fragments),
        "fragment_contact_A": contacts,
        "level": "GFN2-xTB / ALPB(toluene), --opt normal",
    }, indent=2))
    return atoms


def _run_soap_for_library(
    library_path: Path, kind: str, index_name: str, label: str
) -> None:
    """Optimise every member of a solvent or base library and write its SOAP.

    Writes ``<kind>_soap/`` with the unnormalised ``raw_model_output.parquet``,
    the L2-normalised ``descriptors.parquet`` and ``metadata.json``, indexed by
    ``index_name``. The species list is the library's own, so the width differs
    from the catalyst SOAP.

    Args:
        library_path: ``paths.SOLVENTS_PARQUET`` or ``paths.BASES_PARQUET``.
        kind: ``"solvent"`` or ``"base"``; the directory prefix.
        index_name: ``"solvent_id"`` or ``"base_id"``.
        label: Human label for progress lines.
    """
    df = pd.read_parquet(library_path)
    print(f"\n[{label}] xTB geometries for {len(df)} molecules ...")
    geometries = {}
    for member_id, name, smiles in zip(df["id"], df["name"], df["smiles"]):
        geometries[int(member_id)] = _component_geometry(
            smiles, paths.XTB_COMPONENTS_DIR / f"{kind}_{int(member_id)}"
        )
        print(f"  ✓ {name}")

    ids, raw, species = fp.soap_matrix(geometries)
    normed = common.l2_normalize(raw)

    out_dir = paths.FEAT_DIR / f"{kind}_soap"
    common.save_feature_matrix(ids, raw, out_dir / "raw_model_output.parquet", "soap_raw",
                               index_name=index_name)
    common.save_feature_matrix(ids, normed, out_dir / "descriptors.parquet", "soap",
                               index_name=index_name)
    (out_dir / "metadata.json").write_text(json.dumps({
        "n_molecules": len(ids),
        "n_features": int(normed.shape[1]),
        "label": label,
        "names": df.set_index("id").loc[ids, "name"].tolist(),
        "raw_model_output_file": "raw_model_output.parquet",
        "descriptors_file": "descriptors.parquet",
        "soap": {
            "r_cut": fp.SOAP_R_CUT,
            "n_max": fp.SOAP_N_MAX,
            "l_max": fp.SOAP_L_MAX,
            "average": "inner",
            "species": species,
        },
        "normalisation": "row-wise L2",
        "geometry_source": (
            "GFN2-xTB / ALPB(toluene) optimised geometry of the library SMILES, "
            "salts as the full ion pair; data/xtb/components/"
            f"{kind}_<id>/xtbopt.xyz"
        ),
    }, indent=2))
    print(f"[{label}] {normed.shape[1]} dims → {out_dir.relative_to(paths.PKG_ROOT)}")


def run_component_soap() -> None:
    """SOAP on xTB geometries of the 14 solvents and the 8 bases."""
    _run_soap_for_library(paths.SOLVENTS_PARQUET, "solvent", "solvent_id", "Solvent-SOAP")
    _run_soap_for_library(paths.BASES_PARQUET, "base", "base_id", "Base-SOAP")


# ── PCA of the learned solvent / base featurizations ─────────────────────────

COMPONENT_PCA_SOURCES: tuple[str, ...] = (
    "solvent_grover", "solvent_soap", "base_grover", "base_soap",
)
"""Full-width solvent/base featurizations reduced to ``<name>_pca/``."""

BASE_PCA_NONE_LOCATION = (
    "No-base rows are zero-filled at assembly. PC scores are centred, so the "
    "zero vector is the centroid of the 8 bases in PC space rather than a point "
    "no base occupies. KS2-KS4 mask the base block through base_active; KS1 "
    "sees no-base rows at that centroid."
)


def run_component_pca() -> None:
    """Frozen PCA-knee reduction of every ``COMPONENT_PCA_SOURCES`` entry.

    Uses ``dim_reduction.build_pca_featurization`` unchanged, fit over the full
    14-solvent or 8-base library. The base reductions record
    ``BASE_PCA_NONE_LOCATION`` in their metadata.
    """
    from pablo.featurization_and_kernel_scan.features.dim_reduction import (
        build_pca_featurization,
    )

    for name in COMPONENT_PCA_SOURCES:
        extra = {"none_location": BASE_PCA_NONE_LOCATION} if name.startswith("base_") else None
        meta = build_pca_featurization(name, extra)
        print(
            f"  ✓ {name}_pca: {meta['source_n_features']} -> {meta['n_features']} pc "
            f"({100 * meta['variance_retained']:.1f}% var)"
        )
