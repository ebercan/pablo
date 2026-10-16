"""Molecular descriptors for the interpretable catalyst featurizations.

Provides:

* **Sterimol / %Vbur block** — LP-axis whole-ligand Sterimol, quadrant-resolved
  %Vbur, and permutation-invariant N/C-block substituent aggregates.
* **Electronics block** — Fukui indices at C2, HOMO weight on C2 (parsed
  directly from xTB molden output), Mulliken charge at C2, and probe-binding
  ΔE for H⁺ and BH₃ adducts.
* **ΔG_deprot block** — Boltzmann-weighted C2 deprotonation free energy from
  independent azolium and carbene conformer ensembles.

The Sterimol, %Vbur, electronics and probe descriptors consume xTB-optimised
free-carbene geometries stored under ``data/xtb_inputs/C<id>/``. ΔG_deprot
embeds its own ETKDG conformers from the precatalyst SMILES and falls back to
the stored geometry only for catalysts ETKDG cannot embed. All intermediate
caches go to ``data/featurizations/_intermediate/``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from pablo.utils.catalyst_form_classifier import (
    classify_library,
    locate_c2_center,
    locate_carbene_center,
    to_azolium,
    to_mono_carbene,
)

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
    HARTREE_TO_KCAL,
    ff_optimize,
    is_polymacrocyclic,
    iter_catalyst_xtb_geometries,
    load_intermediate,
    load_per_catalyst_json,
    parse_total_energy,
    read_charges_at_index,
    read_xyz,
    run_xtb_in_tempdir,
    save_intermediate,
    save_per_catalyst_json,
    xyz_text_from_elements_coords,
)

# ─── Constants ───────────────────────────────────────────────────────────────

VBUR_SPHERE_RADIUS = 3.50
LP_HE_OFFSET_ANGST = 1.5
VBUR_HE_OFFSET_ANGST = 2.0

PROBE_H_DIST = 1.10
PROBE_B_DIST = 1.60
PROBE_BH_DIST = 1.20
BH3_BOND_LENGTH = 1.19

# ─── ΔG_deprot protocol constants ────────────────────────────────────────────
# Fixed library-wide: the descriptor is only meaningful as a relative quantity,
# so every catalyst and both species must use identical settings.

DEPROT_N_CONFORMERS = 20
DEPROT_RMS_PRUNE_ANGST = 0.5
DEPROT_MMFF_WINDOW_KCAL = 10.0
DEPROT_TEMPERATURE_K = 298.15
DEPROT_IMAG_FREQ_CUTOFF_CM1 = -50.0
"""Modes below this are treated as genuine imaginary frequencies. xTB's own
``# imaginary freq.`` counter uses a 5 cm⁻¹ cutoff and flags floppy modes around
−18 cm⁻¹ as significant, which would discard perfectly good conformers."""

DEPROT_DUPLICATE_G_KCAL = 0.1
DEPROT_DUPLICATE_RMSD_ANGST = 0.1
"""Two optimized conformers closer than both thresholds are the same minimum.
They must be collapsed before Boltzmann weighting or that minimum is
double-counted in the partition function."""

GAS_CONSTANT_KCAL = 1.987204259e-3
CH_BOND_MAX_ANGST = 1.35

DEPROT_MIN_CONTACT_ANGST = 0.90
DEPROT_SEED_H_CLEARANCE_ANGST = 1.50
COVALENT_CONTACT_ANGST = 1.75
"""Seed-geometry sanity thresholds. A stored geometry with any interatomic
distance below ``DEPROT_MIN_CONTACT_ANGST`` is not a physical structure; a
constructed azolium whose new C2-H sits within ``DEPROT_SEED_H_CLEARANCE_ANGST``
of any other atom was placed into an occupied region and will relax into an
unrelated basin. ``COVALENT_CONTACT_ANGST`` is the cutoff used to count
heavy-atom neighbours when checking that the carbene centre is still bonded to
exactly its two ring heteroatoms."""

# Canonical Verloop Sterimol values for an H substituent.
# dbstep's distance-based bond perception reaches across the ring for
# single-H substituents, returning inflated values. These literature values
# (Verloop, 1976) keep min_B1 informative as a "thinnest substituent"
# indicator across the multiset.
_H_STERIMOL_VERLOOP = {"L": 2.06, "B1": 1.00, "B5": 1.00}

_AGG_COLS = (
    "n_block_card",
    "n_block_max_L",
    "n_block_sum_L",
    "n_block_max_B5",
    "n_block_min_B1",
    "c_block_card",
    "c_block_max_L",
    "c_block_sum_L",
    "c_block_max_B5",
    "c_block_min_B1",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Sterimol / %Vbur helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _lp_unit_vector(
    coords: np.ndarray, carbene_c: int, hetero_neighbors: list[int]
) -> np.ndarray:
    """Return the lone-pair unit vector at the carbene.

    Points opposite the average direction from the carbene C to its ring
    heteroatom neighbours, i.e. out of the open coordination site.

    Args:
        coords: ``(n_atoms, 3)`` xyz coordinates.
        carbene_c: 0-based index of the carbene C.
        hetero_neighbors: 0-based indices of ring heteroatoms bonded to
            the carbene C.

    Returns:
        Unit vector in 3D pointing along the lone-pair direction.
    """
    c_coord = coords[carbene_c]
    v_sum = np.zeros(3, dtype=np.float64)
    for h_idx in hetero_neighbors:
        v = coords[h_idx] - c_coord
        v_sum += v / np.linalg.norm(v)
    norm = np.linalg.norm(v_sum)
    assert norm > 1e-6, "LP direction degenerate (hetero-neighbour vectors cancel)."
    return -v_sum / norm


def _write_augmented_xyz(elements: list[str], coords: np.ndarray, label: str) -> Path:
    """Write a temporary xyz and return its path. Caller is responsible for unlink.
        Returns:
            Path: Return value of _write_augmented_xyz.
    
    Returns:
        Path: Return value of _write_augmented_xyz.
"""
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
    fh.write(f"{len(elements)}\n{label}\n")
    for el, (x, y, z) in zip(elements, coords):
        fh.write(f"{el}  {float(x):.6f}  {float(y):.6f}  {float(z):.6f}\n")
    fh.close()
    return Path(fh.name)


def _compute_sterimol_from_xyz(
    xyz_path: Path, bond_atom_idx: int, attached_atom_idx: int
) -> dict[str, float]:
    """Compute Sterimol L/B1/B5 via dbstep for a given bond in an xyz file.

    Args:
        xyz_path: Path to the xyz file.
        bond_atom_idx: 1-based index of the bond reference atom (atom1).
        attached_atom_idx: 1-based index of the attached atom (atom2).

    Returns:
        ``{"L": ..., "B1": ..., "B5": ...}`` in Å.
    """
    from dbstep.Dbstep import dbstep as DbstepCalc

    result = DbstepCalc(
        str(xyz_path),
        atom1=bond_atom_idx,
        atom2=attached_atom_idx,
        sterimol=True,
        commandline=True,
    )
    return {"L": result.L, "B1": result.Bmin, "B5": result.Bmax}


# ─── LP-axis Sterimol ───────────────────────────────────────────────────────


def _compute_lp_sterimol_for_catalyst(
    xtbopt_path: Path, mol: Chem.Mol
) -> dict[str, float]:
    """Whole-ligand Sterimol with the reference axis aligned to the lone-pair.

    A ``He`` dummy is placed ``LP_HE_OFFSET_ANGST`` Å from the carbene C
    in the lone-pair direction (where a coordinating metal would sit).
    Sterimol is measured with the dummy as ``atom1`` and the carbene C as
    ``atom2``, so dbstep treats the entire ligand as a substituent on the
    dummy.

    Args:
        xtbopt_path: Path to ``xtbopt.xyz``.
        mol: RDKit Mol matching the xyz atom order.

    Returns:
        Mapping with keys ``lp_L``, ``lp_B1`` and ``lp_B5``, each in Å.
    """
    elements, coords = read_xyz(xtbopt_path)
    carbene_c, hetero_neighbors, _backbone, _ring_atoms = locate_carbene_center(mol)
    assert carbene_c is not None, f"No carbene located for {xtbopt_path}"

    lp_unit = _lp_unit_vector(coords, carbene_c, hetero_neighbors)
    he_coord = coords[carbene_c] + LP_HE_OFFSET_ANGST * lp_unit

    augmented_elements = elements + ["He"]
    augmented_coords = np.vstack([coords, he_coord])
    tmp = _write_augmented_xyz(augmented_elements, augmented_coords, "LP-axis Sterimol")

    he_idx_1based = len(augmented_elements)
    carbene_1based = carbene_c + 1
    sterimol = _compute_sterimol_from_xyz(tmp, he_idx_1based, carbene_1based)
    tmp.unlink()

    return {"lp_L": sterimol["L"], "lp_B1": sterimol["B1"], "lp_B5": sterimol["B5"]}


def generate_lp_sterimol() -> None:
    """Build the LP-axis Sterimol intermediate cache. Resumes from existing parquet."""
    cache_name = "lp_sterimol"
    col_order = ["lp_L", "lp_B1", "lp_B5"]
    results = load_intermediate(cache_name)
    already = set(results.keys())
    if already:
        print(f"  Resuming: {len(already)} catalysts already cached.")

    for cid, xtbopt, mol in iter_catalyst_xtb_geometries():
        if cid in already:
            continue
        results[cid] = _compute_lp_sterimol_for_catalyst(xtbopt, mol)
        r = results[cid]
        print(
            f"  C{cid}: lp_L={r['lp_L']:.3f}  lp_B1={r['lp_B1']:.3f}  lp_B5={r['lp_B5']:.3f}"
        )
        save_intermediate(cache_name, results, col_order)

    out_path = paths.INTERMEDIATE_DIR / f"{cache_name}.parquet"
    print(
        f"\n✓ LP-axis Sterimol ({len(col_order)} cols × {len(results)} rows) → {out_path.relative_to(paths.PKG_ROOT)}"
    )


# ─── Quadrant %Vbur ──────────────────────────────────────────────────────────


def _compute_quadrant_vbur_for_catalyst(
    xtbopt_path: Path, mol: Chem.Mol
) -> dict[str, float]:
    """Quadrant-resolved %Vbur at the carbene center (4 dims).

    A ``He`` dummy is placed 2.0 Å from the carbene C along the LP
    direction (the SambVca metal-position convention). Local frame:
    z from metal to carbene C; xz-plane fixed by the lowest-index ring
    heteroatom (reproducible but **not chirality-canonical**).

    Args:
        xtbopt_path: Path to ``xtbopt.xyz``.
        mol: RDKit Mol matching the xyz atom order.

    Returns:
        Mapping with keys ``vbur_q1`` to ``vbur_q4``, each the mean of the two
        octants above and below that quadrant, in percent.
    """
    from morfeus import BuriedVolume

    elements, coords = read_xyz(xtbopt_path)
    carbene_c, hetero_neighbors, _backbone, _ring_atoms = locate_carbene_center(mol)
    assert carbene_c is not None, f"No carbene located for {xtbopt_path}"

    lp_unit = _lp_unit_vector(coords, carbene_c, hetero_neighbors)
    he_coord = coords[carbene_c] + VBUR_HE_OFFSET_ANGST * lp_unit

    augmented_elements = list(elements) + ["He"]
    augmented_coords = np.vstack([coords, he_coord])
    metal_idx_1based = len(augmented_elements)
    carbene_1based = carbene_c + 1
    plane_neighbor_1based = sorted(hetero_neighbors)[0] + 1

    bv = BuriedVolume(
        augmented_elements,
        augmented_coords,
        metal_idx_1based,
        include_hs=True,
        radius=VBUR_SPHERE_RADIUS,
        z_axis_atoms=[carbene_1based],
        xz_plane_atoms=[plane_neighbor_1based],
    )
    bv.octant_analysis()
    octants = bv.octants["percent_buried_volume"]
    quadrants = {}
    for q_idx, (oct_above, oct_below) in enumerate(
        [(0, 4), (1, 5), (2, 6), (3, 7)], start=1
    ):
        quadrants[f"vbur_q{q_idx}"] = 0.5 * (
            float(octants[oct_above]) + float(octants[oct_below])
        )
    return quadrants


def generate_vbur_quadrants() -> None:
    """Build the quadrant %Vbur intermediate cache. Resumes from existing parquet."""
    cache_name = "vbur_quadrants"
    col_order = ["vbur_q1", "vbur_q2", "vbur_q3", "vbur_q4"]
    results = load_intermediate(cache_name)
    already = set(results.keys())
    if already:
        print(f"  Resuming: {len(already)} catalysts already cached.")

    for cid, xtbopt, mol in iter_catalyst_xtb_geometries():
        if cid in already:
            continue
        results[cid] = _compute_quadrant_vbur_for_catalyst(xtbopt, mol)
        q = results[cid]
        print(
            f"  C{cid}: q1={q['vbur_q1']:.2f}  q2={q['vbur_q2']:.2f}  "
            f"q3={q['vbur_q3']:.2f}  q4={q['vbur_q4']:.2f}"
        )
        save_intermediate(cache_name, results, col_order)

    out_path = paths.INTERMEDIATE_DIR / f"{cache_name}.parquet"
    print(
        f"\n✓ Quadrant %Vbur ({len(col_order)} cols × {len(results)} rows) → {out_path.relative_to(paths.PKG_ROOT)}"
    )


# ─── N/C-block substituent aggregates ────────────────────────────────────────


def _collect_substituent_pairs(
    mol: Chem.Mol, ring_atoms: set[int], carbene_c: int, hetero_neighbors: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Collect (parent_atom, substituent) index pairs for the N- and C-blocks.

    Walks the union of (a) ring atoms other than the carbene C and (b)
    the carbene's heteroatom neighbours. N atoms feed the N-block, backbone
    C atoms the C-block; S/O are skipped. One substituent per parent atom.

    Returns:
        ``(n_pairs, c_pairs)`` — lists of ``(parent_0based, sub_0based)``
        tuples for the N- and C-blocks respectively.
    """
    n_pairs: list[tuple[int, int]] = []
    c_pairs: list[tuple[int, int]] = []
    candidates = (set(ring_atoms) - {carbene_c}) | set(hetero_neighbors)
    for parent_idx in candidates:
        atom = mol.GetAtomWithIdx(parent_idx)
        z = atom.GetAtomicNum()
        if z not in (6, 7):
            continue
        target = n_pairs if z == 7 else c_pairs
        exocyclic = [
            nb.GetIdx()
            for nb in atom.GetNeighbors()
            if nb.GetIdx() != carbene_c and nb.GetIdx() not in ring_atoms
        ]
        if not exocyclic:
            continue
        non_h = [i for i in exocyclic if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        chosen = (non_h or exocyclic)[0]
        target.append((parent_idx, chosen))
    return n_pairs, c_pairs


def _aggregate_triples(
    triples: list[dict[str, float]],
) -> tuple[float, float, float, float, float]:
    """Reduce a list of Sterimol triples to ``(card, max L, sum L, max B5, min B1)``.

    Empty multiset → all zeros.
    
        Returns:
            tuple[float, float, float, float, float]: Return value of _aggregate_triples.
    
    Returns:
        tuple[float, float, float, float, float]: Return value of _aggregate_triples.
"""
    if not triples:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(len(triples)),
        max(t["L"] for t in triples),
        sum(t["L"] for t in triples),
        max(t["B5"] for t in triples),
        min(t["B1"] for t in triples),
    )


def _sterimol_for_pair(
    xtbopt_path: Path, mol: Chem.Mol, ring_idx: int, sub_idx: int
) -> dict[str, float]:
    """Sterimol for one (ring_atom, substituent) pair. Special-cases lone H.
        Returns:
            dict[str, float]: Return value of _sterimol_for_pair.
    
    Returns:
        dict[str, float]: Return value of _sterimol_for_pair.
"""
    if mol.GetAtomWithIdx(sub_idx).GetAtomicNum() == 1:
        return dict(_H_STERIMOL_VERLOOP)
    return _compute_sterimol_from_xyz(xtbopt_path, ring_idx + 1, sub_idx + 1)


def _compute_aggregates_for_catalyst(
    xtbopt_path: Path, mol: Chem.Mol
) -> dict[str, float]:
    """Compute the 10-dim N/C-block aggregate vector for a single catalyst.
        Returns:
            dict[str, float]: Return value of _compute_aggregates_for_catalyst.
    
    Returns:
        dict[str, float]: Return value of _compute_aggregates_for_catalyst.
"""
    carbene_c, hetero_neighbors, _backbone, ring_atoms = locate_carbene_center(mol)
    assert carbene_c is not None, f"No carbene located for {xtbopt_path}"

    n_pairs, c_pairs = _collect_substituent_pairs(
        mol, ring_atoms, carbene_c, hetero_neighbors
    )
    n_triples = [_sterimol_for_pair(xtbopt_path, mol, ri, si) for ri, si in n_pairs]
    c_triples = [_sterimol_for_pair(xtbopt_path, mol, ri, si) for ri, si in c_pairs]

    n_agg = _aggregate_triples(n_triples)
    c_agg = _aggregate_triples(c_triples)
    return dict(zip(_AGG_COLS, n_agg + c_agg))


def generate_nc_aggregates() -> None:
    """Build the N/C-block aggregate intermediate cache (10-dim).

    Resumes from any existing ``_intermediate/n_c_aggregates.parquet``.
    """
    cache_name = "n_c_aggregates"
    results = load_intermediate(cache_name)
    already = set(results.keys())
    if already:
        print(f"  Resuming: {len(already)} catalysts already cached.")
    n_anomalous = sum(1 for cid in already if results[cid]["n_block_card"] == 0)

    for cid, xtbopt, mol in iter_catalyst_xtb_geometries():
        if cid in already:
            continue
        results[cid] = _compute_aggregates_for_catalyst(xtbopt, mol)
        if results[cid]["n_block_card"] == 0:
            n_anomalous += 1
            warnings.warn(
                f"C{cid}: empty N-block — no exocyclic atom on any N "
                f"adjacent to or in the carbene ring. Verify the SMILES.",
                stacklevel=2,
            )
        print(
            f"  C{cid}: |Sn|={int(results[cid]['n_block_card'])} "
            f"max_L_n={results[cid]['n_block_max_L']:.2f}  "
            f"|Sc|={int(results[cid]['c_block_card'])} "
            f"max_L_c={results[cid]['c_block_max_L']:.2f}"
        )
        save_intermediate(cache_name, results, list(_AGG_COLS))

    out_path = paths.INTERMEDIATE_DIR / f"{cache_name}.parquet"
    print(
        f"\n✓ N/C-block aggregates ({len(_AGG_COLS)} cols × {len(results)} rows) → "
        f"{out_path.relative_to(paths.PKG_ROOT)}"
    )
    if n_anomalous:
        print(
            f"  (Anomalous empty N-block: {n_anomalous} catalysts — see warnings above.)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Carbene-localised electronics (Stage 2 — Mechthild)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_homo_weight_on_c2(molden_path: Path, carbene_c_0based: int) -> float:
    """Parse xTB molden directly for the HOMO weight on the carbene C2.

    xTB GFN2 writes Cartesian d/f (6d/10f) and no ``[Core]`` section.  Some
    ``iodata`` versions crash on the missing ``[Core]`` block — this parser
    avoids that entirely.

    Strategy:
    1. Detect ``[5d]``/``[7f]`` markers (absent in xTB → Cartesian d/f assumed).
    2. Walk ``[GTO]`` to count AOs per atom; derive the slice for C2.
    3. Walk ``[MO]`` to find the highest-energy majority-occupied MO
       (``occ >= 1.0``). xTB uses Fermi smearing, so near-degenerate frontier
       orbitals can carry small fractional occupations (e.g. LUMO with
       ``occ ≈ 0.13``); a strict ``occ > 0`` cutoff would mis-select those
       virtual-side orbitals as "the HOMO".
    4. Return sum of squared HOMO coefficients over the C2 AO slice.

    Args:
        molden_path: Path to the xTB molden file.
        carbene_c_0based: 0-based index of the carbene C2 atom.

    Returns:
        Sum of squared HOMO coefficients over the C2 atomic orbitals.
    """
    lines = molden_path.read_text().splitlines()

    sph_d = any(l.strip().lower() == "[5d]" for l in lines)
    sph_f = any(l.strip().lower() == "[7f]" for l in lines)

    _N_AO: dict[str, int] = {
        "s": 1,
        "p": 3,
        "sp": 4,
        "d": 5 if sph_d else 6,
        "f": 7 if sph_f else 10,
        "g": 9,
    }

    sec: dict[str, int] = {}
    for i, line in enumerate(lines):
        t = line.strip().lower()
        if t.startswith("[") and t.endswith("]"):
            sec[t[1:-1]] = i

    def _sec_lines(name: str) -> list[str]:
        s = sec.get(name)
        if s is None:
            return []
        e = min((v for v in sec.values() if v > s), default=len(lines))
        return lines[s + 1 : e]

    # ── [GTO]: count AOs per atom in order ───────────────────────────────
    n_ao_per_atom: list[int] = []
    cur = 0
    in_atom = False

    for line in _sec_lines("gto"):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "****":
            if in_atom:
                n_ao_per_atom.append(cur)
                cur = 0
                in_atom = False
        elif len(parts) >= 2 and parts[0].isdigit() and parts[1] == "0":
            if in_atom:
                n_ao_per_atom.append(cur)
                cur = 0
            in_atom = True
        elif in_atom and len(parts) >= 3 and parts[0].lower() in _N_AO:
            cur += _N_AO[parts[0].lower()]

    if in_atom:
        n_ao_per_atom.append(cur)

    assert carbene_c_0based < len(n_ao_per_atom), (
        f"GTO section yielded {len(n_ao_per_atom)} atoms; "
        f"carbene_idx={carbene_c_0based} is out of range."
    )
    c2_start = sum(n_ao_per_atom[:carbene_c_0based])
    c2_end = c2_start + n_ao_per_atom[carbene_c_0based]

    # ── [MO]: highest-energy majority-occupied MO = HOMO ─────────────────
    last_occ_coeffs: list[float] = []
    cur_occ = 0.0
    cur_coeffs: list[float] = []

    def _flush() -> None:
        if cur_occ >= 1.0 and cur_coeffs:
            last_occ_coeffs[:] = cur_coeffs

    for line in _sec_lines("mo"):
        s = line.strip().lower()
        if s.startswith("ene="):
            _flush()
            cur_occ = 0.0
            cur_coeffs = []
        elif s.startswith("occ") and "=" in s:
            cur_occ = float(line.split("=", 1)[1])
        elif s.startswith("sym=") or s.startswith("spin="):
            pass
        else:
            parts = line.split()
            if len(parts) == 2:
                cur_coeffs.append(float(parts[1]))

    _flush()

    assert last_occ_coeffs, f"No occupied MO found in {molden_path}."
    assert c2_end <= len(last_occ_coeffs), (
        f"C2 AO slice [{c2_start}:{c2_end}] exceeds MO vector length "
        f"{len(last_occ_coeffs)} — GTO/MO atom count mismatch."
    )
    arr = np.array(last_occ_coeffs)
    return float(np.sum(arr[c2_start:c2_end] ** 2))


# ─── Probe adduct geometry builders ─────────────────────────────────────────


def _build_h_plus_adduct_xyz(
    elements: list[str],
    coords: np.ndarray,
    carbene_c: int,
    lp_unit: np.ndarray,
) -> str:
    """Build xyz string for carbene + H⁺ adduct (H placed along LP axis).
        Returns:
            str: Return value of _build_h_plus_adduct_xyz.
    
    Returns:
        str: Return value of _build_h_plus_adduct_xyz.
"""
    h_pos = coords[carbene_c] + PROBE_H_DIST * lp_unit
    new_elements = list(elements) + ["H"]
    new_coords = np.vstack([coords, h_pos.reshape(1, 3)])
    return xyz_text_from_elements_coords(new_elements, new_coords, "H+ adduct")


def _build_bh3_adduct_xyz(
    elements: list[str],
    coords: np.ndarray,
    carbene_c: int,
    lp_unit: np.ndarray,
) -> str:
    """Build xyz string for carbene + BH₃ adduct (B along LP, 3 H's perpendicular).
        Returns:
            str: Return value of _build_bh3_adduct_xyz.
    
    Returns:
        str: Return value of _build_bh3_adduct_xyz.
"""
    b_pos = coords[carbene_c] + PROBE_B_DIST * lp_unit
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(lp_unit, seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(lp_unit, seed)
    u /= np.linalg.norm(u)
    v = np.cross(lp_unit, u)
    v /= np.linalg.norm(v)
    h_positions = []
    for k in range(3):
        angle = k * 2.0 * math.pi / 3.0
        offset = PROBE_BH_DIST * (math.cos(angle) * u + math.sin(angle) * v)
        h_positions.append(b_pos + offset)
    new_elements = list(elements) + ["B", "H", "H", "H"]
    new_coords = np.vstack(
        [coords] + [b_pos.reshape(1, 3)] + [h.reshape(1, 3) for h in h_positions]
    )
    return xyz_text_from_elements_coords(new_elements, new_coords, "BH3 adduct")


def _build_bh3_initial_xyz() -> str:
    """Build xyz string for a free BH₃ molecule (planar, D₃h).
        Returns:
            str: Return value of _build_bh3_initial_xyz.
    
    Returns:
        str: Return value of _build_bh3_initial_xyz.
"""
    b = np.array([0.0, 0.0, 0.0])
    elements = ["B", "H", "H", "H"]
    coords = [b]
    for k in range(3):
        angle = k * 2.0 * math.pi / 3.0
        coords.append(
            b + BH3_BOND_LENGTH * np.array([math.cos(angle), math.sin(angle), 0.0])
        )
    return xyz_text_from_elements_coords(elements, np.array(coords), "free BH3")


# ─── Per-catalyst electronic descriptor computation ──────────────────────────


def _load_catalyst_geometry(
    cat_id: int,
) -> tuple[list[str], np.ndarray, dict, Chem.Mol, int, list[int]]:
    """Load and validate xTB-optimised geometry + topology for a catalyst.

    Returns:
        ``(elements, coords, info_dict, rdkit_mol, carbene_c_idx, hetero_nbr_idxs)``
    """
    mol_dir = paths.XTB_INPUTS_DIR / f"C{cat_id}"
    info = json.loads((mol_dir / "info.json").read_text())
    xtbopt = mol_dir / "xtbopt.xyz"
    assert xtbopt.exists(), f"C{cat_id}: no xtbopt.xyz."
    elements, coords = read_xyz(xtbopt)

    assert "carbene_idx" in info and "hetero_nbr_indices" in info, (
        f"C{cat_id}: info.json is missing 'carbene_idx' or 'hetero_nbr_indices'. "
        "Run `python -m pablo.utils.backfill_carbene_indices` to inject these fields."
    )
    carbene_c = int(info["carbene_idx"])
    hetero_nbrs = [int(i) for i in info["hetero_nbr_indices"]]

    mol = Chem.MolFromSmiles(info["free_carbene_smiles"])
    assert mol is not None, f"C{cat_id}: RDKit parse failed."
    mol = Chem.AddHs(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)
    assert mol.GetNumAtoms() == len(elements), (
        f"C{cat_id}: atom count mismatch — mol from free-carbene SMILES has "
        f"{mol.GetNumAtoms()} atoms but xtbopt.xyz has {len(elements)}."
    )
    assert elements[carbene_c] == "C", (
        f"C{cat_id}: element at carbene_idx={carbene_c} is '{elements[carbene_c]}', "
        "not 'C' — carbene_idx in info.json is inconsistent with xtbopt.xyz."
    )
    for ni in hetero_nbrs:
        assert elements[ni] in ("N", "O", "S"), (
            f"C{cat_id}: element at hetero_nbr_index={ni} is '{elements[ni]}', "
            "expected N/O/S — hetero_nbr_indices in info.json is inconsistent with xtbopt.xyz."
        )
    return elements, coords, info, mol, carbene_c, hetero_nbrs


def _compute_xtb_extra_for_catalyst(cat_id: int) -> dict[str, float]:
    """Compute Fukui indices, HOMO weight, and Mulliken charge at C2.

    Runs three xTB single-point calculations (cation, anion, neutral+molden)
    in temporary directories. Each is cleaned up after extracting the needed
    data. The neutral Mulliken charge is read from the stored
    ``xtb_inputs/C<id>/charges`` file.

    Args:
        cat_id: Catalyst ID.

    Returns:
        Mapping with ``mulliken_charge_c2``, ``fukui_plus_c2`` (q_neutral −
        q_anion), ``fukui_minus_c2`` (q_cation − q_neutral) and
        ``homo_weight_c2``.
    """
    elements, coords, info, mol, carbene_c, _ = _load_catalyst_geometry(cat_id)
    charge = int(info["net_charge"])
    xyz_text = xyz_text_from_elements_coords(elements, coords, f"C{cat_id}")

    q_neutral_c2 = read_charges_at_index(
        paths.XTB_INPUTS_DIR / f"C{cat_id}" / "charges",
        carbene_c,
    )

    stdout_cation, tmpdir_cation, _ = run_xtb_in_tempdir(
        xyz_text,
        charge + 1,
        ["--uhf", "1", "--sp"],
    )
    q_cation_c2 = read_charges_at_index(tmpdir_cation / "charges", carbene_c)
    shutil.rmtree(tmpdir_cation)

    stdout_anion, tmpdir_anion, _ = run_xtb_in_tempdir(
        xyz_text,
        charge - 1,
        ["--uhf", "1", "--sp"],
    )
    q_anion_c2 = read_charges_at_index(tmpdir_anion / "charges", carbene_c)
    shutil.rmtree(tmpdir_anion)

    _, tmpdir_molden, _ = run_xtb_in_tempdir(
        xyz_text,
        charge,
        ["--sp", "--molden"],
    )
    molden_path = tmpdir_molden / "molden.input"
    assert molden_path.exists(), f"C{cat_id}: xTB did not produce molden.input."
    molden_copy = molden_path.with_suffix(".molden")
    shutil.copy(molden_path, molden_copy)
    homo_wt = _compute_homo_weight_on_c2(molden_copy, carbene_c)
    shutil.rmtree(tmpdir_molden)

    return {
        "mulliken_charge_c2": q_neutral_c2,
        "fukui_plus_c2": q_neutral_c2 - q_anion_c2,
        "fukui_minus_c2": q_cation_c2 - q_neutral_c2,
        "homo_weight_c2": homo_wt,
    }


def _compute_probe_h_plus_for_catalyst(cat_id: int) -> dict[str, float]:
    """Compute ΔE for carbene + H⁺ → adduct (kcal/mol).

    Args:
        cat_id: Catalyst ID.

    Returns:
        Mapping with ``delta_e_h_plus`` (E_adduct − E_carbene, kcal/mol) and
        ``e_adduct_h_plus_hartree`` (optimised adduct energy, Hartree).
    """
    elements, coords, info, mol, carbene_c, hetero_nbrs = _load_catalyst_geometry(
        cat_id
    )
    charge = int(info["net_charge"])
    lp = _lp_unit_vector(coords, carbene_c, hetero_nbrs)
    adduct_xyz = _build_h_plus_adduct_xyz(elements, coords, carbene_c, lp)

    stdout, tmpdir, _ = run_xtb_in_tempdir(
        adduct_xyz,
        charge + 1,
        ["--opt", "normal", "--iterations", "500"],
    )
    shutil.rmtree(tmpdir)
    e_adduct = parse_total_energy(stdout)

    e_carbene = parse_total_energy(
        (paths.XTB_OUTPUTS_DIR / f"C{cat_id}.txt").read_text(),
    )
    delta_e = (e_adduct - e_carbene) * HARTREE_TO_KCAL
    return {"delta_e_h_plus": delta_e, "e_adduct_h_plus_hartree": e_adduct}


def _compute_probe_bh3_for_catalyst(cat_id: int) -> dict[str, float]:
    """Compute raw adduct and carbene energies for BH₃ probe binding.

    Args:
        cat_id: Catalyst ID.

    Returns:
        Mapping with ``e_adduct_bh3_hartree`` (optimised adduct) and
        ``e_carbene_hartree`` (free carbene), both in Hartree.
    """
    elements, coords, info, mol, carbene_c, hetero_nbrs = _load_catalyst_geometry(
        cat_id
    )
    charge = int(info["net_charge"])
    lp = _lp_unit_vector(coords, carbene_c, hetero_nbrs)
    adduct_xyz = _build_bh3_adduct_xyz(elements, coords, carbene_c, lp)

    stdout, tmpdir, _ = run_xtb_in_tempdir(
        adduct_xyz,
        charge,
        ["--opt", "normal", "--iterations", "500"],
    )
    shutil.rmtree(tmpdir)
    e_adduct = parse_total_energy(stdout)

    e_carbene = parse_total_energy(
        (paths.XTB_OUTPUTS_DIR / f"C{cat_id}.txt").read_text(),
    )
    return {"e_adduct_bh3_hartree": e_adduct, "e_carbene_hartree": e_carbene}


# ─── ΔG_deprot: conformer ensemble ──────────────────────────────────────────


def _seed_geometry_defects(
    elements: list[str],
    coords: np.ndarray,
    carbene_c: int,
    hetero_nbrs: list[int],
) -> list[str]:
    """Report structural defects in a stored geometry before it seeds a ΔG run.

    The Fibonacci-sphere fallback in the geometry pipeline places heavy atoms
    without regard to connectivity. xTB will happily optimize such a start into
    a converged local minimum whose topology bears no relation to the SMILES —
    a structure that passes every downstream per-conformer QC gate (it converges,
    it has no imaginary mode, C2 carries the right number of H) while being the
    wrong molecule. These checks are the only thing standing between such a
    geometry and a silently corrupted descriptor column.

    Args:
        elements: Element symbols in xyz order.
        coords: ``(n_atoms, 3)`` coordinates in Ångström.
        carbene_c: 0-based index of the carbene carbon.
        hetero_nbrs: 0-based indices of the ring heteroatoms expected to be
            bonded to the carbene carbon. Only their presence is required —
            CAAC and mesoionic centres legitimately carry a carbon as their
            second ring neighbour — but the centre must stay two-coordinate.

    Returns:
        Human-readable defect descriptions; empty when the geometry is sane.
    """
    defects: list[str] = []

    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    d_min = float(d.min())
    if d_min < DEPROT_MIN_CONTACT_ANGST:
        i, j = np.unravel_index(int(d[:, :].argmin()), d.shape)
        defects.append(
            f"interatomic distance {d_min:.2f} A ({elements[i]}{i}-{elements[j]}{j}) "
            f"below {DEPROT_MIN_CONTACT_ANGST} A"
        )

    bonded = [
        i
        for i in range(len(elements))
        if i != carbene_c
        and elements[i] != "H"
        and d[carbene_c, i] <= COVALENT_CONTACT_ANGST
    ]
    got = ", ".join(f"{elements[i]}{i}" for i in sorted(bonded)) or "none"
    missing = sorted(set(hetero_nbrs) - set(bonded))
    if missing:
        want = ", ".join(f"{elements[i]}{i}" for i in missing)
        defects.append(
            f"carbene C{carbene_c} has lost its ring bond to [{want}] "
            f"(bonded to [{got}])"
        )
    if len(bonded) != 2:
        defects.append(
            f"carbene C{carbene_c} is {len(bonded)}-coordinate ([{got}]); a "
            f"carbene centre binds exactly two heavy atoms"
        )

    return defects


def _seed_h_clearance(coords: np.ndarray, h_pos: np.ndarray, carbene_c: int) -> float:
    """Distance from a constructed C2-H to the nearest atom other than C2 itself.

    Args:
        coords: ``(n_atoms, 3)`` coordinates of the parent carbene geometry.
        h_pos: ``(3,)`` position of the proton being added.
        carbene_c: 0-based index of the carbene carbon the proton binds to.

    Returns:
        Nearest-neighbour distance in Ångström.
    """
    d = np.linalg.norm(coords - h_pos, axis=1)
    d[carbene_c] = np.inf
    return float(d.min())


def _etkdg_embeddable(cat_id: int) -> bool:
    """Report whether ETKDG can embed this catalyst at acceptable cost.

    Reuses the force-field record the existing xTB input preparation already
    wrote to ``data/xtb_inputs/ff_used.json``. A catalyst marked ``sphere`` fell
    all the way through that pipeline's embedding cascade to Fibonacci-sphere
    placement, which ignores connectivity — such a geometry is not a starting
    point, it is a failure, and ``_seeded_conformer`` now refuses it outright.

    No catalyst in the current 161-molecule library is marked ``sphere``: the
    one that was, C136, is a doubly-strapped cyclophane that ETKDG cannot embed
    and takes hours to fail on, and it is now routed through plain distance
    geometry instead (``is_polymacrocyclic`` / ``_smiles_to_xyz_pure_dg``).
    This gate stays as a guard for any future catalyst that reaches that rung.

    Args:
        cat_id: Catalyst ID.

    Returns:
        True when the standard ETKDG ensemble should be attempted.
    """
    ff_path = paths.XTB_INPUTS_DIR / "ff_used.json"
    assert ff_path.exists(), f"Missing {ff_path}. Run --xtb first."
    ff_used = json.loads(ff_path.read_text())["per_catalyst"]
    return ff_used.get(str(cat_id), "mmff94") != "sphere"


def _embed_conformers(smiles: str) -> tuple[list[str], int, Chem.Mol]:
    """Embed an MMFF-pruned ETKDGv3 conformer ensemble for one species.

    Cascade, first success wins: plain ETKDGv3 with macrocycle/small-ring
    torsions, then a retry with ``useRandomCoords=True`` — the standard rescue
    for bridged macrocycles, which start poorly from the default coordgen
    coordinates — then the same retry with ``ignoreSmoothingFailures=True``, and
    finally plain distance geometry with the ETKDG torsion terms switched off.
    Molecules carrying two or more macrocycles go straight to that last rung:
    ETKDG cannot embed them at all, and each retry costs minutes before failing.

    Survivors are MMFF94-optimized (UFF fallback) and filtered to a
    ``DEPROT_MMFF_WINDOW_KCAL`` window above the best conformer.

    Args:
        smiles: SMILES of the species (azolium or carbene).

    Returns:
        Tuple of ``(xyz_texts, c2_index, mol)`` where ``xyz_texts`` are
        xyz-format strings in the mol's atom order, ``c2_index`` is the 0-based
        index of the C2 carbon in that same order, and ``mol`` is the H-added
        RDKit ``Mol`` carrying the embedded conformers.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    assert mol is not None, f"RDKit failed to parse species SMILES: {smiles}"
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    # Located before AddHs: RDKit appends explicit H's after the heavy atoms,
    # so the heavy-atom index stays valid in the H-added mol and its xyz.
    c2_index = locate_c2_center(mol)
    assert c2_index is not None, f"No C2 centre located for species: {smiles}"

    mol = Chem.AddHs(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    # (useRandomCoords, ignoreSmoothingFailures, pure distance geometry)
    rungs = [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
    if is_polymacrocyclic(smiles):
        rungs = rungs[-1:]
    for use_random_coords, ignore_smoothing, pure_dg in rungs:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useMacrocycleTorsions = True
        params.useSmallRingTorsions = True
        params.maxIterations = 2000
        params.pruneRmsThresh = DEPROT_RMS_PRUNE_ANGST
        params.useRandomCoords = use_random_coords
        params.ignoreSmoothingFailures = ignore_smoothing
        if pure_dg:
            params.useExpTorsionAnglePrefs = False
            params.useBasicKnowledge = False
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol, numConfs=DEPROT_N_CONFORMERS, params=params
            )
        )
        if conf_ids:
            break
        mol.RemoveAllConformers()

    assert conf_ids, f"ETKDGv3 produced no conformers for species: {smiles}"

    energies: list[tuple[float, int]] = []
    use_mmff = AllChem.MMFFHasAllMoleculeParams(mol)
    props = AllChem.MMFFGetMoleculeProperties(mol) if use_mmff else None
    for cid in conf_ids:
        ff_optimize(mol, conf_id=cid)
        if use_mmff:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        energies.append((ff.CalcEnergy(), cid))

    e_min = min(e for e, _ in energies)
    kept = [cid for e, cid in sorted(energies) if e - e_min <= DEPROT_MMFF_WINDOW_KCAL]

    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz_texts = [
        xyz_text_from_elements_coords(
            elements,
            np.array(mol.GetConformer(cid).GetPositions(), dtype=np.float64),
            f"conformer {cid}",
        )
        for cid in kept
    ]
    return xyz_texts, c2_index, mol


def _seeded_conformer(
    cat_id: int, species: str
) -> tuple[list[str], int, list[str], list[str]]:
    """Build a single conformer from the stored xTB geometry, for ETKDG-hostile catalysts.

    The carbene reuses ``xtb_inputs/C<id>/xtbopt.xyz`` — an already converged
    GFN2/ALPB(toluene) geometry — directly. The azolium adds one H along the C2
    lone-pair axis with the same ``_build_h_plus_adduct_xyz`` helper the existing
    H⁺ probe uses; ``--ohess`` then re-optimizes it properly. The appended H goes
    last, so the carbene atom index is unchanged.

    Both the stored geometry and the constructed azolium are validated before
    use. The stored geometry may come from the Fibonacci-sphere fallback, which
    ignores connectivity; and even a sound carbene geometry can have its
    lone-pair axis pointing into a strap or an ortho substituent, in which case
    the added proton is placed inside another atom and the optimizer relaxes into
    an unrelated basin. Neither failure is visible to the per-conformer QC gates.

    Args:
        cat_id: Catalyst ID.
        species: ``'azolium'`` or ``'carbene'``.

    Returns:
        Tuple of ``(xyz_texts, c2_index, elements, defects)`` with exactly one
        xyz text. ``defects`` is empty when the seed is usable; when it is not,
        the caller must record the failure rather than run xTB on the geometry.
    """
    elements, coords, info, mol, carbene_c, hetero_nbrs = _load_catalyst_geometry(
        cat_id
    )
    defects = _seed_geometry_defects(list(elements), coords, carbene_c, hetero_nbrs)
    if defects:
        return [], carbene_c, list(elements), [f"stored geometry: {d}" for d in defects]

    if species == "carbene":
        xyz = xyz_text_from_elements_coords(elements, coords, "xTB seed (carbene)")
        return [xyz], carbene_c, list(elements), []

    lp = _lp_unit_vector(coords, carbene_c, hetero_nbrs)
    clearance = _seed_h_clearance(
        coords, coords[carbene_c] + PROBE_H_DIST * lp, carbene_c
    )
    if clearance < DEPROT_SEED_H_CLEARANCE_ANGST:
        return (
            [],
            carbene_c,
            list(elements) + ["H"],
            [
                f"constructed C2-H clears the nearest atom by only "
                f"{clearance:.2f} A (< {DEPROT_SEED_H_CLEARANCE_ANGST} A); the "
                f"lone-pair axis is blocked, so this seed cannot relax to the "
                f"azolium minimum"
            ],
        )

    xyz = _build_h_plus_adduct_xyz(elements, coords, carbene_c, lp)
    return [xyz], carbene_c, list(elements) + ["H"], []


# ─── ΔG_deprot: xTB output parsing ──────────────────────────────────────────


def _parse_free_energy(stdout: str) -> float:
    """Extract the last TOTAL FREE ENERGY (Eh) from an xTB ``--ohess`` run.

    Args:
        stdout: Raw xTB stdout.

    Returns:
        Total free energy in Hartree, including the G(RRHO) thermal contribution.
    """
    matches = re.findall(r"\|\s*TOTAL FREE ENERGY\s+([-0-9.]+)\s+Eh", stdout)
    assert matches, "Could not parse TOTAL FREE ENERGY from xTB stdout."
    return float(matches[-1])


def _parse_vibrational_frequencies(stdout: str) -> list[float]:
    """Extract the final projected vibrational frequencies (cm⁻¹) from xTB stdout.

    xTB prints the frequency block more than once during an ``--ohess`` run; only
    the last block corresponds to the converged geometry. The six near-zero
    translation/rotation eigenvalues are dropped.

    Args:
        stdout: Raw xTB stdout.

    Returns:
        Vibrational frequencies in cm⁻¹, in the order xTB reports them.
    """
    blocks = stdout.split("projected vibrational frequencies (cm⁻¹)")
    assert len(blocks) > 1, "No projected vibrational frequency block in xTB stdout."

    freqs: list[float] = []
    for line in blocks[-1].splitlines():
        if not line.strip().startswith("eigval :"):
            if freqs:
                break
            continue
        freqs.extend(float(v) for v in line.split(":", 1)[1].split())

    assert freqs, "Frequency block present but no eigval rows parsed."
    return [f for f in freqs if abs(f) > 1e-6]


def _count_hydrogens_on_atom(
    elements: list[str], coords: np.ndarray, atom_idx: int
) -> int:
    """Count hydrogens covalently bound to one atom in an optimized geometry.

    Purely geometric, so it detects proton migration that happened *during* the
    xTB optimization — which no bookkeeping on the input SMILES could catch.

    Args:
        elements: Element symbols from the optimized xyz.
        coords: ``(n_atoms, 3)`` coordinates in Ångström.
        atom_idx: 0-based index of the atom to probe.

    Returns:
        Number of H atoms within ``CH_BOND_MAX_ANGST`` of the probed atom.
    """
    d = np.linalg.norm(coords - coords[atom_idx], axis=1)
    return int(
        sum(
            1
            for i, el in enumerate(elements)
            if el == "H" and i != atom_idx and d[i] <= CH_BOND_MAX_ANGST
        )
    )


# ─── ΔG_deprot: per-conformer xTB ───────────────────────────────────────────


def _ohess_conformer(
    xyz_text: str, charge: int, c2_index: int, expected_h_on_c2: int
) -> dict:
    """Run ``--ohess`` on one conformer and apply the QC gates.

    Args:
        xyz_text: Input geometry in xyz format.
        charge: Net molecular charge passed to xTB as ``--chrg``.
        c2_index: 0-based index of the C2 carbon in the xyz atom order.
        expected_h_on_c2: 1 for the azolium, 0 for the carbene.

    Returns:
        Dict with ``g_hartree``, ``min_freq_cm1``, ``n_h_on_c2``, ``converged``,
        ``qc_pass``, ``qc_reason``, and the optimized ``coords`` used for the
        duplicate-collapse RMSD. ``g_hartree`` is ``None`` when the run failed QC.
    """
    stdout, tmpdir, rc = run_xtb_in_tempdir(
        xyz_text, charge, ["--ohess", "--iterations", "500"], check=False
    )
    opt_path = tmpdir / "xtbopt.xyz"
    opt_exists = opt_path.exists()
    if opt_exists:
        elements, coords = read_xyz(opt_path)
    shutil.rmtree(tmpdir)

    if rc != 0:
        return {
            "g_hartree": None,
            "min_freq_cm1": None,
            "n_h_on_c2": -1,
            "converged": False,
            "qc_pass": False,
            "qc_reason": f"xtb crashed (rc={rc})",
            "coords": None,
        }

    converged = "GEOMETRY OPTIMIZATION CONVERGED" in stdout
    freqs = _parse_vibrational_frequencies(stdout)
    min_freq = min(freqs) if freqs else 0.0
    n_h = (
        _count_hydrogens_on_atom(elements, coords, c2_index) if opt_exists else -1
    )

    reasons = []
    if not converged:
        reasons.append("optimization did not converge")
    if min_freq < DEPROT_IMAG_FREQ_CUTOFF_CM1:
        reasons.append(f"imaginary mode {min_freq:.1f} cm-1")
    if n_h != expected_h_on_c2:
        reasons.append(f"C2 carries {n_h} H, expected {expected_h_on_c2}")

    qc_pass = not reasons
    return {
        "g_hartree": _parse_free_energy(stdout) if qc_pass else None,
        "min_freq_cm1": min_freq,
        "n_h_on_c2": n_h,
        "converged": converged,
        "qc_pass": qc_pass,
        "qc_reason": "; ".join(reasons),
        "coords": coords if opt_exists else None,
    }


# ─── ΔG_deprot: ensemble aggregation ────────────────────────────────────────


def _heavy_atom_rmsd(
    elements: list[str], coords_a: np.ndarray, coords_b: np.ndarray
) -> float:
    """Kabsch-aligned heavy-atom RMSD between two geometries of the same molecule.

    Atom ordering is shared (both come from the same RDKit mol), so no
    substructure matching is needed.

    Args:
        elements: Element symbols, shared by both geometries.
        coords_a: ``(n_atoms, 3)`` coordinates of the first geometry.
        coords_b: ``(n_atoms, 3)`` coordinates of the second geometry.

    Returns:
        RMSD over heavy atoms in Ångström.
    """
    mask = [i for i, el in enumerate(elements) if el != "H"]
    a = coords_a[mask] - coords_a[mask].mean(axis=0)
    b = coords_b[mask] - coords_b[mask].mean(axis=0)

    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ rot - b) ** 2).sum(axis=1).mean()))


def _collapse_duplicate_conformers(conformers: list[dict]) -> list[dict]:
    """Drop conformers that relaxed onto an already-seen minimum.

    Distinct ETKDG starting points routinely converge to the same xTB minimum.
    Left in place they would each contribute a Boltzmann term, over-weighting
    that minimum in the partition function.

    Args:
        conformers: QC-passing conformer records, each with ``g_hartree`` and
            optionally ``coords`` / ``elements`` for the RMSD tie-break.

    Returns:
        The subset of ``conformers`` representing distinct minima.
    """
    unique: list[dict] = []
    for conf in sorted(conformers, key=lambda c: c["g_hartree"]):
        duplicate = False
        for kept in unique:
            dg = abs(conf["g_hartree"] - kept["g_hartree"]) * HARTREE_TO_KCAL
            if dg > DEPROT_DUPLICATE_G_KCAL:
                continue
            if conf.get("coords") is None or kept.get("coords") is None:
                duplicate = True
                break
            rmsd = _heavy_atom_rmsd(
                conf["elements"],
                np.asarray(conf["coords"]),
                np.asarray(kept["coords"]),
            )
            if rmsd <= DEPROT_DUPLICATE_RMSD_ANGST:
                duplicate = True
                break
        if not duplicate:
            unique.append(conf)
    return unique


def _boltzmann_free_energy(g_hartree: list[float]) -> float:
    """Boltzmann-weight an ensemble of conformer free energies at 298.15 K.

    ``G = -RT ln Σ exp(-(G_i - G_min)/RT)``, offset by ``G_min`` for numerical
    stability.

    Args:
        g_hartree: Per-conformer free energies in Hartree.

    Returns:
        Ensemble free energy in Hartree.
    """
    assert g_hartree, "Cannot Boltzmann-average an empty conformer ensemble."
    g = np.asarray(g_hartree, dtype=np.float64)
    g_min = g.min()
    rel_kcal = (g - g_min) * HARTREE_TO_KCAL
    rt = GAS_CONSTANT_KCAL * DEPROT_TEMPERATURE_K
    g_ens_kcal = -rt * np.log(np.exp(-rel_kcal / rt).sum())
    return float(g_min + g_ens_kcal / HARTREE_TO_KCAL)


# ─── ΔG_deprot: per-species and per-catalyst drivers ────────────────────────


def _species_free_energy(
    smiles: str, charge: int, expected_h_on_c2: int, label: str, cat_id: int
) -> dict:
    """Compute the Boltzmann ensemble free energy of one species.

    Args:
        smiles: Species SMILES (azolium or carbene).
        charge: Net charge passed to xTB.
        expected_h_on_c2: 1 for the azolium, 0 for the carbene.
        label: ``'azolium'`` or ``'carbene'``, for progress messages.
        cat_id: Catalyst ID, for progress messages.

    Returns:
        Dict with ``smiles``, ``charge``, ``conformer_source``, ``conformers``
        (per-conformer records), ``n_conformers_used``, ``seed_defects``, and
        ``g_boltzmann_hartree`` (``None`` if the seed was rejected or every
        conformer failed QC).
    """
    seed_defects: list[str] = []
    if _etkdg_embeddable(cat_id):
        source = "etkdgv3"
        xyz_texts, c2_index, mol = _embed_conformers(smiles)
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
    else:
        source = "xtb_seed"
        xyz_texts, c2_index, elements, seed_defects = _seeded_conformer(cat_id, label)

    if seed_defects:
        print(
            f"  C{cat_id} {label}: SEED REJECTED ({source}) — "
            + "; ".join(seed_defects),
            flush=True,
        )
        return {
            "smiles": smiles,
            "charge": charge,
            "conformer_source": source,
            "conformers": [],
            "n_conformers_embedded": 0,
            "n_conformers_used": 0,
            "seed_defects": seed_defects,
            "g_boltzmann_hartree": None,
        }

    print(
        f"  C{cat_id} {label}: {len(xyz_texts)} conformers ({source}), "
        f"charge {charge:+d}",
        flush=True,
    )

    records: list[dict] = []
    for i, xyz in enumerate(xyz_texts):
        rec = _ohess_conformer(xyz, charge, c2_index, expected_h_on_c2)
        rec["conformer"] = i
        records.append(rec)
        status = "ok" if rec["qc_pass"] else f"REJECTED ({rec['qc_reason']})"
        print(f"    conformer {i}: {status}", flush=True)

    passing = [r for r in records if r["qc_pass"]]
    unique = _collapse_duplicate_conformers(
        [dict(r, elements=elements) for r in passing]
    )
    g_values = [r["g_hartree"] for r in unique]

    kept_ids = {r["conformer"] for r in unique}
    for rec in records:
        rec.pop("coords", None)
        rec["used_in_ensemble"] = rec["conformer"] in kept_ids

    return {
        "smiles": smiles,
        "charge": charge,
        "conformer_source": source,
        "conformers": records,
        "n_conformers_embedded": len(xyz_texts),
        "n_conformers_used": len(unique),
        "seed_defects": seed_defects,
        "g_boltzmann_hartree": _boltzmann_free_energy(g_values) if g_values else None,
    }


def _compute_deltaG_deprot_for_catalyst(cat_id: int) -> dict:
    """Compute ΔG_deprot = G(carbene) − G(azolium) for one catalyst, in kcal/mol.

    Both species are embedded independently — the azolium is *not* derived from
    the carbene geometry — because the two can have different lowest conformers.
    The free proton's free energy is dropped: it is a solvent-referenced constant
    identical for every catalyst, so it cancels in every relative comparison.

    ``charge_state_standard`` records whether the pair sat on the ordinary
    azolium(+1) / carbene(0) ladder. Bis-azolium dications and anionic
    (mesoionic) precursors give a chemically correct first-deprotonation free
    energy, but one displaced by the Coulomb work of separating H⁺ from a
    differently charged species — tens of kcal/mol, which is not basicity. Those
    catalysts stay in the column and are reported at assembly time.

    Args:
        cat_id: Catalyst ID.

    Returns:
        Dict with ``catalyst_id``, ``form``, per-species blocks, the scalar
        ``xtb_deltaG_deprot`` (``None`` on failure), ``charge_state_standard``,
        and a ``failures`` list.
    """
    cat_df = pd.read_parquet(paths.CATALYSTS_PARQUET)
    row = cat_df[cat_df["id"] == cat_id]
    assert not row.empty, f"Catalyst {cat_id} not found in catalysts.parquet."
    smiles = str(row["smiles"].iloc[0])

    form = classify_library([cat_id], [smiles]).get(cat_id, "unknown")
    azolium_smi, azolium_q = to_azolium(smiles, form)
    carbene_smi, carbene_q = to_mono_carbene(smiles, form)
    assert azolium_q - carbene_q == 1, (
        f"C{cat_id}: azolium charge {azolium_q} and carbene charge {carbene_q} "
        "differ by more than one proton."
    )

    azolium = _species_free_energy(azolium_smi, azolium_q, 1, "azolium", cat_id)
    carbene = _species_free_energy(carbene_smi, carbene_q, 0, "carbene", cat_id)

    failures = []
    for name, block in (("azolium", azolium), ("carbene", carbene)):
        if block["g_boltzmann_hartree"] is None:
            if block["seed_defects"]:
                failures.append(f"{name}: seed rejected — " + "; ".join(block["seed_defects"]))
            else:
                failures.append(f"{name}: every conformer failed QC")

    delta = None
    if not failures:
        delta = (
            carbene["g_boltzmann_hartree"] - azolium["g_boltzmann_hartree"]
        ) * HARTREE_TO_KCAL

    return {
        "catalyst_id": cat_id,
        "form": form,
        "precatalyst_smiles": smiles,
        "azolium": azolium,
        "carbene": carbene,
        "xtb_deltaG_deprot": delta,
        "charge_state_standard": (azolium_q, carbene_q) == (1, 0),
        "failures": failures,
    }


# ─── Public generation entry points ─────────────────────────────────────────


def generate_xtb_extra(cat_id: int | None) -> None:
    """Compute Fukui/HOMO-weight/Mulliken for one or all catalysts.

    Args:
        cat_id: Single catalyst ID (SLURM array dispatch), or ``None`` for all.
    """
    if cat_id is not None:
        cat_ids = [cat_id]
    else:
        cat_ids = sorted(
            int(p.parent.name[1:]) for p in paths.XTB_INPUTS_DIR.glob("C*/xtbopt.xyz")
        )
    for cid in cat_ids:
        if load_per_catalyst_json("xtb_extra", cid) is not None:
            print(f"  C{cid}: already cached, skipping.")
            continue
        print(f"  C{cid}: running Fukui SPs + molden ...")
        data = _compute_xtb_extra_for_catalyst(cid)
        save_per_catalyst_json("xtb_extra", cid, data)
        print(
            f"  C{cid}: done — fukui+={data['fukui_plus_c2']:.4f}, "
            f"fukui-={data['fukui_minus_c2']:.4f}, "
            f"HOMO@C2={data['homo_weight_c2']:.4f}"
        )


def generate_probe_h_plus(cat_id: int | None) -> None:
    """Compute H⁺ probe-binding ΔE for one or all catalysts.

    Args:
        cat_id: Single catalyst ID (SLURM array dispatch), or ``None`` for all.
    """
    if cat_id is not None:
        cat_ids = [cat_id]
    else:
        cat_ids = sorted(
            int(p.parent.name[1:]) for p in paths.XTB_INPUTS_DIR.glob("C*/xtbopt.xyz")
        )
    for cid in cat_ids:
        if load_per_catalyst_json("probe_h_plus", cid) is not None:
            print(f"  C{cid}: already cached, skipping.")
            continue
        print(f"  C{cid}: running H+ adduct optimization ...")
        data = _compute_probe_h_plus_for_catalyst(cid)
        save_per_catalyst_json("probe_h_plus", cid, data)
        print(f"  C{cid}: done — ΔE(H+) = {data['delta_e_h_plus']:.2f} kcal/mol")


def generate_probe_bh3(cat_id: int | None) -> None:
    """Compute BH₃ probe-binding energies for one or all catalysts.

    Args:
        cat_id: Single catalyst ID (SLURM array dispatch), or ``None`` for all.
    """
    if cat_id is not None:
        cat_ids = [cat_id]
    else:
        cat_ids = sorted(
            int(p.parent.name[1:]) for p in paths.XTB_INPUTS_DIR.glob("C*/xtbopt.xyz")
        )
    for cid in cat_ids:
        if load_per_catalyst_json("probe_bh3", cid) is not None:
            print(f"  C{cid}: already cached, skipping.")
            continue
        print(f"  C{cid}: running BH3 adduct optimization ...")
        data = _compute_probe_bh3_for_catalyst(cid)
        save_per_catalyst_json("probe_bh3", cid, data)
        print(f"  C{cid}: done — E_adduct = {data['e_adduct_bh3_hartree']:.6f} Eh")


def generate_deltaG_deprot(cat_id: int | None) -> None:
    """Compute the xTB C2-deprotonation free energy for one or all catalysts.

    Per-catalyst results are cached under
    ``_intermediate/deltaG_deprot/C<id>.json`` and already-cached catalysts are
    skipped, so a wall-killed SLURM array task resumes rather than restarts.

    Args:
        cat_id: Single catalyst ID (SLURM array dispatch), or ``None`` for all.
    """
    if cat_id is not None:
        cat_ids = [cat_id]
    else:
        cat_df = pd.read_parquet(paths.CATALYSTS_PARQUET)
        cat_ids = sorted(int(x) for x in cat_df["id"].tolist())

    for cid in cat_ids:
        if load_per_catalyst_json("deltaG_deprot", cid) is not None:
            print(f"  C{cid}: already cached, skipping.")
            continue
        print(f"  C{cid}: running azolium + carbene conformer ensembles ...", flush=True)
        data = _compute_deltaG_deprot_for_catalyst(cid)
        save_per_catalyst_json("deltaG_deprot", cid, data)
        if data["xtb_deltaG_deprot"] is None:
            print(f"  C{cid}: FAILED — {'; '.join(data['failures'])}")
        else:
            print(
                f"  C{cid}: done — ΔG_deprot = {data['xtb_deltaG_deprot']:.2f} kcal/mol "
                f"({data['azolium']['n_conformers_used']} azolium / "
                f"{data['carbene']['n_conformers_used']} carbene conformers)"
            )


def generate_bh3_reference() -> None:
    """Compute and cache the free BH₃ reference energy (one-time)."""
    ref_path = paths.INTERMEDIATE_DIR / "bh3_reference.json"
    if ref_path.exists():
        existing = json.loads(ref_path.read_text())
        print(f"BH3 reference already exists: E = {existing['e_bh3_hartree']:.6f} Eh")
        return
    bh3_xyz = _build_bh3_initial_xyz()
    stdout, tmpdir, _ = run_xtb_in_tempdir(
        bh3_xyz,
        0,
        ["--opt", "normal"],
    )
    shutil.rmtree(tmpdir)
    e_bh3 = parse_total_energy(stdout)
    paths.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps({"e_bh3_hartree": e_bh3}, indent=2))
    print(f"✓ BH3 reference energy: {e_bh3:.6f} Eh → {ref_path.relative_to(paths.PKG_ROOT)}")
