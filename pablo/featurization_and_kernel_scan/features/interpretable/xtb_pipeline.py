"""Geometry generation, xTB execution, and the intermediate descriptor caches.

This is the primitive layer of the interpretable featurization: conformer
generation from SMILES, the xTB subprocess wrappers, output parsing, and the
per-catalyst / per-block caches under ``paths.INTERMEDIATE_DIR``. Both
``descriptors`` and ``assemble`` import from here, and this module imports
neither, so the dependency graph is acyclic.

Names that cross a module boundary are public (``read_xyz``,
``run_xtb_in_tempdir``, ``load_intermediate``, ...); helpers used only here keep
their leading underscore.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from pablo.featurization_and_kernel_scan import paths
from pablo.utils.catalyst_form_classifier import (
    FORM_LABELS,
    classify_library,
    form_to_index,
    form_to_onehot,
    locate_carbene_center,
    to_free_carbene,
)

HOMO_PLAUSIBLE_RANGE = (-11.5, -3.0)
HARTREE_TO_KCAL = 627.5095

XTB_CHECKPOINT = paths.XTB_DIR / "results_checkpoint.parquet"

# ─── Per-catalyst JSON caching ───────────────────────────────────────────────


def save_per_catalyst_json(subdir: str, cat_id: int, data: dict) -> None:
    """Persist a single catalyst's results as JSON under ``_intermediate/<subdir>/``."""
    out_dir = paths.INTERMEDIATE_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"C{cat_id}.json").write_text(json.dumps(data, indent=2))


def load_per_catalyst_json(subdir: str, cat_id: int) -> dict | None:
    """Load a single catalyst's cached JSON, or ``None`` if absent."""
    path = paths.INTERMEDIATE_DIR / subdir / f"C{cat_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_per_catalyst_jsons(subdir: str) -> dict[int, dict]:
    """Collect all per-catalyst JSONs from ``_intermediate/<subdir>/`` into a dict."""
    out_dir = paths.INTERMEDIATE_DIR / subdir
    if not out_dir.exists():
        return {}
    result = {}
    for p in sorted(out_dir.glob("C*.json")):
        cid = int(p.stem[1:])
        result[cid] = json.loads(p.read_text())
    return result


# ─── Parquet intermediate caching ────────────────────────────────────────────


def load_intermediate(name: str) -> dict[int, dict[str, float]]:
    """Load an intermediate parquet from ``_intermediate/<name>.parquet``."""
    path = paths.INTERMEDIATE_DIR / f"{name}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {
        int(cid): {c: float(v) for c, v in row.items()} for cid, row in df.iterrows()
    }


def save_intermediate(
    name: str,
    results: dict[int, dict[str, float]],
    col_order: list[str] | None = None,
) -> None:
    """Persist an intermediate cache to ``_intermediate/<name>.parquet``."""
    paths.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(results, orient="index")
    if col_order is not None:
        df = df[col_order]
    df.index.name = "catalyst_id"
    df.to_parquet(paths.INTERMEDIATE_DIR / f"{name}.parquet")


# ─── Catalyst geometry iterator ──────────────────────────────────────────────


def iter_catalyst_xtb_geometries():
    """Yield ``(cat_id, xtbopt_path, mol)`` for every converged xTB geometry.

    The ``mol`` is built from the free-carbene SMILES in ``info.json`` with
    explicit H's added, so its atom indexing matches the xyz produced by
    ``run_all_xtb``. Catalysts without an ``xtbopt.xyz`` are skipped silently.
    """
    for info_path in sorted(paths.XTB_INPUTS_DIR.glob("C*/info.json")):
        xtbopt = info_path.parent / "xtbopt.xyz"
        if not xtbopt.exists():
            continue
        cid = int(info_path.parent.name[1:])
        info = json.loads(info_path.read_text())
        mol = Chem.MolFromSmiles(info["free_carbene_smiles"])
        if mol is None:
            warnings.warn(f"C{cid}: RDKit parse failed for free-carbene SMILES.")
            continue
        mol = Chem.AddHs(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        yield cid, xtbopt, mol


# ─── xyz I/O ─────────────────────────────────────────────────────────────────


def read_xyz(xyz_path: Path) -> tuple[list[str], np.ndarray]:
    """Read an xyz file and return ``(elements, coords)``."""
    lines = xyz_path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    elements, coords = [], []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, np.array(coords, dtype=np.float64)


def xyz_text_from_elements_coords(
    elements: list[str],
    coords: np.ndarray,
    comment: str = "",
) -> str:
    """Render an xyz string from element symbols and coordinate array."""
    lines = [str(len(elements)), comment]
    for el, (x, y, z) in zip(elements, coords):
        lines.append(f"{el}  {x:.6f}  {y:.6f}  {z:.6f}")
    return "\n".join(lines) + "\n"


# ─── xTB subprocess helpers ─────────────────────────────────────────────────


def run_xtb_in_tempdir(
    xyz_text: str,
    charge: int,
    flags: list[str],
    timeout: int = 3300,
    check: bool = True,
) -> tuple[str, Path, int]:
    """Run xTB in a fresh temp directory and return ``(stdout, tmpdir, returncode)``.

    Caller is responsible for cleaning up ``tmpdir_path`` after extracting
    the needed output files.

    Args:
        xyz_text: Full xyz-format string for the input geometry.
        charge: Net molecular charge.
        flags: Additional xTB command-line flags (e.g. ``['--sp', '--molden']``).
        timeout: Subprocess timeout in seconds.
        check: Assert that xTB exited cleanly. Pass ``False`` for ensemble
            calculations where a single crashing conformer should be recorded
            and dropped rather than aborting the whole molecule — xTB 6.6.0's
            Hessian driver segfaults on some geometries.

    Returns:
        Tuple of ``(stdout_text, tmpdir_path, returncode)``.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="xtb_"))
    xyz_path = tmpdir / "input.xyz"
    xyz_path.write_text(xyz_text)
    result = subprocess.run(
        [
            "xtb",
            "input.xyz",
            "--gfn2",
            "--chrg",
            str(charge),
            "--alpb",
            "toluene",
            "--norestart",
        ]
        + flags,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(tmpdir),
    )
    assert not check or result.returncode == 0, (
        f"xTB failed (rc={result.returncode}).\nSTDERR:\n{result.stderr[-2000:]}"
    )
    return result.stdout, tmpdir, result.returncode


def parse_total_energy(stdout: str) -> float:
    """Extract the last TOTAL ENERGY (Eh) from xTB stdout."""
    matches = re.findall(r"\|\s*TOTAL ENERGY\s+([-0-9.]+)\s+Eh", stdout)
    assert len(matches) > 0, "Could not parse TOTAL ENERGY from xTB stdout."
    return float(matches[-1])


def read_charges_at_index(charges_path: Path, idx_0based: int) -> float:
    """Read the Mulliken charge at a specific atom index from an xTB charges file."""
    lines = charges_path.read_text().splitlines()
    assert idx_0based < len(
        lines
    ), f"Atom index {idx_0based} out of range ({len(lines)} atoms) in {charges_path}."
    return float(lines[idx_0based])


def _parse_xtb_output(stdout: str) -> dict[str, float] | None:
    """Extract HOMO/LUMO/gap/dipole/polarizability from xTB stdout.

    Returns:
        Dict with keys ``xtb_homo_eV``, ``xtb_lumo_eV``, ``xtb_gap_eV``,
        ``xtb_dipole_D``, ``xtb_polarizability_angst3``, or ``None`` if
        not all five could be parsed.
    """
    marker = "GEOMETRY OPTIMIZATION CONVERGED"
    if marker not in stdout:
        return None
    # xTB prints a full property block for the *input* geometry before the
    # optimization starts, so a first-match search over the whole log returns
    # HOMO/LUMO/gap at the unrelaxed force-field geometry. Parse only the
    # segment written after the final converged optimization.
    converged = stdout[stdout.rfind(marker) :]
    result: dict[str, float] = {}

    m = re.search(r"HOMO-LUMO [Gg][Aa][Pp]\s+([0-9.\-]+)\s+eV", converged)
    if m:
        result["xtb_gap_eV"] = float(m.group(1))

    m = re.search(r"([0-9.\-]+)\s+\(HOMO\)", converged)
    if m:
        result["xtb_homo_eV"] = float(m.group(1))

    m = re.search(r"([0-9.\-]+)\s+\(LUMO\)", converged)
    if m:
        result["xtb_lumo_eV"] = float(m.group(1))

    m = re.search(r"full:\s+[0-9.\-]+\s+[0-9.\-]+\s+[0-9.\-]+\s+([0-9.]+)", converged)
    if m:
        result["xtb_dipole_D"] = float(m.group(1))

    m = re.search(r"Mol\. α\(0\).*?:\s+([0-9.]+)", converged)
    if m:
        # Bohr^3 → Å^3 conversion: (0.5291772)^3 ≈ 0.14818471.
        result["xtb_polarizability_angst3"] = float(m.group(1)) * 0.14818471

    return result if len(result) == 5 else None


# ─── RDKit conformer embedding ───────────────────────────────────────────────


def ff_optimize(mol, conf_id: int = -1) -> str:
    """Optimize the active conformer with MMFF94; fall back to UFF on missing params.

    Args:
        mol: RDKit Mol with at least one 3D conformer.
        conf_id: Conformer id to optimize. -1 = active.

    Returns:
        'mmff94' if MMFF94 was applied, 'uff' otherwise.
    """
    if AllChem.MMFFHasAllMoleculeParams(mol):
        status = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
        if status != -1:
            return "mmff94"
    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
    return "uff"


def _best_conformer_id(mol, prefer_mmff: bool = True) -> int:
    """Return the conformer id with the lowest MMFF94 (or UFF) energy."""
    use_mmff = prefer_mmff and AllChem.MMFFHasAllMoleculeParams(mol)
    props = AllChem.MMFFGetMoleculeProperties(mol) if use_mmff else None
    best_id, best_e = mol.GetConformers()[0].GetId(), float("inf")
    for conf in mol.GetConformers():
        cid = conf.GetId()
        if use_mmff:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        e = ff.CalcEnergy() if ff else float("inf")
        if e < best_e:
            best_e, best_id = e, cid
    return best_id


def _xyz_from_conformer(
    mol, conf_id: int, cat_id: int, label: str, skip_atom_idx: int | None = None
) -> str:
    """Render an xyz string from a chosen conformer, optionally dropping one atom."""
    conf = mol.GetConformer(conf_id)
    atom_lines = []
    for i in range(mol.GetNumAtoms()):
        if i == skip_atom_idx:
            continue
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        pos = conf.GetAtomPosition(i)
        atom_lines.append(f"{sym}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
    header = [str(len(atom_lines)), f"C{cat_id} {label} geometry"]
    return "\n".join(header + atom_lines) + "\n"


def _smiles_to_xyz(smiles: str, cat_id: int) -> tuple[str, str, int, list[int]] | None:
    """Build a 3D xyz string from SMILES via RDKit ETKDG + MMFF94/UFF.

    Embedding strategy (first success wins):
    1. ETKDGv3 single conformer.
    2. ETKDGv3 with ``useMacrocycleTorsions=True`` + 10 conformers, picking
       the lowest-MMFF (or UFF) energy conformer. Handles large-ring NHCs.

    Returns:
        ``(xyz_string, ff_used, carbene_xyz_idx, hetero_nbr_xyz_idxs)`` where
        the index fields give the 0-based atom positions in the returned xyz
        string. Returns ``None`` if no real 3D conformer could be embedded.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"C{cat_id}: RDKit parse failed for SMILES: {smiles}")
        return None

    mol = Chem.AddHs(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    carbene_c, hetero_nbrs, _, _ = locate_carbene_center(mol)
    assert carbene_c is not None, (
        f"C{cat_id}: locate_carbene_center failed on free-carbene SMILES: {smiles}"
    )

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 200
    if AllChem.EmbedMolecule(mol, params) == 0:
        ff_used = ff_optimize(mol)
        return (
            _xyz_from_conformer(
                mol, mol.GetConformer().GetId(), cat_id, f"ETKDG+{ff_used.upper()}"
            ),
            ff_used,
            carbene_c,
            hetero_nbrs,
        )

    mac_params = AllChem.ETKDGv3()
    mac_params.randomSeed = 42
    mac_params.useMacrocycleTorsions = True
    mac_params.useSmallRingTorsions = True
    mac_params.maxIterations = 200
    confs = list(AllChem.EmbedMultipleConfs(mol, numConfs=10, params=mac_params))
    if not confs:
        return None

    use_mmff = AllChem.MMFFHasAllMoleculeParams(mol)
    if use_mmff:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        ff_used = "mmff94"
    else:
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500)
        ff_used = "uff"
    best = _best_conformer_id(mol, prefer_mmff=use_mmff)
    return (
        _xyz_from_conformer(mol, best, cat_id, f"ETKDG-mac+{ff_used.upper()}"),
        ff_used,
        carbene_c,
        hetero_nbrs,
    )


def _find_salt_carbene_c(mol) -> int | None:
    """Find the prospective carbene C in an imidazolium / triazolium / thiazolium salt.

    The C2 of an NHC salt is a ring carbon flanked by ≥2 ring nitrogens (or
    1 ring N + 1 ring S for thiazoliums) that still carries an explicit H —
    the H that gets removed on deprotonation to the free carbene.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or not atom.IsInRing():
            continue
        ring_hetero = sum(
            1
            for nb in atom.GetNeighbors()
            if nb.GetAtomicNum() in (7, 16) and nb.IsInRing()
        )
        if ring_hetero < 2:
            continue
        if any(nb.GetAtomicNum() == 1 for nb in atom.GetNeighbors()):
            return atom.GetIdx()
    return None


def _smiles_to_xyz_via_salt(
    salt_smiles: str, cat_id: int
) -> tuple[str, str, int, list[int]] | None:
    """Embed the salt cation, then strip the C2-H to recover free-carbene xyz.

    Used when ETKDG fails on the free-carbene SMILES because the divalent
    ``[C]`` carbene atom has no well-defined distance/torsion bounds. The
    salt is fully valence-saturated and embeds cleanly; removing one H from
    the C2 costs negligible geometric error — xTB will relax it.

    Returns ``None`` if the salt cation can't be embedded or no C2-H is
    locatable (e.g. CO₂-adduct betaines have CO₂ on C2, not H).

    The returned ``carbene_xyz_idx`` and ``hetero_nbr_xyz_idxs`` are 0-based
    indices in the returned xyz string (i.e. the salt-mol indices shifted to
    account for the removed H atom).
    """
    mol = Chem.MolFromSmiles(salt_smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    mol = Chem.AddHs(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    mac_params = AllChem.ETKDGv3()
    mac_params.randomSeed = 42
    mac_params.useMacrocycleTorsions = True
    mac_params.useSmallRingTorsions = True
    mac_params.maxIterations = 200
    confs = list(AllChem.EmbedMultipleConfs(mol, numConfs=10, params=mac_params))
    if not confs:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    if mol.GetNumConformers() == 0:
        return None

    use_mmff = AllChem.MMFFHasAllMoleculeParams(mol)
    if use_mmff:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        ff_used = "mmff94"
    else:
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500)
        ff_used = "uff"
    best = _best_conformer_id(mol, prefer_mmff=use_mmff)

    c2_idx = _find_salt_carbene_c(mol)
    if c2_idx is None:
        return None
    h_to_remove = next(
        (
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(c2_idx).GetNeighbors()
            if nb.GetAtomicNum() == 1
        ),
        None,
    )
    if h_to_remove is None:
        return None

    hetero_nbrs_mol = [
        nb.GetIdx()
        for nb in mol.GetAtomWithIdx(c2_idx).GetNeighbors()
        if nb.GetAtomicNum() in (7, 8, 16) and nb.IsInRing()
    ]
    assert hetero_nbrs_mol, f"C{cat_id}: no ring heteroatom neighbors found at C2 in salt."

    carbene_xyz_idx = c2_idx if c2_idx < h_to_remove else c2_idx - 1
    hetero_xyz_idxs = [i if i < h_to_remove else i - 1 for i in hetero_nbrs_mol]

    xyz = _xyz_from_conformer(
        mol,
        best,
        cat_id,
        f"ETKDG-mac+{ff_used.upper()} (via salt, C2-H stripped)",
        skip_atom_idx=h_to_remove,
    )
    return xyz, ff_used, carbene_xyz_idx, hetero_xyz_idxs


MACROCYCLE_MIN_RING = 10
"""Ring size at or above which an SSSR ring counts as a macrocycle."""


def is_polymacrocyclic(smiles: str) -> bool:
    """Report whether a molecule carries two or more independent macrocycles.

    ETKDG's torsion-preference machinery is what makes a doubly-strapped
    cyclophane pathological: the bounds matrix cannot be smoothed, and each of
    the 200 retries re-pays the full setup cost before failing, so the embedding
    runs for hours and then falls through to the sphere fallback anyway. A
    single macrocycle is fine — the library's 12-membered mono-strapped
    catalysts embed normally — so the gate is deliberately set at two.

    Args:
        smiles: SMILES to inspect.

    Returns:
        True when the molecule should skip straight to plain distance geometry.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)
    big = [r for r in mol.GetRingInfo().AtomRings() if len(r) >= MACROCYCLE_MIN_RING]
    return len(big) >= 2


def _smiles_to_xyz_pure_dg(
    smiles: str, cat_id: int
) -> tuple[str, str, int, list[int]] | None:
    """Embed with plain distance geometry, for molecules ETKDG cannot handle.

    ETKDG's experimental-torsion and basic-knowledge terms are what stall on a
    bridged cyclophane: the bounds matrix cannot be triangle-smoothed, and every
    retry re-pays that cost before failing. Switching both terms off, seeding
    from random coordinates and accepting the smoothing failure embeds the same
    molecule in a fraction of a second. The result is a cruder starting geometry
    than ETKDG would give, so it is MMFF/UFF-relaxed before xTB sees it — but it
    is a chemically connected structure, which is the one thing the sphere
    fallback below cannot promise.

    Args:
        smiles: Free-carbene SMILES.
        cat_id: Catalyst ID, for error messages.

    Returns:
        ``(xyz_string, ff_used, carbene_xyz_idx, hetero_nbr_xyz_idxs)``, or
        ``None`` if even plain distance geometry fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    carbene_c, hetero_nbrs, _, _ = locate_carbene_center(mol)
    assert carbene_c is not None, (
        f"C{cat_id}: locate_carbene_center failed on free-carbene SMILES "
        f"(pure-DG path): {smiles}"
    )

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 200
    params.useExpTorsionAnglePrefs = False
    params.useBasicKnowledge = False
    params.useRandomCoords = True
    params.ignoreSmoothingFailures = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None

    ff_used = ff_optimize(mol)
    return (
        _xyz_from_conformer(mol, mol.GetConformer().GetId(), cat_id, f"pureDG+{ff_used.upper()}"),
        ff_used,
        carbene_c,
        hetero_nbrs,
    )


def _smiles_to_xyz_sphere_fallback(
    smiles: str, cat_id: int
) -> tuple[str, str, int, list[int]] | None:
    """Last-resort 3D geometry: place all atoms, explicit H included, on a Fibonacci sphere.

    No force field is applied; the returned force-field label is ``"sphere"``.
    Used when all ETKDG-based embedding fails (e.g. fused macrocycles). The
    sphere placement is physically meaningless but gives xTB a non-flat starting
    point. The GFN1 crude pre-relaxation in _run_single_xtb then fixes the geometry
    before the GFN2 optimization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    mol = Chem.AddHs(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    carbene_c, hetero_nbrs, _, _ = locate_carbene_center(mol)
    assert carbene_c is not None, (
        f"C{cat_id}: locate_carbene_center failed on free-carbene SMILES (sphere path): {smiles}"
    )

    n = mol.GetNumAtoms()
    conf = Chem.Conformer(n)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    radius = 3.0 + 0.1 * n ** (1 / 3)
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        x, z = r * np.cos(theta), r * np.sin(theta)
        conf.SetAtomPosition(
            i, (float(radius * x), float(radius * y), float(radius * z))
        )
    mol.AddConformer(conf, assignId=True)

    ff_used = "sphere"
    lines = [str(n), f"C{cat_id} sphere-placed geometry (macrocycle fallback)"]
    c = mol.GetConformer()
    for i in range(n):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        pos = c.GetAtomPosition(i)
        lines.append(f"{sym}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
    return "\n".join(lines) + "\n", ff_used, carbene_c, hetero_nbrs


# ─── xTB input preparation and execution ────────────────────────────────────


def _prepare_xtb_inputs() -> list[int]:
    """Write one xyz + info.json per catalyst under ``paths.XTB_INPUTS_DIR / "C<id>"``.

    Also writes ``paths.XTB_INPUTS_DIR / "ff_used.json"`` (``data/xtb/inputs/``) reporting the force field
    actually applied for each catalyst (mmff94, uff, or sphere).

    Molecules carrying two or more macrocycles skip rungs 1-2 entirely
    (``is_polymacrocyclic``): ETKDG cannot embed them and takes hours to say so.

    Embedding cascade (first success wins):
    1. ``_smiles_to_xyz(free_smi)`` — direct ETKDG on free-carbene SMILES,
       single conformer then macrocycle multi-conf.
    2. ``_smiles_to_xyz_via_salt(smi)`` — embed the salt cation cleanly (no
       ``[C]`` valence trouble), strip the C2-H to recover free-carbene xyz.
       Handles macrocyclic / fused-ring salts where free-carbene embedding
       fails outright.
    3. ``_smiles_to_xyz_pure_dg(free_smi)`` — plain distance geometry with the
       ETKDG torsion terms switched off. Embeds bridged cyclophanes that stall
       ETKDG, and unlike the sphere fallback it returns a connected structure.
    4. ``_smiles_to_xyz_sphere_fallback(free_smi)`` — last-resort Fibonacci
       sphere placement. xTB's GFN1 crude pre-relax escapes the placement.
       Reaching this rung means the geometry is not trustworthy: it ignores
       connectivity entirely, and xTB will converge it to whatever knot lies
       nearest. Treat any catalyst marked ``sphere`` as unresolved.

    Returns:
        List of catalyst IDs that have a usable input geometry.
    """
    cat_df = pd.read_parquet(paths.CATALYSTS_PARQUET)
    cat_ids = [int(x) for x in cat_df["id"].tolist()]
    smiles_list = cat_df["smiles"].tolist()
    form_map = classify_library(cat_ids, smiles_list)

    paths.XTB_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    runnable: list[int] = []
    ff_used_map: dict[int, str] = {}
    for cid, smi in zip(cat_ids, smiles_list):
        mol_dir = paths.XTB_INPUTS_DIR / f"C{cid}"
        mol_dir.mkdir(exist_ok=True)
        form = form_map.get(cid, "unknown")
        free_smi, net_charge = to_free_carbene(smi, form)
        if is_polymacrocyclic(free_smi):
            result = _smiles_to_xyz_pure_dg(free_smi, cid) or _smiles_to_xyz_sphere_fallback(
                free_smi, cid
            )
        else:
            result = (
                _smiles_to_xyz(free_smi, cid)
                or _smiles_to_xyz_via_salt(smi, cid)
                or _smiles_to_xyz_pure_dg(free_smi, cid)
                or _smiles_to_xyz_sphere_fallback(free_smi, cid)
            )
        if result is None:
            warnings.warn(f"C{cid}: no 3D geometry — skipping xTB.")
            continue
        xyz, ff_used, carbene_idx, hetero_nbr_idxs = result
        (mol_dir / "mol.xyz").write_text(xyz)
        (mol_dir / "info.json").write_text(
            json.dumps(
                {
                    "catalyst_id": cid,
                    "form": form,
                    "free_carbene_smiles": free_smi,
                    "net_charge": net_charge,
                    "ff_used": ff_used,
                    "carbene_idx": carbene_idx,
                    "hetero_nbr_indices": hetero_nbr_idxs,
                },
                indent=2,
            )
        )
        ff_used_map[cid] = ff_used
        runnable.append(cid)

    ff_counts: dict[str, int] = {}
    for ff in ff_used_map.values():
        ff_counts[ff] = ff_counts.get(ff, 0) + 1
    (paths.XTB_INPUTS_DIR / "ff_used.json").write_text(
        json.dumps(
            {
                "per_catalyst": ff_used_map,
                "summary": {
                    "n_total": len(ff_used_map),
                    "counts": ff_counts,
                },
            },
            indent=2,
        )
    )

    print(f"Prepared {len(runnable)} / {len(cat_ids)} xTB input directories.")
    print(f"  Force-field usage: {ff_counts}")
    return runnable


def _run_single_xtb(cat_id: int) -> bool:
    """Run xTB GFN2 + ALPB(toluene) optimization for one catalyst. Returns success.

    For catalysts whose geometry was flagged as ``sphere`` or ``uff`` in
    ``info.json``, a GFN1 crude pre-relaxation is run first to escape the
    worst strain before the full GFN2 optimization.
    """
    mol_dir = paths.XTB_INPUTS_DIR / f"C{cat_id}"
    xyz_path = mol_dir / "mol.xyz"
    info_path = mol_dir / "info.json"
    assert xyz_path.exists() and info_path.exists(), f"C{cat_id}: missing inputs."

    paths.XTB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_log = paths.XTB_OUTPUTS_DIR / f"C{cat_id}.txt"
    if out_log.exists() and "GEOMETRY OPTIMIZATION CONVERGED" in out_log.read_text():
        return True

    info = json.loads(info_path.read_text())
    charge = int(info["net_charge"])
    ff_used = info.get("ff_used", "mmff94")

    def _xtb(
        xyz: Path, method: str, opt_level: str, extra: list[str] | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "xtb",
                str(xyz),
                f"--{method}",
                "--opt",
                opt_level,
                "--chrg",
                str(charge),
                "--alpb",
                "toluene",
                "--norestart",
            ]
            + (extra or []),
            capture_output=True,
            text=True,
            timeout=3300,
            cwd=str(mol_dir),
        )

    log_text = ""

    if ff_used in ("sphere", "uff"):
        pre1 = _xtb(xyz_path, "gfn1", "crude")
        log_text += (
            "=== GFN1 crude pre-relaxation ===\n"
            + pre1.stdout
            + "\nSTDERR:\n"
            + pre1.stderr
            + "\n"
        )
        relaxed = mol_dir / "xtbopt.xyz"
        after_gfn1 = relaxed if relaxed.exists() else xyz_path
        pre2 = _xtb(
            after_gfn1,
            "gfn2",
            "crude",
            extra=["--etemp", "5000", "--iterations", "500"],
        )
        log_text += (
            "=== GFN2 crude pre-relaxation ===\n"
            + pre2.stdout
            + "\nSTDERR:\n"
            + pre2.stderr
            + "\n"
        )
        input_xyz = relaxed if relaxed.exists() else after_gfn1
    else:
        input_xyz = xyz_path

    extra = ["--iterations", "500"] if ff_used in ("sphere", "uff") else []
    result = _xtb(input_xyz, "gfn2", "normal", extra=extra)
    log_text += (
        "=== GFN2 normal optimization ===\n"
        + result.stdout
        + "\nSTDERR:\n"
        + result.stderr
    )
    out_log.write_text(log_text)
    return result.returncode == 0 and "GEOMETRY OPTIMIZATION CONVERGED" in result.stdout


def _load_xtb_checkpoint() -> dict[int, dict]:
    """Load previously saved xTB results from the checkpoint parquet."""
    if not XTB_CHECKPOINT.exists():
        return {}
    df = pd.read_parquet(XTB_CHECKPOINT)
    return {int(cid): row.to_dict() for cid, row in df.iterrows()}


def _save_xtb_checkpoint(results: dict[int, dict]) -> None:
    """Persist current xTB results to checkpoint parquet."""
    paths.XTB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "catalyst_id"
    df.to_parquet(XTB_CHECKPOINT)


def run_all_xtb() -> None:
    """Run xTB on every catalyst sequentially, saving results after each molecule.

    Results are written to ``data/xtb/results_checkpoint.parquet`` (``XTB_CHECKPOINT``)
    after every successful convergence, so a mid-run interruption loses at most
    one molecule's work. Already-saved catalysts are skipped on re-run.
    """
    runnable = _prepare_xtb_inputs()
    results = _load_xtb_checkpoint()
    already_done = set(results.keys())
    if already_done:
        print(
            f"  Resuming: {len(already_done)} already saved, {len(runnable) - len(already_done & set(runnable))} remaining."
        )

    n_ok = n_fail = 0
    failed: list[int] = []
    for i, cid in enumerate(runnable, 1):
        if cid in already_done:
            n_ok += 1
            print(f"[{i}/{len(runnable)}] C{cid}: already done (skipped).")
            continue
        ok = _run_single_xtb(cid)
        if ok:
            log_path = paths.XTB_OUTPUTS_DIR / f"C{cid}.txt"
            parsed = _parse_xtb_output(log_path.read_text())
            if parsed is not None:
                results[cid] = parsed
                _save_xtb_checkpoint(results)
            n_ok += 1
            print(f"[{i}/{len(runnable)}] C{cid}: converged.")
        else:
            n_fail += 1
            failed.append(cid)
            print(f"[{i}/{len(runnable)}] C{cid}: FAILED.")

    print(f"\nxTB done: {n_ok} ok, {n_fail} failed.")
    if failed:
        print(f"  Failed IDs: {failed}")
    if n_fail > 5:
        warnings.warn(
            f"[FLAG] {n_fail} molecules failed xTB optimization. Report to supervisor "
            "before relying on these descriptors.",
            stacklevel=2,
        )


# ─── Structural one-hots and ring-type classification ───────────────────────

_RING_SMARTS = {
    "triazolinylidene": Chem.MolFromSmarts("[#7]1~[#7]~[#6]~[#7]~[#6]1"),
    "thiazolylidene": Chem.MolFromSmarts("[#7]1~[#6]~[#6]~[#16]~[#6]1"),
}

RING_TYPE_LABELS = (
    "imidazolylidene",
    "imidazolinylidene",
    "triazolinylidene",
    "thiazolylidene",
    "other",
)

STRUCTURAL_FEATURE_COLS = [
    "cat_form_id",
    "ring_type_id",
    "backbone_saturated",
    "mw",
    "heavy_atom_count",
]


def classify_ring_type(smiles: str) -> str:
    """Detect the NHC ring type from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "other"
    if mol.HasSubstructMatch(_RING_SMARTS["thiazolylidene"]):
        return "thiazolylidene"
    if mol.HasSubstructMatch(_RING_SMARTS["triazolinylidene"]):
        return "triazolinylidene"

    found_2n_ring = False
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if sum(a.GetAtomicNum() == 7 for a in atoms) < 2:
            continue
        found_2n_ring = True
        ring_set = set(ring)
        for bond in mol.GetBonds():
            if (
                bond.GetBeginAtomIdx() in ring_set
                and bond.GetEndAtomIdx() in ring_set
                and bond.GetBondTypeAsDouble() >= 1.5
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6
            ):
                return "imidazolylidene"
    if found_2n_ring:
        return "imidazolinylidene"
    return "other"


def _ring_type_onehot(ring_type: str) -> list[int]:
    vec = [0] * len(RING_TYPE_LABELS)
    if ring_type in RING_TYPE_LABELS:
        vec[RING_TYPE_LABELS.index(ring_type)] = 1
    return vec


def ring_type_to_index(ring_type: str) -> int:
    """Map an NHC ring-type label to its position in ``RING_TYPE_LABELS``.

    Args:
        ring_type: Ring-type label, e.g. ``"imidazolylidene"``.

    Returns:
        Index into ``RING_TYPE_LABELS``; unrecognised labels map to the index
        of ``"other"`` (the last entry).
    """
    if ring_type in RING_TYPE_LABELS:
        return RING_TYPE_LABELS.index(ring_type)
    return len(RING_TYPE_LABELS) - 1


def has_unsaturated_backbone(smiles: str) -> bool:
    """Check whether an NHC ring carries a C=C backbone.

    Looks at every 5-membered ring with at least two N or S atoms and reports
    whether any bond between two ring carbons has bond order ≥ 1.5 (double or
    aromatic).

    Args:
        smiles: Catalyst or carbene SMILES.

    Returns:
        True if such a bond exists; False otherwise, including when RDKit
        cannot parse ``smiles``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        n_hetero = sum(a.GetAtomicNum() in (7, 16) for a in atoms)
        if n_hetero < 2:
            continue
        ring_set = set(ring)
        for bond in mol.GetBonds():
            if (
                bond.GetBeginAtomIdx() in ring_set
                and bond.GetEndAtomIdx() in ring_set
                and bond.GetBondTypeAsDouble() >= 1.5
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6
            ):
                return True
    return False


def rdkit_scalar_descriptors(smiles: str) -> dict[str, float]:
    """Compute molecular weight, heavy-atom count and formal charge from SMILES.

    All fragments of ``smiles`` (e.g. counterions) are included.

    Args:
        smiles: Catalyst SMILES.

    Returns:
        Dict with float values under ``"mw"``, ``"heavy_atom_count"`` and
        ``"formal_charge"``.

    Raises:
        AssertionError: If RDKit cannot parse ``smiles``.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit parse failed: {smiles}"
    return {
        "mw": Descriptors.MolWt(mol),
        "heavy_atom_count": float(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
    }

