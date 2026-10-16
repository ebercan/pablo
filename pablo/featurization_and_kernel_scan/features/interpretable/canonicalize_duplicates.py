"""Make free-carbene descriptors identical across salt forms of one carbene.

The 161 library entries are only 113 distinct free carbenes; the rest are the
same carbene as a different salt, CO2 adduct or bicarbonate. Every descriptor
computed on the *free carbene* is therefore a property of the carbene, not of
the entry, and must be identical within a duplicate group. It is not: each
entry was embedded and optimised independently and landed in a slightly
different local minimum, so six of the twenty-six groups disagree — worst case
0.81 percentage points on ``vbur_q4``, 7% of that column's library-wide spread.

This module picks one canonical entry per group — lowest xTB total energy, ties
broken by lowest catalyst id, the members being the same molecule so their
energies are directly comparable — and broadcasts its free-carbene descriptors
to the rest of the group. Nothing is recomputed: ``lp_*`` and ``vbur_q*``
re-derived from the canonical geometry reproduce the canonical entry's stored
values exactly, so a broadcast and a recomputation are the same numbers.

Form-dependent columns (``form_*``, ``mw``, ``heavy_atom_count``) are left
alone. Structural columns (``ring_*``, ``backbone_saturated``) are already
consistent and are asserted to stay so.

The intermediate caches and the xTB checkpoint are patched alongside the
assembled tables, so a re-assembly reproduces the canonicalised values. The
per-catalyst JSON caches are raw computation records and are not touched; the
script is idempotent, so re-run it after any rebuild that regenerates them.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.features.interpretable.canonicalize_duplicates
    python -m pablo.featurization_and_kernel_scan.features.interpretable.canonicalize_duplicates --write
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import XTB_CHECKPOINT

RDLogger.DisableLog("rdApp.*")

#: Descriptors that depend only on the free carbene, so must agree within a group.
FREE_CARBENE_COLS = (
    "lp_L", "lp_B1", "lp_B5",
    "vbur_q1", "vbur_q2", "vbur_q3", "vbur_q4",
    "fukui_plus_c2", "fukui_minus_c2", "homo_weight_c2", "mulliken_charge_c2",
    "delta_e_h_plus", "delta_e_bh3",
    "xtb_homo_eV", "xtb_lumo_eV", "xtb_gap_eV", "xtb_dipole_D", "xtb_polarizability_angst3",
    "xtb_deltaG_deprot",
)
#: Columns that legitimately differ between salt forms of one carbene.
FORM_DEPENDENT_COLS = ("form_salt", "form_free_nhc", "form_co2_adduct", "form_bicarbonate", "mw", "heavy_atom_count")
#: Columns that describe the carbene skeleton and are already group-constant.
STRUCTURAL_COLS = (
    "ring_imidazolylidene", "ring_imidazolinylidene", "ring_triazolinylidene",
    "ring_thiazolylidene", "ring_other", "backbone_saturated",
)
FEATURIZATIONS = ("interpretable", "interpretable_light", "interpretable_dG")
_ENERGY = re.compile(r"energy:\s*([-0-9.]+)")


def _library() -> tuple[pd.Series, pd.Series]:
    """Return ``(canonical free-carbene SMILES, xTB total energy)`` per catalyst id."""
    smiles: dict[int, str] = {}
    energy: dict[int, float] = {}
    for info_path in sorted(paths.XTB_INPUTS_DIR.glob("C*/info.json")):
        info = json.loads(info_path.read_text())
        cid = int(info["catalyst_id"])
        mol = Chem.MolFromSmiles(info["free_carbene_smiles"])
        assert mol is not None, f"C{cid}: unparsable free_carbene_smiles."
        smiles[cid] = Chem.MolToSmiles(mol)
        comment = (info_path.parent / "xtbopt.xyz").read_text().split("\n")[1]
        match = _ENERGY.search(comment)
        assert match is not None, f"C{cid}: no energy in the xtbopt.xyz comment line."
        energy[cid] = float(match.group(1))
    return pd.Series(smiles).sort_index(), pd.Series(energy).sort_index()


def duplicate_groups() -> dict[int, list[int]]:
    """Map canonical catalyst id -> every catalyst id sharing its free carbene.

    The canonical entry is the group member with the lowest xTB total energy;
    ties (within 1e-8 Eh) go to the lowest catalyst id. Singleton groups are
    included, so the mapping covers the whole library.
    """
    smiles, energy = _library()
    groups: dict[int, list[int]] = {}
    for _, members in smiles.groupby(smiles):
        ids = sorted(members.index.tolist())
        best = min(ids, key=lambda cid: (round(float(energy[cid]), 8), cid))
        groups[best] = ids
    return groups


def _broadcast(df: pd.DataFrame, groups: dict[int, list[int]]) -> pd.DataFrame:
    """Copy each canonical row's free-carbene columns onto its group members."""
    cols = [c for c in FREE_CARBENE_COLS if c in df.columns]
    out = df.copy()
    for canonical, ids in groups.items():
        if len(ids) == 1 or canonical not in out.index:
            continue
        members = [i for i in ids if i in out.index]
        out.loc[members, cols] = out.loc[canonical, cols].to_numpy()
    return out


def _report(label: str, old: pd.DataFrame, new: pd.DataFrame) -> int:
    cols = [c for c in FREE_CARBENE_COLS if c in old.columns]
    delta = (new[cols] - old[cols]).abs()
    changed = delta.gt(1e-12).any(axis=1)
    n = int(changed.sum())
    print(f"\n{label}: {n} of {len(old)} rows change")
    if n:
        worst = delta.max().sort_values(ascending=False)
        worst = worst[worst > 1e-12]
        for col, val in worst.items():
            print(f"    {col:28s} max |change| = {val:.4g}")
        print(f"    catalysts: {old.index[changed].tolist()}")
    return n


def canonicalize(write: bool) -> None:
    """Broadcast canonical free-carbene descriptors across every duplicate group.

    Args:
        write: When ``False`` only report. When ``True`` patch the two
            intermediate parquets, the xTB checkpoint and every featurization
            in ``FEATURIZATIONS`` that carries the affected columns.
    """
    groups = duplicate_groups()
    dup = {c: ids for c, ids in groups.items() if len(ids) > 1}
    print(f"{len(groups)} distinct free carbenes over {sum(len(v) for v in groups.values())} entries; "
          f"{len(dup)} duplicate groups covering {sum(len(v) for v in dup.values())} entries.")
    print("canonical picks (lowest xTB energy, ties by lowest id):")
    for canonical, ids in sorted(dup.items()):
        print(f"    C{canonical:<4d} <- {ids}")

    targets: list[tuple[str, object, pd.DataFrame, pd.DataFrame]] = []
    for name in FEATURIZATIONS:
        path = paths.FEAT_DIR / name / "descriptors.parquet"
        assert path.exists(), f"Missing {path}."
        old = pd.read_parquet(path)
        new = _broadcast(old, groups)
        for col in STRUCTURAL_COLS:
            if col in new.columns:
                for ids in dup.values():
                    assert new.loc[ids, col].nunique() == 1, f"{name}: {col} varies within {ids}."
        for col in FORM_DEPENDENT_COLS:
            if col in new.columns:
                assert new[col].equals(old[col]), f"{name}: form-dependent {col} was modified."
        if name == "interpretable_light" and "vbur_scalar" in new.columns:
            q = [f"vbur_q{i}" for i in range(1, 5)]
            base = pd.read_parquet(paths.FEAT_DIR / "interpretable" / "descriptors.parquet")
            assert np.allclose(old["vbur_scalar"], base[q].mean(axis=1), atol=1e-6), "vbur_scalar formula mismatch."
            assert np.allclose(
                old["vbur_asymmetry"], base[q].max(axis=1) - base[q].min(axis=1), atol=1e-6
            ), "vbur_asymmetry formula mismatch."
            bq = _broadcast(base, groups)[q]
            new["vbur_scalar"] = bq.mean(axis=1)
            new["vbur_asymmetry"] = bq.max(axis=1) - bq.min(axis=1)
        assert list(new.columns) == list(old.columns)
        _report(f"{name}/descriptors.parquet", old, new)
        targets.append((name, path, old, new))

    caches = [
        ("_intermediate/lp_sterimol.parquet", paths.INTERMEDIATE_DIR / "lp_sterimol.parquet"),
        ("_intermediate/vbur_quadrants.parquet", paths.INTERMEDIATE_DIR / "vbur_quadrants.parquet"),
        ("xtb/results_checkpoint.parquet", XTB_CHECKPOINT),
    ]
    cache_targets: list[tuple[str, object, pd.DataFrame]] = []
    for label, path in caches:
        assert path.exists(), f"Missing {path}."
        old = pd.read_parquet(path)
        old.index = old.index.astype(int)
        new = _broadcast(old, groups)
        _report(label, old, new)
        cache_targets.append((label, path, new))

    if not write:
        print("\ndry run: nothing written. Re-run with --write to apply.")
        return

    for label, path, new in cache_targets:
        new.to_parquet(path)
        print(f"wrote {path}")
    for name, path, _old, new in targets:
        new.to_parquet(path)
        print(f"wrote {path}")

    print("\nverifying every duplicate group is now internally consistent...")
    bad = 0
    for name in FEATURIZATIONS:
        df = pd.read_parquet(paths.FEAT_DIR / name / "descriptors.parquet")
        cols = [c for c in FREE_CARBENE_COLS if c in df.columns]
        for ids in dup.values():
            spread = (df.loc[ids, cols].max() - df.loc[ids, cols].min()).abs()
            offenders = spread[spread > 0]
            if len(offenders):
                bad += 1
                print(f"  STILL INCONSISTENT {name} {ids}: {offenders.to_dict()}")
    print(f"  groups still inconsistent: {bad}")
    assert bad == 0, "canonicalization did not converge."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="Apply. Default is a dry run.")
    args = parser.parse_args()
    canonicalize(write=args.write)


if __name__ == "__main__":
    main()
