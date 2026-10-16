"""Backfill carbene_idx and hetero_nbr_indices into existing info.json files.

Run once on the existing catalysts whose xyz files were all generated via the
standard ETKDG path (free-carbene SMILES), making their atom ordering
topologically consistent with the free_carbene_smiles field.

Usage:
    python -m pablo.utils.backfill_carbene_indices
"""

from __future__ import annotations

import json

from rdkit import Chem

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
    read_xyz,
)
from pablo.utils.catalyst_form_classifier import locate_carbene_center


def backfill_carbene_indices() -> None:
    """Write ``carbene_idx`` and ``hetero_nbr_indices`` into each catalyst's ``info.json``.

    Scans ``paths.XTB_INPUTS_DIR/C<id>/`` in catalyst-id order. Directories
    without ``info.json`` and files that already carry both fields are skipped.
    For the rest, the carbene carbon and its heteroatom neighbours are located
    on the H-added ``free_carbene_smiles`` molecule, checked against the
    elements in ``mol.xyz``, and written back to ``info.json`` in place.

    Raises:
        AssertionError: If ``mol.xyz`` is missing or was generated via the salt
            embedding path, ``free_carbene_smiles`` does not parse, the atom
            count differs from ``mol.xyz``, no carbene centre is found, or the
            located atoms are not C (carbene) and N/O/S (neighbours) in
            ``mol.xyz``.
    """
    cat_dirs = sorted(
        paths.XTB_INPUTS_DIR.glob("C*/"), key=lambda p: int(p.name[1:])
    )
    n_updated = n_skipped = 0

    for mol_dir in cat_dirs:
        info_path = mol_dir / "info.json"
        if not info_path.exists():
            continue

        info = json.loads(info_path.read_text())
        cid = info["catalyst_id"]

        if "carbene_idx" in info and "hetero_nbr_indices" in info:
            print(f"  C{cid}: already has topology fields, skipping.")
            n_skipped += 1
            continue

        xyz_path = mol_dir / "mol.xyz"
        assert xyz_path.exists(), f"C{cid}: mol.xyz not found."

        xyz_comment = xyz_path.read_text().splitlines()[1]
        assert "via salt" not in xyz_comment.lower(), (
            f"C{cid}: mol.xyz was generated via the salt embedding path "
            f"(comment: '{xyz_comment}'). Backfill cannot safely inject topology "
            "for salt-path xyz files — re-run _prepare_xtb_inputs to regenerate "
            "with correct indices."
        )

        elements, _ = read_xyz(xyz_path)

        mol = Chem.MolFromSmiles(info["free_carbene_smiles"])
        assert mol is not None, f"C{cid}: RDKit parse failed on free_carbene_smiles."
        mol = Chem.AddHs(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)

        assert mol.GetNumAtoms() == len(elements), (
            f"C{cid}: atom count mismatch — mol has {mol.GetNumAtoms()} atoms, "
            f"mol.xyz has {len(elements)}."
        )

        carbene_c, hetero_nbrs, _, _ = locate_carbene_center(mol)
        assert carbene_c is not None, (
            f"C{cid}: locate_carbene_center failed on free_carbene_smiles."
        )

        assert elements[carbene_c] == "C", (
            f"C{cid}: element at carbene_idx={carbene_c} is '{elements[carbene_c]}', not 'C'."
        )
        for ni in hetero_nbrs:
            assert elements[ni] in ("N", "O", "S"), (
                f"C{cid}: element at hetero_nbr_index={ni} is '{elements[ni]}', "
                "expected N/O/S."
            )

        info["carbene_idx"] = carbene_c
        info["hetero_nbr_indices"] = hetero_nbrs
        info_path.write_text(json.dumps(info, indent=2))
        print(f"  C{cid}: carbene_idx={carbene_c}, hetero_nbr_indices={hetero_nbrs}")
        n_updated += 1

    print(f"\nDone: {n_updated} updated, {n_skipped} already complete.")


if __name__ == "__main__":
    backfill_carbene_indices()
