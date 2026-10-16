"""Frozen MiniMol embedding wrapper — executed by the MiniMol Desktop venv.

This script is intentionally self-contained: it imports only ``minimol``,
``numpy``, and the standard library, so it runs inside the isolated
``~/Desktop/minimol/.venv`` (Python 3.12) and never touches the main ``pablo``
environment. The orchestrating generator
(``pablo.featurization_and_kernel_scan.features.gnn._run_minimol``) invokes it via subprocess.

MiniMol (Kläser et al., arXiv:2404.14986) emits a fixed 512-d fingerprint per
molecule. The model checkpoint is fetched automatically on first construction.

Usage:
    python minimol_embed.py --smiles-csv <in.csv> --out <out.npy>

The input CSV must have a ``smiles`` header column, one SMILES per row, in the
order embeddings should be returned. The output is a float64 ``.npy`` array of
shape ``(n_molecules, 512)``.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def _read_smiles(csv_path: str) -> list[str]:
    """Read the ``smiles`` column from ``csv_path`` preserving row order.

    Args:
        csv_path: Path to a CSV file with a ``smiles`` header column.

    Returns:
        List of SMILES strings in file order.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["smiles"] for row in reader]


def _embed(smiles: list[str]) -> np.ndarray:
    """Compute frozen MiniMol fingerprints.

    Args:
        smiles: List of ``n`` SMILES strings.

    Returns:
        Float64 array of shape ``(n, 512)`` — one MiniMol fingerprint per input.
    """
    from minimol import Minimol

    model = Minimol()
    fingerprints = model(smiles)  # list of n torch tensors, each shape (512,)
    X = np.stack(
        [np.asarray(fp.detach().cpu().numpy(), dtype=np.float64) for fp in fingerprints],
        axis=0,
    )
    return X


def main() -> None:
    """Parse CLI, embed the SMILES, and save the ``(n, 512)`` matrix."""
    parser = argparse.ArgumentParser(description="MiniMol frozen embedding")
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    smiles = _read_smiles(args.smiles_csv)
    X = _embed(smiles)
    assert X.shape[0] == len(smiles), (
        f"MiniMol returned {X.shape[0]} rows for {len(smiles)} SMILES"
    )
    np.save(args.out, X)
    print(f"MiniMol embeddings: {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
