"""Frozen Frad embedding wrapper — executed by the Frad Desktop venv.

Self-contained: imports only ``torch``, ``numpy``, ``rdkit``, ``torch_scatter``,
the Frad repo modules (added to ``sys.path`` from ``--repo-dir``), and the
standard library, so it runs inside ``~/Desktop/frad/.venv`` (Python 3.12,
``torch==2.6.0`` CPU — ported off the repo's frozen py3.9.10/torch1.10 conda pins)
without touching the main ``pablo`` environment. The orchestrating generator
(``pablo.featurization_and_kernel_scan.features.gnn._run_frad``) invokes it via
subprocess.

Frad (Feng et al., ICML 2023, github fengshikun/Frad) pre-trains a TorchMD-NET
SE(3)-equivariant transformer by **fractional denoising** on PCQM4Mv2 3D
structures. It is a 3D model, so this wrapper generates a single RDKit
ETKDG(v3)+MMFF conformer per molecule (the model was pretrained with explicit
hydrogens, so ``Chem.AddHs`` is applied first).

The frozen-embedding path is reconstructed from ``torchmdnet/models/model.py``:

  - model build + weights : ``load_model(checkpoint, device='cpu')`` reads the
    checkpoint's ``hyper_parameters`` (``model='equivariant-transformer'``,
    ``embedding_dimension=256``) and loads ``state_dict`` (``strict=False``;
    the denoising/output heads are unused here).
  - frozen embedding      : the **representation model's per-atom scalar features**
    ``x`` (``TorchMD_ET.forward`` returns ``x, v, z, pos, batch`` with ``x`` of
    shape ``(n_atoms, 256)``), mean-pooled per molecule to ``(n_molecules, 256)``.
    The scalar regression head (``TorchMD_Net.forward`` → a single property) is
    **not** used. Mean vs sum pooling is irrelevant downstream: the driver
    L2-normalises, under which ``mean`` and ``add`` reductions give the identical
    unit vector.

This box has no GPU, so everything runs on CPU.

Usage:
    python frad_embed.py --smiles-csv <in.csv> --repo-dir <frad_repo> \\
        --checkpoint <frad_pretrained.pt> --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 256)``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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


def _separate_close_atoms(
    pos: np.ndarray, seed: int, min_sep: float = 0.5, max_tries: int = 50
) -> np.ndarray:
    """Jitter apart any atoms closer than ``min_sep`` Å (degenerate-geometry guard).

    Multi-fragment catalyst salts occasionally embed with near-coincident atoms
    (RDKit places fragments in a shared frame; MMFF may not resolve the overlap).
    Coincident atoms give the SE(3) model an undefined unit direction vector
    (``0/0``) → non-finite features → the whitened equivariant layer-norm SVD
    fails. No real conformer has atoms below ~1 Å apart, so enforcing a 0.5 Å floor
    only repairs pathological overlaps and never perturbs valid geometry.

    Args:
        pos: ``(n_atoms, 3)`` conformer coordinates in Å.
        seed: RNG seed for deterministic jitter.
        min_sep: Minimum allowed interatomic distance in Å.
        max_tries: Maximum jitter iterations before returning best effort.

    Returns:
        ``(n_atoms, 3)`` coordinates with all pairwise distances ≥ ``min_sep``
        (best effort within ``max_tries``), guaranteed finite.
    """
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        too_close = dist.min(axis=1) < min_sep
        if not too_close.any():
            break
        pos[too_close] += rng.normal(0.0, min_sep, size=(int(too_close.sum()), 3)).astype(
            np.float32
        )
    return pos


def _smiles_to_conformer(smiles: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Embed a single ETKDGv3 + MMFF conformer for one SMILES.

    Args:
        smiles: A single SMILES string.
        seed: ETKDG random seed for a deterministic conformer.

    Returns:
        Tuple ``(z, pos)`` where ``z`` is an int64 array of shape ``(n_atoms,)``
        of atomic numbers and ``pos`` is a float32 array of shape ``(n_atoms, 3)``
        of conformer coordinates in Ångström (explicit hydrogens included).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit failed to parse SMILES: {smiles}"
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        # Bulky/congested catalyst salts can defeat the default distance-geometry
        # start; random-coordinate init + ignoring bounds-smoothing failures embeds
        # most of them (RDKit returns a status code, not an exception, so this is a
        # plain retry — not try/except).
        params.useRandomCoords = True
        params.ignoreSmoothingFailures = True
        embed_status = AllChem.EmbedMolecule(mol, params)

    if embed_status == 0:
        AllChem.MMFFOptimizeMolecule(mol)
        pos = mol.GetConformer().GetPositions().astype(np.float32)
    else:
        # Last resort for strained polycyclic salts that defeat 3D distance geometry
        # entirely (rare). A planar/overlapping fallback would feed coincident atoms
        # (zero interatomic distance) into the whitened equivariant layer-norm and
        # produce non-finite SVD inputs, so build a NON-degenerate 3D conformer: 2D
        # layout perturbed by small Gaussian jitter (incl. z) to break coplanarity
        # and any atom overlaps. Keeps all 161 molecules embeddable.
        AllChem.Compute2DCoords(mol)
        pos = mol.GetConformer().GetPositions().astype(np.float32)
        rng = np.random.default_rng(seed)
        pos = pos + rng.normal(0.0, 0.5, size=pos.shape).astype(np.float32)

    pos = _separate_close_atoms(pos, seed)
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    return z, pos


def _embed(smiles: list[str], repo_dir: str, checkpoint: str) -> np.ndarray:
    """Compute frozen Frad representation embeddings on CPU.

    Tensor shapes: per-molecule ``z`` ``(n_atoms,)`` and ``pos`` ``(n_atoms, 3)``
    are concatenated into flat ``(N,)`` / ``(N, 3)`` tensors with a ``(N,)`` batch
    index; ``representation_model`` returns per-atom ``x`` ``(N, 256)``, which is
    scatter-mean-pooled per molecule to ``(n, 256)``.

    Args:
        smiles: List of ``n`` SMILES strings.
        repo_dir: Path to the cloned Frad repo (added to ``sys.path``).
        checkpoint: Path to the pretrained PCQM4Mv2 denoising checkpoint.

    Returns:
        Float64 array of shape ``(n, 256)`` — one pooled representation per input.
    """
    sys.path.insert(0, str(Path(repo_dir).resolve()))

    import torch
    from torch_scatter import scatter

    from torchmdnet.models.model import load_model

    device = torch.device("cpu")
    # This pretrained checkpoint predates four args that ``create_model`` now reads;
    # inject the repo's training-time defaults (``scripts/train.py``). ``load_model``
    # merges extra kwargs into the checkpoint's ``hyper_parameters``.
    model = load_model(
        checkpoint,
        device=device,
        md17=False,
        seperate_noise=False,
        mask_atom=False,
        bond_length_scale=0.0,
    )
    model.eval()

    z_list = []
    pos_list = []
    batch_list = []
    for i, smi in enumerate(smiles):
        z_i, pos_i = _smiles_to_conformer(smi)
        z_list.append(torch.from_numpy(z_i))
        pos_list.append(torch.from_numpy(pos_i))
        batch_list.append(torch.full((z_i.shape[0],), i, dtype=torch.long))

    z = torch.cat(z_list).to(device)
    pos = torch.cat(pos_list).to(device)
    batch = torch.cat(batch_list).to(device)

    with torch.no_grad():
        x, _v, _z, _pos, batch_out = model.representation_model(z, pos, batch=batch)
        emb = scatter(x, batch_out, dim=0, reduce="mean")

    return np.asarray(emb.detach().cpu().numpy(), dtype=np.float64)


def main() -> None:
    """Parse CLI, embed the SMILES, and save the ``(n, 256)`` matrix."""
    parser = argparse.ArgumentParser(description="Frad frozen embedding")
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    smiles = _read_smiles(args.smiles_csv)
    X = _embed(smiles, args.repo_dir, args.checkpoint)
    assert X.shape[0] == len(smiles), (
        f"Frad returned {X.shape[0]} rows for {len(smiles)} SMILES"
    )
    np.save(args.out, X)
    print(f"Frad embeddings: {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
