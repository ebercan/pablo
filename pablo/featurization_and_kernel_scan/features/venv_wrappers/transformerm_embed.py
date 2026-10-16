"""Frozen Transformer-M embedding wrapper — executed by the Transformer-M venv.

Self-contained: imports only ``torch``, ``numpy``, ``rdkit``, the Transformer-M /
fairseq repo modules (added to ``sys.path`` from ``--repo-dir``), and the standard
library, so it runs inside ``~/Desktop/transformer_m/.venv`` (**Python 3.10** —
fairseq's ``omegaconf<2.1`` pin has no cp312 wheel, so this one model cannot use
3.12) without touching the main ``pablo`` environment. The orchestrating generator
(``pablo.featurization_and_kernel_scan.features.gnn._run_transformerm``) invokes it via subprocess; the
``.npy`` contract is Python-version-agnostic, so the py3.10 venv talks to the
py3.12 driver unchanged.

Transformer-M (Luo et al., ICLR 2023, github lsj2408/Transformer-M) is a
Graphormer-style transformer with separate 2D and 3D channels, pretrained on
PCQM4Mv2 (L12 checkpoint, 768-d). The frozen-embedding path is reconstructed from
``evaluate.py`` + ``models/transformer_m.py``:

  - registration : ``fairseq.utils.import_user_module`` loads the ``Transformer-M``
    user-dir, registering the ``Transformer-M`` model and ``transformer_m_base`` arch.
  - model build  : ``TransformerMModel.build_model(args, task=None)`` — ``build_model``
    never touches ``task`` (so the data-loading ``GraphPredictionTask`` is bypassed);
    ``args`` is the flat namespace saved in the checkpoint. Weights from
    ``checkpoint['model']`` (``strict=False``).
  - featurisation: per molecule a PyG ``Data`` (OGB atom/bond features + a 3D RDKit
    conformer) → ``preprocess_item`` (Cython ``algos`` spatial/edge encodings) →
    ``collator_3d`` → the ``batched_data`` dict the encoder consumes.
  - frozen embedding : the final-layer hidden state at the **graph/virtual token**
    (sequence position 0), i.e. ``inner_states[-1][0]`` of shape ``(B, 768)``. The
    1-d regression head (``embed_out``) is not used.

Transformer-M uses 3D positions, so the wrapper generates one RDKit ETKDGv3+MMFF
conformer per molecule (with ``AddHs``), with the same robustness ladder + minimum
interatomic-separation guard as ``frad_embed.py``. This box has no GPU → CPU.

Usage:
    python transformerm_embed.py --smiles-csv <in.csv> --repo-dir <transformer_m_repo> \\
        --checkpoint <L12.pt> --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 768)``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from argparse import Namespace
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

    Multi-fragment catalyst salts occasionally embed with near-coincident atoms; no
    real conformer has atoms below ~1 Å apart, so a 0.5 Å floor only repairs
    pathological overlaps and never perturbs valid geometry.

    Args:
        pos: ``(n_atoms, 3)`` conformer coordinates in Å.
        seed: RNG seed for deterministic jitter.
        min_sep: Minimum allowed interatomic distance in Å.
        max_tries: Maximum jitter iterations before returning best effort.

    Returns:
        ``(n_atoms, 3)`` finite coordinates with pairwise distances ≥ ``min_sep``
        (best effort within ``max_tries``).
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


def _smiles_to_data(smiles: str, idx: int, seed: int = 42):
    """Build a PyG ``Data`` object (2D graph + 3D conformer) for one SMILES.

    Args:
        smiles: A single SMILES string.
        idx: Sequential index stored on the ``Data`` object (``collator`` needs it).
        seed: ETKDG / jitter seed for a deterministic conformer.

    Returns:
        A ``torch_geometric.data.Data`` with ``x`` ``(n_atoms, 9)``, ``edge_index``
        ``(2, n_edges)``, ``edge_attr`` ``(n_edges, 3)``, ``pos`` ``(n_atoms, 3)``,
        ``idx`` and a dummy ``y``.
    """
    import torch
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit failed to parse SMILES: {smiles}"
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        params.ignoreSmoothingFailures = True
        status = AllChem.EmbedMolecule(mol, params)
    if status == 0:
        AllChem.MMFFOptimizeMolecule(mol)
        pos = mol.GetConformer().GetPositions().astype(np.float32)
    else:
        AllChem.Compute2DCoords(mol)
        pos = mol.GetConformer().GetPositions().astype(np.float32)
        pos = pos + np.random.default_rng(seed).normal(0.0, 0.5, size=pos.shape).astype(
            np.float32
        )
    pos = _separate_close_atoms(pos, seed)

    atom_feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_feats, dtype=np.int64), dtype=torch.long)

    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = bond_to_feature_vector(bond)
        edges += [(i, j), (j, i)]
        edge_feats += [ef, ef]
    if edges:
        edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feats, dtype=np.int64), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pos = torch.from_numpy(pos)
    data.idx = idx
    data.y = torch.zeros(1, dtype=torch.float)
    return data


def _build_model(repo_dir: str, checkpoint: str):
    """Register Transformer-M, build the model, and load the pretrained weights.

    Args:
        repo_dir: Path to the cloned Transformer-M repo (added to ``sys.path``; also
            holds the compiled ``algos`` extension and the ``Transformer-M`` user-dir).
        checkpoint: Path to the ``L12.pt`` fairseq checkpoint.

    Returns:
        The eval-mode ``TransformerMModel`` on CPU.
    """
    import torch
    from fairseq import utils
    from fairseq.models import ARCH_CONFIG_REGISTRY, MODEL_REGISTRY

    utils.import_user_module(Namespace(user_dir=str(Path(repo_dir) / "Transformer-M")))

    ckpt = torch.load(checkpoint, map_location="cpu")

    if ckpt.get("args", None) is not None:
        args = ckpt["args"]
    elif "cfg" in ckpt:
        from omegaconf import OmegaConf

        cfg = ckpt["cfg"]
        merged: dict = {}
        for section in ("model", "task"):
            if section in cfg and cfg[section] is not None:
                merged.update(OmegaConf.to_container(cfg[section], resolve=True))
        args = Namespace(**merged)
    else:
        # The published L12 checkpoint stores only weights, so reconstruct the exact
        # training args: GraphPredictionConfig task defaults + the transformer_m_base
        # arch (768-d / 12 layers / 32 heads), with the 3D channel enabled
        # (``add_3d`` is NOT a base-arch default). ARCH_CONFIG_REGISTRY applies the
        # remaining arch defaults (dropout, gelu, learned pos, pre-LN, apply_init, …).
        args = Namespace(
            arch="transformer_m_base",
            max_positions=512,
            num_atoms=512 * 9,
            num_edges=512 * 3,
            num_in_degree=512,
            num_out_degree=512,
            num_spatial=512,
            num_edge_dis=128,
            multi_hop_max_dist=5,
            edge_type="multi_hop",
            add_3d=True,
            no_2d=False,
            num_3d_bias_kernel=128,
            load_qm9=False,
            remove_head=False,
        )
        ARCH_CONFIG_REGISTRY["transformer_m_base"](args)

    model = MODEL_REGISTRY["Transformer-M"].build_model(args, task=None)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(
        f"Transformer-M load_state_dict: {len(missing)} missing, "
        f"{len(unexpected)} unexpected keys"
    )
    model.eval()
    return model


def _embed(smiles: list[str], repo_dir: str, checkpoint: str, batch_size: int = 8) -> np.ndarray:
    """Compute frozen Transformer-M graph-token embeddings on CPU.

    Tensor shapes: ``collator_3d`` builds the padded ``batched_data`` dict; the
    encoder's final-layer ``inner_states[-1]`` is ``(seq_len, B, 768)`` and the
    graph token at position 0 gives the ``(B, 768)`` molecular embedding.

    Args:
        smiles: List of ``n`` SMILES strings.
        repo_dir: Path to the cloned Transformer-M repo.
        checkpoint: Path to the ``L12.pt`` checkpoint.
        batch_size: Molecules per forward pass (CPU-friendly chunking).

    Returns:
        Float64 array of shape ``(n, 768)`` — one graph-token embedding per input.
    """
    repo = str(Path(repo_dir).resolve())
    sys.path.insert(0, repo)  # compiled ``algos`` extension lives at the repo root
    sys.path.insert(0, str(Path(repo) / "Transformer-M"))  # so ``data`` is a package

    import torch

    from data.collator import collator_3d
    from data.wrapper import preprocess_item

    model = _build_model(repo, checkpoint)

    out_blocks = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start : start + batch_size]
            items = [
                preprocess_item(_smiles_to_data(s, i))
                for i, s in enumerate(chunk, start=start)
            ]
            batched = collator_3d(items)
            out = model(batched_data=batched)
            inner_states = out[2]["inner_states"]
            last = inner_states[-1]
            graph_token = last[0]  # (B, embed_dim)
            out_blocks.append(graph_token.cpu().numpy())

    return np.concatenate(out_blocks, axis=0).astype(np.float64)


def main() -> None:
    """Parse CLI, embed the SMILES, and save the ``(n, 768)`` matrix."""
    parser = argparse.ArgumentParser(description="Transformer-M frozen embedding")
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    smiles = _read_smiles(args.smiles_csv)
    X = _embed(smiles, args.repo_dir, args.checkpoint)
    assert X.shape[0] == len(smiles), (
        f"Transformer-M returned {X.shape[0]} rows for {len(smiles)} SMILES"
    )
    np.save(args.out, X)
    print(f"Transformer-M embeddings: {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
