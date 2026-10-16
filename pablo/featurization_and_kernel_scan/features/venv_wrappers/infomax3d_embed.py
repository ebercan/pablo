"""Frozen 3D-Infomax embedding wrapper — executed by the 3D-Infomax Desktop venv.

Self-contained: imports only ``torch``, ``dgl``, ``numpy``, ``yaml``, the
3D-Infomax repo modules (added to ``sys.path`` from ``--repo-dir``), and the
standard library, so it runs inside ``~/Desktop/3dinfomax/.venv`` (Python 3.12,
``torch>=2.2``) without touching the main ``pablo`` environment. The
orchestrating generator
(``pablo.featurization_and_kernel_scan.features.gnn._run_infomax3d``) invokes it
via subprocess.

3D-Infomax (Stärk et al., arXiv:2110.04126, github HannesStark/3DInfomax)
pre-trains a 2D PNA message-passing network so that its representation carries
distilled 3D geometry; at inference only the **2D graph** is consumed (no
conformer step). The frozen-embedding path below is reconstructed from the repo's
``inference.py`` + ``configs_clean/fingerprint_inference.yml`` (which point at the
in-repo pretrained checkpoint
``runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt``):

  - per-molecule featurizer : ``InferenceDataset`` (RDKit ``AddHs`` + OGB
    atom/bond feature vectors -> a DGL graph with ``ndata['feat']`` /
    ``edata['feat']``)
  - batch collation         : ``graph_only_collate`` (``dgl.batch``)
  - model                   : ``models.PNA`` built from the checkpoint dir's
    ``train_arguments.yaml`` ``model_parameters`` (``target_dim=256``), with
    weights from ``checkpoint['model_state_dict']``
  - frozen embedding        : ``PNA.forward`` output — the readout MLP fingerprint
    ``(B, 256)``. This is the published "fingerprint" of ``inference.py``.

The model is constructed directly from ``models.PNA`` (mirroring the repo's
``from models import *`` idiom) rather than by importing ``inference.py``, which
pulls heavy unrelated deps (seaborn, icecream, every trainer). ``avg_d=1`` matches
``load_model`` for ``InferenceDataset`` (no ``avg_degree`` attribute).

Unlike the repo's ``inference.py`` (which omits ``model.eval()`` and so makes the
batch-norm fingerprint depend on batch grouping), this wrapper runs under
``model.eval()`` + ``torch.no_grad()`` so the per-molecule embedding is
deterministic and batch-independent — the correct contract for a frozen
featurizer. This box has no GPU, so everything runs on CPU.

Usage:
    python infomax3d_embed.py --smiles-csv <in.csv> --repo-dir <3dinfomax_repo> \\
        --checkpoint <runs/.../best_checkpoint_35epochs.pt> --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 256)``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
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


def _embed(
    smiles: list[str], repo_dir: str, checkpoint: str, batch_size: int = 32
) -> np.ndarray:
    """Compute frozen 3D-Infomax PNA fingerprints on CPU.

    Tensor shapes: ``graph_only_collate`` batches the per-molecule DGL graphs;
    ``PNA.forward`` reduces each graph's node features to the readout MLP output,
    yielding ``(B, 256)`` per batch, concatenated to ``(n, 256)``.

    Args:
        smiles: List of ``n`` SMILES strings.
        repo_dir: Path to the cloned 3D-Infomax repo (added to ``sys.path``).
        checkpoint: Path to the pretrained checkpoint
            (``runs/.../best_checkpoint_35epochs.pt``).
        batch_size: Molecules per forward pass (CPU-friendly chunking).

    Returns:
        Float64 array of shape ``(n, 256)`` — one PNA fingerprint per input.
    """
    sys.path.insert(0, str(Path(repo_dir).resolve()))

    import dgl  # noqa: F401  (registers DGL backend; imported by the repo modules)
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from datasets.custom_collate import graph_only_collate
    from datasets.inference_dataset import InferenceDataset
    from models import PNA

    device = torch.device("cpu")

    ckpt_dir = Path(checkpoint).resolve().parent
    with open(ckpt_dir / "train_arguments.yaml") as f:
        train_args = yaml.safe_load(f)
    assert train_args["model_type"] == "PNA", (
        f"Expected a PNA checkpoint, got model_type={train_args['model_type']}"
    )

    model = PNA(avg_d=1, device=device, **train_args["model_parameters"]).to(device)
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model.eval()

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(smiles))
        smiles_txt = tf.name

    test_data = InferenceDataset(smiles_txt_path=smiles_txt, device=device)
    os.unlink(smiles_txt)
    assert len(test_data) == len(smiles), (
        f"InferenceDataset built {len(test_data)} graphs for {len(smiles)} SMILES"
    )

    loader = DataLoader(test_data, batch_size=batch_size, collate_fn=graph_only_collate)
    out_blocks = []
    with torch.no_grad():
        for batch in loader:
            out_blocks.append(model(batch.to(device)).cpu().numpy())

    return np.concatenate(out_blocks, axis=0).astype(np.float64)


def main() -> None:
    """Parse CLI, embed the SMILES, and save the ``(n, 256)`` matrix."""
    parser = argparse.ArgumentParser(description="3D-Infomax frozen embedding")
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    smiles = _read_smiles(args.smiles_csv)
    X = _embed(smiles, args.repo_dir, args.checkpoint)
    assert X.shape[0] == len(smiles), (
        f"3D-Infomax returned {X.shape[0]} rows for {len(smiles)} SMILES"
    )
    np.save(args.out, X)
    print(f"3D-Infomax embeddings: {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
