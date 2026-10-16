"""Frozen SCAGE embedding wrapper — executed by the SCAGE Desktop venv.

Self-contained: imports only ``torch``, ``numpy``, the SCAGE repo modules (added
to ``sys.path`` from ``--repo-dir``), and the standard library, so it runs inside
``~/Desktop/scage/.venv`` (Python 3.12, ``torch>=2.2``) without touching the main
``pablo`` environment. The orchestrating generator
(``pablo.featurization_and_kernel_scan.features.gnn._run_scage``) invokes it via
subprocess.

SCAGE (Nature Communications 2025, github KazeDog/scage) ships no documented
frozen-inference API. The pipeline below is reconstructed from ``finetune_mpp.py``:

  - per-molecule featurizer : ``_process_smiles_without_label`` -> ``mol_to_data_pkl``
    (RDKit graph + 3D conformer geometry)
  - batch collation         : ``collator_finetune_pkl``  (returns a dict of tensors)
  - model                   : ``models.scage.Scage`` (mode ``finetune``)
  - frozen embedding        : the pretrained encoder's CLS token ``atom[:, 0, :]``,
    i.e. the output of ``atom_feature`` + the ``EncoderAtomList`` stack BEFORE any
    finetune head. ``head_Graph`` is randomly initialised at finetune time, so its
    ``graph_feature`` is NOT used.

``GlobalVar.dist_bar`` is read from the checkpoint and set before the model is
built (its ``__init__`` registers a ``dist_bar`` buffer and the collator bins
distances against it). This box has no GPU, so everything runs on CPU.

Usage:
    python scage_embed.py --smiles-csv <in.csv> --repo-dir <scage_repo> \\
        --checkpoint <weights/pretrain/pretrain.pth> --config <config_finetune.yaml> \\
        --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, atom_embed_dim)``.
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


def _load_config(config_path: str) -> dict:
    """Load the SCAGE finetune config YAML used to construct the model.

    Args:
        config_path: Path to ``config/config_finetune.yaml``.

    Returns:
        Parsed config dictionary (provides ``model`` hyperparameters + ``mode``).
    """
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def _embed(smiles: list[str], repo_dir: str, checkpoint: str, config_path: str) -> np.ndarray:
    """Compute frozen SCAGE encoder embeddings (CLS token) on CPU.

    Tensor shapes: the collated batch maps to ``atom`` of shape
    ``(B, n_atoms+1, atom_embed_dim)``; the CLS token ``atom[:, 0, :]`` is the
    ``(B, atom_embed_dim)`` molecular embedding returned here.

    Args:
        smiles: List of ``n`` SMILES strings.
        repo_dir: Path to the cloned SCAGE repo (added to ``sys.path``).
        checkpoint: Path to the pretrained checkpoint (``weights/pretrain``).
        config_path: Path to the finetune config YAML.

    Returns:
        Float64 array of shape ``(n, atom_embed_dim)`` — one embedding per input.
    """
    sys.path.insert(0, str(Path(repo_dir).resolve()))

    import torch

    from utils.global_var_util import GlobalVar

    config = _load_config(config_path)
    device = torch.device("cpu")

    # dist_bar lives in the checkpoint and must be set before Scage.__init__
    # (which registers it as a buffer) and before collation (which bins against it).
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "dist_bar" in ckpt:
        db = ckpt["dist_bar"]
        GlobalVar.dist_bar = db.cpu().numpy() if hasattr(db, "cpu") else np.asarray(db)
    else:
        # The pretrain checkpoint stores only weights, not dist_bar. Its attention
        # scale_linear is sized len(dist_bar)*hidden, and the published checkpoint
        # was trained with the pretrain default (length 2). Mismatching the length
        # breaks load_state_dict, so reuse that exact default.
        GlobalVar.dist_bar = np.array([20, 50], dtype=np.float64)
    GlobalVar.use_ckpt = True
    GlobalVar.parallel_train = False

    from data_process.compound_tools import CompoundKit
    from data_process.data_collator import collator_finetune_pkl
    from data_process.function_group_constant import nfg
    from data_process.utils_dataprocess import _process_smiles_without_label
    from models.scage import Scage

    m = config["model"]
    model = Scage(
        mode="finetune",
        atom_names=CompoundKit.atom_vocab_dict.keys(),
        atom_embed_dim=m["atom_embed_dim"],
        num_kernel=m["num_kernel"],
        layer_num=m["layer_num"],
        num_heads=m["num_heads"],
        atom_FG_class=nfg() + 1,
        hidden_size=m["hidden_size"],
        num_tasks=1,
    ).to(device)

    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"SCAGE load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()

    items = []
    for s in smiles:
        data = _process_smiles_without_label(s)
        assert data is not None, f"SCAGE failed to featurize SMILES: {s}"
        items.append(data)
    batch = collator_finetune_pkl(items)
    batch = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()
    }

    with torch.no_grad():
        atom = model.atom_feature(batch)
        attn_mask = batch["atom_attention_mask"]
        for i in range(model.layer_num):
            atom = model.EncoderAtomList[i](
                x=atom,
                attn_mask=attn_mask,
                dist=batch["pair_distances"],
                dist_bar=batch["atom_dist_bar"],
            )
        cls = atom[:, 0, :]

    return np.asarray(cls.detach().cpu().numpy(), dtype=np.float64)


def main() -> None:
    """Parse CLI, embed the SMILES, and save the ``(n, atom_embed_dim)`` matrix."""
    parser = argparse.ArgumentParser(description="SCAGE frozen embedding")
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    smiles = _read_smiles(args.smiles_csv)
    X = _embed(smiles, args.repo_dir, args.checkpoint, args.config)
    assert X.shape[0] == len(smiles), (
        f"SCAGE returned {X.shape[0]} rows for {len(smiles)} SMILES"
    )
    np.save(args.out, X)
    print(f"SCAGE embeddings: {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
