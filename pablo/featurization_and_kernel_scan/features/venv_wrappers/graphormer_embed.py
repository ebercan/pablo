"""Frozen Graphormer embedding wrapper — executed by the Graphormer Desktop venv.

Self-contained: imports only ``torch``, ``numpy``, ``ogb``, ``transformers``, and
the standard library, so it runs inside ``~/Desktop/graphormer/.venv`` (Python
3.12, ``torch>=2.2``) without touching the main ``pablo`` environment. The
orchestrating generator (``pablo.featurization_and_kernel_scan.features.gnn._run_graphormer``) invokes
it via subprocess.

Graphormer (Ying et al., NeurIPS 2021) is a pure graph Transformer pretrained on
PCQM4Mv2 to regress the DFT HOMO–LUMO gap. The original ``graphormer-pretrained``
PyPI package depends on **fairseq**, which does not install on Python 3.12; this
wrapper therefore uses the **Hugging Face** port (``clefourrier/graphormer-base-pcqm4mv2``
+ ``transformers``), which is fairseq-free. The only Python-3.12 fix needed is a
one-line patch to HF's Cython collator (``algos_graphormer.pyx``:
``astype(long)``→``astype("int64")``) — see README §3.2.

Molecules are featurised with OGB ``smiles2graph`` (the exact featurisation the
checkpoint was trained on), preprocessed + collated by HF's Graphormer collator,
and the frozen embedding is the **graph/virtual-node CLS token** ``last_hidden_state[:, 0, :]``
(768-d) of the pretrained encoder.

Usage:
    python graphormer_embed.py --smiles-csv <in.csv> --out <out.npy> \\
        [--pretrained clefourrier/graphormer-base-pcqm4mv2]

Output: float64 ``.npy`` array of shape ``(n_molecules, 768)``.
"""

from __future__ import annotations

import argparse
import csv
import warnings

import numpy as np

warnings.filterwarnings("ignore")


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


def _embed(smiles: list[str], pretrained: str, batch_size: int = 32) -> np.ndarray:
    """Compute frozen Graphormer CLS-token embeddings on CPU.

    Tensor shapes: the encoder maps a collated batch to ``last_hidden_state`` of
    shape ``(B, max_nodes+1, 768)``; the virtual-node token at position 0 is the
    ``(B, 768)`` molecular embedding.

    Args:
        smiles: List of ``n`` SMILES strings.
        pretrained: Hugging Face model id of the PCQM4Mv2 Graphormer checkpoint.
        batch_size: Molecules per forward pass (CPU-friendly chunking).

    Returns:
        Float64 array of shape ``(n, 768)`` — one CLS embedding per input.
    """
    import torch
    from ogb.utils.mol import smiles2graph
    from transformers import GraphormerForGraphClassification
    from transformers.models.graphormer.collating_graphormer import (
        GraphormerDataCollator,
        preprocess_item,
    )

    model = GraphormerForGraphClassification.from_pretrained(pretrained).eval()
    collator = GraphormerDataCollator(spatial_pos_max=20, on_the_fly_processing=False)

    def to_item(smi: str) -> dict:
        g = smiles2graph(smi)
        return {
            "edge_index": g["edge_index"],
            "edge_attr": g["edge_feat"],
            "node_feat": g["node_feat"],
            "num_nodes": int(g["num_nodes"]),
            "y": [0.0],
        }

    out_blocks = []
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start : start + batch_size]
        items = [preprocess_item(to_item(s)) for s in chunk]
        batch = collator(items)
        with torch.no_grad():
            enc = model.encoder(**{k: v for k, v in batch.items() if k != "labels"})
        lhs = enc[0] if isinstance(enc, (tuple, list)) else enc.last_hidden_state
        out_blocks.append(lhs[:, 0, :].cpu().numpy())

    return np.concatenate(out_blocks, axis=0).astype(np.float64)


def main() -> None:
    """Parse CLI, embed the SMILES, and save the ``(n, 768)`` matrix."""
    parser = argparse.ArgumentParser(description="Graphormer frozen embedding (HF)")
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pretrained", default="clefourrier/graphormer-base-pcqm4mv2")
    args = parser.parse_args()

    smiles = _read_smiles(args.smiles_csv)
    X = _embed(smiles, args.pretrained)
    assert X.shape[0] == len(smiles), (
        f"Graphormer returned {X.shape[0]} rows for {len(smiles)} SMILES"
    )
    np.save(args.out, X)
    print(f"Graphormer embeddings: {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
