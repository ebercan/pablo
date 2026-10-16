"""Frozen Monroe embedding wrapper — executed by the Monroe Desktop venv.

Imports the ``monroe`` package installed in ``~/Desktop/monroe/.venv``
(``pip install -e`` of github.com/blazejba/monroe) and runs its documented
inference path, ``monroe.eval.embed``: ``featurize_smiles`` builds one graph per
SMILES, ``to_pyg`` converts it, and the encoder restored by
``monroe.model.ckpt.load_ckpt`` returns a 720-d graph-level embedding. The
orchestrating generator (``features.gnn._run_monroe``) invokes it via subprocess.

Monroe (Banaszewski & Fitzgibbon, arXiv 2608.18982) is a GRIT graph transformer
pretrained on ~81M PM6 molecules (ground, triplet, cation and anion targets),
1.56M PCBA molecules and a conformer-denoising task. Its encoder reads 3D
coordinates: the featurizer round-trips the SMILES through InChI, embeds one
ETKDGv3 conformer and relaxes it with MMFF94s (UFF when MMFF has no
parameters). A SMILES that InChI cannot rebuild (e.g. a free carbene) is built
directly from the SMILES instead; a conformer that cannot be embedded, or takes
longer than 10 s, is replaced by a flat 2D layout.

Two changes make this reproducible, everything else is upstream code:

  - ETKDGv3 has no seed upstream, so every run embeds a different conformer.
    ``rdDistGeom.ETKDGv3`` is wrapped to set ``randomSeed = --seed``.
  - Upstream skips molecules whose graph cannot be built and keys the result by
    SMILES. Here every input must come back, or the run raises; the molecules
    that fell back to a 2D layout are listed in ``<out>.flat2d.txt``.

``load_ckpt`` calls ``torch.load`` without ``map_location`` on CUDA-saved
weights, so its three lines are repeated here with ``device="cpu"``, and the
load raises if any encoder parameter is missing from the checkpoint.

Usage:
    python monroe_embed.py --smiles-csv <in.csv> --checkpoint-dir <monroe/checkpoint> \\
        --seed 0 --n-workers 1 --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 720)``.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from rdkit.Chem import rdDistGeom
from torch_geometric.data import Batch

_ETKDGv3 = rdDistGeom.ETKDGv3


def main() -> None:
    """Parse arguments, embed the SMILES in input order, save the ``.npy``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    def seeded_etkdg():
        params = _ETKDGv3()
        params.randomSeed = a.seed
        return params

    rdDistGeom.ETKDGv3 = seeded_etkdg
    torch.manual_seed(a.seed)

    from pathlib import Path

    from monroe.eval.embed import featurize_smiles, to_pyg
    from monroe.model import ckpt

    with open(a.smiles_csv, newline="") as f:
        smiles = [row["smiles"] for row in csv.DictReader(f)]
    unique = list(dict.fromkeys(smiles))

    built = dict(featurize_smiles(unique, n_workers=a.n_workers))
    missing = [s for s in unique if s not in built]
    if missing:
        raise ValueError(f"Monroe could not build {len(missing)} graphs, first: {missing[0]}")
    flat = [s for s in unique if np.allclose(np.asarray(built[s]["pos_rdkit"])[:, 2], 0.0)]

    hp_dict, state_dict = ckpt._load_config_and_weights(Path(a.checkpoint_dir), device="cpu")
    holder = torch.nn.Module()
    holder.encoder = ckpt._build_encoder(hp_dict)
    result = holder.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        raise KeyError(f"checkpoint lacks {len(result.missing_keys)} encoder parameters, first: {result.missing_keys[0]}")
    encoder = holder.encoder.eval()
    graphs = [to_pyg(built[s]) for s in unique]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(graphs), 32):
            embedding, _ = encoder(Batch.from_data_list(graphs[start:start + 32]))
            vectors.append(embedding.cpu().numpy())
    by_smiles = dict(zip(unique, np.concatenate(vectors)))
    X = np.stack([by_smiles[s] for s in smiles]).astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Monroe output carries non-finite values")
    np.save(a.out, X)
    with open(f"{a.out}.flat2d.txt", "w") as f:
        f.writelines(f"{s}\n" for s in flat)


if __name__ == "__main__":
    main()
