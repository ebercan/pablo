"""Frozen CheMeleon embedding wrapper — executed by the CheMeleon Desktop venv.

Self-contained: imports only ``torch``, ``numpy``, ``rdkit``, ``chemprop`` and
the standard library, so it runs inside ``~/Desktop/chemeleon/.venv`` without
touching the main ``pablo`` environment. The orchestrating generator
(``features.gnn._run_chemeleon``) invokes it via subprocess.

CheMeleon (Burns, Zalte, Green, arXiv 2506.15792) is a Chemprop D-MPNN
(``BondMessagePassing``) pretrained to predict 1,613 Mordred descriptors. This
wrapper is ``chemeleon_fingerprint.py`` of github.com/JacksonBurns/chemeleon with
the checkpoint path passed explicitly instead of downloaded to ``~/.chemprop``:

  - featurizer : ``chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()``
    (multi-hot atom features incl. formal charge, atomic numbers 1-36 and 53),
    on ``rdkit.Chem.MolFromSmiles`` of the SMILES as given. A ``.``-separated
    salt is one disconnected graph; a bond-less ion receives no messages and
    enters the readout with its input features only.
  - encoder    : ``BondMessagePassing(**hyper_parameters)`` with the checkpoint's
    ``state_dict``, ``MeanAggregation`` readout, ``MPNN.fingerprint`` (the
    unused regression head is freshly initialised and never called).

Usage:
    python chemeleon_embed.py --smiles-csv <in.csv> --checkpoint <chemeleon_mp.pt> \\
        --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 2048)``.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import MolFromSmiles


def main() -> None:
    """Parse arguments, embed the SMILES, save the ``.npy``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    with open(a.smiles_csv, newline="") as f:
        smiles = [row["smiles"] for row in csv.DictReader(f)]
    mols = [MolFromSmiles(s) for s in smiles]
    bad = [s for s, m in zip(smiles, mols) if m is None]
    if bad:
        raise ValueError(f"RDKit could not parse {len(bad)} SMILES, first: {bad[0]}")

    ck = torch.load(a.checkpoint, weights_only=True)
    mp = nn.BondMessagePassing(**ck["hyper_parameters"])
    mp.load_state_dict(ck["state_dict"])
    model = MPNN(message_passing=mp, agg=nn.MeanAggregation(),
                 predictor=RegressionFFN(input_dim=mp.output_dim))
    model.eval()

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([featurizer(m) for m in mols])
    with torch.no_grad():
        X = model.fingerprint(bmg).numpy(force=True).astype(np.float64)
    if X.shape[0] != len(smiles) or not np.isfinite(X).all():
        raise ValueError(f"CheMeleon output has shape {X.shape} or non-finite values")
    np.save(a.out, X)


if __name__ == "__main__":
    main()
