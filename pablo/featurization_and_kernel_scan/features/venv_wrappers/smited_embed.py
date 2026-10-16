"""Frozen SMI-TED embedding wrapper — executed by the SMI-TED Desktop venv.

Imports the upstream inference module ``load.py`` of
``ibm-research/materials.smi-ted`` (directory given by ``--code-dir``), so it
runs inside ``~/Desktop/smi_ted/.venv`` (``torch``, ``transformers``,
``pytorch-fast-transformers``, ``rdkit``) without touching the main ``pablo``
environment. The orchestrating generator (``features.gnn._run_smited``) invokes
it via subprocess.

SMI-TED-Light (Soares et al., arXiv 2407.20267; 289M parameters) is a SMILES
encoder-decoder pretrained on 91M PubChem SMILES. The frozen embedding is the
documented one, ``Smi_ted.encode``: RDKit canonical SMILES without
stereochemistry (``normalize_smiles``), regex tokenisation against
``bert_vocab_curated.txt``, the encoder's token embeddings, compressed by the
pretrained autoencoder to one 768-d vector per SMILES.

``normalize_smiles`` returns ``None`` for a SMILES RDKit cannot parse, and the
tokenizer truncates silently at ``max_len`` tokens; both are checked here and
raise instead.

Usage:
    python smited_embed.py --smiles-csv <in.csv> --code-dir <dir with load.py> \\
        --checkpoint <smi-ted-Light_40.pt> --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 768)``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    """Parse arguments, embed the SMILES, save the ``.npy``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    sys.path.insert(0, a.code_dir)
    from load import load_smi_ted, normalize_smiles

    with open(a.smiles_csv, newline="") as f:
        smiles = [row["smiles"] for row in csv.DictReader(f)]
    canonical = [normalize_smiles(s) for s in smiles]
    bad = [s for s, c in zip(smiles, canonical) if c is None]
    if bad:
        raise ValueError(f"RDKit could not canonicalise {len(bad)} SMILES, first: {bad[0]}")

    model = load_smi_ted(folder=a.code_dir, ckpt_filename=os.path.abspath(a.checkpoint))
    n_tokens = [len(model.tokenizer.tokenize(c)) + 2 for c in canonical]
    if max(n_tokens) > model.max_len:
        raise ValueError(f"{sum(n > model.max_len for n in n_tokens)} SMILES exceed max_len {model.max_len} tokens")
    unknown = sorted({t for c in canonical for t in model.tokenizer.tokenize(c) if t not in model.tokenizer.vocab})
    if unknown:
        raise ValueError(f"tokens outside the SMI-TED vocabulary: {unknown}")

    with torch.no_grad():
        X = model.encode(smiles, return_torch=True).numpy().astype(np.float64)
    if X.shape[0] != len(smiles) or not np.isfinite(X).all():
        raise ValueError(f"SMI-TED output has shape {X.shape} or non-finite values")
    np.save(a.out, X)


if __name__ == "__main__":
    main()
