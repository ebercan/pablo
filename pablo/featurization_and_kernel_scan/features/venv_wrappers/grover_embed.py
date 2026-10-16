"""Frozen GROVER-large fingerprint wrapper — executed by the GROVER interpreter.

Self-contained: imports only ``torch``, ``numpy`` and the standard library, so
it runs inside the GROVER Desktop venv (``~/Desktop/grover/.venv``) or any
interpreter with GROVER's dependencies (``descriptastorus``, ``rdkit``,
``tensorboard``). The orchestrating generators
(``features.foundation._grover_fingerprints`` and ``features.gnn._run_grover``)
invoke it via subprocess.

It runs the two steps of GROVER's own CLI (tencent-ailab/grover, README
"Fingerprint generation") with the calling interpreter:

  1. ``scripts/save_features.py --features_generator rdkit_2d_normalized``:
     the 200 descriptastorus RDKit 2D descriptors, CDF-normalised to [0, 1].
  2. ``main.py fingerprint --fingerprint_source both``: mean readout of the
     four embedding sets ``atom_from_atom``, ``atom_from_bond``,
     ``bond_from_atom``, ``bond_from_bond`` (1200 each for GROVER-large),
     concatenated in that order, followed by the 200 descriptors.
     Atom-set means run over atoms, bond-set means over bonds, so a bond-less
     counter-ion (``[Cl-]``) enters the two atom sets only.

The released checkpoint predates arguments that the current code reads. They
are added to an in-memory copy, written next to the checkpoint as
``<checkpoint stem>.patched.pt`` (outside the repository); the downloaded
checkpoint is never modified, so its SHA-256 stays verifiable.

Usage:
    python grover_embed.py --smiles-csv <in.csv> --repo-dir <grover_repo> \\
        --checkpoint <grover_large.pt> --workdir <dir> --out <out.npy>

Output: float64 ``.npy`` array of shape ``(n_molecules, 5000)``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_ARG_DEFAULTS = {
    "dropout": 0.0,
    "nencoders": 3,
    "ffn_num_layers": 2,
    "attn_hidden": 4,
    "attn_out": 128,
    "aug_rate": 0.0,
    "model_type": "classification",
    "ensemble_size": 1,
    "distinct_init": False,
    "atom_message": False,
    "coord": False,
}
"""Arguments missing from the released checkpoint's ``args`` Namespace, with the
values ``features.foundation`` has always used. ``ffn_hidden_size`` is set to the
checkpoint's ``hidden_size``."""


def patched_checkpoint(checkpoint: Path) -> Path:
    """Write a copy of ``checkpoint`` whose ``args`` carry every argument the code reads.

    Args:
        checkpoint: The downloaded GROVER checkpoint (left unchanged); the copy
            is written beside it.

    Returns:
        Path of the patched copy.

    Raises:
        KeyError: If the checkpoint has no ``args`` entry.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"]
    for key, value in CHECKPOINT_ARG_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, "ffn_hidden_size"):
        args.ffn_hidden_size = args.hidden_size
    out = checkpoint.with_name(f"{checkpoint.stem}.patched.pt")
    torch.save(ck, out)
    return out


def main() -> None:
    """Parse arguments, run GROVER's feature and fingerprint CLIs, save ``.npy``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smiles-csv", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    repo = Path(a.repo_dir).resolve()
    workdir = Path(a.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    smiles_csv = Path(a.smiles_csv).resolve()
    features_npz = workdir / "features.npz"
    fp_npz = workdir / "fingerprints.npz"
    checkpoint = patched_checkpoint(Path(a.checkpoint).resolve())
    env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}

    subprocess.run(
        [sys.executable, str(repo / "scripts" / "save_features.py"),
         "--data_path", str(smiles_csv), "--save_path", str(features_npz),
         "--features_generator", "rdkit_2d_normalized", "--restart"],
        check=True, cwd=str(repo), env=env,
    )
    subprocess.run(
        [sys.executable, str(repo / "main.py"), "fingerprint",
         "--data_path", str(smiles_csv), "--features_path", str(features_npz),
         "--checkpoint_path", str(checkpoint), "--fingerprint_source", "both",
         "--output", str(fp_npz)],
        check=True, cwd=str(repo), env=env,
    )
    X = np.load(fp_npz)["fps"].astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("GROVER output carries non-finite values")
    np.save(a.out, X)


if __name__ == "__main__":
    main()
