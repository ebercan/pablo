"""Foundation-model catalyst embeddings.

Seven featurizers built on five frozen pretrained encoders, all requiring
either a GPU node or the ``transformers`` stack:

    molformer_xl_precat   MolFormer-XL on pre-catalyst SMILES
    molformer_xl_carbene  MolFormer-XL on free-carbene SMILES
    chembert2             ChemBERTa-2 (DeepChem/ChemBERTa-77M-MTR)
    clamp                 CLAMP molecular MLP encoder (768-d)
    unimol                UniMol2 1.1B on free-carbene 3D conformers
    grover                GROVERlarge fingerprints via CLI subprocess
    grover_components     GROVERlarge on the solvent and base libraries

Every embedding is L2-normalised and delivered at full width — no reduction.
``submit`` sends one model to the Mechthild H100 partition; the registry marks
these featurizers ``runtime="gpu"`` so the CLI refuses them on a laptop unless
forced.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features import common

LOGS_DIR = paths.RESULTS_DIR / "foundation_model_logs"

GROVER_REPO_DIR = Path(os.environ.get("GROVER_REPO_DIR", str(Path.home() / "grover")))
GROVER_PYTHON = Path(os.environ.get("GROVER_PYTHON", sys.executable))
GROVER_CHECKPOINT = Path(
    os.environ.get("GROVER_CHECKPOINT", str(paths.WEIGHTS_DIR / "grover_large.pt"))
)
PARTITION = "gpu"
GPU_TYPE = "H100"
GPUS_PER_TASK = 1
CPUS_PER_TASK = 4
MEMORY = "64G"
TIME_LIMIT = "02:00:00"
EMAIL = "enisbaris@mpi-magdeburg.mpg.de"
HPC_PROJECT_ROOT = str(paths.REPO_ROOT)

VALID_MODELS = (
    "molformer_xl_precat",
    "molformer_xl_carbene",
    "chembert2",
    "clamp",
    "unimol",
    "grover",
    "grover_components",
)

def _run_molformer_xl(use_free_carbene: bool = False) -> None:
    """Generate MolFormer-XL embeddings for the 161-molecule library.

    Args:
        use_free_carbene: If True, embed free-carbene SMILES; if False, embed
            pre-catalyst SMILES as-is. No form one-hot is appended in either case.
    """
    # MolFormer-XL: 46.8M params, 12 layers, 768 hidden, linear attention + RoPE.
    # ibm/MoLFormer-XL-both-10pct is the ONLY publicly available variant.
    # The full-scale model (trained on 1.1B molecules) is NOT released.
    model_name_hf = "ibm/MoLFormer-XL-both-10pct"

    import torch
    from transformers import (  # [VERIFY] correct class for MolFormer
        AutoModel,
        AutoTokenizer,
    )

    cat_ids, smiles, form_map = common.load_catalyst_library_with_forms()

    tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_hf, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    input_smiles = smiles
    if use_free_carbene:
        input_smiles = [
            common.to_free_carbene_smiles(s, form_map.get(cid, "unknown"))
            for cid, s in zip(cat_ids, smiles)
        ]

    batch_size = 32
    embeddings_list = []
    for i in range(0, len(input_smiles), batch_size):
        batch = input_smiles[i : i + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
        # Mean-pool the last hidden state over sequence length.
        # [VERIFY]: Confirm pooling strategy in the MolFormer paper / model card.
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings_list.append(emb)

    X_raw = np.concatenate(embeddings_list, axis=0).astype(np.float64)
    X_norm = common.l2_normalize(X_raw)

    label = "molformer_xl_carbene" if use_free_carbene else "molformer_xl_precat"
    out_dir = paths.FEAT_DIR / label
    common.save_embeddings(
        cat_ids=cat_ids,
        raw_model_output=X_raw,
        descriptors=X_norm,
        out_dir=out_dir,
        model_name=label,
        extra_meta={
            "hf_model": model_name_hf,
            "pooling": "mean_last_hidden",
            "free_carbene": use_free_carbene,
            "form_onehot_appended": False,
        },
    )


# ── ChemBERTa-2 ───────────────────────────────────────────────────────────────


def _run_chembert2() -> None:
    """Generate ChemBERTa-2 mean-pooled embeddings for the 161-molecule library."""
    # ChemBERTa-2: RoBERTa-base arch (~83M params), 12 layers, 768 hidden.
    # Available variants: 77M-MTR, 77M-MLM, 100M-MLM.
    # "77M"/"100M" = training data size (molecules), NOT model params.
    # All share the same ~83M-param RoBERTa architecture.
    # 77M-MTR is optimal: MTR (multi-task regression) pretraining on RDKit
    # descriptors produces structurally richer embeddings than MLM.
    model_name_hf = "DeepChem/ChemBERTa-77M-MTR"

    import torch
    from transformers import AutoModel, AutoTokenizer

    cat_ids, smiles, _ = common.load_catalyst_library_with_forms()
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    model = AutoModel.from_pretrained(model_name_hf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    batch_size = 64
    embeddings_list = []
    for i in range(0, len(smiles), batch_size):
        batch = smiles[i : i + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings_list.append(emb)

    X_raw = np.concatenate(embeddings_list, axis=0).astype(np.float64)
    X_norm = common.l2_normalize(X_raw)

    common.save_embeddings(
        cat_ids=cat_ids,
        raw_model_output=X_raw,
        descriptors=X_norm,
        out_dir=paths.FEAT_DIR / "chembert2",
        model_name="ChemBERTa-2",
        extra_meta={"hf_model": model_name_hf, "pooling": "mean_last_hidden"},
    )


# ── CLAMP ─────────────────────────────────────────────────────────────────────


def _run_clamp() -> None:
    """Generate CLAMP molecular encoder embeddings for the 161-molecule library.

    CLAMP (Contrastive Language-Assay Pretrained Models) from Seidl et al.
    (JKU Linz).

    Installation:
        git clone https://github.com/ml-jku/clamp $HOME/clamp
        cd $HOME/clamp && sed -i 's/import pkg_resources//' setup.py && pip install -e .

    CLAMP encodes molecules via concatenated Morgan count + RDKit count
    fingerprints through an MLP encoder. We extract the learned molecular
    embedding from the internal molecule encoder.

    API reference (verified from CLAMP source; FpEncoder is the public path):
        import clamp
        model = clamp.CLAMP(device='cpu')
        model.eval()

        from clamp.dataset.encode_compound import FpEncoder
        fp_encoder = FpEncoder(fp_size=8192, fp_type="morganc+rdkc", radius=2, njobs=4)
        fps = np.asarray(fp_encoder.encode(smiles_list), dtype=np.float32)
        fps_tensor = torch.tensor(fps, device='cpu')
        embeddings = model.compound_encoder(fps_tensor)  # (n, embedding_dim)

    For embeddings, we access the internal compound encoder which transforms
    fingerprint vectors into the shared CLAMP embedding space.
    """
    import torch

    cat_ids, smiles, _ = common.load_catalyst_library_with_forms()

    import clamp as clamp_pkg  # type: ignore[import]

    model = clamp_pkg.CLAMP(device="cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)
    model.eval()

    # CLAMP exposes fingerprinting via FpEncoder; use it as the public interface.
    from clamp.dataset.encode_compound import FpEncoder  # type: ignore[import]

    fp_encoder = FpEncoder(fp_size=8192, fp_type="morganc+rdkc", radius=2, njobs=4)
    fps = np.asarray(fp_encoder.encode(smiles), dtype=np.float32)
    fps_tensor = torch.tensor(fps, device=model.device)

    with torch.no_grad():
        X_raw = model.compound_encoder(fps_tensor).cpu().numpy().astype(np.float64)

    X_norm = common.l2_normalize(X_raw)

    common.save_embeddings(
        cat_ids=cat_ids,
        raw_model_output=X_raw,
        descriptors=X_norm,
        out_dir=paths.FEAT_DIR / "clamp",
        model_name="CLAMP",
        extra_meta={
            "model": "clamp-pretrained",
            "encoding": "compound_encoder_output",
            "input_fp": "morganc+rdkc_8192",
        },
    )


# ── UniMol ────────────────────────────────────────────────────────────────────


def _run_unimol() -> None:
    """Generate UniMol2-1.1B embeddings for the 161-molecule library.

    Uses the largest available UniMol2 checkpoint (1.1 billion parameters).
    At FP16 inference on 161 molecules this needs ~3-4GB VRAM — trivial for H100.
    Uses free-carbene SMILES (UniMol generates 3D conformers internally).
    The saved descriptors are the L2-normalised CLS embeddings, with no form
    one-hot appended.

    Tested with unimol_tools 0.1.5 (the version pinned in requirements_hpc.txt):
        pip install unimol_tools==0.1.5

    UniMol2 API (verified; version-skew note):
        from unimol_tools import UniMolRepr
        clf = UniMolRepr(
            data_type='molecule', remove_hs=False,
            model_name='unimolv2', model_size='1.1B',
        )
        result = clf.get_repr(smiles_list, return_atomic_reprs=False)
        # unimol_tools 0.1.5 returns a list; newer versions return a dict.
        cls_data = result if isinstance(result, list) else result['cls_repr']
        cls_embeddings = np.asarray(cls_data)

    Available UniMol2 sizes: 84m, 164m, 310m, 570m, 1.1B.
    We use 1.1B — the maximum. Fits H100 80GB VRAM with room to spare.
    """
    cat_ids, smiles, form_map = common.load_catalyst_library_with_forms()

    from unimol_tools import UniMolRepr  # type: ignore[import]

    clf = UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        model_name="unimolv2",
        model_size="1.1B",
    )

    free_smiles = [
        common.to_free_carbene_smiles(s, form_map.get(cid, "unknown"))
        for cid, s in zip(cat_ids, smiles)
    ]

    result = clf.get_repr(free_smiles, return_atomic_reprs=False)
    # unimol_tools 0.1.5 returns a list of cls embeddings (one per molecule);
    # newer versions return {"cls_repr": ..., "atomic_reprs": ...}. Handle both.
    if isinstance(result, dict):
        cls_data = result['cls_repr']
    elif isinstance(result, (list, tuple, np.ndarray)):
        cls_data = result
    else:
        raise TypeError(
            f"UniMolRepr.get_repr returned unexpected type {type(result).__name__}; "
            "expected dict, list, tuple, or ndarray."
        )
    X_raw = np.asarray(cls_data, dtype=np.float64)
    assert X_raw.ndim == 2 and X_raw.shape[0] == len(free_smiles), (
        f"UniMol cls matrix has shape {X_raw.shape}; "
        f"expected ({len(free_smiles)}, D)."
    )
    X_norm = common.l2_normalize(X_raw)

    common.save_embeddings(
        cat_ids=cat_ids,
        raw_model_output=X_raw,
        descriptors=X_norm,
        out_dir=paths.FEAT_DIR / "unimol",
        model_name="UniMol2-1.1B",
        extra_meta={
            "model": "unimolv2",
            "model_size": "1.1B",
            "params": "1.1 billion",
            "free_carbene": True,
            "form_onehot_appended": False,
        },
    )


# ── GROVER ─────────────────────────────────────────────────────────────────────


def _grover_fingerprints(smiles: list[str], tmp_dir: Path) -> np.ndarray:
    """Run GROVER-large fingerprint mode on a list of SMILES.

    GROVER (Rong et al., NeurIPS 2020) is driven through its own CLI by
    ``venv_wrappers/grover_embed.py``, the same wrapper the Desktop GROVER forms
    in ``features.gnn`` use: 200 ``rdkit_2d_normalized`` descriptors, then
    ``main.py fingerprint --fingerprint_source both`` on a patched copy of the
    checkpoint (the downloaded file is left unchanged).

    Setup (``PROVENANCE.md`` records the versions the stored matrices came from):
        git clone https://github.com/tencent-ailab/grover $GROVER_REPO_DIR
        pip install descriptastorus tensorboard   (into $GROVER_PYTHON's env)
        gdown 1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw -O $GROVER_CHECKPOINT

    ``GROVER_PYTHON`` defaults to the interpreter running this module and
    ``GROVER_CHECKPOINT`` to ``data/weights/grover_large.pt``.

    Args:
        smiles: SMILES strings, one fingerprint row each.
        tmp_dir: Working directory for the CSV and intermediate files; one per
            library, so two libraries never overwrite each other's intermediates.

    Returns:
        ``(len(smiles), 5000)`` float64 array of unnormalised fingerprints.
    """
    grover_dir = GROVER_REPO_DIR
    checkpoint = GROVER_CHECKPOINT
    assert grover_dir.exists(), (
        f"GROVER repo not found at {grover_dir}. "
        f"Clone it: git clone https://github.com/tencent-ailab/grover {grover_dir}"
    )
    assert checkpoint.exists(), (
        f"GROVER checkpoint not found at {checkpoint}. Download GROVERlarge from: "
        "https://drive.google.com/file/d/1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw"
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    smiles_csv = tmp_dir / "smiles_input.csv"
    with open(smiles_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for s in smiles:
            writer.writerow([s])

    out_npy = tmp_dir / "fingerprints.npy"
    subprocess.run(
        [
            str(GROVER_PYTHON),
            str(paths.PKG_ROOT / "features" / "venv_wrappers" / "grover_embed.py"),
            "--smiles-csv", str(smiles_csv),
            "--repo-dir", str(grover_dir),
            "--checkpoint", str(checkpoint),
            "--workdir", str(tmp_dir),
            "--out", str(out_npy),
        ],
        check=True,
    )
    X_raw = np.load(out_npy).astype(np.float64)
    assert X_raw.shape[0] == len(
        smiles
    ), f"GROVER output has {X_raw.shape[0]} rows, expected {len(smiles)}"
    return X_raw


def _run_grover() -> None:
    """Generate GROVERlarge molecular fingerprints for the 161-molecule library."""
    cat_ids, smiles, _ = common.load_catalyst_library_with_forms()
    X_raw = _grover_fingerprints(smiles, paths.DATA_DIR / "grover_tmp")
    X_norm = common.l2_normalize(X_raw)

    common.save_embeddings(
        cat_ids=cat_ids,
        raw_model_output=X_raw,
        descriptors=X_norm,
        out_dir=paths.FEAT_DIR / "grover",
        model_name="GROVERlarge",
        extra_meta={
            "model": "grover_large",
            "params": "100M",
            "fingerprint_source": "both",
            "checkpoint": str(GROVER_CHECKPOINT),
        },
    )


def _run_grover_components() -> None:
    """GROVERlarge fingerprints of the 14 solvents and the 8 bases.

    Each library goes through the CLI on its own, as the stored SMILES (salts
    with their counter-cations), and is L2-normalised like the catalyst
    embedding. Writes ``solvent_grover/`` and ``base_grover/``, indexed by
    ``solvent_id`` / ``base_id``.
    """
    for kind, library_path, index_name in (
        ("solvent", paths.SOLVENTS_PARQUET, "solvent_id"),
        ("base", paths.BASES_PARQUET, "base_id"),
    ):
        df = pd.read_parquet(library_path)
        X_raw = _grover_fingerprints(
            df["smiles"].tolist(), paths.DATA_DIR / "grover_tmp" / kind
        )
        common.save_embeddings(
            cat_ids=df["id"].tolist(),
            raw_model_output=X_raw,
            descriptors=common.l2_normalize(X_raw),
            out_dir=paths.FEAT_DIR / f"{kind}_grover",
            model_name="GROVERlarge",
            extra_meta={
                "model": "grover_large",
                "params": "100M",
                "fingerprint_source": "both",
                "checkpoint": str(GROVER_CHECKPOINT),
                "names": df["name"].tolist(),
                "input_smiles": "library SMILES, salts with counter-cations",
            },
            index_name=index_name,
        )


# ── SLURM submission ──────────────────────────────────────────────────────────


def _submit(model: str) -> None:
    """Submit a foundation model embedding job to the Mechthild GPU partition.

    Args:
        model: One of VALID_MODELS.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    paths.RESULTS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    module_path = "pablo.featurization_and_kernel_scan.features"

    # Relative to the project root set by --chdir. Must match LOGS_DIR, which is
    # created above: SLURM cannot open -o/-e in a directory that does not exist,
    # and the job dies before Python starts.
    log_dir_rel = str(LOGS_DIR.relative_to(paths.REPO_ROOT))

    slurm_script = f"""#!/bin/bash
#SBATCH -J fks_{model}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --gpus={GPU_TYPE}:{GPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user={EMAIL}
#SBATCH -o {log_dir_rel}/{model}_%j.out
#SBATCH -e {log_dir_rel}/{model}_%j.err
#SBATCH -D {HPC_PROJECT_ROOT}

# Load CUDA runtime
module load cuda/12.6.2

# Activate virtual environment
source venv/bin/activate

# Pin threads to allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# HuggingFace cache — pre-download models on login node first!
export HF_HOME=$HOME/.cache/huggingface

# Diagnostics
echo "Foundation model {model} on $SLURMD_NODENAME (GPU: $CUDA_VISIBLE_DEVICES)"
nvidia-smi

python -m {module_path} build {model} --force
"""
    script_path = paths.RESULTS_LOGS_DIR / f"submit_{model}.sh"
    script_path.write_text(slurm_script)
    print(f"Submitting {model} to partition '{PARTITION}' (GPU: {GPU_TYPE}) ...")
    subprocess.run(["sbatch", str(script_path)], check=True)
    print("Submitted.")



_DISPATCH: dict[str, "object"] = {
    "molformer_xl_precat": lambda: _run_molformer_xl(use_free_carbene=False),
    "molformer_xl_carbene": lambda: _run_molformer_xl(use_free_carbene=True),
    "chembert2": _run_chembert2,
    "clamp": _run_clamp,
    "unimol": _run_unimol,
    "grover": _run_grover,
    "grover_components": _run_grover_components,
}


def run_model(model: str) -> None:
    """Compute and write one foundation model's embeddings.

    Args:
        model: A member of ``VALID_MODELS``.

    Raises:
        ValueError: If the model is not known.
    """
    if model not in _DISPATCH:
        raise ValueError(f"Unknown foundation model {model!r}; expected one of {VALID_MODELS}")
    _DISPATCH[model]()


def submit(model: str) -> None:
    """Submit one model's embedding job to the Mechthild H100 partition.

    Args:
        model: A member of ``VALID_MODELS``.

    Raises:
        ValueError: If the model is not known.
    """
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown foundation model {model!r}; expected one of {VALID_MODELS}")
    _submit(model)
