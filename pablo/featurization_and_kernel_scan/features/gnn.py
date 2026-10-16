"""Graph-neural-network catalyst embeddings for the 161-molecule library.

Six open graph models are frozen featurizers alongside GROVER (see
``featurizations.html`` for the featurizer overview):

  scage        : SCAGE — graph Transformer + multiscale conformation (KazeDog/scage)
  minimol      : MiniMol — supervised multi-task MPNN, ~10M params (graphcore-research/minimol)
  graphormer   : Graphormer — pure graph Transformer pretrained on PCQM4Mv2 HOMO-LUMO,
                 via the fairseq-free Hugging Face port (clefourrier/graphormer-base-pcqm4mv2)
  infomax3d    : 3D-Infomax — PNA MPNN, 3D-distilled into a 2D net (HannesStark/3DInfomax)
  frad         : Frad — TorchMD-NET SE(3) transformer, fractional denoising (fengshikun/Frad)
  transformerm : Transformer-M — dual 2D/3D Graphormer-style transformer (lsj2408/Transformer-M)
  grover       : GROVER-large — fingerprint mode, 4 x 1200 mean-pooled embeddings + 200
                 RDKit 2D descriptors (tencent-ailab/grover); ``carbene`` and
                 ``ionstripped`` forms here, the pre-catalyst form is ``foundation.grover``
  chemeleon    : CheMeleon — Chemprop D-MPNN pretrained on Mordred descriptors (JacksonBurns/chemeleon)
  monroe       : Monroe — GRIT transformer pretrained on PM6 + PCBA + conformer denoising (blazejba/monroe)
  smited       : SMI-TED-Light — SMILES encoder-decoder, 289M (ibm-research/materials.smi-ted)

Each model is run in **two forms**, mirroring ``molformer_xl_precat`` /
``molformer_xl_carbene``: a ``_precat`` variant on the raw pre-catalyst (salt)
SMILES, and a ``_carbene`` variant on the free-carbene SMILES. No pre-catalyst-form
one-hot is appended. GROVER adds an ``_ionstripped`` form: the pre-catalyst with
its counter-ion removed (``common.strip_counterion_smiles``). Twenty featurizers
total; how each model was installed and run is recorded in ``PROVENANCE.md``.

Each model is installed in its **own virtualenv on the Desktop** (dependency
conflicts keep them out of the main env). Five are Python 3.12; Transformer-M is
Python 3.10 (its fairseq dependency has no cp312 wheel). This module runs in the
main env: it builds the input SMILES, shells out to the per-model Desktop venv to
compute the raw embedding, then L2-normalises it row-wise. The subprocess +
``.npy`` contract is Python-version-agnostic, so the py3.10 venv talks to this
py3.12 module unchanged.

Desktop layout (overridable via the ``*_REPO_DIR`` env vars):
    ~/Desktop/scage/.venv/bin/python         (clone + torch 2.6 + Cython build_ext)
    ~/Desktop/minimol/.venv/bin/python        (pip install minimol)
    ~/Desktop/graphormer/.venv/bin/python     (transformers + ogb; HF Graphormer port)
    ~/Desktop/3dinfomax/.venv/bin/python       (DGL + PyG; torch 2.2.1; in-repo weights)
    ~/Desktop/frad/.venv/bin/python            (TorchMD-NET; torch 2.6; GDrive checkpoint)
    ~/Desktop/transformer_m/.venv/bin/python   (Python 3.10; vendored fairseq; L12 checkpoint)
    ~/Desktop/grover/.venv/bin/python          (clone + descriptastorus; weights/grover_large.pt)
    ~/Desktop/chemeleon/.venv/bin/python       (chemprop; weights/chemeleon_mp.pt from Zenodo)
    ~/Desktop/monroe/.venv/bin/python          (pip install -e; checkpoint/ from Git LFS)
    ~/Desktop/smi_ted/.venv/bin/python         (pytorch-fast-transformers; hf_snapshot/ from HF)

The thin inference wrappers run BY those venvs live in ``features/venv_wrappers/``.

Usage:
    python -m pablo.featurization_and_kernel_scan.features build minimol_precat
    python -m pablo.featurization_and_kernel_scan.features build --family gnn

Outputs per model (under data/featurizations/<model>/) match every other
featurizer: raw_model_output.parquet, descriptors.parquet, metadata.json.
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import numpy as np

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features import common

WRAPPER_DIR = paths.PKG_ROOT / "features" / "venv_wrappers"
VALID_MODELS = (
    "minimol_precat",
    "minimol_carbene",
    "scage_precat",
    "scage_carbene",
    "graphormer_precat",
    "graphormer_carbene",
    "infomax3d_precat",
    "infomax3d_carbene",
    "frad_precat",
    "frad_carbene",
    "transformerm_precat",
    "transformerm_carbene",
    "grover_carbene",
    "grover_ionstripped",
    "chemeleon_precat",
    "chemeleon_carbene",
    "monroe_precat",
    "monroe_carbene",
    "smited_precat",
    "smited_carbene",
)

SCAGE_REPO_DIR = Path(os.environ.get("SCAGE_REPO_DIR", str(Path.home() / "Desktop" / "scage")))
MINIMOL_REPO_DIR = Path(
    os.environ.get("MINIMOL_REPO_DIR", str(Path.home() / "Desktop" / "minimol"))
)
GRAPHORMER_REPO_DIR = Path(
    os.environ.get("GRAPHORMER_REPO_DIR", str(Path.home() / "Desktop" / "graphormer"))
)
INFOMAX3D_REPO_DIR = Path(
    os.environ.get("INFOMAX3D_REPO_DIR", str(Path.home() / "Desktop" / "3dinfomax"))
)
FRAD_REPO_DIR = Path(
    os.environ.get("FRAD_REPO_DIR", str(Path.home() / "Desktop" / "frad"))
)
FRAD_CHECKPOINT = Path(
    os.environ.get(
        "FRAD_CHECKPOINT", str(FRAD_REPO_DIR / "frad_pretrained.pt")
    )
)
TRANSFORMERM_REPO_DIR = Path(
    os.environ.get("TRANSFORMERM_REPO_DIR", str(Path.home() / "Desktop" / "transformer_m"))
)
TRANSFORMERM_CHECKPOINT = Path(
    os.environ.get(
        "TRANSFORMERM_CHECKPOINT", str(TRANSFORMERM_REPO_DIR / "L12.pt")
    )
)

GROVER_REPO_DIR = Path(
    os.environ.get("GROVER_REPO_DIR", str(Path.home() / "Desktop" / "grover"))
)
GROVER_CHECKPOINT = Path(
    os.environ.get("GROVER_CHECKPOINT", str(GROVER_REPO_DIR / "weights" / "grover_large.pt"))
)
CHEMELEON_REPO_DIR = Path(
    os.environ.get("CHEMELEON_REPO_DIR", str(Path.home() / "Desktop" / "chemeleon"))
)
CHEMELEON_CHECKPOINT = Path(
    os.environ.get("CHEMELEON_CHECKPOINT", str(CHEMELEON_REPO_DIR / "weights" / "chemeleon_mp.pt"))
)
MONROE_REPO_DIR = Path(
    os.environ.get("MONROE_REPO_DIR", str(Path.home() / "Desktop" / "monroe"))
)
MONROE_SEED: int = 0
"""ETKDGv3 seed of Monroe's conformer embedding (``monroe_embed.py``)."""
SMITED_REPO_DIR = Path(
    os.environ.get("SMITED_REPO_DIR", str(Path.home() / "Desktop" / "smi_ted"))
)

# 3D-Infomax pretrained checkpoint ships in the repo's runs/ dir; overridable.
INFOMAX3D_CHECKPOINT = Path(
    os.environ.get(
        "INFOMAX3D_CHECKPOINT",
        str(
            INFOMAX3D_REPO_DIR
            / "runs"
            / "PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52"
            / "best_checkpoint_35epochs.pt"
        ),
    )
)

# SCAGE pretrained checkpoint (ships in the repo's weights/pretrain/) + finetune
# config; overridable so nothing is hard-wired.
SCAGE_CHECKPOINT = Path(
    os.environ.get(
        "SCAGE_CHECKPOINT", str(SCAGE_REPO_DIR / "weights" / "pretrain" / "pretrain.pth")
    )
)
SCAGE_CONFIG = Path(
    os.environ.get("SCAGE_CONFIG", str(SCAGE_REPO_DIR / "config" / "config_finetune.yaml"))
)


def _venv_python(repo_dir: Path) -> Path:
    """Return the interpreter path for a model's Desktop virtualenv.

    Args:
        repo_dir: The model's Desktop directory.

    Returns:
        Path to ``<repo_dir>/.venv/bin/python``.
    """
    return repo_dir / ".venv" / "bin" / "python"


def _write_smiles_csv(csv_path: Path, smiles: list[str]) -> None:
    """Write a single-column ``smiles`` CSV for the wrapper to read.

    Args:
        csv_path: Destination CSV path.
        smiles: SMILES strings in catalyst-ID order.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for s in smiles:
            writer.writerow([s])


def _generate(
    model_name: str,
    label: str,
    venv_python: Path,
    wrapper: Path,
    extra_args: list[str],
    extra_meta: dict,
    use_free_carbene: bool,
    strip_counterion: bool = False,
) -> None:
    """Run a Desktop-venv embedding wrapper and write all featurizer artifacts.

    Embeds the library via ``venv_python`` running ``wrapper``, then L2-normalises
    the ``(161, d_raw)`` output and delegates persistence to
    ``common.save_embeddings``. Two forms are supported, mirroring
    ``molformer_xl_precat`` / ``molformer_xl_carbene``:

      - ``use_free_carbene=False``: embed the raw pre-catalyst (salt) SMILES as-is.
      - ``use_free_carbene=True``: embed the free-carbene SMILES.

    No form one-hot is appended in either form.

    Args:
        model_name: Featurizer key (also the output subdirectory name).
        label: Human-readable model label for metadata.
        venv_python: Interpreter of the model's Desktop venv.
        wrapper: Path to the model's ``venv_wrappers`` script.
        extra_args: Model-specific CLI args inserted before ``--out``.
        extra_meta: Extra keys merged into ``metadata.json``.
        use_free_carbene: Whether to deprotonate to the free carbene before
            embedding, or embed the raw salt SMILES.
        strip_counterion: Embed the pre-catalyst without its counter-ion
            (``common.strip_counterion_smiles``). Exclusive with
            ``use_free_carbene``.

    Raises:
        ValueError: If both ``use_free_carbene`` and ``strip_counterion`` are set.
    """
    if use_free_carbene and strip_counterion:
        raise ValueError("use_free_carbene and strip_counterion are exclusive")
    cat_ids, smiles, form_map = common.load_catalyst_library_with_forms()
    if use_free_carbene:
        input_smiles = [
            common.to_free_carbene_smiles(s, form_map.get(cid, "unknown"))
            for cid, s in zip(cat_ids, smiles)
        ]
    elif strip_counterion:
        input_smiles = [common.strip_counterion_smiles(s) for s in smiles]
    else:
        input_smiles = smiles

    tmp_dir = paths.DATA_DIR / f"{model_name}_tmp"
    csv_path = tmp_dir / "smiles_input.csv"
    out_npy = tmp_dir / "embeddings.npy"
    _write_smiles_csv(csv_path, input_smiles)

    assert venv_python.exists(), (
        f"{label} venv interpreter not found at {venv_python}. Create the Desktop "
        f"venv (Python 3.12) per README §3.2 before running."
    )
    assert wrapper.exists(), f"Inference wrapper not found at {wrapper}."

    cmd = [
        str(venv_python),
        str(wrapper),
        "--smiles-csv",
        str(csv_path),
        *extra_args,
        "--out",
        str(out_npy),
    ]
    subprocess.run(cmd, check=True)

    X_raw = np.load(out_npy).astype(np.float64)
    assert X_raw.shape[0] == len(cat_ids), (
        f"{label} returned {X_raw.shape[0]} rows, expected {len(cat_ids)}"
    )

    X_norm = common.l2_normalize(X_raw)

    common.save_embeddings(
        cat_ids=cat_ids,
        raw_model_output=X_raw,
        descriptors=X_norm,
        out_dir=paths.FEAT_DIR / model_name,
        model_name=label,
        extra_meta={
            **extra_meta,
            "free_carbene": use_free_carbene,
            "counterion_stripped": strip_counterion,
            "form_onehot_appended": False,
        },
    )


def _run_minimol(use_free_carbene: bool) -> None:
    """Generate MiniMol (graphcore-research/minimol) embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"minimol_{suffix}",
        label=f"MiniMol-{suffix}",
        venv_python=_venv_python(MINIMOL_REPO_DIR),
        wrapper=WRAPPER_DIR / "minimol_embed.py",
        extra_args=[],
        extra_meta={"repo": "graphcore-research/minimol", "embedding_dim": 512},
        use_free_carbene=use_free_carbene,
    )


def _run_graphormer(use_free_carbene: bool) -> None:
    """Generate Graphormer (HF clefourrier/graphormer-base-pcqm4mv2) embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"graphormer_{suffix}",
        label=f"Graphormer-{suffix}",
        venv_python=_venv_python(GRAPHORMER_REPO_DIR),
        wrapper=WRAPPER_DIR / "graphormer_embed.py",
        extra_args=[],
        extra_meta={
            "hf_model": "clefourrier/graphormer-base-pcqm4mv2",
            "pretraining": "PCQM4Mv2 HOMO-LUMO gap",
            "embedding": "graph CLS token (768-d)",
        },
        use_free_carbene=use_free_carbene,
    )


def _run_scage(use_free_carbene: bool) -> None:
    """Generate SCAGE (KazeDog/scage) embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    assert SCAGE_CHECKPOINT.exists(), (
        f"SCAGE checkpoint not found at {SCAGE_CHECKPOINT}. The pretrained weights "
        "ship in the repo's weights/pretrain/; set SCAGE_CHECKPOINT (see README §3.2)."
    )
    assert SCAGE_CONFIG.exists(), (
        f"SCAGE config not found at {SCAGE_CONFIG}. Point SCAGE_CONFIG at the repo's "
        "finetune config YAML."
    )
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"scage_{suffix}",
        label=f"SCAGE-{suffix}",
        venv_python=_venv_python(SCAGE_REPO_DIR),
        wrapper=WRAPPER_DIR / "scage_embed.py",
        extra_args=[
            "--repo-dir",
            str(SCAGE_REPO_DIR),
            "--checkpoint",
            str(SCAGE_CHECKPOINT),
            "--config",
            str(SCAGE_CONFIG),
        ],
        extra_meta={"repo": "KazeDog/scage", "checkpoint": str(SCAGE_CHECKPOINT)},
        use_free_carbene=use_free_carbene,
    )


def _run_infomax3d(use_free_carbene: bool) -> None:
    """Generate 3D-Infomax (HannesStark/3DInfomax) PNA embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    assert INFOMAX3D_CHECKPOINT.exists(), (
        f"3D-Infomax checkpoint not found at {INFOMAX3D_CHECKPOINT}. The pretrained "
        "weights ship in the repo's runs/ dir; set INFOMAX3D_CHECKPOINT (see README §3.2)."
    )
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"infomax3d_{suffix}",
        label=f"3DInfomax-{suffix}",
        venv_python=_venv_python(INFOMAX3D_REPO_DIR),
        wrapper=WRAPPER_DIR / "infomax3d_embed.py",
        extra_args=[
            "--repo-dir",
            str(INFOMAX3D_REPO_DIR),
            "--checkpoint",
            str(INFOMAX3D_CHECKPOINT),
        ],
        extra_meta={
            "repo": "HannesStark/3DInfomax",
            "checkpoint": str(INFOMAX3D_CHECKPOINT),
            "embedding_dim": 256,
        },
        use_free_carbene=use_free_carbene,
    )


def _run_frad(use_free_carbene: bool) -> None:
    """Generate Frad (fengshikun/Frad) TorchMD-NET representation embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    assert FRAD_CHECKPOINT.exists(), (
        f"Frad checkpoint not found at {FRAD_CHECKPOINT}. Download the pretrained "
        "PCQM4Mv2 denoising checkpoint and set FRAD_CHECKPOINT (see README §3.2)."
    )
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"frad_{suffix}",
        label=f"Frad-{suffix}",
        venv_python=_venv_python(FRAD_REPO_DIR),
        wrapper=WRAPPER_DIR / "frad_embed.py",
        extra_args=[
            "--repo-dir",
            str(FRAD_REPO_DIR),
            "--checkpoint",
            str(FRAD_CHECKPOINT),
        ],
        extra_meta={
            "repo": "fengshikun/Frad",
            "checkpoint": str(FRAD_CHECKPOINT),
            "embedding_dim": 256,
        },
        use_free_carbene=use_free_carbene,
    )


def _run_transformerm(use_free_carbene: bool) -> None:
    """Generate Transformer-M (lsj2408/Transformer-M) graph-token embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    assert TRANSFORMERM_CHECKPOINT.exists(), (
        f"Transformer-M checkpoint not found at {TRANSFORMERM_CHECKPOINT}. Download "
        "the L12 checkpoint and set TRANSFORMERM_CHECKPOINT (see README §3.2)."
    )
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"transformerm_{suffix}",
        label=f"Transformer-M-{suffix}",
        venv_python=_venv_python(TRANSFORMERM_REPO_DIR),
        wrapper=WRAPPER_DIR / "transformerm_embed.py",
        extra_args=[
            "--repo-dir",
            str(TRANSFORMERM_REPO_DIR),
            "--checkpoint",
            str(TRANSFORMERM_CHECKPOINT),
        ],
        extra_meta={
            "repo": "lsj2408/Transformer-M",
            "checkpoint": str(TRANSFORMERM_CHECKPOINT),
            "embedding": "graph/virtual token (768-d)",
            "embedding_dim": 768,
        },
        use_free_carbene=use_free_carbene,
    )




def _run_grover(form: str) -> None:
    """Generate GROVER-large fingerprints (tencent-ailab/grover) in the Desktop venv.

    The pre-catalyst form is ``foundation.grover``; both use ``grover_embed.py``.

    Args:
        form: ``"carbene"`` for free-carbene SMILES, ``"ionstripped"`` for the
            pre-catalyst without its counter-ion.

    Raises:
        ValueError: If ``form`` is neither.
    """
    if form not in ("carbene", "ionstripped"):
        raise ValueError(f"Unknown GROVER form {form!r}")
    model_name = f"grover_{form}"
    _generate(
        model_name=model_name,
        label=f"GROVERlarge-{form}",
        venv_python=_venv_python(GROVER_REPO_DIR),
        wrapper=WRAPPER_DIR / "grover_embed.py",
        extra_args=[
            "--repo-dir", str(GROVER_REPO_DIR),
            "--checkpoint", str(GROVER_CHECKPOINT),
            "--workdir", str(paths.DATA_DIR / f"{model_name}_tmp" / "grover_work"),
        ],
        extra_meta={"repo": "tencent-ailab/grover", "model": "grover_large",
                    "fingerprint_source": "both", "embedding_dim": 5000},
        use_free_carbene=form == "carbene",
        strip_counterion=form == "ionstripped",
    )


def _run_chemeleon(use_free_carbene: bool) -> None:
    """Generate CheMeleon (JacksonBurns/chemeleon) embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"chemeleon_{suffix}",
        label=f"CheMeleon-{suffix}",
        venv_python=_venv_python(CHEMELEON_REPO_DIR),
        wrapper=WRAPPER_DIR / "chemeleon_embed.py",
        extra_args=["--checkpoint", str(CHEMELEON_CHECKPOINT)],
        extra_meta={"repo": "JacksonBurns/chemeleon", "embedding_dim": 2048},
        use_free_carbene=use_free_carbene,
    )


def _run_monroe(use_free_carbene: bool) -> None:
    """Generate Monroe (blazejba/monroe) embeddings with a seeded conformer.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    suffix = "carbene" if use_free_carbene else "precat"
    _generate(
        model_name=f"monroe_{suffix}",
        label=f"Monroe-{suffix}",
        venv_python=_venv_python(MONROE_REPO_DIR),
        wrapper=WRAPPER_DIR / "monroe_embed.py",
        extra_args=[
            "--checkpoint-dir", str(MONROE_REPO_DIR / "checkpoint"),
            "--seed", str(MONROE_SEED),
            "--n-workers", "1",
        ],
        extra_meta={"repo": "blazejba/monroe", "embedding_dim": 720, "conformer_seed": MONROE_SEED},
        use_free_carbene=use_free_carbene,
    )


def _run_smited(use_free_carbene: bool) -> None:
    """Generate SMI-TED-Light (ibm-research/materials.smi-ted) embeddings.

    Args:
        use_free_carbene: Free-carbene SMILES if True, else raw salt SMILES.
    """
    suffix = "carbene" if use_free_carbene else "precat"
    snapshot = SMITED_REPO_DIR / "hf_snapshot"
    _generate(
        model_name=f"smited_{suffix}",
        label=f"SMI-TED-Light-{suffix}",
        venv_python=_venv_python(SMITED_REPO_DIR),
        wrapper=WRAPPER_DIR / "smited_embed.py",
        extra_args=[
            "--code-dir", str(snapshot / "smi-ted" / "inference" / "smi_ted_light"),
            "--checkpoint", str(snapshot / "smi-ted-Light_40.pt"),
        ],
        extra_meta={"repo": "ibm-research/materials.smi-ted", "embedding_dim": 768},
        use_free_carbene=use_free_carbene,
    )


_DISPATCH: dict[str, "object"] = {
    "minimol_precat": lambda: _run_minimol(use_free_carbene=False),
    "minimol_carbene": lambda: _run_minimol(use_free_carbene=True),
    "scage_precat": lambda: _run_scage(use_free_carbene=False),
    "scage_carbene": lambda: _run_scage(use_free_carbene=True),
    "graphormer_precat": lambda: _run_graphormer(use_free_carbene=False),
    "graphormer_carbene": lambda: _run_graphormer(use_free_carbene=True),
    "infomax3d_precat": lambda: _run_infomax3d(use_free_carbene=False),
    "infomax3d_carbene": lambda: _run_infomax3d(use_free_carbene=True),
    "frad_precat": lambda: _run_frad(use_free_carbene=False),
    "frad_carbene": lambda: _run_frad(use_free_carbene=True),
    "transformerm_precat": lambda: _run_transformerm(use_free_carbene=False),
    "transformerm_carbene": lambda: _run_transformerm(use_free_carbene=True),
    "grover_carbene": lambda: _run_grover("carbene"),
    "grover_ionstripped": lambda: _run_grover("ionstripped"),
    "chemeleon_precat": lambda: _run_chemeleon(use_free_carbene=False),
    "chemeleon_carbene": lambda: _run_chemeleon(use_free_carbene=True),
    "monroe_precat": lambda: _run_monroe(use_free_carbene=False),
    "monroe_carbene": lambda: _run_monroe(use_free_carbene=True),
    "smited_precat": lambda: _run_smited(use_free_carbene=False),
    "smited_carbene": lambda: _run_smited(use_free_carbene=True),
}


def run_model(model: str) -> None:
    """Compute and write one GNN model's embeddings via its Desktop virtualenv.

    Args:
        model: A member of ``VALID_MODELS``.

    Raises:
        ValueError: If the model is not known.
    """
    if model not in _DISPATCH:
        raise ValueError(f"Unknown GNN model {model!r}; expected one of {VALID_MODELS}")
    _DISPATCH[model]()
