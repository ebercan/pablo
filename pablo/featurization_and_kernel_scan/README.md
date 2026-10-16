# Featurization & Kernel Scan — Operator Manual

End-to-end pipeline for the NHC featurization × kernel-structure × prior
comparison study. This document is the "how to run it" manual: commands,
HPC setup, queue control, and troubleshooting.

> **Conceptual reference.** The featurizations are described in
> [featurizations.html](featurizations.html). The grid, the CV studies, the
> scoring, the elimination rule and the runtime are specified in
> [WORK-PACKAGE.md](WORK-PACKAGE.md). This manual does not restate that material.

---

## 1. Scope and design invariants

- Every featurization ships its raw, full-width matrix as `descriptors.parquet`.
  A PCA reduction is a stored featurization of its own, `<name>_pca/` (frozen
  Kneedle-knee fit over the 161-catalyst library, X-only,
  `features/dim_reduction.py`); nothing is reduced inside a CV worker.
- The fixed feature scalings below (continuous design bounds, solvent/base
  library-statistics standardization) are **assembly-time** encodings of the
  stored feature tables. On top of them every GP applies a botorch `Normalize`
  input transform fit on the **training split of each fold** (all columns except
  the integer form-ID), so the values the kernel actually sees are min–max
  re-scaled per fold. "Fit once / not per fold" therefore describes the stored
  table, not the modeling space.
- No-base experiments sit at a **"none" base location**: the base block is
  zero-filled with `base_active == 0`, no base block carries an integer index column
  so index 0 is "none". `KS1` is unmasked (none is a real location, and inside a
  pure product a zero mask would annihilate the whole covariance row);
  `KS2`-`KS4` additionally zero-mask their base terms via `base_active`.
- All electronic descriptors are **xTB-level**, computed locally on this
  laptop. DFT is out of scope.
- The catalyst axis is `cat_feat`: its levels are featurization directory names,
  so `grover` and `grover_pca` are two levels. The grid lists its 50 levels
  (`config.CAT_LEVELS`); the source featurization and the form one-hot flag are
  derived from the name. The four `interpretable*` sets have no `_pca` level.
- The grid is `config.ELIMINATION_GRID` (§7.2): every catalyst level at one
  baseline — `matern52`, **KS1**, solvent `OHE`, base `OHE`, `constant` mean,
  `raw`, prior scheme **P1**, `st_y` — plus four one-axis sweeps over kernel
  structure, prior scheme, outcome transform and mean function. **600 outer cells
  × 1 inner cell.**
- Priors are not a grid axis beyond `PRIOR_SCHEMES`. The LogNormal lengthscale
  prior is Hvarfner et al. (2024) throughout; P1 resolves its `log√D` over the
  whole design matrix and P2/P3 per block. Its σ and shift, the lengthscale
  floor, the outputscale Gamma and the mean-module σ are single global constants
  in `config.py`; the replicate-derived noise prior is one constant pair per
  target and outcome transform (`config.NOISE_EVIDENCE`).

---

## 2. Folder map

- `paths.py` — every filesystem anchor. **Import paths from here**; do not derive
  a data path from a module's own `__file__`.
- `core/` — the modeling core.
  * `config.py` — central config; the InverseGamma noise-prior β rates are
    computed live at import from replicate variance in
    `data/libraries/experiments.parquet`, one per target and outcome transform,
    pooled on the scale the GP is trained on.
  * `priors.py` — LogNormal / Gamma / Dirichlet prior constructors.
  * `kernels.py` — the four unit-lengthscale base kernels (`UnitRBFKernel`,
    `UnitMatern52Kernel`, `UnitTanimotoKernel`, `UnitMinMaxKernel`), the
    `PartialARDKernel` lengthscale layer, `MaskedKernel`,
    `DirichletAdditiveKernel`, the kernel builder (`KernelStructure.build`;
    `KS1`–`KS4` live, `Vanilla` off-grid), and `build_partition` (one shared
    lengthscale per raw discrete block, full ARD on the continuous block).
    Every base lengthscale carries a `GreaterThan(0.05)` floor
    (`LENGTHSCALE_FLOOR`), recorded per run in `run_meta.lengthscale_floor`.
  * `models.py` — mean modules (`BlockLinearMean`, `TypeLinearMean`),
    `InverseGammaPrior`, the `LCMMultiTaskGP` wrapper, and the
    `AcyloinCondensationModel` factory. **14 model classes** spanning four
    families × {yield-only, (C,S), (Y,S), (Y,C)}:
    - `st_y` / `mapsaas_y` — single model on yield (`SingleTaskGP` /
      `AdditiveMapSaasSingleTaskGP`).
    - `indep_{cs,ys,yc}` / `mapsaas_{cs,ys,yc}` — two independent single GPs.
    - `icm_{cs,ys,yc}` — full-rank (rank-2) ICM `KroneckerMultiTaskGP` (one data
      kernel ⊗ one `IndexKernel`).
    - `lcm_{cs,ys,yc}` — **true** LCM (`gpytorch.kernels.LCMKernel`, Q=2 latent
      groups, rank-2 each): covariance `Σ_q K_q ⊗ B_q`, two independent data
      kernels.

    The ICM/LCM task covariance is an unconstrained gpytorch `IndexKernel` with a
    symmetric (loc-0) prior on `covar_factor` and the default LKJ stripped, so
    the cross-task correlation is sign-free and unbiased — positive (e.g.
    yield–conversion) or negative (e.g. conversion–selectivity). `mapsaas_*` uses
    a fixed additive Matérn-5/2 SAAS architecture and ignores the kernel
    structure, mean module, and InverseGamma noise prior.
- `features/` — all feature generation behind **one registry and one CLI**
  (§5). `registry.py` is the only enumeration of featurizers; each entry declares
  its `runtime` (`local` / `xtb` / `gpu` / `venv`), its outputs, and its
  prerequisites.
  * `common.py` — shared library loading, matrix writing, L2 normalisation.
  * `fingerprints.py` — MAP4, ECFP4, Mordred, Mol2vec, SOAP. Local; SOAP reads
    the xTB geometries, so `build xtb` must have run first.
  * `components.py` — solvent (`KT`/`Hansen`/`Gutmann`/`OHE`), base
    (`thermo`/`steric`/`OHE`), MAP2 over each library, and the continuous block.
  * `foundation.py` — MolFormer-XL ×2, ChemBERTa-2, CLAMP, UniMol, GROVER,
    dispatched to Mechthild GPU nodes.
  * `gnn.py` — SCAGE, MiniMol, Graphormer, 3D-Infomax, Frad, Transformer-M, each
    in `_precat`/`_carbene` form, run against per-model Desktop venvs (Python
    3.12, except Transformer-M which needs 3.10 for fairseq). See §3.2.
  * `interpretable/` — `xtb_pipeline.py` (geometry, xTB runs, caches) →
    `descriptors.py` (Sterimol / %Vbur / C2 electronics) → `assemble.py`, which
    writes `interpretable_steric/` (10 dims), `interpretable_light/` (14),
    `interpretable/` (23), and `interpretable_dG/` (23, with `delta_e_h_plus`
    **replaced** by `xtb_deltaG_deprot`). Each assembler writes its raw column
    set straight to `descriptors.parquet`. Local for the geometric pieces;
    Mechthild CPU array for the carbene-localised electronics and ΔG_deprot. See
    §5b.6 / §5b.6b and [featurizations.html](featurizations.html) Part III.
  * `venv_wrappers/` — the thin inference scripts those Desktop venvs execute.
- `cross_validation/` — CV split generation (`cv_splits.py`), single-fold
  fit/eval (`fit_eval.py`), the grid enumerator (`generate_inputs.py`, one outer
  cell per configuration), the file-based queue scheduler and its runtime model
  (`scheduler.py`), the Mechthild SLURM submitter (`submit_mechthild.py`), the
  per-level timing probe (`timing_probe.py`) and the result concatenator
  (`concat_results.py`).
- `pablo/utils/metrics.py` — NLPD, MSLL, CRPS, CRPSS, MAE, RMSE, Q², calibration
  KS. Every metric is computed on **both yield and selectivity** (raw fraction
  scale; selectivity for `(Y,C)` is derived as `S = Y/C`, sample-based, with C
  floored and S clipped to [0,1]); yield-only classes leave the selectivity set
  empty. MSLL uses the same constant `N(ȳ_train, σ²_train)` baseline as CRPSS.
  `pearson_spearman` additionally gives, for the multi-output classes, the Pearson
  and (average-rank) Spearman correlation between the raw-scale posterior means of
  the two **modelled** targets across a fold's test points.
- `results_analysis/` — `gem_scores.py` (pooled CRPSS per study, failure flags,
  gem; §8), `gate.py` (paired fold bootstrap), and the flat heatmap +
  summary-table scripts; PNGs go directly into this folder. There is no scan-report generator: the scan report
  is written and edited by hand. `generate_fb_report.py` is separate and builds
  the fully-Bayesian comparison report only.
- `data/` — `libraries/` (the four parquets), `featurizations/`, `xtb/`,
  `weights/`, `grid/` (`inputs.json`, `cv_splits.json`, `queue/`). See §6.
- `results/` — the live run. `archive/` — frozen runs, each with a
  `MANIFEST.json`. Both gitignored.

### Tests

`pytest tests/` from the repo root. The suites that cover this package:

| file | covers |
| --- | --- |
| `test_elimination_grid.py` | grid enumeration, sweep membership, level registries, PCA reductions, cost model |
| `test_gem_scores.py` | pooled CRPSS, failure flags, gem, failed-fold rows in the aggregate |
| `test_queue_integrity.py` | `init` bookkeeping, result-identity guard |
| `test_fold_fail_criteria.py` | the fold-failure criteria of `fit_eval` |
| `test_gate.py` | paired fold bootstrap and vetoes |
| `test_featscan_behaviour.py` | featurizer registry, ECFP4 reproduction, KS1–KS4 |
| `test_featscan_pipeline.py` | feature assembly through a single-fold fit |
| `test_prior_schemes.py`, `test_lengthscale_floor.py` | P1–P3 partitions and the lengthscale floor |
| `test_outcome_transforms.py`, `test_type_quadlog_mean.py` | transforms and the `type_quadlog` mean |
| `test_replicates.py`, `test_xtb_log_parsing.py` | replicate detection, xTB parsing and repairs |
| `test_repo_structure.py` | package layout and imports |

---

## 3. Mechthild HPC environment setup

Only **foundation-model embedding** runs on Mechthild. The xTB, SOAP, and
interpretable pipelines all run locally on this laptop.

### 3.1 Core Python environment

```bash
ssh mechthild
cd $HOME/pablo

# Activate existing venv
source venv/bin/activate
# Ensure uv is available
# pip install uv          (into the venv)  OR
# pipx install uv         (system-wide, recommended)

uv pip install -r requirements_hpc.txt
```

Pinned BO stack:

- `torch==2.6.0+cu124` — highest PyTorch with cu124 wheels (`botorch>=2.2`).
- `torchvision==0.21.0+cu124` — must match torch 2.6.0 exactly.
- `gpytorch==1.15.2`
- `botorch==0.17.2`
- `transformers<4.40.0` — MolFormer's custom config imports
  `transformers.onnx`, which was removed in 4.40.0.
- `unimol_tools==0.1.5` — top of the PyPI line.

### 3.2 Foundation-model dependencies

Two models require git-cloned sources. Everything else (transformers,
sentencepiece, unimol_tools, dscribe, ase) is pip-installable from
`requirements.txt`.

#### Pre-download HuggingFace model weights (on the login node)

```bash
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
"

python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
"
```

#### CLAMP (ml-jku)

```bash
git clone https://github.com/ml-jku/clamp "$HOME/clamp"
cd "$HOME/clamp"
sed -i 's/import pkg_resources//' setup.py
# --no-deps mandatory: clamp's setup.py has an unpinned `torch` dep that
# would pull a fresh cu130 wheel and overwrite the cu124 pin.
uv pip install -e . --no-deps
cd "$HOME/pablo"
```

#### GROVER (tencent-ailab)

```bash
git clone https://github.com/tencent-ailab/grover "$HOME/grover"
uv pip install descriptastorus tensorboard
cd "$HOME/pablo"

# Use gdown — Google Drive intercepts large-file downloads and returns
# an HTML virus-scan confirmation page that wget would save silently.
gdown "1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw" \
  -O pablo/featurization_and_kernel_scan/data/grover_large.pt

# Verify (must be ~400–500 MB and torch-loadable):
ls -lh pablo/featurization_and_kernel_scan/data/grover_large.pt
python -c "
import torch
ck = torch.load('pablo/featurization_and_kernel_scan/data/grover_large.pt',
                map_location='cpu', weights_only=False)
print('OK — keys:', list(ck.keys())[:5])
"
```

#### GNN graph models — local Desktop venvs (scage / minimol / graphormer / infomax3d / frad / transformer_m)

These six graph models (see `features/gnn.py`) do **not** run on
Mechthild. Each is installed in its own **virtualenv on the Desktop** because their
dependency stacks conflict with each other and with the main env. Five are Python
3.12; **Transformer-M is Python 3.10** (its fairseq dependency has no cp312 wheel).
Per-venv torch also varies: 3D-Infomax and Transformer-M pin `torch==2.2.1` (DGL
graphbolt ABI / fairseq era respectively), the rest use `torch==2.6.0` — all CPU.
The generator runs in the main env and shells out to each Desktop venv via the
wrappers in `features/venv_wrappers/`; the subprocess + `.npy` contract is
Python-version-agnostic. Repo paths are overridable via `SCAGE_REPO_DIR` /
`MINIMOL_REPO_DIR` / `GRAPHORMER_REPO_DIR` / `INFOMAX3D_REPO_DIR` / `FRAD_REPO_DIR`
/ `TRANSFORMERM_REPO_DIR`, and checkpoints via `SCAGE_CHECKPOINT` /
`INFOMAX3D_CHECKPOINT` / `FRAD_CHECKPOINT` / `TRANSFORMERM_CHECKPOINT` (defaults
under `~/Desktop/<model>`).

```bash
# MiniMol — torch-sparse needs torch present first, and PyG ships no cp312 wheel
# for bleeding-edge torch, so pin torch 2.6 and pull the prebuilt PyG wheels:
python3.12 -m venv ~/Desktop/minimol/.venv
~/Desktop/minimol/.venv/bin/pip install -U pip
~/Desktop/minimol/.venv/bin/pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
~/Desktop/minimol/.venv/bin/pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
~/Desktop/minimol/.venv/bin/pip install minimol   # checkpoint auto-downloads on first use
# graphium emits float16 adjacency; new scipy rejects it in sparse — pin scipy:
~/Desktop/minimol/.venv/bin/pip install "scipy==1.11.4"

# SCAGE — clone; torch 2.6 (cp312); deps; rebuild the Cython ext (repo ships a 3.10
# .so); pretrained weights ship in-repo at weights/pretrain/pretrain.pth
git clone https://github.com/KazeDog/scage ~/Desktop/scage
python3.12 -m venv ~/Desktop/scage/.venv
~/Desktop/scage/.venv/bin/pip install -U pip
~/Desktop/scage/.venv/bin/pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
~/Desktop/scage/.venv/bin/pip install rdkit pyyaml numpy scipy pandas cython ogb tqdm \
  scikit-learn echo_logger ipython lmdb loguru pubchempy future
# Two repo patches needed (Cython-3 + CPU-only); already applied on this machine:
#   data_process/algos.pyx : replace astype(long, ...) -> astype(numpy.int64, ...)
#   models/layers/encoder.py : device="cuda" -> device=new_dist.device  (lines 83, 85)
cd ~/Desktop/scage/data_process && ~/Desktop/scage/.venv/bin/python setup.py build_ext --inplace
# defaults already point here; override only if you move them:
export SCAGE_CHECKPOINT=~/Desktop/scage/weights/pretrain/pretrain.pth
export SCAGE_CONFIG=~/Desktop/scage/config/config_finetune.yaml

# Graphormer — pure graph Transformer pretrained on PCQM4Mv2 HOMO-LUMO. The
# graphormer-pretrained PyPI package depends on fairseq, which DOES NOT install on
# Python 3.12 (omegaconf<2.1 metadata + broken fairseq sdist). We use the
# fairseq-free Hugging Face port instead (clefourrier/graphormer-base-pcqm4mv2).
python3.12 -m venv ~/Desktop/graphormer/.venv
~/Desktop/graphormer/.venv/bin/pip install -U pip
~/Desktop/graphormer/.venv/bin/pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
~/Desktop/graphormer/.venv/bin/pip install "transformers==4.40.2" ogb datamol cython "numpy<2"
# One patch: HF's Cython collator has the same Cython-3 `long` bug. Use a string dtype:
PYX=~/Desktop/graphormer/.venv/lib/python3.12/site-packages/transformers/models/graphormer/algos_graphormer.pyx
sed -i 's/astype(long,/astype("int64",/g' "$PYX"
# (transformers>=4.43 removed Graphormer; 4.40.2 is the last version that ships it
#  AND supports Python 3.12. The checkpoint downloads from HF on first use.)

# 3D-Infomax — clone; pretrained PNA weights ship IN-REPO (no download). The DGL
# graphbolt C++ lib (libgraphbolt_pytorch_<ver>.so) is only built up to torch 2.2.1,
# so this venv is pinned to torch 2.2.1 (still cp312, still >=2.2) — NOT 2.6.
git clone https://github.com/HannesStark/3DInfomax ~/Desktop/3dinfomax
python3.12 -m venv ~/Desktop/3dinfomax/.venv
~/Desktop/3dinfomax/.venv/bin/pip install -U pip
I=~/Desktop/3dinfomax/.venv/bin
$I/pip install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cpu
$I/pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cpu/repo.html   # serves dgl 2.1.0
$I/pip install "torchdata==0.7.1"     # DGL graphbolt imports torchdata.datapipes (dropped in 0.10+)
$I/pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
$I/pip install torch_geometric "numpy<2" rdkit pyyaml scipy pandas tqdm \
  matplotlib tensorboard scikit-learn pot pydantic
$I/pip install "ogb==1.3.1"           # chirality atom-vocab is len 4 here, matching the checkpoint
                                      # (ogb>=1.3.2 grew it to 5 -> atom_encoder size mismatch on load)
# Two repo patches (Python-3.12 + modern-PyG):
sed -i 's/^from collections import MutableMapping/from collections.abc import MutableMapping/' \
  ~/Desktop/3dinfomax/commons/utils.py
sed -i 's#^from torch_geometric.nn.acts import swish#from torch.nn.functional import silu as swish#' \
  ~/Desktop/3dinfomax/models/spherical_message_passing.py
# defaults already point here; override only if you move them:
export INFOMAX3D_CHECKPOINT=~/Desktop/3dinfomax/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt

# Frad — TorchMD-NET SE(3) transformer, ported off the repo's py3.9.10/torch1.10
# conda pins to torch 2.6 CPU on py3.12. Pretrained PCQM4Mv2 denoising checkpoint
# is on Google Drive (the README's "For QM9" pre-trained link).
git clone --depth 1 https://github.com/fengshikun/Frad ~/Desktop/frad
python3.12 -m venv ~/Desktop/frad/.venv
~/Desktop/frad/.venv/bin/pip install -U pip
R=~/Desktop/frad/.venv/bin
$R/pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
$R/pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
$R/pip install torch_geometric pytorch_lightning torchmetrics einops ase rdkit \
  pyyaml scipy pandas tqdm "numpy<2" gdown
# Download the pretrained PCQM4Mv2 denoising checkpoint (NOT the MD17 one):
$R/gdown "1O6f6FzYogBS2Mp4XsdAAEN4arLtLH38G" -O ~/Desktop/frad/frad_pretrained.pt
# Verify (~83 MB; hyper_parameters must show dataset=PCQM4MV2_Dihedral2, denoising_only=True):
ls -lh ~/Desktop/frad/frad_pretrained.pt
$R/python -c "import torch; hp=torch.load('$HOME/Desktop/frad/frad_pretrained.pt',map_location='cpu',weights_only=False)['hyper_parameters']; print(hp['model'], hp['dataset'], hp['embedding_dimension'])"
# One repo patch (torch 2.6 flipped torch.load default to weights_only=True; the
# checkpoint carries pickled PL callbacks):
sed -i 's/ckpt = torch.load(filepath, map_location="cpu")/ckpt = torch.load(filepath, map_location="cpu", weights_only=False)/' \
  ~/Desktop/frad/torchmdnet/models/model.py
# defaults already point here; override only if you move it:
export FRAD_CHECKPOINT=~/Desktop/frad/frad_pretrained.pt

# Transformer-M — Graphormer-style 2D/3D transformer on a vendored fairseq. fairseq's
# omegaconf<2.1 pin has NO cp312 wheel, so this venv is **Python 3.10** (via uv).
# The PyPI fairseq sdist is broken (missing balanced_assignment.cpp), so install the
# repo's bundled fairseq editably instead.
git clone --depth 1 https://github.com/lsj2408/Transformer-M ~/Desktop/transformer_m
uv venv --python 3.10 ~/Desktop/transformer_m/.venv
TM=~/Desktop/transformer_m/.venv/bin/python
uv pip install --python $TM "torch==2.2.1" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $TM "numpy==1.23.5" "cython<3" rdkit "ogb==1.3.2" scipy pandas tqdm setuptools wheel
uv pip install --python $TM torch_geometric
uv pip install --python $TM torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
# Build the bundled fairseq (has the C++ source the PyPI sdist omits) + the Cython algos:
uv pip install --python $TM --no-build-isolation -e ~/Desktop/transformer_m
( cd ~/Desktop/transformer_m && $TM setup_cython.py build_ext --inplace )   # -> algos.*.so at repo root
# numpy must stay 1.23.x: fairseq uses the removed np.float alias; 1.23.5 is <2 (torch 2.2.1) and still has it.

# Download the L12 checkpoint (189 MB) from OneDrive — the "L12 / 0.0785" row of the
# repo's Checkpoints table — via browser (OneDrive SharePoint links are not curl-able):
#   https://1drv.ms/u/s!AgZyC7AzHtDBdWUZttg6N2TsOxw?e=sUOhox
# Save it to ~/Desktop/transformer_m/L12.pt (the default TRANSFORMERM_CHECKPOINT path).
export TRANSFORMERM_CHECKPOINT=~/Desktop/transformer_m/L12.pt
```

SCAGE ships no frozen-inference API, so `scage_embed.py` reconstructs the path
from `finetune_mpp.py`: per-molecule `mol_to_data_pkl` (2D graph + 3D conformer),
`collator_finetune_pkl`, then the **pretrained encoder's CLS token** `atom[:,0,:]`
(512-d) — *not* the randomly-initialised `head_Graph`. `GlobalVar.dist_bar` is set
to the pretrain default `[20, 50]` (the checkpoint stores only weights; the
attention `scale_linear` is sized `len(dist_bar)*512`, so the length must match).
On load, the encoder params bind fully (0 missing); only the unused finetune/
pretrain heads are skipped. Runs on CPU (this box has no GPU).

3D-Infomax ships `inference.py`, so `infomax3d_embed.py` reconstructs its
fingerprint path without importing that module (which drags in seaborn/icecream/
every trainer): `InferenceDataset` (RDKit `AddHs` + OGB 2D atom/bond features →
DGL graph) → `graph_only_collate` → `models.PNA` built from the checkpoint dir's
`train_arguments.yaml` (`target_dim=256`) with `checkpoint['model_state_dict']`,
and the **PNA readout fingerprint** (256-d) as the embedding. Only the **2D
graph** is used (the 3D net is a pretraining-time teacher). Unlike the repo's
`inference.py`, the wrapper runs under `model.eval()` + `torch.no_grad()` so the
batch-norm embedding is deterministic and batch-independent (the frozen-featurizer
contract). Runs on CPU.

Frad's `frad_embed.py` reconstructs the frozen path from `torchmdnet/models/model.py`:
`load_model(checkpoint)` (kwargs `md17/seperate_noise/mask_atom/bond_length_scale`
injected — the pretrained checkpoint predates them; defaults from `scripts/train.py`)
builds the equivariant transformer and loads `state_dict`; the embedding is the
**representation model's per-atom scalar features** `x` (256-d), mean-pooled per
molecule (the scalar regression head is unused). Frad is 3D, so the wrapper
generates one RDKit ETKDGv3+MMFF conformer per molecule (with `AddHs`). Conformer
robustness, in order: ETKDGv3 → random-coords + `ignoreSmoothingFailures` → 2D
layout + Gaussian jitter; then a **minimum-interatomic-separation guard** (0.5 Å)
jitters apart any near-coincident atoms. This last guard is essential for the
`_precat` (salt, multi-fragment) form: coincident atoms give the SE(3) model an
undefined unit direction (`0/0`) → non-finite features → the **whitened**
equivariant layer-norm (the checkpoint was pretrained with `--layernorm-on-vec
whitened`) fails its SVD. Runs on CPU.

Transformer-M's `transformerm_embed.py` reconstructs the frozen path from
`evaluate.py` + `models/transformer_m.py`: `fairseq.utils.import_user_module`
registers the model, then `TransformerMModel.build_model(args, task=None)` builds it
(`build_model` ignores `task`, so the data-loading `GraphPredictionTask` is
bypassed). The L12 checkpoint stores **only weights** (no `args`/`cfg`), so the
wrapper reconstructs the exact training args — `GraphPredictionConfig` task defaults
(`num_atoms=4608`, `num_spatial=512`, …) + the `transformer_m_base` arch (768-d / 12
layers / 32 heads) with `add_3d=True` — and applies the registered arch function;
`load_state_dict` then binds with **0 missing / 0 unexpected** keys, confirming the
reconstruction. Per molecule a PyG `Data` (OGB features + ETKDG+MMFF 3D conformer,
same robustness ladder + separation guard as Frad) → `preprocess_item` (Cython
`algos`) → `collator_3d`; the embedding is the final-layer hidden state at the
**graph/virtual token** (sequence position 0), 768-d (the 1-d regression head is
unused). Runs on CPU.

Each model runs in **two forms**, mirroring `molformer_xl_precat` /
`molformer_xl_carbene`: a `_precat` variant on the raw pre-catalyst (salt) SMILES,
and a `_carbene` variant on the free-carbene SMILES with the 4-d pre-catalyst-form
one-hot appended. That is **twelve** featurizers:
`{minimol,scage,graphormer,infomax3d,frad,transformerm}_{precat,carbene}`.

Generate (run locally where the Desktop venvs live):

```bash
python -m pablo.featurization_and_kernel_scan.features build minimol_precat
python -m pablo.featurization_and_kernel_scan.features build minimol_carbene
python -m pablo.featurization_and_kernel_scan.features build --family gnn
```

Each writes `data/featurizations/<model>/{raw_model_output,descriptors}.parquet`
and `metadata.json`, exactly like every other featurizer; the CV
grid then picks them up automatically (`_carbene` registered `True`, `_precat`
`False` in `config.py`).

### 3.3 Validate CUDA on a Mechthild H100 node

```bash
sinteractive -p gpu --gpus=H100:1 --time=00:10:00
module load cuda/12.6.2
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

---

## 4. Canonical data tables

The `data/` directory ships with these committed parquets, used as inputs
across the pipeline:

- `catalysts.parquet`
- `bases.parquet`
- `solvents.parquet`
- `experiments.parquet`

The InverseGamma noise-prior β rates are recomputed live from
`experiments.parquet` on every import of `config.py` (see §2).

---

## 5. Feature generation

### 5.1 Local fingerprint/descriptor featurizers

```bash
python -m pablo.featurization_and_kernel_scan.features build --family fingerprint --family component
```

Default Mol2vec model path: `data/weights/mol2vec_weights.pkl`. Override with
`--mol2vec-model /path/to/weights.pkl`. Individual flags: `--map4`, `--ecfp4`,
`--mordred`, `--mol2vec`, `--solvent-base`, `--solvent-map`, `--base-map`.

`--solvent-map` and `--base-map` produce `solvent_map/` and `base_map/`
featurizations: raw `MAPFingerprint(radius=1)` (= MAP2) on the 14
solvents and 8 bases respectively. This matches the encoding used by
`pablo/main.py`, so the scan compares directly against the production
featurization.

### 5.3 Interpretable xTB

All xTB / Sterimol descriptors live in **a single module**:
[features/interpretable/](features/interpretable/).
It owns the GFN2 / ALPB(toluene) optimisations on every free-carbene
SMILES. See §5b.6 for the Mechthild submission protocol and
[featurizations.html](featurizations.html) Part III for descriptor
definitions and column lists. SOAP lives in
[features/fingerprints.py](features/fingerprints.py) and
consumes the xTB-optimised geometries produced here.

```bash
# Shared prerequisite: xTB-optimised geometries.
python -m pablo.featurization_and_kernel_scan.features build structural
python -m pablo.featurization_and_kernel_scan.features build xtb

# Stage 1 (local): intermediate caches + steric + light assembly.
python -m pablo.featurization_and_kernel_scan.features build tally_topology
python -m pablo.featurization_and_kernel_scan.features build n_c_aggregates
python -m pablo.featurization_and_kernel_scan.features build lp_sterimol
python -m pablo.featurization_and_kernel_scan.features build vbur_quadrants
python -m pablo.featurization_and_kernel_scan.features build interpretable_steric
python -m pablo.featurization_and_kernel_scan.features build interpretable_light

# SOAP (separate module; reads xtb_inputs/C*/xtbopt.xyz).
python -m pablo.featurization_and_kernel_scan.features build soap

# Stage 2 (Mechthild CPU array): carbene-localised electronics + full assembly.
# See §5b.6 for the SLURM submission protocol.
python -m pablo.featurization_and_kernel_scan.features build bh3_reference
python -m pablo.featurization_and_kernel_scan.features build xtb_extra
python -m pablo.featurization_and_kernel_scan.features build probe_h_plus
python -m pablo.featurization_and_kernel_scan.features build probe_bh3
python -m pablo.featurization_and_kernel_scan.features build interpretable
# xTB descriptors — Stage 2b, ΔG_deprot (Mechthild, see §5b.6b)
python -m pablo.featurization_and_kernel_scan.features build deltaG_deprot
python -m pablo.featurization_and_kernel_scan.features build interpretable_dG
```

Required binaries / packages:

- `xtb` command-line binary on PATH (e.g. `conda install -c conda-forge xtb`).
- `dscribe` (PyPI) for SOAP.
- `morfeus-ml`, `dbstep` for Sterimol / %Vbur.

### 5.4 Foundation model GPU jobs (Mechthild)

```bash
python -m pablo.featurization_and_kernel_scan.features submit
```

Six SLURM jobs run in parallel on the `gpu` partition (1 × H100, 4 CPUs,
64 G, 2 h, `cuda/12.6.2`):

| Model | Variant | Source |
|-------|---------|--------|
| `molformer_xl_precat` | MolFormer-XL on pre-cat SMILES | `ibm/MoLFormer-XL-both-10pct` |
| `molformer_xl_carbene` | MolFormer-XL on free-carbene SMILES + form id | same |
| `chembert2` | ChemBERTa-2 mean-pooled | `DeepChem/ChemBERTa-77M-MTR` |
| `clamp` | CLAMP `compound_encoder` | `ml-jku/clamp` |
| `unimol` | UniMol2-1.1B on free-carbene 3D + form id | `unimol_tools` |
| `grover` | GROVERlarge fingerprints | `tencent-ailab/grover` |

Logs: `data/foundation_model_logs/<model>_<jobid>.{out,err}`.

Single-model submit / monitor / read-log:

```bash
python -m pablo.featurization_and_kernel_scan.features submit molformer_xl_precat
squeue -u $USER --format="%.18i %.30j %.8T %.10M %.9l %.6D %R"
cat data/foundation_model_logs/<model>_<jobid>.err
```

---

## 5b. xTB descriptor pipeline (operational)

The three interpretable xTB featurizations — `interpretable_steric` (10
dims), `interpretable_light` (21 dims), and `interpretable` (30 dims), each
delivered raw and un-reduced with every category one-hot encoded —
are defined in full in
[featurizations.html](featurizations.html) Part III (descriptor
blocks, column lists, Sterimol/%Vbur/electronics conventions). This
section covers only how to **generate** them.

`interpretable_steric` and `interpretable_light` run entirely locally
(Stage 1). `interpretable` (full) also requires the Mechthild CPU-array
xTB jobs for the carbene-localised electronics (Stage 2, §5b.6).

### 5b.5 Shared xTB geometry pipeline

GFN2-xTB + ALPB(toluene) on every free-carbene SMILES. Shared by all three
featurizations and by SOAP. Already committed:

- `data/xtb/inputs/C<id>/xtbopt.xyz` — optimised geometry (161 catalysts).
- `data/xtb/inputs/C<id>/charges` — Mulliken atomic charges.
- `data/xtb/inputs/C<id>/info.json` — pre-cat form, SMILES, net charge,
  **`carbene_idx`** (0-based atom index of C2 in the xyz file),
  **`hetero_nbr_indices`** (ring-heteroatom neighbour indices, same frame).
  These topology fields are written by `_prepare_xtb_inputs` for all new runs.
  For datasets generated before the topology-index patch, run the one-time
  backfill described in §5b.6 before Stage 2.
- `data/xtb/outputs/C<id>.txt` — raw stdout log.
- `data/xtb/results_checkpoint.parquet` — parsed scalars.

**Embedding cascade** (`_prepare_xtb_inputs`, first success wins):

1. `_smiles_to_xyz` — ETKDGv3 on the free-carbene SMILES.
2. `_smiles_to_xyz_via_salt` — embed the salt cation, strip the C2–H.
3. `_smiles_to_xyz_pure_dg` — plain distance geometry, ETKDG torsion terms off,
   `useRandomCoords` + `ignoreSmoothingFailures`.
4. `_smiles_to_xyz_sphere_fallback` — Fibonacci-sphere placement.

Molecules with two or more macrocycles (`_is_polymacrocyclic`) skip straight to
rung 3. ETKDG cannot embed a doubly-strapped cyclophane and takes hours of retries
to say so; plain distance geometry does it in 0.2 s.

**Rung 4 is not a fallback, it is a failure.** Sphere placement ignores
connectivity entirely, and xTB will converge it to whatever knot lies nearest —
a geometry that passes every downstream QC gate while being the wrong molecule.
C136 shipped that way for one round: a three-coordinate "carbene", a 0.78 Å
H···H contact, a 0.12 eV HOMO–LUMO gap, and 18 of its 23 `interpretable` columns
wrong (`homo_weight_c2` 0.007 instead of 0.849). No catalyst currently reaches
rung 4, and `tests/test_deltaG_deprot.py` asserts both that none does and that
every stored geometry is structurally sane. If a future catalyst lands there,
treat it as unresolved rather than as data.

**If a stored geometry changes, everything derived from it must be rebuilt.**
`xtbopt.xyz` feeds five featurizations, not one:

```bash
# 1. geometry + parsed scalars
python -m pablo.featurization_and_kernel_scan.features build xtb
# 2. geometric intermediates (drop the affected catalyst's row first — both
#    generators resume from cache and will otherwise skip it)
python -m pablo.featurization_and_kernel_scan.features build lp_sterimol
python -m pablo.featurization_and_kernel_scan.features build vbur_quadrants
python -m pablo.featurization_and_kernel_scan.features build n_c_aggregates
# 3. per-catalyst electronics (Stage 2; --cat-id N for a single catalyst)
python -m pablo.featurization_and_kernel_scan.features build xtb_extra --cat-id N
python -m pablo.featurization_and_kernel_scan.features build probe_h_plus --cat-id N
python -m pablo.featurization_and_kernel_scan.features build probe_bh3 --cat-id N
# 4. re-assemble
python -m pablo.featurization_and_kernel_scan.features build interpretable_steric
python -m pablo.featurization_and_kernel_scan.features build interpretable_light
python -m pablo.featurization_and_kernel_scan.features build interpretable
python -m pablo.featurization_and_kernel_scan.features build deltaG_deprot --cat-id N
python -m pablo.featurization_and_kernel_scan.features build interpretable_dG
# 5. SOAP reads xtbopt.xyz directly and rebuilds all 161 rows
python -m pablo.featurization_and_kernel_scan.features build soap
```

`--assemble-full` needs the Stage-2 per-catalyst JSONs for **all** 161 catalysts,
and those are routinely deleted after a sweep. If they are gone and only one
catalyst changed, patch that row into `interpretable/descriptors.parquet`
directly rather than re-running Stage 2 across the library. The SMILES-derived
columns (`cat_form_id`, `ring_type_id`, `backbone_saturated`, `mw`,
`heavy_atom_count`) are geometry-independent and never need touching.

### 5b.6 Stage 2 — Mechthild CPU array (`interpretable` electronics)

Six dimensions require new xTB calls: Fukui⁺/⁻ at C2, HOMO weight on C2,
Mulliken charge at C2, ΔE(carbene+H⁺), ΔE(carbene+BH₃). Each step uses
`--cat-id N` for SLURM array dispatch.

```bash
# ── Prerequisite: topology-index backfill (login node) ──────────────────────
# Required once per dataset if info.json files lack 'carbene_idx'.
# Safe to re-run — already-patched catalysts are skipped automatically.
python -m pablo.utils.backfill_carbene_indices

# ── BH₃ reference (one-time, login node) ────────────────────────────────────
python -m pablo.featurization_and_kernel_scan.features build bh3_reference

# ── SLURM arrays (submit simultaneously) ────────────────────────────────────
mkdir -p logs
sbatch -J xtb_extra  -t 00:30:00 - <<'EOF'
#!/bin/bash
#SBATCH --array=1-161%200
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m pablo.featurization_and_kernel_scan.features build xtb_extra --cat-id "$SLURM_ARRAY_TASK_ID"
EOF

sbatch -J probe_h    -t 01:00:00 - <<'EOF'
#!/bin/bash
#SBATCH --array=1-161%200
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m pablo.featurization_and_kernel_scan.features build probe_h_plus --cat-id "$SLURM_ARRAY_TASK_ID"
EOF

sbatch -J probe_bh3  -t 01:00:00 - <<'EOF'
#!/bin/bash
#SBATCH --array=1-161%200
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m pablo.featurization_and_kernel_scan.features build probe_bh3 --cat-id "$SLURM_ARRAY_TASK_ID"
EOF

# ── Final assembly (after all arrays complete) ───────────────────────────────
python -m pablo.featurization_and_kernel_scan.features build interpretable
```

### 5b.6b Stage 2b — Mechthild CPU array (`interpretable_dG`, ΔG_deprot)

Swaps one column: `delta_e_h_plus` out, `xtb_deltaG_deprot` =
**G(carbene) − G(azolium)** in kcal/mol at GFN2/ALPB(toluene), 298.15 K, in.

The two are the same quantity at different levels of care — `delta_e_h_plus` is
an electronic protonation energy (opposite sign, no Hessian) read off a single
carbene-derived geometry, `xtb_deltaG_deprot` is a free energy over
independently embedded conformer ensembles — and they correlate at
**r = −0.96** across the library. Carrying both would put two near-collinear
columns in the same GP block and would reduce the head-to-head against
`interpretable` to "does ΔG add anything on top of ΔE", which that correlation
answers trivially. Swapping makes the two featurizations differ in exactly one
column and asks whether the conformer ensemble and the thermal correction are
worth their cost.

**Absolute scale.** ΔG lands near 131 kcal/mol (median), not the ~250 kcal/mol
of a literature gas-phase proton affinity. That gap is the GFN2 energy scale, not
an error: running this protocol on 1,3-dimethylimidazolium as a reference gives
135.5 kcal/mol. Only relative values across the library are meaningful, which is
what the descriptor is for. The ring-class ordering is chemically correct —
imidazolinylidene (sat) > imidazolylidene > benzimidazolylidene > thiazolylidene
≈ triazolylidene > oxazolylidene.

**Charge states.** Three catalysts are not on the ordinary azolium(+1) →
carbene(0) ladder: C31 (0 → −1, anionic mesoionic) and C61 / C78 (+2 → +1,
bis-azolium dications, mono-deprotonated). Their values are correct first
deprotonation free energies, but they carry the Coulomb work of a different
charge state — tens of kcal/mol, and not basicity — which is why they sit at the
extremes of the distribution. `--assemble-dG` prints them as a warning block;
`metadata.json` records them under `charge_state_nonstandard`.

Protocol, identical for every catalyst and both species — the descriptor is only
meaningful as a relative quantity, so nothing here may vary across the library:

| step | setting |
|---|---|
| species | `to_azolium` / `to_mono_carbene` (differ by exactly one C2 proton and one charge unit) |
| charges | `net_charge` and `net_charge + 1`; never literal 0/+1 |
| conformers | ETKDGv3, 20 requested, seed 42, macrocycle + small-ring torsions, 0.5 Å RMS prune, MMFF94 (UFF fallback), 10 kcal/mol window |
| xTB | `--gfn2 --alpb toluene --ohess --iterations 500` on **every** conformer |
| geometry gate | the stored seed geometry must be structurally sane (no contact below 0.90 Å, carbene centre two-coordinate and still bonded to its ring heteroatoms) and a constructed C2–H must clear every other atom by ≥ 1.50 Å; a rejected seed is a recorded failure, never a computed number |
| QC | optimization converged; no mode below −50 cm⁻¹; C2 still carries the expected H count in the optimized geometry |
| ensemble | duplicate minima collapsed (0.1 kcal/mol **and** 0.1 Å heavy-atom RMSD), then Boltzmann-weighted at 298.15 K |
| H⁺ term | dropped — a solvent-referenced constant that cancels library-wide |

The free proton is never computed. Failures are recorded in the per-catalyst JSON
and never imputed; `--assemble-dG` refuses to write a table with a missing value.

`xtb` is **not** on the default PATH on Mechthild — it is an environment module.
Without the `module load` line every array task dies instantly on a missing
binary. Also note `module load` must not be piped, or the PATH change is lost to
a subshell.

Write `dg_deprot.sbatch` in the repo root:

```bash
#!/bin/bash
#SBATCH -J dG_deprot
#SBATCH --array=1-161%200
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --partition=medium
#SBATCH -t 08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

module load xtb/6.6.0
source .venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_STACKSIZE=4G

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} xtb=$(which xtb)"

python -m pablo.featurization_and_kernel_scan.features build \
    deltaG_deprot --cat-id "${SLURM_ARRAY_TASK_ID}"
```

Then, from the repo root:

```bash
mkdir -p logs

# Smoke-test one small catalyst first (C149: 46 atoms, ~9 conformers)
sbatch --array=149 --partition=short -t 01:30:00 -J dG_smoke dg_deprot.sbatch

# Full sweep once the smoke task writes a ΔG value
sbatch dg_deprot.sbatch

# ── Assembly (after the array completes) ────────────────────────────────────
python -m pablo.featurization_and_kernel_scan.features build interpretable_dG
```

**Sizing.** Measured on one 8-thread box: `--ohess` takes ~3 s for a 46-atom
catalyst starting from an optimized geometry, **~58 s** from a raw MMFF conformer
(the production case), and 268 s for the 141-atom outlier from an optimized start.
At ~10 surviving conformers × 2 species the median task is ~20 min and the largest
catalysts run several hours, hence the 8 h wall limit. Results are checkpointed per
catalyst, so a requeued task resumes instead of restarting.

**Re-running.** `--deltaG-deprot` skips catalysts that already have a cached JSON.
To force a recompute, trash the cache first:

```bash
gio trash pablo/featurization_and_kernel_scan/data/featurizations/_intermediate/deltaG_deprot/
```

### 5b.7 Known gotchas

1. **Probe-adduct charge.** Use `info.json["net_charge"] + 1` for H⁺
   adducts, not literal +1. Salt precatalysts already carry charge +1.
2. **xTB working-dir contention.** Every xTB invocation overwrites
   `charges`, `molden.input`, `xtbopt.xyz` in its cwd. Run each call in a
   fresh tempdir.
3. **dbstep on a lone H.** dbstep's distance-based bond perception reaches
   across the ring for single-H substituents. `interpretable_steric` uses
   canonical Verloop H values `(L=2.06, B1=B5=1.00)` instead.
4. **He dummy distance.** ≥ 1.5 Å along LP for Sterimol; 2.0 Å for %Vbur
   (SambVca convention).
5. **xTB molden format.** xTB writes Cartesian d/f (6d/10f) under `--molden`
   (no `[5d]`/`[7f]` markers, no `[Core]` section). HOMO weight is parsed
   directly from the molden text — no `iodata` dependency.
6. **HOMO occupation cutoff.** The molden walker selects the HOMO as the
   highest-energy MO with `occ ≥ 1.0` (majority-occupied), not `occ > 0`.
7. **Topology-index contract.** Stage 2 reads `carbene_idx` and
   `hetero_nbr_indices` directly from `info.json`; it does not re-derive them
   from SMILES. These indices must match the atom ordering in `xtbopt.xyz`.
   If Stage 2 asserts on a missing field, run the backfill step above.
8. **xTB `--opt` logs carry two property blocks.** xTB prints HOMO/LUMO/gap
   for the *input* geometry before optimising, then again after convergence.
   A first-match regex over the whole log returns the force-field-geometry
   values — this shipped in `xtb_homo_eV`/`xtb_lumo_eV`/`xtb_gap_eV` for all
   161 catalysts until 18.09.2026. `_parse_xtb_output` now reads only the text
   after the final `GEOMETRY OPTIMIZATION CONVERGED`. Dipole and polarizability
   are printed only post-convergence and were never affected. Do **not** refresh
   `results_checkpoint.parquet` with `build xtb`: `run_all_xtb` calls
   `_prepare_xtb_inputs()` unconditionally and re-embeds every `mol.xyz` /
   `info.json`. Use `features/interpretable/repair_xtb_scalars.py` (dry run by
   default, `--write` to apply); `tests/test_xtb_log_parsing.py` guards it.

To re-run only `xtb_extra` (e.g. to pick up a HOMO@C2 parser fix), wipe
the stale JSONs first so the resume-skip logic re-runs every catalyst:

```bash
gio trash pablo/featurization_and_kernel_scan/data/featurizations/_intermediate/xtb_extra/
sbatch -J xtb_extra -t 00:30:00 - <<'EOF'
#!/bin/bash
#SBATCH --array=1-161%200
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=medium
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m pablo.featurization_and_kernel_scan.features build xtb_extra --cat-id "$SLURM_ARRAY_TASK_ID"
EOF
python -m pablo.featurization_and_kernel_scan.features build interpretable
```

---

## 6. Feature output contract

Every catalyst featurizer under `data/featurizations/<name>/` contains
exactly 161 rows and writes:

- `raw_model_output.parquet` — direct model / fingerprint output, untouched.
- `descriptors.parquet` — the delivered feature matrix, consumed downstream.
- `metadata.json`

Solvent (14 rows, `solvent_<block>/`) and base (8 rows, `base_<block>/`)
featurizers write `descriptors.parquet` + `metadata.json`. Continuous bounds
live in `continuous/metadata.json`.

The `interpretable*/` folders write `descriptors.parquet` (concatenated
structural / xTB / Sterimol columns, `cat_form_id`/`ring_type_id` included) plus
`metadata.json`. The CV assembler consumes `descriptors.parquet` everywhere.

Index convention:

- Catalyst featurizers: `catalyst_id`.
- Solvents: `solvent_id`.
- Bases: `base_id`.

### 6.1 Featurizer input → output reference

What each featurizer consumes, what it emits (**raw** output + dimension), and
the width of the delivered `descriptors.parquet`. Numbers are read from each
featurization's `metadata.json`. All learned embeddings are L2-normalized; the
normalized matrix is what is delivered. Descriptions and citations: [featurizations.html](featurizations.html).

**Catalyst featurizers** (161 rows, `catalyst_id`). The 25 grid featurizations are
the rows below minus the two **⊘ off-grid** entries (`map4`, `ecfp4`), which remain
generatable via `features build` but were dropped from the CV grid
(see §7.2):

| Featurizer | Input → representation | 3D conformer source | Raw output (dim) | Delivered dim | Ref |
|---|---|---|---|---|---|
| `map4` ⊘ | SMILES → MinHash atom-pair fp (r=2) | — | hashed fp (2048) | 2048 | Capecchi 2020 |
| `ecfp4` ⊘ | SMILES → Morgan count fp (r=2) | — | count fp (2048) | 2048 | Rogers 2010 |
| `mordred` | SMILES → 2D descriptors (5-step preprocess) | — | descriptors (427) | 427 | Moriwaki 2018 |
| `mol2vec` | SMILES → summed Word2Vec substructures | — | embedding (300) | 300 | Jaeger 2018 |
| `soap` | 3D geometry → SOAP power spectrum (avg) | xTB GFN2/ALPB(toluene) free-carbene | power spectrum (7425) | 7425 | Bartók 2013 |
| `chembert2` | SMILES → ChemBERTa-2 mean-pooled | — | embedding (384) | 384 | Ahmad 2022 |
| `clamp` | SMILES → CLAMP compound encoder | — | embedding (768) | 768 | Seidl 2023 |
| `molformer_xl_precat` | salt SMILES → MolFormer-XL mean-pooled | — | embedding (768) | 768 | Ross 2022 |
| `molformer_xl_carbene` † | free-carbene SMILES → MolFormer-XL mean-pooled | — | embedding (768) | 768 | Ross 2022 |
| `unimol` † | free-carbene SMILES → UniMol2-1.1B CLS | UniMol-internal ETKDG+MMFF (low-E) | CLS embedding (1536) | 1536 | Zhou 2023 |
| `grover` | SMILES → 2D graph → GROVERlarge fp | — | fingerprint (5000) | 5000 | Rong 2020 |
| `minimol_precat` | salt SMILES → 2D graph → MiniMol fp | — | fingerprint (512) | 512 | Kläser 2024 |
| `minimol_carbene` † | free-carbene SMILES → 2D graph → MiniMol fp | — | fingerprint (512) | 512 | Kläser 2024 |
| `scage_precat` | salt SMILES → 2D graph + 3D conf → encoder CLS | RDKit ETKDG ×10 + MMFF, min-E | CLS token (512) | 512 | SCAGE 2025 |
| `scage_carbene` † | free-carbene → 2D graph + 3D conf → encoder CLS | RDKit ETKDG ×10 + MMFF, min-E | CLS token (512) | 512 | SCAGE 2025 |
| `graphormer_precat` | salt SMILES → OGB 2D graph → virtual token | — | graph token (768) | 768 | Ying 2021 |
| `graphormer_carbene` † | free-carbene → OGB 2D graph → virtual token | — | graph token (768) | 768 | Ying 2021 |
| `infomax3d_precat` | salt SMILES → 2D graph → PNA readout | — (3D distilled at pretraining) | readout fp (256) | 256 | Stärk 2022 |
| `infomax3d_carbene` † | free-carbene → 2D graph → PNA readout | — (3D distilled at pretraining) | readout fp (256) | 256 | Stärk 2022 |
| `frad_precat` | salt SMILES → 3D conf → per-atom scalar, mean-pooled | RDKit ETKDGv3+MMFF (+ fallback/guard) | pooled scalars (256) | 256 | Feng 2023 |
| `frad_carbene` † | free-carbene → 3D conf → per-atom scalar, mean-pooled | RDKit ETKDGv3+MMFF (+ fallback/guard) | pooled scalars (256) | 256 | Feng 2023 |
| `transformerm_precat` | salt SMILES → OGB 2D graph + 3D conf → graph token | RDKit ETKDGv3+MMFF (+ fallback/guard) | graph token (768) | 768 | Luo 2023 |
| `transformerm_carbene` † | free-carbene → OGB 2D graph + 3D conf → graph token | RDKit ETKDGv3+MMFF (+ fallback/guard) | graph token (768) | 768 | Luo 2023 |
| `interpretable_steric` | 3D geometry → Sterimol + %Vbur aggregates | xTB GFN2/ALPB(toluene) free-carbene | descriptors (10) | 10 | Verloop; Bannwarth 2019 |
| `interpretable_light` | + RDKit structural + molecular xTB scalars | same | descriptors (14) | 14 | Verloop; Bannwarth 2019 |
| `interpretable` | + quadrant %Vbur + carbene electronics (Fukui, HOMO@C2, ΔE) | same | descriptors (23) | 23 | Falivene 2019; Bannwarth 2019 |
| `interpretable_dG` | `delta_e_h_plus` → xTB C2-deprotonation free energy ΔG(carbene−azolium) | ETKDGv3 ensembles per species, GFN2/ALPB `--ohess` | descriptors (23) | 23 | Bannwarth 2019; Grimme 2017 |

**Solvent (14 rows) and base (8 rows) featurizers.** Both components are split
into competing descriptor blocks rather than one pooled table, so the grid
reports which account of the component the GP actually needs. Live grid:
solvent = `KT`, `Hansen`, `Gutmann`, `OHE`; base = `thermo`, `steric`, `OHE`.

| Featurizer | Input → representation | Raw output (dim) | Delivered dim | Ref |
|---|---|---|---|---|
| `solvent_KT` | identity → Kamlet–Taft α / β / π\* | descriptors (3) | 3 | Marcus 1993 |
| `solvent_Hansen` | identity → Hansen dD / dP / dH + logP | descriptors (4) | 4 | Hansen 2007; Hansch–Leo |
| `solvent_Gutmann` | identity → Gutmann DN + ε | descriptors (2) | 2 | Gutmann 1978 |
| `solvent_OHE` | identity → one-hot over the 14 solvents | descriptors (14) | 14 | — |
| `base_thermo` | identity → proton affinity + organic/inorganic one-hot | descriptors (3) | 3 | — |
| `base_steric` | identity → Sterimol L / B1 / B5 at the basic atom | descriptors (3) | 3 | Verloop |
| `base_OHE` | identity → one-hot over the 8 bases | descriptors (8) | 8 | — |

The two `OHE` blocks are the no-descriptor controls: they assert only that the
members are distinct, so any descriptor block that fails to beat its `OHE`
sibling is not earning its chemistry.

Off-grid (generatable via `features build`, not enumerated):
`solvent_map` / `base_map` (raw MAP2 fingerprint, 2048 dims each; reproduce the
`pablo/main.py` production encoding).

**† carbene form one-hot, appended at assembly only.** A `_carbene` featurizer
embeds the free-carbene SMILES; the delivered matrix is the L2-normalised raw
output alone. The CV cell-assembler then appends the 4-column form one-hot
`form_salt / form_free_nhc / form_co2_adduct / form_bicarbonate`
(`append_form_onehot_to_cat=True` → effective catalyst block = `Delivered dim`
+ 4), measured by the catalyst block's own base kernel like every other column.
Uniform across all `_carbene` variants, so cross-cell comparisons are
unaffected.
**‡** Solvent and base continuous columns are standardized on full-library
statistics (14 solvents / 8 bases; baked into the stored tables). Standardization
is per column, so cutting the pooled table into blocks leaves every value
unchanged — a block's numbers do not depend on which other blocks exist.
This standardization is fit-once; the GP's per-fold
`Normalize` input transform still re-scales these columns inside each fold (see §1).

---

## 7. CV and kernel scan execution

### 7.1 CV split definitions

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.cv_splits
```

Writes `data/grid/cv_splits.json` with:

1. **Primary** — **10× repeated** stratified catalyst-grouped **6-fold CV
   (60 folds)**.
2. **Leave-one-pre-cat-type-out** — 2 folds (free_nhc / bicarbonate). "salt"
   and "co2_adduct" are excluded due to >10 test points.
3. **Leave-one-scaffold-out** — **3 folds** (imidazolylidene,
   triazolinylidene, thiazolylidene; "other" has no tested catalysts,
   and "imidazolinylidene" is excluded due to >10 test points).
4. **Leave-one-solvent-out** — 7 folds. "THF" is excluded due to >10 test points.
5. **Leave-one-replicate-out** — one fold per replicate group (currently 5).
6. **Leave-one-out** — 60 folds. Non-replicate experiments get standard
   single-point holdout (55 folds). The 14 replicate experiments (four
   triplets + one doublet) are collapsed into 5 leave-group-out folds so
   that no fold ever places one replicate in the test set while another
   stays in training — this would let the model "see" the test input and
   corrupt the LOO metric.

Both replicate-dependent counts follow the groups `core/replicates.py` detects
in `experiments.parquet`, so they move when replicate experiments are added.

Total at 69 experiments: **137 folds per inner cell**.

### 7.2 Cell enumeration

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --dry-run
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --cat-set fingerprint  # one catalyst family
```

The grid is `config.ELIMINATION_GRID`: the 50 catalyst levels, a baseline level
for every other axis, and the four sweeps (`KS1`–`KS4`, `P1`–`P3`, the four outcome
transforms, the four mean functions). Per level, `generate_inputs.configurations`
gives the baseline plus every one-axis move away from it — 12 configurations, the
baseline once — and each becomes one outer cell with one inner cell, level-major:

| cells | configurations | folds per cell | fits |
| --- | --- | --- | --- |
| 600 | 600 | 137 | 82,200 |

Solvent block, base block, base kernel (`matern52` on every level) and model class
never move. The three prior-scheme cells of a level share every outer key and
differ in the inner `prior_scheme`. `generate_inputs.sweep_membership` names the
sweep a configuration belongs to; the baseline belongs to all four. `--cat-set`
keeps only the levels whose source featurization is in one `config.CATALYST_SETS`
family; it overwrites `inputs.json`.

### 7.3 Runtime and timing probe

`scheduler estimate-runtime` projects every cell from the seconds per fold that
`timing_probe` measured for its level at three widths, interpolated in the cell's
design-matrix width (`scheduler._PRIMARY_SECONDS_PER_FOLD`), scaled per fold by
training-set size (`_LOO_TO_PRIMARY_COST`), plus 67 s startup. It lists the slowest
cells, CPU-h per sweep, and the wall time when cells are claimed in id order on
50–600 slots. The measured table and the current projection are in
[WORK-PACKAGE.md](WORK-PACKAGE.md) "Runtime".

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler estimate-runtime
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe list
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe run --task 0 --folds 1   # one task, locally
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe submit --folds 2         # all 36 on Mechthild
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe summarize
```

### 7.4 Submission

One SLURM task per cell (`worker --single`), at most 300 at a time on `medium` with
`--time=23:59:00`; `autoscale --single` hands each freed slot the next pending
cell, so the 600 cells run as two rounds of 300. The generated `sbatch` script is
in [WORK-PACKAGE.md](WORK-PACKAGE.md) "Operational".

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild submit --single --force
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild autoscale --single
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler status
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler submitted
```

### 7.5 Fold failures

A fold fails when `fit_eval` raises; the scheduler records it as
`status: "failed"` with the error string. The next `init` re-queues it; the fold
seed is fixed, so a re-run repeats the same fit.

| criterion | error |
| --- | --- |
| every L-BFGS restart ends with a non-finite MLL | `NoConvergedRestartError` |
| an L-BFGS restart ends ABNORMAL in all of botorch's fit attempts | `ModelFittingError` |
| Cholesky jitter ≥ 1e-3 | `ExcessiveJitterError` |
| fitted noise variance < 10 × final jitter (`config.MIN_NOISE_TO_JITTER_RATIO`) | `NoiseBelowJitterError` |
| any test-point μ or σ is NaN or inf, or σ ≤ 0 | `NonFinitePredictionError` |
| any other exception, including `NotPSDError` | its own type |

Noise and jitter are both on the standardized scale. The final jitter is the
largest jitter gpytorch added while factorising the training covariance at the
selected hyperparameters (the prediction step); with no jitter warning it is 0.
gpytorch escalates 1e-6 → 1e-5 → 1e-4 and then raises `NotPSDError`, so
`ExcessiveJitterError` cannot fire at the default three tries.


## 8. Analysis

```bash
python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results --require-complete
python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --cv-splits pablo/featurization_and_kernel_scan/data/grid/cv_splits.json
python -m pablo.featurization_and_kernel_scan.results_analysis.scan_summary_tables
python -m pablo.featurization_and_kernel_scan.results_analysis.plot_heatmap_tables
```

`concat_results` writes `results/scan_aggregate.parquet`, one row per fold with
`status` `ok` or `failed`; a failed row carries `error`, NaN scalar metrics and
empty per-point arrays, and every consumer filters on `status`.
`--require-complete` stops on an absent fold (neither ok nor failed).

`gem_scores` writes `results/gem_scores.parquet`, one row per configuration. Per
study, failed folds are dropped and CRPSS is pooled,
`1 − ΣCRPS_model / ΣCRPS_baseline` over every test point of the ok folds. The gem
is the smaller of the pooled CRPSS on `primary` and on `leave_one_out`;
`gem_study` names which. Per study it also reports `n_ok`, `n_failed`,
`ineligible` (n_failed > floor(0.05 · n_folds)), `not_production_ready`
(n_failed > 0), and MSLL, MAE, RMSE, Q² and the calibration KS statistic as
medians over ok folds. Flags mark a configuration and never remove it. Every
configuration must carry every fold of `cv_splits.json`, or `gem_scores` stops.

The elimination rule (WORK-PACKAGE.md "Elimination phase") is applied to
`gem_scores.parquet` after the sweep; every configuration stays on disk.

`scan_summary_tables.py` writes per-`model_class` CSVs and Markdown into
`results/`. `plot_heatmap_tables.py` renders flat PNGs into
`results_analysis/` named
`<model_class>_<case_study>[_<solvent>]_<plot_kind>.png` plus
`case_study_winners.png`.

---

## 9. Reproducibility checklist

- Use pinned package versions.
- Keep featurizations raw; a reduction is its own stored `<name>_pca/`
  featurization, never computed inside a worker.
- Keep CV splits frozen across cells.
- Preserve the `raw_model_output.parquet` artifact for every feature pipeline.
- Record model IDs, pooling choices, and preprocessing assumptions in
  `metadata.json` for each featurization.
- Every `result_<idx>.json` carries the SHA-256 of `cv_splits.json` plus a
  git commit hash and SLURM IDs. Noise-prior β rates are recomputed at
  import from `data/libraries/experiments.parquet`.

---

## 10. Troubleshooting

### General

- **Missing Mol2vec weights.** Place `mol2vec_weights.pkl` under `data/` or
  pass `--mol2vec-model`.
- **Foundation-model failure on Mechthild.** Check
  `data/foundation_model_logs/*.err` first.
- **Unexpected embedding dimensions.** Inspect
  `raw_model_output.parquet`, `descriptors.parquet`, and `metadata.json`.

### xTB / SOAP (local)

- `xtb: command not found` → install via conda-forge or set the path:
  `conda install -c conda-forge xtb`.
- `dscribe` import failure → `pip install dscribe`.
- `morfeus`, `dbstep` import failures →
  `pip install morfeus-ml dbstep`.

### Mechthild SLURM

- **`module load cuda/12.6.2` fails.** `module avail cuda` on a login node;
  update the version in `features/foundation.py`.
- **GPU not found.** Mechthild uses `--gpus=H100:1`, not `--gres=gpu:1`.
- **HuggingFace download hangs.** Pre-download model weights on the login
  node (§3.2). The SLURM script sets `HF_HOME`.
- **GROVER `_pickle.UnpicklingError: invalid load key, '<'`.** The
  checkpoint file is an HTML page — re-download with `gdown`.
- **`torch` gets overwritten with cu130.** Any package with an unpinned
  `torch` dep pulls cu130. Restore with
  `uv pip install --reinstall "torch==2.6.0+cu124"
   --index-url https://download.pytorch.org/whl/cu124`. Install third-party
  packages with `--no-deps`.
- **MolFormer fails with `No module named 'transformers.onnx'`** —
  `transformers` must stay `<4.40.0`.
- **ChemBERTa fails with `operator torchvision::nms does not exist`** —
  re-install matching `torchvision==0.21.0+cu124`.

### Interactive GPU debug

```bash
sinteractive -p gpu --gpus=H100:1 --mem=32G --cpus-per-task=4 --time=01:00:00
module load cuda/12.6.2
source venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4

python -m pablo.featurization_and_kernel_scan.features build \
    molformer_xl_precat --force
```

---

## 11. Quick command summary

```bash
# Local descriptor pipelines
python -m pablo.featurization_and_kernel_scan.features build --family fingerprint --family component
# xTB descriptors — Stage 1 (local)
python -m pablo.featurization_and_kernel_scan.features build tally_topology
python -m pablo.featurization_and_kernel_scan.features build n_c_aggregates
python -m pablo.featurization_and_kernel_scan.features build lp_sterimol
python -m pablo.featurization_and_kernel_scan.features build vbur_quadrants
python -m pablo.featurization_and_kernel_scan.features build interpretable_steric
python -m pablo.featurization_and_kernel_scan.features build interpretable_light
# xTB descriptors — Stage 2 (Mechthild, see §5b.6)
python -m pablo.featurization_and_kernel_scan.features build bh3_reference
python -m pablo.featurization_and_kernel_scan.features build xtb_extra
python -m pablo.featurization_and_kernel_scan.features build probe_h_plus
python -m pablo.featurization_and_kernel_scan.features build probe_bh3
python -m pablo.featurization_and_kernel_scan.features build interpretable
# xTB descriptors — Stage 2b, ΔG_deprot (Mechthild, see §5b.6b)
python -m pablo.featurization_and_kernel_scan.features build deltaG_deprot
python -m pablo.featurization_and_kernel_scan.features build interpretable_dG

# Foundation models on Mechthild (6 parallel GPU jobs)
python -m pablo.featurization_and_kernel_scan.features submit

# Inspect
# CV / kernel scan — elimination grid (600 cells × 1 inner × 137 folds)
python -m pablo.featurization_and_kernel_scan.cross_validation.cv_splits
python -m pablo.featurization_and_kernel_scan.cross_validation.timing_probe submit --folds 2      # then: timing_probe summarize
python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler estimate-runtime  # pre-flight projection
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler init
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild submit --single --force
python -m pablo.featurization_and_kernel_scan.cross_validation.submit_mechthild autoscale --single
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler submitted          # live per-worker view
python -m pablo.featurization_and_kernel_scan.cross_validation.scheduler status             # counts + mean wall + ETA
python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results --require-complete

# Analysis
python -m pablo.featurization_and_kernel_scan.results_analysis.gem_scores --cv-splits pablo/featurization_and_kernel_scan/data/grid/cv_splits.json
python -m pablo.featurization_and_kernel_scan.results_analysis.scan_summary_tables
python -m pablo.featurization_and_kernel_scan.results_analysis.plot_heatmap_tables

# Tests
pytest tests/
```
