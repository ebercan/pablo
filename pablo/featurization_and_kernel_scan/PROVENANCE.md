# Provenance of the pretrained catalyst featurizations

This file states how the pretrained-model featurizations of the 161-catalyst library were produced, so that each matrix can be rebuilt and checked. It covers `grover`, `grover_carbene`, `grover_ionstripped`, `chemeleon_precat`, `chemeleon_carbene`, `monroe_precat`, `monroe_carbene`, `smited_precat`, `smited_carbene`, their `_pca` reductions and `monroe_precat_pca8`. The other pretrained featurizations name their source repository in `README.md` §3; their versions and weight checksums are not recorded.

Every model runs outside the main environment, in its own virtual environment under `~/Desktop/<model>/` on the build machine. The repository holds only the code that calls them (`features/gnn.py`, `features/foundation.py`) and one thin wrapper per model (`features/venv_wrappers/<model>_embed.py`) that the model's own interpreter executes. Model code, weights and environments are not in the repository.

## Build machine

| | |
|---|---|
| Machine | AMD Ryzen 5 4600H, 6 cores, no GPU; all inference on CPU |
| OS | Linux Mint 22.3 |
| Python | 3.12.3 in every model environment |
| Environment tool | `uv` 0.11.13 |
| Main `pablo` environment | `requirements.txt` (`.venv/`) |

## Input SMILES

All models read `data/libraries/catalysts.parquet` (161 rows, column `smiles`) in catalyst-id order, through `common.load_catalyst_library_with_forms()`. Three input forms exist:

| Suffix | Input | Code |
|---|---|---|
| `_precat` (`grover` has no suffix) | the pre-catalyst SMILES as stored: salts with counter-ion, CO₂ adducts, bicarbonate, free carbenes | — |
| `_carbene` | the free carbene of each pre-catalyst | `common.to_free_carbene_smiles` |
| `_ionstripped` | the pre-catalyst without its counter-ion: the `.`-fragment with the most heavy atoms; 148 of 161 SMILES change (147 salts, 1 bicarbonate) | `common.strip_counterion_smiles` |

## Post-processing (identical for every model)

1. `raw_model_output.parquet` is the wrapper's output as float64.
2. `descriptors.parquet` is the same matrix L2-normalised per row (`common.l2_normalize`).
3. `<name>_pca/descriptors.parquet` is `features.dim_reduction.build_pca_featurization(<name>)`: `StandardScaler`, `PCA(random_state=42)` over all 161 rows, components up to the Kneedle knee (`S = 1`) of the explained-variance curve. The knee and retained variance are in each `_pca/metadata.json`.
   `monroe_precat_pca8/` is `build_pca_featurization('monroe_precat', n_components=8)`: the same fit truncated at 8 components (73.6% of the variance; the knee is 15). Its columns equal the first 8 columns of `monroe_precat_pca/` to 1e-13.
4. In the CV grid, `_carbene` levels are combined with the four-column pre-catalyst-form one-hot (`append_form_onehot_to_cat = True`), as for every other `_carbene` level.

## Rebuild commands

From the repo root, with the four environments installed as below:

```bash
python -m pablo.featurization_and_kernel_scan.features build \
    grover_carbene grover_ionstripped \
    chemeleon_precat chemeleon_carbene \
    monroe_precat monroe_carbene \
    smited_precat smited_carbene

python -c "from pablo.featurization_and_kernel_scan.features.dim_reduction import build_all; build_all(['grover_carbene', 'grover_ionstripped', 'chemeleon_precat', 'chemeleon_carbene', 'monroe_precat', 'monroe_carbene', 'smited_precat', 'smited_carbene'])"

python -c "from pablo.featurization_and_kernel_scan.features.dim_reduction import build_pca_featurization; build_pca_featurization('monroe_precat', n_components=8)"
```

The pre-catalyst `grover` matrix is rebuilt through `features.foundation` with the Desktop GROVER environment:

```bash
GROVER_REPO_DIR=~/Desktop/grover \
GROVER_PYTHON=~/Desktop/grover/.venv/bin/python \
GROVER_CHECKPOINT=~/Desktop/grover/weights/grover_large.pt \
python -m pablo.featurization_and_kernel_scan.features build grover --force
```

Each `build` writes `data/<model>_tmp/` (input CSV and intermediates) next to `data/featurizations/<model>/`.

## GROVER-large

| | |
|---|---|
| Paper | Rong et al., *Self-Supervised Graph Transformer on Large-Scale Molecular Data*, NeurIPS 2020 |
| Code | https://github.com/tencent-ailab/grover, commit `40b6d97098e4508687912f3c05eca369fc2c6213` (2026-02-25) |
| Weights | GROVER-large, Google Drive file `1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw` (link in the repository README), 428,605,550 bytes, SHA-256 `4b0c436fbd6ed8539fa92a0c9f890878f5e3dd8591d959315e773efb6302baaa`, retrieved 2026-10-08 11:01 UTC |
| Licence | MIT (repository) |
| Output | 5,000 columns: mean readout of `atom_from_atom`, `atom_from_bond`, `bond_from_atom`, `bond_from_bond` (1,200 each, in this order), then 200 `rdkit_2d_normalized` descriptors (descriptastorus, CDF-scaled to [0, 1]) |
| Wrapper | `venv_wrappers/grover_embed.py` |

Installation:

```bash
git clone https://github.com/tencent-ailab/grover ~/Desktop/grover
cd ~/Desktop/grover && git checkout 40b6d97098e4508687912f3c05eca369fc2c6213
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
uv pip install --python .venv/bin/python rdkit==2026.3.1 numpy==2.2.6 pandas==3.0.3 scikit-learn==1.8.0 \
    scipy tqdm descriptastorus==2.8.0 tensorboard gdown
mkdir -p weights && .venv/bin/gdown 1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw -O weights/grover_large.pt
sha256sum weights/grover_large.pt
```

The wrapper runs GROVER's own two CLI steps with the environment's interpreter: `scripts/save_features.py --features_generator rdkit_2d_normalized`, then `main.py fingerprint --fingerprint_source both --features_path <descriptors>`. The released checkpoint lacks eleven arguments that the current code reads (`dropout`, `nencoders`, `ffn_hidden_size`, …); the wrapper adds them to an in-memory copy, writes it beside the checkpoint as `grover_large.patched.pt` and leaves the downloaded file unchanged. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set because the checkpoint pickles an `argparse.Namespace`.

**The RDKit version changes the descriptor block.** With `rdkit==2025.9.3`, `NumHAcceptors` (descriptor column 57, matrix column 4,857) differs in 102 of 161 rows by up to 0.185 on the CDF scale; with `rdkit==2026.3.1` all 200 descriptor columns equal the stored matrix to 1.1e-16. The stored `grover` matrix was computed on Mechthild (data committed 2026-05-03) and matches `rdkit==2026.3.1`.

**Reproduction of the stored `grover` matrix on the build machine:** the 4,800 embedding columns agree to a maximum absolute difference of 1.1e-5 (float32 inference, GPU against CPU), the 200 descriptor columns exactly, and the rebuilt `grover_pca` agrees with the stored one to 2.2e-5 in every component.

## CheMeleon

| | |
|---|---|
| Paper | Burns, Zalte, Green, *Deep Learning Foundation Models from Classical Molecular Descriptors*, arXiv 2506.15792 |
| Code | https://github.com/JacksonBurns/chemeleon, commit `b5e5bd88a7070c3f278e7e65e85b792617bf125c` (2026-07-23); the wrapper reproduces its `chemeleon_fingerprint.py` |
| Weights | Zenodo record 15460715 (DOI 10.5281/zenodo.15426600), file `chemeleon_mp.pt`, 34,859,448 bytes, MD5 `6a80b54fdb7de37ef0374d302f01e8ce` (equal to the Zenodo record), SHA-256 `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`, retrieved 2026-10-08 11:08 UTC |
| Licence | MIT (code and weights) |
| Output | 2,048 columns: Chemprop `BondMessagePassing` with the checkpoint's hyperparameters, mean aggregation, `MPNN.fingerprint` |
| Wrapper | `venv_wrappers/chemeleon_embed.py` |

```bash
git clone https://github.com/JacksonBurns/chemeleon ~/Desktop/chemeleon
cd ~/Desktop/chemeleon && git checkout b5e5bd88a7070c3f278e7e65e85b792617bf125c
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
uv pip install --python .venv/bin/python chemprop==2.3.1 rdkit==2026.3.1 numpy
mkdir -p weights && curl -L -o weights/chemeleon_mp.pt https://zenodo.org/records/15460715/files/chemeleon_mp.pt
md5sum weights/chemeleon_mp.pt
```

Input handling: `rdkit.Chem.MolFromSmiles` on the SMILES as given; Chemprop's default featurizer encodes formal charge and covers atomic numbers 1 to 36 and 53, so every element of the library (H, B, C, N, O, F, P, S, Cl, Br, I). A salt is one disconnected graph; a bond-less ion (`[Cl-]`) receives no messages and enters the mean with its input features. On the first 10 catalysts the wrapper's output equals `CheMeleonFingerprint` from the repository exactly.

## Monroe

| | |
|---|---|
| Paper | Banaszewski, Fitzgibbon, *Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference*, arXiv 2608.18982 |
| Code | https://github.com/blazejba/monroe, commit `57238edfffea03808abe761a00cd9a75fa41bb95` (2026-08-24), installed with `pip install -e` |
| Weights | `checkpoint/weights.pt` of that commit (Git LFS object), downloaded from `https://media.githubusercontent.com/media/blazejba/monroe/57238edfffea03808abe761a00cd9a75fa41bb95/checkpoint/weights.pt`, 300,057,823 bytes, SHA-256 `df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a` (equal to the LFS pointer's `oid`), retrieved 2026-10-08 11:16 UTC; `checkpoint/config.json` from the same commit |
| Licence | MIT |
| Output | 720 columns: GRIT encoder graph-level embedding |
| Wrapper | `venv_wrappers/monroe_embed.py` |

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/blazejba/monroe ~/Desktop/monroe
cd ~/Desktop/monroe && git checkout 57238edfffea03808abe761a00cd9a75fa41bb95
curl -L -o checkpoint/weights.pt \
  https://media.githubusercontent.com/media/blazejba/monroe/57238edfffea03808abe761a00cd9a75fa41bb95/checkpoint/weights.pt
sha256sum checkpoint/weights.pt
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
uv pip install --python .venv/bin/python -e . rdkit==2026.3.1
```

The wrapper runs Monroe's documented path (`monroe.eval.embed.featurize_smiles`, `to_pyg`, encoder forward pass in batches of 32) with three changes:

1. **Seeded conformer.** Monroe's encoder reads 3D coordinates. Upstream embeds one ETKDGv3 conformer without a seed, so every run gives different embeddings. The wrapper sets `randomSeed = 0` (`gnn.MONROE_SEED`); two runs then agree exactly.
2. **No silent drops.** Upstream leaves out molecules whose graph cannot be built; the wrapper raises unless all 161 come back.
3. **CPU loading.** The encoder is restored with `map_location="cpu"`, and the load raises if any encoder parameter is missing from the checkpoint.

Upstream featurization, kept as is: the SMILES is converted to InChI and rebuilt from it. This changes the canonical SMILES of 54 of the 161 pre-catalysts, mostly by moving the formal charge to a resonance-equivalent nitrogen (triazolium salts), and cannot rebuild a free carbene, whose graph is then built directly from the SMILES. The conformer is relaxed with MMFF94s (UFF when MMFF has no parameters). A conformer that fails, or takes longer than 10 s, is replaced by a flat 2D layout. On the build machine this happened for 1 pre-catalyst (the [2.2]paracyclophane-substituted imidazolinium chloride) and for 13 free carbenes (12 thiazol- and benzothiazol-2-ylidenes and the bis-paracyclophane imidazolinylidene); the list is written to `data/monroe_<form>_tmp/embeddings.npy.flat2d.txt`. The 10 s limit depends on machine speed, so a slower machine can produce more flat layouts.

## SMI-TED-Light

| | |
|---|---|
| Paper | Soares et al., *A Large Encoder-Decoder Family of Foundation Models For Chemical Language*, arXiv 2407.20267 |
| Code and weights | Hugging Face `ibm-research/materials.smi-ted`, revision `414c3ea0a8603ef49d1c5bb3db336e09877c01ce`; files `smi-ted-Light_40.pt` (1,155,124,350 bytes, SHA-256 `baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375`), `smi-ted/inference/smi_ted_light/load.py` (SHA-256 `e4fb006a990b4aba5807d64b2dd7f0a17cb4ef79743e1dae130583de2639def9`) and `bert_vocab_curated.txt` (SHA-256 `8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95`), retrieved 2026-10-08 11:10 UTC |
| Licence | Apache-2.0 |
| Output | 768 columns: `Smi_ted.encode`, the pretrained autoencoder's compression of the encoder's token embeddings |
| Wrapper | `venv_wrappers/smited_embed.py` |

```bash
mkdir -p ~/Desktop/smi_ted && cd ~/Desktop/smi_ted
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python huggingface_hub
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-research/materials.smi-ted', revision='414c3ea0a8603ef49d1c5bb3db336e09877c01ce', local_dir='hf_snapshot', allow_patterns=['smi-ted/inference/smi_ted_light/load.py', 'smi-ted/inference/smi_ted_light/bert_vocab_curated.txt', 'smi-ted-Light_40.pt', 'requirements.txt', 'README.md', 'config.json'])"
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
uv pip install --python .venv/bin/python transformers regex rdkit==2026.3.1 numpy pandas tqdm scipy scikit-learn setuptools wheel
uv pip install --python .venv/bin/python --no-build-isolation pytorch-fast-transformers==0.4.0   # compiles C++ extensions, ~3 min
```

Input handling: `load.normalize_smiles` rewrites each SMILES as RDKit canonical SMILES without stereochemistry, then a regex tokenizer maps it onto the 2,393-token vocabulary, which contains every token of the library (`[B-]`, `[P-]`, `[C]`, `[Cl-]`, `[Br-]`, `[I-]`, `[n+]`, `.`). Upstream returns `None` for an unparsable SMILES and truncates silently at 202 tokens; the wrapper raises on either and on any token outside the vocabulary. Embeddings of the first 10 catalysts computed alone and within the full batch agree to 9.3e-6.

## Determinism checks on the build machine

| Model | Check | Result |
|---|---|---|
| GROVER | rebuild of the stored `grover` matrix | embeddings ≤ 1.1e-5, descriptors exact |
| CheMeleon | `build` output against a direct wrapper run | identical |
| Monroe | two runs with seed 0; `build` output against a direct run | identical |
| SMI-TED | `build` output against a direct run; batch of 10 against batch of 161 | identical; ≤ 9.3e-6 |

## Package versions

The complete `uv pip freeze` of each environment follows. The editable `monroe` install (`-e file:///…/monroe`) is the checkout above.

<details><summary>GROVER (`~/Desktop/grover/.venv`), 42 packages</summary>

```
absl-py==2.5.0
beautifulsoup4==4.15.0
certifi==2026.7.22
charset-normalizer==3.5.2
cloudpickle==3.1.2
descriptastorus==2.8.0
filelock==3.32.3
fsspec==2026.7.0
gdown==6.4.1
grpcio==1.84.0
idna==3.20
jinja2==3.1.6
joblib==1.6.0
markdown==3.11
markupsafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.2.6
packaging==26.3
pandas==3.0.3
pandas-flavor==0.8.1
pillow==12.3.0
protobuf==7.36.2
pysocks==1.7.1
python-dateutil==2.9.0.post0
rdkit==2026.3.1
requests==2.34.2
scikit-learn==1.8.0
scipy==1.18.1
setuptools==78.1.0
six==1.17.0
soupsieve==2.10
sympy==1.13.1
tensorboard==2.21.0
tensorboard-data-server==0.7.2
threadpoolctl==3.7.0
torch==2.6.0+cpu
tqdm==4.70.1
typing-extensions==4.16.0
urllib3==2.8.0
werkzeug==3.1.9
xarray==2026.9.0
```

</details>

<details><summary>CheMeleon (`~/Desktop/chemeleon/.venv`), 58 packages</summary>

```
aimsim-core==2.2.3
aiohappyeyeballs==2.7.1
aiohttp==3.14.4
aiosignal==1.4.0
astartes==1.3.3
attrs==26.1.0
chemprop==2.3.1
cloudpickle==3.1.2
configargparse==1.8.0
cuik-molmaker-pin==2026.3.1
descriptastorus==2.8.0
dill==0.4.1
filelock==3.32.3
frozenlist==1.8.0
fsspec==2026.7.0
idna==3.20
jinja2==3.1.6
joblib==1.6.0
lightning==2.6.6
lightning-utilities==0.15.3
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
mhfp==1.9.6
mordredcommunity==2.0.7
mpmath==1.3.0
multidict==6.9.1
multiprocess==0.70.19
myerson==1.0.1
narwhals==2.26.0
networkx==3.6.1
numpy==2.5.3
packaging==26.3
padelpy==0.1.17
pandas==3.0.6
pandas-flavor==0.8.1
pillow==12.3.0
propcache==0.5.4
psutil==7.2.2
pygments==2.21.0
python-dateutil==2.9.0.post0
pytorch-lightning==2.6.6
pyyaml==6.0.3
rdkit==2026.3.1
rich==15.0.0
scikit-learn==1.9.1
scipy==1.18.1
setuptools==78.1.0
six==1.17.0
sympy==1.13.1
tabulate==0.10.0
threadpoolctl==3.7.0
torch==2.6.0+cpu
torchmetrics==1.9.0
tqdm==4.70.1
typing-extensions==4.16.0
xarray==2026.9.0
yarl==1.25.1
```

</details>

<details><summary>Monroe (`~/Desktop/monroe/.venv`), 176 packages</summary>

```
aiobotocore==3.9.2
aiohappyeyeballs==2.7.1
aiohttp==3.14.4
aioitertools==0.13.0
aiosignal==1.4.0
annotated-types==0.8.0
anyio==4.15.1
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
botocore==1.43.106
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
comm==0.2.3
contourpy==1.4.0
cryptography==50.0.2
cycler==0.12.1
datamol==0.13.0
debugpy==1.8.22
decorator==5.3.1
defusedxml==0.7.1
et-xmlfile==2.0.0
executing==2.2.1
fastjsonschema==2.22.2
filelock==3.32.3
fonttools==4.66.1
fqdn==1.6.0
frozenlist==1.8.0
fsspec==2026.7.0
gcsfs==2026.8.1
google-api-core==2.41.0
google-auth==2.61.0
google-auth-oauthlib==1.5.0
google-cloud-core==2.8.0
google-cloud-storage==3.16.0
google-cloud-storage-control==1.16.0
google-crc32c==1.9.0
google-resumable-media==2.11.0
googleapis-common-protos==1.75.5
grpc-google-iam-v1==0.14.5
grpcio==1.84.0
grpcio-status==1.84.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython-pygments-lexers==1.1.1
ipywidgets==8.1.9
isoduration==20.11.0
jedi==0.20.0
jinja2==3.1.6
jmespath==1.1.0
joblib==1.6.0
json5==0.16.0
jsonpointer==3.2.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-builder==1.2.3
jupyter-client==8.10.0
jupyter-core==5.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter-server==2.21.1
jupyter-server-terminals==0.5.4
jupyterlab==4.6.4
jupyterlab-pygments==0.3.0
jupyterlab-server==2.28.1
jupyterlab-widgets==3.0.17
kiwisolver==1.5.1
lark==1.3.1
loguru==0.7.3
markupsafe==3.0.3
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.3.4
mpmath==1.3.0
multidict==6.9.1
narwhals==2.26.0
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.6.1
nglview==4.0.1
notebook==7.6.3
notebook-shim==0.2.4
numpy==2.5.3
oauthlib==4.0.0
openpyxl==3.1.5
opentelemetry-api==1.45.1
opentelemetry-exporter-http-transport==0.66b1
opentelemetry-exporter-otlp-common==0.66b1
opentelemetry-exporter-otlp-proto-common==1.45.1
opentelemetry-exporter-otlp-proto-http==1.45.1
opentelemetry-proto==1.45.1
opentelemetry-sdk==1.45.1
opentelemetry-semantic-conventions==0.66b1
packaging==26.3
pandas==3.0.6
pandocfilters==1.5.1
parso==0.8.7
pexpect==4.9.0
pillow==12.3.0
platformdirs==4.12.4
prometheus-client==0.26.0
prompt-toolkit==3.0.53
propcache==0.5.4
proto-plus==1.29.0
protobuf==7.36.2
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.4
pyarrow==25.0.1
pyasn1==0.6.4
pyasn1-modules==0.4.2
pycparser==3.0
pydantic==2.13.5
pydantic-core==2.46.5
pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
python-json-logger==4.2.0
pyyaml==6.0.3
pyzmq==27.2.0
rdkit==2026.3.1
referencing==0.37.0
requests==2.34.2
requests-oauthlib==2.0.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
s3fs==2026.7.0
scikit-learn==1.9.1
scipy==1.18.1
selfies==2.2.0
send2trash==2.1.0
setuptools==78.1.0
six==1.17.0
soupsieve==2.10
stack-data==0.6.3
sympy==1.13.1
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
torch==2.6.0+cpu
torch-geometric==2.8.0.post1
tornado==6.5.10
tqdm==4.70.1
traitlets==5.16.1
typing-extensions==4.16.0
typing-inspection==0.4.4
tzdata==2026.5
uri-template==1.3.0
urllib3==2.8.0
wandb==0.30.0
wcwidth==0.9.2
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
widgetsnbextension==4.0.16
wrapt==2.5.0
xlsxwriter==3.2.9
xxhash==4.0.1
yarl==1.25.1
```

</details>

<details><summary>SMI-TED (`~/Desktop/smi_ted/.venv`), 50 packages</summary>

```
annotated-doc==0.0.5
anyio==4.15.1
certifi==2026.7.22
click==8.5.0
cloudpickle==3.1.2
filelock==4.0.12
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore==1.0.9
httpcore2==2.13.1
httpx==0.28.1
httpx2==2.13.1
huggingface-hub==1.33.0
idna==3.20
jinja2==3.1.6
joblib==1.6.0
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
narwhals==2.26.0
networkx==3.6.1
numpy==2.5.3
packaging==26.3
pandas==3.0.6
pillow==12.3.0
pygments==2.21.0
python-dateutil==2.9.0.post0
pytorch-fast-transformers==0.4.0
pyyaml==6.0.3
rdkit==2026.3.1
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
setuptools==78.1.0
shellingham==1.5.4
six==1.17.0
sympy==1.13.1
threadpoolctl==3.7.0
tokenizers==0.23.2
torch==2.6.0+cpu
tqdm==4.70.1
transformers==5.19.0
truststore==0.10.4
typer==0.27.3
typing-extensions==4.16.0
wheel==0.48.0
```

</details>
