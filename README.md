# PABLO — Procurement-aware Bayesian Lookahead Optimisation for the NHC-catalyzed acyloin condensation

`PABLO` (Procurement-aware Bayesian Lookahead Optimisation) is a small, opinionated Bayesian Optimization (BO) framework built to drive a wet-lab campaign on the **NHC-catalyzed acyloin condensation of formaldehyde**. Each iteration trains a Gaussian Process (GP) on the lab's accumulated yields and proposes the next q=6 batch of experiments over a mixed continuous–discrete reaction space.

The canonical end-to-end pipeline is `pablo/main.py`. Two larger investigations live alongside it:

- [`pablo/featurization_and_kernel_scan/`](pablo/featurization_and_kernel_scan/) — large GP model-selection study (catalyst featurization × kernel structure × transforms).
- [`pablo/plan_ahead.py`](pablo/plan_ahead.py) — lookahead / value-of-information acquisition (under active redesign).

---

## 1. What the BO loop optimizes

We maximize **isolated yield** of the acyloin condensation by jointly searching over:

**Continuous variables (4)**
- Temperature (°C) — default bounds `(25, 100)`
- Formaldehyde Ratio (eq) — default bounds `(1.0, 3.0)`
- Catalyst Loading (molar ratio) — default bounds `(0.5, 10.0)`
- Time (h) — default bounds `(16, 20)`; currently fixed at 20 h in `generate_batch` via `fixed_dims`.

**Discrete variables (3)**
- **Catalyst** — the primary lever. NHC pre-catalysts in three classes (true ionic salts, bicarbonate salts, free NHC / CO₂ adduct). The classifier `requires_external_base(smiles)` in [`pablo/main.py`](pablo/main.py) routes each catalyst to whether it needs an external base.
- **Solvent**
- **Base** — automatically zeroed out (and duplicates collapsed) for non-salt catalysts, since they self-activate.

Each discrete component is encoded by a **raw, un-reduced featurization** loaded from `pablo/featurization_and_kernel_scan/data/featurizations/<name>/descriptors.parquet`. Which featurization is used is set by `CAT_FEATURES` / `SOLVENT_FEATURES` / `BASE_FEATURES` in [`pablo/main.py`](pablo/main.py). No dimensionality reduction is applied anywhere in the pipeline.

**Batch shape.** The default proposal is q=6 with a hard-wired temperature partition `(2, 2, 2)` — three distinct temperature levels, two candidates each — implemented as a batch constraint in [`pablo/optimize.py`](pablo/optimize.py).

**Acquisition.** `qLogExpectedImprovement` for batch generation; `LogExpectedImprovement` for the q=1 scope-expansion path.

**Surrogate.** `SingleTaskGP` (BoTorch) with `Normalize` (inputs) + `Standardize` (outcomes) transforms, fit via `fit_gpytorch_mll`. Everything runs in `torch.float64`, GPU when available.

**Optimizer.** Mixed-variable acquisition optimization in [`pablo/optimize.py`](pablo/optimize.py). Three backends: a `pymoo` GA (`optimize_acqf_mixed_GA`, default), a greedy enumerator (`optimize_acqf_mixed_greedy`), and a sequential greedy variant (`optimize_acqf_sequential_greedy`).

---

## 2. Install

Two dependency manifests live at the repo root — **don't conflate them**:

- [`requirements.txt`](requirements.txt) — local-development stack (CPU `torch`, no cheminformatics extras beyond `rdkit` / `scikit-fingerprints`). This is what you need to run the main BO loop on your laptop.
- [`requirements_hpc.txt`](requirements_hpc.txt) — Mechthild stack (CUDA-pinned `torch==2.6.0+cu124`, plus xTB-side descriptors, MAP4, Mordred, Mol2Vec, dbstep, morfeus-ml, etc.). Required only for the heavy featurization sweeps.

Recommended local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When adding a dependency, decide which file it belongs in (or both) and follow the existing pinning style.

---

## 3. Quickstart

The canonical "propose the next batch" command is:

```bash
python -m pablo.main
```

This is equivalent to running [`pablo/main.py`](pablo/main.py) and will:

1. Load reference sheets from [`pablo/nhc_doe.xlsx`](pablo/nhc_doe.xlsx) and load the raw featurization blocks for all catalysts, solvents, and bases.
2. Filter to whatever is marked `"y"` in the **`Next Batch`** column of the `Available Catalysts` / `Solvents` / `Bases` sheets.
3. Train a `SingleTaskGP` on the `Experimental Design` sheet.
4. Optimize `qLogEI` over the mixed space with the `(2, 2, 2)` temperature partition and Time fixed at 20 h.
5. Print the decoded q=6 batch (Temperature, Formaldehyde Ratio, Catalyst Loading, Time, Catalyst ID + BLD, Solvent, Base, posterior mean & variance).
6. Write the proposed batch back to the `Experimental Design` sheet — either appending q new rows or filling existing result-less rows.

To also propose a single new catalyst to **order** (selected from the catalysts currently marked `n,n,n`), set `EXTEND_SCOPE = True` at the top of [`pablo/main.py`](pablo/main.py).

Use explicit module paths, e.g. `python -m pablo.featurization_and_kernel_scan.inspect_features`.

---

## 4. Data — `nhc_doe.xlsx`

The single source of truth is [`pablo/nhc_doe.xlsx`](pablo/nhc_doe.xlsx). It is **read-mostly**: only `update_experimental_design_with_batch` writes to it, and only into the `Experimental Design` sheet. Scripts should not modify it otherwise.

**Sheets:**

| Sheet | Role |
|---|---|
| `Experimental Design` | One row per experiment. Columns: Batch, Catalyst (ID), Solvent (name or ID), Base (label or ID), Formaldehyde Ratio, Catalyst Loading, Temperature, Time, **Yield**, plus annotation columns. New BO proposals are appended here (or fill result-less rows). |
| `Available Catalysts` / `Available Solvents` / `Available Bases` | Per-reagent availability flags across three batch windows (`Next Batch`, `Subsequent Batch`, `Following Batch`). `"y"` = available now, `"n"` = not available. Catalysts marked `n,n,n` are eligible for the **scope-expansion** path. |
| `Catalyst References Achiral` | Master list of NHC pre-catalysts: numeric `ID`, `smiles`, and (column L) a `BLD` order number for traceability. |
| `Solvent References` | Numeric `ID`, `Name`, `smiles` for each solvent. |
| `Base References` | Numeric `ID`, `Name`, `Comment` (the human-readable label, used by `decode_candidates`), `smiles`. |
| `Parameter Definition` | Bookkeeping / notes (not consumed by the pipeline). |

**Encoding contract.** `Catalyst` cells are integer IDs; `Solvent` and `Base` cells may be either the integer ID or the textual `Name`/`Comment`. Text resolution is case-insensitive, and the strings `"no" / "none" / "n/a" / "na" / "-"` in the `Base` column fall back to a sentinel base whose embedding is zeroed downstream.

---

## 5. Repository layout

```
pablo/                              ← repo root
├── pablo/                          ← main Python package
│   ├── main.py                     ← BO loop entry point (generate_batch, generate_scope_expansion,
│   │                                  update_experimental_design_with_batch)
│   ├── optimize.py                 ← mixed-variable acquisition optimizers
│   │                                  (GA via pymoo, greedy, sequential greedy, plan-ahead wrappers)
│   ├── local_penalizer.py          ← LocallyPenalizedAcquisitionFunction (L-HLP / soft-erfc variants)
│   ├── plan_ahead.py               ← PlanAheadNHC lookahead acquisition (active redesign → VOI-at-leaves)
│   ├── nhc_doe.xlsx                ← authoritative dataset (treat as read-only)
│   ├── utils/
│   │   ├── tensor_shapes.py        ← shape aliases + check_shape_and_bounds
│   │   ├── dim_reduction.py        ← PCA / UMAP helpers (apply_pca, apply_umap, excel_to_df)
│   │   ├── starting_set_generation.py
│   │   ├── nhc_utils.py            ← SMILES helpers, MolecularCodec
│   │   ├── catalyst_form_classifier.py
│   │   ├── feature_assembly.py
│   │   └── pca_utils.py
│   ├── featurization_and_kernel_scan/   ← WP1: GP model-selection study (see §6)
│   ├── mechthild_runs/                  ← HPC benchmark sweeps (see §7)
│   ├── results_analysis/                ← one-off analysis scripts (Kneedle, LOO, Pareto, …)
│   └── umap_chemical_space/             ← UMAP visualizations of the discrete space
├── benchmark_problems/             ← generic BO benchmarks (synthetic + real)
├── tests/                          ← pytest suite
├── requirements.txt                ← local-dev stack
├── requirements_hpc.txt            ← Mechthild stack
├── CLAUDE.md                       ← agent operational protocols
└── README.md                       ← (this file)
```

---

## 6. Work package: `featurization_and_kernel_scan/`

A large GP model-selection study (~42k fits) comparing combinations of:

- **Catalyst featurizations** — MAP4 / ECFP4 / Mordred / Mol2Vec, plus xTB-level interpretable descriptors, SOAP, and foundation-model embeddings (MolFormer-XL, ChemBERTa-2, CLAMP, UniMol, GROVER).
- **Kernel structures** — masked / additive variants that respect the base-active indicator.
- **Solvent/base treatments** and **outcome transforms**.

All foundation-model embeddings run on Mechthild GPU nodes; xTB, SOAP, and interpretable pipelines run locally. See the WP's own [README](pablo/featurization_and_kernel_scan/README.md), [IMPLEMENTATION.md](pablo/featurization_and_kernel_scan/IMPLEMENTATION.md), and [WORK-PACKAGE.md](pablo/featurization_and_kernel_scan/WORK-PACKAGE.md) for the operator manual, the artifact map, and the research brief respectively.

## 7. Work package: Plan-Ahead acquisition (`plan_ahead.py` + `mechthild_runs/`)

`pablo/plan_ahead.py` implements a rollout / lookahead acquisition (currently being redesigned toward a value-of-information-at-leaves target). The optimizer wrappers live in [`pablo/optimize.py`](pablo/optimize.py).

Large-scale empirical comparisons of acquisition optimizers, GA settings, and lookahead variants run on the **Mechthild SLURM cluster**, organized as self-contained sweeps under [`pablo/mechthild_runs/`](pablo/mechthild_runs/). Each `run_N_*` folder bundles its own config, input enumerator, file-based queue, dynamic SLURM submitter, and results layout — see the individual `README.md` files there for details. The shared HPC reference (partitions, GPU usage, software stack) is mirrored under [`pablo/mechthild_runs/mechthild_manual/`](pablo/mechthild_runs/mechthild_manual/).

---

## 8. Conventions

- **Default dtype:** `torch.float64`. Default device: CUDA if available, else CPU.
- **Transforms:** every `SingleTaskGP` uses `Normalize` (inputs) + `Standardize` (outcomes).
- **Tensor shapes:** see [`pablo/utils/tensor_shapes.py`](pablo/utils/tensor_shapes.py) for shape aliases and checks.

---

## 9. Citation

> *[Citation placeholder — replace with the published paper / preprint.]*

If you use this code, please cite the corresponding publication.

## 10. License

*[License placeholder — TBD.]*
# 🇨🇴 PaBLO - Plan ahead Bayesian Ligand Optimization

Bayesian Optimization based DoE that plans ahead of time for required catalyst ligands and reaction agents, so there are delays caused by ordering or synthesizing them.

> ⚠️ **Note**: This repository is currently under active development. The file structure is temporary and will be reorganized in future updates.

---

## File Overview

| File               | Description |
|--------------------|-------------|
| `main.py`          | Contains the main plan-ahead routine as applied in the actual laboratorial use case. |
| `bo.py`            | just a BO routine example (taken from an old repo of mine) and does not have to do with the plan ahead BO. |
| `moo.py`           | Testing custom  decorator `@torch_batch_evaluator` for PyMoo, that enables parallelisation for solving PyTorch based problems (such as acquisition functions). |
| `plan_ahead.py`    | Contains `LookAhead` class (copied from BoTorch) and based on that `PlanAhead` class (check for functionality!). |
| `example.py`       | compares lookahead and naïve BO performance. |
| `example2.py`      | tests plan ahead functionality **?** |
| `test_run.py`      | Testing similar to example.py **?** |
| `nhc_doe.xlsx`     | Excel file containing the input and output data for the experimental DoE run in `main.py`. The 'interface' of the experimentalist. |
| `requirements.txt`| Python dependencies — install with `pip install -r requirements.txt`. |

---

## Getting Started
**Clone the repository**
 ```bash
   git clone https://github.com/dionyce/pablo.git
   cd pablo
```

**Set up the environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
