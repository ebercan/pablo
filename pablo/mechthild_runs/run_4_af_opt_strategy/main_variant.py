"""run_4 AF-optimizer comparison: prepare + calibrate + worker entrypoints.

The sweep crosses two GP setups against four q=6 batch-decomposition schemes, each at
six GA-budget tiers, repeated over reps (see :mod:`config`):

  * **Models.** ``st_y`` — the validated positive-AF baseline: a plain ``SingleTaskGP``
    on yield only (Vanilla ARD-RBF, constant mean, raw transform), qLogNEI on the yield
    output (``yield_src="direct"``). ``lcm_ys`` — a rank-2 ``LCMMultiTaskGP`` (Q=2 latent
    groups, each with its own data kernel) on (Yield, Selectivity) with KS1; qLogNEI optimizes the **Yield** task only
    (``yield_src="marginal"`` → objective slices ``samples[..., yield_index]``).
  * **Batch-schemes.** A q=6 batch is built in sub-batches of sizes ``steps`` (``[6]``,
    ``[3, 3]``, ``[2, 2, 2]``, ``[1]*6``). At sub-batch ``i`` the GA frees ``steps[i]``
    new candidates with the already-chosen ones held fixed (``pre_batch``); the continuous
    refine then re-optimizes the continuous dims of all chosen points under the partial
    temperature partition for the cumulative count. The full (2, 2, 2) partition is
    enforced at the final sub-batch (cumulative k = 6) regardless of decomposition. The
    fully-sequential ``[1]*6`` reproduces the original greedy method; ``[6]`` is a single
    full-batch GA call.

Targets are rescaled to percent before fitting (see :func:`_scale_cell_targets`) so the
qLogEI values are positive and match ``pablo.main.train_gp_model``.

Each model is fit once by :func:`prepare` (on the featscan ``experiments.parquet``) and
all models serialize into a single shared ``.pt`` bundle (shared AF search space + one
state dict per model). Work is packaged as **per-cell workers**: each :func:`run_worker`
decodes its flat index into ``(rep, model, scheme)`` via :func:`_decode`, reconstructs
that model, and loops over all six tiers, writing the six per-tier records into one
result JSON. The mixed optimizers come from :mod:`pablo.optimize`; the gem model factory
and feature assembly from :mod:`pablo.featurization_and_kernel_scan`; availability + the
salt rule from :mod:`pablo.main`. A thin acqf adapter (:class:`_GemAcqfAdapter`) bridges
the optimizer's candidate layout to the model's feature layout.

Usage (from the repo root):
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant prepare
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant smoke [--ga-n-evals N] [--ga-pop-size N] [--model NAME]
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant calibrate [--probe-evals N]
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant worker --worker-id ID
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.main_variant cell --model NAME --scheme NAME --tier T [--rep R]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import time
from collections.abc import Callable
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction, qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from rdkit import Chem
from torch import Tensor

import pablo.main as pablo_main
from pablo.main import DTYPE, load_available_agents, requires_external_base
from pablo.optimize import (
    _build_step_batch_constraints,
    _normalize_batch_constraints,
    optimize_acqf_continuous,
    optimize_acqf_mixed_GA,
)

import pablo.featurization_and_kernel_scan.core.config as fks_config
import pablo.featurization_and_kernel_scan.paths as fks_paths
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import (
    _FIT_DISPATCH,
    _build_train_targets,
)
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    build_mean_module,
)
from pablo.utils.feature_assembly import (
    FORM_ID_ORDER,
    _load_numeric_descriptors,
    _resolve_catalyst_parquet,
    _resolve_feature_parquet,
    assemble_features,
)

from pablo.mechthild_runs.run_4_af_opt_strategy import config

ROOT = Path(__file__).parent
EXCEL_PATH = ROOT / "nhc_doe_copy.xlsx"
BUNDLE_PATH = ROOT / "data" / "prepared_bundle.pt"
RESULTS_DIR = ROOT / "data" / "results"

FKS_DATA_ROOT = fks_paths.DATA_DIR
FKS_LIB_ROOT = fks_paths.LIB_DIR
CANONICAL_FEATURIZATIONS_ROOT = fks_paths.FEAT_DIR
EXPERIMENTS_PATH = FKS_LIB_ROOT / "experiments.parquet"
CATALYSTS_PARQUET = FKS_LIB_ROOT / "catalysts.parquet"
SOLVENTS_PARQUET = FKS_LIB_ROOT / "solvents.parquet"
BASES_PARQUET = FKS_LIB_ROOT / "bases.parquet"
INTERPRETABLE_DESCRIPTORS = (
    CANONICAL_FEATURIZATIONS_ROOT / "interpretable" / "descriptors.parquet"
)

AVAILABILITY_COLUMN = "Next Batch"
D_CONTINUOUS = 4

# The gem model and optimizers run on CPU (float64), matching the featscan study
# and the CPU-only SLURM workers.
GEM_DEVICE = torch.device("cpu")


def _canon(smiles: str) -> str:
    """Return the RDKit canonical SMILES, hard-failing on a parse error."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def _map_to_fks_ids(
    smiles_list: list[str], library_parquet: Path, label: str
) -> list[int]:
    """Map reference SMILES to featscan library ids by canonical-SMILES match."""
    lib = pd.read_parquet(library_parquet)
    canon_to_id = {
        _canon(s): int(i) for i, s in zip(lib["id"].tolist(), lib["smiles"].tolist())
    }
    ids: list[int] = []
    for s in smiles_list:
        c = _canon(s)
        if c not in canon_to_id:
            raise ValueError(
                f"Available {label} SMILES not in featscan library: "
                f"{s!r} (canonical {c!r})."
            )
        ids.append(canon_to_id[c])
    return ids


def _form_ids_for(catalyst_ids: list[int]) -> np.ndarray:
    """Integer pre-catalyst form id (0..3) per featscan catalyst id."""
    df = pd.read_parquet(INTERPRETABLE_DESCRIPTORS)
    one_hot = [f"form_{label}" for label in FORM_ID_ORDER]
    labels = np.array(FORM_ID_ORDER)[df[one_hot].to_numpy().argmax(axis=1)]
    label_by_id = pd.Series(labels, index=df.index).reindex(catalyst_ids)
    if label_by_id.isna().any():
        raise ValueError("Missing pre-catalyst form for some available catalysts.")
    lookup = {f: i for i, f in enumerate(FORM_ID_ORDER)}
    return np.array([lookup[x] for x in label_by_id.to_numpy()], dtype=np.float64)


def _build_search_space() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the AF candidate pool: gem features for the available agents.

    Availability comes from the nhc_doe copy (``Next Batch``); each available agent's
    reference SMILES is matched (canonically) to a featscan library id and its raw
    feature row fetched. The catalyst table is augmented with two trailing columns
    ``[form_id, is_salt]`` consumed by :class:`_GemAcqfAdapter`. This is shared across
    both models (same grover / map / map features).
    """
    cat_ref = pd.read_excel(EXCEL_PATH, sheet_name="Catalyst References Achiral").dropna(subset=["smiles"])
    sol_ref = pd.read_excel(EXCEL_PATH, sheet_name="Solvent References").dropna(subset=["smiles"])
    bas_ref = pd.read_excel(EXCEL_PATH, sheet_name="Base References").dropna(subset=["smiles"])
    cat_smiles_by_id = dict(zip(cat_ref["ID"].tolist(), cat_ref["smiles"].astype(str).tolist()))
    sol_smiles_by_id = dict(zip(sol_ref["ID"].tolist(), sol_ref["smiles"].astype(str).tolist()))
    bas_smiles_by_id = dict(zip(bas_ref["ID"].tolist(), bas_ref["smiles"].astype(str).tolist()))

    available_cats, available_sols, available_bases = load_available_agents(
        EXCEL_PATH, AVAILABILITY_COLUMN
    )

    cat_fids = _map_to_fks_ids([cat_smiles_by_id[c] for c in available_cats], CATALYSTS_PARQUET, "catalyst")
    sol_fids = _map_to_fks_ids([sol_smiles_by_id[s] for s in available_sols], SOLVENTS_PARQUET, "solvent")
    bas_fids = _map_to_fks_ids([bas_smiles_by_id[b] for b in available_bases], BASES_PARQUET, "base")

    grover = _load_numeric_descriptors(
        _resolve_catalyst_parquet(CANONICAL_FEATURIZATIONS_ROOT, config.CAT_FEATURIZATION)
    )
    physical = _load_numeric_descriptors(
        _resolve_feature_parquet(CANONICAL_FEATURIZATIONS_ROOT / f"solvent_{config.SOLVENT_FEATURIZATION}")
    )
    structured = _load_numeric_descriptors(
        _resolve_feature_parquet(CANONICAL_FEATURIZATIONS_ROOT / f"base_{config.BASE_FEATURIZATION}")
    )

    cat_feats = grover.reindex(cat_fids).to_numpy(dtype=np.float64)
    sol_feats = physical.reindex(sol_fids).to_numpy(dtype=np.float64)
    bas_feats = structured.reindex(bas_fids).to_numpy(dtype=np.float64)
    for name, arr in (("grover", cat_feats), ("physical", sol_feats), ("structured", bas_feats)):
        if np.isnan(arr).any():
            raise ValueError(f"Missing {name} features for some available agents.")

    form_ids = _form_ids_for(cat_fids).reshape(-1, 1)
    is_salt = np.array(
        [float(requires_external_base(cat_smiles_by_id[c])) for c in available_cats]
    ).reshape(-1, 1)

    cat_aug = np.concatenate([cat_feats, form_ids, is_salt], axis=1)
    return (
        torch.tensor(cat_aug, dtype=DTYPE),
        torch.tensor(sol_feats, dtype=DTYPE),
        torch.tensor(bas_feats, dtype=DTYPE),
        torch.tensor(is_salt.reshape(-1) > 0.5, dtype=torch.bool),
    )


def _scale_cell_targets(cell):
    """Rescale the cell's raw fraction targets to percent, in place.

    The featscan loader stores yield / conversion / selectivity as fractions in
    ``[0, 1]`` (``*_pct / 100``). The model builder fits its ``Standardize`` outcome
    transform from these, and the fitter (``_fit_multistart``) ignores any externally
    passed ``train_y`` — so the only way to put the GP posterior (and hence the qLogEI
    objective) on the percent scale is to scale the cell's raw targets before the model
    is built. Only the ``raw`` fields are touched (both setups use ``OUTCOME_TRANSFORM
    = "raw"``); the logit fields are left untouched. See
    :data:`config.TARGET_PERCENT_SCALE`.
    """
    s = config.TARGET_PERCENT_SCALE
    cell.y_raw = cell.y_raw * s
    cell.y_conversion_raw = cell.y_conversion_raw * s
    cell.y_selectivity_raw = cell.y_selectivity_raw * s
    return cell


def _gem_spec_str(model_cfg: dict) -> str:
    """Human-readable spec string for one model config."""
    return (
        f"{config.CAT_FEATURIZATION}/{config.SOLVENT_FEATURIZATION}/"
        f"{config.BASE_FEATURIZATION}/{model_cfg['kernel_structure']}/"
        f"{model_cfg['model_class']}/{model_cfg['mean_function']}/"
        f"{model_cfg['outcome_transform']}"
    )


def _build_gem_model(cell, model_cfg: dict):
    """Build the (unfitted) model skeleton for one model config.

    The same call is used by :func:`prepare` (then fit) and :func:`_reconstruct_model`
    (then ``load_state_dict``), so the skeletons match exactly.
    """
    train_idx = list(range(cell.X.shape[0]))
    mean_module = build_mean_module(
        model_cfg["mean_function"], cell, train_x=cell.X[train_idx]
    )
    builder = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure=model_cfg["kernel_structure"],
        base_kernel_type=config.BASE_KERNEL,
        outcome_transform=model_cfg["outcome_transform"],
        lengthscale_scale=fks_config.LENGTHSCALE_LOGNORMAL_SCALE,
        mean_function=model_cfg["mean_function"],
    )
    return builder.build(model_cfg["model_class"], mean_module)


class _GemAcqfAdapter(AcquisitionFunction):
    """Remap optimizer-layout candidates into the gem model's feature layout.

    The mixed-variable optimizers emit candidates as
    ``[cont(d_cont) | cat_aug | sol | base]`` where ``cat_aug`` is the grover
    catalyst block followed by two trailing scalars ``[form_id, is_salt]``
    (carried verbatim through the GA's component lookup; the base slice is zeroed
    by the optimizer for non-salt catalysts). The wrapped acqf consumes the
    featscan layout ``[cat | solv | base | base_active | cont | form_id]``.
    ``base_active`` is set to ``is_salt``. The remap is index-slice + concatenate,
    so gradients still flow to the continuous dims during the L-BFGS refine.
    """

    def __init__(
        self,
        inner: AcquisitionFunction,
        d_cat: int,
        d_solv: int,
        d_base: int,
        d_continuous: int,
    ) -> None:
        super().__init__(model=inner.model)
        self.inner = inner
        self.d_cat = int(d_cat)
        self.d_solv = int(d_solv)
        self.d_base = int(d_base)
        self.d_continuous = int(d_continuous)

    def _to_model_layout(self, X: Tensor) -> Tensor:
        """Map ``(..., d_opt)`` optimizer-layout X to ``(..., d_model)``."""
        c = self.d_continuous
        cat_aug_w = self.d_cat + 2
        cont = X[..., :c]
        cat_aug = X[..., c : c + cat_aug_w]
        sol = X[..., c + cat_aug_w : c + cat_aug_w + self.d_solv]
        base = X[..., c + cat_aug_w + self.d_solv : c + cat_aug_w + self.d_solv + self.d_base]
        cat = cat_aug[..., : self.d_cat]
        form_id = cat_aug[..., self.d_cat : self.d_cat + 1]
        base_active = cat_aug[..., self.d_cat + 1 : self.d_cat + 2]  # is_salt
        return torch.cat([cat, sol, base, base_active, cont, form_id], dim=-1)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the wrapped acquisition function on optimizer-layout candidates.

        Args:
            X: A ``batch_shape x q x d_opt`` tensor of candidates in the optimizer
                layout ``[cont | cat | form_id | is_salt | sol | base]``.

        Returns:
            A ``batch_shape`` tensor of acquisition values.
        """
        return self.inner(self._to_model_layout(X))


def _build_gem_acqf(
    model, X_baseline: Tensor, d_cat: int, d_solv: int, d_base: int, model_cfg: dict
) -> _GemAcqfAdapter:
    """Wrap qLogNEI on the yield objective in the layout adapter.

    The objective depends on the model's ``yield_src``: ``"direct"`` (the ``st_y``
    SingleTaskGP outputs yield) uses the identity objective; ``"marginal"`` (a multitask
    model where yield is a task) slices ``samples[..., yield_index]``; ``"product_cs"``
    (a (C, S) Kronecker model) maps the two-task samples to yield = C·S.
    """
    spec = AcyloinCondensationModel.get_spec(model_cfg["model_class"])
    yield_src = spec.yield_src
    if yield_src == "product_cs":
        objective = GenericMCObjective(lambda samples, X=None: samples[..., 0] * samples[..., 1])
    elif yield_src == "direct":
        objective = None  # IdentityMCObjective on the single yield output
    elif yield_src == "marginal":
        yi = int(spec.yield_index)
        objective = GenericMCObjective(lambda samples, X=None, _yi=yi: samples[..., _yi])
    else:
        raise ValueError(f"Unsupported yield_src {yield_src!r} for the AF objective.")
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([config.N_MC_SAMPLES]))
    inner = qLogNoisyExpectedImprovement(
        model=model, X_baseline=X_baseline, sampler=sampler, objective=objective
    )
    return _GemAcqfAdapter(inner, d_cat, d_solv, d_base, D_CONTINUOUS)


def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, and Torch global RNGs with ``seed``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare() -> None:
    """Fit each model in :data:`config.MODELS` once and serialize a shared bundle.

    Assembles the featscan features (grover / map / map) once, rescales targets to
    percent, fits every model on the same inputs (per-spec targets), and writes a single
    CPU bundle to :data:`BUNDLE_PATH`: one state dict per model plus the shared AF search
    space (catalyst augmented with ``form_id`` + ``is_salt``), salt flags, full bounds,
    the temperature batch constraint, the fixed time dim, and feature-block widths.
    """
    cell = assemble_features(
        config.CAT_FEATURIZATION,
        config.SOLVENT_FEATURIZATION,
        config.BASE_FEATURIZATION,
        EXPERIMENTS_PATH,
        CANONICAL_FEATURIZATIONS_ROOT,
        libraries_root=FKS_LIB_ROOT,
    )
    cell = _scale_cell_targets(cell)
    train_X = cell.X.to(device=GEM_DEVICE, dtype=DTYPE)

    models_blob: dict[str, dict] = {}
    for model_cfg in config.MODELS:
        model = _build_gem_model(cell, model_cfg)
        spec = AcyloinCondensationModel.get_spec(model_cfg["model_class"])
        train_y = _build_train_targets(
            cell, list(range(cell.X.shape[0])), spec, model_cfg["outcome_transform"]
        ).to(device=GEM_DEVICE, dtype=DTYPE)
        with gpytorch.settings.cholesky_jitter(double_value=fks_config.CHOLESKY_JITTER_F64):
            _FIT_DISPATCH[spec.architecture](
                model, train_X, train_y, fks_config.N_RESTARTS, fks_config.MAX_ITER
            )
        models_blob[model_cfg["name"]] = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "gem_spec": _gem_spec_str(model_cfg),
        }

    cat_emb_avail, sol_emb_avail, bas_emb_avail, cat_is_salt_flags = _build_search_space()

    d_cat = len(cell.feature_dims["cat"])
    d_solv = len(cell.feature_dims["solv"])
    d_base = len(cell.feature_dims["base"])
    if cat_emb_avail.shape[1] != d_cat + 2:
        raise ValueError(f"Catalyst table width {cat_emb_avail.shape[1]} != d_cat+2 ({d_cat + 2}).")
    if sol_emb_avail.shape[1] != d_solv or bas_emb_avail.shape[1] != d_base:
        raise ValueError("Candidate solvent/base feature widths disagree with training.")

    d_total = D_CONTINUOUS + (d_cat + 2) + d_solv + d_base
    bounds = torch.zeros(2, d_total, dtype=DTYPE)
    bounds[1, :] = 1.0
    bounds[0, :D_CONTINUOUS] = -1.0  # continuous block standardized to [-1, 1]
    bounds[1, :D_CONTINUOUS] = 1.0

    # Gem continuous order is [fa_ratio, loading, temperature, time]: temperature is
    # dim 2 (partitioned), time is dim 3 (fixed at its max, +1.0 on the [-1,1] scale).
    batch_constraints = {2: tuple(config.TEMPERATURE_PARTITION)}
    fixed_dims = dict(config.FIXED_DIMS)

    bundle = {
        "models": models_blob,
        "bounds": bounds.cpu(),
        "cat_emb_avail": cat_emb_avail.cpu(),
        "sol_emb_avail": sol_emb_avail.cpu(),
        "bas_emb_avail": bas_emb_avail.cpu(),
        "cat_is_salt_flags": cat_is_salt_flags.cpu(),
        "batch_constraints": batch_constraints,
        "fixed_dims": fixed_dims,
        "d_continuous": D_CONTINUOUS,
        "d_cat": d_cat,
        "d_solv": d_solv,
        "d_base": d_base,
    }

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, BUNDLE_PATH)

    print(f"[prepare] gem bundle saved to {BUNDLE_PATH}")
    for name, blob in models_blob.items():
        print(f"  model {name:>8} : {blob['gem_spec']}")
    print(f"  train rows    : {cell.X.shape[0]} (percent-scaled targets)")
    print(f"  feature dims  : cat={d_cat} solv={d_solv} base={d_base} cont={D_CONTINUOUS}")
    print(
        "  available     : "
        f"cat={cat_emb_avail.shape[0]} sol={sol_emb_avail.shape[0]} base={bas_emb_avail.shape[0]}"
    )
    print(f"  d_total (opt) : {d_total}")
    print(f"  batch_constr. : {batch_constraints}  fixed_dims: {fixed_dims}")


def _load_bundle() -> dict:
    """Load the prepared bundle onto CPU."""
    return torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)


def _reconstruct_model(bundle: dict, model_cfg: dict) -> tuple[object, Tensor]:
    """Rebuild and load one fitted model from a bundle (no refitting).

    Re-assembles the (percent-scaled) training ``CellFeatures`` deterministically,
    rebuilds the skeleton via :func:`_build_gem_model`, loads the model's serialized
    state dict, and returns it in eval mode on :data:`GEM_DEVICE` together with its
    ``(n, d_model)`` training inputs (the qLogNEI ``X_baseline``).
    """
    cell = assemble_features(
        config.CAT_FEATURIZATION,
        config.SOLVENT_FEATURIZATION,
        config.BASE_FEATURIZATION,
        EXPERIMENTS_PATH,
        CANONICAL_FEATURIZATIONS_ROOT,
        libraries_root=FKS_LIB_ROOT,
    )
    cell = _scale_cell_targets(cell)
    model = _build_gem_model(cell, model_cfg)
    # keep_transforms=False: load the fitted transform buffers verbatim and skip
    # botorch's train-target re-untransform. Mirrors fit_eval._fit_multistart's load.
    model.load_state_dict(bundle["models"][model_cfg["name"]]["state_dict"], keep_transforms=False)
    model.to(GEM_DEVICE)
    model.eval()
    return model, cell.X.to(device=GEM_DEVICE, dtype=DTYPE)


def _run_scheme(
    scheme_steps: list[int],
    acq_function: qLogNoisyExpectedImprovement,
    bounds: Tensor,
    cat_emb_avail: Tensor,
    sol_emb_avail: Tensor,
    bas_emb_avail: Tensor,
    cat_is_salt_flags: Tensor,
    batch_constraints: dict,
    fixed_dims: dict[int, float],
    d_continuous: int,
    setting: dict[str, int],
    seed: int,
    heartbeat: Callable[..., None] | None = None,
) -> dict:
    """Build the q=6 batch in sub-batches of sizes ``scheme_steps``.

    At sub-batch ``i`` (cumulative count ``k = sum(scheme_steps[:i+1])``),
    ``optimize_acqf_mixed_GA`` frees ``scheme_steps[i]`` new candidates with the
    already-chosen ones passed as ``pre_batch`` (no batch constraint on the GA), then
    ``optimize_acqf_continuous`` re-optimizes the continuous dims of all ``k`` candidates
    under the partial temperature partition for ``k``. ``scheme_steps=[1]*6`` reproduces
    the original sequential method; ``[6]`` is a single full-batch GA call. The full
    (2, 2, 2) partition is imposed at the final sub-batch (k=6).

    Args:
        scheme_steps: Sub-batch sizes, summing to q=6 (e.g. ``[2, 2, 2]``).
        acq_function: The acquisition function on optimizer-layout candidates (the
            :class:`_GemAcqfAdapter` from :func:`_build_gem_acqf`).
        bounds: A ``2 x d_total`` tensor of lower/upper bounds, with
            ``d_total = d_continuous + (d_cat + 2) + d_solv + d_base``.
        cat_emb_avail: A ``n_cat x (d_cat + 2)`` tensor of available-catalyst features
            with trailing ``[form_id, is_salt]`` columns.
        sol_emb_avail: A ``n_sol x d_solv`` tensor of available-solvent features.
        bas_emb_avail: A ``n_base x d_base`` tensor of available-base features.
        cat_is_salt_flags: A ``n_cat`` bool tensor, True where the catalyst needs a base.
        batch_constraints: Dimension index to temperature partition, e.g. ``{2: (2, 2, 2)}``.
        fixed_dims: Continuous dimension index to its pinned value.
        d_continuous: Number of continuous dimensions at the front of a candidate.
        setting: Tier settings with keys ``ga_n_evals``, ``ga_pop_size``,
            ``lbfgs_num_restarts`` and ``lbfgs_raw_samples``.
        seed: Base seed; sub-batch ``i`` uses ``seed + i``.
        heartbeat: Optional callback receiving run-state fields at each stage.

    Returns:
        A dict with ``steps`` (per-sub-batch ``post_ga`` / ``post_refinement`` records,
        each holding a ``k x d_total`` candidate list), the final ``candidates``
        (``6 x d_total`` list), ``final_acq_value`` and ``total_wall_time_s``.
    """
    t0 = time.perf_counter()
    _grouped_counts, explicit_partitions = _normalize_batch_constraints(batch_constraints)

    n_sub = len(scheme_steps)
    current_batch: Tensor | None = None
    cumulative = 0
    steps: list[dict] = []
    final_acq_value = float("-inf")

    for i, increment in enumerate(scheme_steps):
        cumulative += increment
        k = cumulative
        step_seed = seed + i

        t_step = time.perf_counter()
        if heartbeat is not None:
            heartbeat(stage="ga", sequential_step=i + 1, sequential_step_total=n_sub)
        ga_cands, ga_acq = optimize_acqf_mixed_GA(
            acq_function=acq_function,
            bounds=bounds,
            q=k,
            n_evals_ga=setting["ga_n_evals"],
            pop_size=setting["ga_pop_size"],
            cat_emb_avail=cat_emb_avail,
            sol_emb_avail=sol_emb_avail,
            bas_emb_avail=bas_emb_avail,
            cat_is_salt_flags=cat_is_salt_flags,
            fixed_dims=fixed_dims,
            pre_batch=current_batch,
            seed=step_seed,
        )
        t_ga = time.perf_counter()

        step_constraints = _build_step_batch_constraints(
            batch_constraints, explicit_partitions, k
        )
        if heartbeat is not None:
            heartbeat(stage="cont_refine", sequential_step=i + 1, sequential_step_total=n_sub)
        refined, acq_val = optimize_acqf_continuous(
            acq_function=acq_function,
            bounds=bounds,
            X_discrete=ga_cands[:, d_continuous:],
            pre_batch=None,
            batch_constraints=step_constraints,
            fixed_dims=fixed_dims,
            num_restarts=setting["lbfgs_num_restarts"],
            raw_samples=setting["lbfgs_raw_samples"],
            seed=step_seed,
        )
        t_ref = time.perf_counter()

        current_batch = refined
        final_acq_value = float(acq_val.item())

        steps.append(
            {
                "sub_batch": i,
                "increment": increment,
                "k": k,
                "post_ga": {
                    "candidates": ga_cands.detach().cpu().tolist(),
                    "acq_value": float(ga_acq.item()),
                    "wall_time_s": t_ga - t_step,
                },
                "post_refinement": {
                    "candidates": refined.detach().cpu().tolist(),
                    "acq_value": float(acq_val.item()),
                    "wall_time_s": t_ref - t_ga,
                },
            }
        )

    return {
        "steps": steps,
        "candidates": current_batch.detach().cpu().tolist(),
        "final_acq_value": final_acq_value,
        "total_wall_time_s": time.perf_counter() - t0,
    }


def _meta() -> dict:
    """Host/runtime metadata for the result file."""
    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(GEM_DEVICE),
    }


def _config_snapshot(
    model_idx: int, scheme_idx: int, tier_idx: int, rep_idx: int, setting: dict[str, int]
) -> dict:
    """Snapshot the work-unit coordinates and optimizer settings into the result."""
    model_cfg = config.MODELS[model_idx]
    scheme = config.BATCH_SCHEMES[scheme_idx]
    return {
        "model": model_cfg["name"],
        "model_class": model_cfg["model_class"],
        "kernel_structure": model_cfg["kernel_structure"],
        "mean_function": model_cfg["mean_function"],
        "outcome_transform": model_cfg["outcome_transform"],
        "scheme": scheme["name"],
        "scheme_steps": list(scheme["steps"]),
        "tier_idx": tier_idx,
        "rep_idx": rep_idx,
        "q_batch": config.Q_BATCH,
        "temperature_partition": list(config.TEMPERATURE_PARTITION),
        "n_mc_samples": config.N_MC_SAMPLES,
        "target_percent_scale": config.TARGET_PERCENT_SCALE,
        "ga_n_evals": setting["ga_n_evals"],
        "ga_pop_size": setting["ga_pop_size"],
        "lbfgs_num_restarts": setting["lbfgs_num_restarts"],
        "lbfgs_raw_samples": setting["lbfgs_raw_samples"],
        "cat_featurization": config.CAT_FEATURIZATION,
        "solvent_featurization": config.SOLVENT_FEATURIZATION,
        "base_featurization": config.BASE_FEATURIZATION,
    }


def _decode(worker_id: int) -> tuple[int, int, int]:
    """Decode a flat worker index into ``(rep_idx, model_idx, scheme_idx)``.

    The queue is a flat range ``[0, N_WORKERS)`` laid out as ``rep -> model -> scheme``
    (``rep`` slowest-varying, ``scheme`` fastest); each worker runs all
    :data:`config.N_TIERS` tiers for the cell ``(rep, MODELS[model_idx], BATCH_SCHEMES[scheme_idx])``.
    """
    cells_per_rep = len(config.MODELS) * len(config.BATCH_SCHEMES)
    rep_idx, rem = divmod(worker_id, cells_per_rep)
    model_idx, scheme_idx = divmod(rem, len(config.BATCH_SCHEMES))
    return rep_idx, model_idx, scheme_idx


def _opt_seed(rep_idx: int, model_idx: int, scheme_idx: int, tier_idx: int) -> int:
    """Unique, well-spaced seed for one optimization in the grid.

    The 1000-stride per ``(rep, model, scheme, tier)`` cell leaves room for the
    sub-batch ``seed + i`` bumps inside :func:`_run_scheme` without colliding.
    """
    flat = (
        (rep_idx * len(config.MODELS) + model_idx) * len(config.BATCH_SCHEMES) + scheme_idx
    ) * config.N_TIERS + tier_idx
    return flat * 1000


def _load_opt_context(model_idx: int) -> dict:
    """Load the bundle and reconstruct everything one model's optimizations need.

    Returns the gem model + qLogNEI baseline, the model config, the optimizer embedding
    tables and salt flags, the batch constraint / fixed dims, and the gem feature-block
    widths for the acqf adapter. Shared across a worker's six tiers.
    """
    bundle = _load_bundle()
    model_cfg = config.MODELS[model_idx]
    model, x_baseline = _reconstruct_model(bundle, model_cfg)
    return {
        "model": model,
        "x_baseline": x_baseline,
        "model_cfg": model_cfg,
        "model_idx": model_idx,
        "bounds": bundle["bounds"].to(device=GEM_DEVICE, dtype=DTYPE),
        "cat_emb_avail": bundle["cat_emb_avail"].to(device=GEM_DEVICE, dtype=DTYPE),
        "sol_emb_avail": bundle["sol_emb_avail"].to(device=GEM_DEVICE, dtype=DTYPE),
        "bas_emb_avail": bundle["bas_emb_avail"].to(device=GEM_DEVICE, dtype=DTYPE),
        "cat_is_salt_flags": bundle["cat_is_salt_flags"].to(GEM_DEVICE),
        "batch_constraints": bundle["batch_constraints"],
        "fixed_dims": bundle["fixed_dims"],
        "d_continuous": int(bundle["d_continuous"]),
        "d_cat": int(bundle["d_cat"]),
        "d_solv": int(bundle["d_solv"]),
        "d_base": int(bundle["d_base"]),
    }


def _run_one_opt(
    ctx: dict,
    scheme_idx: int,
    tier_idx: int,
    rep_idx: int,
    heartbeat: Callable[..., None] | None = None,
) -> dict:
    """Run one optimization (one scheme at one tier/rep) on the gem acqf.

    Seeds from the cell coordinate, builds the qLogNEI (yield objective per the model's
    ``yield_src``) wrapped in the layout adapter, runs :func:`_run_scheme`, and returns
    a per-tier result record.
    """
    model_idx = ctx["model_idx"]
    scheme = config.BATCH_SCHEMES[scheme_idx]
    setting = config.TIER_SETTINGS_BY_SCHEME[scheme["name"]][tier_idx]
    seed = _opt_seed(rep_idx, model_idx, scheme_idx, tier_idx)

    _seed_all(seed)
    acq_function = _build_gem_acqf(
        ctx["model"], ctx["x_baseline"], ctx["d_cat"], ctx["d_solv"], ctx["d_base"],
        ctx["model_cfg"],
    )
    method_result = _run_scheme(
        scheme["steps"],
        acq_function,
        ctx["bounds"],
        ctx["cat_emb_avail"],
        ctx["sol_emb_avail"],
        ctx["bas_emb_avail"],
        ctx["cat_is_salt_flags"],
        ctx["batch_constraints"],
        ctx["fixed_dims"],
        ctx["d_continuous"],
        setting,
        seed=seed,
        heartbeat=heartbeat,
    )
    return {
        "tier_idx": tier_idx,
        "model": ctx["model_cfg"]["name"],
        "model_idx": model_idx,
        "scheme": scheme["name"],
        "scheme_steps": list(scheme["steps"]),
        "rep_idx": rep_idx,
        "seed": seed,
        "config_snapshot": _config_snapshot(model_idx, scheme_idx, tier_idx, rep_idx, setting),
        "result": method_result,
    }


def run_worker(worker_id: int, state_path: Path | None = None) -> Path:
    """Run one per-cell worker — all tiers for ``(rep, model, scheme)`` — and write its result.

    Decodes ``worker_id`` via :func:`_decode`, reconstructs that model once, loops over
    :data:`config.N_TIERS` tiers, and serializes the six per-tier records to
    ``data/results/result_<worker_id>.json``.

    Args:
        worker_id: Flat worker index in ``[0, N_WORKERS)``.
        state_path: Run-state JSON file to update with progress; ``None`` skips
            progress reporting (the ``worker`` CLI subcommand passes ``None``).

    Returns:
        Path to the written result JSON file.
    """
    rep_idx, model_idx, scheme_idx = _decode(worker_id)
    model_name = config.MODELS[model_idx]["name"]
    scheme = config.BATCH_SCHEMES[scheme_idx]
    opt_total = config.N_TIERS
    t0 = time.perf_counter()

    _update: Callable[..., None] | None = None
    if state_path is not None:
        from pablo.mechthild_runs.run_4_af_opt_strategy.run_state import (
            update_run_state,
        )
        update_run_state(
            state_path, worker_id, opt_total,
            status="running",
            rep_idx=rep_idx,
            model_name=model_name,
            current_scheme=scheme["name"],
            current_tier=0,
            opt_done=0,
            started_at_epoch=time.time(),
            wall_time_so_far_s=0.0,
            hostname=socket.gethostname(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )

        def _update(**kw: object) -> None:
            update_run_state(
                state_path, worker_id, opt_total,
                wall_time_so_far_s=time.perf_counter() - t0, **kw,
            )

    ctx = _load_opt_context(model_idx)

    tier_records: list[dict] = []
    for tier_idx in range(config.N_TIERS):
        if _update is not None:
            _update(
                opt_done=tier_idx,
                model_name=model_name,
                current_scheme=scheme["name"],
                current_tier=tier_idx,
                stage=None,
                sequential_step=None,
                sequential_step_total=None,
            )
        entry = _run_one_opt(ctx, scheme_idx, tier_idx, rep_idx, heartbeat=_update)
        tier_records.append(entry)

    result = {
        "worker_id": worker_id,
        "rep_idx": rep_idx,
        "model_idx": model_idx,
        "model": model_name,
        "scheme_idx": scheme_idx,
        "scheme": scheme["name"],
        "meta": _meta(),
        "results": tier_records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"result_{worker_id}.json"
    out_path.write_text(json.dumps(result, indent=2))
    acqs = [f"{r['result']['final_acq_value']:.3f}" for r in tier_records]
    wall = sum(r["result"]["total_wall_time_s"] for r in tier_records)
    print(
        f"[worker {worker_id}] rep={rep_idx} {model_name} {scheme['name']} "
        f"tier_acqs=[{', '.join(acqs)}] wall={wall:.1f}s -> {out_path}"
    )

    if _update is not None:
        _update(
            status="done",
            opt_done=opt_total,
            current_tier=None,
            current_scheme=None,
            stage=None,
            sequential_step=None,
            sequential_step_total=None,
        )

    return out_path


def run_cell(model_name: str, scheme_name: str, tier_idx: int, rep_idx: int) -> Path:
    """Run a single (model, scheme, tier, rep) optimization for the local test-run gate.

    Uses the fitted gem model from the bundle and the tier's production GA / L-BFGS
    settings. Writes ``data/results/cell_<model>_<scheme>_t<tier>_r<rep>.json`` for
    manual inspection.

    Args:
        model_name: Name of a model in :data:`config.MODELS`.
        scheme_name: Name of a scheme in :data:`config.BATCH_SCHEMES`.
        tier_idx: GA-budget tier index in ``[0, config.N_TIERS)``.
        rep_idx: Repetition index; sets the seed together with the other three.

    Returns:
        Path to the written result JSON file.
    """
    model_idx = next(i for i, m in enumerate(config.MODELS) if m["name"] == model_name)
    scheme_idx = next(i for i, s in enumerate(config.BATCH_SCHEMES) if s["name"] == scheme_name)
    ctx = _load_opt_context(model_idx)
    entry = _run_one_opt(ctx, scheme_idx, tier_idx, rep_idx)
    result = {"single_cell": True, "meta": _meta(), "result": entry}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"cell_{model_name}_{scheme_name}_t{tier_idx}_r{rep_idx}.json"
    out_path.write_text(json.dumps(result, indent=2))
    block = entry["result"]
    print(
        f"[cell] {model_name} {scheme_name} tier={tier_idx} rep={rep_idx} "
        f"final_acq={block['final_acq_value']:.4f} wall={block['total_wall_time_s']:.1f}s -> {out_path}"
    )
    return out_path


def smoke(ga_n_evals: int = 1000, ga_pop_size: int = 100, model_name: str = "st_y") -> None:
    """Quick AF-scale sanity check: one full-batch run with tiny settings, prints acq.

    Verifies the acqf produces positive values (qLogNEI on percent-scale yield; the
    validated ``st_y`` baseline lands around 0.4–0.6 with a full GA budget). Does not
    write a result file.

    Args:
        ga_n_evals: GA evaluation budget for the single full-batch call.
        ga_pop_size: GA population size.
        model_name: Name of a model in :data:`config.MODELS`.
    """
    model_idx = next(i for i, m in enumerate(config.MODELS) if m["name"] == model_name)
    ctx = _load_opt_context(model_idx)
    _seed_all(0)
    acqf = _build_gem_acqf(
        ctx["model"], ctx["x_baseline"], ctx["d_cat"], ctx["d_solv"], ctx["d_base"],
        ctx["model_cfg"],
    )
    setting = {
        "ga_n_evals": ga_n_evals,
        "ga_pop_size": ga_pop_size,
        "lbfgs_num_restarts": 1,
        "lbfgs_raw_samples": 16,
    }
    t0 = time.perf_counter()
    result = _run_scheme(
        [config.Q_BATCH],
        acqf,
        ctx["bounds"],
        ctx["cat_emb_avail"],
        ctx["sol_emb_avail"],
        ctx["bas_emb_avail"],
        ctx["cat_is_salt_flags"],
        ctx["batch_constraints"],
        ctx["fixed_dims"],
        ctx["d_continuous"],
        setting,
        seed=0,
    )
    elapsed = time.perf_counter() - t0
    pre = result["steps"][0]["post_ga"]["acq_value"]
    post = result["steps"][0]["post_refinement"]["acq_value"]
    print(f"[smoke] model={model_name}  ga_n_evals={ga_n_evals}  pop={ga_pop_size}  wall={elapsed:.1f}s")
    print(f"[smoke] pre-refine acq  = {pre:.6f}")
    print(f"[smoke] post-refine acq = {post:.6f}")
    print("[smoke] expected: positive, ~0.4-0.6 for st_y on this dataset at a full GA budget")


def calibrate(probe_evals: int = 2000) -> None:
    """Time GA-only cost per (model, scheme) and recommend ``ga_n_evals`` per tier.

    For each model x scheme, runs one cheap :func:`_run_scheme` probe (at the scheme's
    tier-0 population and the production L-BFGS settings), measures the total GA-only and
    refine wall times, and prints the per-call ``ga_n_evals`` that fills each
    :data:`config.TIER_TARGET_HOURS` target. Total GA time scales linearly with
    ``ga_n_evals``; the refine cost is roughly budget-independent and subtracted as a
    fixed allowance. Also prints the summed six-tier worker wall time so the SLURM time
    limit can be set. Estimates are approximate — validate against one real worker and
    adjust once. Run on the same node type / cpus-per-task as the sweep.

    Args:
        probe_evals: GA evaluation budget of each probe :func:`_run_scheme` call.
    """
    refine = dict(
        lbfgs_num_restarts=config.LBFGS_NUM_RESTARTS,
        lbfgs_raw_samples=config.LBFGS_RAW_SAMPLES,
    )
    print(f"[calibrate] N_MC_SAMPLES={config.N_MC_SAMPLES}  probe_evals={probe_evals}")
    print(f"[calibrate] TIER_TARGET_HOURS={config.TIER_TARGET_HOURS}")

    for model_idx, model_cfg in enumerate(config.MODELS):
        ctx = _load_opt_context(model_idx)
        acqf = _build_gem_acqf(
            ctx["model"], ctx["x_baseline"], ctx["d_cat"], ctx["d_solv"], ctx["d_base"],
            ctx["model_cfg"],
        )
        common = dict(
            acq_function=acqf,
            bounds=ctx["bounds"],
            cat_emb_avail=ctx["cat_emb_avail"],
            sol_emb_avail=ctx["sol_emb_avail"],
            bas_emb_avail=ctx["bas_emb_avail"],
            cat_is_salt_flags=ctx["cat_is_salt_flags"],
            batch_constraints=ctx["batch_constraints"],
            fixed_dims=ctx["fixed_dims"],
            d_continuous=ctx["d_continuous"],
            seed=0,
        )
        print(f"\n=== model {model_cfg['name']} ===")
        for scheme in config.BATCH_SCHEMES:
            pop = config.TIER_SETTINGS_BY_SCHEME[scheme["name"]][0]["ga_pop_size"]
            _seed_all(0)
            r = _run_scheme(
                scheme["steps"],
                setting={"ga_n_evals": probe_evals, "ga_pop_size": pop, **refine},
                **common,
            )
            ga_time = sum(s["post_ga"]["wall_time_s"] for s in r["steps"])
            ref_time = sum(s["post_refinement"]["wall_time_s"] for s in r["steps"])
            per_call_cost = ga_time / probe_evals  # GA wall per per-call eval, this scheme

            print(
                f"  {scheme['name']:>12} (pop {pop}, steps {scheme['steps']}): "
                f"GA {ga_time:.1f}s/{probe_evals} evals, refine {ref_time:.1f}s"
            )
            worker_wall = 0.0
            recs = []
            for hours in config.TIER_TARGET_HOURS:
                budget_s = hours * 3600.0
                ga_n_evals = max(0, int((budget_s - ref_time) / per_call_cost)) if per_call_cost > 0 else 0
                recs.append(ga_n_evals)
                worker_wall += hours
            print(f"      ga_n_evals per tier: {recs}")
            print(f"      summed 6-tier worker wall ~ {worker_wall:.2f} h (+ {len(config.TIER_TARGET_HOURS)} refines already counted)")


def main() -> None:
    """CLI: ``prepare`` | ``smoke`` | ``calibrate`` | ``worker`` | ``cell``."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sp = sub.add_parser("smoke")
    sp.add_argument("--ga-n-evals", type=int, default=1000)
    sp.add_argument("--ga-pop-size", type=int, default=100)
    sp.add_argument("--model", default="st_y", choices=[m["name"] for m in config.MODELS])
    clp = sub.add_parser("calibrate")
    clp.add_argument("--probe-evals", type=int, default=2000)
    wp = sub.add_parser("worker")
    wp.add_argument("--worker-id", type=int, required=True)
    cp = sub.add_parser("cell")
    cp.add_argument("--model", required=True, choices=[m["name"] for m in config.MODELS])
    cp.add_argument("--scheme", required=True, choices=[s["name"] for s in config.BATCH_SCHEMES])
    cp.add_argument("--tier", type=int, required=True, choices=range(config.N_TIERS))
    cp.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "smoke":
        smoke(args.ga_n_evals, args.ga_pop_size, args.model)
    elif args.cmd == "calibrate":
        calibrate(args.probe_evals)
    elif args.cmd == "worker":
        run_worker(args.worker_id)
    elif args.cmd == "cell":
        run_cell(args.model, args.scheme, args.tier, args.rep)


if __name__ == "__main__":
    main()
