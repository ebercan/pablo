"""run_6: pool-greedy against the GA by acquisition value at matched walltime.

``prepare`` fits the featscan GP of :mod:`config` on every experiment and writes one
bundle (state dict, AF search space, incumbents) that all workers share. A worker
decodes its flat index into ``(optimizer, tier, rep)``, builds a fresh qLogNEI seeded by
the rep, runs one optimizer at the tier's settings, scores the batch with a common
qLogNEI on :data:`config.SCORE_QMC_SAMPLES` fixed QMC base samples, and writes
``data/results/result_<idx>.json`` as its last action.

Candidates are in the optimizer layout ``[cont(4) | cat | form_id | is_salt | sol |
base]``; ``_GemAcqfAdapter`` (from run_4) maps them to the featscan model layout
``[cat | sol | base | base_active | cont | form_id]``.

Usage (from the repo root):
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant prepare
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant smoke
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant calibrate
    python -m pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant worker --worker-id ID [--smoke] [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from collections.abc import Callable
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

import pablo.featurization_and_kernel_scan.core.config as fks_config
import pablo.featurization_and_kernel_scan.paths as fks_paths
from pablo.featurization_and_kernel_scan.core.kernels import build_partition
from pablo.featurization_and_kernel_scan.core.models import AcyloinCondensationModel, build_mean_module
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import _FIT_DISPATCH, _build_train_targets
from pablo.main import DEFAULT_EXCEL_FILE, DTYPE, load_available_agents, requires_external_base
from pablo.mechthild_runs.run_4_af_opt_strategy.main_variant import (
    _form_ids_for,
    _GemAcqfAdapter,
    _map_to_fks_ids,
    _seed_all,
)
from pablo.mechthild_runs.run_6_af_optimizer_af_value import config
from pablo.optimize import optimize_acqf_continuous, optimize_acqf_mixed_GA, optimize_acqf_mixed_pool_greedy
from pablo.utils.feature_assembly import (
    CellFeatures,
    _load_numeric_descriptors,
    _resolve_catalyst_parquet,
    _resolve_feature_parquet,
    assemble_features,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
BUNDLE_PATH = DATA_DIR / "prepared_bundle.pt"
RESULTS_DIR = DATA_DIR / "results"
CALIBRATION_PATH = DATA_DIR / "calibration.json"

DEVICE = torch.device("cpu")
SMOKE_GA = {"n_evals_ga": 400, "pop_size": 40, "lbfgs_num_restarts": 2, "lbfgs_raw_samples": 64}
SMOKE_PG_SCALE = 0.1


def _time_pinned_value() -> float:
    """``TIME_PINNED_H`` on the featscan [-1, 1] continuous scale."""
    meta = json.loads((fks_paths.FEAT_DIR / "continuous" / "metadata.json").read_text())
    if meta["raw_columns"][config.TIME_DIM] != "time" or meta["raw_columns"][config.TEMPERATURE_DIM] != "temperature":
        raise ValueError(f"unexpected continuous column order {meta['raw_columns']}")
    lo, hi = meta["bounds"]["time"]
    return 2.0 * (config.TIME_PINNED_H - lo) / (hi - lo) - 1.0


def _continuous_to_raw(cont: Tensor) -> list[dict[str, float]]:
    """Map ``(q, 4)`` [-1, 1] continuous rows to raw units keyed by column name."""
    meta = json.loads((fks_paths.FEAT_DIR / "continuous" / "metadata.json").read_text())
    rows = []
    for r in cont.tolist():
        row = {}
        for j, col in enumerate(meta["raw_columns"]):
            lo, hi = meta["bounds"][col]
            row[col] = lo + (r[j] + 1.0) * (hi - lo) / 2.0
        rows.append(row)
    return rows


def _build_cell() -> CellFeatures:
    """Assemble the experiments' features and the P2 lengthscale partition."""
    cell = assemble_features(
        cat_featurization=config.CAT_FEATURIZATION,
        solv_featurization=config.SOLVENT_FEATURIZATION,
        base_featurization=config.BASE_FEATURIZATION,
        experiments_path=fks_paths.EXPERIMENTS_PARQUET,
        featurizations_root=fks_paths.FEAT_DIR,
        libraries_root=fks_paths.LIB_DIR,
    )
    cell.partition = build_partition(
        cell.feature_dims, mode=fks_config.PRIOR_SCHEMES[config.PRIOR_SCHEME].partition_mode
    )
    return cell


def _build_model(cell: CellFeatures):
    """Build the unfitted ``st_y`` model of :mod:`config` on every experiment."""
    train_idx = list(range(cell.X.shape[0]))
    mean_module = build_mean_module(
        config.MEAN_FUNCTION, cell, prior_scale=fks_config.MEAN_MODULE_NORMAL_SCALE, train_x=cell.X
    )
    return AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure=config.KERNEL_STRUCTURE,
        base_kernel_type=config.BASE_KERNEL,
        outcome_transform=config.OUTCOME_TRANSFORM,
        lengthscale_scale=fks_config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme=config.PRIOR_SCHEME,
        mean_function=config.MEAN_FUNCTION,
    ).build(config.MODEL_CLASS, mean_module)


def _raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw catalyst, solvent and base feature tables, indexed by featscan library id."""
    cat = _load_numeric_descriptors(_resolve_catalyst_parquet(fks_paths.FEAT_DIR, config.CAT_FEATURIZATION))
    sol = _load_numeric_descriptors(_resolve_feature_parquet(fks_paths.FEAT_DIR / f"solvent_{config.SOLVENT_FEATURIZATION}"))
    bas = _load_numeric_descriptors(_resolve_feature_parquet(fks_paths.FEAT_DIR / f"base_{config.BASE_FEATURIZATION}"))
    return cat, sol, bas


def _rows_in_table(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``(n,)`` bool: whether each row of ``rows`` equals some row of ``table`` (atol 1e-10)."""
    return np.isclose(rows[:, None, :], table[None, :, :], atol=1e-10, rtol=0.0).all(-1).any(-1)


def _check_cell_uses_raw_tables(cell: CellFeatures) -> None:
    """Raise unless every training block row is a raw library row.

    The search-space tables are built from the raw featurization parquets, so the model
    must see the same unscaled rows.
    """
    cat, sol, bas = (t.to_numpy(dtype=np.float64) for t in _raw_tables())
    x = cell.X.cpu().numpy()
    active = x[:, cell.active_flag_dim] > 0.5
    checks = (
        ("cat", x[:, cell.feature_dims["cat"]], cat),
        ("solv", x[:, cell.feature_dims["solv"]], sol),
        ("base", x[active][:, cell.feature_dims["base"]], bas),
    )
    for name, rows, table in checks:
        missing = ~_rows_in_table(rows, table)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} training {name} rows are not raw library rows")


def _build_search_space() -> dict:
    """Feature tables of the available agents in the optimizer layout.

    Returns:
        A dict with ``cat_emb_avail`` ``(n_cat, d_cat + 2)`` (trailing ``[form_id,
        is_salt]``), ``sol_emb_avail`` ``(n_sol, d_sol)``, ``bas_emb_avail``
        ``(n_bas, d_base)``, ``cat_is_salt_flags`` ``(n_cat,)`` and the workbook ids
        and featscan ids of every available agent.
    """
    cat_ref = pd.read_excel(DEFAULT_EXCEL_FILE, sheet_name="Catalyst References Achiral").dropna(subset=["smiles"])
    sol_ref = pd.read_excel(DEFAULT_EXCEL_FILE, sheet_name="Solvent References").dropna(subset=["smiles"])
    bas_ref = pd.read_excel(DEFAULT_EXCEL_FILE, sheet_name="Base References").dropna(subset=["smiles"])
    cat_smiles = dict(zip(cat_ref["ID"].tolist(), cat_ref["smiles"].astype(str).tolist()))
    sol_smiles = dict(zip(sol_ref["ID"].tolist(), sol_ref["smiles"].astype(str).tolist()))
    bas_smiles = dict(zip(bas_ref["ID"].tolist(), bas_ref["smiles"].astype(str).tolist()))
    cats, sols, bases = load_available_agents(DEFAULT_EXCEL_FILE, config.AVAILABILITY_COLUMN)

    cat_fids = _map_to_fks_ids([cat_smiles[c] for c in cats], fks_paths.LIB_DIR / "catalysts.parquet", "catalyst")
    sol_fids = _map_to_fks_ids([sol_smiles[s] for s in sols], fks_paths.LIB_DIR / "solvents.parquet", "solvent")
    bas_fids = _map_to_fks_ids([bas_smiles[b] for b in bases], fks_paths.LIB_DIR / "bases.parquet", "base")

    cat_t, sol_t, bas_t = _raw_tables()
    cat_feats = cat_t.reindex(cat_fids).to_numpy(dtype=np.float64)
    sol_feats = sol_t.reindex(sol_fids).to_numpy(dtype=np.float64)
    bas_feats = bas_t.reindex(bas_fids).to_numpy(dtype=np.float64)
    for name, arr in (("catalyst", cat_feats), ("solvent", sol_feats), ("base", bas_feats)):
        if np.isnan(arr).any():
            raise ValueError(f"missing {name} features for some available agents")

    form_ids = _form_ids_for(cat_fids).reshape(-1, 1)
    is_salt = np.array([float(requires_external_base(cat_smiles[c])) for c in cats]).reshape(-1, 1)
    return {
        "cat_emb_avail": torch.tensor(np.concatenate([cat_feats, form_ids, is_salt], axis=1), dtype=DTYPE),
        "sol_emb_avail": torch.tensor(sol_feats, dtype=DTYPE),
        "bas_emb_avail": torch.tensor(bas_feats, dtype=DTYPE),
        "cat_is_salt_flags": torch.tensor(is_salt.reshape(-1) > 0.5),
        "cat_ids": [int(c) for c in cats],
        "sol_ids": [int(s) for s in sols],
        "bas_ids": [int(b) for b in bases],
        "cat_fids": cat_fids,
        "sol_fids": sol_fids,
        "bas_fids": bas_fids,
    }


def _to_optimizer_layout(cell: CellFeatures, x_model: Tensor, cat_emb_avail: Tensor) -> Tensor:
    """Map ``(n, d_model)`` featscan-layout rows to ``(n, d_opt)`` optimizer-layout rows.

    The trailing ``[form_id, is_salt]`` catalyst columns are taken from the matching
    available-catalyst row; a catalyst that is not available keeps its own form id and
    ``base_active`` there, and the optimizer treats it as unmatched.
    """
    d_cat = len(cell.feature_dims["cat"])
    cat = x_model[:, cell.feature_dims["cat"]]
    hits = torch.isclose(cat.unsqueeze(1), cat_emb_avail[:, :d_cat].unsqueeze(0), atol=1e-10, rtol=0.0).all(-1)
    trailing = torch.stack([x_model[:, cell.form_id_dim], x_model[:, cell.active_flag_dim]], dim=-1)
    matched = hits.any(-1)
    trailing[matched] = cat_emb_avail[hits[matched].float().argmax(-1), d_cat:]
    return torch.cat(
        [
            x_model[:, cell.feature_dims["cont"]],
            cat,
            trailing,
            x_model[:, cell.feature_dims["solv"]],
            x_model[:, cell.feature_dims["base"]],
        ],
        dim=-1,
    )


def prepare() -> None:
    """Fit the GP on every experiment and write the shared bundle to :data:`BUNDLE_PATH`."""
    cell = _build_cell()
    _check_cell_uses_raw_tables(cell)
    model = _build_model(cell)
    spec = AcyloinCondensationModel.get_spec(config.MODEL_CLASS)
    train_idx = list(range(cell.X.shape[0]))
    train_y = _build_train_targets(cell, train_idx, spec, config.OUTCOME_TRANSFORM)
    if not isinstance(train_y, Tensor):
        raise TypeError(f"{config.MODEL_CLASS} must have a single target tensor")
    with gpytorch.settings.cholesky_jitter(double_value=fks_config.CHOLESKY_JITTER_F64):
        _FIT_DISPATCH[spec.architecture](model, cell.X, train_y, fks_config.N_RESTARTS, fks_config.MAX_ITER)

    space = _build_search_space()
    d_cat = len(cell.feature_dims["cat"])
    d_solv = len(cell.feature_dims["solv"])
    d_base = len(cell.feature_dims["base"])
    if space["cat_emb_avail"].shape[1] != d_cat + 2:
        raise ValueError(f"catalyst table width {space['cat_emb_avail'].shape[1]} != d_cat + 2 = {d_cat + 2}")
    if space["sol_emb_avail"].shape[1] != d_solv or space["bas_emb_avail"].shape[1] != d_base:
        raise ValueError("available solvent/base widths differ from the training blocks")

    x_opt = _to_optimizer_layout(cell, cell.X, space["cat_emb_avail"])
    top = train_y.squeeze(-1).topk(config.PG_BASE["n_incumbents"]).indices
    lo = torch.cat(
        [
            -torch.ones(config.D_CONTINUOUS, dtype=DTYPE),
            space["cat_emb_avail"].min(0).values,
            space["sol_emb_avail"].min(0).values,
            torch.minimum(space["bas_emb_avail"].min(0).values, torch.zeros(d_base, dtype=DTYPE)),
        ]
    )
    hi = torch.cat(
        [
            torch.ones(config.D_CONTINUOUS, dtype=DTYPE),
            space["cat_emb_avail"].max(0).values,
            space["sol_emb_avail"].max(0).values,
            torch.maximum(space["bas_emb_avail"].max(0).values, torch.zeros(d_base, dtype=DTYPE)),
        ]
    )
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "bounds": torch.stack([lo, hi]),
        "batch_constraints": {config.TEMPERATURE_DIM: tuple(config.TEMPERATURE_PARTITION)},
        "fixed_dims": {config.TIME_DIM: _time_pinned_value()},
        "incumbents": x_opt[top],
        "d_cat": d_cat,
        "d_solv": d_solv,
        "d_base": d_base,
        **space,
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, BUNDLE_PATH)

    noise = model.likelihood.noise.item()
    print(f"[prepare] bundle -> {BUNDLE_PATH}")
    print(
        f"  model      : {config.CAT_FEATURIZATION}/{config.SOLVENT_FEATURIZATION}/{config.BASE_FEATURIZATION} "
        f"{config.BASE_KERNEL} {config.OUTCOME_TRANSFORM} {config.PRIOR_SCHEME} {config.KERNEL_STRUCTURE} "
        f"{config.MEAN_FUNCTION} {config.MODEL_CLASS}"
    )
    print(f"  train rows : {cell.X.shape[0]}, fitted noise (standardized) {noise:.4g}")
    print(f"  dims       : cat={d_cat} solv={d_solv} base={d_base} cont={config.D_CONTINUOUS}")
    print(
        f"  available  : cat={len(space['cat_ids'])} ({int(space['cat_is_salt_flags'].sum())} salts) "
        f"sol={len(space['sol_ids'])} base={len(space['bas_ids'])}"
    )
    print(f"  constraints: {bundle['batch_constraints']}  fixed: {bundle['fixed_dims']}")


def _load_context() -> dict:
    """Load the bundle, rebuild the fitted model in eval mode and return the AF context."""
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
    cell = _build_cell()
    model = _build_model(cell)
    model.load_state_dict(bundle["state_dict"], keep_transforms=False)
    model.to(DEVICE)
    model.eval()
    return {**bundle, "model": model, "x_baseline": cell.X.to(device=DEVICE, dtype=DTYPE)}


def _acqf(ctx: dict, n_mc: int, seed: int, prune_baseline: bool = True) -> _GemAcqfAdapter:
    """qLogNEI on the probit-scale output, wrapped for optimizer-layout candidates."""
    inner = qLogNoisyExpectedImprovement(
        model=ctx["model"],
        X_baseline=ctx["x_baseline"],
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([n_mc]), seed=seed),
        prune_baseline=prune_baseline,
    )
    return _GemAcqfAdapter(inner, ctx["d_cat"], ctx["d_solv"], ctx["d_base"], config.D_CONTINUOUS)


def _shared_kwargs(ctx: dict) -> dict:
    """Search-space keyword arguments common to both optimizers."""
    return {
        "bounds": ctx["bounds"],
        "q": config.Q_BATCH,
        "cat_emb_avail": ctx["cat_emb_avail"],
        "sol_emb_avail": ctx["sol_emb_avail"],
        "bas_emb_avail": ctx["bas_emb_avail"],
        "cat_is_salt_flags": ctx["cat_is_salt_flags"],
        "batch_constraints": ctx["batch_constraints"],
        "fixed_dims": ctx["fixed_dims"],
    }


def run_ga(ctx: dict, setting: dict[str, int], seed: int, heartbeat: Callable[..., None] | None = None) -> dict:
    """GA then the production continuous refine on a fresh qLogNEI.

    Returns:
        ``candidates`` ``(q, d_opt)``, ``own_acq_value`` (after refine),
        ``ga_acq_value`` (before refine), ``wall_time_s``, ``ga_wall_time_s`` and
        ``refine_wall_time_s``.
    """
    _seed_all(seed)
    acqf = _acqf(ctx, config.N_MC_SAMPLES, seed)
    t0 = time.perf_counter()
    if heartbeat is not None:
        heartbeat(stage="ga")
    ga_cands, ga_val = optimize_acqf_mixed_GA(
        acq_function=acqf, n_evals_ga=setting["n_evals_ga"], pop_size=setting["pop_size"], seed=seed,
        **_shared_kwargs(ctx),
    )
    t1 = time.perf_counter()
    if heartbeat is not None:
        heartbeat(stage="refine")
    cands, val = optimize_acqf_continuous(
        acq_function=acqf,
        bounds=ctx["bounds"],
        X_discrete=ga_cands[:, config.D_CONTINUOUS:],
        batch_constraints=ctx["batch_constraints"],
        fixed_dims=ctx["fixed_dims"],
        num_restarts=setting["lbfgs_num_restarts"],
        raw_samples=setting["lbfgs_raw_samples"],
        seed=seed,
    )
    t2 = time.perf_counter()
    return {
        "candidates": cands,
        "own_acq_value": float(val.item()),
        "ga_acq_value": float(ga_val.item()),
        "wall_time_s": t2 - t0,
        "ga_wall_time_s": t1 - t0,
        "refine_wall_time_s": t2 - t1,
    }


def run_pool_greedy(ctx: dict, setting: dict[str, int], seed: int, heartbeat: Callable[..., None] | None = None) -> dict:
    """Pool-greedy on a fresh qLogNEI, local rows drawn around the bundle's incumbents.

    Returns:
        ``candidates`` ``(q, d_opt)``, ``own_acq_value`` and ``wall_time_s``.
    """
    _seed_all(seed)
    acqf = _acqf(ctx, config.N_MC_SAMPLES, seed)
    if heartbeat is not None:
        heartbeat(stage="pool_greedy")
    t0 = time.perf_counter()
    cands, val = optimize_acqf_mixed_pool_greedy(
        acq_function=acqf, incumbents=ctx["incumbents"], n_mc=config.N_MC_SAMPLES, seed=seed,
        **setting, **_shared_kwargs(ctx),
    )
    return {"candidates": cands, "own_acq_value": float(val.item()), "wall_time_s": time.perf_counter() - t0}


def score(ctx: dict, candidates: Tensor) -> float:
    """Common-scorer qLogNEI of a ``(q, d_opt)`` batch.

    The scorer uses :data:`config.SCORE_QMC_SAMPLES` QMC base samples with a fixed
    seed and the full (unpruned) baseline, so its value depends on the batch only.
    """
    scorer = _acqf(ctx, config.SCORE_QMC_SAMPLES, config.SCORE_QMC_SEED, prune_baseline=False)
    with torch.no_grad():
        return float(scorer(candidates.unsqueeze(0)).item())


def decode_batch(ctx: dict, candidates: Tensor) -> list[dict]:
    """Workbook ids and raw continuous values of each row of a ``(q, d_opt)`` batch.

    Raises:
        ValueError: If a row's catalyst, solvent or (for a salt) base slice is not a row
            of the available tables.
    """
    c = config.D_CONTINUOUS
    d_cat_aug = ctx["cat_emb_avail"].shape[1]
    d_solv = ctx["d_solv"]
    out = []
    raw = _continuous_to_raw(candidates[:, :c])
    for i, row in enumerate(candidates):
        cat_hits = torch.isclose(row[c:c + d_cat_aug], ctx["cat_emb_avail"], atol=1e-10, rtol=0.0).all(-1).nonzero()
        sol_hits = torch.isclose(row[c + d_cat_aug:c + d_cat_aug + d_solv], ctx["sol_emb_avail"], atol=1e-10, rtol=0.0).all(-1).nonzero()
        if cat_hits.numel() == 0 or sol_hits.numel() == 0:
            raise ValueError(f"batch row {i} has a catalyst or solvent outside the available tables")
        ci, si = int(cat_hits[0]), int(sol_hits[0])
        base_slice = row[c + d_cat_aug + d_solv:]
        if bool(ctx["cat_is_salt_flags"][ci]):
            bas_hits = torch.isclose(base_slice, ctx["bas_emb_avail"], atol=1e-10, rtol=0.0).all(-1).nonzero()
            if bas_hits.numel() == 0:
                raise ValueError(f"batch row {i} has a salt catalyst and a base outside the available table")
            base_id = ctx["bas_ids"][int(bas_hits[0])]
        else:
            if bool((base_slice != 0).any()):
                raise ValueError(f"batch row {i} has a non-salt catalyst and a non-zero base slice")
            base_id = None
        out.append(
            {
                "catalyst_id": ctx["cat_ids"][ci],
                "solvent_id": ctx["sol_ids"][si],
                "base_id": base_id,
                "is_salt": bool(ctx["cat_is_salt_flags"][ci]),
                **raw[i],
            }
        )
    return out


def decode_worker(worker_id: int) -> tuple[str, int, int]:
    """Flat index → ``(optimizer, tier, rep)``.

    The index runs through :data:`config.TIER_BLOCKS` in order; inside a block it
    decodes optimizer (slowest) → tier → rep (fastest), so indices 0-29 are tiers 0-4
    and 30-41 tiers 5-6.
    """
    if not 0 <= worker_id < config.N_WORKERS:
        raise ValueError(f"worker_id must be in [0, {config.N_WORKERS}), got {worker_id}")
    offset = worker_id
    for block in config.TIER_BLOCKS:
        size = len(config.OPTIMIZERS) * len(block) * config.N_REPS
        if offset < size:
            opt_idx, rem = divmod(offset, len(block) * config.N_REPS)
            tier_pos, rep = divmod(rem, config.N_REPS)
            return config.OPTIMIZERS[opt_idx], block[tier_pos], rep
        offset -= size
    raise AssertionError("TIER_BLOCKS do not cover N_WORKERS")


def _meta() -> dict:
    """Host and runtime metadata for a result file."""
    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }


def run_worker(
    worker_id: int,
    heartbeat: Callable[..., None] | None = None,
    smoke_settings: bool = False,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Run one ``(optimizer, tier, rep)`` unit and write its result file last.

    Args:
        worker_id: Flat index in ``[0, N_WORKERS)``.
        heartbeat: Optional callback receiving progress fields.
        smoke_settings: Replace the tier's settings by the tiny smoke settings.
        results_dir: Directory of ``result_<worker_id>.json``.

    Returns:
        Path of the written result file.
    """
    optimizer, tier, rep = decode_worker(worker_id)
    if smoke_settings:
        setting = dict(SMOKE_GA) if optimizer == "ga" else config.pool_greedy_setting(SMOKE_PG_SCALE)
    else:
        setting = config.optimizer_setting(optimizer, tier)
    ctx = _load_context()
    runner = run_ga if optimizer == "ga" else run_pool_greedy
    res = runner(ctx, setting, rep, heartbeat)
    cands = res.pop("candidates")
    if heartbeat is not None:
        heartbeat(stage="score")
    result = {
        "worker_id": worker_id,
        "optimizer": optimizer,
        "tier": tier,
        "rep": rep,
        "seed": rep,
        "target_minutes": config.TIER_TARGET_MINUTES[tier],
        "smoke_settings": smoke_settings,
        "settings": setting,
        **res,
        "score": score(ctx, cands),
        "candidates": cands.tolist(),
        "decoded": decode_batch(ctx, cands),
        "fixed_dims": {str(k): v for k, v in ctx["fixed_dims"].items()},
        "batch_constraints": {str(k): list(v) for k, v in ctx["batch_constraints"].items()},
        "meta": _meta(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"result_{worker_id}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    print(
        f"[worker {worker_id}] {optimizer} tier {tier} rep {rep}: score {result['score']:.4f} "
        f"own {result['own_acq_value']:.4f} in {result['wall_time_s']:.1f} s -> {out}",
        flush=True,
    )
    return out


def smoke() -> None:
    """Tiny GA and pool-greedy runs with their common-scorer values; writes nothing."""
    ctx = _load_context()
    for name, runner, setting in (
        ("ga", run_ga, dict(SMOKE_GA)),
        ("pool_greedy", run_pool_greedy, config.pool_greedy_setting(SMOKE_PG_SCALE)),
    ):
        res = runner(ctx, setting, 0)
        decoded = decode_batch(ctx, res["candidates"])
        temps = sorted(round(r["temperature"], 3) for r in decoded)
        print(
            f"[smoke] {name:>11}: own {res['own_acq_value']:.4f} score {score(ctx, res['candidates']):.4f} "
            f"in {res['wall_time_s']:.1f} s; temperatures {temps}"
        )


def calibrate() -> None:
    """Time both optimizers on this node and print the tier settings for the targets.

    GA: one GA probe of ``CALIBRATION_GA_PROBE_EVALS`` evaluations gives seconds per
    evaluation; the refine runs on its batch at the two
    ``CALIBRATION_REFINE_PROBE_RESTARTS`` and ``time = a + b · restarts`` is fitted. A
    tier's ``n_evals`` fills its target minus the predicted refine time at its
    restarts, rounded down to whole generations. Pool-greedy: runs at the two
    ``CALIBRATION_PG_PROBE_SCALES``, fits ``time = a + b · scale`` and solves for each
    target. Writes the measurements to :data:`CALIBRATION_PATH`.
    """
    ctx = _load_context()
    targets_s = [m * 60.0 for m in config.TIER_TARGET_MINUTES]
    print(f"[calibrate] host {socket.gethostname()}, torch threads {torch.get_num_threads()}", flush=True)

    r0, r1 = config.CALIBRATION_REFINE_PROBE_RESTARTS
    probe = {
        "n_evals_ga": config.CALIBRATION_GA_PROBE_EVALS,
        "pop_size": config.GA_POP_SIZE,
        "lbfgs_num_restarts": r0,
        "lbfgs_raw_samples": config.LBFGS_RAW_SAMPLES,
    }
    res = run_ga(ctx, probe, seed=0)
    per_eval = res["ga_wall_time_s"] / probe["n_evals_ga"]
    t_r0 = res["refine_wall_time_s"]
    print(f"  GA: {per_eval * 1e3:.3f} ms/eval; refine {r0} restart(s) {t_r0:.1f} s", flush=True)
    t = time.perf_counter()
    optimize_acqf_continuous(
        acq_function=_acqf(ctx, config.N_MC_SAMPLES, 0),
        bounds=ctx["bounds"],
        X_discrete=res["candidates"][:, config.D_CONTINUOUS:],
        batch_constraints=ctx["batch_constraints"],
        fixed_dims=ctx["fixed_dims"],
        num_restarts=r1,
        raw_samples=config.LBFGS_RAW_SAMPLES,
        seed=0,
    )
    t_r1 = time.perf_counter() - t
    refine_slope = (t_r1 - t_r0) / (r1 - r0)
    refine_intercept = t_r0 - refine_slope * r0
    print(f"  refine {r1} restarts {t_r1:.1f} s -> {refine_intercept:.1f} + {refine_slope:.1f} s/restart", flush=True)
    ga_n_evals = []
    for tier, restarts in enumerate(config.GA_REFINE_RESTARTS):
        budget = targets_s[tier] - (refine_intercept + refine_slope * restarts)
        if budget <= 0:
            raise ValueError(f"tier {tier}: {restarts} refine restarts alone exceed the {targets_s[tier]:.0f} s target")
        ga_n_evals.append(max(2, int(budget / per_eval / config.GA_POP_SIZE)) * config.GA_POP_SIZE)

    pg_times = []
    for scale in config.CALIBRATION_PG_PROBE_SCALES:
        pg = run_pool_greedy(ctx, config.pool_greedy_setting(scale), seed=0)
        pg_times.append(pg["wall_time_s"])
        print(f"  pool-greedy scale {scale}: {pg['wall_time_s']:.1f} s", flush=True)
    (s0, s1), (p0, p1) = config.CALIBRATION_PG_PROBE_SCALES, pg_times
    pg_slope = (p1 - p0) / (s1 - s0)
    if pg_slope <= 0:
        raise ValueError(f"pool-greedy time does not grow with scale ({pg_times}); probe larger scales")
    pg_intercept = p0 - pg_slope * s0
    pg_scales = [round((t - pg_intercept) / pg_slope, 1) for t in targets_s]
    if min(pg_scales) <= 0:
        raise ValueError(f"pool-greedy fixed cost {pg_intercept:.0f} s exceeds the smallest target")

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(
            {
                "meta": _meta(),
                "ga_s_per_eval": per_eval,
                "refine_probe_restarts": [r0, r1],
                "refine_probe_times_s": [t_r0, t_r1],
                "ga_n_evals": ga_n_evals,
                "pg_probe_scales": list(config.CALIBRATION_PG_PROBE_SCALES),
                "pg_probe_times_s": pg_times,
                "pg_scales": pg_scales,
            },
            indent=2,
        )
    )
    print("\n# paste into config.py")
    print(f"GA_N_EVALS: tuple[int, ...] = ({', '.join(str(n) for n in ga_n_evals)})")
    print(f"PG_TIERS: tuple[float, ...] = ({', '.join(str(s) for s in pg_scales)})")


def main() -> None:
    """CLI: ``prepare`` | ``smoke`` | ``calibrate`` | ``worker``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("smoke")
    sub.add_parser("calibrate")
    wp = sub.add_parser("worker")
    wp.add_argument("--worker-id", type=int, required=True)
    wp.add_argument("--smoke", action="store_true")
    wp.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "smoke":
        smoke()
    elif args.cmd == "calibrate":
        calibrate()
    elif args.cmd == "worker":
        run_worker(args.worker_id, smoke_settings=args.smoke, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
