"""Rebuild every fold's predictive distribution of a finished run, without refitting.

Each ok fold of ``cells/result_<outer_id>.json`` stores the fitted raw parameters
(``hyperparams``) and ``train_idx``. Building the cell's model on ``train_idx`` with
the current code and loading those parameters reproduces the fitted GP, whose
predictive distribution at a test point is Gaussian on the model's outcome scale::

    Z ~ N(m, s²),    s² = posterior variance + noise · stdv²,    Y = inverse(Z)

with ``inverse`` the outcome back-transform (``pablo.utils.outcome_transforms``).
``m`` and ``s`` per test point are what this module writes; any score on the raw
yield scale is then exact for every outcome transform, where the stored
``mu_pred_per_point`` / ``var_pred_per_point`` of a non-raw cell are moments of
2000 posterior draws.

Every cell is checked against what the run stored before it is written:

* ``raw`` cells: rebuilt mean and variance equal the stored ones to 1e-6;
* other transforms: the exact raw-scale mean and variance (Gauss–Hermite, 200
  nodes) differ from the stored sample moments by Monte Carlo error only — the
  root-mean-square z-score of both over all test points of the cell must stay
  below ``MC_RMS_Z_MAX``. A model built differently from the fitted one shifts
  the predictions far beyond that.

A failed check raises; a cell is never written from a model that does not
reproduce the run. Folds with ``status != "ok"`` carry no parameters and are
skipped, their count recorded.

Output: ``runs/<run>/predictive/cell_<outer_id>.parquet``, one row per test point
of every ok fold of every study, with columns ``outer_cell_id``, ``split_type``,
``repeat``, ``fold``, ``idx``, ``y`` (raw yield fraction), ``transform``, ``m``,
``s``, ``ybar_train``, ``sd_train`` (mean and unbiased SD of the fold's training
yields, the constant baseline of CRPSS). A cell whose file exists is skipped.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.results_analysis.rebuild_predictive --run cross_1 --workers 2
    python -m pablo.featurization_and_kernel_scan.results_analysis.rebuild_predictive --run cross_1 --cells 1339 2235
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.polynomial.hermite_e import hermegauss
from torch import Tensor

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import build_partition
from pablo.featurization_and_kernel_scan.core.models import AcyloinCondensationModel, build_mean_module
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import N_PRED_SAMPLES
from pablo.utils.feature_assembly import CellFeatures, assemble_features
from pablo.utils.outcome_transforms import ARCSIN_SQRT_MAX, inverse

RAW_TOL: float = 1e-6
"""Absolute tolerance on the mean and relative tolerance on the variance of raw cells."""

MC_RMS_Z_MAX: float = 1.5
"""Largest root-mean-square z-score of stored sample moments against exact moments."""

N_GH: int = 200
"""Gauss–Hermite nodes for the exact raw-scale moments."""


def predictive_dir(run: str) -> Path:
    """Folder of the rebuilt predictive files of one run."""
    return paths.run_paths(run).cells_dir.parent / "predictive"


def build_cell(cfg: dict, inner: dict) -> CellFeatures:
    """Assemble one cell's features exactly as ``scheduler._run_outer_cell`` does."""
    cell = assemble_features(
        cat_featurization=cfg["cat_feat"],
        solv_featurization=cfg["solv_feat"],
        base_featurization=inner["base_feat"],
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
        append_form_onehot_to_cat=cfg.get("append_form_onehot_to_cat", False),
    )
    cell.partition = build_partition(
        cell.feature_dims, mode=config.PRIOR_SCHEMES[inner["prior_scheme"]].partition_mode
    )
    return cell


def rebuild_fold(cell: CellFeatures, cfg: dict, inner: dict, fold: dict) -> tuple[Tensor, Tensor]:
    """Predictive mean and SD on the model's outcome scale for one fold's test points.

    Args:
        cell: The cell's assembled features.
        cfg: The outer cell configuration (``inputs.json`` entry).
        inner: The inner configuration (``base_feat``, ``prior_scheme``).
        fold: An ok fold record with ``train_idx``, ``test_idx`` and ``hyperparams``.

    Returns:
        ``(m, s)``, each of shape ``n_test``: mean and SD of the Gaussian predictive
        distribution of the observation on the outcome-transform scale.

    Raises:
        ValueError: If the model's parameter names differ from the stored ones, or
            the single-task class assumption does not hold.
    """
    if cfg["model_class"] != "st_y":
        raise ValueError(f"rebuild supports the single-task class st_y only, got {cfg['model_class']!r}")
    train_idx, test_idx = fold["train_idx"], fold["test_idx"]
    mean_module = build_mean_module(
        cfg["mean_function"], cell, prior_scale=config.MEAN_MODULE_NORMAL_SCALE, train_x=cell.X[train_idx]
    )
    model = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure=cfg["kernel_structure"],
        base_kernel_type=cfg["base_kernel"],
        outcome_transform=cfg["outcome_transform"],
        lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme=inner["prior_scheme"],
        mean_function=cfg["mean_function"],
    ).build(cfg["model_class"], mean_module)
    params = dict(model.named_parameters())
    stored = fold["hyperparams"]
    if set(params) != set(stored):
        raise ValueError(f"parameter names differ from the stored fit: {sorted(set(params) ^ set(stored))}")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(torch.as_tensor(stored[name], dtype=p.dtype).reshape(p.shape))
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(cell.X[test_idx])
        noise = model.likelihood.noise.squeeze() * model.outcome_transform.stdvs.squeeze() ** 2
        var = post.variance.squeeze(-1) + noise
    return post.mean.squeeze(-1), var.sqrt()


def _arcsin_sqrt_nodes(m: np.ndarray, s: np.ndarray) -> tuple[Tensor, np.ndarray]:
    """Quadrature of ``Z ~ N(m, s²)`` clamped to ``[0, π/2]``, the domain ``inverse`` uses.

    Two atoms carry ``P(Z ≤ 0)`` at 0 and ``P(Z ≥ π/2)`` at π/2; ``N_GH``
    Gauss-Legendre nodes on ``[0, π/2]`` carry the density in between.

    Returns:
        ``(z, w)``: nodes ``(n, N_GH + 2)`` and weights ``(n, N_GH + 2)`` summing to 1 per row.
    """
    from scipy.stats import norm

    top = ARCSIN_SQRT_MAX
    x, wl = np.polynomial.legendre.leggauss(N_GH)
    z_in = (x + 1.0) * top / 2.0  # (N_GH,)
    dens = norm.pdf((z_in[None, :] - m[:, None]) / s[:, None]) / s[:, None]
    w_in = dens * (wl * top / 2.0)[None, :]
    lo = norm.cdf(-m / s)
    hi = norm.sf((top - m) / s)
    z = np.concatenate([np.zeros((len(m), 1)), np.broadcast_to(z_in, (len(m), N_GH)), np.full((len(m), 1), top)], axis=1)
    w = np.concatenate([lo[:, None], w_in, hi[:, None]], axis=1)
    return torch.as_tensor(z, dtype=torch.float64), w / w.sum(axis=1, keepdims=True)


def raw_moments(m: np.ndarray, s: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact mean, variance and fourth central moment of ``inverse(Z)``, ``Z ~ N(m, s²)``.

    Gauss-Hermite for the smooth inverses; for ``arcsin_sqrt``, whose inverse
    clamps ``Z`` to ``[0, π/2]``, two atoms plus Gauss-Legendre on that interval
    (see :func:`_arcsin_sqrt_nodes`).

    Args:
        m: Predictive means on the transform scale, shape ``n``.
        s: Predictive SDs on the transform scale, shape ``n``.
        transform: Outcome transform name.

    Returns:
        ``(mean, var, m4)``, each of shape ``n``.
    """
    if transform == "arcsin_sqrt":
        z, w = _arcsin_sqrt_nodes(m, s)
    else:
        nodes, weights = hermegauss(N_GH)
        z = torch.as_tensor(m[:, None] + s[:, None] * nodes[None, :], dtype=torch.float64)
        w = np.broadcast_to(weights / weights.sum(), z.shape)
    y = inverse(z, transform).numpy()
    mean = (y * w).sum(axis=1)
    dev = y - mean[:, None]
    return mean, (dev ** 2 * w).sum(axis=1), (dev ** 4 * w).sum(axis=1)


def check_against_stored(frame: pd.DataFrame, mu_stored: np.ndarray, var_stored: np.ndarray, oid: int) -> None:
    """Raise when the rebuilt predictive distributions do not reproduce the stored moments.

    Args:
        frame: Rebuilt rows of one cell (``m``, ``s``, ``transform``).
        mu_stored: Stored ``mu_pred_per_point`` aligned with ``frame``.
        var_stored: Stored ``var_pred_per_point`` aligned with ``frame``.
        oid: Outer cell id, for the message.

    Raises:
        RuntimeError: If a raw cell differs beyond ``RAW_TOL`` or a transformed
            cell's RMS z-score exceeds ``MC_RMS_Z_MAX``.
    """
    transform = frame["transform"].iloc[0]
    m, s = frame["m"].to_numpy(), frame["s"].to_numpy()
    if transform == "raw":
        d_mu = np.abs(m - mu_stored).max()
        d_var = np.abs(s ** 2 / var_stored - 1).max()
        if d_mu > RAW_TOL or d_var > RAW_TOL:
            raise RuntimeError(f"cell {oid}: rebuilt raw moments differ (|dmu| {d_mu:.2e}, |var ratio-1| {d_var:.2e})")
        return
    mean, var, m4 = raw_moments(m, s, transform)
    z_mu = (mu_stored - mean) / np.sqrt(var / N_PRED_SAMPLES)
    z_var = (var_stored - var) / np.sqrt((m4 - var ** 2) / N_PRED_SAMPLES)
    rms_mu, rms_var = np.sqrt(np.mean(z_mu ** 2)), np.sqrt(np.mean(z_var ** 2))
    if rms_mu > MC_RMS_Z_MAX or rms_var > MC_RMS_Z_MAX:
        raise RuntimeError(
            f"cell {oid} ({transform}): stored sample moments are not Monte Carlo draws of the rebuilt "
            f"distribution (RMS z mean {rms_mu:.2f}, RMS z variance {rms_var:.2f}, limit {MC_RMS_Z_MAX})"
        )


def rebuild_cell(job: tuple[str, int]) -> tuple[int, int, int]:
    """Rebuild, check and write one outer cell.

    Args:
        job: ``(run, outer_cell_id)``.

    Returns:
        ``(outer_cell_id, n_rows_written, n_failed_folds)``; ``n_rows_written`` is
        ``-1`` when the file already existed.
    """
    run, oid = job
    out = predictive_dir(run) / f"cell_{oid}.parquet"
    if out.exists():
        return oid, -1, 0
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rp = paths.run_paths(run)
    cfg = json.loads(rp.inputs_json.read_text())[str(oid)]
    result = json.loads((rp.cells_dir / f"result_{oid}.json").read_text())
    if len(result["inner_cells"]) != 1:
        raise ValueError(f"cell {oid}: expected one inner cell, found {len(result['inner_cells'])}")
    inner_res = result["inner_cells"][0]
    inner = inner_res["inner_config"]
    cell = build_cell(cfg, inner)
    y_all = cell.y_raw.squeeze(-1).numpy()
    rows: list[dict] = []
    mu_stored: list[float] = []
    var_stored: list[float] = []
    n_failed = 0
    for fold in inner_res["folds"]:
        if fold["status"] != "ok":
            n_failed += 1
            continue
        m, s = rebuild_fold(cell, cfg, inner, fold)
        y_train = y_all[fold["train_idx"]]
        for j, i in enumerate(fold["test_idx"]):
            rows.append(dict(
                outer_cell_id=oid, split_type=fold["split_type"], repeat=fold["repeat"], fold=fold["fold"],
                idx=i, y=float(y_all[i]), transform=cfg["outcome_transform"], m=float(m[j]), s=float(s[j]),
                ybar_train=float(y_train.mean()), sd_train=float(y_train.std(ddof=1)),
            ))
        mu_stored.extend(fold["mu_pred_per_point"])
        var_stored.extend(fold["var_pred_per_point"])
    frame = pd.DataFrame(rows)
    check_against_stored(frame, np.asarray(mu_stored), np.asarray(var_stored), oid)
    tmp = out.with_suffix(".tmp")
    frame.to_parquet(tmp)
    tmp.replace(out)
    return oid, len(frame), n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="cross_1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", type=int, nargs="*", help="outer cell ids (default: every cell of the run)")
    args = parser.parse_args()
    rp = paths.run_paths(args.run)
    ids = args.cells or sorted(int(k) for k in json.loads(rp.inputs_json.read_text()))
    predictive_dir(args.run).mkdir(exist_ok=True)
    jobs = [(args.run, oid) for oid in ids]
    done = 0
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for oid, n_rows, n_failed in pool.imap_unordered(rebuild_cell, jobs):
            done += 1
            state = "exists" if n_rows < 0 else f"{n_rows} rows"
            print(f"[{done}/{len(jobs)}] cell {oid}: {state}, {n_failed} failed folds skipped", flush=True)


if __name__ == "__main__":
    main()
