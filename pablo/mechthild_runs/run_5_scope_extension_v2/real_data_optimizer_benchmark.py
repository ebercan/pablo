"""Pool-greedy against the production GA on the NHC experimental dataset.

The live-campaign setting of ``pablo.main``: the workbook's training data and
"Next Batch" components, a ``SingleTaskGP`` from ``train_gp_model``, qLogNEI
with the raw training inputs as baseline, q = 6, temperature partition
(2, 2, 2) and time pinned to 20 h. For every data fraction and seed the GP is
fitted on a yield-stratified subset; ``optimize_acqf_mixed_pool_greedy`` and
``optimize_acqf_mixed_GA`` (``GA_SETTINGS``) each optimize their own fresh
qLogNEI, and both batches are scored by a third qLogNEI with
``REAL_DATA_SCORE_QMC_SAMPLES`` fixed base samples, so neither optimizer is
scored on the MC samples it optimized.

Run from the repo root::

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.real_data_optimizer_benchmark \
        [--fractions F ...] [--seeds S ...] [--ga_n_evals N] [--ga_pop_size P] \
        [--n_pool N] [--n_refine N] [--out PATH]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from sklearn.model_selection import train_test_split

from pablo.main import (
    DEFAULT_EXCEL_FILE,
    DEVICE,
    DTYPE,
    GA_SETTINGS,
    TEMPERATURE_PARTITION,
    ContinuousBounds,
    load_available_agents,
    load_reference_data,
    load_training_data,
    train_gp_model,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    Q_BATCH_SIZE,
    REAL_DATA_FRACTIONS,
    REAL_DATA_N_SEEDS,
    REAL_DATA_N_YIELD_BINS,
    REAL_DATA_POLISH_MAXITER,
    REAL_DATA_POOL_SIZE,
    REAL_DATA_QMC_SAMPLES,
    REAL_DATA_REFINE,
    REAL_DATA_RESULTS_FILENAME,
    REAL_DATA_SCORE_QMC_SAMPLES,
    REAL_DATA_SCORE_QMC_SEED,
)
from pablo.optimize import optimize_acqf_mixed_GA, optimize_acqf_mixed_pool_greedy

TIME_DIM = 3
TIME_PINNED_H = 20.0


def stratified_subset(
    x: torch.Tensor, y: torch.Tensor, fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep ``fraction`` of the rows, stratified on yield quantile bins.

    Args:
        x: ``(n, d)`` inputs.
        y: ``(n, 1)`` yields.
        fraction: Share of rows kept, in ``(0, 1]``; ``1`` returns the inputs.
        seed: Seed of the split.

    Returns:
        ``(x_sub, y_sub)`` of shapes ``(m, d)`` and ``(m, 1)``, rows in their
        original order, ``m = floor(n · fraction)``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return x, y
    n = y.shape[0]
    bins = pd.qcut(y.squeeze(-1).cpu().numpy(), q=REAL_DATA_N_YIELD_BINS, labels=False, duplicates="drop")
    keep, _ = train_test_split(np.arange(n), train_size=int(n * fraction), stratify=bins, random_state=seed)
    idx = torch.as_tensor(np.sort(keep), device=x.device)
    return x[idx], y[idx]


def make_acqf(model: SingleTaskGP, x_baseline: torch.Tensor, sampler=None) -> qLogNoisyExpectedImprovement:
    """qLogNEI with the raw (untransformed) ``(n, d)`` training inputs as baseline."""
    return qLogNoisyExpectedImprovement(model=model, X_baseline=x_baseline, sampler=sampler)


def run_benchmark(
    fractions: list[float],
    seeds: list[int],
    ga_n_evals: int,
    ga_pop_size: int,
    n_pool: int,
    n_refine: int,
    out_path: Path,
) -> pd.DataFrame:
    """Run every (fraction, seed) cell and write one CSV row per cell.

    Args:
        fractions: Data fractions, each in ``(0, 1]``.
        seeds: Seeds of the subset, the GP fit and both optimizers.
        ga_n_evals: GA evaluation budget.
        ga_pop_size: GA population size.
        n_pool: Pool-greedy random pool size.
        n_refine: Pool-greedy refined rows.
        out_path: CSV path, rewritten after every cell.

    Returns:
        One row per cell: ``fraction``, ``seed``, ``n_train``, ``pg_time``,
        ``pg_own_val``, ``pg_score``, ``ga_time``, ``ga_own_val``, ``ga_score``.
        ``*_own_val`` is the value each optimizer reports on its own qLogNEI;
        ``*_score`` is the common scorer's value.
    """
    cat_map, sol_map, bas_map, cat_is_salt, *_ = load_reference_data(DEFAULT_EXCEL_FILE)
    cats, sols, bases = load_available_agents(DEFAULT_EXCEL_FILE, "Next Batch")
    tkwargs = {"dtype": DTYPE, "device": DEVICE}
    cat_emb = torch.stack([cat_map[c] for c in cats]).to(**tkwargs)
    sol_emb = torch.stack([sol_map[s] for s in sols]).to(**tkwargs)
    bas_emb = torch.stack([bas_map[b] for b in bases]).to(**tkwargs)
    salt_flags = torch.tensor([cat_is_salt[c] for c in cats], dtype=torch.bool, device=DEVICE)

    cont_bounds = ContinuousBounds()
    cb = cont_bounds.to_tensor()
    fixed_dims = {TIME_DIM: (TIME_PINNED_H - cb[0, TIME_DIM].item()) / (cb[1, TIME_DIM].item() - cb[0, TIME_DIM].item())}
    x_full, y_full = load_training_data(DEFAULT_EXCEL_FILE, cat_map, sol_map, bas_map, cont_bounds, cat_is_salt)
    x_full, y_full = x_full.to(**tkwargs), y_full.to(**tkwargs)
    bounds = torch.zeros(2, x_full.shape[1], **tkwargs)
    bounds[1] = 1.0
    shared = {
        "bounds": bounds,
        "q": Q_BATCH_SIZE,
        "cat_emb_avail": cat_emb,
        "sol_emb_avail": sol_emb,
        "bas_emb_avail": bas_emb,
        "cat_is_salt_flags": salt_flags,
        "batch_constraints": {0: TEMPERATURE_PARTITION},
        "fixed_dims": fixed_dims,
    }
    print(f"{x_full.shape[0]} experiments, d = {x_full.shape[1]}, {len(cats)} catalysts available", flush=True)

    rows = []
    for fraction in fractions:
        for seed in seeds:
            x_train, y_train = stratified_subset(x_full, y_full, fraction, seed)
            torch.manual_seed(seed)
            model = train_gp_model(x_train, y_train)
            scorer = make_acqf(
                model,
                x_train,
                SobolQMCNormalSampler(sample_shape=torch.Size([REAL_DATA_SCORE_QMC_SAMPLES]), seed=REAL_DATA_SCORE_QMC_SEED),
            )

            t0 = time.perf_counter()
            pg_batch, pg_val = optimize_acqf_mixed_pool_greedy(
                acq_function=make_acqf(model, x_train),
                n_pool=n_pool,
                n_refine=n_refine,
                polish_maxiter=REAL_DATA_POLISH_MAXITER,
                n_mc=REAL_DATA_QMC_SAMPLES,
                seed=seed,
                **shared,
            )
            pg_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            ga_batch, ga_val = optimize_acqf_mixed_GA(
                acq_function=make_acqf(model, x_train),
                n_evals_ga=ga_n_evals,
                pop_size=ga_pop_size,
                seed=seed,
                **shared,
            )
            ga_time = time.perf_counter() - t0

            with torch.no_grad():
                pg_score = scorer(pg_batch.unsqueeze(0)).item()
                ga_score = scorer(ga_batch.unsqueeze(0)).item()
            rows.append(
                {
                    "fraction": fraction,
                    "seed": seed,
                    "n_train": x_train.shape[0],
                    "pg_time": pg_time,
                    "pg_own_val": pg_val.item(),
                    "pg_score": pg_score,
                    "ga_time": ga_time,
                    "ga_own_val": ga_val.item(),
                    "ga_score": ga_score,
                }
            )
            print(
                f"fraction {fraction:.2f} seed {seed} (n={x_train.shape[0]}): "
                f"pool-greedy {pg_score:.4f} in {pg_time:.1f} s | GA {ga_score:.4f} in {ga_time:.1f} s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    """Per fraction: median score Δ (pool-greedy − GA), share Δ ≥ 0 and median times."""
    delta = df.pg_score - df.ga_score
    g = df.assign(delta=delta, pg_ge_ga=delta >= 0).groupby("fraction")
    table = pd.DataFrame(
        {
            "n_train": g.n_train.first(),
            "delta_median": g.delta.median(),
            "delta_min": g.delta.min(),
            "share_pg>=ga": g.pg_ge_ga.mean(),
            "pg_s_median": g.pg_time.median(),
            "ga_s_median": g.ga_time.median(),
            "runs": g.size(),
        }
    ).round(4)
    return table.to_string()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fractions", type=float, nargs="+", default=list(REAL_DATA_FRACTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(REAL_DATA_N_SEEDS)))
    parser.add_argument("--ga_n_evals", type=int, default=GA_SETTINGS[0])
    parser.add_argument("--ga_pop_size", type=int, default=GA_SETTINGS[1])
    parser.add_argument("--n_pool", type=int, default=REAL_DATA_POOL_SIZE)
    parser.add_argument("--n_refine", type=int, default=REAL_DATA_REFINE)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / REAL_DATA_RESULTS_FILENAME)
    args = parser.parse_args()
    df = run_benchmark(
        args.fractions, args.seeds, args.ga_n_evals, args.ga_pop_size, args.n_pool, args.n_refine, args.out
    )
    print(f"\nResults written to {args.out}\n")
    print(summarize(df))


if __name__ == "__main__":
    main()
