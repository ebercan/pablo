"""Paired benchmark of the legacy, corrected-operator and canonical batch GAs.

Three arms are compared on the same surrogate and the same acquisition function:

- **legacy**: :func:`pablo.optimize.optimize_acqf_mixed_GA`, which encodes the
  catalyst / solvent / base indices as ``Integer`` and lets pymoo breed them
  with SBX and polynomial mutation.
- **fixed**: :func:`pablo.optimize.optimize_acqf_mixed_canonical_GA` with
  ``canonicalize=False`` — ``Choice`` encoding with uniform crossover and
  random-reset mutation, batched population decoding, no symmetry handling.
- **canonical**: the same with ``canonicalize=True``, projecting every
  individual onto a canonical labeling of the batch.

Self-reported acquisition values are not comparable across arms. The joint
qLogEI estimate is a set function only in expectation: its QMC sampler couples
base samples to candidate positions, so relabeling one physical batch moves the
estimate. An arm free to relabel can therefore inflate its own score without
finding a better batch. Every returned batch is re-scored here on a common
footing — rows sorted into one labeling, one fixed high-sample sampler — and
only those numbers are compared.

Run from the repository root::

    python -m pablo.utils.canonical_ga_benchmark --seeds 30 --n-evals 20000
"""

import argparse
import json
import time

import numpy as np
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.stats import wilcoxon
from torch import Tensor

from pablo.optimize import optimize_acqf_mixed_GA, optimize_acqf_mixed_canonical_GA

D_CAT, D_SOL, D_BAS = 32, 8, 8
N_CAT, N_SOL, N_BAS = 22, 14, 8
D_CONTINUOUS = 4
RESCORE_SAMPLES = 4096
RESCORE_SEED = 12345


def build_problem(seed: int, n_train: int):
    """Build a surrogate and a matching mixed search space for one replicate.

    Args:
        seed: Replicate seed, controlling both the surrogate's training data and
            the random component embeddings.
        n_train: Number of training points for the surrogate.

    Returns:
        A five-tuple ``(model, acq_function, bounds, ga_kwargs, best_f)``, where
        ``bounds`` is ``(2, d)`` with ``d = 4 + d_cat + d_sol + d_bas`` and
        ``ga_kwargs`` carries the component embedding tables and constraints.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = D_CONTINUOUS + D_CAT + D_SOL + D_BAS
    train_x = torch.rand(n_train, d, dtype=torch.float64)
    train_y = torch.rand(n_train, 1, dtype=torch.float64)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    best_f = train_y.max()
    acq_function = qLogExpectedImprovement(model, best_f=best_f)
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(torch.float64)
    ga_kwargs = dict(
        cat_emb_avail=torch.rand(N_CAT, D_CAT, dtype=torch.float64),
        sol_emb_avail=torch.rand(N_SOL, D_SOL, dtype=torch.float64),
        bas_emb_avail=torch.rand(N_BAS, D_BAS, dtype=torch.float64),
        cat_is_salt_flags=torch.rand(N_CAT) > 0.5,
        batch_constraints={0: (2, 2, 2)},
        fixed_dims={3: 0.5},
    )
    return model, acq_function, bounds, ga_kwargs, best_f


def rescore(model: SingleTaskGP, best_f: Tensor, batch: Tensor) -> float:
    """Re-score a batch on a labeling-independent footing.

    Args:
        model: The surrogate the batch was optimized against.
        best_f: Incumbent used by the acquisition function.
        batch: A ``(q, d)`` tensor of candidates in arbitrary row order.

    Returns:
        The joint qLogEI value of the row-sorted batch under a fixed
        high-sample sampler.
    """
    order = sorted(range(batch.shape[0]), key=lambda i: tuple(batch[i].tolist()))
    acq_function = qLogExpectedImprovement(
        model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(
            sample_shape=torch.Size([RESCORE_SAMPLES]), seed=RESCORE_SEED
        ),
    )
    with torch.no_grad():
        return acq_function(batch[order].unsqueeze(0)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-evals", type=int, default=20000)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=65)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path for a JSON dump of the per-seed records.",
    )
    args = parser.parse_args()

    arms = {
        "legacy": lambda acq, bounds, kw: optimize_acqf_mixed_GA(
            acq_function=acq,
            bounds=bounds,
            q=args.q,
            n_evals_ga=args.n_evals,
            pop_size=args.pop_size,
            **kw,
        ),
        "fixed": lambda acq, bounds, kw: optimize_acqf_mixed_canonical_GA(
            acq_function=acq,
            bounds=bounds,
            q=args.q,
            n_evals_ga=args.n_evals,
            pop_size=args.pop_size,
            canonicalize=False,
            **kw,
        ),
        "canonical": lambda acq, bounds, kw: optimize_acqf_mixed_canonical_GA(
            acq_function=acq,
            bounds=bounds,
            q=args.q,
            n_evals_ga=args.n_evals,
            pop_size=args.pop_size,
            canonicalize=True,
            **kw,
        ),
    }

    reported = {name: [] for name in arms}
    fair = {name: [] for name in arms}
    wall = {name: [] for name in arms}

    for seed in range(args.seeds):
        model, acq_function, bounds, ga_kwargs, best_f = build_problem(
            seed, args.n_train
        )
        for name, run in arms.items():
            start = time.perf_counter()
            candidates, acq_value = run(acq_function, bounds, ga_kwargs)
            wall[name].append(time.perf_counter() - start)
            reported[name].append(acq_value.item())
            fair[name].append(rescore(model, best_f, candidates))
        print(
            f"seed {seed:3d}  "
            + "  ".join(f"{name}={fair[name][-1]:8.4f}" for name in arms),
            flush=True,
        )

    print(
        f"\n{args.seeds} replicates, q={args.q}, n_evals={args.n_evals}, "
        f"pop_size={args.pop_size}"
    )
    header = (
        f"{'arm':12s} {'self-reported':>14s} {'re-scored':>12s} {'SE':>9s} "
        f"{'inflation':>11s} {'wall (s)':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name in arms:
        values = np.asarray(fair[name])
        inflation = np.asarray(reported[name]) - values
        print(
            f"{name:12s} {np.mean(reported[name]):14.5f} {values.mean():12.5f} "
            f"{values.std(ddof=1) / np.sqrt(len(values)):9.5f} "
            f"{inflation.mean():+11.5f} "
            f"{np.mean(wall[name]):10.2f}"
        )

    base = np.asarray(fair["legacy"])
    print("\npaired against legacy, on re-scored values")
    for name in ("fixed", "canonical"):
        delta = np.asarray(fair[name]) - base
        stat = wilcoxon(delta) if np.any(delta) else None
        p_value = f"{stat.pvalue:.4f}" if stat is not None else "n/a"
        print(
            f"  {name:10s} mean delta {delta.mean():+.5f}  "
            f"SE {delta.std(ddof=1) / np.sqrt(len(delta)):.5f}  "
            f"wins {int((delta > 0).sum())}/{len(delta)}  wilcoxon p={p_value}"
        )

    delta = np.asarray(fair["canonical"]) - np.asarray(fair["fixed"])
    stat = wilcoxon(delta) if np.any(delta) else None
    p_value = f"{stat.pvalue:.4f}" if stat is not None else "n/a"
    print(
        f"\ncanonical vs fixed (isolates the symmetry): mean delta {delta.mean():+.5f}  "
        f"SE {delta.std(ddof=1) / np.sqrt(len(delta)):.5f}  "
        f"wins {int((delta > 0).sum())}/{len(delta)}  wilcoxon p={p_value}"
    )

    if args.out is not None:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "config": vars(args),
                    "reported": reported,
                    "rescored": fair,
                    "wall": wall,
                },
                handle,
                indent=2,
            )
        print(f"\nper-seed records written to {args.out}")


if __name__ == "__main__":
    main()
