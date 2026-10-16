"""Self-calibrating runtime estimator for NHC batch / scope acquisition optimization.

Background — how this model was derived
========================================
The batch generation in :mod:`pablo.main` spends essentially all of its wall
time inside two operations, and an empirical profiling study (CPU, q=6,
temperature partition ``(2, 2, 2)``) pinned down how each scales:

1. **The GA** (:func:`pablo.optimize.optimize_acqf_mixed_GA`). Wall time is
   ~87% pymoo framework + per-individual candidate building, ~13% GP evaluation.
   It is **linear in ``n_evals``** and grows as **``sqrt(pop)``** (per-generation
   cost is ~O(pop^1.5) over ``n_evals/pop`` generations). Feature dimensionality
   is negligible. Fitted: ``t_ga ~= K * n_evals * sqrt(pop)`` with R^2 = 0.995.

2. **The continuous refine** (:func:`pablo.optimize.optimize_acqf_continuous`),
   called by both the batch final-refinement step and the sequential optimizer.
   For the ``(2, 2, 2)`` partition it enumerates ``N_groupings`` temperature
   groupings (15 for the batch final pass, 39 summed over the 6 sequential
   steps), each running ``num_restarts`` scipy L-BFGS-B optimizations. Cost is
   ``N_groupings * num_restarts * E[nfev] * t_eval`` where ``E[nfev]`` is the
   mean L-BFGS function-evaluations per restart and ``t_eval`` the per-evaluation
   cost. ``raw_samples`` is negligible (one batched Sobol evaluation).

Why the estimate is stable (the variance story)
------------------------------------------------
Naive wall-time timing of the refine looked like it had ~40% variance. That was
an artefact of three things: (a) measuring at ``num_restarts`` of 1-2 where the
average over groupings has not kicked in, (b) **unseeded** Sobol multistarts, and
(c) the machine's own ~+/-10% CPU-speed jitter (thermal/frequency/load). The key
realisation is that wall time = *deterministic work* x *machine speed*:

* The **work** (``n_evals`` for the GA; ``nfev`` for the refine) is deterministic
  once the optimizers are seeded. The GA already uses ``seed=1``; the refine's
  Sobol draw is seeded too (see ``seed`` in
  :func:`pablo.optimize.optimize_acqf_continuous`). With seeding, repeated refines
  agree to ~0.6% (pure machine jitter), down from 16% unseeded at nr=1.
* The **machine speed** and ``E[nfev]`` both depend on hardware and on the *real*
  GP (``E[nfev]`` rises with the training-set size ``n_train``: ~48 at n_train=120,
  ~141 at 240). Rather than bake hardware/data-specific constants, we
  **self-calibrate at runtime**: a few cheap GA and refine probes on the actual
  fitted model measure the GA slope and the per-restart refine cost directly,
  then we extrapolate to the configured settings. The extrapolation is in the
  exactly-known integer counts (``n_evals``, ``N_groupings``, ``num_restarts``),
  so the estimate inherits only the ~+/-10% machine jitter.

Resulting model (per repeat, ``R = GA_OPTIMIZATION_REPEATS`` repeats)
--------------------------------------------------------------------
* Batch:       ``T = R * ( s_ga * n_evals  +  15 * num_restarts * c1 )``
* Sequential:  ``T = R * ( q * s_ga * n_evals  +  39 * num_restarts * c1 )``
  (the per-step GA term is an upper bound: real sequential steps optimize one
  new candidate so they are somewhat cheaper than a full q-batch GA.)
* Scope (q=1, no refine): ``T = R * s_ga * n_evals``.

``s_ga`` (seconds per GA evaluation) and ``c1`` (seconds per grouping-restart)
are measured by :func:`calibrate`; ``N_groupings`` is computed exactly from
``q`` and the partition via the same helpers the optimizer uses.

Caveat: ``c1`` is calibrated with a single-grouping (unconstrained) refine for
speed, which has slightly more free parameters than a real ``(2, 2, 2)`` grouping,
so the refine term carries a few-% positive bias. The GA term (often dominant at
large ``n_evals``) is calibrated faithfully.
"""

import contextlib
import io
import time
from typing import Dict, Optional, Tuple

from torch import Tensor

from pablo.optimize import (
    _enumerate_partition_groupings,
    _partial_partition_for_step,
    optimize_acqf_continuous,
    optimize_acqf_mixed_GA,
)



def grouping_counts(q: int, partition: Tuple[int, ...]) -> Tuple[int, int]:
    """Count L-BFGS grouping-solves for the batch and sequential refine paths.

    Args:
        q: Batch size.
        partition: Temperature partition (e.g. ``(2, 2, 2)``).

    Returns:
        A tuple ``(n_batch, n_sequential)`` of grouping-solve counts. ``n_batch``
        is the number of groupings of ``q`` into ``partition``; ``n_sequential``
        is the sum over greedy steps ``k = 1..q`` of the per-step groupings.
    """
    target = tuple(sorted(partition, reverse=True))
    n_batch = len(_enumerate_partition_groupings(q, target))
    n_seq = 0
    for k in range(1, q + 1):
        partial = _partial_partition_for_step(target, k)
        n_seq += 1 if partial is None else len(_enumerate_partition_groupings(k, partial))
    return n_batch, n_seq


def calibrate(
    acqf,
    bounds: Tensor,
    cat_emb_avail: Tensor,
    sol_emb_avail: Tensor,
    bas_emb_avail: Tensor,
    cat_is_salt_flags: Tensor,
    q: int,
    pop: int,
    raw_samples: int,
    fixed_dims: Dict[int, float],
    batch_constraints: Optional[Dict] = None,
    calibrate_refine: bool = True,
    ga_n_cal: Tuple[int, int] = (1500, 3000),
    refine_nr_cal: int = 4,
) -> Dict[str, float]:
    """Measure the GA slope and per-restart refine cost on the real model.

    Runs two small GA optimizations (to fit a slope free of the fixed startup
    intercept) and, when ``calibrate_refine`` is True, one single-grouping refine
    (``batch_constraints=None``) to measure the per-restart cost ``c1``.

    Args:
        acqf: The acquisition function that will be optimized.
        bounds: ``(2, d_total)`` bounds tensor.
        cat_emb_avail: ``(n_cat, d_cat)`` available-catalyst embeddings.
        sol_emb_avail: ``(n_sol, d_sol)`` available-solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` available-base embeddings.
        cat_is_salt_flags: ``(n_cat,)`` bool tensor; False zeroes the base block
            for that catalyst.
        q: Batch size used for the GA calibration.
        pop: GA population size (calibrate at the real ``pop`` so the ``sqrt(pop)``
            dependence is already baked into ``s_ga``).
        raw_samples: Refine Sobol sample count (passed through; ~irrelevant to time).
        fixed_dims: Fixed-dimension map (e.g. Time).
        batch_constraints: Constraints for the GA calibration (e.g. the partition).
        calibrate_refine: Whether to also calibrate the refine cost.
        ga_n_cal: Two ``n_evals`` points for the GA slope fit.
        refine_nr_cal: ``num_restarts`` for the single-grouping refine probe.

    Returns:
        A dict with ``s_ga`` (s per GA eval), ``ga_intercept`` (s), and, when
        calibrated, ``c1`` (s per grouping-restart). Also echoes the raw probe
        timings under ``probes``.
    """
    d_discrete = (
        cat_emb_avail.shape[1] + sol_emb_avail.shape[1] + bas_emb_avail.shape[1]
    )
    d_cont = bounds.shape[1] - d_discrete

    ga_times = []
    last_cand = None
    for ne in ga_n_cal:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            cand, _ = optimize_acqf_mixed_GA(
                acq_function=acqf, bounds=bounds, q=q, n_evals_ga=ne, pop_size=pop,
                cat_emb_avail=cat_emb_avail, sol_emb_avail=sol_emb_avail,
                bas_emb_avail=bas_emb_avail, cat_is_salt_flags=cat_is_salt_flags,
                batch_constraints=batch_constraints, fixed_dims=fixed_dims,
            )
        ga_times.append(time.perf_counter() - t0)
        last_cand = cand

    s_ga = (ga_times[1] - ga_times[0]) / (ga_n_cal[1] - ga_n_cal[0])
    ga_intercept = ga_times[0] - s_ga * ga_n_cal[0]

    out = {
        "s_ga": s_ga,
        "ga_intercept": ga_intercept,
        "probes": {"ga": list(zip(ga_n_cal, ga_times))},
    }

    if calibrate_refine:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            optimize_acqf_continuous(
                acq_function=acqf, bounds=bounds,
                X_discrete=last_cand[:, d_cont:], batch_constraints=None,
                fixed_dims=fixed_dims, num_restarts=refine_nr_cal,
                raw_samples=raw_samples,
            )
        c1 = (time.perf_counter() - t0) / refine_nr_cal
        out["c1"] = c1
        out["probes"]["refine"] = {"nr": refine_nr_cal, "c1": c1}

    return out


def estimate_runtime(
    calib: Dict[str, float],
    *,
    n_evals: int,
    q: int,
    partition: Tuple[int, ...],
    num_restarts: int,
    repeats: int,
    sequential: bool,
    refine: bool,
) -> Dict[str, float]:
    """Compose a runtime estimate from calibration constants and the settings.

    Args:
        calib: Output of :func:`calibrate` (needs ``s_ga``; ``c1`` if ``refine``).
        n_evals: Configured GA evaluations per GA call.
        q: Batch size.
        partition: Temperature partition (for grouping counts).
        num_restarts: L-BFGS restarts per grouping.
        repeats: ``GA_OPTIMIZATION_REPEATS``.
        sequential: Whether the sequential optimizer is used.
        refine: Whether a continuous refine runs (False for scope expansion).

    Returns:
        A dict with ``t_ga``, ``t_refine``, ``t_total`` (seconds), plus the
        ``n_groupings`` and ``n_ga_calls`` used.
    """
    n_ga_calls = q if sequential else 1
    t_ga = repeats * n_ga_calls * calib["s_ga"] * n_evals

    t_refine = 0.0
    n_groupings = 0
    if refine:
        n_batch, n_seq = grouping_counts(q, partition)
        n_groupings = n_seq if sequential else n_batch
        t_refine = repeats * n_groupings * num_restarts * calib["c1"]

    return {
        "t_ga": t_ga,
        "t_refine": t_refine,
        "t_total": t_ga + t_refine,
        "n_groupings": n_groupings,
        "n_ga_calls": n_ga_calls,
    }


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def format_eta(label: str, est: Dict[str, float], jitter: float = 0.12) -> str:
    """Render a one-line ETA with a +/- machine-jitter band and a breakdown.

    Args:
        label: Prefix (e.g. ``"Batch generation"``).
        est: Output of :func:`estimate_runtime`.
        jitter: Fractional machine-speed band to show (default ~12%).

    Returns:
        A formatted string ready to print.
    """
    total = est["t_total"]
    lo, hi = total * (1 - jitter), total * (1 + jitter)
    parts = [f"GA ~{_fmt(est['t_ga'])} ({est['n_ga_calls']} call(s))"]
    if est["t_refine"] > 0:
        parts.append(
            f"refine ~{_fmt(est['t_refine'])} "
            f"({est['n_groupings']} groupings)"
        )
    return (
        f"  Estimated {label} time: ~{_fmt(total)} "
        f"({_fmt(lo)}-{_fmt(hi)})  [{', '.join(parts)}]"
    )
