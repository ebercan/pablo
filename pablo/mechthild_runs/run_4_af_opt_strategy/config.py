"""Configuration for run_4: AF-optimizer comparison over models x batch-schemes x tiers.

The sweep compares **two GP setups** crossed with **four q=6 batch decompositions**,
each at **six GA-budget tiers**, repeated ``N_REPS`` times.

**Models** (:data:`MODELS`). Both use ``grover / map / map`` featurization and train on
percent-scale targets (see :data:`TARGET_PERCENT_SCALE`).
  * ``st_y``   — the validated positive-AF baseline: plain ``SingleTaskGP`` on yield
    only (Vanilla ARD-RBF, constant mean, raw transform), qLogNEI on the yield output.
    Reproduces ``pablo.main.train_gp_model``.
  * ``lcm_ys`` — rank-2 LCM ``KroneckerMultiTaskGP`` on (Yield, Selectivity) with KS1;
    qLogNEI optimizes the **Yield** task only (objective slices ``samples[..., 0]``,
    no C·S product).

**Batch-schemes** (:data:`BATCH_SCHEMES`). Four ways to decompose the q=6 batch into
sub-batches: ``[6]`` (full batch), ``[3, 3]``, ``[2, 2, 2]``, ``[1]*6`` (fully
sequential). At sub-batch ``i`` the GA frees ``steps[i]`` new candidates with the
already-chosen ones held fixed; the (2, 2, 2) temperature partition is enforced on the
full q=6 batch regardless of the decomposition.

**Tiers** (:data:`TIER_SETTINGS_BY_SCHEME`). Six GA-budget rungs per scheme. Each worker
runs all six tiers for its one ``(rep, model, scheme)`` cell. The per-eval cost differs
by scheme, so each scheme has its own ladder; the rungs below are PLACEHOLDERS to be
filled from the scheme-aware ``main_variant.calibrate`` output against
:data:`TIER_TARGET_HOURS` before submission.

**Work packaging.** One worker = one ``(rep, model, scheme)`` cell; the six tiers are an
inner loop, all six results written into one ``result_<id>.json``.
``N_WORKERS = N_REPS * len(MODELS) * len(BATCH_SCHEMES) = 25 * 2 * 4 = 200``. The flat
queue index decodes via ``main_variant._decode``.
"""

from __future__ import annotations

Q_BATCH: int = 6
TEMPERATURE_PARTITION: tuple[int, ...] = (2, 2, 2)

FIXED_DIMS: dict[int, float] = {3: 1.0}

# Shared featurization (both models): grover catalyst + MAP2 solvent/base. Built via
# pablo.featurization_and_kernel_scan.models.AcyloinCondensationModel and fit on
# pablo/featurization_and_kernel_scan/data/experiments.parquet.
CAT_FEATURIZATION: str = "grover"
SOLVENT_FEATURIZATION: str = "map"
BASE_FEATURIZATION: str = "map"
BASE_KERNEL: str = "matern52"

# The two GP setups crossed against the batch-scheme axis. Each entry is the full
# per-model spec the AcyloinCondensationModel factory needs; featurization and
# base_kernel are shared above.
MODELS: list[dict[str, str]] = [
    {
        "name": "st_y",
        "model_class": "st_y",
        "kernel_structure": "Vanilla",
        "mean_function": "constant",
        "outcome_transform": "raw",
    },
    {
        "name": "lcm_ys",
        "model_class": "lcm_ys",
        "kernel_structure": "KS1",
        "mean_function": "type_linear",
        "outcome_transform": "raw",
    },
]

# Four q=6 decompositions. ``steps`` lists the per-sub-batch increments (must sum to
# Q_BATCH); the GA frees ``steps[i]`` new candidates at sub-batch i.
BATCH_SCHEMES: list[dict] = [
    {"name": "full_batch", "steps": [6]},
    {"name": "split_3_3", "steps": [3, 3]},
    {"name": "split_2_2_2", "steps": [2, 2, 2]},
    {"name": "sequential", "steps": [1, 1, 1, 1, 1, 1]},
]

# qLogNEI Monte-Carlo sample count.
N_MC_SAMPLES: int = 128

# The featscan target path returns yield (and selectivity) as fractions in [0, 1]
# (``yield_pct / 100``). qLogEI scales linearly with the outcome units, so fraction
# targets push log-EI strongly negative (a constant log(100) offset) while leaving
# the argmax unchanged. We rescale targets to percent at fit time so absolute AF
# values are positive and match pablo.main.train_gp_model / REPRODUCE.py, which
# train on percent-scale yield directly.
TARGET_PERCENT_SCALE: float = 100.0

N_REPS: int = 25
N_TIERS: int = 6

# Refinement is deliberately minimal and FIXED across all tiers so ~95% of each tier's
# runtime is GA, which picks the discrete catalyst x solvent x base set.
LBFGS_NUM_RESTARTS: int = 3
LBFGS_RAW_SAMPLES: int = 128


def _tier(ga_n_evals: int, ga_pop_size: int) -> dict[str, int]:
    """One GA-budget rung (GA budget + the fixed minimal L-BFGS refine)."""
    return {
        "ga_n_evals": ga_n_evals,
        "ga_pop_size": ga_pop_size,
        "lbfgs_num_restarts": LBFGS_NUM_RESTARTS,
        "lbfgs_raw_samples": LBFGS_RAW_SAMPLES,
    }


# PLACEHOLDER six-rung ladders, one list per scheme name. ``ga_n_evals`` is the GA
# budget PER SUB-BATCH GA CALL. These vary population at a fixed ~150-generation budget
# as a starting point; the scheme-aware ``calibrate`` command replaces ``ga_n_evals`` so
# the summed six-tier worker wall time hits TIER_TARGET_HOURS. DO NOT submit before
# calibrating.
_PLACEHOLDER_GENS: int = 150
_PLACEHOLDER_POPS: dict[str, tuple[int, ...]] = {
    "full_batch": (300, 450, 600, 750, 900, 1100),
    "split_3_3": (250, 375, 500, 625, 750, 900),
    "split_2_2_2": (250, 375, 500, 625, 750, 900),
    "sequential": (150, 250, 350, 450, 550, 650),
}
TIER_SETTINGS_BY_SCHEME: dict[str, list[dict[str, int]]] = {
    name: [_tier(_PLACEHOLDER_GENS * pop, pop) for pop in pops]
    for name, pops in _PLACEHOLDER_POPS.items()
}

assert {s["name"] for s in BATCH_SCHEMES} == set(TIER_SETTINGS_BY_SCHEME)
assert all(sum(s["steps"]) == Q_BATCH for s in BATCH_SCHEMES)
assert all(len(ladder) == N_TIERS for ladder in TIER_SETTINGS_BY_SCHEME.values())

# Per-cell workers: idx decodes to (rep, model, scheme); each runs all N_TIERS tiers.
N_WORKERS: int = N_REPS * len(MODELS) * len(BATCH_SCHEMES)

# Per-TIER wall-clock targets used by `calibrate` to recommend ga_n_evals; the summed
# six values are the per-worker wall-time budget the SLURM time limit must cover.
TIER_TARGET_HOURS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

SLURM_PARTITION: str = "medium"
SLURM_CPUS_PER_TASK: int = 2
SLURM_MEMORY: str = "32G"
# Each worker now runs all six tiers (~6x a single-cell worker). Set from the summed
# TIER_TARGET_HOURS reported by `calibrate` before submitting (currently a placeholder).
SLURM_TIME_LIMIT: str = "24:00:00"
SLURM_MAX_CONCURRENT_WORKERS: int = 200
SLURM_EMAIL: str = "enisbaris@mpi-magdeburg.mpg.de"
SLURM_POLL_INTERVAL_SECONDS: int = 120
