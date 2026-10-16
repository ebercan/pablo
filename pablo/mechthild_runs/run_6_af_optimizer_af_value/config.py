"""Configuration for run_6: pool-greedy against the GA by acquisition value.

One fitted GP, one acquisition function, two optimizers, seven walltime tiers, three
repetitions. Every worker optimizes the same qLogNEI with one optimizer at one tier and
one seed, and scores its batch with a common qLogNEI on fixed QMC base samples.

**Model.** The kernel_mean_1 anchor of the ``grover_pca`` catalyst level: ``st_y``,
catalyst ``grover_pca``, solvent ``Hansen``, base ``grover_pca``, Matérn-5/2, probit
targets, prior scheme P2, KS1, constant mean, fitted on every row of the featscan
``experiments.parquet``.

**Acquisition.** qLogNEI on the probit-scale GP output (identity objective), the
training inputs as ``X_baseline``, q = 6, temperature partition (2, 2, 2), time pinned
at 20 h, availability from the ``Next Batch`` column of ``pablo/nhc_doe.xlsx``.

**Optimizers.**
  * ``ga`` — ``optimize_acqf_mixed_GA`` followed by ``optimize_acqf_continuous``, the
    pipeline of ``pablo.main.generate_batch``. A tier sets the GA evaluations and the
    refine restarts; population and raw samples are the production values.
  * ``pool_greedy`` — ``optimize_acqf_mixed_pool_greedy``; a tier multiplies the pool
    sizes of :data:`PG_BASE` by its scale.

**Work packaging.** The tiers form the blocks of :data:`TIER_BLOCKS`; the flat index
runs through the blocks in order, and inside a block decodes optimizer (slowest) → tier
→ rep (fastest). Block 0 (tiers 0-4) holds indices 0-29, block 1 (tiers 5-6) indices
30-41, so ``N_WORKERS = 2 × 7 × 3 = 42`` and adding a block leaves every earlier index
and result file valid.
"""

from __future__ import annotations

Q_BATCH: int = 6
TEMPERATURE_PARTITION: tuple[int, ...] = (2, 2, 2)

CAT_FEATURIZATION: str = "grover_pca"
SOLVENT_FEATURIZATION: str = "Hansen"
BASE_FEATURIZATION: str = "grover_pca"
BASE_KERNEL: str = "matern52"
MODEL_CLASS: str = "st_y"
KERNEL_STRUCTURE: str = "KS1"
MEAN_FUNCTION: str = "constant"
OUTCOME_TRANSFORM: str = "probit"
PRIOR_SCHEME: str = "P2"

AVAILABILITY_COLUMN: str = "Next Batch"

# Featscan continuous block order is [fa_ratio, loading, temperature, time], each
# mapped to [-1, 1] by the library bounds in continuous/metadata.json.
D_CONTINUOUS: int = 4
TEMPERATURE_DIM: int = 2
TIME_DIM: int = 3
TIME_PINNED_H: float = 20.0

N_MC_SAMPLES: int = 128
SCORE_QMC_SAMPLES: int = 4096
SCORE_QMC_SEED: int = 12345

OPTIMIZERS: tuple[str, ...] = ("ga", "pool_greedy")
N_TIERS: int = 7
N_REPS: int = 3
TIER_BLOCKS: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3, 4), (5, 6))
N_WORKERS: int = sum(len(OPTIMIZERS) * len(block) * N_REPS for block in TIER_BLOCKS)

TIER_TARGET_MINUTES: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 25.0, 60.0, 120.0)

# Production GA population and refine raw samples (pablo.main.GA_SETTINGS[1] and
# LBFGS_SETTINGS[1]). One refine restart costs ~65 s on a worker node, so the
# production 20 restarts (~22 min) do not fit the short tiers; restarts grow per tier
# up to the production 20.
GA_POP_SIZE: int = 500
LBFGS_RAW_SAMPLES: int = 1024
GA_REFINE_RESTARTS: tuple[int, ...] = (1, 2, 4, 6, 10, 20, 20)

# GA evaluations per tier. Tiers 0-4 from `main_variant calibrate` on node005 (1.149 ms
# per GA evaluation, 64.7 s per refine restart); tiers 5-6 from the tier-4 runs (1.28 ms
# per evaluation, 58 s per restart): 2,400 s and 6,000 s of GA after a ~20 min refine.
GA_N_EVALS: tuple[int, ...] = (49500, 149500, 298000, 446500, 743000, 1875000, 4687500)

# Pool-greedy settings at scale 1. A tier multiplies n_pool, n_override, n_local and
# n_refine by its scale; the iteration counts and swap rounds stay fixed.
PG_BASE: dict[str, int] = {
    "n_pool": 1024,
    "n_override": 256,
    "n_local": 256,
    "n_incumbents": 8,
    "n_refine": 32,
    "refine_maxiter": 30,
    "n_swap_rounds": 1,
    "polish_maxiter": 50,
}
PG_SCALED_KEYS: tuple[str, ...] = ("n_pool", "n_override", "n_local", "n_refine")

# Pool-greedy scales per tier. Tiers 0-4 from the same calibration as GA_N_EVALS (160 s
# at scale 100, 636 s at scale 400); tiers 5-6 from the tier-4 runs (2.0 s per scale
# unit on a shared node, ~6.4 MB of peak memory per unit).
PG_TIERS: tuple[float, ...] = (74.7, 188.3, 377.5, 566.8, 945.3, 1800.0, 3600.0)

CALIBRATION_GA_PROBE_EVALS: int = 100000
CALIBRATION_REFINE_PROBE_RESTARTS: tuple[int, int] = (1, 3)
CALIBRATION_PG_PROBE_SCALES: tuple[float, float] = (100.0, 400.0)

assert len(TIER_TARGET_MINUTES) == N_TIERS
assert sorted(t for block in TIER_BLOCKS for t in block) == list(range(N_TIERS))
assert len(GA_N_EVALS) == N_TIERS and len(GA_REFINE_RESTARTS) == N_TIERS and len(PG_TIERS) == N_TIERS


def pool_greedy_setting(scale: float) -> dict[str, int]:
    """Pool-greedy keyword arguments at ``scale`` times the scaled sizes of :data:`PG_BASE`."""
    if scale <= 0:
        raise ValueError(f"pool-greedy scale must be positive, got {scale}")
    return {
        k: max(1, round(v * scale)) if k in PG_SCALED_KEYS else v
        for k, v in PG_BASE.items()
    }


def optimizer_setting(optimizer: str, tier: int) -> dict[str, int]:
    """Settings of ``optimizer`` at ``tier``; raises on an unknown optimizer or tier."""
    if not 0 <= tier < N_TIERS:
        raise ValueError(f"tier must be in [0, {N_TIERS}), got {tier}")
    if optimizer == "ga":
        return {
            "n_evals_ga": GA_N_EVALS[tier],
            "pop_size": GA_POP_SIZE,
            "lbfgs_num_restarts": GA_REFINE_RESTARTS[tier],
            "lbfgs_raw_samples": LBFGS_RAW_SAMPLES,
        }
    if optimizer == "pool_greedy":
        return pool_greedy_setting(PG_TIERS[tier])
    raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")


SLURM_JOB_NAME: str = "afopt_r6"
SLURM_CPUS_PER_TASK: int = 1

# (partition, time limit, memory) per tier. `short` caps the wall time at 2 h, so the
# 2 h tier runs on `medium`. One array covers tiers that share all three values.
SLURM_BY_TIER: dict[int, tuple[str, str, str]] = {
    **{t: ("short", "00:45:00", "16G") for t in range(5)},
    5: ("short", "01:45:00", "24G"),
    6: ("medium", "03:30:00", "40G"),
}
assert set(SLURM_BY_TIER) == set(range(N_TIERS))

SLURM_CALIBRATE_PARTITION: str = "short"
SLURM_CALIBRATE_TIME_LIMIT: str = "01:00:00"
SLURM_CALIBRATE_MEMORY: str = "16G"
SLURM_MAX_CONCURRENT_WORKERS: int = 30
SLURM_EMAIL: str = "enisbaris@mpi-magdeburg.mpg.de"
