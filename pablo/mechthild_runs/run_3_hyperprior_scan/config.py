"""Configuration for the run_3 hyperprior scan.

Grid: 4 hyperprior configs × 2 cat featurizations × 8 inner configs × 135 folds.

Outer axis (= workers): hyperprior × cat_feat = 4 × 2 = 8 outer cells.
Inner axis: outcome_transform(2) × model_class(2) × mean_function(2) = 8 configs.
Fixed: KS4, matern52, solvent=physical, base=structured.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HyperPriorConfig:
    """One set of GP hyper-prior values to monkey-patch into the scan config.

    Attributes:
        name: Identifier of the hyper-prior configuration.
        ls_scale: σ of the LogNormal lengthscale prior, written to
            ``LENGTHSCALE_LOGNORMAL_SCALE``.
        ls_floor: Lower bound of the lengthscale ``GreaterThan`` constraint,
            written to ``LENGTHSCALE_FLOOR``; ``None`` removes that attribute
            from the scan config.
        dirichlet_conc: Concentration of the symmetric Dirichlet prior on the
            additive kernel weights, written to ``CONCENTRATION_DIRICHLET``.
    """

    name: str
    ls_scale: float
    ls_floor: float | None
    dirichlet_conc: float


HYPERPRIOR_CONFIGS: list[HyperPriorConfig] = [
    HyperPriorConfig(name="baseline",   ls_scale=2.5, ls_floor=0.1, dirichlet_conc=2.0),
    HyperPriorConfig(name="tighter_ls", ls_scale=2.0, ls_floor=0.1, dirichlet_conc=2.0),
    HyperPriorConfig(name="wider_ls",   ls_scale=3.0, ls_floor=0.1, dirichlet_conc=2.0),
    HyperPriorConfig(name="strong_reg", ls_scale=2.5, ls_floor=0.1, dirichlet_conc=5.0),
]

HYPERPRIOR_BY_NAME: dict[str, HyperPriorConfig] = {
    hp.name: hp for hp in HYPERPRIOR_CONFIGS
}

CAT_FEATURIZATIONS: list[tuple[str, bool]] = [
    ("interpretable", False),
    ("grover", False),
]

SOLV_FEATURIZATION: str = "physical"
BASE_FEATURIZATION: str = "structured"

KERNEL_STRUCTURE: str = "KS4"
BASE_KERNEL: str = "matern52"

OUTCOME_TRANSFORMS: list[str] = ["raw", "logit"]
MODEL_CLASSES: list[str] = ["st_y", "lcm_cs"]
MEAN_FUNCTIONS: list[str] = ["constant", "type_linear"]

N_INNER_CONFIGS: int = (
    len(OUTCOME_TRANSFORMS) * len(MODEL_CLASSES) * len(MEAN_FUNCTIONS)
)

FOLD_TOTAL: int = 135

SLURM_PARTITION: str = "medium"
SLURM_CPUS_PER_TASK: int = 1
SLURM_MEMORY: str = "8G"
SLURM_TIME_LIMIT: str = "12:00:00"
SLURM_EMAIL: str = "enisbaris@mpi-magdeburg.mpg.de"
SLURM_MAX_CONCURRENT_WORKERS: int = 8
SLURM_POLL_INTERVAL_SECONDS: int = 30
