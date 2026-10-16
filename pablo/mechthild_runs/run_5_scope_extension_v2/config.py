"""Configuration constants for the run_5 scope-expansion benchmark (v2)."""

from pathlib import Path

from benchmark_problems.synthetic import Ackley, Gallagher101, LunacekBiRastrigin


BENCHMARK_CLASS_MAP = {
    "Ackley": Ackley,
    "Gallagher101": Gallagher101,
    "LunacekBiRastrigin": LunacekBiRastrigin,
}

# The surrogate is the featurization-scan model
# grover_pca / Hansen / grover_pca / matern52 / probit / P2 / KS1 / constant, and the
# search space has its dimensionality: 4 continuous + 16 catalyst + 4 solvent + 4 base.
CAT_FEATURIZATION = "grover_pca"
SOLVENT_FEATURIZATION = "Hansen"
BASE_FEATURIZATION = "grover_pca"
BASE_KERNEL = "matern52"
OUTCOME_TRANSFORM = "probit"
PRIOR_SCHEME = "P2"
KERNEL_STRUCTURE = "KS1"
MEAN_FUNCTION = "constant"

# Point layout [cont | cat | solv | base]; continuous dims in raw units, bounds from
# the featurization scan's continuous/metadata.json.
CONTINUOUS_COLUMNS = ("fa_ratio", "loading", "temperature", "time")
N_CONTINUOUS_DIMS = len(CONTINUOUS_COLUMNS)
CAT_DIM = 16
SOLVENT_DIM = 4
BASE_DIM = 4
TOTAL_DIM = N_CONTINUOUS_DIMS + CAT_DIM + SOLVENT_DIM + BASE_DIM

# Lab constraints: temperature partition (2, 2, 2) over q = 6 and time pinned at 20 h.
TEMPERATURE_DIM = CONTINUOUS_COLUMNS.index("temperature")
FIXED_DIM_INDEX = CONTINUOUS_COLUMNS.index("time")
FIXED_DIM_VALUE = 20.0
DEFAULT_BATCH_CONSTRAINTS = {TEMPERATURE_DIM: (2, 2, 2)}

# Synthetic landscape per (benchmark, seed). Column j of a point is mapped from its
# library range to the window [o_j, o_j + LANDSCAPE_WINDOW] of the benchmark bounds
# (as fractions), reversed with probability 1/2, with
# o_j ~ U(0.25 - LANDSCAPE_OFFSET_HALF_WIDTH, 0.25 + LANDSCAPE_OFFSET_HALF_WIDTH).
# Ackley's optimum (bounds centre) then sits in the middle half of every column's range.
LANDSCAPE_WINDOW = 0.5
LANDSCAPE_OFFSET_HALF_WIDTH = 0.125
# Synthetic yield = (f - f_min) / (f_max - f_min) over the attainable space (library rows,
# continuous bounds, time 20 h): f on every (cat, solv, base) row at RANGE_SOBOL_POINTS
# continuous points, the RANGE_REFINE best and worst refined by L-BFGS-B. Observed
# yield = synthetic yield + N(0, sd²), sd the pooled replicate SD of raw yield
# (featscan NOISE_EVIDENCE[("Y", "raw")]), censored to [0, 1] like a measured yield.
RANGE_SOBOL_POINTS = 128
RANGE_REFINE = 16
RANGE_REFINE_MAXITER = 200

SEEDS = list(range(20))

RANDOM_OPERATING_MODE = "random"
MYOPIC_OPERATING_MODE = "myopic"
FANTASIZED_OPERATING_MODES = ("only_terminal", "averaged", "layer_averaged")
ALL_OPERATING_MODES = (
    RANDOM_OPERATING_MODE,
    MYOPIC_OPERATING_MODE,
    *FANTASIZED_OPERATING_MODES,
)
EXPANSION_CRITERIA = (MYOPIC_OPERATING_MODE, *FANTASIZED_OPERATING_MODES)
"""Acquisition criteria every iteration scores the expansion picks under."""

OPERATING_MODE_TO_STRATEGY = {
    RANDOM_OPERATING_MODE: "random",
    MYOPIC_OPERATING_MODE: "myopic",
    "only_terminal": "fantasized",
    "averaged": "fantasized",
    "layer_averaged": "fantasized",
}

Q_BATCH_SIZE = 6
Q_EXPAND_SIZE = 1
N_BO_BATCHES = 100
QMC_SAMPLES = 1024

# optimize_acqf_mixed_pool_greedy settings. The batch and every tree node use a
# (2, 2, 2)-grouped greedy pass; every expansion pick shares one setting.
BATCH_POOL_SIZE = 4096
TREE_POOL_SIZE = 4096
TREE_SWAP_ROUNDS = 0
TREE_POLISH_MAXITER = 20
TREE_QMC_SAMPLES = 128
EXPANSION_POOL_SIZE = 4096
EXPANSION_REFINE = 0

# An ordered catalyst is dead for SOURCE_A_AVAILABILITY_DELAY batches and usable
# in the next one; the tree's LOOKAHEAD_HORIZON depths cover the dead batches.
SOURCE_A_AVAILABILITY_DELAY = 4
LOOKAHEAD_HORIZON = 4
NUM_FANTASIES = [12, 4, 2, 1]

# real_data_optimizer_benchmark.py: pool-greedy against the production GA
# (pablo.main.GA_SETTINGS) on yield-stratified subsets of the NHC dataset. Both
# batches are scored by one qLogNEI with its own fixed QMC base samples.
REAL_DATA_FRACTIONS = (1.0, 0.7, 0.3)
REAL_DATA_N_SEEDS = 10
REAL_DATA_N_YIELD_BINS = 5
REAL_DATA_POOL_SIZE = 65536
REAL_DATA_REFINE = 128
REAL_DATA_POLISH_MAXITER = 200
REAL_DATA_QMC_SAMPLES = 1024
REAL_DATA_SCORE_QMC_SAMPLES = 4096
REAL_DATA_SCORE_QMC_SEED = 12345
REAL_DATA_RESULTS_FILENAME = "real_data_optimizer_results.csv"

RUN_DIR = Path(__file__).parent
INPUTS_PATH = RUN_DIR / "inputs.json"
LOGS_DIRNAME = "logs"
RUN_STATE_DIRNAME = "run_state"

# SLURM: one array task runs one job on 1 CPU. A 100-iteration job takes ≈ 13.5-23 h on a
# full worker node and peaks at ≈ 3.5 GB (663 rows); `long` has no cap, 2 days leave room for the tail.
SLURM_PARTITION = "long"
SLURM_TIME_LIMIT = "2-00:00:00"
SLURM_MEMORY = "8G"
SLURM_CPUS_PER_TASK = 1
SLURM_EMAIL = "enisbaris@mpi-magdeburg.mpg.de"
SLURM_MAX_CONCURRENT_TASKS = 512


def _check_mode(operating_mode: str) -> str:
    if operating_mode not in ALL_OPERATING_MODES:
        raise ValueError(
            f"operating_mode must be one of {list(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return operating_mode


def get_strategy_for_operating_mode(operating_mode: str) -> str:
    """Strategy label of one operating mode.

    Args:
        operating_mode: One of ``ALL_OPERATING_MODES``.

    Returns:
        ``"random"``, ``"myopic"`` or ``"fantasized"`` (the three fantasized modes).

    Raises:
        ValueError: If ``operating_mode`` is not in ``ALL_OPERATING_MODES``.
    """
    return OPERATING_MODE_TO_STRATEGY[_check_mode(operating_mode)]


def combination_operating_mode(combination: dict) -> str:
    """Operating mode of one ``inputs.json`` combination, checked against its strategy.

    Args:
        combination: One entry of ``inputs.json``.

    Returns:
        The combination's ``operating_mode``.

    Raises:
        ValueError: If the mode is unknown or the stored ``strategy`` is not the mode's.
    """
    mode = _check_mode(str(combination["operating_mode"]))
    if combination["strategy"] != OPERATING_MODE_TO_STRATEGY[mode]:
        raise ValueError(
            f"Combination has strategy {combination['strategy']!r} but operating_mode {mode!r} "
            f"(expected strategy {OPERATING_MODE_TO_STRATEGY[mode]!r})"
        )
    return mode


def results_dir(operating_mode: str) -> Path:
    """Results folder of one operating mode, ``RUN_DIR / results_<mode>``.

    Raises:
        ValueError: If ``operating_mode`` is not in ``ALL_OPERATING_MODES``.
    """
    return RUN_DIR / f"results_{_check_mode(operating_mode)}"


def queue_dir(operating_mode: str) -> Path:
    """Queue folder of one operating mode, ``RUN_DIR / queue_state_<mode>``.

    Raises:
        ValueError: If ``operating_mode`` is not in ``ALL_OPERATING_MODES``.
    """
    return RUN_DIR / f"queue_state_{_check_mode(operating_mode)}"


def job_name(operating_mode: str) -> str:
    """SLURM job name of one operating mode's workers, ``r5_<mode>``.

    Raises:
        ValueError: If ``operating_mode`` is not in ``ALL_OPERATING_MODES``.
    """
    return f"r5_{_check_mode(operating_mode)}"
