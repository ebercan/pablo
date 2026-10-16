"""Configuration constants for the run_1 PlanAhead scope-extension benchmark sweep."""

from benchmark_problems.synthetic import Gallagher101, LunacekBiRastrigin, Rastrigin


BENCHMARK_CLASS_MAP = {
    "LunacekBiRastrigin": LunacekBiRastrigin,
    "Gallagher101": Gallagher101,
    "Rastrigin": Rastrigin,
}

N_CONTINUOUS_DIMS = 4
N_DISCRETE_DIMS = 22
TOTAL_DIM = N_CONTINUOUS_DIMS + N_DISCRETE_DIMS

DEPENDENT_BLOCK_SIZES = (18, 2, 2)
DEPENDENT_SOURCE_CARDINALITIES = (225, 14, 8)

SEED_LIST_1 = [42, 69, 420, 123132, 6534745, 243234, 41231, 5232, 7454]
SEED_LIST_2 = [19, 31, 41]
INITIAL_AVAILABLE_SOURCE_A_SEEDS = [1, 3, 3123]

PLAN_AHEAD_STRATEGIES = ("fantasized", "myopic")

MYOPIC_OPERATING_MODE = "myopic"
FANTASIZED_OPERATING_MODES = ("only_terminal", "averaged", "layer_averaged")
ALL_OPERATING_MODES = (MYOPIC_OPERATING_MODE, *FANTASIZED_OPERATING_MODES)

OPERATING_MODE_TO_STRATEGY = {
    MYOPIC_OPERATING_MODE: "myopic",
    "only_terminal": "fantasized",
    "averaged": "fantasized",
    "layer_averaged": "fantasized",
}

Q_BATCH_SIZE = 6
Q_EXPAND_SIZE = 1
N_BO_BATCHES = 70
N_STARTING_POINTS = 47
N_OPTIMIZE_ACQF_REPETITIONS = 1
QMC_SAMPLES = 1024

# Fixed GA budget for run_1 (no GA setting sweep).
N_EVALS_GA = 2000
POP_SIZE = 35

# Availability model for the first dependent source block (225 choices).
INITIAL_AVAILABLE_SOURCE_A_COUNT = 85
SOURCE_A_AVAILABILITY_DELAY = 4
LOOKAHEAD_HORIZON = 4
NUM_FANTASIES = [12, 4, 2, 1]

DEFAULT_BATCH_CONSTRAINTS = {0: 1}

INPUTS_FILENAME = "inputs.json"
RESULTS_DIRNAME = "results"
LOGS_DIRNAME = "logs"
QUEUE_DIRNAME = "queue_state"
RUN_STATE_DIRNAME = "run_state"

RESULTS_DIRNAME_BY_OPERATING_MODE = {
    "myopic": "results_myopic",
    "only_terminal": "results_only_terminal",
    "averaged": "results_averaged",
    "layer_averaged": "results_layer_averaged",
}

QUEUE_DIRNAME_BY_OPERATING_MODE = {
    "myopic": "queue_state_myopic",
    "only_terminal": "queue_state_only_terminal",
    "averaged": "queue_state_averaged",
    "layer_averaged": "queue_state_layer_averaged",
}

JOB_NAME_PREFIX_BY_OPERATING_MODE = {
    "myopic": "plan_ahead_bo_myopic_cpu",
    "only_terminal": "plan_ahead_bo_only_terminal_cpu",
    "averaged": "plan_ahead_bo_averaged_cpu",
    "layer_averaged": "plan_ahead_bo_layer_averaged_cpu",
}

CPU_PARTITION_BY_OPERATING_MODE = {
    "myopic": "medium",
    "only_terminal": "long",
    "averaged": "long",
    "layer_averaged": "long",
}

TIME_LIMIT_BY_OPERATING_MODE = {
    "myopic": "24:00:00",
    "only_terminal": "7-00:00:00",
    "averaged": "7-00:00:00",
    "layer_averaged": "7-00:00:00",
}


def get_strategy_for_operating_mode(operating_mode: str) -> str:
    """Resolve strategy label for one operating mode.

    Args:
        operating_mode: One of the run_1 operating modes in ``ALL_OPERATING_MODES``.

    Returns:
        Strategy name ("myopic" or "fantasized").

    Raises:
        ValueError: If operating_mode is unknown.
    """
    if operating_mode not in OPERATING_MODE_TO_STRATEGY:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return OPERATING_MODE_TO_STRATEGY[operating_mode]


def get_results_dirname_for_operating_mode(operating_mode: str) -> str:
    """Resolve per-mode results directory name.

    Args:
        operating_mode: One of the run_1 operating modes in ``ALL_OPERATING_MODES``.

    Returns:
        Directory name where results for this mode are written.

    Raises:
        ValueError: If operating_mode is unknown.
    """
    if operating_mode not in RESULTS_DIRNAME_BY_OPERATING_MODE:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return RESULTS_DIRNAME_BY_OPERATING_MODE[operating_mode]


def get_queue_dirname_for_operating_mode(operating_mode: str) -> str:
    """Resolve per-mode queue_state directory name.

    Args:
        operating_mode: One of the run_1 operating modes in ``ALL_OPERATING_MODES``.

    Returns:
        Directory name where queue locks/state are stored for this mode.

    Raises:
        ValueError: If operating_mode is unknown.
    """
    if operating_mode not in QUEUE_DIRNAME_BY_OPERATING_MODE:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return QUEUE_DIRNAME_BY_OPERATING_MODE[operating_mode]


def get_job_name_prefix_for_operating_mode(operating_mode: str) -> str:
    """Resolve SLURM job-name prefix for one operating mode.

    Args:
        operating_mode: One of the run_1 operating modes in ``ALL_OPERATING_MODES``.

    Returns:
        SLURM job-name prefix.

    Raises:
        ValueError: If operating_mode is unknown.
    """
    if operating_mode not in JOB_NAME_PREFIX_BY_OPERATING_MODE:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return JOB_NAME_PREFIX_BY_OPERATING_MODE[operating_mode]


def get_cpu_partition_for_operating_mode(operating_mode: str) -> str:
    """Resolve default CPU partition for one operating mode.

    Args:
        operating_mode: One of the run_1 operating modes in ``ALL_OPERATING_MODES``.

    Returns:
        CPU partition name.

    Raises:
        ValueError: If operating_mode is unknown.
    """
    if operating_mode not in CPU_PARTITION_BY_OPERATING_MODE:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return CPU_PARTITION_BY_OPERATING_MODE[operating_mode]


def get_time_limit_for_operating_mode(operating_mode: str) -> str:
    """Resolve default SLURM time limit for one operating mode.

    Args:
        operating_mode: One of the run_1 operating modes in ``ALL_OPERATING_MODES``.

    Returns:
        SLURM walltime string.

    Raises:
        ValueError: If operating_mode is unknown.
    """
    if operating_mode not in TIME_LIMIT_BY_OPERATING_MODE:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(ALL_OPERATING_MODES)}, got {operating_mode!r}"
        )
    return TIME_LIMIT_BY_OPERATING_MODE[operating_mode]
