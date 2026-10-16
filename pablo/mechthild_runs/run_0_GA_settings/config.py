"""Configuration constants for the run_0 GA settings benchmark sweep."""

from benchmark_problems.synthetic import (
    Ackley,
    Gallagher101,
    Levy,
    LunacekBiRastrigin,
    Michalewicz,
    Rastrigin,
)


BENCHMARK_CLASS_MAP = {
    "LunacekBiRastrigin": LunacekBiRastrigin,
    "Gallagher101": Gallagher101,
    "Michalewicz": Michalewicz,
    "Ackley": Ackley,
    "Levy": Levy,
    "Rastrigin": Rastrigin,
}

TOTAL_DIM = 33
N_CONTINUOUS_DIMS = 3
N_DISCRETE_DIMS = 30

DEPENDENT_BLOCK_SIZES = (26, 2, 2)
DEPENDENT_SOURCE_CARDINALITIES = (225, 14, 8)

SEED_LIST_1 = [42, 69, 420, 49]
SEED_LIST_2 = [19, 31, 41, 7]

GA_RATIOS = [100, 125, 150, 175]
POP_SIZES = [35, 70, 100]

Q_BATCH_SIZE = 6
N_BO_BATCHES = 100
N_STARTING_POINTS = 47
N_OPTIMIZE_ACQF_REPETITIONS = 1
QMC_SAMPLES = 1024

DEVICE_CHOICES = ("cpu", "cuda", "auto")
DEFAULT_DEVICE_REQUEST = "cpu"

# When CUDA is requested but unavailable/incompatible, run_0 jobs may continue on CPU.
ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES = True

# P100 nodes (gpu01-gpu05) have compute capability sm_60, which is below the
# minimum sm_70 supported by our torch cu128 build.  Exclude them so GPU jobs
# land only on A100 (gpua*, sm_80) and H100 (gpuh*, sm_90) nodes.
GPU_EXCLUDE_NODES = "gpu[01-05]"

DEFAULT_BATCH_CONSTRAINTS = {0: 3}

INPUTS_FILENAME = "inputs.json"
RESULTS_DIRNAME = "results"
LOGS_DIRNAME = "logs"
QUEUE_DIRNAME = "queue_state"
RUN_STATE_DIRNAME = "run_state"
