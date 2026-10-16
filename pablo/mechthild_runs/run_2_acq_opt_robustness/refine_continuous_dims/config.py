"""HPC configuration for the refine-continuous-dims sweep.

Worker ``i`` of ``WORKERS`` optimises batches
``[i · BATCHES_PER_WORKER, min((i + 1) · BATCHES_PER_WORKER, TOTAL_BATCHES))``
from ``ranked_results_merged.json`` using ``optimize_acqf_continuous``, so the
last worker gets one batch when ``TOTAL_BATCHES`` is odd.
Results are saved as individual JSON files and merged afterwards.
"""

from __future__ import annotations

WORKERS = 244
BATCHES_PER_WORKER = 2
TOTAL_BATCHES = 487

RESULTS_DIRNAME = "results"
PARTITION = "medium"
CPUS_PER_TASK = 1
MEMORY = "16G"
TIME_LIMIT = "01:00:00"
EMAIL = "enisbaris@mpi-magdeburg.mpg.de"
JOB_NAME = "run3_refine_cont"
