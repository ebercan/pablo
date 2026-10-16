"""Canonical filesystem anchors for the featurization and kernel scan package.

Every path in this package resolves from ``PKG_ROOT``, which is anchored on this
module's own location rather than on a caller's ``__file__``. Modules therefore
keep working when they move between subpackages.

Layout::

    data/
      libraries/       catalysts, solvents, bases, experiments parquets
      featurizations/  one directory per featurization, the delivered matrices
      xtb/             inputs/ (per-catalyst geometries), outputs/, and
                       components/ (per-solvent / per-base geometries)
      weights/         downloaded model weights too large to regenerate
      grid/            inputs.json, cv_splits.json, noise_estimate.json, queue/
    results/           the elimination run's per-cell JSONs and aggregates
    runs/<name>/       every other run: inputs.json, queue/, cells/, logs/, aggregates
    archive/           frozen runs, one self-contained directory each

``run_paths(name)`` returns the per-run locations: the ``elimination`` run lives in
``data/grid/`` and ``results/``, every other run in its own ``runs/<name>/``. The
fold definitions ``data/grid/cv_splits.json`` are shared by every run.

There are no fallback locations outside this layout;
``test_no_legacy_data_layout_left_behind`` fails if a data directory outside it
(for example ``data/xtb_inputs/``) exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PKG_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PKG_ROOT.parents[1]

DATA_DIR: Path = PKG_ROOT / "data"
LIB_DIR: Path = DATA_DIR / "libraries"
FEAT_DIR: Path = DATA_DIR / "featurizations"
INTERMEDIATE_DIR: Path = FEAT_DIR / "_intermediate"
XTB_DIR: Path = DATA_DIR / "xtb"
XTB_INPUTS_DIR: Path = XTB_DIR / "inputs"
XTB_OUTPUTS_DIR: Path = XTB_DIR / "outputs"
XTB_COMPONENTS_DIR: Path = XTB_DIR / "components"
WEIGHTS_DIR: Path = DATA_DIR / "weights"
GRID_DIR: Path = DATA_DIR / "grid"
QUEUE_DIR: Path = GRID_DIR / "queue"
RUN_STATE_DIR: Path = QUEUE_DIR / "run_state"

RESULTS_DIR: Path = PKG_ROOT / "results"
RESULTS_CELLS_DIR: Path = RESULTS_DIR / "cells"
RESULTS_LOGS_DIR: Path = RESULTS_DIR / "logs"
TIMING_PROBE_DIR: Path = RESULTS_DIR / "timing_probe"
NOISE_STRUCTURE_DIR: Path = RESULTS_DIR / "noise_structure"
ARCHIVE_DIR: Path = PKG_ROOT / "archive"

CATALYSTS_PARQUET: Path = LIB_DIR / "catalysts.parquet"
SOLVENTS_PARQUET: Path = LIB_DIR / "solvents.parquet"
BASES_PARQUET: Path = LIB_DIR / "bases.parquet"
EXPERIMENTS_PARQUET: Path = LIB_DIR / "experiments.parquet"

INPUTS_JSON: Path = GRID_DIR / "inputs.json"
CV_SPLITS_JSON: Path = GRID_DIR / "cv_splits.json"
NOISE_ESTIMATE_JSON: Path = GRID_DIR / "noise_estimate.json"
SCAN_AGGREGATE_PARQUET: Path = RESULTS_DIR / "scan_aggregate.parquet"

RUNS_DIR: Path = PKG_ROOT / "runs"
ELIMINATION_RUN: str = "elimination"


@dataclass(frozen=True)
class RunPaths:
    """The files and folders one CV run owns.

    Attributes:
        name: The run name, a key of ``config.RUN_GRIDS``.
        inputs_json: The enumerated grid.
        queue_dir: ``pending/``, ``running/`` and ``done/`` lock folders.
        run_state_dir: Per-worker heartbeat JSONs.
        cells_dir: ``result_<outer_id>.json`` per outer cell.
        logs_dir: SLURM worker logs.
        aggregate_parquet: ``concat_results`` output.
        gem_scores_parquet: ``gem_scores`` output.
    """

    name: str
    inputs_json: Path
    queue_dir: Path
    run_state_dir: Path
    cells_dir: Path
    logs_dir: Path
    aggregate_parquet: Path
    gem_scores_parquet: Path


def run_paths(name: str) -> RunPaths:
    """The locations of one run.

    Args:
        name: ``"elimination"`` for the run in ``data/grid/`` and ``results/``;
            any other name resolves under ``runs/<name>/``.

    Returns:
        The run's paths. Nothing is created.
    """
    if name == ELIMINATION_RUN:
        return RunPaths(
            name=name,
            inputs_json=INPUTS_JSON,
            queue_dir=QUEUE_DIR,
            run_state_dir=RUN_STATE_DIR,
            cells_dir=RESULTS_CELLS_DIR,
            logs_dir=RESULTS_LOGS_DIR,
            aggregate_parquet=SCAN_AGGREGATE_PARQUET,
            gem_scores_parquet=RESULTS_DIR / "gem_scores.parquet",
        )
    root = RUNS_DIR / name
    return RunPaths(
        name=name,
        inputs_json=root / "inputs.json",
        queue_dir=root / "queue",
        run_state_dir=root / "queue" / "run_state",
        cells_dir=root / "cells",
        logs_dir=root / "logs",
        aggregate_parquet=root / "scan_aggregate.parquet",
        gem_scores_parquet=root / "gem_scores.parquet",
    )
