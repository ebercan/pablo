"""The crossed ``cross_1`` grid: enumeration, cost ordering, run folders, submitter and the zeroed base block."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.cross_validation import scheduler, submit_mechthild
from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import (
    build_run_cells,
    configurations,
    sweep_membership,
)
from pablo.utils.feature_assembly import assemble_features

CROSSED = config.CROSSED_GRID_1
ELIMINATION = config.ELIMINATION_GRID
CELLS = build_run_cells(CROSSED)


def _config_of(cell: dict) -> dict[str, str]:
    (inner,) = cell["inner_cells"]
    merged = {**cell, **inner}
    return {"cat_feat": cell["cat_feat"], **{axis: merged[axis] for axis in ELIMINATION.baseline}}


@pytest.fixture
def restore_run():
    """Put the scheduler and submitter back on the default run after a test selects another."""
    yield
    scheduler.select_run(config.DEFAULT_RUN)
    submit_mechthild.select_run(config.DEFAULT_RUN)


def test_crossed_grid_has_2520_distinct_configurations():
    assert len(CELLS) == 2520
    assert len({tuple(sorted(_config_of(c).items())) for c in CELLS.values()}) == 2520


def test_every_axis_level_appears_equally_often():
    """Each level of each axis appears in 2520 / n_levels cells; single-level axes are constant."""
    axes = {"cat_feat": CROSSED.cat_feats, **CROSSED.axes}
    for axis, levels in axes.items():
        counts = Counter(_config_of(c)[axis] for c in CELLS.values())
        assert set(counts) == set(levels), axis
        assert set(counts.values()) == {2520 // len(levels)}, axis
    fixed = {axis: levels[0] for axis, levels in CROSSED.axes.items() if len(levels) == 1}
    assert fixed == {"kernel_structure": "KS1", "mean_function": "constant", "model_class": "st_y"}


def test_crossed_cells_use_the_outer_inner_record_format():
    """One inner cell per outer cell, ids 0..n-1 matching their keys, the elimination record keys."""
    reference = next(iter(build_run_cells(ELIMINATION).values()))
    for key, cell in CELLS.items():
        assert cell["outer_cell_id"] == int(key)
        assert set(cell) == set(reference)
        assert len(cell["inner_cells"]) == 1
        assert set(cell["inner_cells"][0]) == set(reference["inner_cells"][0])
    assert sorted(map(int, CELLS)) == list(range(2520))


def test_crossed_ids_run_most_expensive_first():
    seconds = [scheduler._per_cell_seconds(CELLS[str(i)]) for i in range(len(CELLS))]
    assert all(a >= b for a, b in zip(seconds, seconds[1:]))
    assert CELLS["0"]["cat_feat"] == "unimol"


def test_elimination_grid_still_writes_the_tracked_inputs():
    """The elimination run enumerates 600 cells, byte-identical to data/grid/inputs.json."""
    cells = build_run_cells(ELIMINATION)
    assert len(cells) == 600
    assert json.dumps(cells, indent=2) == paths.INPUTS_JSON.read_text()


def test_cost_model_on_elimination_cells_is_the_probed_one_axis_time():
    """A one-axis configuration gets exactly the probed time of its level at its width."""
    for cell in build_run_cells(ELIMINATION).values():
        (inner,) = cell["inner_cells"]
        sweeps = sweep_membership({**cell, **inner}, ELIMINATION)
        label = "baseline" if len(sweeps) > 1 else f"{sweeps[0]}={({**cell, **inner})[sweeps[0]]}"
        expected = np.interp(
            scheduler._design_matrix_width(cell, inner),
            scheduler._PROBE_WIDTHS,
            scheduler._PRIMARY_SECONDS_PER_FOLD[label],
        )
        assert scheduler._primary_seconds_per_fold(cell, inner) == pytest.approx(expected, rel=1e-12)


def test_cost_model_composes_crossed_axes_multiplicatively():
    cell = next(
        c for c in CELLS.values()
        if _config_of(c)["outcome_transform"] == "probit" and _config_of(c)["prior_scheme"] == "P2"
    )
    (inner,) = cell["inner_cells"]
    width = scheduler._design_matrix_width(cell, inner)
    table = scheduler._PRIMARY_SECONDS_PER_FOLD
    at = lambda label: np.interp(width, scheduler._PROBE_WIDTHS, table[label])
    expected = at("baseline") * (at("prior_scheme=P2") / at("baseline")) * (
        at("outcome_transform=probit") / at("baseline")
    )
    assert scheduler._primary_seconds_per_fold(cell, inner) == pytest.approx(expected, rel=1e-12)


def test_cost_model_projects_the_crossed_grid():
    """Every crossed cell has a finite cost above startup; the grid totals about 616 CPU-h."""
    seconds = [scheduler._per_cell_seconds(c) for c in CELLS.values()]
    assert all(np.isfinite(s) and s > scheduler.STARTUP_SECONDS for s in seconds)
    assert sum(seconds) / 3600 == pytest.approx(616.1, abs=0.5)


def test_cost_model_refuses_an_unprobed_level():
    """A level on a probed axis that was never timed raises instead of costing as the baseline."""
    cell = {**CELLS["0"], "model_class": "icm_cs"}
    with pytest.raises(KeyError):
        scheduler._primary_seconds_per_fold(cell, cell["inner_cells"][0])


def test_estimate_runtime_runs_on_both_grids(capsys):
    scheduler.estimate_runtime(CELLS, CROSSED)
    assert "outcome_transform=probit" in capsys.readouterr().out
    scheduler.estimate_runtime(build_run_cells(ELIMINATION), ELIMINATION)
    assert "mean_function" in capsys.readouterr().out


def test_run_folders_do_not_overlap():
    """No path of the cross_1 run resolves to, or inside, an elimination-run path."""
    elimination = paths.run_paths("elimination")
    crossed = paths.run_paths("cross_1")
    own = [v for f, v in dataclasses.asdict(elimination).items() if f != "name"]
    for field, path in dataclasses.asdict(crossed).items():
        if field == "name":
            continue
        assert path.is_relative_to(paths.RUNS_DIR / "cross_1"), field
        assert all(path != p and not path.is_relative_to(p) for p in own), field
    assert elimination.inputs_json == paths.INPUTS_JSON
    assert elimination.cells_dir == paths.RESULTS_CELLS_DIR
    assert elimination.queue_dir == paths.QUEUE_DIR


def test_every_run_grid_has_folders():
    assert set(config.RUN_GRIDS) == {"elimination", "cross_1", "monroe_m3", "kernel_mean_1", "product_1"}
    assert config.RUN_GRIDS["cross_1"] is CROSSED
    assert config.RUN_GRIDS["monroe_m3"] is config.MONROE_M3_GRID
    assert config.RUN_GRIDS["kernel_mean_1"] is config.KERNEL_MEAN_1_GRID
    assert config.RUN_GRIDS["product_1"] is config.PRODUCT_1_GRID


def test_scheduler_select_run_moves_every_path(restore_run):
    scheduler.select_run("cross_1")
    run = paths.run_paths("cross_1")
    assert scheduler.INPUTS_PATH == run.inputs_json
    assert scheduler.QUEUE_DIR == run.queue_dir
    assert scheduler.RUN_STATE_DIR == run.run_state_dir
    assert scheduler.RESULTS_DIR == run.cells_dir
    assert scheduler.JOB_NAME_PREFIX == "feat_kernel_scan_cross_1"
    assert scheduler.SPLITS_PATH == paths.CV_SPLITS_JSON


def test_array_script_targets_the_selected_run(restore_run):
    submit_mechthild.select_run("cross_1")
    script = submit_mechthild.array_script(300, single=True)
    assert "#SBATCH -J feat_kernel_scan_cross_1\n" in script
    assert f"#SBATCH -o {paths.run_paths('cross_1').logs_dir}/worker_%A_%a.out" in script
    assert "scheduler worker --single --run cross_1\n" in script
    assert submit_mechthild.script_path().name == ".submit_array_cross_1.sh"

    submit_mechthild.select_run("elimination")
    script = submit_mechthild.array_script(300, single=True)
    assert "#SBATCH -J feat_kernel_scan\n" in script
    assert "scheduler worker --single\n" in script


def test_crossed_grid_rejects_a_missing_axis_or_a_repeated_level():
    axes = dict(CROSSED.axes)
    with pytest.raises(ValueError, match="must be exactly"):
        config.CrossedGrid(cat_feats=("grover_pca",), axes={k: v for k, v in axes.items() if k != "base_feat"}, note="")
    with pytest.raises(ValueError, match="distinct"):
        config.CrossedGrid(cat_feats=("grover_pca",), axes={**axes, "prior_scheme": ("P1", "P1")}, note="")


def test_crossed_configurations_are_the_product():
    assert len(configurations(CROSSED)) == int(np.prod([len(v) for v in CROSSED.axes.values()]))


def test_base_block_is_zero_wherever_base_active_is_zero():
    """Non-salt rows, including the one with a base added anyway, carry an all-zero base block."""
    experiments = pd.read_parquet(paths.EXPERIMENTS_PARQUET).dropna(subset=["yield_pct"]).reset_index(drop=True)
    assert ((experiments["base_active"] == 0) & (experiments["base"] != "no")).any()
    features = assemble_features(
        cat_featurization="grover_pca",
        solv_featurization="OHE",
        base_featurization="thermo",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )
    X = features.X.numpy() if hasattr(features.X, "numpy") else np.asarray(features.X)
    base = X[:, features.feature_dims["base"]]
    inactive = experiments["base_active"].to_numpy() == 0
    assert np.all(base[inactive] == 0)
    assert np.all(np.any(base[~inactive] != 0, axis=1))
