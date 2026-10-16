"""The elimination grid: its enumeration, its levels, the stored PCA reductions, and its cost model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import BASE_KERNELS
from pablo.featurization_and_kernel_scan.core.models import (
    _MEAN_BUILDERS,
    AcyloinCondensationModel,
)
from pablo.featurization_and_kernel_scan.cross_validation import scheduler
from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import (
    INNER_AXES,
    OUTER_AXES,
    catalyst_spec,
    check_cat_feats,
    configurations,
    enumerate_cells,
    select_cat_feats,
    sweep_membership,
)
from pablo.featurization_and_kernel_scan.features.dim_reduction import (
    is_reduced,
    reduced_name,
    source_name,
)
from pablo.utils.outcome_transforms import OUTCOME_TRANSFORM_NAMES

GRID = config.ELIMINATION_GRID
INTERPRETABLE = {s.name for s in config.CATALYST_SETS["interpretable"]}
CELLS = enumerate_cells(GRID.cat_feats, GRID)


def _config_of(cell: dict) -> dict[str, str]:
    (inner,) = cell["inner_cells"]
    return {axis: ({**cell, **inner})[axis] for axis in GRID.baseline}


# ── the stored PCA reductions ────────────────────────────────────────────────


def test_every_reducible_catalyst_has_a_stored_reduction():
    """All 24 non-interpretable featurizations have a built <name>_pca/."""
    reducible = [s.name for s in config.CATALYST_SETS["all"] if s.name not in INTERPRETABLE]
    assert len(reducible) == 24
    missing = [
        n for n in reducible
        if not (paths.FEAT_DIR / reduced_name(n) / "descriptors.parquet").exists()
    ]
    assert not missing, f"PCA reductions not built: {missing}"


def test_reductions_are_narrower_than_their_source_and_self_describing():
    """Each reduction records its source, its knee, and the variance it kept."""
    for cat_feat in filter(is_reduced, GRID.cat_feats):
        cell_dir = paths.FEAT_DIR / cat_feat
        df = pd.read_parquet(cell_dir / "descriptors.parquet")
        meta = json.loads((cell_dir / "metadata.json").read_text())
        source = pd.read_parquet(paths.FEAT_DIR / source_name(cat_feat) / "descriptors.parquet")

        assert meta["source_featurization"] == source_name(cat_feat)
        assert meta["n_features"] == df.shape[1] == len(meta["feature_cols"])
        assert list(df.columns) == meta["feature_cols"]
        assert df.shape[0] == source.shape[0] == 161
        assert 0 < df.shape[1] < source.shape[1]
        assert 0.0 < meta["variance_retained"] <= 1.0
        assert np.isfinite(df.to_numpy(dtype=float)).all()


def test_interpretable_sets_have_no_reduced_level():
    """The grid lists no interpretable _pca level, and none is on disk."""
    for name in INTERPRETABLE:
        assert not (paths.FEAT_DIR / reduced_name(name)).exists()
    assert not [c for c in GRID.cat_feats if is_reduced(c) and source_name(c) in INTERPRETABLE]


# ── enumeration ──────────────────────────────────────────────────────────────


def test_grid_has_50_catalyst_levels():
    """27 featurizations plus a _pca level for the 23 that are not interpretable."""
    assert GRID.cat_feats == config.CAT_LEVELS
    assert len(GRID.cat_feats) == len(set(GRID.cat_feats)) == 50
    assert sum(map(is_reduced, GRID.cat_feats)) == 23


def test_600_cells_one_configuration_each():
    """50 levels × 12 unique configurations, one inner cell per outer cell, ids 0..599."""
    assert len(configurations(GRID)) == 12
    assert len(CELLS) == 600
    assert list(CELLS) == [str(i) for i in range(600)]
    for cid, cell in CELLS.items():
        assert cell["outer_cell_id"] == int(cid)
        assert len(cell["inner_cells"]) == 1
        assert cell["inner_cells"][0]["inner_id"] == 0


def test_no_configuration_is_enumerated_twice():
    """Every (catalyst level, configuration) pair is unique; the baseline appears once per level."""
    keys = [(c["cat_feat"], tuple(sorted(_config_of(c).items()))) for c in CELLS.values()]
    assert len(set(keys)) == len(keys)
    baseline = tuple(sorted(GRID.baseline.items()))
    for cat_feat in GRID.cat_feats:
        assert sum(k == (cat_feat, baseline) for k in keys) == 1


def test_every_cell_differs_from_the_baseline_on_at_most_one_swept_axis():
    """The grid is a union of one-axis sweeps, never a cross of two."""
    for cell in CELLS.values():
        moved = [a for a, level in GRID.baseline.items() if _config_of(cell)[a] != level]
        assert len(moved) <= 1
        assert set(moved) <= set(GRID.sweeps)


@pytest.mark.parametrize("axis, n_cells", [
    ("kernel_structure", 200), ("prior_scheme", 150),
    ("outcome_transform", 200), ("mean_function", 200),
])
def test_each_sweep_covers_every_level_of_its_axis_for_every_catalyst(axis, n_cells):
    """A sweep holds all 50 levels × all its axis levels, the baseline cell included."""
    members = [c for c in CELLS.values() if axis in sweep_membership(_config_of(c), GRID)]
    assert len(members) == n_cells
    covered = {(c["cat_feat"], _config_of(c)[axis]) for c in members}
    assert covered == {(cat, level) for cat in GRID.cat_feats for level in GRID.sweeps[axis]}


def test_sweep_membership_rejects_a_two_axis_configuration():
    """A configuration that moves two axes belongs to no sweep."""
    crossed = {**GRID.baseline, "kernel_structure": "KS2", "mean_function": "type_linear"}
    with pytest.raises(ValueError, match="in no sweep"):
        sweep_membership(crossed, GRID)


def test_unswept_axes_stay_at_the_baseline():
    """Solvent, base, base kernel and model class never move."""
    for cell in CELLS.values():
        cfg = _config_of(cell)
        for axis in set(GRID.baseline) - set(GRID.sweeps):
            assert cfg[axis] == GRID.baseline[axis]


def test_matern52_on_every_level_including_the_fingerprints():
    """No tanimoto, minmax or rbf anywhere, full-dimensional ecfp4 and map4 included."""
    assert {c["base_kernel"] for c in CELLS.values()} == {"matern52"}
    assert {"ecfp4", "map4"} <= {c["cat_feat"] for c in CELLS.values()}


def test_inputs_schema_is_unchanged():
    """Outer and inner records carry exactly the keys scheduler, concat_results and gem_scores read."""
    outer_keys = {"outer_cell_id", "cat_feat", *OUTER_AXES, "append_form_onehot_to_cat", "inner_cells"}
    for cell in CELLS.values():
        assert set(cell) == outer_keys
        assert set(cell["inner_cells"][0]) == {"inner_id", *INNER_AXES}
    assert set(OUTER_AXES) | set(INNER_AXES) == set(GRID.baseline)
    assert set(OUTER_AXES) >= set(scheduler.CELL_IDENTITY_KEYS) - {"cat_feat"}


# ── every level resolves ─────────────────────────────────────────────────────


def test_every_level_resolves_to_an_implementation():
    """Each axis level names a registered kernel, scheme, transform, mean or model class."""
    registries = {
        "kernel_structure": set(config.KERNEL_STRUCTURES),
        "prior_scheme": set(config.PRIOR_SCHEMES),
        "outcome_transform": set(OUTCOME_TRANSFORM_NAMES),
        "mean_function": set(_MEAN_BUILDERS),
        "base_kernel": set(BASE_KERNELS),
        "model_class": set(AcyloinCondensationModel.SPECS),
    }
    for axis, known in registries.items():
        levels = set(GRID.sweeps.get(axis, ())) | {GRID.baseline[axis]}
        assert levels <= known, f"{axis}: {levels - known} not implemented"
    assert {"P2", "P3"} <= set(config.PRIOR_SCHEMES)
    assert {"arcsin_sqrt", "probit"} <= set(OUTCOME_TRANSFORM_NAMES)
    assert {"block_linear", "type_quadlog"} <= set(_MEAN_BUILDERS)
    assert set(config.MEAN_FUNCTIONS) == set(_MEAN_BUILDERS)


def test_every_featurization_block_is_on_disk():
    """Every catalyst level and the baseline solvent and base blocks exist."""
    names = [*GRID.cat_feats, f"solvent_{GRID.baseline['solv_feat']}", f"base_{GRID.baseline['base_feat']}"]
    for name in names:
        assert (paths.FEAT_DIR / name / "descriptors.parquet").exists(), name


def test_form_onehot_flag_follows_the_source_featurization():
    """A reduced level carries the form-one-hot flag of the featurization it reduces."""
    for cell in CELLS.values():
        spec = catalyst_spec(cell["cat_feat"])
        assert spec.name == source_name(cell["cat_feat"])
        assert cell["append_form_onehot_to_cat"] == spec.append_form_onehot


def test_unbuilt_reduction_is_refused_at_enumeration():
    """A level naming a reduction that was never built fails before submission."""
    with pytest.raises(ValueError, match="has not been built"):
        check_cat_feats(["interpretable_pca"])


def test_unknown_level_is_refused_at_enumeration():
    """A level that resolves to no catalyst featurization fails before submission."""
    with pytest.raises(ValueError, match="not in CATALYST_SETS"):
        enumerate_cells(["not_a_featurization"], GRID)


def test_cat_set_narrows_to_one_family_in_grid_order():
    """--cat-set keeps the grid's levels whose source is in that family."""
    assert select_cat_feats(GRID, "fingerprint") == ("ecfp4", "ecfp4_pca", "map4", "map4_pca")
    assert select_cat_feats(GRID, None) == GRID.cat_feats


def test_a_sweep_must_contain_its_baseline_level():
    """A sweep that omits the baseline level is refused at definition."""
    with pytest.raises(ValueError, match="omit the baseline"):
        config.EliminationGrid(
            cat_feats=("grover",), baseline=dict(GRID.baseline),
            sweeps={"kernel_structure": ("KS2", "KS3")}, note="",
        )


# ── cost model and result identity ───────────────────────────────────────────


def test_every_cell_has_a_cost():
    """Each level of each sweep has a multiplier, so estimate-runtime projects every cell."""
    for cell in CELLS.values():
        assert scheduler._per_cell_seconds(cell) > scheduler.STARTUP_SECONDS


def test_list_schedule_is_bounded_by_the_longest_cell_and_the_mean_load():
    """Wall time lies between max(longest cell, work / slots) and their sum."""
    seconds = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    wall = scheduler._list_schedule_wall(seconds, 2)
    assert max(max(seconds), sum(seconds) / 2) <= wall <= max(seconds) + sum(seconds) / 2
    assert scheduler._list_schedule_wall(seconds, 6) == 5.0


def test_a_result_from_another_prior_scheme_is_refused():
    """Cells of the prior sweep share every outer key; the inner prior_scheme still tells them apart."""
    p1, p2 = (
        next(c for c in CELLS.values() if c["cat_feat"] == "grover" and _config_of(c) == {**GRID.baseline, "prior_scheme": p})
        for p in ("P1", "P2")
    )
    stored = {
        "outer_config": {k: v for k, v in p2.items() if k != "inner_cells"},
        "inner_cells": [{"inner_config": p2["inner_cells"][0], "folds": []}],
    }
    with pytest.raises(ValueError, match="prior_scheme"):
        scheduler._assert_result_matches_cell(stored, p1, p1["outer_cell_id"])
    scheduler._assert_result_matches_cell(stored, p2, p2["outer_cell_id"])
