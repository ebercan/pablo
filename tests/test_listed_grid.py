"""The listed ``monroe_m3`` grid and the fixed-component-count PCA reductions it uses."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.cross_validation.generate_inputs import (
    build_run_cells,
    configurations,
)
from pablo.featurization_and_kernel_scan.features.dim_reduction import (
    is_reduced,
    reduced_name,
    source_name,
)

M3 = config.MONROE_M3_GRID
CELLS = build_run_cells(M3)


def _config_of(cell: dict) -> dict[str, str]:
    (inner,) = cell["inner_cells"]
    merged = {**cell, **inner}
    return {axis: merged[axis] for axis in config.ELIMINATION_GRID.baseline}


def test_m3_is_every_catalyst_level_under_every_listed_configuration():
    """30 cells: each of the 3 catalyst levels once under each of the 10 configurations."""
    assert len(CELLS) == 30
    pairs = {(c["cat_feat"], tuple(sorted(_config_of(c).items()))) for c in CELLS.values()}
    assert pairs == {
        (cat, tuple(sorted(cfg.items()))) for cat in M3.cat_feats for cfg in M3.configs
    }
    assert sorted(int(k) for k in CELLS) == list(range(30))
    assert all(int(k) == c["outer_cell_id"] for k, c in CELLS.items())


def test_m3_configurations_are_the_cross_1_top_ten():
    """The listed configurations equal cross_1 gem ranks 1-10, in rank order."""
    scores = pd.read_parquet(paths.run_paths("cross_1").gem_scores_parquet)
    scores = scores.sort_values("gem", ascending=False).head(10)
    assert (scores["cat_feat"] == "grover_pca").all()
    top = [
        {axis: row[axis] for axis in config.ELIMINATION_GRID.baseline}
        for _, row in scores.iterrows()
    ]
    assert configurations(M3) == top


def test_m3_carbene_free_levels_carry_no_form_onehot():
    assert not any(c["append_form_onehot_to_cat"] for c in CELLS.values())


def test_listed_grid_rejects_a_missing_axis_or_a_repeated_configuration():
    cfg = dict(M3.configs[0])
    with pytest.raises(ValueError, match="exactly the keys"):
        config.ListedGrid(
            cat_feats=("grover_pca",),
            configs=({k: v for k, v in cfg.items() if k != "base_feat"},),
            note="",
        )
    with pytest.raises(ValueError, match="listed twice"):
        config.ListedGrid(cat_feats=("grover_pca",), configs=(cfg, dict(cfg)), note="")


def test_listed_grid_axes_are_the_levels_in_use():
    assert M3.axes["solv_feat"] == ("Hansen", "grover_pca", "Gutmann")
    assert M3.axes["kernel_structure"] == ("KS1",)


def test_fixed_count_reduction_names_resolve_to_their_source():
    assert reduced_name("monroe_precat", 8) == "monroe_precat_pca8"
    assert reduced_name("monroe_precat") == "monroe_precat_pca"
    for name in ("monroe_precat_pca8", "monroe_precat_pca", "solvent_grover_pca"):
        assert is_reduced(name)
    assert source_name("monroe_precat_pca8") == "monroe_precat"
    assert source_name("solvent_grover_pca") == "solvent_grover"
    assert not is_reduced("monroe_precat") and not is_reduced("grover_pcax")


def test_monroe_pca8_is_the_knee_reduction_truncated_at_8():
    """The stored 8-PC block equals the first 8 columns of the 15-PC knee block."""
    pca8 = pd.read_parquet(paths.FEAT_DIR / "monroe_precat_pca8" / "descriptors.parquet")
    knee = pd.read_parquet(paths.FEAT_DIR / "monroe_precat_pca" / "descriptors.parquet")
    meta = json.loads((paths.FEAT_DIR / "monroe_precat_pca8" / "metadata.json").read_text())
    assert pca8.shape == (161, 8) and meta["n_features"] == 8 and meta["knee"] == knee.shape[1]
    assert list(pca8.columns) == list(knee.columns[:8])
    assert pca8.index.equals(knee.index)
    np.testing.assert_allclose(pca8.to_numpy(), knee.to_numpy()[:, :8], atol=1e-10)
