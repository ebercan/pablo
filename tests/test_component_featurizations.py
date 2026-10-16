"""The GROVER and SOAP featurizations of the solvent and base libraries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.components import COMPONENT_PCA_SOURCES
from pablo.utils.feature_assembly import assemble_features

LIBRARIES = {"solvent": paths.SOLVENTS_PARQUET, "base": paths.BASES_PARQUET}
COMPONENT_DIRS = COMPONENT_PCA_SOURCES + tuple(f"{n}_pca" for n in COMPONENT_PCA_SOURCES)


def _kind(name: str) -> str:
    return name.split("_", 1)[0]


def _load(name: str) -> tuple[pd.DataFrame, dict]:
    feat_dir = paths.FEAT_DIR / name
    return (
        pd.read_parquet(feat_dir / "descriptors.parquet"),
        json.loads((feat_dir / "metadata.json").read_text()),
    )


@pytest.mark.parametrize("name", COMPONENT_DIRS)
def test_covers_the_whole_library_with_finite_values(name):
    """Indexed by every library id, finite, and as wide as the metadata says."""
    df, meta = _load(name)
    library = pd.read_parquet(LIBRARIES[_kind(name)])
    assert df.index.name == f"{_kind(name)}_id"
    assert sorted(df.index) == sorted(library["id"])
    assert np.isfinite(df.to_numpy()).all()
    assert df.shape[1] == meta["n_features"]


@pytest.mark.parametrize("name", COMPONENT_DIRS)
def test_no_two_library_members_share_a_row(name):
    """Every member is distinguishable, including K/Na and Rb/Cs salts of one anion."""
    X = _load(name)[0].to_numpy()
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    assert dist.min() > 1e-6


@pytest.mark.parametrize("source", COMPONENT_PCA_SOURCES)
def test_pca_metadata_points_at_its_source(source):
    """A reduction records its source, width, knee provenance and retained variance."""
    df, meta = _load(f"{source}_pca")
    assert meta["source_featurization"] == source
    assert meta["source_n_features"] == _load(source)[0].shape[1]
    assert 0 < meta["n_features"] < len(df)
    assert 0 < meta["variance_retained"] <= 1
    assert isinstance(meta["knee_from_fallback"], bool)
    assert ("none_location" in meta) == (_kind(source) == "base")


@pytest.mark.parametrize("feat", ["grover_pca", "soap_pca"])
def test_assembles_on_the_solvent_and_base_axes(feat):
    """The PCA-reduced solvent and base levels load through assemble_features, and rows run without base have a zero base block."""
    cell = assemble_features(
        cat_featurization="grover_pca",
        solv_featurization=feat,
        base_featurization=feat,
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
    )
    assert len(cell.feature_dims["solv"]) == _load(f"solvent_{feat}")[0].shape[1]
    assert len(cell.feature_dims["base"]) == _load(f"base_{feat}")[0].shape[1]
    experiments = pd.read_parquet(paths.EXPERIMENTS_PARQUET).dropna(subset=["yield_pct"])
    no_base = torch.as_tensor((experiments["base"] == "no").to_numpy())
    assert no_base.any()
    assert (cell.X[no_base][:, cell.feature_dims["base"]] == 0).all()
