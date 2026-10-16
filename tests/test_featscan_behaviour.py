"""Behavioural checks on the featurization and kernel scan package.

The structural suite checks that the code imports and resolves its names. These
tests check what it computes: the registry drives real featurizers, the two
slug functions stay distinct, a stored featurization is reproduced by its
source function, and KS1-KS4 build finite covariance matrices under every base
kernel.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import (
    BASE_KERNELS,
    KS_TERMS,
    build_partition,
    make_kernel_from_config,
)
from pablo.featurization_and_kernel_scan.features import common
from pablo.featurization_and_kernel_scan.features.registry import (
    FAMILIES,
    RUNTIMES,
    build_registry,
)

KS_NAMES = ("KS1", "KS2", "KS3", "KS4", "KS5", "KS6")
"""Every structure that builds without library catalyst rows; KS7 is in test_kernel_mean_1."""


# ── registry ─────────────────────────────────────────────────────────────────


def test_registry_is_consistent():
    """Every entry has a known family/runtime, a unique name, and a callable."""
    registry = build_registry()
    assert len(registry) >= 40
    for name, entry in registry.items():
        assert entry.name == name
        assert entry.family in FAMILIES
        assert entry.runtime in RUNTIMES
        assert callable(entry.build)


def test_registry_prerequisites_exist():
    """Every ``requires`` entry names another registered featurizer."""
    registry = build_registry()
    for entry in registry.values():
        unknown = [r for r in entry.requires if r not in registry]
        assert not unknown, f"{entry.name} requires unknown step(s) {unknown}"


def test_registry_covers_the_configured_grid():
    """Every catalyst/solvent/base featurization the grid names is buildable.

    ``config`` enumerates featurization names on the CV axes; each must either
    be produced by a registry entry or already exist on disk. A name that is
    neither is a grid referring to something nothing can build.
    """
    registry = build_registry()
    produced = {out for e in registry.values() for out in e.outputs}
    expected = (
        [s.name for specs in config.CATALYST_SETS.values() for s in specs]
        + [f"solvent_{s}" for s in config.SOLVENT_FEATURIZATIONS]
        + [f"base_{b}" for b in config.BASE_FEATURIZATIONS]
    )
    unbuildable = [
        name for name in sorted(set(expected))
        if name not in produced and not (paths.FEAT_DIR / name).exists()
    ]
    assert not unbuildable, f"grid names nothing can build: {unbuildable}"


def test_all_featurization_dirs_have_descriptors():
    """Every delivered featurization directory carries a descriptors.parquet."""
    missing = [
        d.name for d in sorted(paths.FEAT_DIR.iterdir())
        if d.is_dir() and not d.name.startswith("_")
        and not (d / "descriptors.parquet").exists()
    ]
    assert not missing, f"featurizations without descriptors.parquet: {missing}"


# ── the two slug functions must not be merged ────────────────────────────────


def test_no_integer_category_columns_survive_in_any_featurization():
    """No stored featurization ships an unordered category as an integer column.

    Every kernel on the grid measures a real-valued distance over raw feature
    columns — there is no Hamming/``CategoricalKernel`` route — so an integer
    id column would be read as an ordinal axis (``thiazolylidene = 3`` three
    times as far from ``salt = 0`` as ``free_nhc = 1``). Categories must be
    one-hot before they are stored.
    """
    banned = {"cat_form_id", "ring_type_id"}
    offenders = {}
    for cell_dir in sorted(paths.FEAT_DIR.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue
        parquet = cell_dir / "descriptors.parquet"
        if not parquet.exists():
            continue
        hits = banned & set(pd.read_parquet(parquet).columns)
        if hits:
            offenders[cell_dir.name] = sorted(hits)
    assert not offenders, f"Integer category columns still stored: {offenders}"


def test_interpretable_metadata_matches_its_parquet():
    """Each interpretable table's metadata describes the columns actually stored."""
    for name in ("interpretable_steric", "interpretable_light", "interpretable", "interpretable_dG"):
        cell_dir = paths.FEAT_DIR / name
        df = pd.read_parquet(cell_dir / "descriptors.parquet")
        meta = json.loads((cell_dir / "metadata.json").read_text())
        assert list(df.columns) == meta["feature_cols"], name
        assert meta["n_features"] == df.shape[1] == len(meta["feature_cols"]), name
        assert "categorical_cols" not in meta, (
            f"{name}: the categorical_cols contract is retired; nothing reads it."
        )


def test_component_slug_and_common_slug_differ():
    """They are not interchangeable; merging them would rename one-hot columns.

    ``component_slug`` collapses runs of non-alphanumerics and preserves case,
    which the ``is_Me_THF``-style one-hot column names depend on.
    ``common.slug`` lowercases and maps each character individually.
    """
    from pablo.featurization_and_kernel_scan.features.components import component_slug

    assert component_slug("2-Me THF") == "2_Me_THF"
    assert common.slug("2-Me THF") == "2_me_thf"
    assert component_slug("MolFormer-XL") != common.slug("MolFormer-XL")


def test_onehot_column_names_match_stored_featurization():
    """Stored solvent one-hot columns still match what component_slug produces."""
    from pablo.featurization_and_kernel_scan.features.components import component_slug

    stored = pd.read_parquet(paths.FEAT_DIR / "solvent_OHE" / "descriptors.parquet")
    solvents = pd.read_parquet(paths.SOLVENTS_PARQUET)
    expected = {f"is_{component_slug(n)}" for n in solvents["name"]}
    assert set(stored.columns) == expected


# ── a featurizer still reproduces its stored output ──────────────────────────


def test_ecfp4_reproduces_stored_matrix():
    """Recomputing ECFP4 for the first 12 catalysts matches the stored parquet exactly.

    Read-only: the fingerprint is recomputed in memory and compared, so the
    delivered featurization is never rewritten.
    """
    from pablo.featurization_and_kernel_scan.features.fingerprints import (
        _ecfp4_fingerprint,
    )

    stored = pd.read_parquet(paths.FEAT_DIR / "ecfp4" / "descriptors.parquet")
    cat_ids, smiles = common.load_catalyst_library()
    assert stored.index.tolist() == cat_ids

    recomputed = np.array([_ecfp4_fingerprint(s) for s in smiles[:12]], dtype=np.float64)
    np.testing.assert_allclose(recomputed, stored.to_numpy()[:12], rtol=0, atol=0)


def test_library_loaders_agree():
    """Both catalyst library loaders return the same 161 IDs and SMILES, and every ID has a form."""
    cat_ids, smiles = common.load_catalyst_library()
    ids2, smiles2, forms = common.load_catalyst_library_with_forms()
    assert cat_ids == ids2 and smiles == smiles2
    assert len(cat_ids) == len(smiles) == 161
    assert set(forms) == set(cat_ids)


# ── kernels ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def toy_design():
    """A small synthetic design matrix with a no-base block, float64."""
    feature_dims = {
        "cat": list(range(6)), "solv": [6, 7], "base": [8, 9, 10],
        "cont": [12, 13, 14, 15],
    }
    active_flag_dim = 11
    torch.manual_seed(0)
    X = torch.rand(9, 16, dtype=torch.float64)
    X[:3, 8:11] = 0.0
    X[:3, active_flag_dim] = 0.0
    X[3:, active_flag_dim] = 1.0
    return feature_dims, active_flag_dim, X


def test_build_partition_recipe(toy_design):
    """Discrete blocks get one shared lengthscale; cont gets full ARD."""
    feature_dims, _, _ = toy_design
    part = build_partition(feature_dims)
    assert [len(g) for g in part["cat"]] == [6]
    assert [len(g) for g in part["solv"]] == [2]
    assert [len(g) for g in part["base"]] == [3]
    assert [len(g) for g in part["cont"]] == [1, 1, 1, 1]


def test_build_partition_rejects_incomplete_dims():
    """A missing block is an error, not a silently-ignored feature set."""
    with pytest.raises(ValueError):
        build_partition({"cat": [0], "solv": [1], "base": [2]})
    with pytest.raises(ValueError):
        build_partition({"cat": [0], "solv": [1], "base": [2], "cont": []})


@pytest.mark.parametrize("ks", KS_NAMES)
@pytest.mark.parametrize("base_kernel", sorted(BASE_KERNELS))
def test_every_structure_and_base_kernel_builds(ks, base_kernel, toy_design):
    """Every structure builds under all four base kernels, diag consistent."""
    feature_dims, active_flag_dim, X = toy_design
    kernel = make_kernel_from_config(
        kernel_structure=ks,
        feature_dims=feature_dims,
        active_flag_dim=active_flag_dim,
        base_kernel_type=base_kernel,
    ).to(torch.float64)

    K = kernel(X).to_dense()
    assert K.shape == (9, 9)
    assert torch.isfinite(K).all()
    torch.testing.assert_close(K.diagonal(), kernel(X, diag=True), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("ks", KS_NAMES)
def test_one_lengthscale_prior_per_block(ks, toy_design):
    """Each block's lengthscale prior is registered exactly once.

    KS3 and KS4 reference a block kernel in several additive terms. Because the
    terms are index tuples rather than nested ProductKernels, ``named_priors``
    still yields four lengthscale priors — not one per term reference.
    """
    feature_dims, active_flag_dim, _ = toy_design
    kernel = make_kernel_from_config(
        kernel_structure=ks,
        feature_dims=feature_dims,
        active_flag_dim=active_flag_dim,
        base_kernel_type="matern52",
    )
    names = [t[0] for t in kernel.named_priors()]
    assert sum("lengthscale" in n for n in names) == 4


def test_structure_term_counts():
    """KS2/KS3/KS4 carry 4, 10 and 11 additive terms."""
    assert {k: len(v) for k, v in KS_TERMS.items()} == {"KS2": 4, "KS3": 10, "KS4": 11}


def test_unknown_structure_is_rejected():
    """``KS_flat``, ``KS_pairwise`` and ``KS8`` raise ``ValueError``.

    ``KS_flat`` and ``KS_pairwise`` are named on purpose: the test fails if
    either is added to ``config.KERNEL_STRUCTURES`` or ``KernelStructure.build``.
    Keep these literals rather than generic invalid names.
    """
    for bad in ("KS_flat", "KS_pairwise", "KS8"):
        with pytest.raises(ValueError):
            make_kernel_from_config(
                kernel_structure=bad,
                feature_dims={"cat": [0], "solv": [1], "base": [2], "cont": [4]},
                active_flag_dim=3,
                base_kernel_type="matern52",
            )


def test_config_grid_axes():
    """The live grid names the seven structures and the four base kernels."""
    assert config.KERNEL_STRUCTURES == [*KS_NAMES, "KS7"]
    assert set(BASE_KERNELS) == {"rbf", "matern52", "tanimoto", "minmax"}
