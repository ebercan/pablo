"""The P1/P2/P3 lengthscale-prior schemes: granularity and prior location.

Each scheme pairs a partition granularity with the ``D`` the Hvarfner
``log sqrt(D)`` location resolves to. The pairing is the whole point, so these
tests check both halves of each scheme and that a mismatched pair is refused
rather than half-applied.
"""

from __future__ import annotations

import math

import pytest
import torch

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import (
    BLOCKS,
    PARTITION_MODES,
    build_partition,
    make_kernel_from_config,
)

FEATURE_DIMS = {"cat": [0, 1, 2, 3, 4, 5], "solv": [6, 7], "base": [8, 9, 10], "cont": [12, 13, 14, 15]}
ACTIVE_FLAG_DIM = 11
BLOCK_WIDTHS = {b: len(d) for b, d in FEATURE_DIMS.items()}
D_TOTAL = sum(BLOCK_WIDTHS.values())


def _lengthscale_prior_locations(kernel) -> list[float]:
    """Location of each lengthscale prior, in block order."""
    return [
        float(torch.as_tensor(item[2].loc).flatten()[0])
        for item in kernel.named_priors()
        if "lengthscale" in item[0]
    ]


def _lengthscale_widths(kernel) -> list[int]:
    """Number of lengthscale scalars each block kernel carries, in block order."""
    return [
        int(torch.as_tensor(p).numel())
        for name, p in kernel.named_parameters()
        if "raw_lengthscale" in name
    ]


def _expected_location(width: int) -> float:
    return math.log(math.sqrt(width)) + config.LENGTHSCALE_LOGNORMAL_LOC_SHIFT


def test_every_scheme_names_valid_modes():
    """PRIOR_SCHEMES only references modes the partition builder and priors know."""
    from pablo.featurization_and_kernel_scan.core.priors import LENGTHSCALE_DIM_MODES

    assert set(config.PRIOR_SCHEMES) == {"P1", "P2", "P3"}
    for name, scheme in config.PRIOR_SCHEMES.items():
        assert scheme.partition_mode in PARTITION_MODES, name
        assert scheme.lengthscale_dim_mode in LENGTHSCALE_DIM_MODES, name
    assert config.DEFAULT_PRIOR_SCHEME in config.PRIOR_SCHEMES


@pytest.mark.parametrize("mode,expected", [
    ("full_ard", {"cat": [1] * 6, "solv": [1, 1], "base": [1, 1, 1], "cont": [1, 1, 1, 1]}),
    ("block_shared", {"cat": [6], "solv": [2], "base": [3], "cont": [4]}),
    ("legacy", {"cat": [6], "solv": [2], "base": [3], "cont": [1, 1, 1, 1]}),
])
def test_partition_modes_group_as_documented(mode, expected):
    """Each mode produces the documented group sizes and covers every column."""
    partition = build_partition(FEATURE_DIMS, mode=mode)
    assert {b: [len(g) for g in gs] for b, gs in partition.items()} == expected
    for block in BLOCKS:
        flat = [d for group in partition[block] for d in group]
        assert sorted(flat) == sorted(FEATURE_DIMS[block])


def test_unknown_partition_mode_is_rejected():
    """A typo'd mode fails instead of silently falling back to the legacy recipe."""
    with pytest.raises(ValueError, match="Unknown partition mode"):
        build_partition(FEATURE_DIMS, mode="isotropic")


def test_build_partition_default_is_unchanged():
    """The no-argument call still returns the recipe archived runs were fit under."""
    assert build_partition(FEATURE_DIMS) == build_partition(FEATURE_DIMS, mode="legacy")


def test_p1_is_full_ard_under_one_global_location():
    """P1: one lengthscale per column, every block at log sqrt(D_total)."""
    kernel = make_kernel_from_config("KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme="P1")
    assert _lengthscale_widths(kernel) == [BLOCK_WIDTHS[b] for b in BLOCKS]
    locations = _lengthscale_prior_locations(kernel)
    assert len(locations) == len(BLOCKS)
    for location in locations:
        assert location == pytest.approx(_expected_location(D_TOTAL))


def test_p2_is_full_ard_under_per_block_locations():
    """P2: same granularity as P1, but each block at its own log sqrt(D_block)."""
    kernel = make_kernel_from_config("KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme="P2")
    assert _lengthscale_widths(kernel) == [BLOCK_WIDTHS[b] for b in BLOCKS]
    assert _lengthscale_prior_locations(kernel) == pytest.approx(
        [_expected_location(BLOCK_WIDTHS[b]) for b in BLOCKS]
    )


def test_p3_is_one_lengthscale_per_block():
    """P3: ARD off — one lengthscale per block, at its own log sqrt(D_block)."""
    kernel = make_kernel_from_config("KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme="P3")
    assert _lengthscale_widths(kernel) == [1, 1, 1, 1]
    assert _lengthscale_prior_locations(kernel) == pytest.approx(
        [_expected_location(BLOCK_WIDTHS[b]) for b in BLOCKS]
    )


def test_p1_and_p2_differ_only_in_prior_location():
    """The two ARD schemes are the same model under a different prior."""
    p1 = make_kernel_from_config("KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme="P1")
    p2 = make_kernel_from_config("KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme="P2")
    assert _lengthscale_widths(p1) == _lengthscale_widths(p2)
    assert _lengthscale_prior_locations(p1) != pytest.approx(_lengthscale_prior_locations(p2))


def test_unknown_scheme_is_rejected():
    """A scheme name outside the registry fails loudly."""
    with pytest.raises(ValueError, match="Unknown prior scheme"):
        make_kernel_from_config("KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme="P4")


def test_partition_contradicting_the_scheme_is_rejected():
    """A P1 scheme handed a block-shared partition is an error, not a silent half-apply.

    Granularity and prior location are set in two different places (the cell
    assembler and the kernel builder). If only one of them follows the scheme,
    the fit runs and the results table still says "P1".
    """
    with pytest.raises(ValueError, match="granularity"):
        make_kernel_from_config(
            "KS1", FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52",
            partition=build_partition(FEATURE_DIMS, mode="block_shared"),
            prior_scheme="P1",
        )


@pytest.mark.parametrize("scheme", ["P1", "P2", "P3"])
@pytest.mark.parametrize("ks", ["KS1", "KS2"])
def test_every_scheme_builds_a_finite_kernel(scheme, ks):
    """All schemes build under the additive and product structures alike."""
    kernel = make_kernel_from_config(
        ks, FEATURE_DIMS, ACTIVE_FLAG_DIM, "matern52", prior_scheme=scheme
    ).to(torch.float64)
    K = kernel(torch.rand(9, 16, dtype=torch.float64)).to_dense()
    assert torch.isfinite(K).all()


@pytest.mark.parametrize("scheme", ["P1", "P2", "P3"])
def test_end_to_end_single_fold_fit_under_each_scheme(scheme):
    """One real fold fits under each scheme with the partition built to match."""
    from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import fit_eval
    from pablo.utils.feature_assembly import assemble_features

    cell = assemble_features(
        cat_featurization="interpretable_steric",
        solv_featurization="KT",
        base_featurization="thermo",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )
    cell.partition = build_partition(
        cell.feature_dims, mode=config.PRIOR_SCHEMES[scheme].partition_mode
    )
    n = cell.X.shape[0]
    result = fit_eval(
        cell,
        train_idx=list(range(n - 8)),
        test_idx=list(range(n - 8, n)),
        kernel_structure="KS1",
        outcome_transform="raw",
        model_class="st_y",
        base_kernel="matern52",
        mean_function="type_linear",
        n_restarts=1,
        max_iter=40,
        prior_scheme=scheme,
    )
    assert result.status == "ok", result
    assert result.rmse is not None and result.rmse > 0
