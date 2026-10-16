"""Tests for ``TypeQuadLogMean``: Normalize inversion, formula, priors, guards."""

import math

import pytest
import torch

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    TypeQuadLogMean,
    _cont_column_index,
    build_mean_module,
)
from pablo.utils.feature_assembly import assemble_features

TRAIN_IDX = list(range(0, 60))
TEST_IDX = list(range(60, 65))


@pytest.fixture(scope="module")
def cell():
    return assemble_features(
        cat_featurization="interpretable",
        solv_featurization="OHE",
        base_featurization="OHE",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )


@pytest.fixture(scope="module")
def built(cell):
    """The mean module plus the fitted-free model that owns the input transform."""
    train_x = cell.X[TRAIN_IDX]
    mean = build_mean_module("type_quadlog", cell, train_x=train_x)
    builder = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=TRAIN_IDX,
        kernel_structure="KS1",
        base_kernel_type="matern52",
        outcome_transform="raw",
        lengthscale_scale=1.0,
    )
    model = builder.build("st_y", mean)
    model.train()
    model.transform_inputs(train_x)
    model.eval()
    return mean, model


def test_normalize_inversion_recovers_library_scale(cell, built):
    """The stored affine constants invert the model's learned Normalize exactly."""
    mean, model = built
    x = cell.X[TEST_IDX]
    u = model.transform_inputs(x)
    z = u.index_select(-1, mean._tl_dims) * mean._norm_span + mean._norm_offset
    torch.testing.assert_close(z, x.index_select(-1, mean._tl_dims))


def test_forward_matches_raw_scale_formula(cell, built):
    """mu equals the hand-written quadratic in raw °C and log in raw catalyst loading (molar ratio), plus the linear and offset terms."""
    import pandas as pd

    mean, model = built
    torch.manual_seed(0)
    for p in mean.parameters():
        p.data = torch.randn_like(p) * 0.2

    exp = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    temp = torch.tensor(exp["temperature"].to_numpy(float))[TEST_IDX]
    load = torch.tensor(exp["loading"].to_numpy(float))[TEST_IDX]

    z_temp = 2.0 * (temp - 25.0) / (100.0 - 25.0) - 1.0
    log_load = torch.log(load) - math.log(0.5 * (0.5 + 10.0))

    x = cell.X[TEST_IDX]
    form_id = x[:, cell.form_id_dim].long()
    coeffs = mean.c_shape + mean.d_shape[form_id]
    expected = (
        coeffs[:, 0] * z_temp**2 + coeffs[:, 1] * z_temp + coeffs[:, 2] * log_load
    )

    u = model.transform_inputs(x)
    expected = expected + (u.index_select(-1, mean._rest_dims) * mean.a_rest).sum(-1)
    expected = expected + mean.b_cont
    expected = expected + (u.index_select(-1, mean._disc_dims) * mean.a_disc[form_id]).sum(-1)
    expected = expected + mean.b_disc[form_id]

    torch.testing.assert_close(mean(u), expected)


def test_partial_pooling_prior_is_tighter_than_global(built):
    """Per-type deviations carry a prior config.MEAN_TYPE_DEVIATION_FRAC as wide."""
    mean, _ = built
    priors = {name: prior for name, _, prior, _, _ in mean.named_priors()}
    ratio = priors["d_shape_prior"].scale / priors["c_shape_prior"].scale
    assert math.isclose(float(ratio), config.MEAN_TYPE_DEVIATION_FRAC)


def test_shape_and_dims(cell, built):
    """Parameter shapes and the dims split match the cell layout."""
    mean, model = built
    assert mean.c_shape.shape == (3,)
    assert mean.d_shape.shape == (config.N_PRECATALYST_FORM_TYPES, 3)
    assert mean._rest_dims.tolist() == [
        d
        for d in cell.feature_dims["cont"]
        if d not in (_cont_column_index(cell, "temperature"), _cont_column_index(cell, "loading"))
    ]
    assert mean(model.transform_inputs(cell.X)).shape == (cell.X.shape[0],)


def test_missing_train_x_raises(cell):
    """The builder refuses to guess the Normalize bounds."""
    with pytest.raises(ValueError, match="training inputs"):
        build_mean_module("type_quadlog", cell)
