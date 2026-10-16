"""Tests for the product-structured model classes ``logsum_cy`` and ``condprod_cy``
and for the corrected ``*_cs`` yield product."""

import math

import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import build_partition
from pablo.featurization_and_kernel_scan.core.models import AcyloinCondensationModel
from pablo.featurization_and_kernel_scan.core.product_models import (
    ColumnScaledMean,
    ConversionScaledKernel,
    FixedMixingKernel,
    ScaleOnlyStandardize,
    append_column,
    make_condprod_model,
    make_logsum_model,
)
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import (
    _product_cs_yield_samples,
    fit_eval,
)
from pablo.utils.feature_assembly import assemble_features

DTYPE = torch.float64


@pytest.fixture(scope="module")
def cell():
    """The product_1 cell: grover_pca catalyst, Hansen solvent, grover_pca base, P2 partition."""
    features = assemble_features(
        cat_featurization="grover_pca",
        solv_featurization="Hansen",
        base_featurization="grover_pca",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )
    features.partition = build_partition(
        features.feature_dims, mode=config.PRIOR_SCHEMES["P2"].partition_mode
    )
    return features


def _rbf(dims: list[int], lengthscale: float, outputscale: float) -> ScaleKernel:
    kernel = ScaleKernel(RBFKernel(active_dims=dims)).to(DTYPE)
    kernel.base_kernel.lengthscale = lengthscale
    kernel.outputscale = outputscale
    return kernel


def test_scale_only_standardize_divides_without_centring():
    """Targets are divided by their standard deviation; the mean is kept."""
    Y = torch.tensor([[1.0], [2.0], [4.0]], dtype=DTYPE)
    tf = ScaleOnlyStandardize(m=1)
    Y_tf, _ = tf(Y)
    std = Y.std()
    assert torch.allclose(Y_tf, Y / std)
    assert torch.all(tf.means == 0)
    tf.eval()
    Y_back, _ = tf.untransform(Y_tf)
    assert torch.allclose(Y_back, Y)


def test_fixed_mixing_kernel_gives_the_covariance_of_c_y_s():
    """Blocks equal Cov(g_C, g_C + g_S, g_S) for independent g_C, g_S."""
    torch.manual_seed(0)
    X = torch.rand(5, 2, dtype=DTYPE)
    k_c, k_s = _rbf([0, 1], 0.4, 1.3), _rbf([0, 1], 0.9, 0.7)
    kernel = FixedMixingKernel(k_c, k_s, task_column=2).to(DTYPE)
    flat = torch.cat([append_column(X, t) for t in (0, 1, 2)])
    K = kernel(flat).to_dense().detach()
    kc, ks = k_c(X).to_dense().detach(), k_s(X).to_dense().detach()
    block = {(i, j): K[5 * i:5 * i + 5, 5 * j:5 * j + 5] for i in range(3) for j in range(3)}
    zero = torch.zeros_like(kc)
    expected = {
        (0, 0): kc, (0, 1): kc, (0, 2): zero,
        (1, 1): kc + ks, (1, 2): ks, (2, 2): ks,
    }
    for (i, j), value in expected.items():
        assert torch.allclose(block[(i, j)], value), (i, j)
        assert torch.allclose(block[(j, i)], value.T), (j, i)
    assert torch.allclose(kernel(flat, diag=True).detach(), torch.diagonal(K))


def test_conversion_scaled_kernel_and_mean():
    """k = c c' k_s and m = c μ, with c read from the appended column."""
    torch.manual_seed(1)
    X = torch.rand(4, 2, dtype=DTYPE)
    c = torch.tensor([[0.1], [0.5], [0.9], [0.3]], dtype=DTYPE)
    Xa = append_column(X, c)
    base = _rbf([0, 1], 0.5, 2.0)
    K = ConversionScaledKernel(base, column=2)(Xa).to_dense().detach()
    expected = c @ c.T * base(X).to_dense().detach()
    assert torch.allclose(K, expected)
    inner = ConstantMean().to(DTYPE)
    inner.constant.data.fill_(0.4)
    mean = ColumnScaledMean(inner, column=2)(Xa).detach()
    assert torch.allclose(mean, 0.4 * c.squeeze(-1))


@pytest.mark.parametrize("ks", ["KS1", "KS2", "KS3", "KS4"])
def test_scan_kernels_ignore_an_appended_column(cell, ks):
    """The scan's kernels read only their own columns, so the task / conversion column is invisible to them."""
    builder = AcyloinCondensationModel(
        cell, list(range(20)), ks, "matern52", "raw", config.LENGTHSCALE_LOGNORMAL_SCALE, prior_scheme="P2"
    )
    kernel = builder._build_kernel().to(DTYPE)
    X = cell.X[:20]
    Xa = append_column(X, torch.rand(20, 1, dtype=DTYPE))
    assert torch.allclose(kernel(X).to_dense().detach(), kernel(Xa).to_dense().detach())


def test_condprod_selectivity_posterior_is_the_heteroscedastic_gp_on_s():
    """The c c' k_s model of Y equals a GP on S = Y/C with noise τ²/c², by closed form."""
    torch.manual_seed(2)
    n, d = 12, 2
    X = torch.rand(n, d, dtype=DTYPE)
    c = 0.05 + 0.9 * torch.rand(n, 1, dtype=DTYPE)
    s_true = 0.3 + 0.2 * torch.sin(4 * X[:, :1])
    Y = c * s_true + 0.01 * torch.randn(n, 1, dtype=DTYPE)
    conversion_gp = SingleTaskGP(X, c)
    model = make_condprod_model(
        conversion_gp, X, c, Y, _rbf([0, 1], 0.6, 1.5),
        selectivity_mean=None, form_id_dim=-1,
        yield_noise_prior=GammaPrior(2.0, 2.0), conversion_transform="raw",
    )
    gp = model.selectivity_gp
    gp.mean_module.inner.constant.data.fill_(1.2)
    gp.likelihood.noise = torch.tensor([0.03], dtype=DTYPE)
    gp.eval()
    X_test = torch.rand(5, d, dtype=DTYPE)
    with torch.no_grad():
        post = gp.posterior(append_column(X_test, 1.0))
        Xn_train = gp.input_transform(model.selectivity_train_x)
        Xn_test = gp.input_transform(append_column(X_test, 1.0))
        k_s = gp.covar_module.base_kernel
        sy = gp.outcome_transform.stdvs.squeeze()
        s_obs = (Y / sy / c).squeeze(-1)
        K = k_s(Xn_train).to_dense() + torch.diag(0.03 / c.squeeze(-1) ** 2)
        k_star = k_s(Xn_test, Xn_train).to_dense()
        mean_ref = 1.2 + k_star @ torch.linalg.solve(K, s_obs - 1.2)
        var_ref = k_s(Xn_test, diag=True) - (k_star * torch.linalg.solve(K, k_star.T).T).sum(-1)
    assert torch.allclose(post.mean.squeeze(-1), sy * mean_ref, atol=1e-8)
    assert torch.allclose(post.variance.squeeze(-1), sy ** 2 * var_ref, atol=1e-8)


def _logsum_inputs(n: int = 10):
    torch.manual_seed(3)
    X = torch.rand(n, 2, dtype=DTYPE)
    c = 0.1 + 0.8 * torch.rand(n, 1, dtype=DTYPE)
    y = c * (0.2 + 0.5 * torch.rand(n, 1, dtype=DTYPE))
    y[0] = 0.0
    return X, c, y


def _make_logsum(X, c, y, zero_yield=1e-3, zero_anchor=1e-2):
    prior = GammaPrior(2.0, 2.0)
    return make_logsum_model(
        X, c, y, c.squeeze(-1), y.squeeze(-1),
        _rbf([0, 1], 0.5, 1.0), _rbf([0, 1], 0.5, 1.0),
        conversion_mean=ConstantMean(), selectivity_mean=ConstantMean(),
        form_id_dim=-1, replicate_variances=(4e-4, 1e-4), noise_priors=(prior, prior),
        zero_yield=zero_yield, zero_noise_anchor=zero_anchor, noise_floor=1e-4,
    )


def test_logsum_targets_and_noise_shapes_of_zero_rows():
    """Zero yield enters as log(zero_yield) with noise evaluated at zero_anchor; other rows at their value."""
    X, c, y = _logsum_inputs()
    model = _make_logsum(X, c, y)
    n = X.shape[0]
    scale = model.gp.outcome_transform.stdvs.squeeze()
    targets = model.gp.train_targets * scale
    assert torch.allclose(targets[:n], c.squeeze(-1).log())
    assert math.isclose(float(targets[n]), math.log(1e-3), rel_tol=1e-12)
    assert torch.allclose(targets[n + 1:], y[1:].squeeze(-1).log())
    shape = model.noise_module().point_shape * scale ** 2
    assert math.isclose(float(shape[n]), 1e-2 ** -2, rel_tol=1e-12)
    assert torch.allclose(shape[n + 1:], y[1:].squeeze(-1) ** -2)
    assert torch.allclose(shape[:n], c.squeeze(-1) ** -2)


def test_logsum_rejects_non_positive_conversion():
    """A zero conversion has no logarithm and raises."""
    X, c, y = _logsum_inputs()
    c[2] = 0.0
    with pytest.raises(ValueError, match="positive conversion"):
        _make_logsum(X, c, y)


def test_logsum_samples_are_positive_and_shaped():
    """Conversion and latent selectivity are exponentials, so conversion is positive; shapes are (S, n)."""
    X, c, y = _logsum_inputs()
    model = _make_logsum(X, c, y)
    model.gp.eval()
    with torch.no_grad():
        samples = model.predictive_samples(torch.rand(4, 2, dtype=DTYPE), 300)
    for t in (samples.conversion, samples.yield_, samples.selectivity):
        assert t.shape == (300, 4)
        assert torch.isfinite(t).all()
    assert (samples.conversion > 0).all()


def test_product_cs_adds_the_yield_noise_once():
    """Constant latent C and S give yield C·S plus N(0, σ²_Y) with σ²_Y the replicate posterior mean."""
    torch.manual_seed(4)
    slots = torch.empty(200_000, 1, 2, dtype=DTYPE)
    slots[..., 0], slots[..., 1] = 0.6, 0.5
    samples = _product_cs_yield_samples(slots)
    expected_var = config.NOISE_EVIDENCE[("Y", "raw")].prior_mean
    assert samples.shape == (200_000, 1, 1)
    assert abs(float(samples.mean()) - 0.3) < 1e-3
    assert abs(float(samples.var()) / expected_var - 1.0) < 0.02


@pytest.mark.parametrize(("model_class", "transform"), [("logsum_cy", "probit"), ("condprod_cy", "raw")])
def test_product_classes_reject_other_transforms(cell, model_class, transform):
    """logsum_cy takes raw targets only, condprod_cy fits conversion on probit only."""
    builder = AcyloinCondensationModel(
        cell, list(range(30)), "KS1", "matern52", transform, config.LENGTHSCALE_LOGNORMAL_SCALE, prior_scheme="P2"
    )
    with pytest.raises(ValueError, match=model_class):
        builder.build(model_class, None)


@pytest.mark.parametrize(
    ("model_class", "transform"), [("logsum_cy", "raw"), ("condprod_cy", "probit"), ("lcm_cs", "raw")]
)
def test_product_classes_fit_one_fold(cell, model_class, transform):
    """Each class fits one fold whose test rows include zero yields, with finite metrics on yield and selectivity."""
    y = cell.y_raw.squeeze(-1)
    n = y.shape[0]
    zeros = torch.nonzero(y == 0).flatten().tolist()
    test = zeros[:2] + [i for i in range(n) if y[i] > 0.3][:4]
    train = [i for i in range(n) if i not in test]
    torch.manual_seed(5)
    result = fit_eval(
        cell, train, test, "KS1", transform, model_class, "matern52", "constant",
        n_restarts=1, max_iter=40, prior_scheme="P2",
    )
    assert result.status == "ok"
    assert len(result.crps_per_point) == len(test)
    assert len(result.crps_per_point_sel) == len(test)
    assert math.isfinite(result.crpss) and math.isfinite(result.crpss_sel)
    assert math.isfinite(result.posterior_mean_pearson)


def test_product_1_grid_lists_runnable_classes():
    """Every product_1 configuration names an implemented class at a transform it accepts."""
    grid = config.RUN_GRIDS["product_1"]
    assert grid.cat_feats == ("grover_pca",)
    pairs = {(cfg["model_class"], cfg["outcome_transform"]) for cfg in grid.configs}
    assert pairs == {
        ("st_y", "probit"), ("st_y", "raw"), ("lcm_cs", "raw"), ("lcm_cs", "probit"),
        ("logsum_cy", "raw"), ("condprod_cy", "probit"),
    }
    for model_class, _ in pairs:
        assert model_class in config.MODEL_CLASSES
        AcyloinCondensationModel.get_spec(model_class)
