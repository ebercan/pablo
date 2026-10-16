"""Predictive metrics for GP posteriors: NLPD, CRPS, MAE, RMSE, calibration KS.

All functions take aligned tensors of shape `(N,)` or `(N, 1)` (squeezed automatically).
NLPD and CRPS work on the modeling scale (whatever scale `mu` and `sigma` are on).
MAE / RMSE operate on whatever scale `y` is given on — the caller is responsible
for back-transforming logit-scale predictions to original yield space if desired.

Reference: Gneiting & Raftery (2007), "Strictly Proper Scoring Rules".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


_LOG_2PI = math.log(2.0 * math.pi)




def nlpd_gaussian(y: Tensor, mu: Tensor, var_total: Tensor) -> Tensor:
    """Per-point Gaussian NLPD: −log N(y; μ, σ²_total).

    Args:
        y: observations, shape (N,) or (N, 1).
        mu: posterior predictive mean, same shape as y.
        var_total: total predictive variance = posterior variance + likelihood noise,
            same shape as y. Must be strictly positive.

    Returns:
        Per-point NLPD, shape (N,).
    """
    y, mu, var_total = y.reshape(-1), mu.reshape(-1), var_total.reshape(-1)
    return 0.5 * ((y - mu) ** 2 / var_total + var_total.log() + _LOG_2PI)


def crps_gaussian(y: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Per-point Gaussian CRPS, closed form.

    CRPS(y; μ, σ) = σ · [z·(2Φ(z) − 1) + 2φ(z) − 1/√π]   with z = (y − μ)/σ.

    Lower is better. Sign convention: CRPS ≥ 0; equals 0 only at a perfect
    delta prediction matching y.

    Args:
        y: observations, shape (N,) or (N, 1).
        mu: predictive mean, same shape as y.
        sigma: predictive standard deviation, same shape as y. Must be
            strictly positive.

    Returns:
        Per-point CRPS, shape (N,).
    """
    y, mu, sigma = y.reshape(-1), mu.reshape(-1), sigma.reshape(-1)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def mae(y: Tensor, mu: Tensor) -> float:
    """Mean absolute error.

    Args:
        y: observations, shape (N,) or (N, 1).
        mu: predictions, same shape as y.

    Returns:
        Mean of |y − μ| as a Python float.
    """
    y, mu = y.reshape(-1), mu.reshape(-1)
    return (y - mu).abs().mean().item()


def rmse(y: Tensor, mu: Tensor) -> float:
    """Root mean squared error.

    Args:
        y: observations, shape (N,) or (N, 1).
        mu: predictions, same shape as y.

    Returns:
        √mean((y − μ)²) as a Python float.
    """
    y, mu = y.reshape(-1), mu.reshape(-1)
    return ((y - mu) ** 2).mean().sqrt().item()


def crps_gaussian_baseline(y: Tensor, y_train: Tensor) -> Tensor:
    """Per-point Gaussian CRPS of the constant N(mean(y_train), var(y_train)) forecast.

    The baseline is the "braindead" model that ignores the inputs and predicts the
    sample mean and sample variance of the training set for every test point.
    Used as the reference forecast for CRPSS.

    Args:
        y: test observations, shape (N,) or (N, 1).
        y_train: training observations, shape (M,) or (M, 1). Must have M >= 2.

    Returns:
        Per-point baseline CRPS, shape (N,).
    """
    y = y.reshape(-1)
    y_train = y_train.reshape(-1)
    mu_b = y_train.mean()
    sigma_b = y_train.std(unbiased=True).clamp_min(1e-8)
    mu = mu_b.expand_as(y)
    sigma = sigma_b.expand_as(y)
    return crps_gaussian(y, mu, sigma)


def nlpd_gaussian_baseline(y: Tensor, y_train: Tensor) -> Tensor:
    """Per-point Gaussian NLPD of the constant ``N(mean(y_train), var(y_train))`` forecast.

    The MSLL reference forecast: the "braindead" model that ignores the inputs and
    predicts the training-set mean and variance for every test point. Shares the
    baseline convention with ``crps_gaussian_baseline`` so MSLL and CRPSS are scored
    against the same null model.

    Args:
        y: test observations, shape (N,) or (N, 1).
        y_train: training observations, shape (M,) or (M, 1). Must have M >= 2.

    Returns:
        Per-point baseline NLPD, shape (N,).
    """
    y = y.reshape(-1)
    y_train = y_train.reshape(-1)
    mu_b = y_train.mean().expand_as(y)
    var_b = y_train.var(unbiased=True).clamp_min(1e-16).expand_as(y)
    return nlpd_gaussian(y, mu_b, var_b)


def mean_standardized_log_loss(
    nlpd_model: Tensor | float, nlpd_baseline: Tensor | float
) -> float:
    """Mean Standardized Log Loss (Rasmussen & Williams, eq. 2.34).

    MSLL = mean(nlpd_model) − mean(nlpd_baseline), where the baseline is the
    constant ``N(mean(y_train), var(y_train))`` forecast. Sign convention: < 0 is
    better than the training-mean baseline; 0 matches it; > 0 is worse. Tensors are
    reduced to scalars by mean before the difference.

    Args:
        nlpd_model: per-point model NLPD (Tensor) or a pre-aggregated scalar.
        nlpd_baseline: per-point baseline NLPD (Tensor) or a pre-aggregated scalar.

    Returns:
        Scalar MSLL value.
    """
    m = nlpd_model.reshape(-1).mean().item() if torch.is_tensor(nlpd_model) else float(nlpd_model)
    b = nlpd_baseline.reshape(-1).mean().item() if torch.is_tensor(nlpd_baseline) else float(nlpd_baseline)
    return m - b


def crps_skill_score(crps_model: Tensor | float, crps_baseline: Tensor | float) -> float:
    """CRPS skill score against a reference forecast.

    CRPSS = 1 − mean(crps_model) / mean(crps_baseline).

    Sign convention: +1 = perfect; 0 = same as baseline; < 0 = worse than
    baseline. Tensors are reduced to scalars by mean before the ratio is taken.

    Args:
        crps_model: per-point model CRPS (Tensor) or a pre-aggregated scalar.
        crps_baseline: per-point baseline CRPS (Tensor) or a pre-aggregated scalar.

    Returns:
        Scalar skill score.
    """
    if torch.is_tensor(crps_model):
        m = crps_model.reshape(-1).mean().item()
    else:
        m = float(crps_model)
    if torch.is_tensor(crps_baseline):
        b = crps_baseline.reshape(-1).mean().item()
    else:
        b = float(crps_baseline)
    return 1.0 - m / b


def pooled_crps_skill_score(
    crps_model: Tensor | Sequence[float], crps_baseline: Tensor | Sequence[float]
) -> float:
    """CRPS skill score pooled over every test point of a set of folds.

    CRPSS_pooled = 1 − Σ crps_model / Σ crps_baseline.

    The sums run over all test points of all folds, so a fold weighs by its
    baseline CRPS mass rather than counting once, and one fold with a large model
    CRPS moves the score in proportion to that CRPS. Per-point values and per-fold
    sums give the same result.

    Args:
        crps_model: Model CRPS, per point or per fold sum, shape ``(N,)``.
        crps_baseline: Baseline CRPS aligned with ``crps_model``, shape ``(N,)``.

    Returns:
        Scalar pooled skill score; +1 is perfect, 0 ties the baseline.

    Raises:
        ValueError: If the inputs are empty or misaligned, or the baseline sum is
            not positive.
    """
    m, b = (
        x.to(torch.float64).reshape(-1) if torch.is_tensor(x)
        else torch.tensor(x, dtype=torch.float64).reshape(-1)
        for x in (crps_model, crps_baseline)
    )
    if m.numel() == 0 or m.shape != b.shape:
        raise ValueError(
            f"pooled CRPSS needs aligned non-empty inputs; got {m.numel()} model "
            f"and {b.numel()} baseline values"
        )
    b_sum = b.sum().item()
    if not b_sum > 0.0:
        raise ValueError(f"baseline CRPS sums to {b_sum}; the skill score is undefined")
    return 1.0 - m.sum().item() / b_sum


def q_squared(y: Tensor, mu: Tensor, y_train: Tensor) -> float:
    """Cross-validated coefficient of determination Q^2.

    Q² = 1 − Σ(y − μ)² / Σ(y − mean(y_train))².

    Baseline is the training-set mean (not the test-set mean) — this is what
    makes it Q² rather than the usual R². Sign convention: +1 = perfect; 0 =
    same as predicting the training mean for every test point; < 0 = worse
    than predicting the training mean.

    Args:
        y: test observations, shape (N,) or (N, 1).
        mu: model posterior mean, same shape as y.
        y_train: training observations, shape (M,) or (M, 1).

    Returns:
        Scalar Q² value.
    """
    y, mu = y.reshape(-1), mu.reshape(-1)
    y_train = y_train.reshape(-1)
    y_bar_train = y_train.mean()
    sse = ((y - mu) ** 2).sum().item()
    tss = ((y - y_bar_train) ** 2).sum().item()
    if tss <= 0.0:
        return float("nan")
    return 1.0 - sse / tss


def calibration_ks(y: Tensor, mu: Tensor, sigma_total: Tensor) -> float:
    """Kolmogorov–Smirnov statistic of standardized residuals against N(0, 1).

    Computes (y − μ) / σ_total for every test point pooled from the caller, then
    returns sup_x |F_emp(x) − Φ(x)|. Lower is better; 0 indicates perfectly
    calibrated normality.

    Args:
        y: observations, shape (N,) or (N, 1).
        mu: predictive mean, same shape as y.
        sigma_total: total predictive standard deviation (posterior + noise),
            same shape as y.

    Returns:
        KS statistic in [0, 1].
    """
    y, mu, sigma_total = y.reshape(-1), mu.reshape(-1), sigma_total.reshape(-1)
    z = ((y - mu) / sigma_total).detach().cpu().numpy()
    z_sorted = sorted(z)
    n = len(z_sorted)
    max_diff = 0.0
    sqrt2 = math.sqrt(2.0)
    for i, zi in enumerate(z_sorted):
        f_emp_low = i / n
        f_emp_high = (i + 1) / n
        f_norm = 0.5 * (1.0 + math.erf(zi / sqrt2))
        diff = max(abs(f_emp_low - f_norm), abs(f_emp_high - f_norm))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def _pearson(a: Tensor, b: Tensor) -> float:
    """Pearson correlation between two 1-D tensors; NaN on zero variance or n < 2."""
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom <= 0.0:
        return float("nan")
    return (a @ b / denom).item()


def _average_ranks(x: Tensor) -> Tensor:
    """Ascending ranks of a 1-D tensor with ties broken by their mean rank."""
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[order] = torch.arange(1, x.numel() + 1, dtype=torch.float64, device=x.device)
    x_sorted = x[order]
    i = 0
    n = x.numel()
    while i < n:
        j = i + 1
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        if j - i > 1:
            tie_idx = order[i:j]
            ranks[tie_idx] = ranks[tie_idx].mean()
        i = j
    return ranks


def pearson_spearman(a: Tensor, b: Tensor) -> tuple[float, float]:
    """Pearson and (average-rank) Spearman correlation between two 1-D tensors.

    Args:
        a, b: shape (N,) or (N, 1), on the same scale.

    Returns:
        (pearson, spearman). Both are NaN when N < 2 or either vector has zero
        variance (constant predictions).
    """
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    pearson = _pearson(a, b)
    spearman = _pearson(_average_ranks(a), _average_ranks(b))
    return pearson, spearman


def crps_sample_based(y: Tensor, samples: Tensor) -> Tensor:
    """Empirical CRPS from posterior predictive samples on the raw scale.

    Unbiased estimator of ``CRPS = E|X − y| − 0.5 · E|X − X'|`` (Gneiting &
    Raftery 2007):

        CRPS ≈ S⁻¹ Σ_s |x_s − y| − (2S(S−1))⁻¹ Σ_{s≠t} |x_s − x_t|

    The pairwise sum uses the sorted-sample identity
    ``Σ_{s≠t} |x_s − x_t| = 2 Σ_k (2k − S − 1) x_(k)``, so each point costs
    O(S log S) instead of O(S²).

    Args:
        y: observations, shape (N,) or (N, 1).
        samples: posterior predictive samples on the raw scale, shape (S, N)
            or (S, N, 1). Requires S ≥ 2.

    Returns:
        Per-point CRPS, shape (N,).
    """
    y = y.reshape(-1)
    if samples.dim() == 3:
        samples = samples.squeeze(-1)
    s_sorted, _ = samples.sort(dim=0)
    s_count = s_sorted.shape[0]
    term1 = (s_sorted - y.unsqueeze(0)).abs().mean(dim=0)
    ranks = torch.arange(1, s_count + 1, dtype=samples.dtype, device=samples.device)
    weight = (2.0 * ranks - 1.0 - s_count) / (s_count * (s_count - 1))
    term2 = (weight.unsqueeze(-1) * s_sorted).sum(dim=0)
    return term1 - term2


def nlpd_sample_based(
    y: Tensor, samples: Tensor, bandwidth: float | None = None
) -> Tensor:
    """Per-point sample-based NLPD via a 1-D Gaussian KDE on the raw scale.

    The bandwidth defaults to a per-point Silverman rule of thumb on the
    samples for that point: h = 1.06 · σ_samples · S^(−1/5).

    Args:
        y: observations, shape (N,) or (N, 1).
        samples: posterior predictive samples, shape (S, N) or (S, N, 1).
        bandwidth: optional explicit bandwidth (scalar).

    Returns:
        Per-point NLPD = −log p(y), shape (N,).
    """
    y = y.reshape(-1)
    if samples.dim() == 3:
        samples = samples.squeeze(-1)
    s_count = samples.shape[0]

    if bandwidth is None:
        sample_std = samples.std(dim=0).clamp_min(1e-6)
        h = 1.06 * sample_std * (s_count ** (-1.0 / 5.0))
    else:
        h = torch.full_like(samples[0], float(bandwidth))

    z = (samples - y.unsqueeze(0)) / h.unsqueeze(0)
    log_kernels = -0.5 * z * z - h.log().unsqueeze(0) - 0.5 * _LOG_2PI
    log_p = torch.logsumexp(log_kernels, dim=0) - math.log(s_count)
    return -log_p


def calibration_ks_sample_based(y: Tensor, samples: Tensor) -> float:
    """KS statistic of PIT values vs Uniform(0,1).

    For each test point, PIT = empirical CDF of samples evaluated at y.
    Pool PITs across points and compute sup |F_emp(p) − p|.

    Args:
        y: observations, shape (N,) or (N, 1).
        samples: posterior predictive samples, shape (S, N) or (S, N, 1).

    Returns:
        KS statistic in [0, 1].
    """
    y = y.reshape(-1)
    if samples.dim() == 3:
        samples = samples.squeeze(-1)
    s_count = samples.shape[0]
    pit = (samples < y.unsqueeze(0)).to(samples.dtype).mean(dim=0)
    pit_np = pit.detach().cpu().numpy().tolist()
    pit_sorted = sorted(pit_np)
    n = len(pit_sorted)
    max_diff = 0.0
    for i, p in enumerate(pit_sorted):
        diff = max(abs(i / n - p), abs((i + 1) / n - p))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def all_metrics_sample_based(y: Tensor, samples: Tensor, y_train: Tensor) -> dict:
    """Sample-based per-cell metric bundle on the raw fraction scale.

    Used wherever the predictive distribution on the raw scale is non-Gaussian:
    every outcome transform other than ``raw``, and targets derived from
    samples of two modeled outputs (yield as a product, selectivity as a ratio).
    The predictive mean and variance are the sample mean and sample variance.

    Args:
        y: test observations, shape (N,) or (N, 1).
        samples: posterior predictive samples, shape (S, N) or (S, N, 1).
        y_train: training observations on the same scale as ``y``, shape
            (M,) or (M, 1) with M ≥ 2; used for the constant baseline forecast
            of CRPSS and MSLL and the training-mean baseline of Q².

    Returns:
        Dict with per-point lists of length N (``nlpd_per_point``,
        ``crps_per_point``, ``baseline_crps_per_point``,
        ``baseline_nlpd_per_point``, ``mu_pred_per_point``,
        ``var_pred_per_point``) and float scalars (``crpss``, ``msll``,
        ``q2``, ``mae``, ``rmse``, ``calibration_ks``).
    """
    if samples.dim() == 3:
        samples = samples.squeeze(-1)
    mu_hat = samples.mean(dim=0)
    var_hat = samples.var(dim=0)
    crps_pp = crps_sample_based(y, samples)
    baseline_crps_pp = crps_gaussian_baseline(y, y_train)
    nlpd_pp = nlpd_sample_based(y, samples)
    baseline_nlpd_pp = nlpd_gaussian_baseline(y, y_train)
    return {
        "nlpd_per_point": nlpd_pp.tolist(),
        "crps_per_point": crps_pp.tolist(),
        "baseline_crps_per_point": baseline_crps_pp.tolist(),
        "baseline_nlpd_per_point": baseline_nlpd_pp.tolist(),
        "crpss": crps_skill_score(crps_pp, baseline_crps_pp),
        "msll": mean_standardized_log_loss(nlpd_pp, baseline_nlpd_pp),
        "q2": q_squared(y, mu_hat, y_train),
        "mae": mae(y, mu_hat),
        "rmse": rmse(y, mu_hat),
        "calibration_ks": calibration_ks_sample_based(y, samples),
        "mu_pred_per_point": mu_hat.reshape(-1).tolist(),
        "var_pred_per_point": var_hat.reshape(-1).tolist(),
    }


def all_metrics(
    y: Tensor,
    mu: Tensor,
    var_post: Tensor,
    noise_var: Tensor | float,
    y_train: Tensor,
) -> dict:
    """Compute the full per-cell metric bundle.

    Args:
        y: observations, shape (N,) or (N, 1).
        mu: posterior mean, same shape as y.
        var_post: posterior variance (epistemic only), same shape as y.
        noise_var: scalar likelihood noise (or shape-broadcastable tensor).
        y_train: training observations on the same scale as ``y``, shape
            (M,) or (M, 1) with M ≥ 2; used to build the constant-mean /
            constant-variance baseline forecast for CRPSS and MSLL and the
            training-mean baseline for Q².

    Returns:
        Dict with per-point lists of length N (``nlpd_per_point``,
        ``crps_per_point``, ``baseline_crps_per_point``,
        ``baseline_nlpd_per_point``, ``mu_pred_per_point``, and
        ``var_pred_per_point`` holding the total variance
        ``var_post + noise_var``) plus float scalars (``crpss``, ``msll``,
        ``q2``, ``mae``, ``rmse``, ``calibration_ks``). The per-point lists
        support paired statistical tests across folds.
    """
    if not torch.is_tensor(noise_var):
        noise_var = torch.tensor(noise_var, dtype=var_post.dtype, device=var_post.device)
    var_total = var_post + noise_var
    sigma_total = var_total.sqrt()
    crps_pp = crps_gaussian(y, mu, sigma_total)
    baseline_crps_pp = crps_gaussian_baseline(y, y_train)
    nlpd_pp = nlpd_gaussian(y, mu, var_total)
    baseline_nlpd_pp = nlpd_gaussian_baseline(y, y_train)
    return {
        "nlpd_per_point": nlpd_pp.tolist(),
        "crps_per_point": crps_pp.tolist(),
        "baseline_crps_per_point": baseline_crps_pp.tolist(),
        "baseline_nlpd_per_point": baseline_nlpd_pp.tolist(),
        "crpss": crps_skill_score(crps_pp, baseline_crps_pp),
        "msll": mean_standardized_log_loss(nlpd_pp, baseline_nlpd_pp),
        "q2": q_squared(y, mu, y_train),
        "mae": mae(y, mu),
        "rmse": rmse(y, mu),
        "calibration_ks": calibration_ks(y, mu, sigma_total),
        "mu_pred_per_point": mu.reshape(-1).tolist(),
        "var_pred_per_point": var_total.reshape(-1).tolist(),
    }
