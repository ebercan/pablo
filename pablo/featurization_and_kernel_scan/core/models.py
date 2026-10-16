"""GP model factory and mean modules for the featurization x kernel scan.

The ``AcyloinCondensationModel`` class catalogs 16 model classes: four families
× {yield-only, (C,S), (Y,S), (Y,C)} target combinations, plus two classes that
build ``Y = C·S`` into the model (``core/product_models.py``):

  st_y / mapsaas_y      single model on yield (yield-only).
  indep_{cs,ys,yc}      two independent SingleTaskGPs.
  icm_{cs,ys,yc}        full-rank (rank-2) ICM KroneckerMultiTaskGP.
  lcm_{cs,ys,yc}        true LCM (LCMKernel, Q=2 latent groups, rank-2 each).
  mapsaas_{cs,ys,yc}    two independent AdditiveMapSaasSingleTaskGPs.
  logsum_cy             log C and log Y = log C + log S, fixed-mixing LMC.
  condprod_cy           Y = C·s(x) + noise with measured C, plus a conversion GP.

ICM is a single data kernel ⊗ one IndexKernel task covariance; LCM is a sum of
two data-kernel ⊗ IndexKernel terms (one shared structure per latent group). The
IndexKernel task covariance is unconstrained, so cross-task correlation is free in
sign with a symmetric (loc-0) prior — no bias toward positive or negative.

The live grid runs the four kernel structures ``KS1`` (scaled product of the
four block kernels), ``KS2`` (4-term additive mixture), ``KS3`` (+6 pairwise
products) and ``KS4`` (+ the 4-way product, 11 terms); ``kernel_structure ==
"Vanilla"`` (flat ARD RBF, no ScaleKernel) remains buildable but is off-grid.
The ``mapsaas_*`` family uses a fixed additive Matérn-5/2 SAAS architecture and
ignores the kernel structure, mean module, and InverseGamma noise prior.

Mean modules (``config.MEAN_FUNCTIONS``):
  M1  ConstantMean — gpytorch built-in, the default.
  M2  BlockLinearMean — four (a, b) pairs (cat, solv, base, cont), summed.
      Base block is masked to zero for non-salt rows.
  M3  TypeLinearMean — one global continuous (a, b) + four per-pre-cat-type
      discrete (a, b) pairs, two active at a time.
  M4  TypeQuadLogMean — M3 with the global continuous linear term replaced by a
      quadratic in temperature plus a log in catalyst loading, whose three
      coefficients are partially pooled (global + per-pre-cat-type deviation).
      fa_ratio and time stay linear.
  cond_linear  CondLinearMean — intercept, solvent, base and continuous slopes,
      base terms gated by base_active; no catalyst slopes.
  cond_trend   ConstantMean plus ``LinearTrendKernel``: cond_linear's
      coefficients integrated out.
  form_offset  ConstantMean plus ``FormOffsetKernel``: an integrated offset per
      pre-catalyst form.

To add a new model class, add a method on ``AcyloinCondensationModel`` and
an entry to ``AcyloinCondensationModel.SPECS``. All priors and concentrations
come from ``config.py``.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass

import pyro
import pyro.distributions as pdist
import torch
from pyro.infer.autoguide.initialization import init_to_value
from pyro.infer.mcmc import MCMC, NUTS
from pyro.ops.stats import effective_sample_size, split_gelman_rubin

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import (
    AbstractFullyBayesianSingleTaskGP,
    PyroModel,
    reshape_and_detach,
)
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputTransform,
    Normalize,
)
from botorch.models.transforms.outcome import OutcomeTransform, Standardize

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import (
    AdditiveKernel,
    IndexKernel,
    Kernel,
    LCMKernel,
    MaternKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood, Likelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.means import ConstantMean, Mean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior, LogNormalPrior, NormalPrior, Prior

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import (
    BASE_KERNELS,
    BLOCKS,
    FormOffsetKernel,
    LinearTrendKernel,
    PartialARDKernel,
    make_kernel_from_config,
)
from pablo.featurization_and_kernel_scan.core.product_models import (
    CondProdModel,
    LogSumModel,
    make_condprod_model,
    make_logsum_model,
)
from pablo.featurization_and_kernel_scan.core.replicates import (
    ReplicateNoiseEvidence,
    replicate_anchor_values,
)
from pablo.utils.feature_assembly import CellFeatures
from pablo.utils.outcome_transforms import forward as forward_outcome
from torch import Tensor, nn
from torch.distributions import InverseGamma


def _vanilla_lengthscale_constraint() -> GreaterThan | None:
    """``GreaterThan(config.LENGTHSCALE_FLOOR)`` for the flat-ARD vanilla kernels.

    The vanilla models build their ``MaternKernel`` inline rather than via
    ``make_kernel_from_config``, so they apply the same lengthscale floor here.
    Returns ``None`` (default ``Positive`` constraint) when no floor is set.
    """
    floor = getattr(config, "LENGTHSCALE_FLOOR", None)
    return None if floor is None else GreaterThan(floor)


@dataclass(frozen=True)
class ModelSpec:
    """Structured descriptor for a model-class string.

    Each model reports metrics on two quantities — yield (Y) and, except for the
    yield-only classes, selectivity (S). ``yield_src`` / ``sel_src`` say how each
    is obtained from the fitted model; ``yield_index`` / ``sel_index`` give the
    task column for the ``marginal`` source.

    Attributes:
        model_class: Canonical ID, e.g. ``"icm_ys"``.
        architecture: ``"singletask"`` | ``"independent"`` | ``"icm"`` | ``"lcm"``
            | ``"logsum"`` | ``"condprod"``.
        model_family: ``"gp"`` (SingleTaskGP / Kronecker / LCM) or ``"mapsaas"``
            (``AdditiveMapSaasSingleTaskGP``).
        targets: Tuple of modeled target labels in column order. Each label is one
            of ``"Y"`` (yield), ``"C"`` (conversion), ``"S"`` (selectivity).
        rank: Task-covariance rank for ICM/LCM (2 = full rank for 2 tasks); 0 for
            singletask/independent.
        yield_src: How yield is obtained — ``"direct"`` (single model on Y),
            ``"marginal"`` (Y is a task of the joint posterior),
            ``"product_cs"`` (sample-based ``Y = C * S``), or ``"logsum"`` /
            ``"condprod"`` (the samples of a ``product_models`` class).
        yield_index: Task column of yield for ``yield_src == "marginal"``, else
            ``None``. For ``independent``, ``0``/``1`` selects which member model.
        sel_src: How selectivity is obtained — ``"none"`` (yield-only model),
            ``"direct"`` (single model on S), ``"marginal"`` (S is a task),
            ``"ratio_yc"`` (sample-based ``S = Y / C``), or ``"logsum"`` /
            ``"condprod"`` (the samples of a ``product_models`` class).
        sel_index: Task column of selectivity for ``sel_src == "marginal"`` (or
            the member-model index for ``independent`` ``"direct"``), else ``None``.
    """

    model_class: str
    architecture: str
    model_family: str
    targets: tuple[str, ...]
    rank: int
    yield_src: str
    yield_index: int | None
    sel_src: str
    sel_index: int | None


class BlockLinearMean(Mean):
    """Block-wise linear mean: one (a, b) pair per feature block, summed.

    mu(x) = (a_cat . x_cat + b_cat)
           + (a_solv . x_solv + b_solv)
           + mask * (a_base . x_base + b_base)
           + (a_cont . x_cont + b_cont)

    The base block is masked: for non-salt rows (base_active == 0) both the
    linear and intercept terms are zeroed out.

    Args:
        feature_dims: dict mapping block name -> list of column indices.
            Keys must include "cat", "solv", "base", "cont".
        active_flag_dim: column index of the base_active indicator in x.
        prior_loc: location μ of the Normal prior on each parameter.
        prior_scale: stddev of the Normal prior on each parameter.
    """

    def __init__(
        self,
        feature_dims: dict[str, list[int]],
        active_flag_dim: int,
        prior_loc: float,
        prior_scale: float,
    ) -> None:
        super().__init__()
        self.active_flag_dim = int(active_flag_dim)

        self._block_names = ["cat", "solv", "base", "cont"]
        for name in self._block_names:
            dims = feature_dims[name]
            idx_tensor = torch.tensor(dims, dtype=torch.long)
            self.register_buffer(f"_dims_{name}", idx_tensor)
            setattr(self, f"a_{name}", nn.Parameter(torch.zeros(len(dims), dtype=torch.float64)))
            setattr(self, f"b_{name}", nn.Parameter(torch.zeros((), dtype=torch.float64)))

        loc = torch.tensor(prior_loc, dtype=torch.float64)
        intercept_scale = torch.tensor(prior_scale, dtype=torch.float64)
        for name in self._block_names:
            d_block = len(feature_dims[name])
            slope_s = prior_scale / math.sqrt(d_block) if d_block > 0 else prior_scale
            self.register_prior(
                f"a_{name}_prior",
                NormalPrior(loc, torch.tensor(slope_s, dtype=torch.float64)),
                f"a_{name}",
            )
            self.register_prior(f"b_{name}_prior", NormalPrior(loc, intercept_scale), f"b_{name}")

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the block-wise linear mean.

        Args:
            x: ``... x N x D`` inputs.

        Returns:
            ``... x N`` mean values.
        """
        mu = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

        for name in self._block_names:
            dims = getattr(self, f"_dims_{name}")
            a = getattr(self, f"a_{name}")
            b = getattr(self, f"b_{name}")
            x_block = x.index_select(-1, dims.to(x.device))
            contrib = (x_block * a).sum(dim=-1) + b
            if name == "base":
                contrib = contrib * x[..., self.active_flag_dim]
            mu = mu + contrib

        return mu


class CondLinearMean(Mean):
    """Linear mean over the solvent, base and continuous blocks; no catalyst slopes.

    mu(x) = c + a_solv . x_solv + a_cont . x_cont + base_active * (b_base + a_base . x_base)

    The catalyst block does not enter: its slopes stay at their prior on every
    catalyst level of the elimination run, and they are what makes
    ``block_linear`` cost grow with catalyst width. The parameter count is
    ``d_solv + d_base + d_cont + 2``.

    Args:
        feature_dims: Block name -> column indices; keys "solv", "base", "cont" are read.
        active_flag_dim: Column index of ``base_active``.
        prior_loc: Location of every Normal prior.
        prior_scale: Standard deviation of the intercept priors; each slope of a
            ``d``-column block has standard deviation ``prior_scale / sqrt(d)``.
    """

    def __init__(
        self,
        feature_dims: dict[str, list[int]],
        active_flag_dim: int,
        prior_loc: float,
        prior_scale: float,
    ) -> None:
        super().__init__()
        self.active_flag_dim = int(active_flag_dim)
        self._block_names = ["solv", "base", "cont"]
        loc = torch.tensor(prior_loc, dtype=torch.float64)
        for name in self._block_names:
            dims = feature_dims[name]
            if len(dims) == 0:
                raise ValueError(f"cond_linear needs a non-empty {name!r} block")
            self.register_buffer(f"_dims_{name}", torch.tensor(dims, dtype=torch.long))
            setattr(self, f"a_{name}", nn.Parameter(torch.zeros(len(dims), dtype=torch.float64)))
            self.register_prior(
                f"a_{name}_prior",
                NormalPrior(loc, torch.tensor(prior_scale / math.sqrt(len(dims)), dtype=torch.float64)),
                f"a_{name}",
            )
        for name in ("c", "b_base"):
            setattr(self, name, nn.Parameter(torch.zeros((), dtype=torch.float64)))
            self.register_prior(
                f"{name}_prior", NormalPrior(loc, torch.tensor(prior_scale, dtype=torch.float64)), name
            )

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the mean.

        Args:
            x: ``... x N x D`` inputs.

        Returns:
            ``... x N`` mean values.
        """
        def linear(name: str) -> Tensor:
            dims = getattr(self, f"_dims_{name}").to(x.device)
            return (x.index_select(-1, dims) * getattr(self, f"a_{name}")).sum(dim=-1)

        gate = x[..., self.active_flag_dim]
        return self.c + linear("solv") + linear("cont") + gate * (self.b_base + linear("base"))


class TypeLinearMean(Mean):
    """Hierarchical linear mean: one global continuous block + per-type discrete blocks.

    mu(x) = a_cont . x[cont_dims] + b_cont
          + a_disc[t] . x[disc_dims] + b_disc[t]

    where t = int(x[..., form_id_dim]) selects one of ``n_types`` pre-catalyst
    form groups (salt / free_nhc / co2_adduct / bicarbonate).

    Args:
        cont_dims: column indices of the continuous block in x.
        disc_dims: column indices of the discrete block (cat + solv + base)
            in x. base_active and form_id are excluded.
        form_id_dim: column index of the integer pre-cat-form ID in x.
        n_types: number of pre-catalyst form groups.
        prior_loc: location μ of the Normal prior on each parameter.
        prior_scale: stddev of the Normal prior on each parameter.
    """

    def __init__(
        self,
        cont_dims: list[int],
        disc_dims: list[int],
        form_id_dim: int,
        n_types: int,
        prior_loc: float,
        prior_scale: float,
    ) -> None:
        super().__init__()
        self.cont_dims = torch.tensor(cont_dims, dtype=torch.long)
        self.disc_dims = torch.tensor(disc_dims, dtype=torch.long)
        self.form_id_dim = int(form_id_dim)
        self.n_types = int(n_types)

        d_cont = len(cont_dims)
        d_disc = len(disc_dims)

        self.a_cont = nn.Parameter(torch.zeros(d_cont, dtype=torch.float64))
        self.b_cont = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.a_disc = nn.Parameter(torch.zeros(n_types, d_disc, dtype=torch.float64))
        self.b_disc = nn.Parameter(torch.zeros(n_types, dtype=torch.float64))

        loc = torch.tensor(prior_loc, dtype=torch.float64)
        intercept_scale = torch.tensor(prior_scale, dtype=torch.float64)
        scale_a_cont = prior_scale / math.sqrt(d_cont) if d_cont > 0 else prior_scale
        scale_a_disc = prior_scale / math.sqrt(d_disc) if d_disc > 0 else prior_scale
        self.register_prior(
            "a_cont_prior",
            NormalPrior(loc, torch.tensor(scale_a_cont, dtype=torch.float64)),
            "a_cont",
        )
        self.register_prior("b_cont_prior", NormalPrior(loc, intercept_scale), "b_cont")
        self.register_prior(
            "a_disc_prior",
            NormalPrior(loc, torch.tensor(scale_a_disc, dtype=torch.float64)),
            "a_disc",
        )
        self.register_prior("b_disc_prior", NormalPrior(loc, intercept_scale), "b_disc")

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the hierarchical linear mean.

        The form-ID column is cast to an integer and clamped to
        ``[0, n_types - 1]`` before it selects the per-type coefficients.

        Args:
            x: ``... x N x D`` inputs.

        Returns:
            ``... x N`` mean values.
        """
        x_cont = x.index_select(-1, self.cont_dims.to(x.device))
        x_disc = x.index_select(-1, self.disc_dims.to(x.device))
        form_id = x[..., self.form_id_dim].long().clamp(0, self.n_types - 1)

        mu_cont = (x_cont * self.a_cont).sum(dim=-1) + self.b_cont

        a_t = self.a_disc[form_id]
        b_t = self.b_disc[form_id]
        mu_disc = (x_disc * a_t).sum(dim=-1) + b_t

        return mu_cont + mu_disc


class TypeQuadLogMean(Mean):
    """Partially pooled thermal mean: quadratic in temperature, logarithmic in loading.

    mu(x) = (c_quad + d_quad[t]) * zT^2
          + (c_lin  + d_lin[t])  * zT
          + (c_log  + d_log[t])  * log(L / L_ref)
          + a_rest . x[rest_dims] + b_cont
          + a_disc[t] . x[disc_dims] + b_disc[t]

    where ``t = int(x[..., form_id_dim])`` is the pre-catalyst form group and
    ``b_disc[t]`` carries the per-type intercept (the ``c3`` of the
    temperature/loading polynomial; a separate per-type constant would be
    perfectly confounded with it).

    The three shape coefficients are **partially pooled**: a global value shared
    by all types plus a per-type deviation whose Normal prior is
    ``deviation_frac`` times as wide. Types with no temperature or loading
    variation in the training fold (in the 69-experiment table, ``free_nhc``
    contributes three rows at a single condition) therefore collapse onto the
    global curve instead of extrapolating an unidentified parabola.

    ``zT`` is the library-bounds standardised temperature on [-1, 1] and ``L``
    the catalyst loading in raw molar-ratio units. Both are reconstructed inside
    ``forward`` because the mean module is invoked *after* the model's
    ``Normalize`` input transform, whose learned per-fold [0, 1] map would
    otherwise put ``log`` on a column whose minimum is exactly zero.
    ``L_ref`` is the midpoint of the library loading bounds, which keeps the log
    term orthogonal to the intercepts.

    Args:
        temp_dim: column index of the standardised temperature in x.
        loading_dim: column index of the standardised loading in x.
        rest_dims: column indices of the remaining continuous columns, which
            stay linear.
        disc_dims: column indices of the discrete block (cat + solv + base);
            base_active and form_id are excluded.
        form_id_dim: column index of the integer pre-cat-form ID in x.
        n_types: number of pre-catalyst form groups.
        norm_offset: ``(2,)`` offset of the model's learned ``Normalize`` on the
            (temperature, loading) columns.
        norm_span: ``(2,)`` span of that same transform.
        loading_bounds: ``(lo, hi)`` raw loading bounds used by the library
            standardisation.
        prior_loc: location mu of the Normal prior on each parameter.
        prior_scale: stddev of the Normal prior on each parameter.
        deviation_frac: multiplier on ``prior_scale`` for the per-type
            deviations, i.e. the strength of the partial pooling.
    """

    def __init__(
        self,
        temp_dim: int,
        loading_dim: int,
        rest_dims: list[int],
        disc_dims: list[int],
        form_id_dim: int,
        n_types: int,
        norm_offset: Tensor,
        norm_span: Tensor,
        loading_bounds: tuple[float, float],
        prior_loc: float,
        prior_scale: float,
        deviation_frac: float,
    ) -> None:
        super().__init__()
        self.form_id_dim = int(form_id_dim)
        self.n_types = int(n_types)

        self.register_buffer(
            "_tl_dims", torch.tensor([int(temp_dim), int(loading_dim)], dtype=torch.long)
        )
        self.register_buffer("_rest_dims", torch.tensor(rest_dims, dtype=torch.long))
        self.register_buffer("_disc_dims", torch.tensor(disc_dims, dtype=torch.long))
        self.register_buffer("_norm_offset", norm_offset.to(torch.float64))
        self.register_buffer("_norm_span", norm_span.to(torch.float64))

        lo, hi = float(loading_bounds[0]), float(loading_bounds[1])
        self.register_buffer("_load_lo", torch.tensor(lo, dtype=torch.float64))
        self.register_buffer("_load_half_range", torch.tensor((hi - lo) / 2.0, dtype=torch.float64))
        self.register_buffer(
            "_log_load_ref", torch.tensor(math.log(0.5 * (lo + hi)), dtype=torch.float64)
        )

        d_rest = len(rest_dims)
        d_disc = len(disc_dims)
        n_shape = 3

        self.c_shape = nn.Parameter(torch.zeros(n_shape, dtype=torch.float64))
        self.d_shape = nn.Parameter(torch.zeros(n_types, n_shape, dtype=torch.float64))
        self.a_rest = nn.Parameter(torch.zeros(d_rest, dtype=torch.float64))
        self.b_cont = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.a_disc = nn.Parameter(torch.zeros(n_types, d_disc, dtype=torch.float64))
        self.b_disc = nn.Parameter(torch.zeros(n_types, dtype=torch.float64))

        loc = torch.tensor(prior_loc, dtype=torch.float64)
        intercept_scale = torch.tensor(prior_scale, dtype=torch.float64)
        scale_shape = prior_scale / math.sqrt(n_shape)
        scale_dev = scale_shape * deviation_frac
        scale_rest = prior_scale / math.sqrt(d_rest) if d_rest > 0 else prior_scale
        scale_disc = prior_scale / math.sqrt(d_disc) if d_disc > 0 else prior_scale

        self.register_prior(
            "c_shape_prior",
            NormalPrior(loc, torch.tensor(scale_shape, dtype=torch.float64)),
            "c_shape",
        )
        self.register_prior(
            "d_shape_prior",
            NormalPrior(loc, torch.tensor(scale_dev, dtype=torch.float64)),
            "d_shape",
        )
        self.register_prior(
            "a_rest_prior",
            NormalPrior(loc, torch.tensor(scale_rest, dtype=torch.float64)),
            "a_rest",
        )
        self.register_prior("b_cont_prior", NormalPrior(loc, intercept_scale), "b_cont")
        self.register_prior(
            "a_disc_prior",
            NormalPrior(loc, torch.tensor(scale_disc, dtype=torch.float64)),
            "a_disc",
        )
        self.register_prior("b_disc_prior", NormalPrior(loc, intercept_scale), "b_disc")

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the mean on ``(..., N, D)`` inputs, returning ``(..., N)``."""
        z = x.index_select(-1, self._tl_dims) * self._norm_span + self._norm_offset
        z_temp = z[..., 0]
        loading = self._load_lo + self._load_half_range * (z[..., 1] + 1.0)

        basis = torch.stack(
            [z_temp * z_temp, z_temp, torch.log(loading) - self._log_load_ref], dim=-1
        )

        form_id = x[..., self.form_id_dim].long().clamp(0, self.n_types - 1)
        coeffs = self.c_shape + self.d_shape[form_id]
        mu = (basis * coeffs).sum(dim=-1)

        x_rest = x.index_select(-1, self._rest_dims)
        mu = mu + (x_rest * self.a_rest).sum(dim=-1) + self.b_cont

        x_disc = x.index_select(-1, self._disc_dims)
        mu = mu + (x_disc * self.a_disc[form_id]).sum(dim=-1) + self.b_disc[form_id]

        return mu


class InverseGammaPrior(Prior):
    """InverseGamma(concentration, rate). Mean = rate / (concentration - 1) for c > 1."""

    arg_constraints = {}

    def __init__(self, concentration: Tensor, rate: Tensor, transform=None) -> None:
        super(Prior, self).__init__()
        self.register_buffer("concentration", concentration)
        self.register_buffer("rate", rate)
        self._transform = transform

    def _dist(self) -> InverseGamma:
        return InverseGamma(concentration=self.concentration, rate=self.rate)

    def log_prob(self, parameter: Tensor) -> Tensor:
        if self._transform is not None:
            parameter = self._transform(parameter)
        return self._dist().log_prob(parameter).sum()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self._dist().sample(sample_shape)

    def forward(self, x):
        pass


class _MultitaskDataAdapter(Kernel):
    """Adapter so a kernel using ``active_dims`` works inside ``MultitaskKernel``.

    ``MultitaskKernel.forward`` invokes ``data_covar_module.forward(x1, x2)``
    directly, which bypasses the ``active_dims`` slicing done in
    ``Kernel.__call__``. This adapter wraps the inner kernel so its
    ``forward`` invokes ``__call__`` on the original, restoring the slicing.
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, wrapped: Kernel) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(
        self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **params
    ):
        out = self.wrapped(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        return out.to_dense()


def _build_normalize(d: int, form_id_dim: int) -> Normalize:
    if form_id_dim < 0:
        return Normalize(d=d)
    indices = [i for i in range(d) if i != form_id_dim]
    return Normalize(d=d, indices=indices)


def _patch_multitask_likelihood_priors(
    likelihood: MultitaskGaussianLikelihood, noise_prior: Prior
) -> None:
    """Re-bind noise priors with explicit setting closures (upstream gpytorch bug fix)."""
    likelihood._priors["raw_task_noises_prior"] = (
        noise_prior,
        lambda m: m.task_noises,
        lambda m, v: m._set_task_noises(v),
    )
    likelihood._priors["raw_noise_prior"] = (
        noise_prior,
        lambda m: m.noise,
        lambda m, v: m._set_noise(v),
    )


def _shared_standardized_noise_rate(
    train_y: Tensor, noise_prior_betas: tuple[float, float]
) -> Tensor:
    """InverseGamma rate shared by the noises of a two-task GP, standardized scale.

    ``Standardize(m=2)`` divides task ``t`` by its own standard deviation, so
    task ``t``'s replicate rate on the standardized scale is ``β_t / var_t``.
    One prior serves both task noises and the global noise, and its rate is the
    mean of the two.

    Args:
        train_y: ``(N, 2)`` training targets on the outcome scale.
        noise_prior_betas: InverseGamma β per column of ``train_y``, each in
            squared units of that column.

    Returns:
        Scalar rate tensor in ``train_y``'s dtype.
    """
    betas = torch.tensor(noise_prior_betas, dtype=train_y.dtype, device=train_y.device)
    return (betas / train_y.var(dim=0)).mean()


def _strip_task_covar_lkj_prior(task_kernel: IndexKernel) -> None:
    """Remove BoTorch's default LKJCovariancePrior from an ``IndexKernel``."""
    task_kernel._priors.pop("IndexKernelPrior", None)


def _floor_task_covar_var(task_kernel: IndexKernel) -> None:
    """Replace the ``raw_var`` ``Positive`` constraint with ``GreaterThan(floor)``."""
    new_constraint = GreaterThan(config.TASK_COVAR_VAR_FLOOR)
    task_kernel._modules["raw_var_constraint"] = new_constraint
    task_kernel._constraints["raw_var_constraint"] = new_constraint


def _register_task_covar_priors(
    task_kernel: IndexKernel,
    noise_prior_alpha: float,
    beta_standardized: Tensor,
) -> None:
    """Register Normal on covar_factor and InverseGamma on var in an ``IndexKernel``.

    Uses the same InverseGamma parameterisation as the noise prior so the task
    covariance scale stays consistent with the standardised output scale. The
    Normal prior on ``covar_factor`` is centred at ``config.TASK_COVAR_FACTOR_NORMAL_LOC``
    (0.0), so the off-diagonal task correlation is unbiased in sign — positive and
    negative cross-task correlations are equally favoured.

    Args:
        task_kernel: The ``IndexKernel`` whose task covariance is being primed.
        noise_prior_alpha: InverseGamma α (concentration) for the ``var`` prior.
        beta_standardized: InverseGamma β (rate) on the standardised scale.
    """
    cf_shape = task_kernel.covar_factor.shape
    dtype = beta_standardized.dtype

    cf_loc = torch.full(cf_shape, config.TASK_COVAR_FACTOR_NORMAL_LOC, dtype=dtype)
    cf_scale = torch.full(cf_shape, config.TASK_COVAR_FACTOR_NORMAL_SCALE, dtype=dtype)

    task_kernel.register_prior(
        "covar_factor_prior",
        NormalPrior(cf_loc, cf_scale),
        lambda m: m.covar_factor,
        lambda m, v: m.covar_factor.data.copy_(v.detach()),
    )
    task_kernel.register_prior(
        "var_prior",
        InverseGammaPrior(
            concentration=torch.tensor(noise_prior_alpha, dtype=dtype),
            rate=beta_standardized.reshape(()).clone(),
        ),
        lambda m: m.var,
        lambda m, v: m.raw_var.data.copy_(
            m.raw_var_constraint.inverse_transform(v).detach()
        ),
    )


def _assert_all_priors_sampleable(model: ExactGP) -> None:
    offenders = [
        name
        for name, _, _, _, setting_closure in model.named_priors()
        if setting_closure is None
    ]
    if offenders:
        raise RuntimeError(
            "Model contains priors without setting_closure (would crash "
            f"botorch's fit fallback): {offenders}"
        )


def _continuous_library_bounds() -> dict[str, tuple[float, float]]:
    """Raw (lo, hi) bounds behind the ``*_std`` continuous columns.

    Returns:
        Mapping from raw column name (``"fa_ratio"``, ``"loading"``,
        ``"temperature"``, ``"time"``) to the library bounds used by
        ``feature_assembly`` to map it onto [-1, 1].
    """
    meta = json.loads((paths.FEAT_DIR / "continuous" / "metadata.json").read_text())
    return {k: (float(v[0]), float(v[1])) for k, v in meta["bounds"].items()}


def _cont_column_index(cell_features: CellFeatures, raw_name: str) -> int:
    """Column index of a standardised continuous feature in ``cell_features.X``."""
    name = f"{raw_name}_std"
    if name not in cell_features.feature_names:
        raise KeyError(
            f"{name!r} not among feature_names; the continuous block is "
            f"{[cell_features.feature_names[i] for i in cell_features.feature_dims['cont']]}"
        )
    return cell_features.feature_names.index(name)


def _normalize_inverse_affine(
    train_x: Tensor, dims: list[int], min_range: float = 1e-8
) -> tuple[Tensor, Tensor]:
    """Coefficients that undo the model's learned ``Normalize`` on ``dims``.

    ``Normalize`` learns ``offset``/``coefficient`` from the training inputs, so
    ``u = (z - offset) / coefficient`` and ``z = u * span + off`` with the values
    returned here. Mirrors ``Normalize._update_coefficients``, including its
    collapse to the identity map on a column whose range underflows
    ``min_range``.

    Args:
        train_x: ``(N, D)`` *untransformed* training inputs for the fold.
        dims: column indices to invert.
        min_range: ``Normalize``'s degeneracy threshold.

    Returns:
        ``(off, span)``, each ``(len(dims),)``.
    """
    cols = train_x[..., dims]
    off = cols.amin(dim=-2)
    span = cols.amax(dim=-2) - off
    degenerate = span < min_range
    return torch.where(degenerate, torch.zeros_like(off), off), torch.where(
        degenerate, torch.ones_like(span), span
    )


def _build_type_quadlog_mean(
    cell_features: CellFeatures, prior_scale: float, train_x: Tensor | None
) -> "TypeQuadLogMean":
    """Assemble a ``TypeQuadLogMean`` from a cell and its fold training inputs."""
    if train_x is None:
        raise ValueError(
            "mean_function='type_quadlog' needs the fold's training inputs to "
            "invert the model's learned Normalize transform; pass "
            "build_mean_module(..., train_x=cell_features.X[train_idx])."
        )

    temp_dim = _cont_column_index(cell_features, "temperature")
    loading_dim = _cont_column_index(cell_features, "loading")
    rest_dims = [d for d in cell_features.feature_dims["cont"] if d not in (temp_dim, loading_dim)]

    bounds = _continuous_library_bounds()
    lo, hi = bounds["loading"]
    z_loading = cell_features.X[:, loading_dim]
    raw_loading = lo + 0.5 * (hi - lo) * (z_loading + 1.0)
    if bool((raw_loading <= 0).any()):
        raise ValueError(
            "type_quadlog takes log(loading): the cell contains non-positive raw "
            f"loading values (min {float(raw_loading.min())})."
        )

    off, span = _normalize_inverse_affine(train_x, [temp_dim, loading_dim])
    return TypeQuadLogMean(
        temp_dim=temp_dim,
        loading_dim=loading_dim,
        rest_dims=rest_dims,
        disc_dims=list(cell_features.feature_dims["cat"])
        + list(cell_features.feature_dims["solv"])
        + list(cell_features.feature_dims["base"]),
        form_id_dim=cell_features.form_id_dim,
        n_types=config.N_PRECATALYST_FORM_TYPES,
        norm_offset=off,
        norm_span=span,
        loading_bounds=(lo, hi),
        prior_loc=config.MEAN_MODULE_NORMAL_LOC,
        prior_scale=prior_scale,
        deviation_frac=config.MEAN_TYPE_DEVIATION_FRAC,
    )


def build_mean_module(
    mean_function: str,
    cell_features: CellFeatures,
    prior_scale: float | None = None,
    train_x: Tensor | None = None,
) -> Mean | None:
    """Construct a mean module by name, using prior parameters from config.

    Args:
        mean_function: One of ``"constant"``, ``"block_linear"``,
            ``"type_linear"``, ``"type_quadlog"``.
        cell_features: Provides feature_dims, active_flag_dim, form_id_dim.
        prior_scale: σ of the Normal prior on every slope/intercept. If ``None``,
            ``config.MEAN_MODULE_NORMAL_SCALE`` is used. Hyperprior sets that vary
            the mean-prior strength pass their own value here.
        train_x: ``(N, D)`` untransformed training inputs of the fold. Required
            by ``"type_quadlog"``, which must invert the model's learned
            ``Normalize`` to recover raw loading; ignored by the others.

    Returns:
        ``None`` for ``"constant"`` (gpytorch ConstantMean is the model default),
        ``BlockLinearMean`` for ``"block_linear"``, ``TypeLinearMean`` for
        ``"type_linear"``, ``TypeQuadLogMean`` for ``"type_quadlog"``.
    """
    scale = config.MEAN_MODULE_NORMAL_SCALE if prior_scale is None else prior_scale
    return _MEAN_BUILDERS[mean_function](cell_features, scale, train_x)


_MEAN_BUILDERS: dict[str, "callable"] = {
    "constant": lambda cf, scale, train_x: None,
    "cond_linear": lambda cf, scale, train_x: CondLinearMean(
        feature_dims=cf.feature_dims,
        active_flag_dim=cf.active_flag_dim,
        prior_loc=config.MEAN_MODULE_NORMAL_LOC,
        prior_scale=scale,
    ),
    "cond_trend": lambda cf, scale, train_x: None,
    "form_offset": lambda cf, scale, train_x: None,
    "block_linear": lambda cf, scale, train_x: BlockLinearMean(
        feature_dims=cf.feature_dims,
        active_flag_dim=cf.active_flag_dim,
        prior_loc=config.MEAN_MODULE_NORMAL_LOC,
        prior_scale=scale,
    ),
    "type_linear": lambda cf, scale, train_x: TypeLinearMean(
        cont_dims=list(cf.feature_dims["cont"]),
        disc_dims=list(cf.feature_dims["cat"])
        + list(cf.feature_dims["solv"])
        + list(cf.feature_dims["base"]),
        form_id_dim=cf.form_id_dim,
        n_types=config.N_PRECATALYST_FORM_TYPES,
        prior_loc=config.MEAN_MODULE_NORMAL_LOC,
        prior_scale=scale,
    ),
    "type_quadlog": _build_type_quadlog_mean,
}


def _cond_trend_term(cell_features: CellFeatures) -> LinearTrendKernel:
    """The ``cond_trend`` kernel term: the ``cond_linear`` coefficients integrated out.

    Basis: every solvent and continuous column centred at 0.5, the
    ``base_active`` flag (the base intercept) and every base column centred at
    0.5 times ``base_active``. Prior variance factors ``1/d`` per column of a
    ``d``-column block and 1 for the base intercept, the ``cond_linear`` priors.
    """
    fd = cell_features.feature_dims
    columns, weights, centre, gated = [], [], [], []
    for name, is_gated in (("solv", False), ("base", True), ("cont", False)):
        dims = fd[name]
        columns += list(dims)
        weights += [1.0 / len(dims)] * len(dims)
        centre += [True] * len(dims)
        gated += [is_gated] * len(dims)
    columns.append(cell_features.active_flag_dim)
    weights.append(1.0)
    centre.append(False)
    gated.append(False)
    return LinearTrendKernel(
        columns=columns,
        weights=weights,
        centre=centre,
        gate_dim=cell_features.active_flag_dim,
        gated=gated,
        scale_halfnormal=config.COND_TREND_SCALE_HALFNORMAL,
        scale_init=config.COND_TREND_SCALE_INIT,
    )


_MEAN_KERNEL_TERMS: dict[str, "callable"] = {
    "cond_trend": _cond_trend_term,
    "form_offset": lambda cf: FormOffsetKernel(
        form_id_dim=cf.form_id_dim,
        scale_halfnormal=config.FORM_OFFSET_SCALE_HALFNORMAL,
        scale_init=config.FORM_OFFSET_SCALE_INIT,
    ),
}
"""Mean functions realised as an integrated kernel term next to a constant mean."""


def build_mean_kernel_term(mean_function: str, cell_features: CellFeatures) -> Kernel | None:
    """The covariance term a mean function adds, or ``None``.

    ``cond_trend`` and ``form_offset`` integrate their coefficients out, so they
    are a constant mean (``build_mean_module`` returns ``None`` for both) plus
    this term added to the kernel. Every other mean function adds nothing.

    Args:
        mean_function: One of ``config.MEAN_FUNCTIONS``.
        cell_features: Provides the block columns, ``base_active`` and ``form_id``.

    Returns:
        A fresh kernel term, or ``None``.

    Raises:
        ValueError: If ``mean_function`` is not a key of ``_MEAN_BUILDERS``.
    """
    if mean_function not in _MEAN_BUILDERS:
        raise ValueError(f"Unknown mean function {mean_function!r}; expected one of {list(_MEAN_BUILDERS)}")
    builder = _MEAN_KERNEL_TERMS.get(mean_function)
    return None if builder is None else builder(cell_features)


def library_cat_reference(cell_features: CellFeatures, train_x: Tensor) -> Tensor:
    """Library catalyst rows in the normalised coordinates the kernel sees.

    Mirrors the ``Normalize`` the model fits on ``train_x``: each catalyst column
    is shifted by its training minimum and divided by its training range (the
    identity on a column whose range underflows). The rows are full assembled
    width, zero outside the catalyst block.

    Args:
        cell_features: Carries ``cat_library`` and the catalyst columns.
        train_x: ``(N, D)`` untransformed training inputs of the fold.

    Returns:
        ``(L, D)`` reference rows for ``LibraryCentredKernel``.

    Raises:
        ValueError: If ``cell_features.cat_library`` is ``None`` or its width is
            not the catalyst block's.
    """
    library = cell_features.cat_library
    cat_dims = list(cell_features.feature_dims["cat"])
    if library is None or library.shape[-1] != len(cat_dims):
        raise ValueError(
            "KS7 needs cell_features.cat_library with one column per catalyst column; got "
            f"{None if library is None else tuple(library.shape)} for {len(cat_dims)} columns"
        )
    off, span = _normalize_inverse_affine(train_x, cat_dims)
    reference = torch.zeros(library.shape[0], train_x.shape[-1], dtype=train_x.dtype)
    reference[:, cat_dims] = (library.to(train_x) - off) / span
    return reference





def make_singletask_gp(
    train_x: Tensor,
    train_y: Tensor,
    kernel: Kernel,
    *,
    mean_module: Mean | None,
    form_id_dim: int,
    noise_prior_alpha: float,
    noise_prior_beta: float,
) -> SingleTaskGP:
    """Construct a ``SingleTaskGP`` with InverseGamma noise prior.

    The ``noise_prior_beta`` is rescaled by the empirical variance of
    ``train_y`` so the prior mean β/(α-1) lives on the standardized scale
    that ``Standardize(m=1)`` produces.

    Args:
        train_x: ``(N, D)`` training inputs.
        train_y: ``(N, 1)`` training targets.
        kernel: Pre-built covariance module.
        mean_module: Optional mean module; ``None`` defers to ``ConstantMean``.
        form_id_dim: Column index of the form-ID integer in ``train_x``, or
            ``-1`` if absent.
        noise_prior_alpha: InverseGamma α (concentration).
        noise_prior_beta: InverseGamma β (rate) in squared units of
            ``train_y`` — replicate evidence pooled on the same target and
            outcome scale.

    Returns:
        Unfitted ``SingleTaskGP`` with ``Normalize`` inputs (form-ID column
        excluded) and ``Standardize(m=1)`` outcomes.

    Raises:
        ValueError: If ``train_y`` is not ``(N, 1)``.
        RuntimeError: If a registered prior has no setting closure, which
            botorch's fit fallback needs to sample it.
    """
    if train_y.ndim != 2 or train_y.shape[-1] != 1:
        raise ValueError(f"train_y must be (N, 1); got {tuple(train_y.shape)}")

    std_y = train_y.squeeze(-1).std()
    beta_standardized = noise_prior_beta / (std_y ** 2)
    noise_prior = InverseGammaPrior(
        concentration=torch.tensor(noise_prior_alpha, dtype=train_y.dtype),
        rate=beta_standardized.to(dtype=train_y.dtype),
    )
    likelihood = GaussianLikelihood(noise_prior=noise_prior)

    kwargs = {}
    if mean_module is not None:
        kwargs["mean_module"] = mean_module
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=kernel,
        likelihood=likelihood,
        input_transform=_build_normalize(train_x.shape[-1], form_id_dim),
        outcome_transform=Standardize(m=1),
        **kwargs,
    )
    _assert_all_priors_sampleable(model)
    return model


def make_kronecker_gp(
    train_x: Tensor,
    train_y: Tensor,
    kernel: Kernel,
    *,
    rank: int,
    mean_module: Mean | None,
    form_id_dim: int,
    noise_prior_alpha: float,
    noise_prior_betas: tuple[float, float],
) -> KroneckerMultiTaskGP:
    """Construct a ``KroneckerMultiTaskGP`` with InverseGamma noise prior.

    One InverseGamma is shared by both task noises and the global noise; its
    rate is the mean of the two tasks' standardized replicate rates
    (:func:`_shared_standardized_noise_rate`).

    Args:
        train_x: ``(N, D)`` training inputs.
        train_y: ``(N, 2)`` training targets.
        kernel: Pre-built data-covariance kernel (will be wrapped in
            ``_MultitaskDataAdapter``).
        rank: Rank of the ``IndexKernel`` task covariance; 2 is full rank
            for two tasks. Either value gives a single ICM term.
        mean_module: Optional single-task mean module; wrapped in
            ``MultitaskMean(num_tasks=2)`` automatically.
        form_id_dim: Column index of the form-ID integer in ``train_x``, or
            ``-1`` if absent.
        noise_prior_alpha: InverseGamma α.
        noise_prior_betas: InverseGamma β per task, in ``train_y`` column order,
            each in squared units of its own column — replicate evidence pooled
            on that target and outcome scale.

    Returns:
        Unfitted ``KroneckerMultiTaskGP`` with ``Standardize(m=2)`` outcomes.

    Raises:
        ValueError: If ``train_y`` is not ``(N, 2)``.
        RuntimeError: If a registered prior has no setting closure, which
            botorch's fit fallback needs to sample it.
    """
    if train_y.ndim != 2 or train_y.shape[-1] != 2:
        raise ValueError(f"train_y must be (N, 2); got {tuple(train_y.shape)}")

    data_covar_module = _MultitaskDataAdapter(kernel)

    beta_standardized = _shared_standardized_noise_rate(train_y, noise_prior_betas)
    noise_prior = InverseGammaPrior(
        concentration=torch.tensor(noise_prior_alpha, dtype=train_y.dtype),
        rate=beta_standardized.to(dtype=train_y.dtype),
    )
    likelihood = MultitaskGaussianLikelihood(num_tasks=2, noise_prior=noise_prior)
    _patch_multitask_likelihood_priors(likelihood, noise_prior)

    model = KroneckerMultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        likelihood=likelihood,
        data_covar_module=data_covar_module,
        rank=rank,
        task_covar_prior=None,
        input_transform=_build_normalize(train_x.shape[-1], form_id_dim),
        outcome_transform=Standardize(m=2),
    )
    task_kernel = model.covar_module.task_covar_module
    _strip_task_covar_lkj_prior(task_kernel)
    _floor_task_covar_var(task_kernel)
    _register_task_covar_priors(
        task_kernel, noise_prior_alpha, beta_standardized.to(dtype=train_y.dtype)
    )

    if mean_module is not None:
        model.mean_module = MultitaskMean(mean_module, num_tasks=2)

    _assert_all_priors_sampleable(model)
    return model


class LCMMultiTaskGP(ExactGP, GPyTorchModel):
    """Block-design multitask GP with a true Linear Coregionalization kernel.

    Unlike ``KroneckerMultiTaskGP`` (a single ICM term, one data kernel ⊗ one
    ``IndexKernel``), this model uses ``gpytorch.kernels.LCMKernel`` with ``Q``
    latent groups: the covariance is ``Σ_q K_q(x, x') ⊗ B_q``, a sum of ``Q``
    independent data kernels each with its own ``IndexKernel`` task covariance.
    It assumes the block-design case (all tasks observed at every input).

    The posterior is computed by the inherited ``GPyTorchModel.posterior`` (no
    Kronecker shortcut), which applies the input transform in eval mode and the
    outcome transform on the returned posterior. ``forward`` returns a
    ``MultitaskMultivariateNormal``.

    Args:
        train_X: ``(N, D)`` training inputs (raw; the input transform is applied
            in ``forward`` during training and in ``posterior`` during eval).
        train_Y: ``(N, T)`` training targets (raw; standardized by
            ``outcome_transform`` at construction).
        data_covar_modules: The ``Q`` data-space kernels, one per latent group.
        likelihood: A ``MultitaskGaussianLikelihood`` over ``T`` tasks.
        rank: Rank of each group's ``IndexKernel`` task covariance.
        input_transform: Applied to inputs (e.g. ``Normalize``).
        outcome_transform: Applied to targets (e.g. ``Standardize(m=T)``).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        data_covar_modules: list[Kernel],
        likelihood: MultitaskGaussianLikelihood,
        rank: int,
        input_transform: InputTransform,
        outcome_transform: OutcomeTransform,
    ) -> None:
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        outcome_transform.train()
        train_Y, _ = outcome_transform(train_Y, X=transformed_X)

        num_tasks = train_Y.shape[-1]
        super().__init__(train_X, train_Y, likelihood)
        self._num_outputs = num_tasks
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = LCMKernel(
            base_kernels=data_covar_modules,
            num_tasks=num_tasks,
            rank=rank,
            task_covar_prior=None,
        )
        self.outcome_transform = outcome_transform
        self.input_transform = input_transform
        self.to(train_X)

    def forward(self, X: Tensor) -> MultitaskMultivariateNormal:
        """Evaluate the prior over all tasks.

        The input transform is applied here only in training mode; in eval mode
        ``posterior`` has already applied it.

        Args:
            X: ``... x N x D`` inputs.

        Returns:
            ``MultitaskMultivariateNormal`` with mean of shape ``... x N x T``.
        """
        if self.training:
            X = self.transform_inputs(X)
        return MultitaskMultivariateNormal(self.mean_module(X), self.covar_module(X))


def make_lcm_gp(
    train_x: Tensor,
    train_y: Tensor,
    kernels: list[Kernel],
    *,
    rank: int,
    mean_module: Mean | None,
    form_id_dim: int,
    noise_prior_alpha: float,
    noise_prior_betas: tuple[float, float],
) -> LCMMultiTaskGP:
    """Construct a true LCM multitask GP (``LCMKernel``, Q = ``len(kernels)``).

    Mirrors ``make_kronecker_gp`` for the noise prior, likelihood patch, and the
    per-``IndexKernel`` task-covariance priors — but applies them to **each** of
    the ``Q`` latent groups' ``IndexKernel``s, so every group's cross-task
    correlation is independently sign-free and unbiased.

    Args:
        train_x: ``(N, D)`` training inputs.
        train_y: ``(N, 2)`` training targets.
        kernels: The ``Q`` data-space kernels (each pre-built and independent);
            each is wrapped in ``_MultitaskDataAdapter`` for ``active_dims``.
        rank: Rank of each group's task covariance (2 = full rank for 2 tasks).
        mean_module: Optional single-task mean; wrapped in ``MultitaskMean``.
        form_id_dim: Column index of the form-ID integer, or ``-1`` if absent.
        noise_prior_alpha: InverseGamma α.
        noise_prior_betas: InverseGamma β per task, in ``train_y`` column order,
            each in squared units of its own column — replicate evidence pooled
            on that target and outcome scale.

    Returns:
        Unfitted ``LCMMultiTaskGP`` with ``Q = len(kernels)`` latent groups.

    Raises:
        ValueError: If ``train_y`` is not ``(N, 2)``.
        RuntimeError: If a registered prior has no setting closure, which
            botorch's fit fallback needs to sample it.
    """
    if train_y.ndim != 2 or train_y.shape[-1] != 2:
        raise ValueError(f"train_y must be (N, 2); got {tuple(train_y.shape)}")

    data_covar_modules = [_MultitaskDataAdapter(k) for k in kernels]

    beta_standardized = _shared_standardized_noise_rate(train_y, noise_prior_betas)
    noise_prior = InverseGammaPrior(
        concentration=torch.tensor(noise_prior_alpha, dtype=train_y.dtype),
        rate=beta_standardized.to(dtype=train_y.dtype),
    )
    likelihood = MultitaskGaussianLikelihood(num_tasks=2, noise_prior=noise_prior)
    _patch_multitask_likelihood_priors(likelihood, noise_prior)

    model = LCMMultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        data_covar_modules=data_covar_modules,
        likelihood=likelihood,
        rank=rank,
        input_transform=_build_normalize(train_x.shape[-1], form_id_dim),
        outcome_transform=Standardize(m=2),
    )

    for multitask_kernel in model.covar_module.covar_module_list:
        task_kernel = multitask_kernel.task_covar_module
        _strip_task_covar_lkj_prior(task_kernel)
        _floor_task_covar_var(task_kernel)
        _register_task_covar_priors(
            task_kernel, noise_prior_alpha, beta_standardized.to(dtype=train_y.dtype)
        )

    if mean_module is not None:
        model.mean_module = MultitaskMean(mean_module, num_tasks=2)

    _assert_all_priors_sampleable(model)
    return model


def make_mapsaas_gp(
    train_x: Tensor,
    train_y: Tensor,
    *,
    form_id_dim: int,
) -> AdditiveMapSaasSingleTaskGP:
    """Construct an ``AdditiveMapSaasSingleTaskGP`` on a single target.

    MAP-SAAS uses a fixed additive Matérn-5/2 SAAS architecture with MAP
    inference. It deliberately ignores the configured kernel structure, the mean
    module, and the InverseGamma noise prior — only the input/outcome transforms
    (``Normalize`` excluding the form-ID column, ``Standardize(m=1)``) and
    ``num_taus`` are threaded through from this study. The form-ID column is
    dropped before the kernels (``FilterFeatures``), as in every other builder;
    the SAAS kernels see the remaining ``D - 1`` columns.

    Args:
        train_x: ``(N, D)`` training inputs.
        train_y: ``(N, 1)`` training targets.
        form_id_dim: Column index of the form-ID integer, or ``-1`` if absent.

    Returns:
        Unfitted ``AdditiveMapSaasSingleTaskGP``.

    Raises:
        ValueError: If ``train_y`` is not ``(N, 1)``.
    """
    if train_y.ndim != 2 or train_y.shape[-1] != 1:
        raise ValueError(f"train_y must be (N, 1); got {tuple(train_y.shape)}")

    d = train_x.shape[-1]
    input_transform: InputTransform = _build_normalize(d, form_id_dim)
    if form_id_dim >= 0:
        kept = torch.tensor([i for i in range(d) if i != form_id_dim], device=train_x.device)
        input_transform = ChainedInputTransform(
            normalize=input_transform, drop_form_id=FilterFeatures(feature_indices=kept)
        )
    return AdditiveMapSaasSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
        num_taus=config.MAPSAAS_NUM_TAUS,
    )


def make_vanilla_gp(
    train_x: Tensor,
    train_y: Tensor,
    *,
    form_id_dim: int,
    noise_prior_alpha: float,
    noise_prior_beta: float,
) -> SingleTaskGP:
    """Vanilla GP: flat ARD RBF, fixed outputscale, Hvarfner priors.

    Uses the dimension-scaled LogNormal prior (DSP) on ARD lengthscales and
    fixes the kernel outputscale to 1 (no ``ScaleKernel`` wrapper). Constant
    mean is the gpytorch default.

    Args:
        train_x: ``(N, D)`` training inputs.
        train_y: ``(N, 1)`` training targets.
        form_id_dim: Column index of the form-ID integer in ``train_x``, or
            ``-1`` if absent.
        noise_prior_alpha: InverseGamma α (concentration).
        noise_prior_beta: InverseGamma β (rate) in squared units of
            ``train_y`` — replicate evidence pooled on the same target and
            outcome scale.

    Returns:
        Unfitted ``SingleTaskGP`` with an ARD ``RBFKernel`` over every column
        except the form ID.

    Raises:
        ValueError: If ``train_y`` is not ``(N, 1)``.
        RuntimeError: If a registered prior has no setting closure, which
            botorch's fit fallback needs to sample it.
    """
    if train_y.ndim != 2 or train_y.shape[-1] != 1:
        raise ValueError(f"train_y must be (N, 1); got {tuple(train_y.shape)}")

    D = train_x.shape[-1]
    if form_id_dim >= 0:
        active_dims = [i for i in range(D) if i != form_id_dim]
    else:
        active_dims = list(range(D))
    ard_d = len(active_dims)

    ls_loc = config.LENGTHSCALE_LOGNORMAL_LOC_SHIFT + math.log(math.sqrt(ard_d))
    ls_prior = LogNormalPrior(
        torch.tensor(ls_loc, dtype=train_y.dtype),
        torch.tensor(config.LENGTHSCALE_LOGNORMAL_SCALE, dtype=train_y.dtype),
    )
    kernel = RBFKernel(
        ard_num_dims=ard_d,
        active_dims=active_dims,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=_vanilla_lengthscale_constraint(),
    )

    std_y = train_y.squeeze(-1).std()
    beta_standardized = noise_prior_beta / (std_y ** 2)
    noise_prior = InverseGammaPrior(
        concentration=torch.tensor(noise_prior_alpha, dtype=train_y.dtype),
        rate=beta_standardized.to(dtype=train_y.dtype),
    )
    likelihood = GaussianLikelihood(noise_prior=noise_prior)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=kernel,
        likelihood=likelihood,
        input_transform=_build_normalize(D, form_id_dim),
        outcome_transform=Standardize(m=1),
    )
    _assert_all_priors_sampleable(model)
    return model


def make_vanilla_kronecker_gp(
    train_x: Tensor,
    train_y: Tensor,
    *,
    rank: int,
    form_id_dim: int,
    noise_prior_alpha: float,
    noise_prior_betas: tuple[float, float],
) -> KroneckerMultiTaskGP:
    """Vanilla Kronecker multitask GP: flat ARD RBF, no ScaleKernel, ConstantMean.

    Same kernel philosophy as ``make_vanilla_gp`` but extended to a
    ``KroneckerMultiTaskGP`` with ``rank`` task covariance. The kernel has no
    ``ScaleKernel`` wrapper — outputscale is implicitly 1.0, absorbed by
    ``Standardize(m=2)``.

    Args:
        train_x: ``(N, D)`` training inputs.
        train_y: ``(N, 2)`` training targets.
        rank: Rank of the ``IndexKernel`` task covariance; 2 is full rank
            for two tasks. Either value gives a single ICM term.
        form_id_dim: Column index of the form-ID integer in ``train_x``, or
            ``-1`` if absent.
        noise_prior_alpha: InverseGamma α (concentration).
        noise_prior_betas: InverseGamma β per task, in ``train_y`` column order,
            each in squared units of its own column — replicate evidence pooled
            on that target and outcome scale.

    Returns:
        Unfitted ``KroneckerMultiTaskGP`` with ``Standardize(m=2)`` outcomes.

    Raises:
        ValueError: If ``train_y`` is not ``(N, 2)``.
        RuntimeError: If a registered prior has no setting closure, which
            botorch's fit fallback needs to sample it.
    """
    if train_y.ndim != 2 or train_y.shape[-1] != 2:
        raise ValueError(f"train_y must be (N, 2); got {tuple(train_y.shape)}")

    D = train_x.shape[-1]
    if form_id_dim >= 0:
        active_dims = [i for i in range(D) if i != form_id_dim]
    else:
        active_dims = list(range(D))
    ard_d = len(active_dims)

    ls_loc = config.LENGTHSCALE_LOGNORMAL_LOC_SHIFT + math.log(math.sqrt(ard_d))
    ls_prior = LogNormalPrior(
        torch.tensor(ls_loc, dtype=train_y.dtype),
        torch.tensor(config.LENGTHSCALE_LOGNORMAL_SCALE, dtype=train_y.dtype),
    )
    kernel = RBFKernel(
        ard_num_dims=ard_d,
        active_dims=active_dims,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=_vanilla_lengthscale_constraint(),
    )

    data_covar_module = _MultitaskDataAdapter(kernel)

    beta_standardized = _shared_standardized_noise_rate(train_y, noise_prior_betas)
    noise_prior = InverseGammaPrior(
        concentration=torch.tensor(noise_prior_alpha, dtype=train_y.dtype),
        rate=beta_standardized.to(dtype=train_y.dtype),
    )
    likelihood = MultitaskGaussianLikelihood(num_tasks=2, noise_prior=noise_prior)
    _patch_multitask_likelihood_priors(likelihood, noise_prior)

    model = KroneckerMultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        likelihood=likelihood,
        data_covar_module=data_covar_module,
        rank=rank,
        task_covar_prior=None,
        input_transform=_build_normalize(D, form_id_dim),
        outcome_transform=Standardize(m=2),
    )
    task_kernel = model.covar_module.task_covar_module
    _strip_task_covar_lkj_prior(task_kernel)
    _floor_task_covar_var(task_kernel)
    _register_task_covar_priors(
        task_kernel, noise_prior_alpha, beta_standardized.to(dtype=train_y.dtype)
    )

    _assert_all_priors_sampleable(model)
    return model


def make_independent_singletask_pair(
    train_x: Tensor,
    train_y_task0: Tensor,
    train_y_task1: Tensor,
    kernel_task0: Kernel,
    kernel_task1: Kernel,
    *,
    mean_module: Mean | None,
    form_id_dim: int,
    noise_prior_alpha: float,
    noise_prior_betas: tuple[float, float],
) -> tuple[SingleTaskGP, SingleTaskGP]:
    """Build two independent ``SingleTaskGP`` models, one per task.

    ``mean_module`` is deep-copied for the second model so the two GPs hold
    independent parameter tensors. Each model gets its own pre-built kernel.

    Args:
        train_x: ``(N, D)`` training inputs, shared by both models.
        train_y_task0: ``(N, 1)`` targets of the first task.
        train_y_task1: ``(N, 1)`` targets of the second task.
        kernel_task0: Pre-built covariance module for the first model.
        kernel_task1: Pre-built covariance module for the second model.
        mean_module: Optional mean module; ``None`` defers to ``ConstantMean``.
        form_id_dim: Column index of the form-ID integer in ``train_x``, or
            ``-1`` if absent.
        noise_prior_alpha: InverseGamma α (concentration).
        noise_prior_betas: InverseGamma β (rate) per task, each in squared units
            of its own targets; each model rescales its β by its own target
            variance.

    Returns:
        ``(model_0, model_1)``, each built by :func:`make_singletask_gp`.
    """
    mean_task1 = copy.deepcopy(mean_module) if mean_module is not None else None
    model_0 = make_singletask_gp(
        train_x, train_y_task0, kernel_task0,
        mean_module=mean_module,
        form_id_dim=form_id_dim,
        noise_prior_alpha=noise_prior_alpha,
        noise_prior_beta=noise_prior_betas[0],
    )
    model_1 = make_singletask_gp(
        train_x, train_y_task1, kernel_task1,
        mean_module=mean_task1,
        form_id_dim=form_id_dim,
        noise_prior_alpha=noise_prior_alpha,
        noise_prior_beta=noise_prior_betas[1],
    )
    return model_0, model_1


def make_independent_mapsaas_pair(
    train_x: Tensor,
    train_y_task0: Tensor,
    train_y_task1: Tensor,
    *,
    form_id_dim: int,
) -> tuple[AdditiveMapSaasSingleTaskGP, AdditiveMapSaasSingleTaskGP]:
    """Build two independent ``AdditiveMapSaasSingleTaskGP`` models, one per task.

    Args:
        train_x: ``(N, D)`` training inputs, shared by both models.
        train_y_task0: ``(N, 1)`` targets of the first task.
        train_y_task1: ``(N, 1)`` targets of the second task.
        form_id_dim: Column index of the form-ID integer, or ``-1`` if absent.

    Returns:
        ``(model_0, model_1)``, each built by :func:`make_mapsaas_gp`.
    """
    model_0 = make_mapsaas_gp(train_x, train_y_task0, form_id_dim=form_id_dim)
    model_1 = make_mapsaas_gp(train_x, train_y_task1, form_id_dim=form_id_dim)
    return model_0, model_1


def get_y_yield(cell_features: CellFeatures, outcome_transform: str) -> Tensor:
    """Return the yield target on the requested outcome scale.

    ``raw`` and ``logit`` return the tensors precomputed at assembly time; any
    other transform is computed from ``y_raw`` with
    :func:`pablo.utils.outcome_transforms.forward`.

    Args:
        cell_features: Assembled features holding the yield columns.
        outcome_transform: One of
            :data:`pablo.utils.outcome_transforms.OUTCOME_TRANSFORM_NAMES`.

    Returns:
        ``(N, 1)`` yield targets for every row of ``cell_features`` (not only
        the training rows).

    Raises:
        ValueError: If ``outcome_transform`` is not an implemented transform.
    """
    if outcome_transform == "raw":
        return cell_features.y_raw
    if outcome_transform == "logit":
        return cell_features.y_logit
    return forward_outcome(cell_features.y_raw, outcome_transform)


def get_y_cs(
    cell_features: CellFeatures, outcome_transform: str
) -> tuple[Tensor, Tensor]:
    """Return the conversion and selectivity targets on the requested outcome scale.

    ``raw`` and ``logit`` return the tensors precomputed at assembly time; any
    other transform is computed from the raw columns with
    :func:`pablo.utils.outcome_transforms.forward`.

    Args:
        cell_features: Assembled features holding the conversion and
            selectivity columns.
        outcome_transform: One of
            :data:`pablo.utils.outcome_transforms.OUTCOME_TRANSFORM_NAMES`.

    Returns:
        ``(conversion, selectivity)``, each ``(N, 1)`` over every row of
        ``cell_features``.

    Raises:
        ValueError: If ``outcome_transform`` is not an implemented transform.
    """
    if outcome_transform == "raw":
        return cell_features.y_conversion_raw, cell_features.y_selectivity_raw
    if outcome_transform == "logit":
        return cell_features.y_conversion_logit, cell_features.y_selectivity_logit
    return (
        forward_outcome(cell_features.y_conversion_raw, outcome_transform),
        forward_outcome(cell_features.y_selectivity_raw, outcome_transform),
    )


def assemble_kronecker_targets(
    cell_features: CellFeatures,
    targets: tuple[str, str],
    outcome_transform: str,
    train_idx: list[int],
) -> Tensor:
    """Stack two targets on the training rows for a multitask GP.

    Args:
        cell_features: Assembled features holding all target columns.
        targets: Two labels from ``"Y"`` (yield), ``"C"`` (conversion) and
            ``"S"`` (selectivity), in output column order.
        outcome_transform: One of
            :data:`pablo.utils.outcome_transforms.OUTCOME_TRANSFORM_NAMES`.
        train_idx: Row indices of the training set.

    Returns:
        ``(len(train_idx), 2)`` target tensor.

    Raises:
        ValueError: If ``outcome_transform`` is not an implemented transform.
    """
    y_yield = get_y_yield(cell_features, outcome_transform)
    y_c, y_s = get_y_cs(cell_features, outcome_transform)
    target_map: dict[str, Tensor] = {"Y": y_yield, "C": y_c, "S": y_s}
    return torch.cat(
        [target_map[targets[0]][train_idx], target_map[targets[1]][train_idx]],
        dim=-1,
    )


class AcyloinCondensationModel:
    """Catalog of GP model classes for NHC acyloin condensation.

    Each public method (``st_y``, ``indep_*``, ``icm_*``, ``lcm_*``,
    ``mapsaas_*``) builds one fitted-ready model — to add a new model class, add a
    method and an entry in ``SPECS``; to drop one, delete its method.

    All priors come from ``config.py``. Each target's noise prior is
    ``config.NOISE_EVIDENCE[(target, outcome_transform)]``: the replicate
    evidence of that target on the scale the model is trained on.

    Args:
        cell_features: Assembled features for all data points.
        train_idx: Training-set indices.
        kernel_structure: One of ``config.KERNEL_STRUCTURES``, or "Vanilla".
        base_kernel_type: "matern52", "matern32", "matern12", "rbf",
            "tanimoto", or "minmax".
        outcome_transform: Outcome scale of the training targets, one of
            :data:`pablo.utils.outcome_transforms.OUTCOME_TRANSFORM_NAMES`;
            passed to ``get_y_yield`` / ``get_y_cs`` when a model is built.
        lengthscale_scale: σ of the LogNormal lengthscale prior — typically
            data-adapted as ``min(1.0, sqrt(N_train / N_params))``.
        prior_scheme: A key of ``config.PRIOR_SCHEMES`` ("P1", "P2", "P3").
            ``cell_features.partition`` must already have been built at that
            scheme's granularity — ``make_kernel_from_config`` rejects the pair
            if it was not. ``None`` leaves the dim mode to ``config``.
        mean_function: One of ``config.MEAN_FUNCTIONS``. Only its kernel term
            (``build_mean_kernel_term``) is used here; the mean module itself is
            passed to ``build``.
        target_scale: Factor the cell's raw targets were multiplied by before
            the build (e.g. 100 for percent). Every noise evidence β is
            multiplied by ``target_scale²`` so the prior stays in the targets'
            squared units. Only ``outcome_transform="raw"`` allows a value other
            than 1.

    Raises:
        ValueError: If ``target_scale != 1`` with a non-raw outcome transform,
            or ``target_scale`` is not positive.
    """

    SPECS: dict[str, ModelSpec] = {
        # model_class      architecture   family     targets    rank yield_src    y_idx sel_src     s_idx
        "st_y":       ModelSpec("st_y",       "singletask",  "gp",      ("Y",),     0, "direct",     None, "none",     None),
        "indep_cs":   ModelSpec("indep_cs",   "independent", "gp",      ("C", "S"), 0, "product_cs", None, "direct",   1),
        "indep_ys":   ModelSpec("indep_ys",   "independent", "gp",      ("Y", "S"), 0, "direct",     0,    "direct",   1),
        "indep_yc":   ModelSpec("indep_yc",   "independent", "gp",      ("Y", "C"), 0, "direct",     0,    "ratio_yc", None),
        "icm_cs":     ModelSpec("icm_cs",     "icm",         "gp",      ("C", "S"), 2, "product_cs", None, "marginal", 1),
        "icm_ys":     ModelSpec("icm_ys",     "icm",         "gp",      ("Y", "S"), 2, "marginal",   0,    "marginal", 1),
        "icm_yc":     ModelSpec("icm_yc",     "icm",         "gp",      ("Y", "C"), 2, "marginal",   0,    "ratio_yc", None),
        "lcm_cs":     ModelSpec("lcm_cs",     "lcm",         "gp",      ("C", "S"), 2, "product_cs", None, "marginal", 1),
        "lcm_ys":     ModelSpec("lcm_ys",     "lcm",         "gp",      ("Y", "S"), 2, "marginal",   0,    "marginal", 1),
        "lcm_yc":     ModelSpec("lcm_yc",     "lcm",         "gp",      ("Y", "C"), 2, "marginal",   0,    "ratio_yc", None),
        "mapsaas_y":  ModelSpec("mapsaas_y",  "singletask",  "mapsaas", ("Y",),     0, "direct",     None, "none",     None),
        "mapsaas_cs": ModelSpec("mapsaas_cs", "independent", "mapsaas", ("C", "S"), 0, "product_cs", None, "direct",   1),
        "mapsaas_ys": ModelSpec("mapsaas_ys", "independent", "mapsaas", ("Y", "S"), 0, "direct",     0,    "direct",   1),
        "mapsaas_yc": ModelSpec("mapsaas_yc", "independent", "mapsaas", ("Y", "C"), 0, "direct",     0,    "ratio_yc", None),
        "logsum_cy":  ModelSpec("logsum_cy",  "logsum",      "gp",      ("C", "Y"), 0, "logsum",     None, "logsum",   None),
        "condprod_cy": ModelSpec("condprod_cy", "condprod",  "gp",      ("C", "Y"), 0, "condprod",   None, "condprod", None),
    }

    def __init__(
        self,
        cell_features: CellFeatures,
        train_idx: list[int],
        kernel_structure: str,
        base_kernel_type: str,
        outcome_transform: str,
        lengthscale_scale: float,
        prior_scheme: str | None = None,
        mean_function: str = "constant",
        target_scale: float = 1.0,
    ) -> None:
        if target_scale <= 0.0:
            raise ValueError(f"target_scale must be positive, got {target_scale}")
        if target_scale != 1.0 and outcome_transform != "raw":
            raise ValueError(
                f"target_scale={target_scale} rescales raw targets; outcome_transform "
                f"{outcome_transform!r} is not linear in them"
            )
        self.target_scale = target_scale
        self.cell_features = cell_features
        self.mean_function = mean_function
        self.train_idx = train_idx
        self.kernel_structure = kernel_structure
        self.base_kernel_type = base_kernel_type
        self.outcome_transform = outcome_transform

        self.lengthscale_scale = lengthscale_scale
        self.prior_scheme = prior_scheme

        self.train_x = cell_features.X[train_idx]

    @classmethod
    def get_spec(cls, model_class: str) -> ModelSpec:
        """Return the ``ModelSpec`` for *model_class* or raise ``ValueError``."""
        if model_class not in cls.SPECS:
            raise ValueError(
                f"Unknown model_class {model_class!r}; "
                f"expected one of {sorted(cls.SPECS)}"
            )
        return cls.SPECS[model_class]

    def build(self, model_class: str, mean_module: Mean | None):
        """Dispatch to the method named ``model_class``.

        Args:
            model_class: One of the keys of ``SPECS``.
            mean_module: Mean module (or ``None`` for ``ConstantMean``).

        Returns:
            A ``SingleTaskGP`` / ``AdditiveMapSaasSingleTaskGP`` for the single
            classes, a 2-tuple of single models for the ``indep_*`` / ``mapsaas_*``
            pairs, a ``KroneckerMultiTaskGP`` for ``icm_*``, an
            ``LCMMultiTaskGP`` for ``lcm_*``, a ``LogSumModel`` for
            ``logsum_cy`` or a ``CondProdModel`` for ``condprod_cy``.
        """
        method = getattr(self, model_class, None)
        if method is None or model_class not in self.SPECS:
            raise ValueError(
                f"Unknown model_class {model_class!r}; "
                f"expected one of {sorted(self.SPECS)}"
            )
        return method(mean_module)

    def st_y(self, mean_module: Mean | None) -> SingleTaskGP:
        """Single-task GP on yield."""
        train_y = self._target_tensor("Y")
        evidence = self._noise_evidence("Y")
        if self.kernel_structure == "Vanilla":
            return make_vanilla_gp(
                self.train_x, train_y,
                form_id_dim=self.cell_features.form_id_dim,
                noise_prior_alpha=evidence.alpha,
                noise_prior_beta=evidence.beta,
            )
        return self._build_singletask(train_y, evidence, self._build_kernel(), mean_module)

    def indep_cs(self, mean_module: Mean | None):
        """Two independent SingleTaskGPs on (conversion, selectivity)."""
        return self._build_independent(("C", "S"), mean_module)

    def indep_ys(self, mean_module: Mean | None):
        """Two independent SingleTaskGPs on (yield, selectivity)."""
        return self._build_independent(("Y", "S"), mean_module)

    def indep_yc(self, mean_module: Mean | None):
        """Two independent SingleTaskGPs on (yield, conversion)."""
        return self._build_independent(("Y", "C"), mean_module)

    def icm_cs(self, mean_module: Mean | None) -> KroneckerMultiTaskGP:
        """Full-rank (rank-2) ICM KroneckerMultiTaskGP on (conversion, selectivity)."""
        return self._build_kronecker(("C", "S"), rank=2, mean_module=mean_module)

    def icm_ys(self, mean_module: Mean | None) -> KroneckerMultiTaskGP:
        """Full-rank (rank-2) ICM KroneckerMultiTaskGP on (yield, selectivity)."""
        return self._build_kronecker(("Y", "S"), rank=2, mean_module=mean_module)

    def icm_yc(self, mean_module: Mean | None) -> KroneckerMultiTaskGP:
        """Full-rank (rank-2) ICM KroneckerMultiTaskGP on (yield, conversion)."""
        return self._build_kronecker(("Y", "C"), rank=2, mean_module=mean_module)

    def lcm_cs(self, mean_module: Mean | None) -> LCMMultiTaskGP:
        """True LCM (Q=2, rank-2) multitask GP on (conversion, selectivity)."""
        return self._build_lcm(("C", "S"), rank=2, mean_module=mean_module)

    def lcm_ys(self, mean_module: Mean | None) -> LCMMultiTaskGP:
        """True LCM (Q=2, rank-2) multitask GP on (yield, selectivity)."""
        return self._build_lcm(("Y", "S"), rank=2, mean_module=mean_module)

    def lcm_yc(self, mean_module: Mean | None) -> LCMMultiTaskGP:
        """True LCM (Q=2, rank-2) multitask GP on (yield, conversion)."""
        return self._build_lcm(("Y", "C"), rank=2, mean_module=mean_module)

    def mapsaas_y(self, mean_module: Mean | None) -> AdditiveMapSaasSingleTaskGP:
        """Single AdditiveMapSaasSingleTaskGP on yield (ignores kernel/mean)."""
        return make_mapsaas_gp(
            self.train_x, self._target_tensor("Y"),
            form_id_dim=self.cell_features.form_id_dim,
        )

    def mapsaas_cs(self, mean_module: Mean | None):
        """Two independent AdditiveMapSaasSingleTaskGPs on (conversion, selectivity)."""
        return self._build_independent_mapsaas(("C", "S"))

    def mapsaas_ys(self, mean_module: Mean | None):
        """Two independent AdditiveMapSaasSingleTaskGPs on (yield, selectivity)."""
        return self._build_independent_mapsaas(("Y", "S"))

    def mapsaas_yc(self, mean_module: Mean | None):
        """Two independent AdditiveMapSaasSingleTaskGPs on (yield, conversion)."""
        return self._build_independent_mapsaas(("Y", "C"))

    def logsum_cy(self, mean_module: Mean | None) -> LogSumModel:
        """Log-space LMC on (conversion, yield) with fixed mixing: ``log Y = log C + log S``.

        Requires ``outcome_transform == "raw"``: the targets enter as raw
        fractions and the class takes their logarithm itself. Each latent
        (``g_C``, ``g_S``) gets its own kernel and its own copy of the mean. A row's
        log-scale noise is ``σ²_g / a_i²`` with ``a_i`` the training-row replicate
        mean of its target (:func:`replicate_anchor_values`) and ``σ²_g`` the raw
        replicate variance times a multiplier under ``InverseGamma(α, β / v)``. Rows
        with ``Y = 0`` are modelled at ``config.LOGSUM_ZERO_YIELD`` with noise
        evaluated at ``config.LOGSUM_ZERO_NOISE_ANCHOR``.

        Raises:
            ValueError: If ``outcome_transform`` is not ``"raw"``.
        """
        if self.outcome_transform != "raw":
            raise ValueError(f"logsum_cy takes raw targets and logs them itself; got outcome_transform={self.outcome_transform!r}")
        conversion = self._target_tensor("C")
        yield_ = self._target_tensor("Y")
        group_ids = self.cell_features.replicate_id[self.train_idx]
        evidence = (self._noise_evidence("C"), self._noise_evidence("Y"))
        priors = tuple(
            InverseGammaPrior(
                concentration=torch.tensor(e.alpha, dtype=yield_.dtype),
                rate=torch.tensor(e.beta / e.variance, dtype=yield_.dtype),
            )
            for e in evidence
        )
        conversion_mean = ConstantMean() if mean_module is None else mean_module
        model = make_logsum_model(
            self.train_x,
            conversion,
            yield_,
            replicate_anchor_values(conversion.squeeze(-1), group_ids),
            replicate_anchor_values(yield_.squeeze(-1), group_ids),
            self._build_kernel(),
            self._build_kernel(),
            conversion_mean=conversion_mean,
            selectivity_mean=copy.deepcopy(conversion_mean),
            form_id_dim=self.cell_features.form_id_dim,
            replicate_variances=(evidence[0].variance, evidence[1].variance),
            noise_priors=(priors[0], priors[1]),
            zero_yield=config.LOGSUM_ZERO_YIELD * self.target_scale,
            zero_noise_anchor=config.LOGSUM_ZERO_NOISE_ANCHOR * self.target_scale,
            noise_floor=config.LOGSUM_NOISE_MULTIPLIER_FLOOR,
        )
        _assert_all_priors_sampleable(model.gp)
        return model

    def condprod_cy(self, mean_module: Mean | None) -> CondProdModel:
        """Conditional product: ``Y_i = C_i · s(x_i) + ε_i`` plus a GP on conversion.

        Requires ``outcome_transform == "probit"``, the scale of the conversion
        GP; the yield model is on the raw scale by construction. The noise ``τ²``
        of the yield model carries the raw yield replicate prior. Each GP gets
        its own kernel and its own copy of the mean.

        Raises:
            ValueError: If ``outcome_transform`` is not ``"probit"``.
        """
        if self.outcome_transform != "probit":
            raise ValueError(f"condprod_cy fits its conversion GP on probit; got outcome_transform={self.outcome_transform!r}")
        conversion_evidence = self._noise_evidence("C")
        conversion_gp = self._build_singletask(
            self._target_tensor("C"), conversion_evidence, self._build_kernel(), mean_module
        )
        yield_raw = get_y_yield(self.cell_features, "raw")[self.train_idx]
        yield_evidence = config.NOISE_EVIDENCE[("Y", "raw")]
        std_y = yield_raw.squeeze(-1).std()
        noise_prior = InverseGammaPrior(
            concentration=torch.tensor(yield_evidence.alpha, dtype=yield_raw.dtype),
            rate=(yield_evidence.beta / std_y ** 2).to(dtype=yield_raw.dtype),
        )
        model = make_condprod_model(
            conversion_gp,
            self.train_x,
            get_y_cs(self.cell_features, "raw")[0][self.train_idx],
            yield_raw,
            self._build_kernel(),
            selectivity_mean=None if mean_module is None else copy.deepcopy(mean_module),
            form_id_dim=self.cell_features.form_id_dim,
            yield_noise_prior=noise_prior,
            conversion_transform=self.outcome_transform,
        )
        _assert_all_priors_sampleable(model.selectivity_gp)
        return model

    def _build_kernel(self) -> Kernel:
        """A fresh covariance module: the kernel structure plus the mean function's term.

        KS7 gets the library catalyst rows normalised on this fold's training
        inputs; ``cond_trend`` and ``form_offset`` add their integrated term.
        """
        kernel = make_kernel_from_config(
            kernel_structure=self.kernel_structure,
            feature_dims=self.cell_features.feature_dims,
            active_flag_dim=self.cell_features.active_flag_dim,
            base_kernel_type=self.base_kernel_type,
            partition=self.cell_features.partition,
            lengthscale_scale=self.lengthscale_scale,
            prior_scheme=self.prior_scheme,
            cat_reference=(
                library_cat_reference(self.cell_features, self.train_x)
                if self.kernel_structure == "KS7" else None
            ),
        )
        term = build_mean_kernel_term(self.mean_function, self.cell_features)
        return kernel if term is None else AdditiveKernel(kernel, term)

    def _build_singletask(
        self,
        train_y: Tensor,
        evidence: ReplicateNoiseEvidence,
        kernel: Kernel,
        mean_module: Mean | None,
    ) -> SingleTaskGP:
        return make_singletask_gp(
            self.train_x, train_y, kernel,
            mean_module=mean_module,
            form_id_dim=self.cell_features.form_id_dim,
            noise_prior_alpha=evidence.alpha,
            noise_prior_beta=evidence.beta,
        )

    def _build_independent(
        self, targets: tuple[str, str], mean_module: Mean | None,
    ) -> tuple[SingleTaskGP, SingleTaskGP]:
        alpha, betas = self._noise_prior_pair(targets)
        return make_independent_singletask_pair(
            self.train_x,
            self._target_tensor(targets[0]),
            self._target_tensor(targets[1]),
            self._build_kernel(),
            self._build_kernel(),
            mean_module=mean_module,
            form_id_dim=self.cell_features.form_id_dim,
            noise_prior_alpha=alpha,
            noise_prior_betas=betas,
        )

    def _build_independent_mapsaas(
        self, targets: tuple[str, str],
    ) -> tuple[AdditiveMapSaasSingleTaskGP, AdditiveMapSaasSingleTaskGP]:
        return make_independent_mapsaas_pair(
            self.train_x,
            self._target_tensor(targets[0]),
            self._target_tensor(targets[1]),
            form_id_dim=self.cell_features.form_id_dim,
        )

    def _build_kronecker(
        self,
        targets: tuple[str, str],
        rank: int,
        mean_module: Mean | None,
    ) -> KroneckerMultiTaskGP:
        alpha, betas = self._noise_prior_pair(targets)
        if self.kernel_structure == "Vanilla":
            return make_vanilla_kronecker_gp(
                self.train_x,
                self._assemble_kronecker_targets(targets),
                rank=rank,
                form_id_dim=self.cell_features.form_id_dim,
                noise_prior_alpha=alpha,
                noise_prior_betas=betas,
            )
        return make_kronecker_gp(
            self.train_x,
            self._assemble_kronecker_targets(targets),
            self._build_kernel(),
            rank=rank,
            mean_module=mean_module,
            form_id_dim=self.cell_features.form_id_dim,
            noise_prior_alpha=alpha,
            noise_prior_betas=betas,
        )

    def _build_lcm(
        self,
        targets: tuple[str, str],
        rank: int,
        mean_module: Mean | None,
    ) -> LCMMultiTaskGP:
        alpha, betas = self._noise_prior_pair(targets)
        return make_lcm_gp(
            self.train_x,
            self._assemble_kronecker_targets(targets),
            [self._build_kernel(), self._build_kernel()],
            rank=rank,
            mean_module=mean_module,
            form_id_dim=self.cell_features.form_id_dim,
            noise_prior_alpha=alpha,
            noise_prior_betas=betas,
        )

    def _noise_evidence(self, target: str) -> ReplicateNoiseEvidence:
        """Replicate noise evidence for ``target`` on this model's outcome scale.

        Variance-valued fields are multiplied by ``target_scale²``.
        """
        evidence = config.NOISE_EVIDENCE[(target, self.outcome_transform)]
        s2 = self.target_scale ** 2
        return dataclasses.replace(
            evidence,
            sum_squares=evidence.sum_squares * s2,
            variance=evidence.variance * s2,
            beta=evidence.beta * s2,
        )

    def _noise_prior_pair(
        self, targets: tuple[str, str]
    ) -> tuple[float, tuple[float, float]]:
        """Shared α and per-target β for a two-target model, in target order.

        Raises:
            ValueError: If the two targets' replicate evidence differs in
                degrees of freedom, so no single α describes both.
        """
        first, second = (self._noise_evidence(t) for t in targets)
        if first.dof != second.dof:
            raise ValueError(
                f"Replicate evidence for {targets} differs in degrees of freedom "
                f"({first.dof} vs {second.dof}); a shared InverseGamma α needs both equal."
            )
        return first.alpha, (first.beta, second.beta)

    def _target_tensor(self, target: str) -> Tensor:
        """Train-subset target tensor for one of ``"Y"``, ``"C"``, ``"S"``."""
        y_yield = get_y_yield(self.cell_features, self.outcome_transform)
        y_c, y_s = get_y_cs(self.cell_features, self.outcome_transform)
        return {"Y": y_yield, "C": y_c, "S": y_s}[target][self.train_idx]

    def _assemble_kronecker_targets(self, targets: tuple[str, str]) -> Tensor:
        return assemble_kronecker_targets(
            self.cell_features, targets, self.outcome_transform, self.train_idx
        )


_FB_CONFIG_KEYS: tuple[str, ...] = (
    "cat_feat", "solv_feat", "base_feat", "kernel_structure", "base_kernel",
    "model_class", "mean_function", "outcome_transform", "prior_scheme",
)
"""Configuration axes a fully Bayesian model is built from: the config columns of
``gem_scores.parquet``."""

_FB_SUPPORTED: dict[str, str] = {
    "model_class": "st_y",
    "kernel_structure": "KS1",
    "mean_function": "constant",
}
"""The one level of each of these axes that ``FullyBayesianScanGP`` implements."""


@dataclass(frozen=True)
class NutsSettings:
    """Budget of one fully Bayesian fit.

    Attributes:
        num_chains: Independent NUTS chains, each seeded ``seed + chain``.
        warmup_steps: Adaptation steps per chain, discarded.
        num_samples: Post-warmup draws per chain.
        thinning: Every ``thinning``-th post-warmup draw of each chain is kept
            for the posterior mixture; diagnostics use every draw.
        max_tree_depth: NUTS maximum tree depth (at most ``2**depth - 1``
            leapfrog steps per iteration).
    """

    num_chains: int
    warmup_steps: int
    num_samples: int
    thinning: int
    max_tree_depth: int = 6

    def __post_init__(self) -> None:
        for name in ("num_chains", "num_samples", "thinning", "max_tree_depth"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")

    @property
    def draws_per_chain(self) -> int:
        """Draws each chain contributes to the mixture after thinning."""
        return len(range(0, self.num_samples, self.thinning))


@dataclass(frozen=True)
class ChainResult:
    """One NUTS chain of a ``FullyBayesianScanGP``.

    Attributes:
        seed: The pyro RNG seed the chain ran under.
        samples: Site name -> ``num_samples x *site_shape`` post-warmup draws,
            unthinned, on the constrained scale.
        divergences: Post-warmup divergent iterations.
        nonpd_evaluations: Log-joint evaluations, warmup included, whose
            covariance was not positive definite (log density -inf).
        potential_evaluations: Log-joint evaluations, warmup included.
        wall_time_s: Wall time of the chain.
    """

    seed: int
    samples: dict[str, Tensor]
    divergences: int
    nonpd_evaluations: int
    potential_evaluations: int
    wall_time_s: float


@dataclass(frozen=True)
class MCMCDiagnostics:
    """Convergence record of a fully Bayesian fit.

    Attributes:
        divergences_per_chain: Post-warmup divergences of each chain.
        nonpd_evaluations: Non-positive-definite log-joint evaluations, all chains.
        potential_evaluations: Log-joint evaluations, all chains.
        rhat_max: Largest split-R̂ over every scalar hyperparameter; NaN when
            any R̂ is NaN.
        ess_min: Smallest bulk effective sample size over every scalar
            hyperparameter, from all post-warmup draws of all chains.
        num_draws_kept: Mixture components after thinning, ``S``.
        wall_time_s: Summed wall time of the chains.
    """

    divergences_per_chain: tuple[int, ...]
    nonpd_evaluations: int
    potential_evaluations: int
    rhat_max: float
    ess_min: float
    num_draws_kept: int
    wall_time_s: float


class MCMCDiagnosticsError(RuntimeError):
    """Raised when a fully Bayesian fit fails ``config.FB_*`` convergence thresholds."""


def check_mcmc_diagnostics(diagnostics: MCMCDiagnostics) -> None:
    """Raise unless a fit meets every ``config.FB_*`` threshold.

    A NaN R̂ or ESS fails its check.

    Args:
        diagnostics: The record of the fit.

    Raises:
        MCMCDiagnosticsError: Naming every threshold the fit fails.
    """
    failures = []
    divergences = sum(diagnostics.divergences_per_chain)
    if divergences > config.FB_MAX_DIVERGENCES:
        failures.append(
            f"{divergences} post-warmup divergences > {config.FB_MAX_DIVERGENCES} "
            f"(per chain {diagnostics.divergences_per_chain})"
        )
    if not diagnostics.rhat_max <= config.FB_RHAT_MAX:
        failures.append(f"max split-R̂ {diagnostics.rhat_max} > {config.FB_RHAT_MAX}")
    if not diagnostics.ess_min >= config.FB_ESS_MIN:
        failures.append(f"min bulk ESS {diagnostics.ess_min} < {config.FB_ESS_MIN}")
    if failures:
        raise MCMCDiagnosticsError("; ".join(failures))


@dataclass(frozen=True)
class KS1BlockPrior:
    """One block kernel of KS1 as the fully Bayesian model rebuilds it.

    Attributes:
        name: One of ``kernels.BLOCKS``.
        active_dims: Columns of the assembled tensor the block reads.
        groups: Lengthscale groups in the block's local coordinates.
        base_kernel_type: A key of ``kernels.BASE_KERNELS``.
        lengthscale_loc: Location of the LogNormal prior on each group lengthscale.
        lengthscale_scale: Scale of that LogNormal.
    """

    name: str
    active_dims: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    base_kernel_type: str
    lengthscale_loc: float
    lengthscale_scale: float


@dataclass(frozen=True)
class KS1PriorSpec:
    """Every prior of an ``st_y`` · KS1 · constant model, read from its MAP build.

    Attributes:
        blocks: One entry per block, in ``kernels.BLOCKS`` order.
        outputscale_concentration: Gamma α of the outputscale.
        outputscale_rate: Gamma β of the outputscale.
        noise_concentration: InverseGamma α of the noise variance.
        noise_rate: InverseGamma β of the noise variance on the standardized scale.
    """

    blocks: tuple[KS1BlockPrior, ...]
    outputscale_concentration: float
    outputscale_rate: float
    noise_concentration: float
    noise_rate: float


def ks1_prior_spec(template: SingleTaskGP) -> KS1PriorSpec:
    """Read the priors of a MAP ``st_y`` · KS1 · constant model.

    The fully Bayesian model takes every prior parameter from this MAP build, so
    the two share one source. The MAP floors (``GreaterThan`` constraints) are
    not part of the spec: the fully Bayesian priors are untruncated.

    Args:
        template: ``AcyloinCondensationModel(...).st_y(None)`` for KS1.

    Returns:
        The prior spec.

    Raises:
        ValueError: If the model is not a ScaleKernel over a product of one
            ``PartialARDKernel`` per block with LogNormal lengthscale priors, a
            Gamma outputscale prior and an InverseGamma noise prior.
    """
    covar = template.covar_module
    if not (isinstance(covar, ScaleKernel) and isinstance(covar.base_kernel, ProductKernel)):
        raise ValueError(f"expected ScaleKernel(ProductKernel(...)), got {type(covar).__name__}")
    kernels = list(covar.base_kernel.kernels)
    if len(kernels) != len(BLOCKS) or not all(isinstance(k, PartialARDKernel) for k in kernels):
        raise ValueError(f"expected {len(BLOCKS)} PartialARDKernel blocks, got {[type(k).__name__ for k in kernels]}")
    names_by_type = {kernel_type: name for name, kernel_type in BASE_KERNELS.items()}
    blocks = []
    for name, kernel in zip(BLOCKS, kernels):
        prior = kernel.lengthscale_prior
        if not isinstance(prior, LogNormalPrior):
            raise ValueError(f"block {name!r} lengthscale prior is {type(prior).__name__}, not LogNormalPrior")
        group_index = kernel.group_index.tolist()
        groups = tuple(
            tuple(i for i, g in enumerate(group_index) if g == group) for group in range(kernel.num_groups)
        )
        active_dims = kernel.active_dims
        if active_dims is None:
            raise ValueError(f"block {name!r} has no active_dims")
        blocks.append(KS1BlockPrior(
            name=name,
            active_dims=tuple(int(d) for d in active_dims.tolist()),
            groups=groups,
            base_kernel_type=names_by_type[type(kernel.base_kernel)],
            lengthscale_loc=float(prior.loc),
            lengthscale_scale=float(prior.scale),
        ))
    outputscale_prior = covar.outputscale_prior
    noise_prior = template.likelihood.noise_covar.noise_prior
    if not isinstance(outputscale_prior, GammaPrior):
        raise ValueError(f"outputscale prior is {type(outputscale_prior).__name__}, not GammaPrior")
    if not isinstance(noise_prior, InverseGammaPrior):
        raise ValueError(f"noise prior is {type(noise_prior).__name__}, not InverseGammaPrior")
    return KS1PriorSpec(
        blocks=tuple(blocks),
        outputscale_concentration=float(outputscale_prior.concentration),
        outputscale_rate=float(outputscale_prior.rate),
        noise_concentration=float(noise_prior.concentration),
        noise_rate=float(noise_prior.rate),
    )


class KS1PyroModel(PyroModel):
    """Pyro log joint of ``st_y`` · KS1 · constant on the transformed training data.

    Sites, all on the standardized outcome scale:

      ``lengthscale_<block>``  ``(G_b,)`` LogNormal, one per lengthscale group.
      ``outputscale``          ``()`` Gamma.
      ``noise``                ``()`` InverseGamma.
      ``mean``                 ``()`` flat (improper) on the real line.

    ``K = outputscale · Π_b k_b(X_b / ℓ_b) + noise · I`` with ``k_b`` the block's
    unit-lengthscale base kernel from ``kernels.BASE_KERNELS``, the kernel the MAP
    model evaluates. A covariance whose Cholesky factorization fails gives log
    density -inf, which NUTS records as a divergence; ``nonpd_evaluations``
    counts those evaluations.

    Shapes: ``train_X`` ``N x D`` after the input transform, ``train_Y``
    ``N x 1`` standardized; ``K`` is ``N x N``.

    Args:
        spec: The priors; ``None`` until ``set_spec`` is called.
        use_input_warping: Must be ``False``.
        indices_to_warp: Unused.
        eps: Unused.
    """

    def __init__(
        self,
        spec: KS1PriorSpec | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
        eps: float = 1e-7,
    ) -> None:
        if use_input_warping:
            raise ValueError("KS1PyroModel does not support input warping")
        super().__init__(use_input_warping=False, indices_to_warp=indices_to_warp, eps=eps)
        self.spec: KS1PriorSpec | None = None
        self.potential_evaluations = 0
        self.nonpd_evaluations = 0
        if spec is not None:
            self.set_spec(spec)

    def set_spec(self, spec: KS1PriorSpec) -> None:
        """Attach the priors and build the unit base kernels."""
        self.spec = spec
        self._unit_kernels = [BASE_KERNELS[b.base_kernel_type]() for b in spec.blocks]
        self._dims = [torch.tensor(b.active_dims, dtype=torch.long) for b in spec.blocks]
        self._group_index = []
        for b in spec.blocks:
            index = torch.empty(len(b.active_dims), dtype=torch.long)
            for g, group in enumerate(b.groups):
                index[list(group)] = g
            self._group_index.append(index)

    def _require_spec(self) -> KS1PriorSpec:
        if self.spec is None:
            raise RuntimeError("KS1PyroModel needs set_spec before it is sampled")
        return self.spec

    def covariance(self, lengthscales: Sequence[Tensor], outputscale: Tensor, X: Tensor) -> Tensor:
        """Noiseless KS1 covariance.

        Args:
            lengthscales: One ``(G_b,)`` group-lengthscale vector per block.
            outputscale: Scalar outputscale.
            X: ``N x D`` transformed inputs.

        Returns:
            ``N x N`` covariance.
        """
        K = outputscale * torch.ones(X.shape[-2], X.shape[-2], dtype=X.dtype, device=X.device)
        for kernel, dims, index, ls in zip(self._unit_kernels, self._dims, self._group_index, lengthscales):
            xb = X.index_select(-1, dims.to(X.device)) / ls.index_select(-1, index.to(X.device))
            K = K * kernel.forward(xb, xb)
        return K

    def sample(self) -> None:
        """Pyro model of the hyperparameters and the training targets."""
        spec = self._require_spec()
        X, y = self.train_X, self.train_Y.squeeze(-1)
        tkwargs = {"dtype": X.dtype, "device": X.device}
        lengthscales = [
            pyro.sample(
                f"lengthscale_{b.name}",
                pdist.LogNormal(
                    torch.full((len(b.groups),), b.lengthscale_loc, **tkwargs),
                    torch.full((len(b.groups),), b.lengthscale_scale, **tkwargs),
                ).to_event(1),
            )
            for b in spec.blocks
        ]
        outputscale = pyro.sample(
            "outputscale",
            pdist.Gamma(
                torch.tensor(spec.outputscale_concentration, **tkwargs),
                torch.tensor(spec.outputscale_rate, **tkwargs),
            ),
        )
        noise = pyro.sample(
            "noise",
            pdist.InverseGamma(
                torch.tensor(spec.noise_concentration, **tkwargs),
                torch.tensor(spec.noise_rate, **tkwargs),
            ),
        )
        mean = pyro.sample("mean", pdist.ImproperUniform(pdist.constraints.real, (), ()))
        n = X.shape[-2]
        covariance = self.covariance(lengthscales, outputscale, X) + noise * torch.eye(n, **tkwargs)
        scale_tril, info = torch.linalg.cholesky_ex(covariance)
        self.potential_evaluations += 1
        if int(info) != 0:
            self.nonpd_evaluations += 1
            # The flat-priored mean has no other path into the graph; NUTS differentiates w.r.t. it.
            pyro.factor("Y", torch.tensor(-math.inf, **tkwargs) + 0.0 * mean)
            return
        pyro.factor("Y", pdist.MultivariateNormal(mean.expand(n), scale_tril=scale_tril).log_prob(y))

    def postprocess_mcmc_samples(self, mcmc_samples: dict[str, Tensor]) -> dict[str, Tensor]:
        """The samples as drawn."""
        return mcmc_samples

    def get_dummy_mcmc_samples(self, num_mcmc_samples: int, **tkwargs) -> dict[str, Tensor]:
        """Ones of the shape ``load_mcmc_samples`` expects, for state-dict loading."""
        spec = self._require_spec()
        samples = {
            f"lengthscale_{b.name}": torch.ones(num_mcmc_samples, len(b.groups), **tkwargs)
            for b in spec.blocks
        }
        for site in ("outputscale", "noise", "mean"):
            samples[site] = torch.ones(num_mcmc_samples, **tkwargs)
        return samples

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood, None]:
        """Batched gpytorch modules holding ``S`` draws.

        The block kernels are ``PartialARDKernel`` with an MCMC batch dimension and
        ``Positive`` constraints, inside ``ScaleKernel(ProductKernel(...))``.

        Args:
            mcmc_samples: Site name -> ``S x *site_shape``.

        Returns:
            ``(mean_module, covar_module, likelihood, None)``, each with batch shape ``S``.
        """
        spec = self._require_spec()
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        num = mcmc_samples["mean"].shape[0]
        batch_shape = torch.Size([num])
        blocks = []
        for b in spec.blocks:
            kernel = PartialARDKernel(
                base_kernel_type=b.base_kernel_type,
                num_dims=len(b.active_dims),
                partition=[list(group) for group in b.groups],
                lengthscale_prior=LogNormalPrior(b.lengthscale_loc, b.lengthscale_scale),
                lengthscale_constraint=Positive(),
                active_dims=list(b.active_dims),
                batch_shape=batch_shape,
            ).to(**tkwargs)
            kernel._set_lengthscale(mcmc_samples[f"lengthscale_{b.name}"].to(**tkwargs).reshape(num, 1, -1))
            blocks.append(kernel)
        covar_module = ScaleKernel(ProductKernel(*blocks), batch_shape=batch_shape).to(**tkwargs)
        covar_module.outputscale = mcmc_samples["outputscale"].to(**tkwargs).reshape(num)
        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data, new_value=mcmc_samples["mean"].to(**tkwargs)
        )
        likelihood = GaussianLikelihood(batch_shape=batch_shape, noise_constraint=Positive()).to(**tkwargs)
        likelihood.noise_covar.noise = reshape_and_detach(
            target=likelihood.noise_covar.noise, new_value=mcmc_samples["noise"].to(**tkwargs)
        )
        return mean_module, covar_module, likelihood, None


class FullyBayesianScanGP(AbstractFullyBayesianSingleTaskGP):
    """Fully Bayesian counterpart of a MAP scan configuration, fitted with NUTS.

    Supports ``st_y`` · ``KS1`` · ``constant`` under any featurization, base
    kernel, outcome transform and prior scheme. The priors are those of the MAP
    model of the same configuration (``ks1_prior_spec``), without its
    lengthscale and noise floors, and the constant mean is flat. Inputs pass the
    MAP ``Normalize`` (form_id excluded), targets ``Standardize(m=1)``.

    Fit with ``fit_nuts``, or ``run_chain`` per chain and ``load_chains`` to merge
    chains run elsewhere. ``posterior(X)`` for ``X`` ``b x q x D`` returns a
    ``GaussianMixturePosterior`` whose mean and variance are ``S x b x q x 1``
    (``S`` kept draws) on the outcome-transform scale (e.g. probit).

    Args:
        train_X: ``N x D`` untransformed training inputs.
        train_Y: ``N x 1`` targets on the outcome-transform scale.
        prior_spec: The priors, from ``ks1_prior_spec``.
        input_transform: The MAP model's input transform.
        outcome_transform: The MAP model's outcome transform.
    """

    _pyro_model_class = KS1PyroModel

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        prior_spec: KS1PriorSpec,
        input_transform: InputTransform,
        outcome_transform: OutcomeTransform,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        pyro_model = KS1PyroModel(prior_spec)
        pyro_model.set_inputs(train_X=self.pyro_model.train_X, train_Y=self.pyro_model.train_Y)
        self.pyro_model = pyro_model
        self.ks1_pyro_model = pyro_model
        self.prior_spec = prior_spec
        self.configuration: dict[str, str] = {}
        self.diagnostics: MCMCDiagnostics | None = None

    @classmethod
    def from_config(
        cls,
        configuration: Mapping[str, object],
        cell_features: CellFeatures,
        train_idx: list[int],
    ) -> FullyBayesianScanGP:
        """Build the fully Bayesian model of one scan configuration.

        Args:
            configuration: At least the keys of ``_FB_CONFIG_KEYS``; a row of
                ``gem_scores.parquet`` as a dict works, extra keys are ignored.
                ``cell_features`` must be the cell of its ``cat_feat`` /
                ``solv_feat`` / ``base_feat``, with the partition of its
                ``prior_scheme``.
            cell_features: Assembled features for all data points.
            train_idx: Training-set indices.

        Returns:
            The unfitted model.

        Raises:
            KeyError: If a configuration key is missing.
            ValueError: If the configuration is outside ``st_y`` · KS1 · constant.
        """
        missing = [key for key in _FB_CONFIG_KEYS if key not in configuration]
        if missing:
            raise KeyError(f"configuration lacks {missing}")
        cfg = {key: str(configuration[key]) for key in _FB_CONFIG_KEYS}
        for axis, level in _FB_SUPPORTED.items():
            if cfg[axis] != level:
                raise ValueError(
                    f"FullyBayesianScanGP supports {axis}={level!r} only, got {cfg[axis]!r}"
                )
        template = AcyloinCondensationModel(
            cell_features=cell_features,
            train_idx=train_idx,
            kernel_structure=cfg["kernel_structure"],
            base_kernel_type=cfg["base_kernel"],
            outcome_transform=cfg["outcome_transform"],
            lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE,
            prior_scheme=cfg["prior_scheme"],
            mean_function=cfg["mean_function"],
        ).st_y(None)
        train_X = cell_features.X[train_idx]
        model = cls(
            train_X=train_X,
            train_Y=get_y_yield(cell_features, cfg["outcome_transform"])[train_idx],
            prior_spec=ks1_prior_spec(template),
            input_transform=_build_normalize(train_X.shape[-1], cell_features.form_id_dim),
            outcome_transform=Standardize(m=1),
        )
        model.configuration = cfg
        return model

    def run_chain(self, settings: NutsSettings, seed: int) -> ChainResult:
        """Run one NUTS chain without loading it.

        Seeds pyro's global RNG with ``seed``. The chain starts from the mean
        constant at 0 and every other site uniform in ``(-2, 2)`` on its
        unconstrained scale.

        Args:
            settings: Warmup, draws and tree depth; ``num_chains`` and
                ``thinning`` are not used here.
            seed: RNG seed of the chain.

        Returns:
            The chain's unthinned post-warmup draws and counters.
        """
        pyro_model = self.ks1_pyro_model
        pyro_model.potential_evaluations = 0
        pyro_model.nonpd_evaluations = 0
        tkwargs = {"dtype": pyro_model.train_X.dtype, "device": pyro_model.train_X.device}
        pyro.set_rng_seed(seed)
        kernel = NUTS(
            pyro_model.sample,
            full_mass=True,
            max_tree_depth=settings.max_tree_depth,
            init_strategy=init_to_value(values={"mean": torch.zeros((), **tkwargs)}),
            jit_compile=False,
            max_plate_nesting=0,
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=settings.warmup_steps,
            num_samples=settings.num_samples,
            num_chains=1,
            disable_progbar=True,
        )
        start = time.time()
        mcmc.run()
        wall = time.time() - start
        divergences = mcmc.diagnostics()["divergences"]["chain 0"]
        return ChainResult(
            seed=seed,
            samples={k: v.detach().clone() for k, v in mcmc.get_samples().items()},
            divergences=len(divergences),
            nonpd_evaluations=pyro_model.nonpd_evaluations,
            potential_evaluations=pyro_model.potential_evaluations,
            wall_time_s=wall,
        )

    def load_chains(
        self, chains: Sequence[ChainResult], settings: NutsSettings, strict: bool = True
    ) -> MCMCDiagnostics:
        """Merge chains into the model and record their diagnostics.

        Split-R̂ and bulk ESS are computed per scalar hyperparameter over every
        post-warmup draw; each chain is then thinned and the draws concatenated
        chain-major into ``S = num_chains · draws_per_chain`` mixture components.

        Args:
            chains: ``settings.num_chains`` results of ``run_chain``.
            settings: The settings the chains ran under.
            strict: Whether to call ``check_diagnostics`` after loading.

        Returns:
            The diagnostics, also stored as ``self.diagnostics``.

        Raises:
            ValueError: If the chain count, sites or draw counts disagree with
                ``settings``.
            MCMCDiagnosticsError: If ``strict`` and a threshold is failed.
        """
        if len(chains) != settings.num_chains:
            raise ValueError(f"expected {settings.num_chains} chains, got {len(chains)}")
        sites = sorted(chains[0].samples)
        for chain in chains:
            if sorted(chain.samples) != sites:
                raise ValueError(f"chain {chain.seed} has sites {sorted(chain.samples)}, expected {sites}")
            counts = {site: chain.samples[site].shape[0] for site in sites}
            if any(count != settings.num_samples for count in counts.values()):
                raise ValueError(f"chain {chain.seed} has {counts} draws, expected {settings.num_samples}")
        rhats, esses = [], []
        for site in sites:
            stacked = torch.stack([chain.samples[site] for chain in chains])
            flat = stacked.reshape(len(chains), settings.num_samples, -1)
            rhats.append(split_gelman_rubin(flat, chain_dim=0, sample_dim=1).reshape(-1))
            esses.append(effective_sample_size(flat, chain_dim=0, sample_dim=1).reshape(-1))
        kept = {
            site: torch.cat([chain.samples[site][:: settings.thinning] for chain in chains])
            for site in sites
        }
        self.load_mcmc_samples(kept)
        self.eval()
        self.diagnostics = MCMCDiagnostics(
            divergences_per_chain=tuple(chain.divergences for chain in chains),
            nonpd_evaluations=sum(chain.nonpd_evaluations for chain in chains),
            potential_evaluations=sum(chain.potential_evaluations for chain in chains),
            rhat_max=float(torch.cat(rhats).max()),
            ess_min=float(torch.cat(esses).min()),
            num_draws_kept=int(kept["mean"].shape[0]),
            wall_time_s=sum(chain.wall_time_s for chain in chains),
        )
        if strict:
            self.check_diagnostics()
        return self.diagnostics

    def fit_nuts(self, settings: NutsSettings, seed: int, strict: bool = True) -> MCMCDiagnostics:
        """Run ``settings.num_chains`` chains in sequence, seeds ``seed + c``, and load them.

        Args:
            settings: The NUTS budget.
            seed: Seed of chain 0.
            strict: Whether to raise on failed diagnostics.

        Returns:
            The diagnostics.
        """
        chains = [self.run_chain(settings, seed + c) for c in range(settings.num_chains)]
        return self.load_chains(chains, settings, strict=strict)

    def check_diagnostics(self) -> None:
        """Raise ``MCMCDiagnosticsError`` unless the loaded fit meets ``config.FB_*``.

        Raises:
            RuntimeError: If no chains have been loaded.
            MCMCDiagnosticsError: If a threshold is failed.
        """
        if self.diagnostics is None:
            raise RuntimeError("no chains loaded; call fit_nuts or load_chains first")
        check_mcmc_diagnostics(self.diagnostics)

    def predictive_components(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Per-draw Gaussian predictive of an observation, on the outcome-transform scale.

        The mixture over draws, weighted equally, is the predictive distribution.

        Args:
            X: ``n x D`` untransformed inputs.

        Returns:
            ``(m, s)``, each ``S x n``: mean and SD of the observation (latent
            variance plus noise) under each draw.

        Raises:
            ValueError: If ``X`` is not two-dimensional.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be n x D, got {tuple(X.shape)}")
        with torch.no_grad():
            posterior = self.posterior(X, observation_noise=True)
        return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1).sqrt()
