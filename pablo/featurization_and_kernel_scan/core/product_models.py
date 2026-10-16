"""Two-output GPs that build ``yield = conversion × selectivity`` into the model.

Each experiment measures conversion ``C`` and yield ``Y``; selectivity is the
derived ``S = Y / C``. The two model classes here impose that product exactly
instead of leaving a task covariance to learn it, and both are fitted on the
measured pair ``(C, Y)``:

  ``logsum_cy``    On the log scale the product is a sum. Two independent latent
                   GPs ``g_C = log C`` and ``g_S = log S`` are observed through
                   ``log C = g_C`` and ``log Y = g_C + g_S``: a linear model of
                   coregionalization whose mixing is fixed, not learned. Exact
                   and conjugate. A row's log-scale noise is its raw measurement
                   variance divided by the squared value, so low-yield rows count
                   less.
  ``condprod_cy``  ``Y_i = C_i · s(x_i) + ε_i`` with the measured ``C_i`` as a known
                   scale at the training rows: a GP on ``S`` with per-row noise
                   ``τ² / C_i²``, built as the kernel ``C_i C_j k_s(x_i, x_j)``. A
                   separate GP on transformed conversion supplies ``C`` at new
                   inputs.

Both return raw-fraction predictive samples of conversion, yield and selectivity
(:class:`ProductSamples`), which ``fit_eval`` scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.means import ConstantMean, Mean
from gpytorch.priors import Prior
from linear_operator.operators import DiagLinearOperator
from torch import Tensor

from pablo.utils.outcome_transforms import inverse as inverse_outcome

TASK_CONVERSION: int = 0
TASK_YIELD: int = 1
TASK_SELECTIVITY: int = 2
"""Task indices of the ``logsum_cy`` task column. Training rows carry conversion
and yield; selectivity rows exist only at prediction."""

CONVERSION_LOADING: tuple[float, float, float] = (1.0, 1.0, 0.0)
"""Loading of ``g_C`` on the tasks (conversion, yield, selectivity)."""

SELECTIVITY_LOADING: tuple[float, float, float] = (0.0, 1.0, 1.0)
"""Loading of ``g_S`` on the tasks (conversion, yield, selectivity)."""


def append_column(X: Tensor, value: float | Tensor) -> Tensor:
    """Append one column to ``X``.

    Args:
        X: ``... x n x d`` inputs.
        value: A scalar broadcast to every row, or an ``... x n x 1`` tensor.

    Returns:
        ``... x n x (d + 1)`` inputs.
    """
    if isinstance(value, Tensor):
        column = value.to(X)
    else:
        column = torch.full((*X.shape[:-1], 1), float(value), dtype=X.dtype, device=X.device)
    return torch.cat([X, column], dim=-1)


def normalize_excluding(d: int, excluded: list[int]) -> Normalize:
    """``Normalize`` over every column of a ``d``-column input except ``excluded``.

    Negative entries of ``excluded`` (an absent form-ID column) are ignored.
    """
    skip = {i for i in excluded if i >= 0}
    return Normalize(d=d, indices=[i for i in range(d) if i not in skip])


class ScaleOnlyStandardize(Standardize):
    """``Standardize`` that divides by the training standard deviation and does not centre.

    A product or sum relation between outputs, ``Y = C·s`` or ``log Y = log C +
    log S``, survives a common rescaling but not a shift, so the means stay 0.
    """

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Divide by the standard deviation, learned from ``Y`` in train mode.

        Args:
            Y: ``batch_shape x n x m`` targets.
            Yvar: ``batch_shape x n x m`` observation noises, or ``None``.
            X: Unused.

        Returns:
            The scaled targets and noises.
        """
        if self.training:
            super().forward(Y=Y, Yvar=Yvar, X=X)
            self.means = torch.zeros_like(self.means)
        Yvar_tf = None if Yvar is None else Yvar / self._stdvs_sq
        return Y / self.stdvs, Yvar_tf


class ConversionScaledKernel(Kernel):
    """``k((x, c), (x', c')) = c · c' · k_s(x, x')``, with ``c`` read from one input column.

    ``k_s`` must not read the conversion column (its ``active_dims`` exclude it).

    Args:
        base_kernel: The selectivity kernel ``k_s``.
        column: Index of the conversion column.
    """

    is_stationary = False

    def __init__(self, base_kernel: Kernel, column: int) -> None:
        super().__init__()
        self.base_kernel = base_kernel
        self.column = column

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """Covariance ``n1 x n2`` (``n1`` when ``diag``) of ``x1`` ``... x n1 x D`` and ``x2``."""
        c1 = x1[..., self.column]
        c2 = x2[..., self.column]
        if diag:
            return c1 * c2 * self.base_kernel(x1, x2, diag=True)
        base = self.base_kernel(x1, x2).to_dense()
        return c1.unsqueeze(-1) * c2.unsqueeze(-2) * base


class ColumnScaledMean(Mean):
    """``m(x) = x[column] · inner(x)``.

    Args:
        inner: The mean of the latent function.
        column: Index of the scaling column.
    """

    def __init__(self, inner: Mean, column: int) -> None:
        super().__init__()
        self.inner = inner
        self.column = column

    def forward(self, x: Tensor) -> Tensor:
        """Mean ``... x n`` of ``x`` ``... x n x D``."""
        return x[..., self.column] * self.inner(x)


class FixedMixingKernel(Kernel):
    """LMC kernel with fixed loadings over the tasks (conversion, yield, selectivity).

    ``K((x, t), (x', t')) = a_t a_t' k_C(x, x') + b_t b_t' k_S(x, x')`` with
    ``a = CONVERSION_LOADING = (1, 1, 0)`` and ``b = SELECTIVITY_LOADING = (0, 1, 1)``,
    the covariance of ``(g_C, g_C + g_S, g_S)`` for independent ``g_C ~ GP(0, k_C)``
    and ``g_S ~ GP(0, k_S)``. ``k_C`` and ``k_S`` must not read the task column.

    Args:
        conversion_kernel: ``k_C``.
        selectivity_kernel: ``k_S``.
        task_column: Index of the task column (values 0, 1, 2).
    """

    is_stationary = False

    def __init__(self, conversion_kernel: Kernel, selectivity_kernel: Kernel, task_column: int) -> None:
        super().__init__()
        self.conversion_kernel = conversion_kernel
        self.selectivity_kernel = selectivity_kernel
        self.task_column = task_column
        self.register_buffer("conversion_loading", torch.tensor(CONVERSION_LOADING, dtype=torch.float64))
        self.register_buffer("selectivity_loading", torch.tensor(SELECTIVITY_LOADING, dtype=torch.float64))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """Covariance ``n1 x n2`` (``n1`` when ``diag``) of ``x1`` ``... x n1 x D`` and ``x2``."""
        t1 = x1[..., self.task_column].round().long()
        t2 = x2[..., self.task_column].round().long()
        a1, a2 = self.conversion_loading[t1], self.conversion_loading[t2]
        b1, b2 = self.selectivity_loading[t1], self.selectivity_loading[t2]
        if diag:
            return (
                a1 * a2 * self.conversion_kernel(x1, x2, diag=True)
                + b1 * b2 * self.selectivity_kernel(x1, x2, diag=True)
            )
        k_c = self.conversion_kernel(x1, x2).to_dense()
        k_s = self.selectivity_kernel(x1, x2).to_dense()
        return (
            a1.unsqueeze(-1) * a2.unsqueeze(-2) * k_c
            + b1.unsqueeze(-1) * b2.unsqueeze(-2) * k_s
        )


class LogSumMean(Mean):
    """``m(x, t) = a_t m_C(x) + b_t m_S(x)``, the mean of :class:`FixedMixingKernel`'s tasks.

    Args:
        conversion_mean: ``m_C``.
        selectivity_mean: ``m_S``.
        task_column: Index of the task column.
    """

    def __init__(self, conversion_mean: Mean, selectivity_mean: Mean, task_column: int) -> None:
        super().__init__()
        self.conversion_mean = conversion_mean
        self.selectivity_mean = selectivity_mean
        self.task_column = task_column
        self.register_buffer("conversion_loading", torch.tensor(CONVERSION_LOADING, dtype=torch.float64))
        self.register_buffer("selectivity_loading", torch.tensor(SELECTIVITY_LOADING, dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        """Mean ``... x n`` of ``x`` ``... x n x D``."""
        t = x[..., self.task_column].round().long()
        return (
            self.conversion_loading[t] * self.conversion_mean(x)
            + self.selectivity_loading[t] * self.selectivity_mean(x)
        )


class TaskShapedNoise(Noise):
    """Diagonal noise ``σ²_{g(i)} · shape_i`` on the training rows, one ``σ²`` per task group.

    ``σ²_g = v_g λ_g``: ``v_g`` is the pooled replicate variance of task ``g`` on
    the raw scale and ``λ_g`` a learnable multiplier with an InverseGamma prior;
    ``shape_i`` turns the raw variance into the variance on the fitted scale. Only
    the training rows have shapes.

    Args:
        point_shape: ``(n_train,)`` shapes.
        task_index: ``(n_train,)`` group of each row, ``0`` conversion, ``1`` yield.
        replicate_variance: ``(2,)`` ``v_g``, raw-scale squared units.
        priors: Prior on ``λ_conversion`` and on ``λ_yield``.
        floor: Lower bound of each ``λ``.
    """

    def __init__(
        self,
        point_shape: Tensor,
        task_index: Tensor,
        replicate_variance: Tensor,
        priors: tuple[Prior, Prior],
        floor: float,
    ) -> None:
        super().__init__()
        self.register_buffer("point_shape", point_shape)
        self.register_buffer("task_index", task_index)
        self.register_buffer("replicate_variance", replicate_variance)
        dtype = point_shape.dtype
        self.register_parameter("raw_noise_conversion", torch.nn.Parameter(torch.zeros((), dtype=dtype)))
        self.register_parameter("raw_noise_yield", torch.nn.Parameter(torch.zeros((), dtype=dtype)))
        self.register_constraint("raw_noise_conversion", GreaterThan(floor))
        self.register_constraint("raw_noise_yield", GreaterThan(floor))
        self.register_prior(
            "noise_conversion_prior", priors[0],
            lambda m: m.noise_conversion, lambda m, v: m._set_noise_conversion(v),
        )
        self.register_prior(
            "noise_yield_prior", priors[1],
            lambda m: m.noise_yield, lambda m, v: m._set_noise_yield(v),
        )

    @property
    def noise_conversion(self) -> Tensor:
        """``λ_conversion``."""
        return self.raw_noise_conversion_constraint.transform(self.raw_noise_conversion)

    @property
    def noise_yield(self) -> Tensor:
        """``λ_yield``."""
        return self.raw_noise_yield_constraint.transform(self.raw_noise_yield)

    def _set_noise_conversion(self, value: Tensor) -> None:
        raw = self.raw_noise_conversion_constraint.inverse_transform(value.to(self.raw_noise_conversion))
        self.initialize(raw_noise_conversion=raw)

    def _set_noise_yield(self, value: Tensor) -> None:
        raw = self.raw_noise_yield_constraint.inverse_transform(value.to(self.raw_noise_yield))
        self.initialize(raw_noise_yield=raw)

    def measurement_variances(self) -> tuple[Tensor, Tensor]:
        """``(σ²_C, σ²_Y)``, raw-scale squared units."""
        return (
            self.replicate_variance[0] * self.noise_conversion,
            self.replicate_variance[1] * self.noise_yield,
        )

    def training_variances(self) -> Tensor:
        """``(n_train,)`` noise variances on the fitted scale."""
        sigma2 = torch.stack(self.measurement_variances())
        return sigma2[self.task_index] * self.point_shape

    def forward(self, *params, shape: torch.Size | None = None, **kwargs) -> DiagLinearOperator:
        """Diagonal noise of the ``n_train`` training rows.

        Raises:
            RuntimeError: If asked for a number of rows other than ``n_train``.
        """
        n = shape[-1] if shape is not None else params[0].shape[-2]
        if n != self.point_shape.shape[-1]:
            raise RuntimeError(f"TaskShapedNoise holds {self.point_shape.shape[-1]} shapes, asked for {n} rows")
        return DiagLinearOperator(self.training_variances())


@dataclass
class ProductSamples:
    """Raw-fraction predictive samples at ``n`` test inputs.

    Attributes:
        conversion: ``(S, n)`` latent conversion (no measurement noise).
        yield_: ``(S, n)`` yield of a new measurement.
        selectivity: ``(S, n)`` selectivity of a new measurement, ``Y / C`` with the
            yield and conversion noise propagated.
    """

    conversion: Tensor
    yield_: Tensor
    selectivity: Tensor


@dataclass
class LogSumModel:
    """A fitted-ready ``logsum_cy`` model.

    Attributes:
        gp: Single-output ``SingleTaskGP`` over task-augmented inputs ``(x, t)``,
            trained on ``log C`` (``t = 0``) and ``log Y`` (``t = 1``) rows.
        train_x: ``(2N, D + 1)`` its training inputs, conversion rows first.
    """

    gp: SingleTaskGP
    train_x: Tensor

    def noise_module(self) -> TaskShapedNoise:
        """The likelihood's :class:`TaskShapedNoise`.

        Raises:
            TypeError: If the likelihood holds another noise model.
        """
        noise = self.gp.likelihood.noise_covar
        if not isinstance(noise, TaskShapedNoise):
            raise TypeError(f"logsum_cy likelihood holds {type(noise).__name__}, not TaskShapedNoise")
        return noise

    def predictive_samples(self, test_X: Tensor, n_samples: int) -> ProductSamples:
        """Draw ``(g_C, g_S)`` jointly at ``test_X`` and add raw-scale measurement noise.

        ``C = exp(g_C)``, ``S = exp(g_S)``, ``Y = C·S + ε_Y`` with ``ε_Y ~ N(0, σ²_Y)``.
        Selectivity adds ``N(0, (σ²_Y + S² σ²_C) / C²)``, the first-order noise of
        ``Y / C`` for independent yield and conversion errors.

        Args:
            test_X: ``(n, D)`` inputs without the task column.
            n_samples: Number of joint draws ``S``.

        Returns:
            The samples, each ``(S, n)``.
        """
        n = test_X.shape[-2]
        flat = torch.cat(
            [append_column(test_X, TASK_CONVERSION), append_column(test_X, TASK_SELECTIVITY)], dim=-2
        )
        latent = self.gp.posterior(flat).rsample(sample_shape=torch.Size([n_samples])).squeeze(-1)
        conversion = latent[..., :n].exp()
        selectivity = latent[..., n:].exp()
        var_c, var_y = self.noise_module().measurement_variances()
        latent_yield = conversion * selectivity
        yield_ = latent_yield + torch.randn_like(latent_yield) * var_y.sqrt()
        sel_sd = ((var_y + selectivity.pow(2) * var_c) / conversion.pow(2)).sqrt()
        return ProductSamples(
            conversion=conversion,
            yield_=yield_,
            selectivity=selectivity + torch.randn_like(selectivity) * sel_sd,
        )


def make_logsum_model(
    train_x: Tensor,
    conversion: Tensor,
    yield_: Tensor,
    conversion_anchor: Tensor,
    yield_anchor: Tensor,
    conversion_kernel: Kernel,
    selectivity_kernel: Kernel,
    *,
    conversion_mean: Mean,
    selectivity_mean: Mean,
    form_id_dim: int,
    replicate_variances: tuple[float, float],
    noise_priors: tuple[Prior, Prior],
    zero_yield: float,
    zero_noise_anchor: float,
    noise_floor: float,
) -> LogSumModel:
    """Build a ``logsum_cy`` model on the training rows.

    Rows with ``Y = 0`` take ``log(zero_yield)`` as their log yield and
    ``zero_noise_anchor`` as the value their yield noise is evaluated at. Targets
    are divided by one shared standard deviation and not centred
    (:class:`ScaleOnlyStandardize`), so ``log Y = log C + log S`` holds on the
    fitted scale.

    Args:
        train_x: ``(N, D)`` training inputs.
        conversion: ``(N, 1)`` raw conversion fractions, all positive.
        yield_: ``(N, 1)`` raw yield fractions, non-negative.
        conversion_anchor: ``(N,)`` conversion each row's noise is evaluated at.
        yield_anchor: ``(N,)`` yield each row's noise is evaluated at.
        conversion_kernel: ``k_C``; must not read column ``D``.
        selectivity_kernel: ``k_S``; must not read column ``D``.
        conversion_mean: ``m_C``.
        selectivity_mean: ``m_S``.
        form_id_dim: Column index of the form-ID integer, or ``-1``.
        replicate_variances: ``(v_C, v_Y)`` pooled raw replicate variances.
        noise_priors: Priors on the multipliers ``λ_C``, ``λ_Y`` of ``v_C``, ``v_Y``.
        zero_yield: Raw yield a zero-yield row is modelled at.
        zero_noise_anchor: Raw yield a zero-yield row's noise is evaluated at.
        noise_floor: Lower bound of ``λ_C`` and ``λ_Y``.

    Returns:
        The unfitted model.

    Raises:
        ValueError: If a conversion is not positive, a yield is negative, or an
            anchor of a non-zero row is not positive.
    """
    c = conversion.squeeze(-1)
    y = yield_.squeeze(-1)
    bad_c = torch.nonzero(c <= 0).flatten().tolist()
    if bad_c:
        raise ValueError(f"logsum_cy needs positive conversion; training rows {bad_c} have {c[bad_c].tolist()}")
    bad_y = torch.nonzero(y < 0).flatten().tolist()
    if bad_y:
        raise ValueError(f"logsum_cy needs non-negative yield; training rows {bad_y} have {y[bad_y].tolist()}")
    zero = y == 0
    y_model = torch.where(zero, torch.full_like(y, zero_yield), y)
    y_anchor = torch.where(zero, torch.full_like(y, zero_noise_anchor), yield_anchor.to(y))
    anchors = torch.cat([conversion_anchor.to(c), y_anchor])
    bad_anchor = torch.nonzero(anchors <= 0).flatten().tolist()
    if bad_anchor:
        raise ValueError(f"logsum_cy noise anchors must be positive; flat rows {bad_anchor} are not")

    n, d = train_x.shape
    flat_x = torch.cat(
        [append_column(train_x, TASK_CONVERSION), append_column(train_x, TASK_YIELD)], dim=-2
    )
    flat_y = torch.cat([c.log(), y_model.log()]).unsqueeze(-1)
    gp = SingleTaskGP(
        train_X=flat_x,
        train_Y=flat_y,
        likelihood=GaussianLikelihood(),
        covar_module=FixedMixingKernel(conversion_kernel, selectivity_kernel, task_column=d),
        mean_module=LogSumMean(conversion_mean, selectivity_mean, task_column=d),
        input_transform=normalize_excluding(d + 1, [form_id_dim, d]),
        outcome_transform=ScaleOnlyStandardize(m=1),
    )
    scale = gp.outcome_transform.stdvs.squeeze()
    task_index = torch.cat([
        torch.full((n,), TASK_CONVERSION, dtype=torch.long),
        torch.full((n,), TASK_YIELD, dtype=torch.long),
    ])
    gp.likelihood.noise_covar = TaskShapedNoise(
        point_shape=1.0 / (anchors.pow(2) * scale.pow(2)),
        task_index=task_index,
        replicate_variance=torch.tensor(replicate_variances, dtype=train_x.dtype),
        priors=noise_priors,
        floor=noise_floor,
    ).to(train_x)
    return LogSumModel(gp=gp, train_x=flat_x)


@dataclass
class CondProdModel:
    """A fitted-ready ``condprod_cy`` model.

    Attributes:
        conversion_gp: GP on conversion under ``conversion_transform``.
        selectivity_gp: GP on yield over ``(x, c)`` with kernel ``c c' k_s`` and mean
            ``c μ_s``; queried at ``c = 1`` it returns the latent selectivity.
        selectivity_train_x: ``(N, D + 1)`` its training inputs, measured raw
            conversion in the last column.
        conversion_transform: Outcome transform of the conversion GP.
    """

    conversion_gp: SingleTaskGP
    selectivity_gp: SingleTaskGP
    selectivity_train_x: Tensor
    conversion_transform: str

    def yield_noise_variance(self) -> Tensor:
        """Fitted ``τ²``, raw yield squared units."""
        stdvs = self.selectivity_gp.outcome_transform.stdvs.squeeze()
        return self.selectivity_gp.likelihood.noise.squeeze() * stdvs.pow(2)

    def predictive_samples(self, test_X: Tensor, n_samples: int) -> ProductSamples:
        """Draw conversion and selectivity independently and form the yield.

        ``Y = C·s + ε`` with ``ε ~ N(0, τ²)``; ``C`` is the back-transformed latent
        conversion, since ``τ`` already absorbs the conversion error of the
        training rows. Selectivity is ``s + ε / C̄`` with ``C̄`` the per-point mean
        of the conversion draws.

        Args:
            test_X: ``(n, D)`` inputs.
            n_samples: Number of draws ``S``.

        Returns:
            The samples, each ``(S, n)``.

        Raises:
            ValueError: If a test point's mean conversion is not positive.
        """
        shape = torch.Size([n_samples])
        f_c = self.conversion_gp.posterior(test_X).rsample(sample_shape=shape).squeeze(-1)
        conversion = inverse_outcome(f_c, self.conversion_transform)
        s = self.selectivity_gp.posterior(append_column(test_X, 1.0)).rsample(sample_shape=shape).squeeze(-1)
        eps = torch.randn_like(s) * self.yield_noise_variance().sqrt()
        mean_conversion = conversion.mean(dim=0)
        if bool((mean_conversion <= 0).any()):
            raise ValueError(f"condprod_cy mean conversion is not positive at test points: {mean_conversion.tolist()}")
        return ProductSamples(
            conversion=conversion,
            yield_=conversion * s + eps,
            selectivity=s + eps / mean_conversion,
        )


def make_condprod_model(
    conversion_gp: SingleTaskGP,
    train_x: Tensor,
    conversion: Tensor,
    yield_: Tensor,
    selectivity_kernel: Kernel,
    *,
    selectivity_mean: Mean | None,
    form_id_dim: int,
    yield_noise_prior: Prior,
    conversion_transform: str,
) -> CondProdModel:
    """Build a ``condprod_cy`` model around an unfitted conversion GP.

    The selectivity GP sees ``(x, c)`` with ``c`` the measured raw conversion; its
    targets are divided by their standard deviation and not centred
    (:class:`ScaleOnlyStandardize`), so ``E[Y_i] = C_i μ_s(x_i)`` holds on the
    fitted scale.

    Args:
        conversion_gp: GP on conversion under ``conversion_transform``.
        train_x: ``(N, D)`` training inputs.
        conversion: ``(N, 1)`` measured raw conversion fractions.
        yield_: ``(N, 1)`` raw yield fractions.
        selectivity_kernel: ``k_s``; must not read column ``D``.
        selectivity_mean: ``μ_s``; ``None`` for a constant.
        form_id_dim: Column index of the form-ID integer, or ``-1``.
        yield_noise_prior: Prior on ``τ²`` on the scaled yield scale.
        conversion_transform: Outcome transform the conversion GP was built on.

    Returns:
        The unfitted model.
    """
    d = train_x.shape[-1]
    x_aug = append_column(train_x, conversion)
    inner = ConstantMean() if selectivity_mean is None else selectivity_mean
    selectivity_gp = SingleTaskGP(
        train_X=x_aug,
        train_Y=yield_,
        likelihood=GaussianLikelihood(noise_prior=yield_noise_prior),
        covar_module=ConversionScaledKernel(selectivity_kernel, column=d),
        mean_module=ColumnScaledMean(inner, column=d),
        input_transform=normalize_excluding(d + 1, [form_id_dim, d]),
        outcome_transform=ScaleOnlyStandardize(m=1),
    )
    return CondProdModel(
        conversion_gp=conversion_gp,
        selectivity_gp=selectivity_gp,
        selectivity_train_x=x_aug,
        conversion_transform=conversion_transform,
    )
