"""Prior distributions and prior constructors for the featurization x kernel scan.

Four things live here, all consumed by ``kernels.py``:

  1. ``DirichletPrior`` — a symmetric Dirichlet density evaluated on the
     softmax-transformed additive weights of ``DirichletAdditiveKernel``.
  2. The LogNormal lengthscale prior, whose location is ``log(sqrt(D)) + shift``.
     Which ``D`` that is — the block's own width or the whole design's width — is
     a swept axis, not a fixed convention; see ``LENGTHSCALE_DIM_MODES``.
  3. The Gamma outputscale prior, shared by the ``ScaleKernel`` of KS1 and the
     global ``total_outputscale`` of KS2-KS4.
  4. ``BetaPrior`` — the prior on the KS5 catalyst correlation floor.

The numerical values of every hyper-hyperparameter (sigma, shift, alpha, beta)
stay in ``config.py``; this module only turns them into distributions.
"""

from __future__ import annotations

import math

import torch
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from torch import Tensor
from torch.distributions import Beta, Dirichlet


LENGTHSCALE_DIM_MODES: tuple[str, ...] = ("block", "total")
"""Conventions for the dimension ``D`` in the LogNormal location ``log(sqrt(D)) + shift``.

The dimension-scaled lengthscale prior of Hvarfner et al. (2024) grows the prior
lengthscale as ``sqrt(D)`` so that the expected pairwise distance in a
``D``-dimensional input stays comparable across problem sizes. A block-structured
kernel makes ``D`` ambiguous, and the two readings are swept against each other
on the CV grid rather than assumed:

  ``"block"``  ``D`` is the width of the block the kernel acts on. Each block
               kernel is treated as its own GP input space, so a 3-column solvent
               block gets a shorter prior lengthscale than a 2048-column
               fingerprint block.
  ``"total"``  ``D`` is the summed width of all four kernel blocks
               (cat + solv + base + cont). Every block kernel then shares one
               prior location, set by the dimensionality of the full design.
               ``base_active`` and ``form_id`` are not kernel inputs and are
               excluded from the sum.
"""


class DirichletPrior(Prior):
    """Symmetric Dirichlet prior on a simplex-valued parameter.

    The parameter is not itself constrained to the simplex; ``transform`` maps it
    there (a softmax, for ``DirichletAdditiveKernel``'s unconstrained weight
    vector) before the density is evaluated.

    Args:
        concentration: ``(k,)`` concentration vector of the Dirichlet.
        transform: Callable mapping the raw parameter onto the simplex, or
            ``None`` to evaluate the density on the parameter as given.
    """

    arg_constraints = {}

    def __init__(self, concentration: Tensor, transform=None):
        super(Prior, self).__init__()
        self.register_buffer("concentration", concentration)
        self._transform = transform

    def forward(self, x):
        pass

    def log_prob(self, parameter: Tensor) -> Tensor:
        """Log density of the transformed parameter, summed over batch dims.

        Args:
            parameter: Raw parameter of shape ``(..., k)``.

        Returns:
            Scalar log density.
        """
        if self._transform is not None:
            parameter = self._transform(parameter)
        dist = Dirichlet(self.concentration)
        return dist.log_prob(parameter).sum()


class BetaPrior(Prior):
    """Beta(a, b) prior on a parameter constrained to the open unit interval.

    gpytorch ships no Beta prior. ``KernelStructure.ks5`` puts this on the
    catalyst correlation floor ``rho``; with ``a = 1`` the density is finite at
    zero, so a MAP fit can return to ``rho = 0``.

    Args:
        concentration1: First shape parameter ``a``.
        concentration0: Second shape parameter ``b``.
    """

    arg_constraints = {}

    def __init__(self, concentration1: float, concentration0: float):
        super(Prior, self).__init__()
        self.register_buffer("concentration1", torch.tensor(float(concentration1)))
        self.register_buffer("concentration0", torch.tensor(float(concentration0)))

    def _dist(self) -> Beta:
        return Beta(self.concentration1, self.concentration0)

    def forward(self, x):
        pass

    def log_prob(self, parameter: Tensor) -> Tensor:
        """Log density summed over all elements of ``parameter``."""
        return self._dist().log_prob(parameter).sum()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw from the Beta."""
        return self._dist().sample(sample_shape)


def get_lengthscale_prior(
    *,
    dim_mode: str,
    block_dims: int,
    total_dims: int,
    shift: float,
    scale: float,
) -> LogNormalPrior:
    """LogNormal lengthscale prior with location ``log(sqrt(D)) + shift``.

    Keyword-only so that the two dimension counts cannot be transposed at a call
    site. ``D`` is resolved from ``dim_mode`` (see ``LENGTHSCALE_DIM_MODES``).

    Args:
        dim_mode: One of ``LENGTHSCALE_DIM_MODES``: ``"block"`` uses
            ``block_dims``, ``"total"`` uses ``total_dims``.
        block_dims: Number of columns the kernel acts on.
        total_dims: Summed width of all four kernel blocks.
        shift: Additive offset on the location, biasing toward longer
            lengthscales.
        scale: Standard deviation sigma of the LogNormal.

    Returns:
        The LogNormal prior, applied elementwise to a kernel's group-lengthscale
        vector.

    Raises:
        ValueError: If ``dim_mode`` is not in ``LENGTHSCALE_DIM_MODES``.
    """
    if dim_mode not in LENGTHSCALE_DIM_MODES:
        raise ValueError(
            f"Unknown lengthscale dim_mode {dim_mode!r}; "
            f"expected one of {LENGTHSCALE_DIM_MODES}"
        )
    num_dims = block_dims if dim_mode == "block" else total_dims
    return LogNormalPrior(loc=math.log(math.sqrt(num_dims)) + shift, scale=scale)


def get_outputscale_prior(alpha: float, beta: float) -> GammaPrior:
    """Gamma prior on a kernel outputscale, with mean ``alpha / beta``.

    Args:
        alpha: Concentration of the Gamma.
        beta: Rate of the Gamma.

    Returns:
        The Gamma prior, used by the KS1 ``ScaleKernel`` and by the global
        ``total_outputscale`` of ``DirichletAdditiveKernel``.
    """
    return GammaPrior(alpha, beta)


def dirichlet_weight_prior(num_kernels: int, concentration: float) -> DirichletPrior:
    """Symmetric Dirichlet prior on softmax-transformed additive weights.

    Args:
        num_kernels: Number of additive terms ``k``; the concentration vector is
            ``concentration`` repeated ``k`` times.
        concentration: Symmetric Dirichlet concentration. Values below 1 make the
            density diverge at the simplex corners and every fit fail; keep >= 1.

    Returns:
        The prior, to be registered on an unconstrained ``(k,)`` weight vector.
    """
    return DirichletPrior(
        concentration=torch.full((num_kernels,), concentration),
        transform=lambda x: torch.nn.functional.softmax(x, dim=-1),
    )
