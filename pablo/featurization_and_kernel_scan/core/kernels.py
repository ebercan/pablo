"""Base kernels, the partial-ARD layer, and the KS1-KS7 structure catalog.

The design matrix is four blocks wide — ``cat | solv | base | cont`` (plus the
``base_active`` flag column and ``form_id``, which no kernel consumes) — and a
kernel structure is a rule for combining one kernel per block:

  KS1  ``ScaleKernel`` over the product k_cat · k_solv · k_base · k_cont.
  KS2  Dirichlet-weighted additive mixture of the 4 single-block kernels.
  KS3  KS2's 4 singles + the 6 pairwise products = 10 terms.
  KS4  KS3's 10 + the full 4-way product = 11 terms.
  KS5  KS1 with a catalyst correlation floor: [rho + (1 - rho) k_cat] in the product.
  KS6  The KS1 product plus solvent, base and condition main effects with their
       own half-normal-priored scale.
  KS7  [s_g² + s_d² k̃_cat] · k_solv · k_base · k_cont, with k̃_cat the catalyst
       kernel centred over the 161-catalyst library.

KS5-KS7 leave the base block unmasked, as KS1 does, and contain KS1 as the
special case rho = 0, s_+ = 0 and s_g = 0.

KS2-KS4 all instantiate ``DirichletAdditiveKernel`` and differ only in their
entry of ``KS_TERMS``, the index tuples naming which blocks each additive term
multiplies. A term is an index tuple rather than a nested ``ProductKernel`` so
that each block kernel is a child of the mixture exactly once — see that class
for why the distinction changes the MAP objective and not just the runtime.

Every block kernel is a ``PartialARDKernel``: one raw lengthscale per group of a
**partition** over the block's columns, gathered to a per-column vector that
divides the inputs before a unit-lengthscale base kernel (RBF, Matern-5/2,
Tanimoto, or MinMax) sees them. Full ARD is the partition ``[[0], [1], ...]`` and
a single shared lengthscale is ``[[0, 1, ..., d-1]]``; both run through that one
code path, and gpytorch's native ``ard_num_dims`` is never used. The partition is
supplied by the caller, or left to ``build_partition`` below, which is the
default recipe: one shared lengthscale over each raw cat/solv/base block and full
ARD over the continuous block. There is no Hamming ``CategoricalKernel`` route —
a discrete block is described entirely by its featurization's columns.

No-base handling. Rows with no base carry an all-zero base block and
``base_active == 0``. KS2-KS4 zero-mask the base block through that flag, so the
k_base single and every base-containing product term vanish for those rows while
the base-free terms carry them. KS1 is unmasked: inside a pure product a zero
mask annihilates the entire covariance row, and a neutral-1 mask is not PSD, so
the all-zero base vector is instead left to act as the "none" location — a point
no real base occupies under one-hot encoding.

One consequence, unhandled here on purpose: the Tanimoto and MinMax ratios both
evaluate to 0 on an all-zero row, including against itself, so under those two
base kernels a KS1 no-base row contributes nothing but noise. Restricting the
ratio kernels to the catalyst block is a grid decision and belongs in
``config.py``, not here.

All hyper-hyperparameters (prior parameters, initial outputscale, Dirichlet
concentration) live in ``config.py``; the prior objects themselves are built in
``priors.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from gpytorch.constraints import GreaterThan, Interval, Positive
from gpytorch.kernels import Kernel, ProductKernel, ScaleKernel
from gpytorch.priors import HalfNormalPrior, Prior
from linear_operator import to_dense
from torch import Tensor

from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.priors import (
    BetaPrior,
    dirichlet_weight_prior,
    get_lengthscale_prior,
    get_outputscale_prior,
)

BLOCKS: tuple[str, ...] = ("cat", "solv", "base", "cont")
"""The four feature blocks every kernel structure is built from, in column order."""

_SINGLE_TERMS: tuple[tuple[int, ...], ...] = tuple((i,) for i in range(len(BLOCKS)))
_PAIRWISE_TERMS: tuple[tuple[int, ...], ...] = tuple(
    (i, j) for i in range(len(BLOCKS)) for j in range(i + 1, len(BLOCKS))
)
_FULL_TERM: tuple[tuple[int, ...], ...] = (tuple(range(len(BLOCKS))),)

KS_TERMS: dict[str, tuple[tuple[int, ...], ...]] = {
    "KS2": _SINGLE_TERMS,
    "KS3": _SINGLE_TERMS + _PAIRWISE_TERMS,
    "KS4": _SINGLE_TERMS + _PAIRWISE_TERMS + _FULL_TERM,
}
"""Additive terms of the mixture structures, as index tuples into ``BLOCKS``.

KS2 is the 4 single-block terms, KS3 adds the 6 pairwise products, KS4 adds the
full 4-way product. KS1 is not here: it is a single product wrapped in a
``ScaleKernel``, not a mixture."""


_SQRT5: float = math.sqrt(5.0)
_RATIO_EPS: float = 1e-8


class UnitRBFKernel(Kernel):
    """Squared-exponential kernel at unit lengthscale.

    K(x, z) = exp(-‖x - z‖² / 2). Scaling is the responsibility of the
    ``PartialARDKernel`` above it, which divides its inputs by the per-column
    lengthscale vector before calling this.

    Shapes: x1 ``(..., n1, d)``, x2 ``(..., n2, d)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        sq_dist = self.covar_dist(x1, x2, diag=diag, square_dist=True, **params)
        return torch.exp(-0.5 * sq_dist)


class UnitMatern52Kernel(Kernel):
    """Matern kernel with ν = 5/2 at unit lengthscale.

    K(x, z) = (1 + √5 r + 5r²/3) · exp(-√5 r) with r = ‖x - z‖. Scaling is the
    responsibility of the ``PartialARDKernel`` above it.

    Shapes: x1 ``(..., n1, d)``, x2 ``(..., n2, d)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        r = self.covar_dist(x1, x2, diag=diag, **params)
        return (1.0 + _SQRT5 * r + (5.0 / 3.0) * r.pow(2)) * torch.exp(-_SQRT5 * r)


class UnitTanimotoKernel(Kernel):
    """Tanimoto (Jaccard) similarity kernel at unit lengthscale.

    K(x, z) = ⟨x, z⟩ / (‖x‖² + ‖z‖² - ⟨x, z⟩), the continuous generalization of
    the fingerprint Tanimoto coefficient. Well defined for non-negative features;
    the denominator is clamped away from 0, so an all-zero row (the "no base"
    location) has similarity 0 against every row, itself included.

    Shapes: x1 ``(..., n1, d)``, x2 ``(..., n2, d)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False
    is_stationary = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        if diag:
            cross = (x1 * x2).sum(dim=-1)
            denom = x1.pow(2).sum(dim=-1) + x2.pow(2).sum(dim=-1) - cross
            return cross / denom.clamp_min(_RATIO_EPS)
        cross = torch.matmul(x1, x2.transpose(-2, -1))
        norm1 = x1.pow(2).sum(dim=-1, keepdim=True)
        norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
        denom = norm1 + norm2.transpose(-2, -1) - cross
        return cross / denom.clamp_min(_RATIO_EPS)


class UnitMinMaxKernel(Kernel):
    """MinMax (Ruzicka) similarity kernel at unit lengthscale.

    K(x, z) = Σ_d min(x_d, z_d) / Σ_d max(x_d, z_d). Ruzicka similarity is only
    well defined for non-negative features; on a z-scored or L2-normalized block
    the ratio can leave [0, 1]. The pairwise path materializes ``(..., n1, n2,
    d)``, which is the memory-binding operation on wide blocks.

    Shapes: x1 ``(..., n1, d)``, x2 ``(..., n2, d)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False
    is_stationary = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        if diag:
            sum_min = torch.minimum(x1, x2).sum(dim=-1)
            sum_max = torch.maximum(x1, x2).sum(dim=-1)
            return sum_min / sum_max.clamp_min(_RATIO_EPS)
        a = x1.unsqueeze(-2)
        b = x2.unsqueeze(-3)
        sum_min = torch.minimum(a, b).sum(dim=-1)
        sum_max = torch.maximum(a, b).sum(dim=-1)
        return sum_min / sum_max.clamp_min(_RATIO_EPS)


BASE_KERNELS: dict[str, type[Kernel]] = {
    "rbf": UnitRBFKernel,
    "matern52": UnitMatern52Kernel,
    "tanimoto": UnitTanimotoKernel,
    "minmax": UnitMinMaxKernel,
}
"""The four base kernels, keyed by the name used on the CV grid."""


def make_base_kernel(kernel_type: str) -> Kernel:
    """Instantiate one unit-lengthscale base kernel by name.

    Args:
        kernel_type: A key of ``BASE_KERNELS``.

    Returns:
        The base kernel, with no ``active_dims`` and no lengthscale of its own.

    Raises:
        ValueError: If ``kernel_type`` is not a known base kernel.
    """
    if kernel_type not in BASE_KERNELS:
        raise ValueError(
            f"Unknown base kernel {kernel_type!r}; expected one of "
            f"{tuple(BASE_KERNELS)}"
        )
    return BASE_KERNELS[kernel_type]()


def validate_partition(
    partition: Sequence[Sequence[int]],
    dims: Iterable[int],
    label: str,
) -> None:
    """Check that a partition's groups disjointly cover exactly ``dims``.

    A column assigned to no group would never enter the kernel — its feature
    would be silently ignored by the covariance — and a column in two groups
    would be scaled twice, so both are hard errors.

    Args:
        partition: Groups of dimension indices, one shared lengthscale each.
        dims: The dimensions the kernel is defined over.
        label: Name used in the error messages, e.g. ``"partition['cat']"``.

    Raises:
        ValueError: If ``dims`` is empty, or a group is empty, or the groups
            overlap, or they leave a dimension uncovered, or they include a
            dimension outside ``dims``.
    """
    dims = list(dims)
    expected = set(dims)
    if not dims:
        raise ValueError(f"{label}: no dimensions to partition")
    if len(expected) != len(dims):
        raise ValueError(f"{label}: duplicate dimensions in {dims}")

    seen: set[int] = set()
    for group in partition:
        if len(group) == 0:
            raise ValueError(f"{label}: empty group in {partition}")
        overlap = seen.intersection(group)
        if overlap or len(set(group)) != len(group):
            raise ValueError(f"{label}: group {list(group)} repeats dimensions")
        seen.update(group)

    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"{label}: dimensions {missing} are in no group")
    extra = sorted(seen - expected)
    if extra:
        raise ValueError(f"{label}: dimensions {extra} are outside {dims}")


PARTITION_MODES: tuple[str, ...] = ("full_ard", "block_shared", "legacy")
"""How many lengthscales a partition gives each block.

``full_ard``      one lengthscale per column, every block. This is the BoTorch
                  default granularity and what the P1/P2 prior schemes use.
``block_shared``  one shared lengthscale per block, continuous block included.
                  ARD off everywhere; the P3 baseline.
``legacy``        one shared lengthscale per discrete block, full ARD over the
                  continuous block. The recipe the archived sweeps ran under;
                  it is none of P1-P3 and is kept only so those stay reproducible.

Granularity is the axis that decides whether the partition-and-interpolate
degeneracy is reachable at all: per-column ARD on a wide catalyst block can
single out one entity-identifying coordinate and shrink it, which tying the
block's columns to one lengthscale forbids outright.
"""


def build_partition(
    feature_dims: dict[str, list[int]],
    mode: str = "legacy",
) -> dict[str, list[list[int]]]:
    """Build a lengthscale partition over the four blocks.

    Args:
        feature_dims: Block name to column indices in the assembled tensor, as
            carried by ``CellFeatures.feature_dims``. Must cover all of
            ``BLOCKS``.
        mode: One of ``PARTITION_MODES``. Defaults to ``"legacy"`` so existing
            callers keep the recipe they were written against.

    Returns:
        Block name to groups of column indices, one shared lengthscale per
        group, in the form ``KernelStructure`` expects.

    Raises:
        ValueError: If ``mode`` is unknown, or a block of ``BLOCKS`` is absent
            or carries no columns.
    """
    if mode not in PARTITION_MODES:
        raise ValueError(f"Unknown partition mode {mode!r}; expected one of {PARTITION_MODES}")
    missing = [block for block in BLOCKS if block not in feature_dims]
    if missing:
        raise ValueError(f"feature_dims is missing blocks {missing}")

    partition: dict[str, list[list[int]]] = {}
    for block in BLOCKS:
        dims = list(feature_dims[block])
        if not dims:
            raise ValueError(f"Block {block!r} has no feature columns")
        per_column = mode == "full_ard" or (mode == "legacy" and block == "cont")
        partition[block] = [[dim] for dim in dims] if per_column else [dims]
    return partition


def validate_terms(
    terms: Sequence[Sequence[int]],
    num_blocks: int,
) -> tuple[tuple[int, ...], ...]:
    """Check an additive-mixture term list and normalize it to tuples.

    Args:
        terms: One index tuple per term, indexing into the block kernels.
        num_blocks: Number of block kernels available.

    Returns:
        The terms as a tuple of tuples.

    Raises:
        ValueError: If ``terms`` is empty, a term is empty or repeats an index,
            two terms are identical, or an index is out of range.
    """
    if len(terms) == 0:
        raise ValueError("Additive mixture needs at least one term")
    normalized = tuple(tuple(term) for term in terms)
    seen: set[tuple[int, ...]] = set()
    for term in normalized:
        if len(term) == 0:
            raise ValueError(f"Empty term in {normalized}")
        if len(set(term)) != len(term):
            raise ValueError(f"Term {term} repeats a block index")
        if any(index < 0 or index >= num_blocks for index in term):
            raise ValueError(f"Term {term} indexes outside {num_blocks} blocks")
        key = tuple(sorted(term))
        if key in seen:
            raise ValueError(f"Term {term} is a duplicate")
        seen.add(key)
    return normalized


class PartialARDKernel(Kernel):
    """A base kernel under one lengthscale per group of a dimension partition.

    Holds ``len(partition)`` raw lengthscales, gathers them into a length-``d``
    vector, divides both inputs by it, and delegates to a unit-lengthscale base
    kernel. The three ARD regimes are the same code path, differing only in the
    partition:

      full ARD      ``[[0], [1], ..., [d-1]]`` — d lengthscales.
      no ARD        ``[[0, 1, ..., d-1]]`` — one shared lengthscale.
      partial ARD   anything between, e.g. ``[[0, 1, 2, 3], [4, 5]]`` — two
                    lengthscales over six dimensions.

    Note on the ratio kernels: a *uniform* positive rescaling cancels in both the
    Tanimoto and the MinMax ratio, so under those base kernels a single-group
    lengthscale is unidentifiable (the likelihood is flat in it and the MAP sits
    at the prior mode) and only *relative* per-group scales matter. This is the
    standard fixed-Tanimoto GP baseline and is left as is.

    ``active_dims`` is applied by ``Kernel.__call__`` before ``forward`` runs, so
    ``partition`` and ``num_dims`` are expressed in the kernel's own local
    coordinates ``0 ... num_dims - 1``, not in assembled-tensor column indices.

    Args:
        base_kernel_type: A key of ``BASE_KERNELS``.
        num_dims: Number of columns this kernel acts on.
        partition: Groups of local dimension indices, one lengthscale each.
        lengthscale_prior: Prior applied elementwise to the group-lengthscale
            vector.
        lengthscale_constraint: Constraint on the group lengthscales; ``None``
            uses the default ``Positive()``.
        **kwargs: Passed to ``Kernel``, notably ``active_dims`` (the block's
            global column indices) and ``batch_shape``.

    Shapes: x1 ``(..., n1, d)``, x2 ``(..., n2, d)`` with ``d = num_dims`` →
    ``(..., n1, n2)``, or ``(..., n1)`` when ``diag=True``. The lengthscale
    parameter is ``(*batch_shape, 1, len(partition))`` and the gathered vector is
    ``(*batch_shape, 1, d)``.
    """

    has_lengthscale = False

    def __init__(
        self,
        base_kernel_type: str,
        num_dims: int,
        partition: Sequence[Sequence[int]],
        lengthscale_prior,
        lengthscale_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        validate_partition(partition, range(num_dims), f"{base_kernel_type} partition")

        self.base_kernel = make_base_kernel(base_kernel_type)
        self.num_dims = num_dims
        self.num_groups = len(partition)

        group_index = torch.empty(num_dims, dtype=torch.long)
        for group_id, group in enumerate(partition):
            group_index[torch.tensor(list(group), dtype=torch.long)] = group_id
        self.register_buffer("group_index", group_index)

        self.register_parameter(
            name="raw_lengthscale",
            parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1, self.num_groups)),
        )
        self.register_constraint(
            "raw_lengthscale",
            Positive() if lengthscale_constraint is None else lengthscale_constraint,
        )
        self.register_prior(
            "lengthscale_prior",
            lengthscale_prior,
            lambda m: m.lengthscale,
            lambda m, v: m._set_lengthscale(v),
        )

    @property
    def lengthscale(self) -> Tensor:
        """The group lengthscales, shape ``(*batch_shape, 1, num_groups)``."""
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(
            raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value)
        )

    def expanded_lengthscale(self) -> Tensor:
        """Group lengthscales gathered to one per dimension, ``(*batch, 1, num_dims)``."""
        return self.lengthscale.index_select(-1, self.group_index)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        assert x1.size(-1) == self.num_dims and x2.size(-1) == self.num_dims, (
            f"Expected {self.num_dims} columns, got {x1.size(-1)} and {x2.size(-1)}."
        )
        lengthscale = self.expanded_lengthscale()
        return self.base_kernel.forward(
            x1 / lengthscale, x2 / lengthscale, diag=diag, **params
        )


class MaskedKernel(Kernel):
    """Zeroes a kernel wherever either input is inactive.

    The covariance of the wrapped kernel is multiplied by
    ``flag(x1) · flag(x2)``, with the flag read from column ``active_flag_dim``
    of the *unsliced* input — which is why this wrapper carries no
    ``active_dims`` of its own and calls the wrapped kernel through ``__call__``,
    letting that kernel apply its own column selection.

    Used for the base block in KS2-KS4: with ``base_active == 0`` on no-base
    rows, the k_base term and every product term containing it drop out for those
    rows. Not used in KS1, where a zero factor would annihilate the whole
    product.

    Args:
        base_kernel: The kernel to mask.
        active_flag_dim: Column index of the 0/1 activity flag in the assembled
            feature tensor.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` over the full assembled
    width → ``(..., n1, n2)``, or ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, active_flag_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.active_flag_dim = active_flag_dim

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        covar = self.base_kernel(x1, x2, diag=diag, **params)
        mask1 = x1[..., self.active_flag_dim]
        mask2 = x2[..., self.active_flag_dim]
        if diag:
            return covar * (mask1 * mask2)
        return covar * (mask1.unsqueeze(-1) * mask2.unsqueeze(-2))


class DirichletAdditiveKernel(Kernel):
    """Dirichlet-weighted additive mixture over products of block kernels.

    K = total_outputscale · Σ_i weights_i · Π_{j ∈ terms_i} block_kernels_j, with
    the weights the softmax of an unconstrained vector under a symmetric
    Dirichlet prior. The single global outputscale keeps the mixture
    identifiable — per-term outputscales would be redundant with the weights.

    Terms are index tuples into ``block_kernels``, not nested kernel modules, for
    two reasons. Each block kernel stays a child of this module exactly once, so
    ``named_priors`` — which recurses without a memo, unlike ``named_parameters``
    and ``named_constraints`` — yields its lengthscale prior once instead of once
    per term referencing it; nesting the same block module in several
    ``ProductKernel``s would otherwise raise that prior to the power of the term
    multiplicity (4 for KS3, 5 for KS4). And each block covariance is evaluated
    once per forward rather than once per term, which matters most for the
    ``UnitMinMaxKernel`` pairwise path on wide blocks.

    Args:
        block_kernels: The block kernels, indexed by ``terms``.
        terms: One index tuple per additive term; ``(0,)`` is the single k_cat
            term, ``(0, 2)`` the product k_cat · k_base.
        concentration: Symmetric Dirichlet concentration on the weights.
        outputscale_gamma_alpha: Concentration of the Gamma prior on
            ``total_outputscale``.
        outputscale_gamma_beta: Rate of that Gamma prior.
        outputscale_init: Initial value of ``total_outputscale``.
        **kwargs: Passed to ``Kernel``.

    Raises:
        ValueError: If ``terms`` is empty, a term is empty or repeats an index,
            two terms are identical, or an index is out of range.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` over the full assembled
    width → ``(..., n1, n2)``, or ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(
        self,
        block_kernels: list[Kernel],
        terms: Sequence[Sequence[int]],
        concentration: float,
        outputscale_gamma_alpha: float,
        outputscale_gamma_beta: float,
        outputscale_init: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_kernels = nn.ModuleList(block_kernels)
        self.terms = validate_terms(terms, len(block_kernels))

        num_terms = len(self.terms)
        self.register_parameter(
            name="raw_weights",
            parameter=nn.Parameter(torch.zeros(num_terms)),
        )
        self.register_prior(
            "raw_weights_prior",
            dirichlet_weight_prior(num_terms, concentration),
            "raw_weights",
        )

        self.register_parameter(
            name="raw_total_outputscale",
            parameter=nn.Parameter(torch.tensor(0.0)),
        )
        self.register_constraint("raw_total_outputscale", Positive())
        self.register_prior(
            "total_outputscale_prior",
            get_outputscale_prior(outputscale_gamma_alpha, outputscale_gamma_beta),
            lambda m: m.total_outputscale,
            lambda m, v: m._set_total_outputscale(v),
        )

        self.total_outputscale = outputscale_init

    @property
    def total_outputscale(self) -> Tensor:
        """The scalar global outputscale multiplying the weighted sum."""
        return self.raw_total_outputscale_constraint.transform(
            self.raw_total_outputscale
        )

    @total_outputscale.setter
    def total_outputscale(self, value):
        self._set_total_outputscale(value)

    def _set_total_outputscale(self, value) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(
            raw_total_outputscale=self.raw_total_outputscale_constraint.inverse_transform(
                value
            )
        )

    @property
    def weights(self) -> Tensor:
        """The simplex weights, shape ``(len(terms),)``."""
        return torch.nn.functional.softmax(self.raw_weights, dim=-1)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        covars = [
            to_dense(kernel(x1, x2, diag=diag, **params))
            for kernel in self.block_kernels
        ]
        weights = self.weights
        res = None
        for i, term in enumerate(self.terms):
            component = covars[term[0]]
            for index in term[1:]:
                component = component * covars[index]
            component = component * weights[i]
            res = component if res is None else res + component
        return res * self.total_outputscale


def _register_scalar(
    module: Kernel, name: str, init: float, constraint: Interval | Positive, prior: Prior
) -> None:
    """Register ``raw_<name>``, its constraint and a prior on the constrained value.

    The prior gets both closures, so botorch's fit fallback can sample it.

    Args:
        module: The kernel to register on.
        name: Parameter name without the ``raw_`` prefix.
        init: Initial constrained value.
        constraint: Maps ``raw_<name>`` to the constrained value.
        prior: Prior on the constrained value.
    """
    raw = f"raw_{name}"
    module.register_parameter(raw, nn.Parameter(torch.tensor(0.0, dtype=torch.float64)))
    module.register_constraint(raw, constraint)

    def _value(m: Kernel) -> Tensor:
        return getattr(m, f"{raw}_constraint").transform(getattr(m, raw))

    def _set(m: Kernel, value) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float64)
        m.initialize(**{raw: getattr(m, f"{raw}_constraint").inverse_transform(value)})

    module.register_prior(f"{name}_prior", prior, _value, _set)
    _set(module, init)


def _scalar(module: Kernel, name: str) -> Tensor:
    """The constrained value of a scalar registered by ``_register_scalar``."""
    return getattr(module, f"raw_{name}_constraint").transform(getattr(module, f"raw_{name}"))


class CatalystFloorKernel(Kernel):
    """``rho + (1 - rho) · k_cat``: the catalyst kernel with a correlation floor.

    Inside the KS5 product, two experiments under the same conditions correlate at
    least ``rho`` whatever their catalysts, so a held-out catalyst far from every
    training catalyst keeps covariance ``sigma² · rho · k_rest`` with them.
    ``rho = 0`` is the catalyst kernel of KS1.

    Args:
        base_kernel: The catalyst block kernel; its ``active_dims`` select the
            catalyst columns.
        concentration1: First Beta shape parameter of the prior on ``rho``.
        concentration0: Second Beta shape parameter.
        floor_init: Initial ``rho``.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` over the full assembled
    width → ``(..., n1, n2)``, or ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(
        self,
        base_kernel: Kernel,
        concentration1: float,
        concentration0: float,
        floor_init: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        _register_scalar(
            self, "floor", floor_init, Interval(0.0, 1.0),
            BetaPrior(concentration1, concentration0),
        )

    @property
    def floor(self) -> Tensor:
        """The correlation floor ``rho`` in (0, 1)."""
        return _scalar(self, "floor")

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        covar = to_dense(self.base_kernel(x1, x2, diag=diag, **params))
        rho = self.floor
        return rho + (1.0 - rho) * covar


class ProductPlusMainEffectsKernel(Kernel):
    """``s_x² · Π_b k_b + s_+² · mean_{b ∈ main} k_b``: the KS6 covariance.

    The product of all block kernels carries the interactions; the main-effect
    term is the average of the listed block kernels and has its own scale under a
    half-normal prior whose mode is zero, so a fit that finds no use for it
    returns to KS1. Every block kernel is a child of this module once and is
    evaluated once per forward, as in ``DirichletAdditiveKernel``.

    Args:
        block_kernels: The four block kernels in ``BLOCKS`` order, unmasked.
        main_effects: Indices into ``block_kernels`` of the main-effect blocks.
        outputscale_gamma_alpha: Concentration of the Gamma prior on ``s_x²``.
        outputscale_gamma_beta: Rate of that Gamma prior.
        outputscale_init: Initial ``s_x²``.
        main_scale_halfnormal: Scale of the half-normal prior on ``s_+``.
        main_scale_init: Initial ``s_+``.

    Raises:
        ValueError: If ``main_effects`` is empty, repeats an index or is out of range.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(
        self,
        block_kernels: list[Kernel],
        main_effects: Sequence[int],
        outputscale_gamma_alpha: float,
        outputscale_gamma_beta: float,
        outputscale_init: float,
        main_scale_halfnormal: float,
        main_scale_init: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_kernels = nn.ModuleList(block_kernels)
        self.main_effects = validate_terms([tuple(main_effects)], len(block_kernels))[0]
        _register_scalar(
            self, "product_outputscale", outputscale_init, Positive(),
            get_outputscale_prior(outputscale_gamma_alpha, outputscale_gamma_beta),
        )
        _register_scalar(
            self, "main_scale", main_scale_init, Positive(),
            HalfNormalPrior(main_scale_halfnormal),
        )

    @property
    def product_outputscale(self) -> Tensor:
        """``s_x²``, the variance of the product term."""
        return _scalar(self, "product_outputscale")

    @property
    def main_scale(self) -> Tensor:
        """``s_+``, the standard deviation of the main-effect term."""
        return _scalar(self, "main_scale")

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        covars = [to_dense(k(x1, x2, diag=diag, **params)) for k in self.block_kernels]
        product = covars[0]
        for covar in covars[1:]:
            product = product * covar
        main = sum(covars[i] for i in self.main_effects) / len(self.main_effects)
        return self.product_outputscale * product + self.main_scale ** 2 * main


class LibraryCentredKernel(Kernel):
    """``k(c, c') − m(c) m(c') / M``: a kernel orthogonal to its library average.

    ``m(c)`` is the mean of ``k(c, c_l)`` over the ``L`` reference rows and ``M``
    the mean of ``k(c_l, c_l')`` over all reference pairs. The result is the
    covariance of ``f`` given the library average of ``f``, which is zero, so it
    is positive semidefinite and a process drawn from it averages to zero over the
    reference rows (Lu, Boukouvalas and Hensman 2022, Appendix C, with uniform
    weights over the library).

    Args:
        base_kernel: The kernel to centre; its ``active_dims`` select the columns.
        reference: ``(L, D)`` reference rows over the full assembled width, in the
            coordinates the kernel is evaluated in (after the model's input
            transform). Only the base kernel's active columns are read.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, reference: Tensor, **kwargs):
        super().__init__(**kwargs)
        if reference.ndim != 2 or reference.shape[0] < 2:
            raise ValueError(f"reference must be (L, D) with L >= 2; got {tuple(reference.shape)}")
        self.base_kernel = base_kernel
        self.register_buffer("reference", reference)

    def _reference_like(self, x: Tensor) -> Tensor:
        return self.reference.to(x).expand(*x.shape[:-2], *self.reference.shape)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        ref = self._reference_like(x1)
        m1 = to_dense(self.base_kernel(x1, ref, **params)).mean(dim=-1)
        total = to_dense(self.base_kernel(ref, ref, **params)).mean(dim=(-2, -1))
        if diag:
            covar = to_dense(self.base_kernel(x1, x2, diag=True, **params))
            return covar - m1 ** 2 / total.unsqueeze(-1)
        m2 = to_dense(self.base_kernel(x2, self._reference_like(x2), **params)).mean(dim=-1)
        covar = to_dense(self.base_kernel(x1, x2, **params))
        return covar - m1.unsqueeze(-1) * m2.unsqueeze(-2) / total.unsqueeze(-1).unsqueeze(-1)


class SharedDeviationKernel(Kernel):
    """``s_g² + s_d² · k̃_cat``: shared response plus catalyst deviation (KS7).

    Inside the KS7 product with the solvent, base and condition kernels, ``s_g²``
    times that product is the response shared by all catalysts and the centred
    catalyst kernel ``k̃_cat`` carries the catalyst-specific deviation.

    Args:
        centred_kernel: The library-centred catalyst kernel.
        outputscale_gamma_alpha: Concentration of the Gamma prior on ``s_d²``.
        outputscale_gamma_beta: Rate of that Gamma prior.
        outputscale_init: Initial ``s_d²``.
        shared_scale_halfnormal: Scale of the half-normal prior on ``s_g``.
        shared_scale_init: Initial ``s_g``.

    Shapes: as ``LibraryCentredKernel``.
    """

    has_lengthscale = False

    def __init__(
        self,
        centred_kernel: Kernel,
        outputscale_gamma_alpha: float,
        outputscale_gamma_beta: float,
        outputscale_init: float,
        shared_scale_halfnormal: float,
        shared_scale_init: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.centred_kernel = centred_kernel
        _register_scalar(
            self, "deviation_outputscale", outputscale_init, Positive(),
            get_outputscale_prior(outputscale_gamma_alpha, outputscale_gamma_beta),
        )
        _register_scalar(
            self, "shared_scale", shared_scale_init, Positive(),
            HalfNormalPrior(shared_scale_halfnormal),
        )

    @property
    def deviation_outputscale(self) -> Tensor:
        """``s_d²``, the variance of the catalyst deviation."""
        return _scalar(self, "deviation_outputscale")

    @property
    def shared_scale(self) -> Tensor:
        """``s_g``, the standard deviation of the shared response."""
        return _scalar(self, "shared_scale")

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        covar = to_dense(self.centred_kernel(x1, x2, diag=diag, **params))
        return self.shared_scale ** 2 + self.deviation_outputscale * covar


class LinearTrendKernel(Kernel):
    """``s² Σ_j w_j h_j(x) h_j(x')``: a linear trend with its coefficients integrated out.

    A mean ``h(x)ᵀ β`` with ``β ~ N(0, s² diag(w))`` integrated out adds this term to
    the covariance (Rasmussen and Williams 2006, Eq. 2.40). The basis is built from
    the normalised inputs: every listed column centred at 0.5, the midpoint of the
    unit box the input transform maps the training range to, and optionally
    multiplied by a gate column (``base_active`` for the base block); gate columns
    themselves can enter as a basis function.

    Args:
        columns: Column index of each basis function.
        weights: Prior variance factor ``w_j`` of each basis function.
        centre: Whether each basis function is its column minus 0.5 (``True``) or
            the column as it is (``False``, for a 0/1 gate column).
        gate_dim: Column whose value multiplies the gated basis functions, or ``None``.
        gated: Whether each basis function is multiplied by the gate column.
        scale_halfnormal: Scale of the half-normal prior on ``s``.
        scale_init: Initial ``s``.

    Raises:
        ValueError: If the per-function sequences differ in length, are empty, or
            a basis function is gated with no ``gate_dim``.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(
        self,
        columns: Sequence[int],
        weights: Sequence[float],
        centre: Sequence[bool],
        gate_dim: int | None,
        gated: Sequence[bool],
        scale_halfnormal: float,
        scale_init: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n = len(columns)
        if n == 0 or not (len(weights) == len(centre) == len(gated) == n):
            raise ValueError(
                f"columns, weights, centre and gated must be non-empty and of equal "
                f"length; got {n}, {len(weights)}, {len(centre)}, {len(gated)}"
            )
        if any(gated) and gate_dim is None:
            raise ValueError("a gated basis function needs gate_dim")
        self.register_buffer("columns", torch.tensor(list(columns), dtype=torch.long))
        self.register_buffer("weights", torch.tensor(list(weights), dtype=torch.float64))
        self.register_buffer("shift", torch.tensor([0.5 if c else 0.0 for c in centre], dtype=torch.float64))
        self.register_buffer("gated", torch.tensor(list(gated), dtype=torch.bool))
        self.gate_dim = gate_dim
        _register_scalar(self, "scale", scale_init, Positive(), HalfNormalPrior(scale_halfnormal))

    @property
    def scale(self) -> Tensor:
        """``s``, the scale of the integrated coefficients."""
        return _scalar(self, "scale")

    def basis(self, x: Tensor) -> Tensor:
        """``(..., n, J)`` basis functions ``h(x)`` times ``sqrt(w)``."""
        h = x.index_select(-1, self.columns.to(x.device)) - self.shift.to(x)
        if self.gate_dim is not None:
            gate = x[..., self.gate_dim].unsqueeze(-1)
            h = torch.where(self.gated.to(x.device), h * gate, h)
        return h * self.weights.to(x).sqrt()

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        h1 = self.basis(x1)
        s2 = self.scale ** 2
        if diag:
            return s2 * (h1 * h1).sum(dim=-1)
        return s2 * h1 @ self.basis(x2).transpose(-2, -1)


class FormOffsetKernel(Kernel):
    """``tau² 1[form(x) = form(x')]``: an integrated offset per pre-catalyst form.

    The offsets ``b_t ~ N(0, tau²)`` of the pre-catalyst forms, integrated out.
    The form is read from the integer ``form_id`` column, which the input
    transform leaves untouched.

    Args:
        form_id_dim: Column index of ``form_id``.
        scale_halfnormal: Scale of the half-normal prior on ``tau``.
        scale_init: Initial ``tau``.

    Shapes: x1 ``(..., n1, D)``, x2 ``(..., n2, D)`` → ``(..., n1, n2)``, or
    ``(..., n1)`` when ``diag=True``.
    """

    has_lengthscale = False

    def __init__(self, form_id_dim: int, scale_halfnormal: float, scale_init: float, **kwargs):
        super().__init__(**kwargs)
        if form_id_dim < 0:
            raise ValueError("form_offset needs the form_id column")
        self.form_id_dim = form_id_dim
        _register_scalar(self, "scale", scale_init, Positive(), HalfNormalPrior(scale_halfnormal))

    @property
    def scale(self) -> Tensor:
        """``tau``, the standard deviation of the form offsets."""
        return _scalar(self, "scale")

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        assert not last_dim_is_batch, "last_dim_is_batch is not supported."
        t2 = self.scale ** 2
        f1 = x1[..., self.form_id_dim]
        if diag:
            return t2 * torch.ones_like(f1)
        f2 = x2[..., self.form_id_dim]
        return t2 * (f1.unsqueeze(-1) == f2.unsqueeze(-2)).to(x1.dtype)


class KernelStructure:
    """Catalog of the GP covariance structures scanned on the CV grid.

    Builds one ``PartialARDKernel`` per block from that block's slice of the
    partition — all four sharing the same base kernel type and the same
    lengthscale prior family — and combines them per structure: KS1 as one
    scaled product, KS2-KS4 as Dirichlet-weighted mixtures over ``KS_TERMS``. A
    block kernel is instantiated once per structure, so every term referencing
    that block shares its lengthscales.

    Args:
        feature_dims: Block name → column indices in the assembled tensor. Must
            cover all four of ``BLOCKS``; excludes the ``base_active`` column.
        partition: Block name → groups of column indices, one shared lengthscale
            per group. Each block's groups must disjointly cover exactly
            ``feature_dims[block]``.
        active_flag_dim: Column index of the ``base_active`` indicator.
        base_kernel_type: A key of ``BASE_KERNELS``, used for every block.
        lengthscale_dim_mode: A value of ``priors.LENGTHSCALE_DIM_MODES``,
            selecting whether the LogNormal prior location scales with the
            block's width or with the whole design's width.
        lengthscale_shift: Additive offset on the LogNormal prior location.
        lengthscale_scale: Standard deviation sigma of the LogNormal prior.
        dirichlet_concentration: Concentration of the Dirichlet weight prior
            (KS2-KS4).
        outputscale_gamma_alpha: Concentration of the Gamma outputscale prior.
        outputscale_gamma_beta: Rate of the Gamma outputscale prior.
        outputscale_init: Initial outputscale.
        lengthscale_floor: ``GreaterThan`` floor on every group lengthscale, or
            ``None`` for the default ``Positive`` constraint. The floor is a
            guard against the partition-and-interpolate degeneracy in which a
            lengthscale collapses toward zero during the multi-start fits.
        cat_reference: ``(L, D)`` library catalyst rows over the full assembled
            width, in the model's normalised coordinates, or ``None``. Required
            by KS7 only.

    The KS5-KS7 prior parameters are read from ``config`` (``KS5_*``,
    ``KS6_*``, ``KS7_*``).
    """

    def __init__(
        self,
        feature_dims: dict[str, list[int]],
        partition: dict[str, list[list[int]]],
        active_flag_dim: int,
        base_kernel_type: str,
        lengthscale_dim_mode: str,
        lengthscale_shift: float,
        lengthscale_scale: float,
        dirichlet_concentration: float,
        outputscale_gamma_alpha: float,
        outputscale_gamma_beta: float,
        outputscale_init: float,
        lengthscale_floor: float | None = None,
        cat_reference: Tensor | None = None,
    ):
        self.feature_dims = feature_dims
        self.cat_reference = cat_reference
        self.partition = partition
        self.active_flag_dim = active_flag_dim
        self.base_kernel_type = base_kernel_type
        self.lengthscale_dim_mode = lengthscale_dim_mode
        self.lengthscale_shift = lengthscale_shift
        self.lengthscale_scale = lengthscale_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.outputscale_gamma_alpha = outputscale_gamma_alpha
        self.outputscale_gamma_beta = outputscale_gamma_beta
        self.outputscale_init = outputscale_init
        self.lengthscale_floor = lengthscale_floor
        self.total_dims = sum(len(feature_dims[block]) for block in BLOCKS)

    def build(self, kernel_structure: str) -> Kernel:
        """Dispatch to the named structure builder.

        Args:
            kernel_structure: One of ``config.KERNEL_STRUCTURES``.

        Returns:
            The constructed kernel.

        Raises:
            ValueError: If the name is not a known structure.
        """
        builders = {
            "KS1": self.ks1, "KS2": self.ks2, "KS3": self.ks3, "KS4": self.ks4,
            "KS5": self.ks5, "KS6": self.ks6, "KS7": self.ks7,
        }
        if kernel_structure not in builders:
            raise ValueError(
                f"Unknown kernel structure {kernel_structure!r}; expected one of "
                f"{tuple(builders)}"
            )
        return builders[kernel_structure]()

    def ks1(self) -> ScaleKernel:
        """KS1 — ScaleKernel over the product k_cat · k_solv · k_base · k_cont.

        The blocks combine multiplicatively, so shrinking any one block's
        lengthscale toward zero zeroes the whole covariance and the MLL rejects
        it. The base block is **not** masked here: a zero factor would annihilate
        the covariance row of every no-base experiment. Those rows sit at the
        all-zero base location instead and their base similarity is learned.

        Returns:
            The scaled product kernel, one Gamma-priored outputscale.
        """
        blocks = [self._block_kernel(block) for block in BLOCKS]
        kernel = ScaleKernel(
            ProductKernel(*blocks),
            outputscale_prior=get_outputscale_prior(
                self.outputscale_gamma_alpha, self.outputscale_gamma_beta
            ),
        )
        kernel.outputscale = self.outputscale_init
        return kernel

    def ks2(self) -> DirichletAdditiveKernel:
        """KS2 — additive main effects: the 4 single-block terms.

        Returns:
            The 4-term Dirichlet-weighted mixture, base block masked.
        """
        return self._mixture(KS_TERMS["KS2"])

    def ks3(self) -> DirichletAdditiveKernel:
        """KS3 — KS2's 4 singles plus the 6 pairwise products.

        Returns:
            The 10-term Dirichlet-weighted mixture, base block masked.
        """
        return self._mixture(KS_TERMS["KS3"])

    def ks4(self) -> DirichletAdditiveKernel:
        """KS4 — KS3's 10 terms plus the full 4-way product.

        Returns:
            The 11-term Dirichlet-weighted mixture, base block masked.
        """
        return self._mixture(KS_TERMS["KS4"])

    def ks5(self) -> ScaleKernel:
        """KS5 — KS1 with a catalyst correlation floor.

        ``sigma² · [rho + (1 - rho) k_cat] · k_solv · k_base · k_cont`` with
        ``rho ~ Beta(KS5_FLOOR_BETA)``. Base block unmasked, as in KS1.

        Returns:
            The scaled product kernel with the floored catalyst factor.
        """
        a, b = config.KS5_FLOOR_BETA
        cat = CatalystFloorKernel(
            self._block_kernel("cat"), concentration1=a, concentration0=b,
            floor_init=config.KS5_FLOOR_INIT,
        )
        blocks = [cat] + [self._block_kernel(block) for block in BLOCKS[1:]]
        kernel = ScaleKernel(
            ProductKernel(*blocks),
            outputscale_prior=get_outputscale_prior(
                self.outputscale_gamma_alpha, self.outputscale_gamma_beta
            ),
        )
        kernel.outputscale = self.outputscale_init
        return kernel

    def ks6(self) -> ProductPlusMainEffectsKernel:
        """KS6 — the KS1 product plus solvent, base and condition main effects.

        ``s_x² Π_b k_b + s_+² (k_solv + k_base + k_cont) / 3`` with the main-effect
        scale ``s_+`` under a half-normal prior. No catalyst main effect; base
        block unmasked, as in KS1.

        Returns:
            The product-plus-main-effects kernel.
        """
        blocks = [self._block_kernel(block) for block in BLOCKS]
        return ProductPlusMainEffectsKernel(
            blocks,
            main_effects=tuple(BLOCKS.index(b) for b in ("solv", "base", "cont")),
            outputscale_gamma_alpha=self.outputscale_gamma_alpha,
            outputscale_gamma_beta=self.outputscale_gamma_beta,
            outputscale_init=self.outputscale_init,
            main_scale_halfnormal=config.KS6_MAIN_SCALE_HALFNORMAL,
            main_scale_init=config.KS6_MAIN_SCALE_INIT,
        )

    def ks7(self) -> ProductKernel:
        """KS7 — shared condition response plus a library-centred catalyst deviation.

        ``[s_g² + s_d² k̃_cat] · k_solv · k_base · k_cont``, with ``k̃_cat`` the
        catalyst kernel centred over ``cat_reference``. Base block unmasked.

        Returns:
            The product kernel.

        Raises:
            ValueError: If ``cat_reference`` was not given.
        """
        if self.cat_reference is None:
            raise ValueError("KS7 needs cat_reference, the library catalyst rows")
        cat = SharedDeviationKernel(
            LibraryCentredKernel(self._block_kernel("cat"), self.cat_reference),
            outputscale_gamma_alpha=self.outputscale_gamma_alpha,
            outputscale_gamma_beta=self.outputscale_gamma_beta,
            outputscale_init=self.outputscale_init,
            shared_scale_halfnormal=config.KS7_SHARED_SCALE_HALFNORMAL,
            shared_scale_init=config.KS7_SHARED_SCALE_INIT,
        )
        return ProductKernel(cat, *[self._block_kernel(block) for block in BLOCKS[1:]])

    def _mixture(self, terms: Sequence[Sequence[int]]) -> DirichletAdditiveKernel:
        """Build the four block kernels, base masked, and mix them over ``terms``."""
        blocks = [self._block_kernel(block, mask=(block == "base")) for block in BLOCKS]
        return DirichletAdditiveKernel(
            blocks,
            terms,
            concentration=self.dirichlet_concentration,
            outputscale_gamma_alpha=self.outputscale_gamma_alpha,
            outputscale_gamma_beta=self.outputscale_gamma_beta,
            outputscale_init=self.outputscale_init,
        )

    def _block_kernel(self, block: str, mask: bool = False) -> Kernel:
        """Build one block's partial-ARD kernel from its slice of the partition.

        The block's partition is validated against ``feature_dims[block]`` and
        then remapped from assembled-tensor column indices to the kernel's local
        coordinates, since ``active_dims`` slicing happens before ``forward``.

        Args:
            block: One of ``BLOCKS``.
            mask: Whether to wrap the result in ``MaskedKernel``.

        Returns:
            The block kernel, masked when requested.
        """
        dims = list(self.feature_dims[block])
        groups = self.partition[block]
        validate_partition(groups, dims, f"partition[{block!r}]")
        position = {dim: i for i, dim in enumerate(dims)}
        local_groups = [[position[dim] for dim in group] for group in groups]

        kernel: Kernel = PartialARDKernel(
            base_kernel_type=self.base_kernel_type,
            num_dims=len(dims),
            partition=local_groups,
            lengthscale_prior=get_lengthscale_prior(
                dim_mode=self.lengthscale_dim_mode,
                block_dims=len(dims),
                total_dims=self.total_dims,
                shift=self.lengthscale_shift,
                scale=self.lengthscale_scale,
            ),
            lengthscale_constraint=(
                None
                if self.lengthscale_floor is None
                else GreaterThan(self.lengthscale_floor)
            ),
            active_dims=dims,
        )
        if mask:
            return MaskedKernel(kernel, active_flag_dim=self.active_flag_dim)
        return kernel


def make_kernel_from_config(
    kernel_structure: str,
    feature_dims: dict[str, list[int]],
    active_flag_dim: int,
    base_kernel_type: str,
    partition: dict[str, list[list[int]]] | None = None,
    lengthscale_scale: float | None = None,
    lengthscale_dim_mode: str | None = None,
    prior_scheme: str | None = None,
    cat_reference: Tensor | None = None,
) -> Kernel:
    """Build a kernel structure from the global config.

    Args:
        kernel_structure: One of ``config.KERNEL_STRUCTURES``.
        feature_dims: Block name → column indices.
        active_flag_dim: Column index of the ``base_active`` indicator.
        base_kernel_type: A key of ``BASE_KERNELS``.
        partition: Block name → groups of column indices, one shared lengthscale
            per group. ``None`` uses ``build_partition(feature_dims,
            mode=scheme.partition_mode)`` when ``prior_scheme`` is given, else
            ``build_partition(feature_dims, mode="legacy")``.
        lengthscale_scale: σ of the LogNormal lengthscale prior. ``None`` takes
            ``config.LENGTHSCALE_LOGNORMAL_SCALE``; callers that adapt σ to the
            training-set size pass it explicitly.
        lengthscale_dim_mode: A value of ``priors.LENGTHSCALE_DIM_MODES``.
            ``None`` uses ``config.LENGTHSCALE_DIM_MODE``.
        prior_scheme: A key of ``config.PRIOR_SCHEMES`` ("P1", "P2", "P3")
            supplying both the partition granularity and the ``log sqrt(D)``
            dim mode. An explicit ``lengthscale_dim_mode`` wins over the
            scheme's. An explicit ``partition`` must have, in every block, the
            same number of groups as the scheme's granularity. ``None`` leaves
            both to their own defaults.
        cat_reference: ``(L, D)`` library catalyst rows in the model's
            normalised coordinates; required by ``"KS7"``, ignored otherwise.

    Returns:
        The constructed kernel.

    Raises:
        ValueError: If ``prior_scheme`` is not a key of ``config.PRIOR_SCHEMES``;
            if an explicit ``partition`` disagrees with the scheme's
            granularity in the number of groups of any block; or if
            ``kernel_structure`` is not one of ``config.KERNEL_STRUCTURES``.
    """
    scheme = None
    if prior_scheme is not None:
        if prior_scheme not in config.PRIOR_SCHEMES:
            raise ValueError(
                f"Unknown prior scheme {prior_scheme!r}; "
                f"expected one of {sorted(config.PRIOR_SCHEMES)}"
            )
        scheme = config.PRIOR_SCHEMES[prior_scheme]
    if partition is None:
        partition = build_partition(
            feature_dims, mode="legacy" if scheme is None else scheme.partition_mode
        )
    elif scheme is not None:
        expected = build_partition(feature_dims, mode=scheme.partition_mode)
        mismatched = [
            block for block in BLOCKS
            if len(partition[block]) != len(expected[block])
        ]
        if mismatched:
            raise ValueError(
                f"Prior scheme {prior_scheme!r} wants "
                f"{scheme.partition_mode!r} granularity but the supplied partition "
                f"disagrees on blocks {mismatched}. The scheme fixes the number of "
                "lengthscales and the prior location together; a partition built "
                "under a different mode silently changes only one of them."
            )
    if lengthscale_scale is None:
        lengthscale_scale = config.LENGTHSCALE_LOGNORMAL_SCALE
    if lengthscale_dim_mode is None:
        lengthscale_dim_mode = (
            config.LENGTHSCALE_DIM_MODE if scheme is None else scheme.lengthscale_dim_mode
        )
    builder = KernelStructure(
        feature_dims=feature_dims,
        partition=partition,
        active_flag_dim=active_flag_dim,
        base_kernel_type=base_kernel_type,
        lengthscale_dim_mode=lengthscale_dim_mode,
        lengthscale_shift=config.LENGTHSCALE_LOGNORMAL_LOC_SHIFT,
        lengthscale_scale=lengthscale_scale,
        dirichlet_concentration=config.CONCENTRATION_DIRICHLET,
        outputscale_gamma_alpha=config.ALPHA_GAMMA_OUTPUTSCALE,
        outputscale_gamma_beta=config.BETA_GAMMA_OUTPUTSCALE,
        outputscale_init=config.OUTPUTSCALE_INIT,
        lengthscale_floor=config.LENGTHSCALE_FLOOR,
        cat_reference=cat_reference,
    )
    return builder.build(kernel_structure)
