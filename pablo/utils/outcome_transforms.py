"""Outcome transforms for bounded [0, 1] targets (yield, conversion, selectivity).

Every transform is a link applied to a boundary-guarded squash of the raw
fraction, so no transform ever sees an exact 0 or 1::

    p = (y + eps) / (1 + 2 * eps)       y in [0, 1]  ->  p in (0, 1)
    z = link(p)

with ``eps = LOGIT_EPS``. ``inverse`` is the exact algebraic inverse of
``forward`` for every name, which matters because ``fit_eval`` draws posterior
samples on the transform scale and back-transforms them to the raw fraction
scale before any metric is computed — the metrics and their constant-model
baselines are all defined on the raw scale, so a transform is only comparable
to the others if its inverse is exact.

``raw`` is the identity and bypasses the squash. ``logit`` is computed in the
closed form ``log((y + eps) / (1 - y + eps))``, which is algebraically identical
to squash-then-link and differs from it only in the last ULP.

Tensors are elementwise and shape-preserving throughout; every function accepts
any shape and returns that same shape.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pablo.utils.feature_assembly import LOGIT_EPS

OUTCOME_TRANSFORM_NAMES: tuple[str, ...] = ("raw", "logit", "arcsin_sqrt", "probit")
"""Every implemented outcome transform, in the order they are reported."""

ARCSIN_SQRT_MAX: float = math.pi / 2.0
"""Upper bound of the ``arcsin_sqrt`` image, ``arcsin(sqrt(1)) = pi/2``."""


def _squash(y: Tensor) -> Tensor:
    """Map a raw fraction in ``[0, 1]`` into the open interval ``(0, 1)``."""
    return (y + LOGIT_EPS) / (1.0 + 2.0 * LOGIT_EPS)


def _unsquash(p: Tensor) -> Tensor:
    """Inverse of :func:`_squash`."""
    return p * (1.0 + 2.0 * LOGIT_EPS) - LOGIT_EPS


def forward(y: Tensor, name: str) -> Tensor:
    """Map raw fractions onto the modelling scale named by ``name``.

    Args:
        y: Raw fractions in ``[0, 1]``, any shape.
        name: One of :data:`OUTCOME_TRANSFORM_NAMES`.

    Returns:
        Tensor of the same shape and dtype as ``y``, on the transform scale.

    Raises:
        ValueError: If ``name`` is not an implemented transform.
    """
    if name == "raw":
        return y
    if name == "logit":
        return torch.log((y + LOGIT_EPS) / (1.0 - y + LOGIT_EPS))
    p = _squash(y)
    if name == "arcsin_sqrt":
        return torch.asin(torch.sqrt(p))
    if name == "probit":
        return torch.special.ndtri(p)
    raise ValueError(f"Unknown outcome_transform: {name!r}")


def inverse(z: Tensor, name: str) -> Tensor:
    """Map values on the ``name`` scale back to raw fractions.

    Exact algebraic inverse of :func:`forward`. ``sin²`` is not monotone
    outside ``[0, π/2]``, so ``arcsin_sqrt`` clamps ``z`` to that interval first:
    a Gaussian draw below 0 maps to the lowest yield, not to ``sin²(|z|)``.

    Args:
        z: Values on the transform scale, any shape.
        name: One of :data:`OUTCOME_TRANSFORM_NAMES`.

    Returns:
        Tensor of the same shape and dtype as ``z``, on the raw fraction scale.

    Raises:
        ValueError: If ``name`` is not an implemented transform.
    """
    if name == "raw":
        return z
    if name == "logit":
        return (1.0 + 2.0 * LOGIT_EPS) / (1.0 + torch.exp(-z)) - LOGIT_EPS
    if name == "arcsin_sqrt":
        p = torch.sin(z.clamp(0.0, ARCSIN_SQRT_MAX)) ** 2
    elif name == "probit":
        p = torch.special.ndtr(z)
    else:
        raise ValueError(f"Unknown outcome_transform: {name!r}")
    return _unsquash(p)
