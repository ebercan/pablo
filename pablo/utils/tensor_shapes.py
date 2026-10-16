"""Tensor shape annotations for better code readability and type checking.

Module: pablo.utils.tensor_shapes
"""

from dataclasses import dataclass
from typing import Annotated

import torch


@dataclass
class Tensor:
    """Namespace of ``Annotated[torch.Tensor, ...]`` aliases for type hints.

    Each ``shape_*`` attribute names a tensor layout, e.g. ``Tensor.shape_NxD``
    for an ``n x d`` tensor. The annotations carry no runtime check.
    """

    shape_AxBxNx1 = Annotated[torch.Tensor, "Shape: [A, B, N, 1]"]
    shape_BxNxD = Annotated[torch.Tensor, "Shape: [B, N, D]"]
    shape_NxD = Annotated[torch.Tensor, "Shape: [N, D]"]
    shape_BxNx1 = Annotated[torch.Tensor, "Shape: [B, N, 1]"]
    shape_Nx1 = Annotated[torch.Tensor, "Shape: [N, 1]"]
    shape_Bx1 = Annotated[torch.Tensor, "Shape: [B, 1]"]
    shape_BxN = Annotated[torch.Tensor, "Shape: [B, N]"]
    shape_1x2xD = Annotated[torch.Tensor, "Shape: [1, 2, D]"]
    shape_2xD = Annotated[torch.Tensor, "Shape: [2, D]"]


def check_shape_and_bounds(self, x: torch.Tensor) -> None:
    """Validate batched candidate tensor shape and bound feasibility.

    Called as ``check_shape_and_bounds(self, x)`` from a benchmark problem's
    methods.

    Args:
        self: Benchmark problem with dimension attribute ``d`` and a
            ``bounds()`` method returning a ``2 x d`` tensor.
        x: Candidate tensor with shape ``[B, N, D]``; ``D`` must equal
            ``self.d``.

    Raises:
        ValueError: If the tensor shape is invalid or points are outside bounds.
    """
    if x.dim() != 3 or x.shape[-1] != self.d:
        raise ValueError(
            f"Input must be (b,n,{self.d}) or (n,{self.d}), got {tuple(x.shape)}"
        )

    # Keep bounds on the same device/dtype as x to avoid CPU/CUDA mismatch.
    bounds = self.bounds().to(device=x.device, dtype=x.dtype)
    out_of_bounds = ((x < bounds[0]) | (x > bounds[1])).any(dim=-1)
    if out_of_bounds.any():
        indices = torch.where(out_of_bounds)
        b_idx = indices[0]
        n_idx = indices[1]
        error_indices: list[str] = []
        for i in range(len(b_idx)):
            error_indices.append(f"(batch={b_idx[i]}, point={n_idx[i]})")
        raise ValueError(f"Points out of bounds at indices: {error_indices}")
