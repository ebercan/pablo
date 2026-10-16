"""
Base classes for defining optimization problems and synthetic functions.

This module provides abstract base classes to standardize the interface for optimization
problems used in benchmarking and algorithm development. All optimization problems and
synthetic functions intended for use in this framework should inherit from these classes
and provide concrete implementations for the required methods.

Module: benchmark_problems.base_classes

Classes:
    - OptimizationProblem: Abstract base class for general optimization problems.
    - SyntheticFunction: Abstract base class for synthetic benchmark functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.

    This class defines the interface for optimization problems used in benchmarking
    and algorithm development. Subclasses must implement the following:

    - bounds: Returns the recommended bounds for the optimization variables as a 2 x d tensor,
      where the first row contains the lower bounds and the second row contains the upper bounds
      for each dimension.
    - global_max: Returns the global maximum value of the objective function, or None if unknown.
    - global_maximizers: Returns a list of tensors, each a global maximizer of the objective
      function, or None if unknown.
    - value: Evaluates the objective at an n x d tensor of points and returns an n x 1 tensor
      (some subclasses also accept b x n x d and return b x n x 1).

    All optimization problems intended for use in this framework should inherit from this class
    and provide concrete implementations for these methods.
    """

    @abstractmethod
    def bounds(self) -> torch.Tensor | None:
        """Return the bounds that are recommended for the optimization benchmarking.

        Returns:
            torch.Tensor: 2 x d tensor of lower and upper bounds for each dimension.
        """
        pass

    @abstractmethod
    def global_max(self) -> float | None:
        """Return the global maximum value of the optimization problem.

        Leave as None if not known
        """
        pass

    @abstractmethod
    def global_maximizers(self) -> list[torch.Tensor] | None:
        """Return the list of global maximizers of the function."""
        pass

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the objective function at point(s) x.

        Args:
            x (torch.Tensor): Input tensor with shape (n, d),
              where n is the number of points and d is the dimension.
        Returns:
            torch.Tensor: Function value at x with shape (n, 1).
        """
        pass


class SyntheticFunction(OptimizationProblem):
    """Abstract base class for synthetic functions."""
