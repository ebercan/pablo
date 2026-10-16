"""Synthetic functions with known global optima for benchmarking global optimization.

Module: benchmark_problems.synthetic

The coefficients are taken from
https://www.sfu.ca/~ssurjano/optimization.html


"""

import warnings

import numpy as np
import torch

from benchmark_problems.base_classes import SyntheticFunction
from pablo.utils.tensor_shapes import check_shape_and_bounds


def _bbob_compute_rotation(seed: int, dim: int) -> torch.Tensor:
    """Create a deterministic orthogonal matrix for BBOB-style transforms.

    Args:
        seed: Random seed used to generate the matrix.
        dim: Matrix dimension.

    Returns:
        torch.Tensor: Orthogonal matrix with shape (dim, dim).
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(matrix)
    return torch.tensor(q, dtype=torch.double)


def _bbob_monotone_tf_osc(values: torch.Tensor) -> torch.Tensor:
    """Apply the BBOB monotone oscillatory transform elementwise.

    Args:
        values: Input tensor.

    Returns:
        torch.Tensor: Transformed tensor with same shape as input.
    """
    transformed = values.clone()

    positive = values > 0
    if positive.any():
        positive_vals = torch.log(values[positive]) / 0.1
        transformed[positive] = (
            torch.exp(
                positive_vals
                + 0.49 * (torch.sin(positive_vals) + torch.sin(0.79 * positive_vals))
            )
            ** 0.1
        )

    negative = values < 0
    if negative.any():
        negative_vals = torch.log(-values[negative]) / 0.1
        transformed[negative] = (
            -torch.exp(
                negative_vals
                + 0.49
                * (torch.sin(0.55 * negative_vals) + torch.sin(0.31 * negative_vals))
            )
            ** 0.1
        )

    return transformed


class Branin(SyntheticFunction):
    """
    Negative of the 2D Branin function.

    As defined in
    https://www.sfu.ca/~ssurjano/branin.html.
    It's negative because maximization convention is followed.
    Input x should be in the (0,1),(0,1) square.

    Supports batch evaluation with input shape (b,n,2) where:
    b: number of batches
    n: number of points per batch
    2: input dimension (fixed for Branin)
    """

    def __init__(
        self,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Branin benchmark.

        Args:
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If mixed-discrete metadata is inconsistent.
        """
        self.d = 2
        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self) -> torch.Tensor:
        """
        Return the bounds for the Branin function.

        Returns:
        torch.Tensor, recommended bounds for the Branin function
        """
        # Rescale from [0,1]^2 to the standard Branin domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
        # The bounds are given in the form of a 2x2 tensor.

        return torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    def global_max(self) -> float:
        """Return the global maximum value (−0.397887) under the maximization convention.

        Returns:
            float: Negated Branin minimum.
        """
        return -0.397887

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Branin function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor is a global maximizer.
        """
        x_opt_1 = torch.Tensor([[0.1239, 0.8183]]).unsqueeze(0)  # Shape (1,1,2)
        x_opt_2 = torch.Tensor([[0.5428, 0.1517]]).unsqueeze(0)  # Shape (1,1,2)
        x_opt_3 = torch.Tensor([[0.9617, 0.1650]]).unsqueeze(0)  # Shape (1,1,2)
        return [x_opt_1, x_opt_2, x_opt_3]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the negated Branin function.

        Args:
            x: Input tensor with shape (b, n, 2) or (n, 2), in [0, 1]².

        Returns:
            torch.Tensor: Function values with shape (b, n, 1) or (n, 1).

        Raises:
            ValueError: If input shape is invalid or points are out of bounds.
        """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        # Rescale from [0,1]^2 to standard Branin domain

        x1 = x[..., 0:1] * 15 - 5  # Shape (b,n,1)
        x2 = x[..., 1:2] * 15  # Shape (b,n,1)

        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * torch.cos(x1)
        value = -(term1 + term2 + s)

        if is_x_2d:
            value = value.squeeze(0)
        return value


class Hartmann(SyntheticFunction):
    """
    Negative of the Hartmann function.

    As defined in
    https://www.sfu.ca/~ssurjano/hart6.html. It's
    negative because maximization convention is followed.
    Input should be in the (0,1)^d hypersquare. d is by default 6,
    but can be 3,4,6. The function is defined for 3,4,6 dimensions.
    """

    def __init__(
        self,
        d: int = 6,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Hartmann benchmark.

        Args:
            d: Input dimension (3, 4, or 6).
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If d is not in {3, 4, 6} or mixed-discrete metadata is
                inconsistent.
        """
        if d not in [3, 4, 6]:
            raise ValueError("Valid dimensions for the Hartmann function are 3, 4, 6.")
        self.d = d
        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self) -> torch.Tensor:
        """
        Return the bounds for the Hartmann function.

        Returns:
        torch.Tensor, recommended bounds for the Hartmann function
        """
        return torch.stack([torch.zeros(self.d), torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the global maximum value for the Hartmann function.

        Returns:
            float: Maximum value (3.86278 for d=3, 3.7297 for d=4, 3.32237 for d=6).

        Raises:
            ValueError: If d is not in {3, 4, 6}.
        """
        # Only valid for d=3,4,6; index 0 for d=3, 1 for d=4, 3 for d=6

        if self.d == 3:
            return 3.86278
        elif self.d == 4:
            return 3.7297
        elif self.d == 6:
            return 3.32237
        else:
            raise ValueError("Valid dimensions for the Hartmann function are 3, 4, 6.")

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Hartmann function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor is a global maximizer.
        """
        maximizer = []
        if self.d == 3:
            maximizer = [torch.Tensor([[0.114614, 0.555649, 0.852547]]).unsqueeze(0)]
        elif self.d == 4:
            maximizer = [torch.tensor([[0.1873, 0.1906, 0.5566, 0.2647]]).unsqueeze(0)]
        elif self.d == 6:
            maximizer = [
                torch.Tensor(
                    [[0.20169, 0.15000, 0.47687, 0.27530, 0.31165, 0.65730]]
                ).unsqueeze(0)
            ]
        return maximizer

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Give negative Hartmann function value.

        Parameters:
        x : torch.Tensor with shape (b,n,d) or (n,d) where
            b: number of batches
            n: number of points per batch
            d: input dimension (3,4, or 6)

        Returns:
        torch.Tensor with shape (b,n,1), function value at x for each batch and point

        Raises:
        ValueError: If input shape is incorrect or if points are out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)
        value = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        alpha = torch.tensor([[1.0, 1.2, 3.0, 3.2]], device=x.device, dtype=x.dtype)

        if self.d == 6:
            # Coefficients from the SFU website

            matrix_a = torch.tensor(
                [
                    [10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00],
                ],
                device=x.device, dtype=x.dtype,
            )
            matrix_p = torch.tensor(
                [
                    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
                ],
                device=x.device, dtype=x.dtype,
            )
        elif self.d == 3:
            # Coefficients from the SFU website

            matrix_a = torch.tensor(
                [
                    [3.00, 10.00, 30.00],
                    [0.10, 10.00, 35.00],
                    [3.00, 10.00, 30.00],
                    [0.10, 10.00, 35.00],
                ],
                device=x.device, dtype=x.dtype,
            )
            matrix_p = torch.tensor(
                [
                    [0.3689, 0.1170, 0.2673],
                    [0.4699, 0.4387, 0.7470],
                    [0.1091, 0.8732, 0.5547],
                    [0.0381, 0.5743, 0.8828],
                ],
                device=x.device, dtype=x.dtype,
            )
        elif self.d == 4:
            # Coefficients from the SFU website

            matrix_a = torch.tensor(
                [
                    [10.00, 3.00, 17.00, 3.50],
                    [0.05, 10.00, 17.00, 0.10],
                    [3.00, 3.50, 1.70, 10.00],
                    [17.00, 8.00, 0.05, 10.00],
                ],
                device=x.device, dtype=x.dtype,
            )
            matrix_p = torch.tensor(
                [
                    [0.1312, 0.1696, 0.5569, 0.0124],
                    [0.2329, 0.4135, 0.8307, 0.3736],
                    [0.2348, 0.1451, 0.3522, 0.2883],
                    [0.4047, 0.8828, 0.8732, 0.5743],
                ],
                device=x.device, dtype=x.dtype,
            )
        else:
            raise ValueError("Valid dimensions for the Hartmann function are 3, 4, 6.")
        for i in range(4):
            # Reshape matrices for broadcasting

            torch.reshape(matrix_a[i, :], (1, self.d))  # Shape (1,4,d)
            A_i = torch.reshape(matrix_a[i, :], (1, self.d)).unsqueeze(
                1
            )  # Shape (1,1,d)
            P_i = torch.reshape(matrix_p[i, :], (1, self.d)).unsqueeze(
                1
            )  # Shape (1,1,d)

            # Compute exponent maintaining batch dimension

            exponent = -torch.sum(A_i * torch.square(x - P_i), dim=-1, keepdim=True)
            value += alpha[0, i] * torch.exp(exponent)
        if is_x_2d:
            value = value.squeeze(0)
        return value


class Ackley(SyntheticFunction):
    """
    Negative and shifted Ackley function.

    As defined in
    https://www.sfu.ca/~ssurjano/ackley.html. It's
    negated because maximization convention is followed.
    Then shifted by 1, so that the global maximum is 1
    instead of 0. Input should be in the (-33,33)^d
    hypersquare. d is by default 10, but can be any
    positive integer.
    """

    def __init__(
        self,
        d: int = 10,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Ackley benchmark.

        Args:
            d: Input dimension.
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If mixed-discrete metadata is inconsistent.
        """
        self.d = d
        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self):
        """
        Return the bounds for the Ackley function.

        Returns:
        torch.Tensor, recommended bounds for the Ackley function
        """
        return torch.stack([(-33) * torch.ones(self.d), 33 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the global maximum value (1.0) of the negated and shifted Ackley function.

        Returns:
            float: Global maximum under the maximization convention.
        """
        return 1.0

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Ackley function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor has shape (1,1,d)
        """
        return [torch.zeros(1, 1, self.d)]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ackley Function.

        Parameters:
        x : torch.Tensor with shape (b,n,d) or (n,d) where
            b: number of batches
            n: number of points per batch
            d: input dimension

        Returns:
        torch.Tensor with shape (b,n,1), function value at x for each batch and point

        Raises:
        ValueError: If input shape is incorrect or if points are out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)
        _kw = dict(device=x.device, dtype=x.dtype)
        a = 20 * torch.ones(x.shape[0], x.shape[1], 1, **_kw)
        b = 0.2 * torch.ones(x.shape[0], x.shape[1], 1, **_kw)
        c = 2 * np.pi * torch.ones(x.shape[0], x.shape[1], 1, **_kw)

        sum1 = torch.zeros(x.shape[0], x.shape[1], 1, **_kw)
        sum2 = torch.zeros(x.shape[0], x.shape[1], 1, **_kw)

        for i in range(self.d):
            xi = torch.reshape(x[:, :, i], (x.shape[0], x.shape[1], 1))
            sum1 += torch.square(xi)
            sum2 += torch.cos(c * xi)

        term1 = -a * torch.exp(-b * torch.sqrt(sum1 / self.d))
        term2 = -torch.exp(sum2 / self.d)
        value = -(
            term1
            + term2
            + a
            + torch.exp(torch.ones(x.shape[0], x.shape[1], 1, **_kw))
            - torch.ones(x.shape[0], x.shape[1], 1, **_kw)
        )

        if is_x_2d:
            value = value.squeeze(0)
        return value


class Rastrigin(SyntheticFunction):
    """
    Negative and shifted Rastrigin function.

    As defined in
    https://www.sfu.ca/~ssurjano/rastr.html. It's
    negated because maximization convention is followed.
    Then shifted by 1, so that the global maximum is 1
    instead of 0. Input should be in the (-5.12,5.12)^d
    hypersquare. d is by default 8, but can be any
    positive integer.
    """

    def __init__(
        self,
        d: int = 8,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Rastrigin benchmark.

        Args:
            d: Input dimension.
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If mixed-discrete metadata is inconsistent.
        """
        self.d = d
        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self):
        """
        Return the bounds for the Rastrigin function.

        Returns:
        torch.Tensor, recommended bounds for the Rastrigin function
        """
        return torch.stack([(-5.12) * torch.ones(self.d), 5.12 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the global maximum value (1.0) of the negated and shifted Rastrigin function.

        Returns:
            float: Global maximum under the maximization convention.
        """
        return 1.0

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Rastrigin function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor is a global maximizer.
        """
        return [torch.zeros(1, 1, self.d)]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rastrigin Function.

        Parameters:
        x : torch.Tensor with shape (b,n,d) or (n,d) where
            b: number of batches
            n: number of points per batch
            d: input dimension

        Returns:
        torch.Tensor with shape (b,n,1), function value at x for each batch and point

        Raises:
        ValueError: If input shape is incorrect or if points are out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)
        _kw = dict(device=x.device, dtype=x.dtype)
        A = 10 * torch.ones(x.shape[0], x.shape[1], 1, **_kw)
        sum_term = torch.zeros(x.shape[0], x.shape[1], 1, **_kw)

        for i in range(self.d):
            xi = torch.reshape(x[:, :, i], (x.shape[0], x.shape[1], 1))

            sum_term += xi**2 - 10 * torch.cos(2 * np.pi * xi)
        value = -(A * self.d + sum_term - torch.ones(x.shape[0], x.shape[1], 1, **_kw))

        if is_x_2d:
            value = value.squeeze(0)
        return value


class Levy(SyntheticFunction):
    """
    Negative and shifted Levy function.

    As defined in
    https://www.sfu.ca/~ssurjano/levy.html. It's
    negated because maximization convention is followed.
    Then shifted by 1, so that the global maximum is 1
    instead of 0. Input should be in the (-10, 10)^d
    hypersquare. d is by default 10, but can be any
    positive integer.
    """

    def __init__(
        self,
        d: int = 10,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Levy benchmark.

        Args:
            d: Input dimension.
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If mixed-discrete metadata is inconsistent.
        """
        self.d = d
        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self):
        """
        Return the bounds for the Levy function.

        Returns:
            torch.Tensor, recommended bounds for the Levy function
        """
        return torch.stack([-10.0 * torch.ones(self.d), 10.0 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the global maximum value (1.0) of the negated and shifted Levy function.

        Returns:
            float: Global maximum under the maximization convention.
        """
        return 1.0

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Levy function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor is a global maximizer.
        """
        return [torch.ones(1, 1, self.d)]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Levy Function.

        Parameters:
            x : torch.Tensor with shape (b,n,d) or (n,d) where
            b: number of batches
            n: number of points per batch
            d: input dimension

        Returns:
            torch.Tensor with shape (b,n,1), function value at x for each batch and point

        Raises:
            ValueError: If input shape is incorrect or if points are out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)
        w = 1 + (x - 1) / 4
        w1 = w[..., 0:1]
        term1 = torch.sin(np.pi * w1) ** 2

        term2 = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        if self.d >= 2:
            for i in range(0, self.d - 1):
                wi = w[..., i : i + 1]
                term2 += (wi - 1) ** 2 * (1 + 10 * torch.sin(np.pi * wi + 1) ** 2)
        w_last = w[..., -1:]
        term3 = (w_last - 1) ** 2 * (1 + torch.sin(2 * np.pi * w_last) ** 2)

        f_levy = term1 + term2 + term3

        value = 1 - f_levy

        if is_x_2d:
            value = value.squeeze(0)
        return value


class Shekel(SyntheticFunction):
    """
    Negative Shekel function.

    As defined in
    https://www.sfu.ca/~ssurjano/shekel.html. It's
    negated because maximization convention is followed.
    Input should be in the [0,10]^4 hypersquare. m=10 is
    implicitly assumed.
    """

    def __init__(
        self,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Shekel benchmark (d=4 fixed).

        Args:
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If mixed-discrete metadata is inconsistent.
        """
        self.d = 4

        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self):
        """
        Return the bounds for the Shekel function.

        Returns:
        torch.Tensor, recommended bounds for the Shekel function
        """
        return torch.stack([torch.zeros(self.d), 10 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the global maximum value (10.5364) of the Shekel function.

        Returns:
            float: Global maximum under the maximization convention.
        """
        return 10.5364

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Shekel function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor has shape (1,1,d)
        """
        return [torch.ones(1, 1, self.d) * 4]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shekel Function.

        Parameters:
        x : torch.Tensor with shape (b,n,d) or (n,d) where
            b: number of batches
            n: number of points per batch
            d: input dimension

        Returns:
        torch.Tensor with shape (b,n,1), function value at x for each batch and point

        Raises:
        ValueError: If input shape is incorrect or if points are out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)

        beta = torch.tensor(
            [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5],
            device=x.device, dtype=x.dtype,
        )
        C = torch.tensor(
            [
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            ],
            device=x.device, dtype=x.dtype,
        )

        b, n, d = x.shape
        m = 10

        value = torch.zeros(b, n, device=x.device, dtype=x.dtype)

        for i in range(m):
            c_i = C[:, i].reshape(1, 1, d)
            diff = x - c_i
            squared_diff = torch.sum(diff**2, dim=-1)
            term = 1.0 / (squared_diff + beta[i])
            value -= term

        value = value.unsqueeze(-1)

        if is_x_2d:
            value = value.squeeze(0)
        return (-1) * value


class Michalewicz(SyntheticFunction):
    """Negated Michalewicz function.

    As defined in
    https://www.sfu.ca/~ssurjano/michal.html. The canonical definition on
    that page is minimized, so this implementation returns its negative to
    follow the maximization convention used in this repository.
    Input should be in the [0, pi]^d hypercube.
    """

    def __init__(
        self,
        d: int = 2,
        m: int = 10,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Initialize the Michalewicz function.

        Args:
            d: Input dimension.
            m: Steepness parameter from the SFU formulation.
            mixed: Whether mixed-variable metadata is provided.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If d <= 0, m <= 0, or mixed-discrete metadata is invalid.
        """
        if d <= 0:
            raise ValueError("d must be a positive integer.")
        if m <= 0:
            raise ValueError("m must be a positive integer.")

        self.d = d
        self.m = m

        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self):
        """Return the recommended bounds for the Michalewicz function.

        Returns:
            torch.Tensor: 2 x d tensor where row 0 is lower bounds and row 1 is
                upper bounds.
        """
        return torch.stack([torch.zeros(self.d), torch.pi * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return known global maxima for common dimensions when m=10.

        Returns:
            float: Global maximum value for known (d, m=10) pairs.

        Raises:
            ValueError: If no benchmark reference value is encoded for this
                dimension/parameter pair.
        """
        if self.m != 10:
            raise ValueError("Known global maxima are only encoded for m=10.")

        known_max = {
            2: 1.8013,
            5: 4.687658,
            10: 9.66015,
        }
        if self.d not in known_max:
            raise ValueError(
                "Known global maxima are only encoded for d in {2, 5, 10} when m=10."
            )
        return known_max[self.d]

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return known global maximizers for the 2D benchmark case.

        Returns:
            list[torch.Tensor]: List with one tensor of shape (1, 1, d).

        Raises:
            ValueError: If no benchmark maximizer is encoded for this dimension/
                parameter pair.
        """
        if self.d == 2 and self.m == 10:
            return [torch.tensor([[2.20, 1.57]]).unsqueeze(0)]
        raise ValueError("Known global maximizers are only encoded for d=2 and m=10.")

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the negated Michalewicz function.

        Args:
            x: Input points with shape (n, d) or (b, n, d), with each coordinate
                in [0, pi].

        Returns:
            torch.Tensor: Function values with shape (n, 1) or (b, n, 1),
                matching input batch convention.

        Raises:
            ValueError: If x has invalid shape or lies out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False

        check_shape_and_bounds(self, x)

        ii = torch.arange(1, self.d + 1, device=x.device, dtype=x.dtype).reshape(
            1, 1, self.d
        )
        term = torch.sin(x) * torch.pow(
            torch.sin(ii * torch.square(x) / np.pi), 2 * self.m
        )
        value = torch.sum(term, dim=-1, keepdim=True)

        if is_x_2d:
            value = value.squeeze(0)
        return value


class StyblinskiTang(SyntheticFunction):
    """
    Negative Styblinski-Tang function.

    As defined in
    https://www.sfu.ca/~ssurjano/stybtang.html. It's
    negated because maximization convention is followed.
    Input should be in the [-5,5]^d
    hypersquare. d is by default 9, but can be any
    positive integer.
    """

    def __init__(
        self,
        d: int = 9,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        """Configure the Styblinski-Tang benchmark.

        Args:
            d: Input dimension.
            mixed: If True, enable mixed continuous–discrete mode.
            discrete_indices: Indices of discrete dimensions when mixed=True.
            discrete_choices: Allowed values for each discrete dimension.

        Raises:
            ValueError: If mixed-discrete metadata is inconsistent.
        """
        self.d = d
        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

    def bounds(self):
        """
        Return the bounds for the Styblinski-Tang function.

        Returns:
        torch.Tensor, recommended bounds for the Styblinski-Tang function
        """
        return torch.stack([(-5) * torch.ones(self.d), 5 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the global maximum value (39.16599 · d) of the negated Styblinski-Tang function.

        Returns:
            float: Global maximum under the maximization convention.
        """
        return 39.16599 * self.d

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the global maximizers of the Styblinski-Tang function.

        Returns:
        list[torch.Tensor]: List of tensors, each tensor has shape (1,1,d)
        """
        return [torch.ones(1, 1, self.d) * (-2.903534)]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Styblinski-Tang Function.

        Parameters:
        x : torch.Tensor with shape (b,n,d) or (n,d) where
            b: number of batches
            n: number of points per batch
            d: input dimension

        Returns:
        torch.Tensor with shape (b,n,1), function value at x for each batch and point

        Raises:
        ValueError: If input shape is incorrect or if points are out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)

        term1 = torch.pow(x, 4)
        term2 = -16 * torch.pow(x, 2)
        term3 = 5 * x

        value = 0.5 * torch.sum(term1 + term2 + term3, dim=-1, keepdim=True)

        if is_x_2d:
            value = value.squeeze(0)

        return (-1) * value


class Gallagher101(SyntheticFunction):
    """Gallagher with 101 Gaussian peaks (BBOB f21) as a maximization benchmark.

    This class implements the deterministic, noise-free structure of BBOB section
    5.21 (f21) with fixed seeded rotations and peak placement, then converts the
    original minimization landscape into maximization via ``1 - f(x)``.

    Args:
        d: Input dimension.
        seed: Seed controlling deterministic rotations and peak locations.
        mixed: Whether mixed-variable metadata is provided.
        discrete_indices: Indices of discrete dimensions when ``mixed=True``.
        discrete_choices: Allowed values for each discrete dimension.

    Raises:
        ValueError: If ``d <= 0`` or mixed-discrete metadata is invalid.
    """

    def __init__(
        self,
        d: int = 10,
        seed: int = 21,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        if d <= 0:
            raise ValueError("d must be a positive integer.")

        self.d = d
        self.seed = seed
        self.n_high_peaks = 101
        self.max_condition = 1000.0
        self.high_peak_cond = np.sqrt(1000.0)

        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

        self.rotation = _bbob_compute_rotation(seed=self.seed, dim=self.d)

        rng = np.random.default_rng(self.seed)
        arrcondition = self.max_condition ** np.linspace(0, 1, self.n_high_peaks - 1)
        peak_perm = np.argsort(rng.uniform(size=self.n_high_peaks - 1))
        arrcondition = np.insert(arrcondition[peak_perm], 0, self.high_peak_cond)

        arrscales = []
        for i, cond in enumerate(arrcondition):
            scale = cond ** np.linspace(-0.5, 0.5, self.d)
            dim_perm_rng = np.random.default_rng(self.seed + 1000 * i)
            dim_perm = np.argsort(dim_perm_rng.uniform(size=self.d))
            arrscales.append(scale[dim_perm])
        self.arrscales = torch.tensor(np.vstack(arrscales), dtype=torch.double)

        peakvalues = np.insert(
            np.linspace(1.1, 9.1, self.n_high_peaks - 1),
            0,
            10.0,
        )
        self.peakvalues = torch.tensor(peakvalues, dtype=torch.double)

        xlocal = (
            10.0 * rng.uniform(size=(self.n_high_peaks, self.d)) - 5.0
        ) @ self.rotation.numpy()
        xlocal[0, :] = 0.8 * xlocal[0, :]
        self.xlocal = torch.tensor(xlocal, dtype=torch.double)
        self.xopt = torch.matmul(self.xlocal[0], self.rotation.T)

    def bounds(self) -> torch.Tensor:
        """Return recommended box constraints for BBOB f21.

        Returns:
            torch.Tensor: 2 x d tensor with lower and upper bounds in [-5, 5].
        """
        return torch.stack([-5.0 * torch.ones(self.d), 5.0 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the deterministic global maximum value.

        Returns:
            float: Global maximum value under the maximization convention.
        """
        return 1.0

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the deterministic global maximizer.

        Returns:
            list[torch.Tensor]: One tensor with shape (1, 1, d).
        """
        return [self.xopt.reshape(1, 1, self.d)]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the deterministic BBOB f21 objective.

        Args:
            x: Input points with shape (n, d) or (b, n, d), each coordinate in
                [-5, 5].

        Returns:
            torch.Tensor: Objective values with shape (n, 1) or (b, n, 1),
                matching input batch convention.

        Raises:
            ValueError: If x has invalid shape or lies out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False

        check_shape_and_bounds(self, x)

        rotation = self.rotation.to(device=x.device, dtype=x.dtype)
        arrscales = self.arrscales.to(device=x.device, dtype=x.dtype)
        peakvalues = self.peakvalues.to(device=x.device, dtype=x.dtype)
        xlocal = self.xlocal.to(device=x.device, dtype=x.dtype)

        x_rot = torch.matmul(x, rotation)
        diff = x_rot.unsqueeze(-2) - xlocal.view(1, 1, self.n_high_peaks, self.d)

        scaled_squared = arrscales.view(1, 1, self.n_high_peaks, self.d) * diff.square()
        exponent = (-0.5 / self.d) * torch.sum(scaled_squared, dim=-1)
        peak_response = peakvalues.view(1, 1, self.n_high_peaks) * torch.exp(exponent)
        max_peak = torch.max(peak_response, dim=-1, keepdim=True).values

        f_min = _bbob_monotone_tf_osc(10.0 - max_peak).square()
        value = 1.0 - f_min

        if is_x_2d:
            value = value.squeeze(0)
        return value


class LunacekBiRastrigin(SyntheticFunction):
    """Lunacek bi-Rastrigin (BBOB f24) as a maximization benchmark.

    This class implements the deterministic, noise-free form of BBOB section 5.24
    (f24) with seeded transformations and converts the original minimization
    objective to maximization via ``1 - f(x)``.

    Args:
        d: Input dimension.
        seed: Seed controlling deterministic sign pattern and rotations.
        mixed: Whether mixed-variable metadata is provided.
        discrete_indices: Indices of discrete dimensions when ``mixed=True``.
        discrete_choices: Allowed values for each discrete dimension.

    Raises:
        ValueError: If ``d <= 0`` or mixed-discrete metadata is invalid.
    """

    def __init__(
        self,
        d: int = 10,
        seed: int = 24,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        if d <= 0:
            raise ValueError("d must be a positive integer.")

        self.d = d
        self.seed = seed
        self.condition = 100.0
        self.mu1 = 2.5
        self.shift = 1.0

        if mixed is True:
            if discrete_choices is None:
                raise ValueError("Discrete choices are not given.")
            if len(discrete_indices) != len(discrete_choices):
                raise ValueError(
                    "Number of discrete indices and discrete choice lists do not match."
                )
            if len(discrete_indices) > self.d:
                raise ValueError("Too many discrete dimensions, d_discrete > d")
            if discrete_indices is None:
                warnings.warn(
                    f"Discrete indices are not given, so the last "
                    f"{len(discrete_choices)} dimensions are set as "
                    "discrete indices.",
                    UserWarning,
                )
                discrete_indices = [
                    self.d - len(discrete_choices) + i
                    for i in range(len(discrete_choices))
                ]
            self.discrete_indices = discrete_indices
            self.discrete_choices = discrete_choices

        rng = np.random.default_rng(self.seed)
        xopt_sign = np.sign(rng.standard_normal(self.d))
        xopt_sign[xopt_sign == 0] = 1.0
        self.xopt = torch.tensor(0.5 * self.mu1 * xopt_sign, dtype=torch.double)
        self.arrscales = torch.tensor(2.0 * np.sign(xopt_sign), dtype=torch.double)

        rotation_a = _bbob_compute_rotation(seed=self.seed, dim=self.d)
        rotation_b = _bbob_compute_rotation(seed=self.seed + 1_000_000, dim=self.d)
        scales = (self.condition**0.5) ** np.linspace(0, 1, self.d)
        scale_diag = torch.diag(torch.tensor(scales, dtype=torch.double))
        self.linear_tf = torch.matmul(torch.matmul(rotation_a, scale_diag), rotation_b)

    def bounds(self) -> torch.Tensor:
        """Return recommended box constraints for BBOB f24.

        Returns:
            torch.Tensor: 2 x d tensor with lower and upper bounds in [-5, 5].
        """
        return torch.stack([-5.0 * torch.ones(self.d), 5.0 * torch.ones(self.d)])

    def global_max(self) -> float:
        """Return the deterministic global maximum value.

        Returns:
            float: Global maximum value under the maximization convention.
        """
        return 1.0

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return the deterministic global maximizer.

        Returns:
            list[torch.Tensor]: One tensor with shape (1, 1, d).
        """
        return [self.xopt.reshape(1, 1, self.d)]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the deterministic BBOB f24 objective.

        Args:
            x: Input points with shape (n, d) or (b, n, d), each coordinate in
                [-5, 5].

        Returns:
            torch.Tensor: Objective values with shape (n, 1) or (b, n, 1),
                matching input batch convention.

        Raises:
            ValueError: If x has invalid shape or lies out of bounds.
        
        Args:
            x (torch.Tensor): Argument x.
    """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False

        check_shape_and_bounds(self, x)

        arrscales = self.arrscales.to(device=x.device, dtype=x.dtype)
        linear_tf = self.linear_tf.to(device=x.device, dtype=x.dtype)

        x_scaled = x * arrscales.view(1, 1, self.d)

        s = 1.0 - 0.5 / (np.sqrt(self.d + 20.0) - 4.1)
        mu2 = -np.sqrt((self.mu1**2 - self.shift) / s)

        bowl_1 = torch.sum((x_scaled - self.mu1) ** 2, dim=-1, keepdim=True)
        bowl_2 = self.shift * self.d + s * torch.sum(
            (x_scaled - mu2) ** 2,
            dim=-1,
            keepdim=True,
        )
        f_min = torch.minimum(bowl_1, bowl_2)

        transformed = torch.matmul(x_scaled - self.mu1, linear_tf)
        rastrigin_part = 10.0 * (
            self.d
            - torch.sum(torch.cos(2.0 * np.pi * transformed), dim=-1, keepdim=True)
        )
        f_min = f_min + rastrigin_part

        value = 1.0 - f_min

        if is_x_2d:
            value = value.squeeze(0)
        return value
