"""
Optimization benchmark problems that either use real data, or fundamental physical
relations. Global optima are not known: ``global_max`` and
``global_maximizers`` return None.

Module: benchmark_problems.real

#TODO Multi-objective benchmarking not supported yet. For such problems (e.g. Penicillin) an
arbitrary weighted single objective (for maximization) is defined.
"""

import warnings

import torch

from benchmark_problems.base_classes import OptimizationProblem
from pablo.utils.dim_reduction import apply_pca, apply_umap, excel_to_df
from pablo.utils.tensor_shapes import check_shape_and_bounds


class PenicillinYield(OptimizationProblem):
    r"""A penicillin production simulator.

    This implementation is adapted from
    https://github.com/HarryQL/TuRBO-Penicillin.

    The goal is to maximize the penicillin yield.
    Nonconstrained, single objective version, i.e.
    time to ferment and the CO2 byproduct are ignored.
    """

    def __init__(
        self,
        mixed: bool = False,
        discrete_indices: list[int] | None = None,
        discrete_choices: list[list[int]] | None = None,
    ):
        self.d = 7
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
        return torch.tensor(
            [
                [60.0, 120.0],
                [0.05, 18.0],
                [293.0, 303.0],
                [0.05, 18.0],
                [0.01, 0.5],
                [500.0, 700.0],
                [5.0, 6.5],
            ]
        ).T

    def global_max(self) -> None:
        return None

    def global_maximizers(self) -> None:
        return None

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate the fed-batch fermentation and return the final penicillin titre.

        Integrates the process ODEs with unit Euler steps for up to 2500 h; a
        point stops early once the volume exceeds 180 L, the substrate goes
        negative or the penicillin production rate drops below 1e-11.

        Args:
            x: ``b x n x 7`` or ``n x 7`` tensor of control variables, in
                order: culture volume V (L), biomass X (g/L), temperature T
                (K), substrate S (g/L), feed rate F (L/h), feed substrate
                concentration s_f (g/L) and pH. Every point must lie within
                ``bounds()``. ``x`` is not modified.

        Returns:
            ``b x n x 1`` or ``n x 1`` tensor (matching ``x``) of penicillin
            concentration P at the end of the simulation.

        Raises:
            ValueError: If ``x`` has the wrong shape or a point lies outside
                ``bounds()``.
        """
        if x.dim() == 2:
            is_x_2d = True
            x = x.unsqueeze(0)
        else:
            is_x_2d = False
        check_shape_and_bounds(self, x)
        x = x.clone()
        # Constants

        Y_xs = 0.45
        Y_ps = 0.90
        K_1 = 10 ** (-10)
        K_2 = 7 * 10 ** (-5)
        m_X = 0.014
        alpha_1 = 0.143
        alpha_2 = 4 * 10 ** (-7)
        alpha_3 = 10 ** (-4)
        mu_X = 0.092
        K_X = 0.15
        mu_p = 0.005
        K_p = 0.0002
        K_I = 0.10
        K = 0.04
        k_g = 7.0 * 10**3
        E_g = 5100.0
        k_d = 10.0**33
        E_d = 50000.0
        lambd = 2.5 * 10 ** (-4)
        T_v = 273.0  # Kelvin
        T_o = 373.0
        R = 1.9872  # CAL/(MOL K)
        V_max = 180.0

        V, X, T, S, F, s_f, H_ = torch.split(x, 1, dim=-1)

        P = torch.zeros_like(V)
        CO2 = torch.zeros_like(V)

        H = torch.full_like(H_, 10.0).pow(-H_)

        active = torch.ones_like(V, dtype=torch.bool)
        t_tensor = torch.full_like(V, 2500)

        for t in range(1, 2501):
            if active.sum() == 0:
                break
            F_loss = (
                V[active]
                * lambd
                * torch.special.expm1(5 * ((T[active] - T_o) / (T_v - T_o)))
            )

            dV_dt = F[active] - F_loss

            term_temp = (k_g * torch.exp(-E_g / (R * T[active]))) - (
                k_d * torch.exp(-E_d / (R * T[active]))
            )

            mu = (
                (mu_X / (1 + K_1 / H[active] + H[active] / K_2))
                * (S[active] / (K_X * X[active] + S[active]))
                * term_temp
            )

            dX_dt = mu * X[active] - (X[active] / V[active]) * dV_dt

            mu_pp = mu_p * (S[active] / (K_p + S[active] + S[active].pow(2) / K_I))

            dS_dt = (
                -(mu / Y_xs) * X[active]
                - (mu_pp / Y_ps) * X[active]
                - m_X * X[active]
                + F[active] * s_f[active] / V[active]
                - (S[active] / V[active]) * dV_dt
            )

            dP_dt = (
                (mu_pp * X[active]) - K * P[active] - (P[active] / V[active]) * dV_dt
            )

            dCO2_dt = alpha_1 * dX_dt + alpha_2 * X[active] + alpha_3

            P[active] += dP_dt
            V[active] += dV_dt
            X[active] += dX_dt
            S[active] += dS_dt
            CO2[active] += dCO2_dt

            full_dpdt = torch.ones_like(P)
            full_dpdt[active] = dP_dt

            inactive = (V > V_max) | (S < 0) | (full_dpdt < 10e-12)

            t_tensor[inactive] = torch.minimum(
                t_tensor[inactive], torch.full_like(t_tensor[inactive], t)
            )

            active[inactive] = False
        value = P

        if is_x_2d:
            value = value.squeeze(0)
        return value


class AsymmetricHydrogenation(OptimizationProblem):
    """Lookup benchmark on the C=C asymmetric hydrogenation dataset.

    Each dataset entry is a vector of reaction conditions (starting material
    through time) followed by a dimensionality-reduced ligand descriptor.
    ``value`` returns the measured outcome of the dataset entry that matches
    the query point, so candidates must be dataset points.
    """

    def __init__(
        self,
        starting_material: str | None = None,
        objective: str = "combined",
        ligand_descriptor_dim: int = 10,
        dim_red_method: str = "pca",
    ):
        """Load the dataset and build the lookup table ``X`` / ``Y``.

        With ``ligand_descriptor_dim == 10`` the precomputed tensors
        ``benchmark_problems/data/asy_hyd_self{x,y}_pca10.pt`` are loaded and
        ``dim_red_method`` has no effect. Otherwise the 36 ligand descriptors
        in ``benchmark_problems/data/C=C_AH_dataset.xlsx`` are reduced with
        PCA or UMAP. Paths are relative, so run from the repo root.

        Args:
            starting_material: One of ``"SM1"`` to ``"SM5"``, or None to vary
                the starting material. Stored on the instance; the lookup
                table always contains all starting materials.
            objective: ``"conversion"``, ``"enantiomeric_excess"`` or
                ``"combined"``.
            ligand_descriptor_dim: Number of reduced ligand dimensions,
                1 to 36.
            dim_red_method: ``"pca"`` or ``"umap"``.

        Raises:
            ValueError: If any argument is outside the values listed above.
        """
        if starting_material not in ["SM1", "SM2", "SM3", "SM4", "SM5", None]:
            raise ValueError(
                "Available starting materials are SM1, SM2, SM3, SM4, SM5. "
                "To vary the starting material, do not enter it."
            )
        if objective not in ["enantiomeric_excess", "conversion", "combined"]:
            raise ValueError(
                "Available starting materials are enantiomeric_excess and conversion."
            )
        if ligand_descriptor_dim not in list(range(1, 37)):
            raise ValueError(
                "There are 36 ligand descriptors. Choose ligand_descriptor_dim 1 to 36."
            )
        if dim_red_method not in ["umap", "pca"]:
            raise ValueError("Only umap and pca are currently supported.")
        self.starting_material = starting_material
        self.objective = objective
        if ligand_descriptor_dim == 10:
            self.X = torch.load(
                "benchmark_problems/data/asy_hyd_selfx_pca10.pt"
            ).double()
            self.Y = torch.load(
                "benchmark_problems/data/asy_hyd_selfy_pca10.pt"
            ).double()
            self.Y = torch.column_stack((self.Y, self.Y.mean(dim=1)))
        else:
            df_descriptors = excel_to_df(
                "benchmark_problems/data/C=C_AH_dataset.xlsx",
                sheet_name="ligand_descriptors",
            )
            if dim_red_method == "umap":
                self.ligand_labels, self.ligand_descriptors = apply_umap(df_descriptors)
            else:
                self.ligand_labels, self.ligand_descriptors, self.pca_variance_list = (
                    apply_pca(df_descriptors)
                )
            df_all_combinations = excel_to_df(
                "benchmark_problems/data/C=C_AH_dataset.xlsx",
                sheet_name="all_combinations",
            )

            df_X = df_all_combinations.loc[:, "SM":"Time"]
            df_ligand_labels = df_all_combinations.loc[:, "Ligand"]

            tensor_non_ligand = torch.from_numpy(df_X.to_numpy())
            tensor_ligand_labels = torch.from_numpy(df_ligand_labels.to_numpy())
            tensor_ligand_descriptors = self.ligand_descriptors[
                tensor_ligand_labels - 1
            ]

            self.X = torch.cat(
                (tensor_non_ligand, tensor_ligand_descriptors), dim=1
            ).double()
            self.Y = torch.from_numpy(
                df_all_combinations.loc[
                    :, "Conversion":"Enantiomeric_excess"
                ].to_numpy()
            ).double()
            self.Y = torch.column_stack((self.Y, self.Y.mean(dim=1)))

    def bounds(self) -> torch.Tensor | None:
        return torch.stack(
            (torch.min(self.X, dim=0).values, torch.max(self.X, dim=0).values), dim=0
        )

    def global_max(self) -> float | None:
        """Return None: the global maximum is not defined for this dataset benchmark.

        #TODO Can wait. Define by conditionals on SM, and target choice.
        """
        pass

    def global_maximizers(self) -> list[torch.Tensor]:
        """Return None: the global maximizers are not defined for this dataset benchmark.

        #TODO Can wait. Return the global maximizer index.
        """
        pass

    def value(self, x: torch.Tensor, tolerance=1e-6) -> torch.Tensor:
        """Look up the measured outcome of each candidate in the dataset.

        Each row of ``x`` is matched to the dataset entry with the smallest
        max-abs difference over all ``d`` columns.

        Args:
            x: ``n x d`` or ``d`` tensor of candidates in the layout of
                ``self.X``.
            tolerance: Largest allowed max-abs difference between a candidate
                and its nearest dataset entry.

        Returns:
            ``n x 1`` tensor: conversion, enantiomeric excess, or for
            ``"combined"`` the mean of the two.

        Raises:
            ValueError: If any candidate has no dataset entry within
                ``tolerance``.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_typed = x.to(self.X.dtype)

        diffs = torch.abs(self.X.unsqueeze(0) - x_typed.unsqueeze(1))

        max_diffs_per_point, _ = torch.max(diffs, dim=2)

        min_errors, indices = torch.min(max_diffs_per_point, dim=1)

        failed_mask = min_errors > tolerance

        if torch.any(failed_mask):
            failed_indices = torch.where(failed_mask)[0]
            error_details = []

            for idx in failed_indices:
                val = x[idx].tolist()
                err = min_errors[idx].item()
                error_details.append(
                    f"Point at index {idx} (Value: {val}) missed nearest match by {err:.2e}"
                )
            error_msg = (
                f"x is not found in the allowed values within tolerance {tolerance}.\n"
                + "\n".join(
                    error_details[:10]
                )  # Limiting to 10 to avoid flooding the console
            )
            if len(error_details) > 10:
                error_msg += f"\n... and {len(error_details) - 10} more failures."
            raise ValueError(error_msg)

        if self.objective == "conversion":
            return self.Y[indices, 0].reshape(-1, 1)
        if self.objective == "enantiomeric_excess":
            return self.Y[indices, 1].reshape(-1, 1)
        return self.Y[indices, 2].reshape(-1, 1)
