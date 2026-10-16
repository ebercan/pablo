"""
Plan ahead problem class. Takes the parsed dataset and orchestrates
the sequential optimization process for the Next/Subsequent/Following batches.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
from pablo.optimize import (
    best_observed_f,
    optimize_acqf_mixed_GA,
    optimize_plan_ahead_qlogEI_fantasized,
)
from botorch.acquisition import qLogExpectedImprovement
from botorch.models.model import Model
from torch import Tensor

from pablo.utils.nhc_utils import read_data


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlanAheadNHC:
    r"""Sequential PlanAhead orchestrator for NHC datasets.

    Manages the parsing of available, ordered, and unordered discrete configurations
    for catalyst, solvent, and base. It generates cartesian products to form
    compatible constraint tensors (`x_available`, `x_ordered`, `x_unordered`) and
    provides optimization methods to generate the next batch of experiments or
    suggest order combinations for search space expansion.
    """

    def __init__(
        self,
        model: Model,
        excel_file: str = "nhc_doe.xlsx",
        bounds: Optional[Tensor] = None,
        num_fantasies: Optional[List[int]] = None,
        q_batch: int = 6,
        q_expand: int = 1,
        n_samples: int = 20000,
        popsize: int = 150,
        batch_constraints: Optional[Dict[int, int]] = None,
        fixed_dims: Optional[Dict[int, float]] = None,
        prebatch_newly_available: bool = True,
    ):
        r"""Initialize the PlanAheadNHC orchestrator from an Excel dataset.

        Args:
            model: Fitted BoTorch surrogate model.
            excel_file: Path to the dataset excel file mapping reaction agents.
            bounds: `(2, d)` boundaries for the full continuous+discrete search space.
            num_fantasies: Number of fantasy samples per lookahead step.
                Defaults to `[10, 5, 3, 2]`.
            q_batch: Batch size for intermediate evaluation steps. Defaults to 6.
            q_expand: Number of space expansion candidates to recommend. Defaults to 1.
            n_samples: Evaluation budget for GA internal optimizations.
            popsize: GA population size.
            batch_constraints: Dictionary mapping dimension indices to ``k``,
                the number of distinct values allowed across the batch.
                Defaults to ``{3: 3}`` (at most 3 distinct values in dimension 3). In
                the ``pablo.main`` continuous layout dimension 3 is Time and
                Temperature is dimension 0.
            fixed_dims: Dictionary enforcing constant dimensions.
            prebatch_newly_available: If True, prevents components that became available in prior batches
                from populating subsequent `x_ordered` recommendation bounds.
        """
        self.model = model
        self.excel_file = excel_file

        agents, var_defs = read_data(
            excel_file=self.excel_file, device=DEVICE, dtype=DTYPE
        )
        self.available_reaction_agents = agents
        self.variable_definitions = var_defs

        self.bounds = bounds
        self.num_fantasies = (
            num_fantasies if num_fantasies is not None else [10, 5, 3, 2]
        )
        self.q_batch = q_batch
        self.q_expand = q_expand
        self.n_samples = n_samples
        self.popsize = popsize
        self.batch_constraints = (
            batch_constraints
            if batch_constraints is not None
            else {3: 3}  # Temperature index, 3 distinct values.
        )
        self.fixed_dims = fixed_dims
        self.prebatch_newly_available = prebatch_newly_available
        self.device = DEVICE
        self.dtype = DTYPE

        self.x_available, self.x_ordered, self.x_unordered = self._build_tensors()

    def _generate_discrete_grid(
        self, catalysts: List[Any], solvents: List[Any], bases: List[Any]
    ) -> set:
        r"""Helper to generate strictly ordered cartesian products of configurations.
            Returns:
                set: Return value of _generate_discrete_grid.
        
        Returns:
            set: Return value of _generate_discrete_grid.
    """
        return set(itertools.product(catalysts, solvents, bases))

    def _build_tensors(self) -> Tuple[Tensor, List[Tensor], Tensor]:
        r"""Parse batch availabilities and generate valid grid tensors for the dimensions.

        Translates the `available_reaction_agents` constraints into cartesian products,
        preventing elements present in prior batches from appearing as "new" components
        in subsequent future batches.

        Returns:
            Tuple[Tensor, List[Tensor], Tensor]:
                - `x_available`: `(n, 3)` configurations physically present in the immediate batch.
                - `x_ordered`: List of `(n_i, 3)` configurations representing newly available
                  structures at each future snapshot.
                - `x_unordered`: `(n_un, 3)` configurations that are never physically ordered.
        """
        agents = self.available_reaction_agents

        all_cats = set(self.variable_definitions["Catalyst"].options)
        all_sols = set(self.variable_definitions["Solvent"].options)
        all_bas = set(self.variable_definitions["Base"].options)

        num_future_batches = len(agents["Catalyst"]) - 1

        accumulated_cat = set(agents["Catalyst"][0])
        accumulated_sol = set(agents["Solvent"][0])
        accumulated_bas = set(agents["Base"][0])

        prev_combos = self._generate_discrete_grid(
            accumulated_cat, accumulated_sol, accumulated_bas
        )

        if len(prev_combos) > 0:
            x_available = torch.tensor(
                list(prev_combos), dtype=self.dtype, device=self.device
            )
        else:
            x_available = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        x_ordered = []

        # Calculate explicitly what newly ordered permutations open up at each consecutive step

        for i in range(1, num_future_batches + 1):
            accumulated_cat.update(agents["Catalyst"][i])
            accumulated_sol.update(agents["Solvent"][i])
            accumulated_bas.update(agents["Base"][i])

            curr_combos = self._generate_discrete_grid(
                accumulated_cat, accumulated_sol, accumulated_bas
            )

            # If prebatch_newly_available is True, we mathematically exclude ANY combo
            # that was previously achievable from arriving in 'x_ordered'

            if self.prebatch_newly_available:
                new_combos = curr_combos - prev_combos
            else:
                new_combos = curr_combos
            prev_combos = curr_combos

            if len(new_combos) > 0:
                grid = torch.tensor(
                    list(new_combos), dtype=self.dtype, device=self.device
                )
            else:
                grid = torch.empty((0, 3), dtype=self.dtype, device=self.device)
            x_ordered.append(grid)
        # Unordered options encompass all existing domains missing from the final snapshot

        all_combos = self._generate_discrete_grid(all_cats, all_sols, all_bas)
        unordered_combos = all_combos - prev_combos

        if len(unordered_combos) > 0:
            x_unordered = torch.tensor(
                list(unordered_combos), dtype=self.dtype, device=self.device
            )
        else:
            x_unordered = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        return x_available, x_ordered, x_unordered

    def generate_next_batch(
        self, best_f: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Generates the optimal recommendations for the current immediate experimental batch.

        Employs the baseline `qLogExpectedImprovement` acqusition function alongside
        a localized GA optimizing strict categorical boundaries tied closely to
        the existing `x_available`.

        Args:
            best_f: Optional override for the model's currently verified `best_f` value.

        Returns:
            tuple[Tensor, Tensor]: A two-element tuple containing:

            - **candidates** (*Tensor*): Joint `(q_batch, d)` suggestions from current sets.
            - **acq_value** (*Tensor*): Expected joint utility projection for the query.
        """
        if best_f is None:
            best_f = best_observed_f(self.model)
        acqf = qLogExpectedImprovement(model=self.model, best_f=best_f)

        if self.x_available.shape[0] > 0:
            cat_emb = self.x_available[:, 0].unique().unsqueeze(-1)
            sol_emb = self.x_available[:, 1].unique().unsqueeze(-1)
            bas_emb = self.x_available[:, 2].unique().unsqueeze(-1)
        else:
            cat_emb = sol_emb = bas_emb = None

        candidates, acq_value = optimize_acqf_mixed_GA(
            acq_function=acqf,
            bounds=self.bounds,
            q=self.q_batch,
            n_evals_ga=self.n_samples,
            pop_size=self.popsize,
            cat_emb_avail=cat_emb,
            sol_emb_avail=sol_emb,
            bas_emb_avail=bas_emb,
            batch_constraints=self.batch_constraints,
            fixed_dims=self.fixed_dims,
        )
        return candidates, acq_value

    def generate_catalyst_order(self, delay: int = 4) -> Tuple[Tensor, Tensor]:
        r"""Acts as an expander optimizer proposing future elements via multi-step lookahead.

        Wraps the internal logic of `optimize_plan_ahead_qlogEI_fantasized` across predefined
        search horizons dictated by `delay`. Candidates are ultimately suggested exclusively
        from the `x_unordered` tensor arrays.

        Args:
            delay: Number of sequential lookahead steps simulating subsequent batches.
                Must equal ``len(self.num_fantasies)`` (4 with the default
                ``num_fantasies``); otherwise ``optimize_plan_ahead_qlogEI_fantasized``
                raises ``ValueError``.

        Returns:
            tuple[Tensor, Tensor]: A two-element tuple containing:

            - **candidates** (*Tensor*): A `(q_expand, d)` sample of discrete+continuous configs.
            - **acq_value** (*Tensor*): Scalar tensor with the joint lookahead acquisition
              value of the completed ``q_batch``.
        """
        candidates, acq_value = optimize_plan_ahead_qlogEI_fantasized(
            model=self.model,
            q_batch=self.q_batch,
            q_expand=self.q_expand,
            x_available=self.x_available,
            x_ordered=self.x_ordered,
            x_unordered=self.x_unordered,
            bounds=self.bounds,
            k=delay,
            num_fantasies=self.num_fantasies,
            n_samples=self.n_samples,
            popsize=self.popsize,
            batch_constraints=self.batch_constraints,
            fixed_dims=self.fixed_dims,
        )
        return candidates, acq_value
