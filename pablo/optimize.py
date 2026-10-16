import functools
import itertools
import math
import random
import time
import warnings
from numbers import Integral
from typing import Callable

import numpy as np
import torch

from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
from botorch.sampling.normal import SobolQMCNormalSampler
from pymoo.core.crossover import Crossover
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableGA,
    MixedVariableMating,
)
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice, Integer, Real
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.optimize import minimize
from torch import Tensor

from pablo.utils.acq_utils import best_observed_f, get_joint_log_ei

torch.set_default_dtype(torch.float64)


BatchConstraintSpec = int | tuple[int, ...] | list[int]
BatchConstraints = dict[int, BatchConstraintSpec]


def _normalize_partition_spec(
    partition: tuple[int, ...] | list[int],
    q_expected: int | None,
    dim_idx: int,
) -> tuple[int, ...]:
    """Validate and canonicalize an explicit partition specification.

    Args:
        partition: Sequence of group sizes for one batch-constrained dimension.
        q_expected: Expected sum of partition entries. If ``None``, sum validation
            is skipped.
        dim_idx: Dimension index being validated.

    Returns:
        A descending-sorted tuple of positive integer group sizes.

    Raises:
        ValueError: If partition values are invalid or the sum does not match
            ``q_expected``.
    """
    normalized = tuple(partition)
    if len(normalized) == 0:
        raise ValueError(
            f"Invalid explicit partition for dim {dim_idx}: partition cannot be empty"
        )

    if any((not isinstance(v, Integral)) or isinstance(v, bool) for v in normalized):
        raise ValueError(
            f"Invalid explicit partition for dim {dim_idx}: all entries must be integers"
        )

    normalized = tuple(int(v) for v in normalized)
    if any(v <= 0 for v in normalized):
        raise ValueError(
            f"Invalid explicit partition for dim {dim_idx}: all entries must be > 0, got {normalized}"
        )

    if q_expected is not None and sum(normalized) != q_expected:
        raise ValueError(
            "Invalid explicit partition for dim "
            f"{dim_idx}: partition {normalized} sums to {sum(normalized)} "
            f"but expected {q_expected}"
        )

    return tuple(sorted(normalized, reverse=True))


def _normalize_batch_constraints(
    batch_constraints: BatchConstraints | None,
    q_expected: int | None = None,
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    """Normalize mixed-style batch constraints into grouped-count and partition maps.

    Args:
        batch_constraints: Optional mapping from dimension index to either:
            - integer ``k`` (legacy mode), or
            - explicit partition sequence (exact mode).
        q_expected: Expected batch size for explicit partition sums. If ``None``,
            explicit partitions are validated except for sum consistency.

    Returns:
        A two-element tuple:

        - grouped_counts: Map from dim index to group count ``k``.
        - explicit_partitions: Map from dim index to exact partition tuple.

    Raises:
        ValueError: If any constraint specification is invalid.
    """
    if batch_constraints is None:
        return {}, {}

    grouped_counts: dict[int, int] = {}
    explicit_partitions: dict[int, tuple[int, ...]] = {}

    for dim_idx, spec in batch_constraints.items():
        if isinstance(spec, Integral) and not isinstance(spec, bool):
            k_val = int(spec)
            if k_val <= 0:
                raise ValueError(
                    f"Invalid batch constraint for dim {dim_idx}: k must be > 0, got {k_val}"
                )
            if q_expected is not None and k_val > q_expected:
                raise ValueError(
                    "Invalid batch constraint for dim "
                    f"{dim_idx}: k={k_val} exceeds q={q_expected}"
                )
            grouped_counts[dim_idx] = k_val
            continue

        if isinstance(spec, (tuple, list)):
            partition = _normalize_partition_spec(spec, q_expected, dim_idx)
            grouped_counts[dim_idx] = len(partition)
            explicit_partitions[dim_idx] = partition
            continue

        raise ValueError(
            "Invalid batch constraint for dim "
            f"{dim_idx}: expected int or partition tuple/list, got {type(spec).__name__}"
        )

    return grouped_counts, explicit_partitions


class MixedAllowedSampling(Sampling):
    """Initial population sampling for mixed-variable GA with allowed sets.

    Generates random samples for continuous variables within their bounds
    and random indices into the allowed discrete set for discrete variables.

    Args:
        n_allowed: Number of allowed discrete configurations.
        d_continuous: Number of continuous dimensions.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
    """

    def __init__(self, n_allowed, d_continuous, bounds):
        super().__init__()
        self.n_allowed = n_allowed
        self.d_continuous = d_continuous
        self.bounds = bounds

    def _do(self, problem, n_samples, **kwargs):
        X = {}
        for k in range(n_samples):
            X[k] = {}
            for var_key in problem.vars.keys():
                var = problem.vars[var_key]
                if isinstance(var, Real):
                    low, high = var.bounds
                    X[k][var_key] = np.random.uniform(low, high)
                elif isinstance(var, Integer):
                    lo, hi = var.bounds
                    X[k][var_key] = np.random.randint(lo, hi + 1)
                else:
                    X[k][var_key] = np.random.choice(var.options)
        return X


class MixedAllowedCrossover(Crossover):
    """Crossover operator for mixed-variable GA with allowed discrete sets.

    Performs blend crossover (BLX-alpha with alpha=1) for continuous variables
    and uniform crossover for discrete (index-based) variables.

    Args:
        n_allowed: Number of allowed discrete configurations.
    """

    def __init__(self, n_allowed):
        super().__init__(n_parents=2, n_offsprings=2)
        self.n_allowed = n_allowed

    def _do(self, problem, X, **kwargs):
        n_matings = len(X) // 2
        offspring = []
        for i in range(n_matings):
            p1 = X[2 * i]
            p2 = X[2 * i + 1]
            o1 = {}
            o2 = {}
            for var_key in problem.vars.keys():
                var = problem.vars[var_key]
                if isinstance(var, Real):
                    alpha = np.random.uniform(0, 1)
                    o1[var_key] = alpha * p1[var_key] + (1 - alpha) * p2[var_key]
                    o2[var_key] = (1 - alpha) * p1[var_key] + alpha * p2[var_key]
                else:
                    if np.random.random() < 0.5:
                        o1[var_key] = p1[var_key]
                        o2[var_key] = p2[var_key]
                    else:
                        o1[var_key] = p2[var_key]
                        o2[var_key] = p1[var_key]
            offspring.append(o1)
            offspring.append(o2)
        return offspring


class MixedAllowedMutation(Mutation):
    """Mutation operator for mixed-variable GA with allowed discrete sets.

    Applies random reset mutation independently to each variable with a
    given probability. Continuous variables are resampled uniformly within
    their bounds; discrete variables are reassigned a random allowed index.

    Args:
        n_allowed: Number of allowed discrete configurations.
        prob: Per-variable mutation probability. Defaults to 0.1.
    """

    def __init__(self, n_allowed, prob=0.1):
        super().__init__()
        self.n_allowed = n_allowed
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        for ind in X:
            for var_key in problem.vars.keys():
                if np.random.random() < self.prob:
                    var = problem.vars[var_key]
                    if isinstance(var, Real):
                        low, high = var.bounds
                        ind[var_key] = np.random.uniform(low, high)
                    elif isinstance(var, Integer):
                        lo, hi = var.bounds
                        ind[var_key] = np.random.randint(lo, hi + 1)
                    else:
                        ind[var_key] = np.random.choice(var.options)
        return X


def _integer_partitions(
    n: int, k: int, _max_val: int | None = None
) -> list[tuple[int, ...]]:
    """Enumerate all unordered partitions of ``n`` into exactly ``k`` positive parts.

    Each partition is returned as a tuple sorted in descending order.

    Args:
        n: The integer to partition.
        k: The exact number of positive parts.
        _max_val: Internal recursion bound (largest allowed part).

    Returns:
        A list of tuples, each a descending-sorted partition of ``n``
        into ``k`` parts.  For example, ``_integer_partitions(6, 3)``
        yields ``[(4, 1, 1), (3, 2, 1), (2, 2, 2)]``.
    """
    if _max_val is None:
        _max_val = n - k + 1
    if k == 1:
        if 1 <= n <= _max_val:
            return [(n,)]
        return []
    results: list[tuple[int, ...]] = []
    for i in range(min(n - k + 1, _max_val), 0, -1):
        for rest in _integer_partitions(n - i, k - 1, i):
            results.append((i,) + rest)
    return results


def _candidate_group_map(partition: tuple[int, ...]) -> list[int]:
    """Map each candidate index to its group index given a partition.

    The partition is expected sorted descending (e.g. ``(3, 2, 1)``).
    Candidates are assigned left-to-right: the first ``partition[0]``
    candidates belong to group 0, the next ``partition[1]`` to group 1,
    and so on.

    Args:
        partition: A tuple of positive integers summing to ``q``.

    Returns:
        A list of length ``q`` where element ``j`` is the group index
        for candidate ``j``.
    """
    mapping: list[int] = []
    for group_idx, size in enumerate(partition):
        mapping.extend([group_idx] * size)
    return mapping


class CanonicalLayout:
    r"""Slot-exchange symmetry of a batch-GA genome.

    A pymoo mixed-variable individual for a ``q``-candidate batch is a flat,
    1-indexed dict. Candidate ``c`` owns the ``n_var_per_cand`` keys starting at
    ``c * n_var_per_cand + 1``; each batch-constrained ("grouped") dimension owns
    ``k`` further keys inside the grouped block that starts at
    ``grouped_offset + 1``.

    A joint batch acquisition function is a set function over the ``q``
    candidates, so any relabeling that preserves the grouped-dimension
    assignment leaves its value unchanged. Two relabelings exist:

    - **within-block**: candidates sharing a grouped-dim group are freely
      interchangeable;
    - **between-block**: two blocks of equal size may swap, provided their
      grouped-dim values swap with them.

    For ``TEMPERATURE_PARTITION = (2, 2, 2)`` at ``q = 6`` the group has order
    ``2!**3 * 3! = 48``. With no grouped dims every slot is interchangeable and
    the order is ``q!``.

    The sort key places discrete keys before continuous ones. Ordering by a
    continuous coordinate would make the canonical labeling flip on almost every
    mutation, teleporting genes between slots and destroying positional linkage;
    discrete indices change rarely, so the labeling stays stable across
    generations.

    Args:
        q: Number of GA-controlled candidates.
        n_var_per_cand: Genome keys owned by each candidate.
        disc_offsets: Candidate-local offsets of the discrete keys, in the order
            they enter the sort key.
        cont_offsets: Candidate-local offsets of the continuous keys.
        grouped_offset: Number of genome keys preceding the grouped block.
        grouped_starts: Offset of each grouped dim within the grouped block.
        grouped_ks: Number of groups ``k`` for each grouped dim.
        blocks: Candidate indices per block.
        block_group_idx: Per block, its group index in each grouped dim.
        permutable_runs: Index runs of ``blocks`` that may permute among
            themselves.
        cat_local: Candidate-local offset of the catalyst index, or ``None``
            outside component mode.
        bas_local: Candidate-local offset of the base index, or ``None``
            outside component mode.
    """

    def __init__(
        self,
        q: int,
        n_var_per_cand: int,
        disc_offsets: list[int],
        cont_offsets: list[int],
        grouped_offset: int,
        grouped_starts: list[int],
        grouped_ks: list[int],
        blocks: list[list[int]],
        block_group_idx: list[tuple[int, ...]],
        permutable_runs: list[list[int]],
        cat_local: int | None = None,
        bas_local: int | None = None,
    ):
        self.q = q
        self.n_var_per_cand = n_var_per_cand
        self.disc_offsets = disc_offsets
        self.cont_offsets = cont_offsets
        self.grouped_offset = grouped_offset
        self.grouped_starts = grouped_starts
        self.grouped_ks = grouped_ks
        self.n_grouped_vars = sum(grouped_ks)
        self.blocks = blocks
        self.block_group_idx = block_group_idx
        self.permutable_runs = permutable_runs
        self.cat_local = cat_local
        self.bas_local = bas_local

    @property
    def group_order(self) -> int:
        """Order of the symmetry group this layout quotients out."""
        order = 1
        for members in self.blocks:
            for i in range(2, len(members) + 1):
                order *= i
        for run in self.permutable_runs:
            for i in range(2, len(run) + 1):
                order *= i
        return order

    def slot_key(self, flat_vals: dict, cand: int) -> tuple:
        """Sort key for one candidate slot, discrete keys first."""
        base = cand * self.n_var_per_cand
        return tuple(flat_vals[base + o + 1] for o in self.disc_offsets) + tuple(
            flat_vals[base + o + 1] for o in self.cont_offsets
        )

    def canonicalize(self, flat_vals: dict, salt_flags: np.ndarray | None = None) -> dict:
        """Project one genome onto its canonical labeling, in place.

        Args:
            flat_vals: 1-indexed genome dict, modified in place.
            salt_flags: Optional ``(n_cat,)`` bool array. Where False, the base
                index is collapsed to 0 because the base slice is zeroed
                downstream and the index is a don't-care.

        Returns:
            The same dict, canonicalized.
        """
        nvc = self.n_var_per_cand

        if salt_flags is not None and self.bas_local is not None:
            for cand in range(self.q):
                base = cand * nvc
                if not salt_flags[int(flat_vals[base + self.cat_local + 1])]:
                    flat_vals[base + self.bas_local + 1] = 0

        slots = [
            [flat_vals[cand * nvc + j + 1] for j in range(nvc)] for cand in range(self.q)
        ]
        keys = [self.slot_key(flat_vals, cand) for cand in range(self.q)]
        grouped = [
            flat_vals[self.grouped_offset + j + 1] for j in range(self.n_grouped_vars)
        ]

        payloads = []
        for b, members in enumerate(self.blocks):
            order = sorted(members, key=lambda c: keys[c])
            payloads.append(
                (
                    [keys[c] for c in order],
                    [slots[c] for c in order],
                    tuple(
                        grouped[start + self.block_group_idx[b][r]]
                        for r, start in enumerate(self.grouped_starts)
                    ),
                )
            )

        for run in self.permutable_runs:
            if len(run) < 2:
                continue
            reordered = sorted((payloads[b] for b in run), key=lambda p: (p[0], p[2]))
            for b, payload in zip(run, reordered):
                payloads[b] = payload

        for b, members in enumerate(self.blocks):
            _, ordered_slots, grouped_vals = payloads[b]
            for cand, slot in zip(members, ordered_slots):
                base = cand * nvc
                for j, value in enumerate(slot):
                    flat_vals[base + j + 1] = value
            for r, start in enumerate(self.grouped_starts):
                grouped[start + self.block_group_idx[b][r]] = grouped_vals[r]

        for j, value in enumerate(grouped):
            flat_vals[self.grouped_offset + j + 1] = value
        return flat_vals


def _build_canonical_layout(
    q: int,
    free_dims: list[int],
    grouped_dims: dict[int, int],
    partition_map: dict[int, tuple[int, ...]],
    d_continuous: int,
    use_component_mode: bool,
) -> CanonicalLayout:
    """Derive the slot-exchange symmetry from the GA's own batch constraints.

    The block structure is read from ``partition_map`` rather than supplied
    separately: ``_candidate_group_map`` already states which candidates share a
    grouped-dim value, and that is exactly the block partition.

    Between-block permutation is enabled only when every participating grouped
    dim has one group per block, so that a block permutation induces a
    well-defined permutation of the grouped values. Otherwise only within-block
    exchange is canonicalized.

    Args:
        q: Number of GA-controlled candidates.
        free_dims: Dimension indices optimized per candidate.
        grouped_dims: Grouped dim index to group count ``k``.
        partition_map: Grouped dim index to its partition tuple.
        d_continuous: Number of continuous dimensions.
        use_component_mode: Whether the discrete part is encoded as three
            component indices per candidate.

    Returns:
        The layout describing the genome's symmetry.
    """
    if use_component_mode:
        free_cont_dims = [i for i in free_dims if i < d_continuous]
        participating = {
            dim: k for dim, k in grouped_dims.items() if dim < d_continuous
        }
        n_free_cont = len(free_cont_dims)
        n_var_per_cand = n_free_cont + 3
        cont_offsets = list(range(n_free_cont))
        disc_offsets = [n_free_cont, n_free_cont + 1, n_free_cont + 2]
        cat_local: int | None = n_free_cont
        bas_local: int | None = n_free_cont + 2
    else:
        participating = dict(grouped_dims)
        n_var_per_cand = len(free_dims)
        cont_offsets = [j for j, dim in enumerate(free_dims) if dim < d_continuous]
        disc_offsets = [j for j, dim in enumerate(free_dims) if dim >= d_continuous]
        cat_local = None
        bas_local = None

    grouped_order = sorted(participating.keys())
    grouped_ks = [participating[dim] for dim in grouped_order]
    grouped_starts: list[int] = []
    running = 0
    for k_val in grouped_ks:
        grouped_starts.append(running)
        running += k_val

    if grouped_order:
        group_maps = [_candidate_group_map(partition_map[dim]) for dim in grouped_order]
        block_ids: list[tuple[int, ...]] = []
        members_by_id: dict[tuple[int, ...], list[int]] = {}
        for cand in range(q):
            bid = tuple(gm[cand] for gm in group_maps)
            if bid not in members_by_id:
                members_by_id[bid] = []
                block_ids.append(bid)
            members_by_id[bid].append(cand)
        blocks = [members_by_id[bid] for bid in block_ids]
        block_group_idx = list(block_ids)
        permutable = all(k_val == len(blocks) for k_val in grouped_ks)
    else:
        blocks = [list(range(q))]
        block_group_idx = [()]
        permutable = False

    permutable_runs: list[list[int]] = []
    if permutable:
        start = 0
        for b in range(1, len(blocks) + 1):
            if b == len(blocks) or len(blocks[b]) != len(blocks[start]):
                permutable_runs.append(list(range(start, b)))
                start = b
    else:
        permutable_runs = [[b] for b in range(len(blocks))]

    return CanonicalLayout(
        q=q,
        n_var_per_cand=n_var_per_cand,
        disc_offsets=disc_offsets,
        cont_offsets=cont_offsets,
        grouped_offset=q * n_var_per_cand,
        grouped_starts=grouped_starts,
        grouped_ks=grouped_ks,
        blocks=blocks,
        block_group_idx=block_group_idx,
        permutable_runs=permutable_runs,
        cat_local=cat_local,
        bas_local=bas_local,
    )


class CanonicalRepair(Repair):
    """Pymoo repair projecting every individual onto its canonical labeling.

    Applied to the initial population and to every offspring before duplicate
    elimination, so that the GA searches the quotient of the genome space by the
    batch symmetry group rather than the full space.

    Args:
        layout: Symmetry layout of the genome.
        salt_flags: Optional ``(n_cat,)`` bool array driving base-index
            collapse for non-salt catalysts.
    """

    def __init__(self, layout: CanonicalLayout, salt_flags: np.ndarray | None = None):
        super().__init__()
        self.layout = layout
        self.salt_flags = salt_flags

    def _do(self, problem, X, **kwargs):
        for flat_vals in X:
            self.layout.canonicalize(flat_vals, self.salt_flags)
        return X


def _enumerate_partition_groupings(
    q: int,
    partition: tuple[int, ...],
) -> list[list[list[int]]]:
    """Enumerate unique groupings for a fixed partition of ``q`` candidates.

    Groups are unlabeled: permutations of same-size groups are treated as
    identical.  The grouping is encoded as a list of groups, where each
    group is a list of candidate indices.

    Args:
        q: Total number of candidates.
        partition: Tuple of group sizes summing to ``q``.

    Returns:
        A list of groupings, where each grouping is a list of index lists.

    Raises:
        ValueError: If the partition is invalid for ``q``.
    """
    if q <= 0:
        raise ValueError(f"q must be > 0, got q={q}")
    if len(partition) == 0:
        raise ValueError("partition cannot be empty")
    if any(v <= 0 for v in partition):
        raise ValueError(f"partition entries must be > 0, got {partition}")
    if sum(partition) != q:
        raise ValueError(
            f"partition {partition} sums to {sum(partition)} but expected q={q}"
        )

    def _labeled(remaining, sizes):
        if not sizes:
            return [[]]
        size = sizes[0]
        result: list[list[list[int]]] = []
        for combo in itertools.combinations(remaining, size):
            rest = tuple(x for x in remaining if x not in combo)
            for tail in _labeled(rest, sizes[1:]):
                result.append([list(combo)] + tail)
        return result

    raw = _labeled(tuple(range(q)), partition)

    seen: set[tuple[tuple[int, ...], ...]] = set()
    unique: list[list[list[int]]] = []
    for grouping in raw:
        canon = list(grouping)
        i = 0
        while i < len(partition):
            j = i + 1
            while j < len(partition) and partition[j] == partition[i]:
                j += 1
            if j > i + 1:
                canon[i:j] = sorted(canon[i:j])
            i = j
        key = tuple(tuple(g) for g in canon)
        if key not in seen:
            seen.add(key)
            unique.append(grouping)
    return unique


def _build_free_dim_info(
    d: int,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
) -> tuple[list[int], dict[int, int], dict[int, float]]:
    """Classify each dimension as free, grouped (batch-constrained), or fixed.

    Args:
        d: Total number of dimensions.
        batch_constraints: Optional dictionary mapping dimension indices to:

            - integer ``k`` for legacy grouped constraints, or
            - explicit partition tuple/list for exact grouped constraints.

            When ``k = 1`` every candidate shares a single value.
        fixed_dims: Optional dictionary mapping dimension indices to fixed
            values that should not be optimized.

    Returns:
        A three-element tuple:

        - **free_dims** (*list[int]*): Dims optimized independently per candidate.
        - **grouped_dims** (*dict[int, int]*): Dims that are batch-constrained,
          mapping dimension index to the allowed number of distinct values ``k``.
        - **fixed_dims_dict** (*dict[int, float]*): Dims pinned to constant values.
    """
    grouped_counts, _ = _normalize_batch_constraints(batch_constraints)
    if fixed_dims is None:
        fixed_dims = {}

    grouped_set = set(grouped_counts.keys())
    fixed_set = set(fixed_dims.keys())

    free_dims = [i for i in range(d) if i not in grouped_set and i not in fixed_set]
    # fixed takes precedence over grouped
    grouped_dims = {dim: grouped_counts[dim] for dim in sorted(grouped_set - fixed_set)}
    return free_dims, grouped_dims, fixed_dims


def _partial_partition_for_step(
    target_partition: tuple[int, ...],
    k: int,
) -> tuple[int, ...] | None:
    """Compute the partial partition for ``k`` points given a target.

    Returns ``None`` when ``k <= len(target_partition)`` (too few points
    to require grouping).  Otherwise distributes ``k`` across
    ``len(target_partition)`` groups, each capped at the corresponding
    target size.

    Args:
        target_partition: Full partition sorted descending, e.g. ``(2, 2, 2)``.
        k: Current number of candidates.

    Returns:
        A descending-sorted partial partition, or ``None``.
    """
    m = len(target_partition)
    if k <= m:
        return None
    partial = [1] * m
    extras = k - m
    for i in range(m):
        room = target_partition[i] - 1
        give = min(extras, room)
        partial[i] += give
        extras -= give
        if extras == 0:
            break
    if extras > 0:
        raise ValueError(
            f"Cannot distribute {k} points across target partition "
            f"{target_partition}"
        )
    return tuple(sorted(partial, reverse=True))


def _block_cumulative_ks(q_free: int, candidates_at_a_time: int) -> list[int]:
    """Cumulative candidate counts for block-wise sequential building.

    The ``q_free`` free candidates are built in successive blocks of
    ``candidates_at_a_time`` (full groups first, a single remainder group
    last). This returns the running total after each block — the ``q`` value
    passed to the GA at each step.

    Args:
        q_free: Number of free candidates to build (excludes any pre-batch).
        candidates_at_a_time: Block size; ``1`` reproduces the one-by-one
            schedule, ``>= q_free`` collapses to a single block.

    Returns:
        Ascending list of cumulative counts ending at ``q_free``. For example,
        ``(6, 2) -> [2, 4, 6]``, ``(7, 3) -> [3, 6, 7]``, ``(6, 1) ->
        [1, 2, 3, 4, 5, 6]``.

    Raises:
        ValueError: If ``candidates_at_a_time < 1``.
    """
    if candidates_at_a_time < 1:
        raise ValueError(
            f"candidates_at_a_time must be >= 1, got {candidates_at_a_time}"
        )
    ks: list[int] = []
    k = 0
    while k < q_free:
        k = min(k + candidates_at_a_time, q_free)
        ks.append(k)
    return ks


def _build_step_batch_constraints(
    full_constraints: BatchConstraints | None,
    explicit_partitions: dict[int, tuple[int, ...]],
    k: int,
) -> BatchConstraints | None:
    """Build batch constraints for an intermediate sequential step.

    For explicit partition specs, computes the partial partition for ``k``
    candidates.  Integer specs are capped at ``k`` and kept only when
    actually constraining (i.e. less than ``k``).

    Args:
        full_constraints: Original batch constraints mapping.
        explicit_partitions: Explicit partitions extracted by
            ``_normalize_batch_constraints``.
        k: Current number of candidates.

    Returns:
        Step-specific batch constraints, or ``None``.
    """
    if full_constraints is None:
        return None
    step: BatchConstraints = {}
    for dim_idx, spec in full_constraints.items():
        if dim_idx in explicit_partitions:
            partial = _partial_partition_for_step(
                explicit_partitions[dim_idx],
                k,
            )
            if partial is not None:
                step[dim_idx] = partial
        elif isinstance(spec, Integral) and not isinstance(spec, bool):
            capped = min(int(spec), k)
            if capped < k:
                step[dim_idx] = capped
    return step if step else None


def _ga_vector_to_candidates(
    flat_vals: dict,
    q: int,
    d: int,
    free_dims: list[int],
    grouped_dims: dict[int, int],
    fixed_dims: dict[int, float],
    partition_map: dict[int, tuple[int, ...]] | None = None,
    cat_emb_avail: Tensor | None = None,
    sol_emb_avail: Tensor | None = None,
    bas_emb_avail: Tensor | None = None,
    cat_is_salt_flags: Tensor | None = None,
) -> Tensor:
    r"""Reconstruct a ``(q, d)`` candidate tensor from the reduced GA vector.

    Two encoding modes:

    - **Component mode** (``cat_emb_avail``/``sol_emb_avail``/``bas_emb_avail``
      provided): each candidate carries three integer indices that pick rows
      from the per-component embedding tables. The discrete slice is assembled
      by concatenation, and the base slice is zeroed when the chosen catalyst
      is non-salt (``cat_is_salt_flags[c_idx]`` is False).
    - **Choice mode** (no component tensors): each free continuous and discrete
      dim is its own pymoo variable per candidate (uses ``parse_pymoo_mixed_variables``).

    Args:
        flat_vals: Dictionary of GA variable values (1-indexed keys).
        q: Batch size (number of candidates).
        d: Total number of dimensions per candidate.
        free_dims: Dimension indices optimized per candidate.
        grouped_dims: Dictionary mapping batch-constrained dimension indices
            to the number of distinct groups ``k``.
        fixed_dims: Dictionary of dimension index to constant value.
        partition_map: Dictionary mapping each grouped dimension index to a
            partition tuple. Defaults to the trivial ``(q,)`` per grouped dim.
        cat_emb_avail: ``(n_cat, d_cat)`` tensor of available catalyst embeddings.
        sol_emb_avail: ``(n_sol, d_sol)`` tensor of available solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` tensor of available base embeddings.
        cat_is_salt_flags: ``(n_cat,)`` bool tensor; when False at index ``c``,
            the base slice is zeroed for candidates choosing catalyst ``c``.
            ``None`` disables base zeroing (every base contributes its row).

    Returns:
        A ``(q, d)`` tensor of reconstructed candidates.
    """
    use_component_mode = cat_emb_avail is not None

    if partition_map is None:
        partition_map = {dim: (q,) for dim in grouped_dims}

    candidates = torch.zeros(q, d, dtype=torch.float64)

    if use_component_mode:
        d_cat = cat_emb_avail.shape[1]
        d_sol = sol_emb_avail.shape[1]
        d_bas = bas_emb_avail.shape[1]
        d_discrete = d_cat + d_sol + d_bas
        d_continuous = d - d_discrete

        free_cont_dims = [i for i in free_dims if i < d_continuous]
        grouped_cont_dims = {
            dim: k for dim, k in grouped_dims.items() if dim < d_continuous
        }
        n_free_cont = len(free_cont_dims)
        n_var_per_q = n_free_cont + 3
        grouped_offset = q * n_var_per_q

        cat_start = d_continuous
        sol_start = cat_start + d_cat
        bas_start = sol_start + d_sol
        bas_end = bas_start + d_bas

        for cand in range(q):
            offset = cand * n_var_per_q
            for j_local, dim_idx in enumerate(free_cont_dims):
                candidates[cand, dim_idx] = flat_vals[offset + j_local + 1]
            c_idx = int(flat_vals[offset + n_free_cont + 1])
            s_idx = int(flat_vals[offset + n_free_cont + 2])
            b_idx = int(flat_vals[offset + n_free_cont + 3])

            candidates[cand, cat_start:sol_start] = cat_emb_avail[c_idx]
            candidates[cand, sol_start:bas_start] = sol_emb_avail[s_idx]
            if cat_is_salt_flags is None or bool(cat_is_salt_flags[c_idx]):
                candidates[cand, bas_start:bas_end] = bas_emb_avail[b_idx]
            else:
                candidates[cand, bas_start:bas_end] = 0.0

        var_idx = grouped_offset + 1
        for dim_idx in sorted(grouped_cont_dims.keys()):
            k_val = grouped_cont_dims[dim_idx]
            group_map = _candidate_group_map(partition_map[dim_idx])
            for cand in range(q):
                group = group_map[cand]
                candidates[cand, dim_idx] = flat_vals[var_idx + group]
            var_idx += k_val
    else:
        free_per_q = len(free_dims)
        grouped_offset = q * free_per_q

        for cand in range(q):
            offset = cand * free_per_q
            for j_local, dim_idx in enumerate(free_dims):
                candidates[cand, dim_idx] = flat_vals[offset + j_local + 1]

        var_idx = grouped_offset + 1
        for dim_idx in sorted(grouped_dims.keys()):
            k_val = grouped_dims[dim_idx]
            group_map = _candidate_group_map(partition_map[dim_idx])
            for cand in range(q):
                group = group_map[cand]
                candidates[cand, dim_idx] = flat_vals[var_idx + group]
            var_idx += k_val

    for dim_idx, val in fixed_dims.items():
        candidates[:, dim_idx] = val

    return candidates


def parse_pymoo_mixed_variables_with_components(
    bounds: Tensor,
    n_cat: int,
    n_sol: int,
    n_bas: int,
    d_discrete: int,
    q: int = 1,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
):
    """Parse bounds into pymoo mixed variables using component-wise indexing.

    The discrete (catalyst, solvent, base) choice is encoded as three separate
    ``Integer`` variables per candidate. The full discrete row is assembled at
    AF-evaluation time by component lookup; salt-zeroing is applied there.

    Args:
        bounds: A ``2 x d`` tensor of lower and upper bounds for each dimension.
        n_cat: Number of available catalysts.
        n_sol: Number of available solvents.
        n_bas: Number of available bases.
        d_discrete: Total discrete dimensionality (``d_cat + d_sol + d_bas``).
        q: Number of candidates to optimize jointly.
        batch_constraints: Optional dictionary mapping dimension indices to:

            - integer ``k`` for legacy grouped constraints, or
            - explicit partition tuple/list for exact grouped constraints.
        fixed_dims: Optional dictionary mapping dimension indices to fixed
            values (these dimensions are excluded from GA optimization).

    Returns:
        A dictionary mapping 1-indexed variable keys to pymoo ``Real`` or
        ``Integer`` variable objects.
    """
    d = bounds.shape[1]
    d_continuous = d - d_discrete
    free_dims, grouped_dims, fixed_dims = _build_free_dim_info(
        d, batch_constraints, fixed_dims
    )

    free_cont_dims = [i for i in free_dims if i < d_continuous]
    grouped_cont_dims = {
        dim: k for dim, k in grouped_dims.items() if dim < d_continuous
    }

    vars_dict = {}
    var_counter = 1

    for _cand in range(q):
        for dim_idx in free_cont_dims:
            xl, xu = bounds[0, dim_idx].item(), bounds[1, dim_idx].item()
            vars_dict[var_counter] = Real(bounds=(xl, xu))
            var_counter += 1
        vars_dict[var_counter] = Integer(bounds=(0, max(n_cat - 1, 0)))
        var_counter += 1
        vars_dict[var_counter] = Integer(bounds=(0, max(n_sol - 1, 0)))
        var_counter += 1
        vars_dict[var_counter] = Integer(bounds=(0, max(n_bas - 1, 0)))
        var_counter += 1

    for dim_idx in sorted(grouped_cont_dims.keys()):
        k_val = grouped_cont_dims[dim_idx]
        xl, xu = bounds[0, dim_idx].item(), bounds[1, dim_idx].item()
        for _g in range(k_val):
            vars_dict[var_counter] = Real(bounds=(xl, xu))
            var_counter += 1

    return vars_dict


def _infer_acqf_device(acq_function: AcquisitionFunction) -> torch.device:
    r"""Infer the torch device used by an acquisition function model.

    Args:
        acq_function: Acquisition function instance.

    Returns:
        The device of the first model parameter when available, otherwise CPU.
    """
    if hasattr(acq_function, "model"):
        model = acq_function.model
        if hasattr(model, "parameters"):
            params = list(model.parameters())
            if params:
                return params[0].device
    return torch.device("cpu")


def _compute_batch_acq_value(
    acq_function: AcquisitionFunction,
    candidates: Tensor,
    q: int,
) -> Tensor:
    r"""Compute a single joint batch acquisition value for a set of candidates.

    For ``q = 1``, evaluates the acquisition function directly. For ``q > 1``,
    evaluates the acquisition function on the full ``1 x q x d`` batch and
    returns the result if it is a scalar. If it is not a scalar and the
    acquisition function has ``model`` and ``best_f`` attributes, returns the
    joint ``qLogExpectedImprovement`` value from ``get_joint_log_ei``;
    otherwise returns the sum of the native evaluation.

    Args:
        acq_function: The acquisition function to evaluate.
        candidates: A ``q x d`` tensor of candidate points.
        q: The number of candidates (batch size).

    Returns:
        A scalar tensor containing the joint batch acquisition value.
    """
    target_device = _infer_acqf_device(acq_function)
    eval_candidates = candidates
    if candidates.device != target_device:
        eval_candidates = candidates.to(target_device)

    if q == 1:
        with torch.no_grad():
            return acq_function(eval_candidates.unsqueeze(0))

    # For q > 1, try to evaluate the full batch jointly
    with torch.no_grad():
        batch_val = acq_function(eval_candidates.unsqueeze(0))  # (1, q, d) -> scalar

    # If the acq function natively handles q > 1, batch_val is already a scalar
    if batch_val.dim() == 0:
        return batch_val

    # Fallback: use qLogEI to compute the joint value
    if hasattr(acq_function, "model") and hasattr(acq_function, "best_f"):
        joint_val = get_joint_log_ei(
            acq_function.model, acq_function.best_f, eval_candidates
        )
        return torch.tensor(
            joint_val,
            dtype=eval_candidates.dtype,
            device=eval_candidates.device,
        )

    # Last resort: return the sum
    return batch_val.sum()


def _normalize_greedy_fixed_dims(
    fixed_dims: dict[int, float | list[float] | tuple[float, ...]] | None,
    d_continuous: int,
    q_free: int,
) -> tuple[dict[int, float], tuple[int, list[float]] | None]:
    """Normalize fixed-dimension specs for greedy optimization.

    Args:
        fixed_dims: Mapping from continuous dim index to either:

            - a scalar float/int: fixed for all newly optimized candidates, or
            - a list/tuple of floats: fixed for exactly ``n`` newly optimized
              candidates where ``n`` is the sequence length.
        d_continuous: Number of continuous dimensions.
        q_free: Number of newly optimized candidates (excluding pre-batch).

    Returns:
        A two-element tuple:

        - scalar_fixed: Mapping of dims fixed for all ``q_free`` candidates.
        - list_fixed: Optional ``(dim_idx, values)`` for one list-valued
          partial-fix dimension.

    Raises:
        ValueError: If fixed-dim specs are invalid.
    """
    if fixed_dims is None:
        return {}, None

    scalar_fixed: dict[int, float] = {}
    list_fixed: tuple[int, list[float]] | None = None

    for dim_idx, value in fixed_dims.items():
        if dim_idx < 0 or dim_idx >= d_continuous:
            raise ValueError(
                "fixed_dims supports continuous dims only: "
                f"got dim {dim_idx}, d_continuous={d_continuous}"
            )

        if isinstance(value, (list, tuple)):
            if list_fixed is not None:
                raise ValueError(
                    "At most one list-valued fixed dim is supported in greedy mode"
                )
            if len(value) == 0:
                raise ValueError(
                    f"List-valued fixed dim for dim {dim_idx} cannot be empty"
                )
            values = [float(v) for v in value]
            if len(values) > q_free:
                raise ValueError(
                    "List-valued fixed dim length exceeds q_free: "
                    f"len={len(values)}, q_free={q_free}"
                )
            list_fixed = (dim_idx, values)
            continue

        if isinstance(value, bool):
            raise ValueError(
                f"Invalid fixed value for dim {dim_idx}: bool is not supported"
            )

        scalar_fixed[dim_idx] = float(value)

    if list_fixed is not None and list_fixed[0] in scalar_fixed:
        raise ValueError(
            f"Dimension {list_fixed[0]} cannot be both scalar-fixed and list-fixed"
        )

    return scalar_fixed, list_fixed


def optimize_acqf_continuous(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    X_discrete: Tensor,
    pre_batch: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float | list[float] | tuple[float, ...]] | None = None,
    num_restarts: int = 20,
    raw_samples: int = 1024,
    progress_callback: Callable[[str], None] | None = None,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    r"""Optimize continuous dims for a fixed discrete q-batch using L-BFGS-B.

    The discrete part of each candidate is fixed by ``X_discrete``. Grouped
    constraints (e.g., temperature partitioning) are enforced by enumerating
    all unique grouping assignments for the specified partition and
    reparameterizing the optimization so that each group uses a single
    temperature variable (no equality constraints).

    Args:
        acq_function: Acquisition function to optimize (q-batch aware when
            ``q > 1``).
        bounds: ``(2, d_total)`` tensor of lower and upper bounds.
        X_discrete: ``(q, d_discrete)`` tensor of fixed discrete values.
        pre_batch: Optional ``(q_pre, d_total)`` tensor of fixed candidates
            prepended to the optimized batch. Must align with ``X_discrete``
            for the discrete slice.
        batch_constraints: Optional mapping from continuous dim index to either
            integer ``k`` (legacy grouped mode) or explicit partition tuple/list
            (exact grouped mode). Grouping applies across the full ``q``
            including ``pre_batch`` rows.
        fixed_dims: Optional mapping from continuous dim index to either:

            - scalar float/int: fixed for all free candidates, or
            - list/tuple of floats: fixed for the first ``n`` free candidates.

            At most one list-valued fixed dim is supported.
        num_restarts: Number of L-BFGS-B restarts.
        raw_samples: Number of Sobol samples for multi-start initialization.
        progress_callback: Optional callback receiving short progress messages.
        seed: Base seed for the per-grouping Sobol multi-start draw. Grouping
            ``i`` (1-indexed) uses ``seed + i``, making the refinement fully
            reproducible run-to-run (only machine speed then varies).

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): ``(q, d_total)`` tensor of optimized
          candidates (``pre_batch`` rows first, if provided).
        - **acq_value** (*Tensor*): Scalar tensor of the joint batch
          acquisition value.

    Raises:
        ValueError: If inputs are invalid or no feasible grouping assignments
            exist.
    """

    if X_discrete.ndim != 2:
        raise ValueError("X_discrete must have shape (q, d_discrete)")
    if num_restarts <= 0:
        raise ValueError(f"num_restarts must be > 0, got {num_restarts}")
    if raw_samples <= 0:
        raise ValueError(f"raw_samples must be > 0, got {raw_samples}")

    q_total = int(X_discrete.shape[0])
    if q_total <= 0:
        raise ValueError(f"X_discrete must contain at least one row, got {q_total}")

    d_total = int(bounds.shape[1])
    d_discrete = int(X_discrete.shape[1])
    d_continuous = d_total - d_discrete
    if d_continuous < 0:
        raise ValueError("X_discrete has too many discrete dims for bounds")

    if q_total > 1 and isinstance(acq_function, AnalyticAcquisitionFunction):
        raise ValueError(
            "optimize_acqf_continuous requires a q-batch acquisition function "
            f"when q > 1; got {type(acq_function).__name__}."
        )

    device = _infer_acqf_device(acq_function)
    dtype = bounds.dtype
    bounds_local = bounds.to(device=device, dtype=dtype)
    X_discrete_local = X_discrete.to(device=device, dtype=dtype)

    if pre_batch is not None:
        if pre_batch.ndim != 2 or pre_batch.shape[1] != d_total:
            raise ValueError(
                "pre_batch must have shape (q_pre, d_total) with d_total matching bounds"
            )
        q_pre = int(pre_batch.shape[0])
        if q_pre > q_total:
            raise ValueError("pre_batch cannot have more rows than X_discrete")
        pre_batch_local = pre_batch.to(device=device, dtype=dtype)
        if d_discrete > 0 and not torch.allclose(
            pre_batch_local[:, d_continuous:],
            X_discrete_local[:q_pre],
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("pre_batch discrete slice must match X_discrete")
    else:
        q_pre = 0
        pre_batch_local = None

    q_free = q_total - q_pre

    scalar_fixed_dims, list_fixed_dim = _normalize_greedy_fixed_dims(
        fixed_dims=fixed_dims,
        d_continuous=d_continuous,
        q_free=q_free,
    )

    if pre_batch_local is not None:
        for dim_idx, value in scalar_fixed_dims.items():
            target = torch.full((q_pre,), float(value), device=device, dtype=dtype)
            if not torch.allclose(
                pre_batch_local[:, dim_idx], target, atol=1e-6, rtol=0.0
            ):
                raise ValueError(
                    "pre_batch conflicts with scalar fixed dim " f"{dim_idx}={value}"
                )

    const_X = torch.zeros(q_total, d_total, device=device, dtype=dtype)
    if pre_batch_local is not None and q_pre > 0:
        const_X[:q_pre] = pre_batch_local
    if d_discrete > 0:
        const_X[:, d_continuous:] = X_discrete_local
    for dim_idx, value in scalar_fixed_dims.items():
        const_X[q_pre:, dim_idx] = float(value)
    if list_fixed_dim is not None:
        list_dim, list_vals = list_fixed_dim
        for s, value in enumerate(list_vals):
            if s >= q_free:
                break
            const_X[q_pre + s, list_dim] = float(value)

    _grouped_counts, explicit_partitions_all = _normalize_batch_constraints(
        batch_constraints, q_expected=q_total
    )
    free_dims, grouped_dims, _ = _build_free_dim_info(
        d_total, batch_constraints=batch_constraints, fixed_dims=scalar_fixed_dims
    )
    for dim_idx in grouped_dims:
        if dim_idx >= d_continuous:
            raise ValueError(
                "batch_constraints supports continuous dims only: "
                f"got dim {dim_idx}, d_continuous={d_continuous}"
            )

    free_cont_dims = [i for i in free_dims if i < d_continuous]
    grouped_cont_dims = {
        dim: k for dim, k in grouped_dims.items() if dim < d_continuous
    }

    per_dim_groupings: list[list[tuple[tuple[int, ...], list[list[int]]]]] = []
    grouped_dim_keys: list[int] = []
    for dim_idx in sorted(grouped_cont_dims.keys()):
        if dim_idx in explicit_partitions_all:
            partitions = [explicit_partitions_all[dim_idx]]
        else:
            partitions = []
            for k in range(1, grouped_cont_dims[dim_idx] + 1):
                partitions.extend(_integer_partitions(q_total, k))
        dim_groupings: list[tuple[tuple[int, ...], list[list[int]]]] = []
        for partition in partitions:
            groupings = _enumerate_partition_groupings(q_total, partition)
            for grouping in groupings:
                dim_groupings.append((partition, grouping))
        per_dim_groupings.append(dim_groupings)
        grouped_dim_keys.append(dim_idx)

    total_groupings = 1
    for options in per_dim_groupings:
        total_groupings *= len(options)
    if total_groupings > 10 and progress_callback is not None:
        progress_callback(
            f"optimize_acqf_continuous: evaluating {total_groupings} grouping assignments"
        )

    def _resolve_fixed_value(values: list[float]) -> float | None:
        if len(values) == 0:
            return None
        ref = values[0]
        ref_t = torch.tensor(ref, device=device, dtype=dtype)
        for v in values[1:]:
            v_t = torch.tensor(v, device=device, dtype=dtype)
            if not torch.isclose(v_t, ref_t, atol=1e-6, rtol=0.0).item():
                return None
        return ref

    def _optimize_for_grouping(
        grouping_map: dict[int, list[list[int]]],
        grouping_seed: int,
    ) -> tuple[Tensor, Tensor] | None:
        const_local = const_X.clone()
        param_slots: list[list[tuple[int, int]]] = []
        param_lows: list[float] = []
        param_highs: list[float] = []

        for s in range(q_free):
            full_slot = q_pre + s
            for dim_idx in free_cont_dims:
                if list_fixed_dim is not None:
                    list_dim, list_vals = list_fixed_dim
                    if dim_idx == list_dim and s < len(list_vals):
                        continue
                param_slots.append([(full_slot, dim_idx)])
                param_lows.append(bounds_local[0, dim_idx].item())
                param_highs.append(bounds_local[1, dim_idx].item())

        for dim_idx in sorted(grouped_cont_dims.keys()):
            for group in grouping_map[dim_idx]:
                fixed_vals: list[float] = []
                if pre_batch_local is not None:
                    for idx in group:
                        if idx < q_pre:
                            fixed_vals.append(
                                float(pre_batch_local[idx, dim_idx].item())
                            )
                if list_fixed_dim is not None and list_fixed_dim[0] == dim_idx:
                    list_vals = list_fixed_dim[1]
                    for idx in group:
                        if idx >= q_pre:
                            local_idx = idx - q_pre
                            if local_idx < len(list_vals):
                                fixed_vals.append(float(list_vals[local_idx]))

                fixed_value = _resolve_fixed_value(fixed_vals)
                if fixed_vals and fixed_value is None:
                    return None

                if fixed_value is not None:
                    for idx in group:
                        if idx >= q_pre:
                            const_local[idx, dim_idx] = fixed_value
                    continue

                free_slots = [idx for idx in group if idx >= q_pre]
                if len(free_slots) == 0:
                    continue
                param_slots.append([(idx, dim_idx) for idx in free_slots])
                param_lows.append(bounds_local[0, dim_idx].item())
                param_highs.append(bounds_local[1, dim_idx].item())

        n_params = len(param_slots)
        if n_params == 0:
            with torch.no_grad():
                val = acq_function(const_local.unsqueeze(0)).reshape(-1).sum()
            return const_local, val.reshape(())

        pair_param_ids: list[int] = []
        pair_slots: list[int] = []
        pair_dims: list[int] = []
        for p_idx, slot_dim_list in enumerate(param_slots):
            for slot, dim_idx in slot_dim_list:
                pair_param_ids.append(p_idx)
                pair_slots.append(slot)
                pair_dims.append(dim_idx)

        pair_param_ids_t = torch.tensor(pair_param_ids, dtype=torch.long, device=device)
        pair_slots_t = torch.tensor(pair_slots, dtype=torch.long, device=device)
        pair_dims_t = torch.tensor(pair_dims, dtype=torch.long, device=device)

        param_lows_t = torch.tensor(param_lows, dtype=dtype, device=device)
        param_highs_t = torch.tensor(param_highs, dtype=dtype, device=device)

        def scatter_into_X(x_params_2d: Tensor) -> Tensor:
            B = x_params_2d.shape[0]
            X_batch = const_local.unsqueeze(0).expand(B, -1, -1).clone()
            vals = x_params_2d[:, pair_param_ids_t]
            X_batch[:, pair_slots_t, pair_dims_t] = vals
            return X_batch

        from botorch.utils.sampling import draw_sobol_samples

        sobol_bounds = torch.stack([param_lows_t, param_highs_t], dim=0)
        X_rnd = draw_sobol_samples(
            bounds=sobol_bounds, n=raw_samples, q=1, seed=grouping_seed
        ).squeeze(1)

        with torch.no_grad():
            af_vals = acq_function(scatter_into_X(X_rnd)).reshape(-1)
        af_vals = torch.nan_to_num(af_vals, nan=-float("inf"))

        k_top = min(num_restarts, X_rnd.shape[0])
        _, top_idx = torch.topk(af_vals, k=k_top)
        starts = X_rnd[top_idx].detach().cpu().numpy()

        from scipy.optimize import minimize as scipy_minimize

        bounds_for_scipy = list(zip(param_lows, param_highs))

        def neg_af_with_grad(x_np):
            x = torch.tensor(x_np, dtype=dtype, device=device, requires_grad=True)
            X_full = scatter_into_X(x.unsqueeze(0))
            v = acq_function(X_full).reshape(-1).sum()
            grad = torch.autograd.grad(v, x)[0]
            return float(-v.item()), (-grad.detach().cpu().numpy()).astype(np.float64)

        best_neg_val = float("inf")
        best_x_np = None
        for x0 in starts:
            res = scipy_minimize(
                neg_af_with_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds_for_scipy,
            )
            if res.fun < best_neg_val:
                best_neg_val = float(res.fun)
                best_x_np = res.x

        best_x = torch.tensor(best_x_np, dtype=dtype, device=device)
        X_full = scatter_into_X(best_x.unsqueeze(0))
        with torch.no_grad():
            best_val = acq_function(X_full).reshape(-1).sum()
        return X_full[0], best_val.reshape(())

    best_candidates: Tensor | None = None
    best_acq_value: Tensor | None = None

    if per_dim_groupings:
        grouping_iter = itertools.product(*per_dim_groupings)
    else:
        grouping_iter = [()]

    for combo_idx, combo in enumerate(grouping_iter, start=1):
        if progress_callback is not None and total_groupings > 10:
            progress_callback(
                f"optimize_acqf_continuous grouping {combo_idx}/{total_groupings}"
            )
        grouping_map: dict[int, list[list[int]]] = {}
        for dim_idx, (_partition, grouping) in zip(grouped_dim_keys, combo):
            grouping_map[dim_idx] = grouping

        result = _optimize_for_grouping(grouping_map, grouping_seed=seed + combo_idx)
        if result is None:
            continue
        candidates, acq_value = result
        if best_acq_value is None or float(acq_value.item()) > float(
            best_acq_value.item()
        ):
            best_candidates = candidates
            best_acq_value = acq_value

    if best_candidates is None or best_acq_value is None:
        raise ValueError(
            "No feasible grouping assignments for optimize_acqf_continuous"
        )

    return best_candidates, best_acq_value


def parse_pymoo_mixed_variables(
    bounds: Tensor,
    discrete_points: list[list[float]],
    q: int = 1,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
):
    """Parse bounds into pymoo mixed variable definitions using choice variables.

    Creates a dictionary of pymoo variable objects for a mixed search space
    where discrete variables are represented as ``Choice`` variables with
    explicitly enumerated allowed values.  Supports grouped batch constraints
    (each constrained dim may allow ``k`` distinct values across the batch)
    and fixed dimensions.

    Args:
        bounds: A ``2 x d`` tensor of lower and upper bounds for each dimension.
        discrete_points: A list of lists, where ``discrete_points[i]`` contains
            the allowed values for discrete dimension ``i``.
        q: Number of candidates to optimize jointly. Defaults to 1.
        batch_constraints: Optional dictionary mapping dimension indices to:

            - integer ``k`` for legacy grouped constraints, or
            - explicit partition tuple/list for exact grouped constraints.
        fixed_dims: Optional dictionary mapping dimension indices to fixed
            values (these dimensions are excluded from GA optimization).

    Returns:
        A dictionary mapping 1-indexed variable keys to pymoo ``Real`` or
        ``Choice`` variable objects.
    """
    discrete_points = [[float(x) for x in inner_list] for inner_list in discrete_points]

    d = bounds.shape[1]
    d_discrete = len(discrete_points)
    d_continuous = d - d_discrete

    free_dims, grouped_dims, fixed_dims_dict = _build_free_dim_info(
        d, batch_constraints, fixed_dims
    )

    vars_dict = {}
    var_counter = 1

    # Per-candidate variables: free dims only
    for cand in range(q):
        for dim_idx in free_dims:
            if dim_idx < d_continuous:
                xu, xl = bounds[0, dim_idx].item(), bounds[1, dim_idx].item()
                vars_dict[var_counter] = Real(bounds=(xu, xl))
            else:
                disc_idx = dim_idx - d_continuous
                vars_dict[var_counter] = Choice(options=discrete_points[disc_idx])
            var_counter += 1

    # Grouped dims: k variables per dim
    for dim_idx in sorted(grouped_dims.keys()):
        k_val = grouped_dims[dim_idx]
        for _g in range(k_val):
            if dim_idx < d_continuous:
                xu, xl = bounds[0, dim_idx].item(), bounds[1, dim_idx].item()
                vars_dict[var_counter] = Real(bounds=(xu, xl))
            else:
                disc_idx = dim_idx - d_continuous
                vars_dict[var_counter] = Choice(options=discrete_points[disc_idx])
            var_counter += 1

    return vars_dict


class MixedAcqFunProblem(Problem):
    """Pymoo Problem wrapper for BoTorch acquisition function optimization.

    Wraps a BoTorch acquisition function as a pymoo ``Problem`` for use with
    ``MixedVariableGA``. Supports two modes:

    - **Choice mode** (``discrete_points``): Each discrete dimension is
      represented as a ``Choice`` variable with enumerated allowed values.
    - **Component mode** (``cat_emb_avail`` / ``sol_emb_avail`` / ``bas_emb_avail``):
      The discrete (catalyst, solvent, base) choice is encoded as three
      ``Integer`` variables per candidate, each indexing a row of the
      corresponding per-component embedding tensor. The discrete slice of
      the candidate is assembled by concatenation; when ``cat_is_salt_flags``
      marks the chosen catalyst as non-salt, the base slice is zeroed.

    Optionally supports:

    - **Batch constraints** (``batch_constraints``): grouped continuous dims
      with ``k`` distinct values across the ``q`` batch candidates.
    - **Fixed dimensions** (``fixed_dims``): Dimensions pinned to constants.
    - **Pre-batch** (``pre_batch``): Pre-selected candidates that are
      included in every AF evaluation.

    Args:
        acq_function: The BoTorch acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: A list of lists of allowed values per discrete
            dimension. Mutually exclusive with the component tensors.
        q: Total number of candidates in the batch (including pre-batch).
        cat_emb_avail: ``(n_cat, d_cat)`` available-catalyst embeddings.
        sol_emb_avail: ``(n_sol, d_sol)`` available-solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` available-base embeddings.
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor. When False at
            index ``c``, the base slice is zeroed for any candidate choosing
            catalyst ``c``. ``None`` disables the rule.
        batch_constraints: Per-dim grouped constraints.
        fixed_dims: Per-dim fixed values.
        pre_batch: Optional ``q_pre x d`` tensor of pre-selected candidates,
            prepended to the ``q - q_pre`` optimized candidates in every AF
            evaluation.
        partition_map: Optional per-grouped-dim partition tuple.

    Raises:
        ValueError: If neither component tensors nor ``discrete_points`` are
            provided, or both are provided simultaneously.
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        discrete_points: list[list[float]] | None = None,
        q: int = 1,
        cat_emb_avail: Tensor | None = None,
        sol_emb_avail: Tensor | None = None,
        bas_emb_avail: Tensor | None = None,
        cat_is_salt_flags: Tensor | None = None,
        batch_constraints: BatchConstraints | None = None,
        fixed_dims: dict[int, float] | None = None,
        pre_batch: Tensor | None = None,
        partition_map: dict[int, tuple[int, ...]] | None = None,
    ):
        component_tensors_given = (
            cat_emb_avail is not None
            or sol_emb_avail is not None
            or bas_emb_avail is not None
        )
        if component_tensors_given and not (
            cat_emb_avail is not None
            and sol_emb_avail is not None
            and bas_emb_avail is not None
        ):
            raise ValueError(
                "Component mode requires all three of cat_emb_avail, "
                "sol_emb_avail, and bas_emb_avail."
            )
        if discrete_points is not None and component_tensors_given:
            raise ValueError(
                "Cannot specify both discrete_points and component tensors."
            )
        if discrete_points is None and not component_tensors_given:
            raise ValueError(
                "Must specify either discrete_points or the component tensors "
                "(cat_emb_avail, sol_emb_avail, bas_emb_avail)."
            )

        self.acq_function = acq_function
        self.q_total = q
        self.pre_batch = pre_batch
        self.q_pre = pre_batch.shape[0] if pre_batch is not None else 0
        self.q = q - self.q_pre
        self.d = bounds.shape[1]
        self.bounds = bounds
        self.discrete_points = discrete_points
        self.cat_emb_avail = cat_emb_avail
        self.sol_emb_avail = sol_emb_avail
        self.bas_emb_avail = bas_emb_avail
        self.cat_is_salt_flags = cat_is_salt_flags
        self.use_component_mode = component_tensors_given
        self.batch_constraints = batch_constraints
        self.fixed_dims_arg = fixed_dims
        _, explicit_partitions = _normalize_batch_constraints(batch_constraints, self.q)

        self.free_dims, self.grouped_dims, self.fixed_dims_dict = _build_free_dim_info(
            self.d, batch_constraints, fixed_dims
        )

        if partition_map is not None:
            self.partition_map = partition_map
        else:
            self.partition_map = {dim: (self.q,) for dim in self.grouped_dims}
            self.partition_map.update(explicit_partitions)

        self.xl = bounds[0].detach().cpu().numpy()
        self.xu = bounds[1].detach().cpu().numpy()

        if self.use_component_mode:
            self.n_cat = cat_emb_avail.shape[0]
            self.n_sol = sol_emb_avail.shape[0]
            self.n_bas = bas_emb_avail.shape[0]
            self.d_cat = cat_emb_avail.shape[1]
            self.d_sol = sol_emb_avail.shape[1]
            self.d_bas = bas_emb_avail.shape[1]
            self.d_discrete = self.d_cat + self.d_sol + self.d_bas
            self.d_continuous = self.d - self.d_discrete
            self.vars = parse_pymoo_mixed_variables_with_components(
                bounds,
                n_cat=self.n_cat,
                n_sol=self.n_sol,
                n_bas=self.n_bas,
                d_discrete=self.d_discrete,
                q=q,
                batch_constraints=batch_constraints,
                fixed_dims=fixed_dims,
            )
        else:
            self.vars = parse_pymoo_mixed_variables(
                bounds,
                discrete_points,
                q=q,
                batch_constraints=batch_constraints,
                fixed_dims=fixed_dims,
            )
            self.d_discrete = len(discrete_points)
            self.d_continuous = self.d - self.d_discrete

        n_var = len(self.vars)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0)

    def _evaluate(self, x, out):
        # x is a list of dicts from MixedVariableGA with 1-indexed keys.
        n = len(x)
        candidates_list = []
        for i in range(n):
            cand = _ga_vector_to_candidates(
                flat_vals=x[i],
                q=self.q,
                d=self.d,
                free_dims=self.free_dims,
                grouped_dims=self.grouped_dims,
                fixed_dims=self.fixed_dims_dict,
                partition_map=self.partition_map,
                cat_emb_avail=self.cat_emb_avail,
                sol_emb_avail=self.sol_emb_avail,
                bas_emb_avail=self.bas_emb_avail,
                cat_is_salt_flags=self.cat_is_salt_flags,
            )  # (q_free, d)
            if self.pre_batch is not None:
                cand = torch.cat([self.pre_batch, cand], dim=0)  # (q_total, d)
            candidates_list.append(cand)

        x_tensor = torch.stack(candidates_list, dim=0)  # (N, q_total, d)
        target_device = _infer_acqf_device(self.acq_function)
        if x_tensor.device != target_device:
            x_tensor = x_tensor.to(target_device)

        with torch.no_grad():
            f = self.acq_function(x_tensor).detach().cpu().numpy()
        out["F"] = -f  # Negate for minimization


class CanonicalMixedAcqFunProblem(MixedAcqFunProblem):
    r"""Symmetry-aware, vectorized pymoo problem for batch acquisition optimization.

    Differs from :class:`MixedAcqFunProblem` in two ways, both opt-in through
    the canonical optimizers and neither reachable from the existing ones:

    - the whole population is decoded to a ``(N, q_total, d)`` tensor with
      batched indexing instead of a per-individual Python loop. Measured at
      ``pop_size = 150`` and 20k evaluations this drops decoding from 30% of GA
      wall-clock to 1%;
    - acquisition values may be cached on the canonical genome.

    The variable encoding is unchanged from :class:`MixedAcqFunProblem`; the
    discrete indices are corrected at the operator level instead, in
    :func:`_run_single_canonical_ga`.

    Caching is off by default. Measured on this search space the hit rate is
    effectively zero: each genome carries ``3 * q + n_blocks`` free float64
    coordinates, so two individuals are never bitwise symmetry-equivalent.
    ``cache_decimals`` rounds the key and trades exactness for hits.

    Args:
        canonicalize: Whether to build the symmetry layout. The layout drives
            :class:`CanonicalRepair` and the cache key.
        cache_acq_values: Whether to memoize acquisition values on the genome.
        cache_decimals: Optional rounding applied to cache keys. ``None`` keys
            on exact values; an integer makes the cache approximate.
        *args: Forwarded to :class:`MixedAcqFunProblem`.
        **kwargs: Forwarded to :class:`MixedAcqFunProblem`.
    """

    def __init__(
        self,
        *args,
        canonicalize: bool = True,
        cache_acq_values: bool = False,
        cache_decimals: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.layout = _build_canonical_layout(
            q=self.q,
            free_dims=self.free_dims,
            grouped_dims=self.grouped_dims,
            partition_map=self.partition_map,
            d_continuous=self.d_continuous,
            use_component_mode=self.use_component_mode,
        )
        self.canonicalize = canonicalize
        self.cache_decimals = cache_decimals
        self.acq_cache: dict[tuple, float] | None = {} if cache_acq_values else None
        self.cache_hits = 0
        self.cache_lookups = 0

        self._var_keys = sorted(self.vars.keys())
        self._n_var_per_cand = self.layout.n_var_per_cand
        self._grouped_order = [
            dim
            for dim in sorted(self.grouped_dims.keys())
            if (not self.use_component_mode) or dim < self.d_continuous
        ]
        self._grouped_starts = self.layout.grouped_starts
        self._grouped_ks = self.layout.grouped_ks
        self._group_maps = [
            torch.as_tensor(_candidate_group_map(self.partition_map[dim]))
            for dim in self._grouped_order
        ]

        if self.use_component_mode:
            free_cont_dims = [i for i in self.free_dims if i < self.d_continuous]
            self._n_free_cont = len(free_cont_dims)
            self._free_idx = torch.as_tensor(free_cont_dims, dtype=torch.long)
            self._cat_start = self.d_continuous
            self._sol_start = self._cat_start + self.d_cat
            self._bas_start = self._sol_start + self.d_sol
            self._bas_end = self._bas_start + self.d_bas
            self._salt_mask = (
                self.cat_is_salt_flags.to(torch.float64)
                if self.cat_is_salt_flags is not None
                else None
            )
            self._salt_flags_np = (
                self.cat_is_salt_flags.detach().cpu().numpy().astype(bool)
                if self.cat_is_salt_flags is not None
                else None
            )
        else:
            self._n_free_cont = 0
            self._free_idx = torch.as_tensor(self.free_dims, dtype=torch.long)
            self._salt_mask = None
            self._salt_flags_np = None

    @property
    def salt_flags(self) -> np.ndarray | None:
        """``(n_cat,)`` bool array of salt flags, or ``None`` outside component mode."""
        return self._salt_flags_np

    def decode_population(self, x: list[dict]) -> Tensor:
        r"""Decode a population of genomes into candidate tensors.

        Args:
            x: List of ``N`` 1-indexed genome dicts.

        Returns:
            A ``(N, q_total, d)`` tensor, pre-batch candidates first.
        """
        n = len(x)
        keys = self._var_keys
        raw = np.empty((n, len(keys)), dtype=np.float64)
        for i, flat_vals in enumerate(x):
            row = raw[i]
            for j, key in enumerate(keys):
                row[j] = flat_vals[key]

        nvc = self._n_var_per_cand
        per_cand = raw[:, : self.q * nvc].reshape(n, self.q, nvc)
        grouped = raw[:, self.q * nvc :]

        candidates = torch.zeros(n, self.q, self.d, dtype=torch.float64)

        if self.use_component_mode:
            if self._n_free_cont > 0:
                candidates[:, :, self._free_idx] = torch.from_numpy(
                    np.ascontiguousarray(per_cand[:, :, : self._n_free_cont])
                )
            idx = per_cand[:, :, self._n_free_cont : self._n_free_cont + 3].astype(
                np.int64
            )
            cat_idx = torch.from_numpy(np.ascontiguousarray(idx[:, :, 0]))
            sol_idx = torch.from_numpy(np.ascontiguousarray(idx[:, :, 1]))
            bas_idx = torch.from_numpy(np.ascontiguousarray(idx[:, :, 2]))
            candidates[:, :, self._cat_start : self._sol_start] = self.cat_emb_avail[
                cat_idx
            ]
            candidates[:, :, self._sol_start : self._bas_start] = self.sol_emb_avail[
                sol_idx
            ]
            base = self.bas_emb_avail[bas_idx]
            if self._salt_mask is not None:
                base = base * self._salt_mask[cat_idx].unsqueeze(-1)
            candidates[:, :, self._bas_start : self._bas_end] = base
        elif self._free_idx.numel() > 0:
            candidates[:, :, self._free_idx] = torch.from_numpy(
                np.ascontiguousarray(per_cand)
            )

        for r, dim_idx in enumerate(self._grouped_order):
            start = self._grouped_starts[r]
            block = torch.from_numpy(
                np.ascontiguousarray(grouped[:, start : start + self._grouped_ks[r]])
            )
            candidates[:, :, dim_idx] = block[:, self._group_maps[r]]

        for dim_idx, value in self.fixed_dims_dict.items():
            candidates[:, :, dim_idx] = value

        if self.pre_batch is not None:
            candidates = torch.cat(
                [self.pre_batch.unsqueeze(0).expand(n, -1, -1), candidates], dim=1
            )
        return candidates

    def _cache_keys(self, x: list[dict]) -> list[tuple]:
        """Cache keys for a population, canonical because repair ran first."""
        keys = self._var_keys
        if self.cache_decimals is None:
            return [tuple(flat_vals[k] for k in keys) for flat_vals in x]
        nd = self.cache_decimals
        return [tuple(round(float(flat_vals[k]), nd) for k in keys) for flat_vals in x]

    def _evaluate(self, x, out):
        target_device = _infer_acqf_device(self.acq_function)

        if self.acq_cache is None:
            x_tensor = self.decode_population(x).to(target_device)
            with torch.no_grad():
                f = self.acq_function(x_tensor).detach().cpu().numpy()
            out["F"] = -f
            return

        cache_keys = self._cache_keys(x)
        values = np.empty(len(x), dtype=np.float64)
        misses = []
        for i, key in enumerate(cache_keys):
            hit = self.acq_cache.get(key)
            if hit is None:
                misses.append(i)
            else:
                values[i] = hit
        self.cache_lookups += len(x)
        self.cache_hits += len(x) - len(misses)

        if misses:
            x_tensor = self.decode_population([x[i] for i in misses]).to(target_device)
            with torch.no_grad():
                f = self.acq_function(x_tensor).detach().cpu().numpy()
            for j, i in enumerate(misses):
                values[i] = f[j]
                self.acq_cache[cache_keys[i]] = f[j]

        out["F"] = -values


def _run_single_ga(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_points: list[list[float]] | None,
    q: int,
    q_free: int,
    n_evals_ga: int,
    pop_size: int,
    cat_emb_avail: Tensor | None,
    sol_emb_avail: Tensor | None,
    bas_emb_avail: Tensor | None,
    cat_is_salt_flags: Tensor | None,
    batch_constraints: BatchConstraints | None,
    fixed_dims: dict[int, float] | None,
    pre_batch: Tensor | None,
    partition_map: dict[int, tuple[int, ...]] | None,
    seed: int = 1,
) -> tuple[Tensor, Tensor]:
    """Execute a single GA run for one specific partition assignment.

    Args:
        acq_function: The acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: Discrete variable allowed values (choice mode).
        q: Total batch size including pre-batch.
        q_free: Number of candidates the GA actually optimises.
        n_evals_ga: Total GA function evaluations.
        pop_size: GA population size.
        cat_emb_avail: ``(n_cat, d_cat)`` catalyst embeddings (component mode).
        sol_emb_avail: ``(n_sol, d_sol)`` solvent embeddings (component mode).
        bas_emb_avail: ``(n_bas, d_bas)`` base embeddings (component mode).
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor for the salt rule.
        batch_constraints: Grouped batch constraint dictionary.
        fixed_dims: Fixed dimension dictionary.
        pre_batch: Optional ``q_pre x d`` tensor of pre-selected candidates,
            or ``None``.
        partition_map: Partition assignment for each grouped dim.
        seed: Seed forwarded to pymoo's ``minimize``. ``minimize`` resets
            NumPy's global RNG to this value at the start of the run, so it
            fully controls the GA trajectory. Defaults to 1.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): A ``q x d`` tensor, the ``pre_batch`` rows
          followed by the ``q_free`` optimized candidates.
        - **acq_value** (*Tensor*): The joint batch acquisition value from
          ``_compute_batch_acq_value`` (a single element).
    """
    problem = MixedAcqFunProblem(
        acq_function,
        bounds,
        discrete_points,
        q=q,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        pre_batch=pre_batch,
        partition_map=partition_map,
    )

    if problem.use_component_mode:
        sampling = MixedAllowedSampling(1, problem.d_continuous, bounds)
        crossover = MixedAllowedCrossover(1)
        mutation = MixedAllowedMutation(1, prob=0.1)
        algorithm = MixedVariableGA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    else:
        algorithm = MixedVariableGA(pop_size=pop_size)

    res = minimize(
        problem, algorithm, termination=("n_evals", n_evals_ga), seed=seed, verbose=False
    )

    candidates = _ga_vector_to_candidates(
        flat_vals=res.X,
        q=q_free,
        d=bounds.shape[1],
        free_dims=problem.free_dims,
        grouped_dims=problem.grouped_dims,
        fixed_dims=problem.fixed_dims_dict,
        partition_map=problem.partition_map,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
    )

    if pre_batch is not None:
        candidates = torch.cat([pre_batch, candidates], dim=0)

    acq_value = _compute_batch_acq_value(acq_function, candidates, q)
    return candidates, acq_value


def optimize_acqf_mixed_GA(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_points: list[list[float]] | None = None,
    q: int = 1,
    n_evals_ga: int = 20000,
    pop_size: int = 150,
    cat_emb_avail: Tensor | None = None,
    sol_emb_avail: Tensor | None = None,
    bas_emb_avail: Tensor | None = None,
    cat_is_salt_flags: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    pre_batch: Tensor | None = None,
    progress_callback: Callable[[str], None] | None = None,
    seed: int = 1,
) -> tuple[Tensor, Tensor]:
    r"""Optimize a mixed acquisition function using a genetic algorithm (GA).

    Uses pymoo's ``MixedVariableGA`` to optimize over the joint space of
    continuous and discrete variables. Supports two discrete representations:

    - **Choice mode**: Each discrete dimension has explicitly enumerated
      allowed values (via ``discrete_points``).
    - **Component mode**: The discrete part is encoded as three integer
      indices ``(cat_idx, sol_idx, bas_idx)`` per candidate, indexing into
      ``cat_emb_avail`` / ``sol_emb_avail`` / ``bas_emb_avail``. The base
      embedding slice is zeroed when ``cat_is_salt_flags[cat_idx]`` is False.

    ``batch_constraints`` supports two forms per constrained dimension:

    - integer ``k`` (legacy mode): enumerate all unordered partitions of ``q``
      into ``1..k`` groups and return the best partition;
    - explicit partition tuple/list (exact mode): evaluate only that exact
      partition (for example ``(2, 2, 2)``).

    For multiple constrained dimensions, the cross-product of per-dimension
    partition options is evaluated.

    When ``q > 1``, a single joint batch acquisition value is returned
    (computed via ``qLogExpectedImprovement`` if the acquisition function
    does not natively support batches).

    Args:
        acq_function: The acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: A list of lists, where ``discrete_points[i]`` contains
            the allowed values for discrete dimension ``i``. Mutually exclusive
            with the component tensors.
        q: Total number of candidates in the batch (including pre-batch).
        n_evals_ga: Total number of function evaluations for the GA.
        pop_size: Population size for the GA.
        cat_emb_avail: ``(n_cat, d_cat)`` available-catalyst embeddings.
            When given (with ``sol_emb_avail`` and ``bas_emb_avail``), the GA
            runs in component mode.
        sol_emb_avail: ``(n_sol, d_sol)`` available-solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` available-base embeddings.
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor. When False at
            index ``c``, the base slice is zeroed for candidates choosing
            catalyst ``c``. ``None`` disables the salt rule.
        batch_constraints: Per-dim grouped constraints.
        fixed_dims: Per-dim fixed values.
        pre_batch: Optional ``(q', d)`` tensor of pre-selected candidates.
        progress_callback: Optional callback receiving short progress messages
            at partition start/end.
        seed: Seed forwarded to every internal pymoo ``minimize`` call (the
            same value is used for each partition combo). ``minimize`` resets
            NumPy's global RNG to this value, so it fully controls the GA
            trajectory. Defaults to 1.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): A ``q x d`` tensor of optimized candidates.
        - **acq_value** (*Tensor*): A scalar tensor with the joint batch acq value.
    """
    q_free = q - (pre_batch.shape[0] if pre_batch is not None else 0)

    _normalized_counts, explicit_partitions_all = _normalize_batch_constraints(
        batch_constraints, q_expected=q_free
    )

    # Build grouped_dims to know which dims need partition enumeration
    _, grouped_dims, _ = _build_free_dim_info(
        bounds.shape[1], batch_constraints, fixed_dims
    )

    # Enumerate partitions per grouped dim
    per_dim_partitions: list[list[tuple[int, ...]]] = []
    grouped_dim_keys: list[int] = []
    for dim_idx in sorted(grouped_dims.keys()):
        if dim_idx in explicit_partitions_all:
            per_dim_partitions.append([explicit_partitions_all[dim_idx]])
            grouped_dim_keys.append(dim_idx)
            continue

        k_val = grouped_dims[dim_idx]
        partitions = []
        for i in range(1, k_val + 1):
            partitions.extend(_integer_partitions(q_free, i))
        per_dim_partitions.append(partitions)
        grouped_dim_keys.append(dim_idx)

    # Cross-product of all per-dim partitions
    if per_dim_partitions:
        partition_combos = list(itertools.product(*per_dim_partitions))
    else:
        partition_combos = [()]

    best_candidates = None
    best_acq_value = None

    total_partition_combos = len(partition_combos)

    if total_partition_combos > 10:
        warnings.warn(
            f"Optimizing {total_partition_combos} partition combinations. This might take a while.",
            UserWarning,
            stacklevel=2,
        )

    for combo_idx, combo in enumerate(partition_combos, start=1):
        partition_map = {grouped_dim_keys[i]: combo[i] for i in range(len(combo))}

        if progress_callback is not None:
            progress_callback(f"partition {combo_idx}/{total_partition_combos} started")
        combo_start = time.perf_counter()

        candidates, acq_value = _run_single_ga(
            acq_function=acq_function,
            bounds=bounds,
            discrete_points=discrete_points,
            q=q,
            q_free=q_free,
            n_evals_ga=n_evals_ga,
            pop_size=pop_size,
            cat_emb_avail=cat_emb_avail,
            sol_emb_avail=sol_emb_avail,
            bas_emb_avail=bas_emb_avail,
            cat_is_salt_flags=cat_is_salt_flags,
            batch_constraints=batch_constraints,
            fixed_dims=fixed_dims,
            pre_batch=pre_batch,
            partition_map=partition_map,
            seed=seed,
        )

        if progress_callback is not None:
            combo_elapsed = time.perf_counter() - combo_start
            progress_callback(
                f"partition {combo_idx}/{total_partition_combos} finished in {combo_elapsed:.1f}s"
            )

        if best_acq_value is None or acq_value.item() > best_acq_value.item():
            best_candidates = candidates
            best_acq_value = acq_value

    return best_candidates, best_acq_value


def _partition_combos(
    q_free: int,
    bounds: Tensor,
    batch_constraints: BatchConstraints | None,
    fixed_dims: dict[int, float] | None,
) -> tuple[list[int], list[tuple[tuple[int, ...], ...]]]:
    """Enumerate the partition assignments a batch GA must try.

    Explicit partitions contribute exactly themselves; an integer ``k``
    contributes every unordered partition of ``q_free`` into ``1..k`` parts.
    Multiple grouped dims contribute their cross-product.

    Args:
        q_free: Number of GA-controlled candidates.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        batch_constraints: Per-dim grouped constraints.
        fixed_dims: Per-dim fixed values.

    Returns:
        A two-element tuple of the grouped dim indices and the list of
        per-dim partition combinations.
    """
    _, explicit_partitions_all = _normalize_batch_constraints(
        batch_constraints, q_expected=q_free
    )
    _, grouped_dims, _ = _build_free_dim_info(
        bounds.shape[1], batch_constraints, fixed_dims
    )

    per_dim_partitions: list[list[tuple[int, ...]]] = []
    grouped_dim_keys: list[int] = []
    for dim_idx in sorted(grouped_dims.keys()):
        if dim_idx in explicit_partitions_all:
            per_dim_partitions.append([explicit_partitions_all[dim_idx]])
        else:
            partitions: list[tuple[int, ...]] = []
            for i in range(1, grouped_dims[dim_idx] + 1):
                partitions.extend(_integer_partitions(q_free, i))
            per_dim_partitions.append(partitions)
        grouped_dim_keys.append(dim_idx)

    if per_dim_partitions:
        return grouped_dim_keys, list(itertools.product(*per_dim_partitions))
    return grouped_dim_keys, [()]


def _build_canonical_mating(
    repair: Repair | None,
    discrete_mutation_prob: float,
) -> MixedVariableMating:
    """Build a mating operator that treats discrete indices as unordered.

    Pymoo maps ``Integer`` and ``Choice`` variables to SBX crossover and
    polynomial mutation by default, which impose an ordered metric on the
    catalyst, solvent and base identifiers — SBX blends index 3 and index 19
    into index 11. Both are replaced here with uniform crossover, which is
    elementwise and therefore value-preserving, and random-reset mutation, which
    resamples from the variable's own domain. ``Real`` variables keep pymoo's
    defaults.

    A fresh :class:`MixedVariableMating` is constructed on every call:
    ``MixedVariableGA`` carries one as a mutable default argument shared across
    every instance, so mutating that would leak into unrelated runs.

    Args:
        repair: Repair applied to offspring before duplicate elimination, or
            ``None``.
        discrete_mutation_prob: Per-gene random-reset probability for the
            discrete variables.

    Returns:
        The configured mating operator.
    """
    mating = MixedVariableMating(
        repair=repair,
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )
    for var_type in (Integer, Choice):
        mating.crossover[var_type] = UX()
        mating.mutation[var_type] = ChoiceRandomMutation(
            prob=1.0, prob_var=discrete_mutation_prob
        )
    return mating


def _run_single_canonical_ga(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_points: list[list[float]] | None,
    q: int,
    q_free: int,
    n_evals_ga: int,
    pop_size: int,
    cat_emb_avail: Tensor | None,
    sol_emb_avail: Tensor | None,
    bas_emb_avail: Tensor | None,
    cat_is_salt_flags: Tensor | None,
    batch_constraints: BatchConstraints | None,
    fixed_dims: dict[int, float] | None,
    pre_batch: Tensor | None,
    partition_map: dict[int, tuple[int, ...]] | None,
    seed: int = 1,
    canonicalize: bool = True,
    discrete_mutation_prob: float = 0.05,
    cache_acq_values: bool = False,
    cache_decimals: int | None = None,
    stats: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Execute one canonical GA run for a single partition assignment.

    The counterpart of :func:`_run_single_ga`, differing in three ways: the
    discrete indices are bred with uniform crossover and random-reset mutation,
    the population is decoded in one batched pass, and — when ``canonicalize``
    is set — every individual is projected onto its canonical labeling before
    duplicate elimination.

    The operator override is the correction that matters. Catalyst, solvent and
    base identifiers are unordered categoricals, but pymoo maps ``Integer``
    variables to SBX crossover and polynomial mutation, which treat them as an
    ordered metric axis and blend catalyst 3 with catalyst 19 into catalyst 11.
    ``UX`` is elementwise and ``ChoiceRandomMutation`` resamples from the
    variable's own domain, which is the correct treatment.

    Encoding the indices as ``Choice`` instead would reach the same operators
    through pymoo's defaults, but ``MixedVariableMating`` gives every ``Choice``
    its own single-variable sub-problem — 18 operator invocations per generation
    at ``q = 6`` instead of one — which measured 1.38x slower at identical
    solution quality. Hence ``Integer`` variables with overridden operators.

    Args:
        acq_function: The acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: Discrete variable allowed values (choice mode).
        q: Total batch size including pre-batch.
        q_free: Number of candidates the GA actually optimises.
        n_evals_ga: Total GA function evaluations.
        pop_size: GA population size.
        cat_emb_avail: ``(n_cat, d_cat)`` catalyst embeddings (component mode).
        sol_emb_avail: ``(n_sol, d_sol)`` solvent embeddings (component mode).
        bas_emb_avail: ``(n_bas, d_bas)`` base embeddings (component mode).
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor for the salt rule.
        batch_constraints: Grouped batch constraint dictionary.
        fixed_dims: Fixed dimension dictionary.
        pre_batch: Optional ``q_pre x d`` tensor of pre-selected candidates,
            or ``None``.
        partition_map: Partition assignment for each grouped dim.
        seed: Seed forwarded to pymoo's ``minimize``.
        canonicalize: Whether to apply :class:`CanonicalRepair`.
        discrete_mutation_prob: Per-gene random-reset probability for the
            discrete variables. Defaults to 0.05; at 20k evaluations 0.02 and
            0.05 match :func:`optimize_acqf_mixed_GA`, while 0.1 and 0.2 find
            worse batches.
        cache_acq_values: Whether to memoize acquisition values.
        cache_decimals: Optional rounding of cache keys.
        stats: Optional dict, filled with GA diagnostics.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): A ``q x d`` tensor, the ``pre_batch`` rows
          followed by the ``q_free`` optimized candidates.
        - **acq_value** (*Tensor*): The joint batch acquisition value from
          ``_compute_batch_acq_value`` (a single element).
    """
    problem = CanonicalMixedAcqFunProblem(
        acq_function,
        bounds,
        discrete_points,
        q=q,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        pre_batch=pre_batch,
        partition_map=partition_map,
        canonicalize=canonicalize,
        cache_acq_values=cache_acq_values,
        cache_decimals=cache_decimals,
    )

    repair = (
        CanonicalRepair(problem.layout, problem.salt_flags) if canonicalize else None
    )
    mating = _build_canonical_mating(repair, discrete_mutation_prob)
    algorithm = MixedVariableGA(
        pop_size=pop_size,
        repair=repair,
        mating=mating,
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    res = minimize(
        problem, algorithm, termination=("n_evals", n_evals_ga), seed=seed, verbose=False
    )

    best = dict(res.X)
    if canonicalize:
        problem.layout.canonicalize(best, problem.salt_flags)
    candidates = problem.decode_population([best])[0]
    acq_value = _compute_batch_acq_value(acq_function, candidates, q)

    if stats is not None:
        stats["symmetry_group_order"] = problem.layout.group_order
        stats["cache_lookups"] = problem.cache_lookups
        stats["cache_hits"] = problem.cache_hits
        stats["cache_hit_rate"] = (
            problem.cache_hits / problem.cache_lookups if problem.cache_lookups else 0.0
        )
    return candidates, acq_value


def optimize_acqf_mixed_canonical_GA(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_points: list[list[float]] | None = None,
    q: int = 1,
    n_evals_ga: int = 20000,
    pop_size: int = 150,
    cat_emb_avail: Tensor | None = None,
    sol_emb_avail: Tensor | None = None,
    bas_emb_avail: Tensor | None = None,
    cat_is_salt_flags: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    pre_batch: Tensor | None = None,
    progress_callback: Callable[[str], None] | None = None,
    seed: int = 1,
    canonicalize: bool = True,
    discrete_mutation_prob: float = 0.05,
    cache_acq_values: bool = False,
    cache_decimals: int | None = None,
    stats: dict | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Symmetry-aware drop-in replacement for :func:`optimize_acqf_mixed_GA`.

    Same signature, same return contract, same partition enumeration. Three
    things differ inside:

    1. **Unordered categoricals.** Catalyst, solvent and base indices are
       ``Choice`` variables bred with uniform crossover and random-reset
       mutation. :func:`optimize_acqf_mixed_GA` encodes them as ``Integer`` and
       pymoo breeds them with SBX and polynomial mutation, which imposes an
       order on identifiers that have none.
    2. **Symmetry canonicalization.** The joint acquisition value is invariant
       under relabelings of the batch that preserve the grouped-dimension
       assignment — order 48 for ``q = 6`` under ``(2, 2, 2)``, ``q!`` with no
       batch constraint. Every individual is projected onto a canonical
       labeling before duplicate elimination, so the GA searches the quotient
       space. The block structure is read from the partition, not passed in.
    3. **Batched decoding.** The population is decoded in one vectorized pass.

    Args:
        acq_function: The acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: Allowed values per discrete dim (choice mode).
        q: Total number of candidates in the batch (including pre-batch).
        n_evals_ga: Total number of function evaluations for the GA.
        pop_size: Population size for the GA.
        cat_emb_avail: ``(n_cat, d_cat)`` available-catalyst embeddings.
        sol_emb_avail: ``(n_sol, d_sol)`` available-solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` available-base embeddings.
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor driving base
            zeroing and base-index collapse.
        batch_constraints: Per-dim grouped constraints.
        fixed_dims: Per-dim fixed values.
        pre_batch: Optional ``(q', d)`` tensor of pre-selected candidates.
        progress_callback: Optional callback receiving progress messages.
        seed: Seed forwarded to every internal pymoo ``minimize`` call.
        canonicalize: Whether to canonicalize. ``False`` keeps the corrected
            operators but searches the full labeling space, which is the
            control arm for measuring what the symmetry is worth.
        discrete_mutation_prob: Per-gene random-reset probability for the
            discrete variables.
        cache_acq_values: Whether to memoize acquisition values on the
            canonical genome. Off by default; measured hit rate on this search
            space is effectively zero.
        cache_decimals: Optional rounding of cache keys, which makes the cache
            approximate in exchange for hits.
        stats: Optional dict, filled with diagnostics from the winning
            partition.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): A ``q x d`` tensor of optimized candidates.
        - **acq_value** (*Tensor*): A scalar tensor with the joint batch acq value.
    """
    q_free = q - (pre_batch.shape[0] if pre_batch is not None else 0)
    grouped_dim_keys, partition_combos = _partition_combos(
        q_free, bounds, batch_constraints, fixed_dims
    )

    best_candidates = None
    best_acq_value = None
    total_partition_combos = len(partition_combos)

    if total_partition_combos > 10:
        warnings.warn(
            f"Optimizing {total_partition_combos} partition combinations. This might take a while.",
            UserWarning,
            stacklevel=2,
        )

    for combo_idx, combo in enumerate(partition_combos, start=1):
        partition_map = {grouped_dim_keys[i]: combo[i] for i in range(len(combo))}

        if progress_callback is not None:
            progress_callback(f"partition {combo_idx}/{total_partition_combos} started")
        combo_start = time.perf_counter()

        combo_stats: dict = {}
        candidates, acq_value = _run_single_canonical_ga(
            acq_function=acq_function,
            bounds=bounds,
            discrete_points=discrete_points,
            q=q,
            q_free=q_free,
            n_evals_ga=n_evals_ga,
            pop_size=pop_size,
            cat_emb_avail=cat_emb_avail,
            sol_emb_avail=sol_emb_avail,
            bas_emb_avail=bas_emb_avail,
            cat_is_salt_flags=cat_is_salt_flags,
            batch_constraints=batch_constraints,
            fixed_dims=fixed_dims,
            pre_batch=pre_batch,
            partition_map=partition_map,
            seed=seed,
            canonicalize=canonicalize,
            discrete_mutation_prob=discrete_mutation_prob,
            cache_acq_values=cache_acq_values,
            cache_decimals=cache_decimals,
            stats=combo_stats,
        )

        if progress_callback is not None:
            combo_elapsed = time.perf_counter() - combo_start
            progress_callback(
                f"partition {combo_idx}/{total_partition_combos} finished in {combo_elapsed:.1f}s"
            )

        if best_acq_value is None or acq_value.item() > best_acq_value.item():
            best_candidates = candidates
            best_acq_value = acq_value
            if stats is not None:
                stats.clear()
                stats.update(combo_stats)
                stats["partition_map"] = partition_map

    return best_candidates, best_acq_value


def optimize_acqf_mixed_sequential_GA(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_points: list[list[float]] | None = None,
    q: int = 1,
    candidates_at_a_time: int = 1,
    n_evals_ga: int = 20000,
    pop_size: int = 150,
    cat_emb_avail: Tensor | None = None,
    sol_emb_avail: Tensor | None = None,
    bas_emb_avail: Tensor | None = None,
    cat_is_salt_flags: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    pre_batch: Tensor | None = None,
    num_restarts: int = 20,
    raw_samples: int = 1024,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Sequential batch builder: GA discrete selection + L-BFGS-B refinement.

    Builds a ``q``-batch in successive blocks of ``candidates_at_a_time``.  At
    each block:

    1. ``optimize_acqf_mixed_GA`` jointly picks the block's new discrete points
       (catalyst / solvent / base in component mode, or choice-mode discrete
       vectors), holding the already-selected candidates fixed as ``pre_batch``.
    2. ``optimize_acqf_continuous`` jointly re-optimises **all** continuous
       dimensions of **all** candidates selected so far.

    ``candidates_at_a_time`` interpolates the whole spectrum: ``1`` builds one
    candidate at a time (fully sequential); ``q`` is a single block (batch GA +
    one refine).  Blocks are full groups of ``candidates_at_a_time`` first with a
    single smaller remainder block last, e.g. ``q=7, c=3`` -> sizes ``(3, 3, 1)``.

    For explicit-partition batch constraints (e.g. temperature ``(2, 2, 2)``),
    the partial partition is computed from the cumulative candidate count after
    each block (via :func:`_build_step_batch_constraints`), so blocks remain
    unconstrained while the running total ``<= len(partition)`` and tighten
    toward the full partition as it grows — independent of the block size.

    Args:
        acq_function: The acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: Allowed values per discrete dim (choice mode).
            Mutually exclusive with the component tensors.
        q: Total batch size (including ``pre_batch``).
        candidates_at_a_time: GA block size (new candidates picked jointly per
            step). ``1`` = one-by-one sequential; ``>= q_free`` = single batch
            block. Must be ``>= 1``.
        n_evals_ga: GA function evaluations per step.
        pop_size: GA population size per step.
        cat_emb_avail: ``(n_cat, d_cat)`` catalyst embeddings (component mode).
        sol_emb_avail: ``(n_sol, d_sol)`` solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` base embeddings.
        cat_is_salt_flags: ``(n_cat,)`` bool tensor for the salt /
            base-zeroing rule.
        batch_constraints: Per-dim grouped constraints (int ``k`` or explicit
            partition tuple).
        fixed_dims: Per-dim fixed values (applied at both GA and refinement).
        pre_batch: ``(q_pre, d)`` fixed candidates included in every
            evaluation.  Their values are never modified.
        num_restarts: L-BFGS-B restarts for continuous refinement.
        raw_samples: Sobol samples for multi-start initialisation.
        progress_callback: Forwarded to sub-optimizers for fine-grained
            messages.

    Returns:
        A two-element tuple ``(candidates, acq_value)``:

        - **candidates**: ``(q, d)`` tensor of optimized candidates.
        - **acq_value**: scalar tensor of the final batch acquisition value.
    """
    q_pre = pre_batch.shape[0] if pre_batch is not None else 0
    q_free = q - q_pre

    if cat_emb_avail is not None:
        d_discrete = (
            cat_emb_avail.shape[1] + sol_emb_avail.shape[1] + bas_emb_avail.shape[1]
        )
    elif discrete_points is not None:
        d_discrete = len(discrete_points)
    else:
        d_discrete = 0
    d_continuous = bounds.shape[1] - d_discrete

    _, explicit_partitions = _normalize_batch_constraints(batch_constraints)

    current_batch = pre_batch
    acq_val = torch.tensor(float("-inf"))

    cumulative_ks = _block_cumulative_ks(q_free, candidates_at_a_time)
    n_blocks = len(cumulative_ks)

    for block_idx, cum_free in enumerate(cumulative_ks):
        k = q_pre + cum_free
        tag = f"[SeqGA blk {block_idx + 1}/{n_blocks} k={k}]"

        # ---- GA: pick the block's new discrete points ----
        print(f"{tag} GA: selecting discrete point(s)...", flush=True)
        ga_cands, ga_acq = optimize_acqf_mixed_GA(
            acq_function=acq_function,
            bounds=bounds,
            discrete_points=discrete_points,
            q=k,
            n_evals_ga=n_evals_ga,
            pop_size=pop_size,
            cat_emb_avail=cat_emb_avail,
            sol_emb_avail=sol_emb_avail,
            bas_emb_avail=bas_emb_avail,
            cat_is_salt_flags=cat_is_salt_flags,
            fixed_dims=fixed_dims,
            pre_batch=current_batch,
            progress_callback=progress_callback,
        )
        print(f"{tag} GA done (acq={ga_acq.item():.4f})", flush=True)

        # ---- Continuous refinement ----
        step_constraints = _build_step_batch_constraints(
            batch_constraints,
            explicit_partitions,
            k,
        )

        info_parts: list[str] = []
        if step_constraints:
            for dim_idx, spec in sorted(step_constraints.items()):
                if isinstance(spec, tuple):
                    n_grp = len(_enumerate_partition_groupings(k, spec))
                    info_parts.append(
                        f"dim{dim_idx}={spec}, {n_grp} groupings",
                    )
                else:
                    info_parts.append(f"dim{dim_idx}: k={spec}")
        info = ", ".join(info_parts) if info_parts else "no constraint"

        print(f"{tag} Refining continuous dims ({info})...", flush=True)
        X_disc = ga_cands[:, d_continuous:]
        refined, acq_val = optimize_acqf_continuous(
            acq_function=acq_function,
            bounds=bounds,
            X_discrete=X_disc,
            pre_batch=pre_batch,
            batch_constraints=step_constraints,
            fixed_dims=fixed_dims,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            progress_callback=progress_callback,
        )
        print(
            f"{tag} Refinement done (acq={acq_val.item():.4f})",
            flush=True,
        )
        current_batch = refined

    return current_batch, acq_val


def optimize_acqf_mixed_sequential_canonical_GA(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_points: list[list[float]] | None = None,
    q: int = 1,
    candidates_at_a_time: int = 1,
    n_evals_ga: int = 20000,
    pop_size: int = 150,
    cat_emb_avail: Tensor | None = None,
    sol_emb_avail: Tensor | None = None,
    bas_emb_avail: Tensor | None = None,
    cat_is_salt_flags: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    pre_batch: Tensor | None = None,
    num_restarts: int = 20,
    raw_samples: int = 1024,
    progress_callback: Callable[[str], None] | None = None,
    canonicalize: bool = True,
    discrete_mutation_prob: float = 0.05,
    cache_acq_values: bool = False,
    cache_decimals: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Canonical counterpart of :func:`optimize_acqf_mixed_sequential_GA`.

    Identical block schedule and continuous-refinement stage; the only change is
    that each block's discrete selection runs through
    :func:`optimize_acqf_mixed_canonical_GA`.

    The symmetry available per block is smaller than for a single joint GA: the
    already-selected candidates enter as ``pre_batch`` and are never relabeled,
    so only the ``candidates_at_a_time`` new slots are exchangeable. At
    ``candidates_at_a_time = 1`` there is no slot symmetry left at all and
    ``canonicalize`` only collapses the base index of non-salt catalysts.

    Args:
        acq_function: The acquisition function to optimize.
        bounds: A ``2 x d`` tensor of lower and upper bounds.
        discrete_points: Allowed values per discrete dim (choice mode).
        q: Total batch size (including ``pre_batch``).
        candidates_at_a_time: GA block size.
        n_evals_ga: GA function evaluations per step.
        pop_size: GA population size per step.
        cat_emb_avail: ``(n_cat, d_cat)`` catalyst embeddings (component mode).
        sol_emb_avail: ``(n_sol, d_sol)`` solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` base embeddings.
        cat_is_salt_flags: ``(n_cat,)`` bool tensor for the salt rule.
        batch_constraints: Per-dim grouped constraints.
        fixed_dims: Per-dim fixed values.
        pre_batch: ``(q_pre, d)`` fixed candidates, never modified.
        num_restarts: L-BFGS-B restarts for continuous refinement.
        raw_samples: Sobol samples for multi-start initialisation.
        progress_callback: Forwarded to sub-optimizers.
        canonicalize: Whether each block's GA canonicalizes its population.
        discrete_mutation_prob: Per-gene random-reset probability for the
            discrete variables.
        cache_acq_values: Whether each block's GA memoizes acquisition values.
        cache_decimals: Optional rounding of cache keys.

    Returns:
        A two-element tuple ``(candidates, acq_value)``:

        - **candidates**: ``(q, d)`` tensor of optimized candidates.
        - **acq_value**: scalar tensor of the final batch acquisition value.
    """
    q_pre = pre_batch.shape[0] if pre_batch is not None else 0
    q_free = q - q_pre

    if cat_emb_avail is not None:
        d_discrete = (
            cat_emb_avail.shape[1] + sol_emb_avail.shape[1] + bas_emb_avail.shape[1]
        )
    elif discrete_points is not None:
        d_discrete = len(discrete_points)
    else:
        d_discrete = 0
    d_continuous = bounds.shape[1] - d_discrete

    _, explicit_partitions = _normalize_batch_constraints(batch_constraints)

    current_batch = pre_batch
    acq_val = torch.tensor(float("-inf"))

    cumulative_ks = _block_cumulative_ks(q_free, candidates_at_a_time)
    n_blocks = len(cumulative_ks)

    for block_idx, cum_free in enumerate(cumulative_ks):
        k = q_pre + cum_free
        tag = f"[SeqCanonGA blk {block_idx + 1}/{n_blocks} k={k}]"

        print(f"{tag} GA: selecting discrete point(s)...", flush=True)
        ga_cands, ga_acq = optimize_acqf_mixed_canonical_GA(
            acq_function=acq_function,
            bounds=bounds,
            discrete_points=discrete_points,
            q=k,
            n_evals_ga=n_evals_ga,
            pop_size=pop_size,
            cat_emb_avail=cat_emb_avail,
            sol_emb_avail=sol_emb_avail,
            bas_emb_avail=bas_emb_avail,
            cat_is_salt_flags=cat_is_salt_flags,
            fixed_dims=fixed_dims,
            pre_batch=current_batch,
            progress_callback=progress_callback,
            canonicalize=canonicalize,
            discrete_mutation_prob=discrete_mutation_prob,
            cache_acq_values=cache_acq_values,
            cache_decimals=cache_decimals,
        )
        print(f"{tag} GA done (acq={ga_acq.item():.4f})", flush=True)

        step_constraints = _build_step_batch_constraints(
            batch_constraints,
            explicit_partitions,
            k,
        )

        info_parts: list[str] = []
        if step_constraints:
            for dim_idx, spec in sorted(step_constraints.items()):
                if isinstance(spec, tuple):
                    n_grp = len(_enumerate_partition_groupings(k, spec))
                    info_parts.append(
                        f"dim{dim_idx}={spec}, {n_grp} groupings",
                    )
                else:
                    info_parts.append(f"dim{dim_idx}: k={spec}")
        info = ", ".join(info_parts) if info_parts else "no constraint"

        print(f"{tag} Refining continuous dims ({info})...", flush=True)
        X_disc = ga_cands[:, d_continuous:]
        refined, acq_val = optimize_acqf_continuous(
            acq_function=acq_function,
            bounds=bounds,
            X_discrete=X_disc,
            pre_batch=pre_batch,
            batch_constraints=step_constraints,
            fixed_dims=fixed_dims,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            progress_callback=progress_callback,
        )
        print(
            f"{tag} Refinement done (acq={acq_val.item():.4f})",
            flush=True,
        )
        current_batch = refined

    return current_batch, acq_val



class _LookupAcquisitionFunction(AcquisitionFunction):
    r"""Index-space view of a component-mode acquisition function.

    Presents the search space as ``d_continuous + 3`` dimensions: the
    continuous block followed by three index dimensions selecting a row of
    the catalyst, solvent and base embedding tables. ``forward`` assembles
    the full ``d``-dimensional point (zeroing the base slice for non-salt
    catalysts), prepends the already-chosen ``pre_batch`` rows and evaluates
    the wrapped acquisition function on the joint batch.

    Gradients flow through the continuous block only; the index dimensions
    are held fixed by the caller during any gradient-based step.

    Args:
        base_acqf: Acquisition function (or any ``nn.Module`` mapping a
            ``(..., q, d)`` tensor to ``(...)``) over the full space.
        d_continuous: Number of leading continuous dimensions.
        cat_emb_avail: ``(n_cat, d_cat)`` catalyst embedding table.
        sol_emb_avail: ``(n_sol, d_sol)`` solvent embedding table.
        bas_emb_avail: ``(n_bas, d_bas)`` base embedding table.
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor; ``False``
            zeroes the base slice of candidates choosing that catalyst.
        pre_batch: Optional ``(q_pre, d)`` tensor of fixed candidates
            prepended to every evaluation.
    """

    def __init__(
        self,
        base_acqf: torch.nn.Module,
        d_continuous: int,
        cat_emb_avail: Tensor,
        sol_emb_avail: Tensor,
        bas_emb_avail: Tensor,
        cat_is_salt_flags: Tensor | None = None,
        pre_batch: Tensor | None = None,
    ) -> None:
        super().__init__(model=getattr(base_acqf, "model", None))
        self.base_acqf = base_acqf
        self.d_continuous = d_continuous
        self.cat_emb_avail = cat_emb_avail
        self.sol_emb_avail = sol_emb_avail
        self.bas_emb_avail = bas_emb_avail
        self.cat_is_salt_flags = cat_is_salt_flags
        self.pre_batch = pre_batch
        self.n_cat = cat_emb_avail.shape[0]
        self.n_sol = sol_emb_avail.shape[0]
        self.n_bas = bas_emb_avail.shape[0]
        self.d_full = (
            d_continuous
            + cat_emb_avail.shape[1]
            + sol_emb_avail.shape[1]
            + bas_emb_avail.shape[1]
        )

    def expand(self, X_idx: Tensor) -> Tensor:
        r"""Map index-space points to full-space points.

        Args:
            X_idx: ``(..., d_continuous + 3)`` tensor.

        Returns:
            ``(..., d)`` tensor in the full embedded space.
        """
        d_c = self.d_continuous
        x_cont = X_idx[..., :d_c]
        c_idx = X_idx[..., d_c].round().long().clamp(0, self.n_cat - 1)
        s_idx = X_idx[..., d_c + 1].round().long().clamp(0, self.n_sol - 1)
        b_idx = X_idx[..., d_c + 2].round().long().clamp(0, self.n_bas - 1)
        cat_rows = self.cat_emb_avail[c_idx]
        sol_rows = self.sol_emb_avail[s_idx]
        bas_rows = self.bas_emb_avail[b_idx]
        if self.cat_is_salt_flags is not None:
            keep = self.cat_is_salt_flags[c_idx].to(bas_rows.dtype).unsqueeze(-1)
            bas_rows = bas_rows * keep
        return torch.cat([x_cont, cat_rows, sol_rows, bas_rows], dim=-1)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the wrapped acquisition on ``pre_batch`` plus ``X``.

        Args:
            X: ``(b, q_new, d_continuous + 3)`` index-space batch.

        Returns:
            ``(b,)`` acquisition values of the joint batch
            ``[pre_batch; expand(X)]``.
        """
        X_full = self.expand(X)  # (b, q_new, d)
        if self.pre_batch is not None:
            pre = self.pre_batch.to(X_full).expand(
                *X_full.shape[:-2], self.pre_batch.shape[0], self.d_full
            )
            X_full = torch.cat([pre, X_full], dim=-2)  # (b, q_pre + q_new, d)
        return self.base_acqf(X_full)


def _greedy_group_maps(
    batch_constraints: BatchConstraints | None,
    q: int,
    d_continuous: int,
) -> dict[int, list[int]]:
    r"""Resolve batch constraints into a fixed candidate-to-group map per dim.

    Explicit partitions use :func:`_candidate_group_map`; an integer ``k=1``
    means all candidates share one value. Larger integer specs are ambiguous
    for a greedy builder and are rejected. A partition summing to more than
    ``q`` is reduced with :func:`_partial_partition_for_step` (the sequential
    GA convention): unconstrained while ``q <= len(partition)``, then
    tightened toward the full partition.

    Args:
        batch_constraints: Per-dim grouped constraints.
        q: Total batch size including any pre-batch rows.
        d_continuous: Number of leading continuous dimensions; grouped dims
            must lie in this block.

    Returns:
        Mapping ``dim -> [group_of_candidate_0, ..., group_of_candidate_q-1]``.

    Raises:
        ValueError: If a grouped dim is not continuous, a partition sums to
            fewer than ``q`` candidates, or an integer spec other than ``1`` is
            given.
    """
    if not batch_constraints:
        return {}
    group_maps: dict[int, list[int]] = {}
    for dim_idx, spec in batch_constraints.items():
        if dim_idx >= d_continuous:
            raise ValueError(
                f"Grouped dim {dim_idx} is not continuous (d_continuous={d_continuous})"
            )
        if isinstance(spec, Integral) and not isinstance(spec, bool):
            if int(spec) != 1:
                raise ValueError(
                    "The greedy lookup optimizer needs an explicit partition for "
                    f"dim {dim_idx}; integer k={spec} is ambiguous"
                )
            group_maps[dim_idx] = [0] * q
            continue
        partition = tuple(int(v) for v in spec)
        if sum(partition) < q:
            raise ValueError(
                f"Partition {partition} for dim {dim_idx} covers fewer than q={q} candidates"
            )
        if sum(partition) > q:
            partition_q = _partial_partition_for_step(partition, q)
            if partition_q is None:
                continue
            partition = partition_q
        group_maps[dim_idx] = _candidate_group_map(partition)
    return group_maps


def optimize_acqf_mixed_lookup_alternating(
    acq_function: torch.nn.Module,
    bounds: Tensor,
    q: int = 1,
    cat_emb_avail: Tensor | None = None,
    sol_emb_avail: Tensor | None = None,
    bas_emb_avail: Tensor | None = None,
    cat_is_salt_flags: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    pre_batch: Tensor | None = None,
    num_restarts: int = 4,
    raw_samples: int = 256,
    options: dict[str, object] | None = None,
    joint_refine: bool = False,
    refine_num_restarts: int = 8,
    refine_raw_samples: int = 256,
    progress_callback: Callable[[str], None] | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Greedy batch optimizer: BoTorch alternating local search in choice space.

    Builds the ``q`` candidates one at a time. Each step runs BoTorch's
    ``optimize_acqf_mixed_alternating`` (Sobol starts, 1-Hamming local search
    over the catalyst / solvent / base indices, L-BFGS-B on the continuous
    dims) on a :class:`_LookupAcquisitionFunction` that maps index-space
    points to the embedded space and evaluates the joint acquisition value
    together with the candidates already chosen. The discrete search space is
    therefore the same cartesian catalyst x solvent x base choice space the GA
    optimizers use in component mode, including the salt / base-zeroing rule.

    Batch constraints are enforced with a fixed candidate-to-group schedule
    (:func:`_greedy_group_maps`): the first candidate of a group optimizes the
    grouped dim freely and later members of the same group inherit its value
    as a fixed feature. ``pre_batch`` rows occupy the leading group slots.

    Args:
        acq_function: Acquisition function over the full space. Any module
            mapping ``(b, q, d)`` to ``(b,)`` is accepted, so averaged
            fantasy acquisitions work as well as BoTorch acquisitions.
        bounds: ``(2, d)`` bounds; the continuous block comes first.
        q: Total batch size including ``pre_batch`` rows.
        cat_emb_avail: ``(n_cat, d_cat)`` available-catalyst embeddings.
        sol_emb_avail: ``(n_sol, d_sol)`` available-solvent embeddings.
        bas_emb_avail: ``(n_bas, d_bas)`` available-base embeddings.
        cat_is_salt_flags: Optional ``(n_cat,)`` bool tensor for the salt rule.
        batch_constraints: Per-dim grouped constraints. Explicit partitions
            (e.g. ``{0: (2, 2, 2)}``) or integer ``1``.
        fixed_dims: Continuous dims pinned to constants.
        pre_batch: Optional ``(q_pre, d)`` tensor of fixed candidates.
        num_restarts: Local-search restarts per greedy step.
        raw_samples: Sobol starting candidates per greedy step.
        options: Extra options forwarded to the BoTorch optimizer
            (``maxiter_alternating``, ``maxiter_discrete``,
            ``maxiter_continuous``, ``batch_limit``, ...). Defaults use random
            Sobol initialization and no spray points.
        joint_refine: If True, finish with :func:`optimize_acqf_continuous`
            over all continuous dims of the whole batch under the exact
            batch constraints.
        refine_num_restarts: L-BFGS-B restarts for the joint refine.
        refine_raw_samples: Sobol samples for the joint refine.
        progress_callback: Optional callback receiving short progress messages.
        seed: Optional seed for the torch and Python RNGs used by the
            BoTorch optimizer, restored afterwards.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): ``(q, d)`` tensor, ``pre_batch`` rows first.
        - **acq_value** (*Tensor*): Scalar joint acquisition value of the batch.

    Raises:
        ValueError: If component tables are missing or ``pre_batch`` is not
            smaller than ``q``.
    """
    if cat_emb_avail is None or sol_emb_avail is None or bas_emb_avail is None:
        raise ValueError(
            "optimize_acqf_mixed_lookup_alternating requires cat_emb_avail, "
            "sol_emb_avail and bas_emb_avail (component mode only)."
        )
    d = bounds.shape[1]
    d_discrete = (
        cat_emb_avail.shape[1] + sol_emb_avail.shape[1] + bas_emb_avail.shape[1]
    )
    d_continuous = d - d_discrete
    q_pre = pre_batch.shape[0] if pre_batch is not None else 0
    if q_pre >= q:
        raise ValueError(f"pre_batch has {q_pre} rows but q={q}; nothing to optimize")

    device = _infer_acqf_device(acq_function)
    dtype = bounds.dtype
    cat_t = cat_emb_avail.to(device=device, dtype=dtype)
    sol_t = sol_emb_avail.to(device=device, dtype=dtype)
    bas_t = bas_emb_avail.to(device=device, dtype=dtype)
    flags_t = (
        cat_is_salt_flags.to(device=device) if cat_is_salt_flags is not None else None
    )
    bounds_t = bounds.to(device=device, dtype=dtype)

    n_cat, n_sol, n_bas = cat_t.shape[0], sol_t.shape[0], bas_t.shape[0]
    bounds_idx = torch.zeros(2, d_continuous + 3, device=device, dtype=dtype)
    bounds_idx[:, :d_continuous] = bounds_t[:, :d_continuous]
    bounds_idx[1, d_continuous] = n_cat - 1
    bounds_idx[1, d_continuous + 1] = n_sol - 1
    bounds_idx[1, d_continuous + 2] = n_bas - 1
    cat_dims = {
        d_continuous: list(range(n_cat)),
        d_continuous + 1: list(range(n_sol)),
        d_continuous + 2: list(range(n_bas)),
    }

    fixed_dims = dict(fixed_dims) if fixed_dims else {}
    group_maps = _greedy_group_maps(batch_constraints, q, d_continuous)
    for dim_idx in fixed_dims:
        group_maps.pop(dim_idx, None)

    resolved_options: dict[str, object] = {
        "initialization_strategy": "random",
        "num_spray_points": 0,
    }
    if options:
        resolved_options.update(options)

    current = (
        pre_batch.to(device=device, dtype=dtype)
        if pre_batch is not None
        else torch.empty(0, d, device=device, dtype=dtype)
    )
    acq_value = torch.tensor(float("nan"), device=device, dtype=dtype)

    py_state = random.getstate()
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        for step in range(q_pre, q):
            fixed_features = {int(k): float(v) for k, v in fixed_dims.items()}
            for dim_idx, group_map in group_maps.items():
                group = group_map[step]
                earlier = [i for i in range(step) if group_map[i] == group]
                if earlier:
                    fixed_features[dim_idx] = float(current[earlier[0], dim_idx].item())

            lookup = _LookupAcquisitionFunction(
                base_acqf=acq_function,
                d_continuous=d_continuous,
                cat_emb_avail=cat_t,
                sol_emb_avail=sol_t,
                bas_emb_avail=bas_t,
                cat_is_salt_flags=flags_t,
                pre_batch=current if current.shape[0] > 0 else None,
            )
            step_start = time.perf_counter()
            cand_idx, step_value = optimize_acqf_mixed_alternating(
                acq_function=lookup,
                bounds=bounds_idx,
                cat_dims=cat_dims,
                q=1,
                raw_samples=raw_samples,
                num_restarts=num_restarts,
                options=dict(resolved_options),
                fixed_features=fixed_features,
            )
            with torch.no_grad():
                new_row = lookup.expand(cand_idx.detach())  # (1, d)
            current = torch.cat([current, new_row], dim=0)
            acq_value = step_value.detach().reshape(())
            if progress_callback is not None:
                progress_callback(
                    f"greedy step {step + 1}/{q} done in "
                    f"{time.perf_counter() - step_start:.1f}s (acq={acq_value.item():.4f})"
                )
    random.setstate(py_state)

    if joint_refine:
        refined, acq_value = optimize_acqf_continuous(
            acq_function=acq_function,
            bounds=bounds_t,
            X_discrete=current[:, d_continuous:],
            pre_batch=pre_batch,
            batch_constraints=batch_constraints,
            fixed_dims=fixed_dims,
            num_restarts=refine_num_restarts,
            raw_samples=refine_raw_samples,
            progress_callback=progress_callback,
        )
        current = refined
    else:
        with torch.no_grad():
            acq_value = acq_function(current.unsqueeze(0)).reshape(())

    return current.detach(), acq_value.detach()



def _decompose_allowed_set_to_components(
    x_allowed: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose an ``(n, 3)`` allowed-set tensor into per-component scalar tables.

    Each column is treated as the raw scalar identifier of one component
    (catalyst / solvent / base). The unique values along each column become the
    rows of the corresponding per-component embedding tensor with ``d=1``.

    Args:
        x_allowed: ``(n, 3)`` tensor of allowed configurations.

    Returns:
        A three-tuple ``(cat_emb, sol_emb, bas_emb)`` of ``(n_unique, 1)`` tensors.
    """
    cat_emb = x_allowed[:, 0].unique().unsqueeze(-1)
    sol_emb = x_allowed[:, 1].unique().unsqueeze(-1)
    bas_emb = x_allowed[:, 2].unique().unsqueeze(-1)
    return cat_emb, sol_emb, bas_emb


def optimize_plan_ahead_qlogEI_fantasized(
    model: Model,
    q_batch: int,
    q_expand: int,
    x_available: Tensor,
    x_ordered: list[Tensor],
    x_unordered: Tensor,
    bounds: Tensor,
    k: int = 4,
    num_fantasies: list[int] | None = None,
    n_samples: int = 15000,
    popsize: int = 100,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    operating_mode: str = "layer_averaged",
    decompose_allowed_set: Callable[
        [Tensor], tuple[Tensor, Tensor, Tensor]
    ] = _decompose_allowed_set_to_components,
    ga_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_GA,
) -> tuple[Tensor, Tensor]:
    r"""Run fantasized PlanAhead with a sequential two-stage terminal step.

    The optimizer performs ``k`` lookahead rounds where each intermediate round
    selects ``q_batch`` candidates from the currently available discrete set,
    then conditions the model on posterior samples — without refitting
    hyperparameters — to produce the next-depth branch models. This preserves
    the "true conditioning of the original posterior" semantics: all branch
    models share the root model's kernel and noise hyperparameters, updated
    only in their training data. At the terminal round, branch models are
    aggregated according to ``operating_mode`` and optimization is carried out
    sequentially:

    1. pick ``q_expand`` expansion candidates from ``x_unordered``;
    2. complete the full ``q_batch`` by optimizing the remaining
       ``q_batch - q_expand`` candidates from the terminal available set using
       ``pre_batch``.

    Returned candidates are only the ``q_expand`` expansion points from stage 1,
    while ``acq_value`` is the full joint value of the completed ``q_batch``.

    Args:
        model: Fitted BoTorch surrogate model.
        q_batch: Total terminal batch size used for the final joint value.
        q_expand: Number of terminal expansion candidates selected from
            ``x_unordered``.
        x_available: ``(n, d_disc)`` tensor of currently available discrete
            configurations.
        x_ordered: List of ``(n_i, d_disc)`` tensors that become available at
            each future lookahead depth.
        x_unordered: ``(n_unordered, d_disc)`` tensor of unavailable expansion
            candidates.
        bounds: ``(2, d)`` bounds for the full search space, where
            ``d = d_cont + d_disc``.
        k: Number of lookahead (fantasy) depths. Must satisfy ``k >= 1``.
        num_fantasies: Number of fantasy samples per depth. Must have length
            ``k``. Defaults are provided for ``k`` in ``{1, 2, 3, 4}``.
        n_samples: Total GA function evaluations for each inner optimization.
        popsize: GA population size.
        batch_constraints: Optional mapping from dimension index to either
            integer ``k`` or explicit partition tuple/list.
        fixed_dims: Optional mapping from dimension index to fixed value.
        ga_optimizer: Batch optimizer used for every inner GA call. Defaults
            to :func:`optimize_acqf_mixed_GA`; pass
            :func:`optimize_acqf_mixed_canonical_GA` for the symmetry-aware,
            corrected-operator variant.
        progress_callback: Optional callback receiving progress messages from
            nested GA calls.
        operating_mode: Aggregation mode for terminal optimization.

            - ``"only_terminal"``: average qLogEI over terminal-layer models only.
            - ``"averaged"``: average qLogEI over all node-level models from
              depths ``1..k`` (weighted by node count).
            - ``"layer_averaged"``: average each depth first, then average
              those per-depth means equally.
        decompose_allowed_set: Maps an ``(n, d_disc)`` allowed-set tensor to
            the ``(cat_emb, sol_emb, bas_emb)`` component tables passed to the
            GA in component mode. Defaults to
            :func:`_decompose_allowed_set_to_components`, which expects
            ``d_disc = 3`` scalar identifiers.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): ``(q_expand, d)`` tensor of selected expansion candidates.
        - **acq_value** (*Tensor*): Scalar full joint acquisition value of the completed ``q_batch``.

    Raises:
        ValueError: If ``k < 1``.
        ValueError: If ``q_batch <= 0`` or ``q_expand <= 0``.
        ValueError: If ``q_expand >= q_batch``.
        ValueError: If ``num_fantasies`` length does not equal ``k``.
        ValueError: If terminal expansion/available pools are empty when needed.
        ValueError: If ``operating_mode`` is unknown.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")
    if q_batch <= 0:
        raise ValueError(f"q_batch must be > 0, got q_batch={q_batch}")
    if q_expand <= 0:
        raise ValueError(f"q_expand must be > 0, got q_expand={q_expand}")
    if q_expand >= q_batch:
        raise ValueError(
            f"q_expand must be strictly smaller than q_batch, got q_expand={q_expand}, q_batch={q_batch}"
        )

    valid_operating_modes = {"only_terminal", "averaged", "layer_averaged"}
    if operating_mode not in valid_operating_modes:
        raise ValueError(
            "operating_mode must be one of "
            f"{sorted(valid_operating_modes)}, got {operating_mode!r}"
        )

    if num_fantasies is None:
        _defaults = {1: [1], 2: [10, 1], 3: [10, 5, 1], 4: [10, 5, 3, 1]}
        if k not in _defaults:
            raise ValueError(
                f"num_fantasies must be given explicitly for k={k} (defaults exist for k=1..4)"
            )
        num_fantasies = _defaults[k]
    if len(num_fantasies) != k:
        raise ValueError(
            f"num_fantasies must have length k={k}, got length={len(num_fantasies)}"
        )
    if any(nf <= 0 for nf in num_fantasies):
        raise ValueError("All entries in num_fantasies must be positive")

    allowed_sets: list[Tensor] = []
    accumulated_discrete_sets = x_available.clone()
    for j in range(k):
        allowed_sets.append(accumulated_discrete_sets.clone())
        if j < len(x_ordered):
            accumulated_discrete_sets = torch.cat(
                [accumulated_discrete_sets, x_ordered[j]], dim=0
            )
    terminal_available_set = accumulated_discrete_sets.clone()

    active_fantasy_models = [model]
    fantasy_models_by_layer: list[list[Model]] = []

    for j in range(k):
        next_depth_fantasy_models = []
        n_allowed = allowed_sets[j].shape[0]

        for fm in active_fantasy_models:
            fm.eval()

            best_f = best_observed_f(fm)

            acqf = qLogExpectedImprovement(model=fm, best_f=best_f)

            if n_allowed > 0:
                cat_emb_j, sol_emb_j, bas_emb_j = decompose_allowed_set(allowed_sets[j])
            else:
                cat_emb_j = sol_emb_j = bas_emb_j = None

            cands, _ = ga_optimizer(
                acq_function=acqf,
                bounds=bounds,
                q=q_batch,
                n_evals_ga=n_samples,
                pop_size=popsize,
                cat_emb_avail=cat_emb_j,
                sol_emb_avail=sol_emb_j,
                bas_emb_avail=bas_emb_j,
                batch_constraints=batch_constraints,
                fixed_dims=fixed_dims,
                progress_callback=progress_callback,
            )

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies[j]]))
            with torch.no_grad():
                posterior = fm.posterior(cands, observation_noise=False)
                y_samples = sampler(posterior)  # (num_fantasies[j], q_batch, 1)

            for branch_idx in range(num_fantasies[j]):
                branch_model = fm.condition_on_observations(
                    X=cands,
                    Y=y_samples[branch_idx],
                )
                next_depth_fantasy_models.append(branch_model)

        active_fantasy_models = next_depth_fantasy_models
        fantasy_models_by_layer.append(active_fantasy_models)

    class AveragedAcquisitionFunction(torch.nn.Module):
        def __init__(
            self,
            models_by_layer: list[list[Model]],
            mode: str,
        ):
            super().__init__()
            self.mode = mode
            self.layer_acqfs: list[list[AcquisitionFunction]] = []

            for layer_models in models_by_layer:
                layer_acqfs: list[AcquisitionFunction] = []
                for fm in layer_models:
                    fm.eval()
                    best_f = best_observed_f(fm)
                    layer_acqfs.append(qLogExpectedImprovement(model=fm, best_f=best_f))
                self.layer_acqfs.append(layer_acqfs)

            self.terminal_acqfs = self.layer_acqfs[-1]
            self.flat_acqfs = [
                acqf for layer_acqfs in self.layer_acqfs for acqf in layer_acqfs
            ]

        def forward(self, X):
            if self.mode == "only_terminal":
                terminal_vals = torch.stack(
                    [af(X) for af in self.terminal_acqfs], dim=0
                )
                return terminal_vals.mean(dim=0)

            if self.mode == "averaged":
                all_vals = torch.stack([af(X) for af in self.flat_acqfs], dim=0)
                return all_vals.mean(dim=0)

            layer_means = []
            for layer_acqfs in self.layer_acqfs:
                layer_vals = torch.stack([af(X) for af in layer_acqfs], dim=0)
                layer_means.append(layer_vals.mean(dim=0))
            return torch.stack(layer_means, dim=0).mean(dim=0)

    terminal_acqf = AveragedAcquisitionFunction(
        models_by_layer=fantasy_models_by_layer,
        mode=operating_mode,
    )

    n_terminal_unordered = x_unordered.shape[0]
    if n_terminal_unordered == 0:
        raise ValueError("x_unordered must be non-empty when q_expand > 0")

    n_terminal_fill = q_batch - q_expand
    n_terminal_available = terminal_available_set.shape[0]
    if n_terminal_fill > 0 and n_terminal_available == 0:
        raise ValueError(
            "Terminal available set is empty, cannot complete q_batch with available candidates"
        )

    cat_unord, sol_unord, bas_unord = decompose_allowed_set(x_unordered)
    cat_term, sol_term, bas_term = decompose_allowed_set(terminal_available_set)

    terminal_expand_candidates, _ = ga_optimizer(
        acq_function=terminal_acqf,
        bounds=bounds,
        q=q_expand,
        n_evals_ga=n_samples,
        pop_size=popsize,
        cat_emb_avail=cat_unord,
        sol_emb_avail=sol_unord,
        bas_emb_avail=bas_unord,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        progress_callback=progress_callback,
    )

    _, terminal_joint_acq_value = ga_optimizer(
        acq_function=terminal_acqf,
        bounds=bounds,
        q=q_batch,
        n_evals_ga=n_samples,
        pop_size=popsize,
        cat_emb_avail=cat_term,
        sol_emb_avail=sol_term,
        bas_emb_avail=bas_term,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        pre_batch=terminal_expand_candidates,
        progress_callback=progress_callback,
    )

    return terminal_expand_candidates, terminal_joint_acq_value


def optimize_plan_ahead_qlogEI_myopic(
    model: Model,
    q_expand: int,
    x_unordered: Tensor,
    bounds: Tensor,
    n_samples: int = 20000,
    popsize: int = 150,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    decompose_allowed_set: Callable[
        [Tensor], tuple[Tensor, Tensor, Tensor]
    ] = _decompose_allowed_set_to_components,
    ga_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_GA,
) -> tuple[Tensor, Tensor]:
    r"""Myopic optimiser for search-space expansion using qLogExpectedImprovement.

    Performs a standard singular myopic step to pick expansion candidates solely from
    `x_unordered` using `qLogExpectedImprovement` evaluated on the original root model.
    No future fantasies or available sets are simulated or assessed.

    Args:
        model: Fitted BoTorch surrogate model.
        q_expand: Number of expansion candidates to suggest from `x_unordered`.
        x_unordered: `(n*, d_disc)` tensor of candidate configurations for
            search-space expansion.
        bounds: `(2, d)` bounds for the full input space (`d = d_cont + d_disc`).
        n_samples: Total GA function evaluations for the myopic optimizer.
            Default is 20,000.
        popsize: GA population size. Default is 150.
        batch_constraints: Optional mapping from dimension index to either
            integer ``k`` or explicit partition tuple/list.
        fixed_dims: Dictionary mapping dimension indices to fixed float values.
        ga_optimizer: Batch optimizer used for the inner GA call. Defaults to
            :func:`optimize_acqf_mixed_GA`; pass
            :func:`optimize_acqf_mixed_canonical_GA` for the symmetry-aware,
            corrected-operator variant.
        progress_callback: Optional callback receiving short progress messages
            from the inner GA optimizer.
        decompose_allowed_set: Maps the ``(n*, d_disc)`` ``x_unordered`` tensor
            to the ``(cat_emb, sol_emb, bas_emb)`` component tables passed to
            the GA in component mode. Defaults to
            :func:`_decompose_allowed_set_to_components`, which expects
            ``d_disc = 3`` scalar identifiers.

    Returns:
        tuple[Tensor, Tensor]: A two-element tuple containing:

        - **candidates** (*Tensor*): A `(q_expand, d)` tensor of selected expansion candidates.
        - **acq_value** (*Tensor*): A scalar tensor representing their joint batch acquisition value.
    """
    best_f = best_observed_f(model)
    acqf = qLogExpectedImprovement(model=model, best_f=best_f)

    if x_unordered.shape[0] > 0:
        cat_un, sol_un, bas_un = decompose_allowed_set(x_unordered)
    else:
        cat_un = sol_un = bas_un = None

    return ga_optimizer(
        acq_function=acqf,
        bounds=bounds,
        q=q_expand,
        n_evals_ga=n_samples,
        pop_size=popsize,
        cat_emb_avail=cat_un,
        sol_emb_avail=sol_un,
        bas_emb_avail=bas_un,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        progress_callback=progress_callback,
    )


def optimize_plan_ahead_qlogEI_fantasized_canonical(
    *args,
    canonicalize: bool = True,
    discrete_mutation_prob: float = 0.05,
    cache_acq_values: bool = False,
    cache_decimals: int | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    r"""Rollout PlanAhead driven by the canonical batch optimizer.

    A thin wrapper over :func:`optimize_plan_ahead_qlogEI_fantasized` that
    replaces every inner GA call with
    :func:`optimize_acqf_mixed_canonical_GA`. The fantasy-tree logic, the
    conditioning semantics and the terminal aggregation are unchanged; only the
    batch optimizer differs.

    Args:
        canonicalize: Whether each inner GA canonicalizes its population.
        discrete_mutation_prob: Per-gene random-reset probability for the
            discrete variables.
        cache_acq_values: Whether each inner GA memoizes acquisition values.
        cache_decimals: Optional rounding of cache keys.
        *args: Forwarded to :func:`optimize_plan_ahead_qlogEI_fantasized`.
        **kwargs: Forwarded to :func:`optimize_plan_ahead_qlogEI_fantasized`.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): ``(q_expand, d)`` expansion candidates.
        - **acq_value** (*Tensor*): Scalar joint value of the completed batch.
    """
    ga = functools.partial(
        optimize_acqf_mixed_canonical_GA,
        canonicalize=canonicalize,
        discrete_mutation_prob=discrete_mutation_prob,
        cache_acq_values=cache_acq_values,
        cache_decimals=cache_decimals,
    )
    return optimize_plan_ahead_qlogEI_fantasized(*args, ga_optimizer=ga, **kwargs)


def optimize_plan_ahead_qlogEI_myopic_canonical(
    *args,
    canonicalize: bool = True,
    discrete_mutation_prob: float = 0.05,
    cache_acq_values: bool = False,
    cache_decimals: int | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    r"""Myopic expansion step driven by the canonical batch optimizer.

    A thin wrapper over :func:`optimize_plan_ahead_qlogEI_myopic` that replaces
    the inner GA call with :func:`optimize_acqf_mixed_canonical_GA`.

    Args:
        canonicalize: Whether the inner GA canonicalizes its population.
        discrete_mutation_prob: Per-gene random-reset probability for the
            discrete variables.
        cache_acq_values: Whether the inner GA memoizes acquisition values.
        cache_decimals: Optional rounding of cache keys.
        *args: Forwarded to :func:`optimize_plan_ahead_qlogEI_myopic`.
        **kwargs: Forwarded to :func:`optimize_plan_ahead_qlogEI_myopic`.

    Returns:
        A two-element tuple containing:

        - **candidates** (*Tensor*): ``(q_expand, d)`` expansion candidates.
        - **acq_value** (*Tensor*): Scalar joint batch acquisition value.
    """
    ga = functools.partial(
        optimize_acqf_mixed_canonical_GA,
        canonicalize=canonicalize,
        discrete_mutation_prob=discrete_mutation_prob,
        cache_acq_values=cache_acq_values,
        cache_decimals=cache_decimals,
    )
    return optimize_plan_ahead_qlogEI_myopic(*args, ga_optimizer=ga, **kwargs)


def _pool_group_options(
    counts: list[int],
    partition: tuple[int, ...] | None,
    max_groups: int | None,
) -> tuple[list[int], bool]:
    """Groups a new point may join, and whether it may open a new group.

    Args:
        counts: Current member count per group.
        partition: Explicit target partition (sorted descending), or ``None``.
        max_groups: Group cap for integer constraints, or ``None``.

    Returns:
        ``(joinable_group_indices, can_open)``.
    """
    if partition is None and max_groups is None:
        return [], True
    if partition is None:
        return list(range(len(counts))), len(counts) < max_groups

    def completable(c: list[int]) -> bool:
        c = sorted(c, reverse=True)
        return len(c) <= len(partition) and all(a <= b for a, b in zip(c, partition))

    joinable = [
        g for g in range(len(counts))
        if completable(counts[:g] + [counts[g] + 1] + counts[g + 1:])
    ]
    return joinable, completable(counts + [1])




class WeightedSumAcquisition(AcquisitionFunction):
    r"""``α(X) = Σᵢ wᵢ αᵢ(X)`` over member acquisition functions.

    :func:`optimize_acqf_mixed_pool_greedy` scores it member by member on its
    fast path when every member is a supported ``qLogExpectedImprovement``.
    """

    def __init__(
        self,
        acq_functions: list[AcquisitionFunction],
        weights: Tensor,
        release_member_caches: bool = False,
    ):
        r"""
        Args:
            acq_functions: ``k`` member acquisition functions.
            weights: ``(k,)`` member weights.
            release_member_caches: After evaluating a member in :meth:`forward`,
                drop its exact GP's prediction caches, so at most one member
                model holds them at a time. Each call then rebuilds every
                member's caches; meant for few evaluations of many large models.
        """
        super().__init__(model=acq_functions[0].model)
        self.acq_functions = torch.nn.ModuleList(acq_functions)
        self.register_buffer("weights", weights)
        self.release_member_caches = release_member_caches

    def forward(self, X: Tensor) -> Tensor:
        r"""Weighted sum over members.

        Args:
            X: ``(b, q, d)`` candidate batches.

        Returns:
            ``(b,)`` acquisition values.
        """
        values = []
        for af in self.acq_functions:
            values.append(af(X))
            if self.release_member_caches:
                release_gp_caches(af.model)
        stacked = torch.stack(values, dim=0)  # (k, b)
        return (self.weights.to(stacked).unsqueeze(-1) * stacked).sum(dim=0)


def release_gp_caches(model) -> None:
    """Drop an exact GP's prediction caches; its next posterior call rebuilds them.

    Args:
        model: A GPyTorch ``ExactGP`` (e.g. a BoTorch ``SingleTaskGP``).
    """
    model.prediction_strategy = None


def _exact_gp_cholesky(model, x: Tensor) -> Tensor:
    """``L`` with ``L Lᵀ = K(x, x) + σ² I`` for transformed inputs ``x`` ``(n, d)``."""
    from linear_operator.utils.cholesky import psd_safe_cholesky

    k = model.covar_module(x).to_dense()
    k = k + model.likelihood.noise * torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    return psd_safe_cholesky(k)


def _pool_key(pool_t: Tensor) -> str:
    """Content hash of a transformed pool ``(N, d)``."""
    import hashlib

    return hashlib.sha1(pool_t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class GPPoolPosteriorCache:
    r"""Root posterior of an exact GP over candidate pools, shared by fantasy models.

    Holds ``L₀`` with ``L₀ L₀ᵀ = K(X₀, X₀) + σ² I`` on the root's transformed
    training inputs ``X₀`` ``(n₀, d)``, ``β₀ = L₀⁻¹ (y₀ − m(X₀))``, and for every
    pool ``P`` it has seen ``W₀ = L₀⁻¹ K(X₀, P)`` ``(n₀, N)`` with the root's
    standardized posterior mean and variance on ``P``. A model whose training
    inputs start with ``X₀`` and whose hyperparameters equal the root's — every
    model built by ``condition_on_observations`` from the root — reuses them and
    adds only a rank-``f`` correction for its ``f`` extra points.
    """

    def __init__(self, root_model):
        r"""
        Args:
            root_model: Fitted single-output exact GP in eval mode.
        """
        self.model = root_model
        self.x = root_model.train_inputs[0]
        with torch.no_grad():
            self.L = _exact_gp_cholesky(root_model, self.x)  # (n0, n0)
            resid = (root_model.train_targets - root_model.mean_module(self.x)).unsqueeze(-1)
            self.beta = torch.linalg.solve_triangular(self.L, resid, upper=False).squeeze(-1)  # (n0,)
        self._hyper = tuple(p.detach() for p in root_model.parameters())
        self._pools: dict[str, dict[str, Tensor]] = {}

    def matches(self, model) -> bool:
        """Whether ``model`` is the root conditioned on extra points."""
        x = model.train_inputs[0]
        hyper = tuple(p.detach() for p in model.parameters())
        return (
            x.shape[0] >= self.x.shape[0]
            and torch.equal(x[: self.x.shape[0]], self.x)
            and len(hyper) == len(self._hyper)
            and all(torch.equal(a, b) for a, b in zip(hyper, self._hyper))
        )

    @torch.no_grad()
    def pool(self, pool_t: Tensor, key: str | None, store: bool = True) -> dict[str, Tensor]:
        r"""Root posterior terms of a candidate pool, cached under ``key``.

        Args:
            pool_t: ``N x d`` pool in the model's transformed input space.
            key: Cache key. A cached entry under ``key`` is returned as is;
                ``None`` computes the entry without caching it.
            store: Whether to cache the computed entry under ``key``.
                ``False`` computes the entry (tagged with ``key``) without
                keeping it.

        Returns:
            A dict with ``x`` (``pool_t``), ``w = L₀⁻¹ K(X₀, P)`` ``(n0, N)``,
            the root's standardized posterior ``mean`` and ``var`` ``(N,)``,
            and ``key``.
        """
        if key is not None and key in self._pools:
            return self._pools[key]
        covar = self.model.covar_module
        w = torch.linalg.solve_triangular(self.L, covar(self.x, pool_t).to_dense(), upper=False)
        entry = {
            "x": pool_t,
            "w": w,
            "mean": self.model.mean_module(pool_t) + w.transpose(-1, -2) @ self.beta,
            "var": covar(pool_t, diag=True) - (w * w).sum(dim=0),
            "key": key,
        }
        if key is not None and store:
            self._pools[key] = entry
        return entry

    @torch.no_grad()
    def batch_terms(self, entry: dict[str, Tensor], batch_t: Tensor) -> dict[str, Tensor]:
        r"""Root posterior terms of a batch against a pool entry.

        Args:
            entry: Root pool terms from :meth:`pool`.
            batch_t: ``s x d`` batch in the model's transformed input space.

        Returns:
            A dict with ``x`` (``batch_t``), ``w = L₀⁻¹ K(X₀, B)`` ``(n0, s)``,
            the root's standardized posterior ``mean`` ``(s,)`` and ``cov``
            ``(s, s)``, and the pool–batch posterior covariance ``cross``
            ``(N, s)``.
        """
        covar = self.model.covar_module
        w_b = torch.linalg.solve_triangular(self.L, covar(self.x, batch_t).to_dense(), upper=False)
        return {
            "x": batch_t,
            "w": w_b,
            "mean": self.model.mean_module(batch_t) + w_b.transpose(-1, -2) @ self.beta,
            "cov": covar(batch_t).to_dense() - w_b.transpose(-1, -2) @ w_b,
            "cross": covar(entry["x"], batch_t).to_dense() - entry["w"].transpose(-1, -2) @ w_b,
        }


class _ExactGPPoolPosterior:
    r"""Joint posterior of one model that is a root GP conditioned on ``f`` extra points."""

    def __init__(self, model, cache: GPPoolPosteriorCache):
        from linear_operator.utils.cholesky import psd_safe_cholesky

        self.model = model
        self.cache = cache
        x = model.train_inputs[0]
        n0 = cache.x.shape[0]
        self.f_x = x[n0:] if x.shape[0] > n0 else None
        self._ext: dict[str, Tensor] = {}
        if self.f_x is not None:
            covar = model.covar_module
            with torch.no_grad():
                self.w_f = torch.linalg.solve_triangular(
                    cache.L, covar(cache.x, self.f_x).to_dense(), upper=False
                )  # (n0, f)
                s_ff = covar(self.f_x).to_dense() - self.w_f.transpose(-1, -2) @ self.w_f
                s_ff = s_ff + model.likelihood.noise * torch.eye(self.f_x.shape[0], dtype=x.dtype, device=x.device)
                self.l_f = psd_safe_cholesky(s_ff)  # (f, f)
                r_f = model.train_targets[n0:] - model.mean_module(self.f_x) - self.w_f.transpose(-1, -2) @ cache.beta
                self.beta_f = torch.linalg.solve_triangular(self.l_f, r_f.unsqueeze(-1), upper=False).squeeze(-1)
        transform = getattr(model, "outcome_transform", None)
        self.y_mean = 0.0 if transform is None else transform.means.reshape(())
        self.y_std = 1.0 if transform is None else transform.stdvs.reshape(())

    def _ext_rows(self, x_t: Tensor, w0: Tensor) -> Tensor:
        """Extension rows ``L_f⁻¹ (K(F, x) − W_fᵀ W₀)`` ``(f, m)``."""
        k_f = self.model.covar_module(self.f_x, x_t).to_dense() - self.w_f.transpose(-1, -2) @ w0
        return torch.linalg.solve_triangular(self.l_f, k_f, upper=False)

    @torch.no_grad()
    def joint(self, entry: dict[str, Tensor], terms: dict[str, Tensor] | None) -> tuple[Tensor, Tensor]:
        r"""Outcome-space joint mean ``(N, s+1)`` and covariance ``(N, s+1, s+1)`` of ``batch ∪ {x}``.

        Args:
            entry: Root pool terms from :meth:`GPPoolPosteriorCache.pool`.
            terms: Root batch terms from :meth:`GPPoolPosteriorCache.batch_terms`,
                ``None`` for an empty batch.
        """
        mean_p, var_p = entry["mean"], entry["var"]
        w1_p = None
        if self.f_x is not None:
            key = entry["key"]
            if key is not None and key in self._ext:
                w1_p = self._ext[key]
            else:
                w1_p = self._ext_rows(entry["x"], entry["w"])  # (f, N)
                if key is not None:
                    self._ext[key] = w1_p
            mean_p = mean_p + w1_p.transpose(-1, -2) @ self.beta_f
            var_p = var_p - (w1_p * w1_p).sum(dim=0)
        n_pool = mean_p.shape[0]
        mean = mean_p.unsqueeze(-1)
        cov = var_p.clamp_min(0.0).reshape(n_pool, 1, 1)
        if terms is not None:
            b_mean, b_cov, cross = terms["mean"], terms["cov"], terms["cross"]
            if self.f_x is not None:
                w1_b = self._ext_rows(terms["x"], terms["w"])  # (f, s)
                b_mean = b_mean + w1_b.transpose(-1, -2) @ self.beta_f
                b_cov = b_cov - w1_b.transpose(-1, -2) @ w1_b
                cross = cross - w1_p.transpose(-1, -2) @ w1_b
            s = b_mean.shape[0]
            mean = torch.cat([b_mean.expand(n_pool, s), mean], dim=-1)
            top = torch.cat([b_cov.expand(n_pool, s, s), cross.unsqueeze(-1)], dim=-1)
            cov = torch.cat([top, torch.cat([cross.unsqueeze(1), cov], dim=-1)], dim=1)
        return mean * self.y_std + self.y_mean, cov * self.y_std**2


def _supports_pool_fast_path(acq_function: AcquisitionFunction) -> bool:
    """``qLogExpectedImprovement`` on a single-output, non-batched ``SingleTaskGP``."""
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.acquisition.objective import IdentityMCObjective
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.likelihoods import GaussianLikelihood

    if type(acq_function) is not qLogExpectedImprovement:
        return False
    model = acq_function.model
    transform = getattr(model, "outcome_transform", None)
    return (
        isinstance(model, SingleTaskGP)
        and model.num_outputs == 1
        and len(model.batch_shape) == 0
        and type(model.likelihood) is GaussianLikelihood
        and (transform is None or type(transform) is Standardize)
        and isinstance(acq_function.objective, IdentityMCObjective)
        and acq_function.X_pending is None
        and acq_function._constraints is None
        and acq_function.posterior_transform is None
    )


class _PoolAcquisitionEvaluator:
    r"""Scores ``batch ∪ {x}`` for every row ``x`` of a candidate pool.

    Fast path: every member (the acquisition function itself, or each member of
    a :class:`WeightedSumAcquisition`) is ``qLogExpectedImprovement`` on a
    single-output exact GP. Joint posteriors come from
    :class:`_ExactGPPoolPosterior` (root factors reused through
    ``posterior_cache``), samples from fixed QMC base samples ``z``
    ``(n_mc, q)``, and each member's own qLogEI reductions turn them into
    values. Each member's value is ``max(MC qLogEI, maxᵢ analytic log EI(xᵢ))``:
    the second term is a lower bound on log qEI that stays informative where
    no QMC sample reaches the improvement region (there the MC estimate
    floors). With ``n_mc_screen`` set, every row is first ranked on the first
    ``n_mc_screen`` samples by a cheap proxy (hard max over q,
    ``log E[softplus(max Y − best_f)]``) and only the best ``n_keep`` are
    scored with the exact reductions on all samples. Other acquisition functions are called on ``(N, s+1, d)`` tensors.
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        q: int,
        n_mc: int | None,
        seed: int,
        n_mc_screen: int | None = None,
        n_keep: int = 64,
        posterior_cache: GPPoolPosteriorCache | None = None,
        chunk_size: int = 2048,
    ):
        from botorch.utils.sampling import draw_sobol_normal_samples
        from linear_operator.utils.cholesky import psd_safe_cholesky

        self.acq_function = acq_function
        self.chunk_size = chunk_size
        self.n_mc_screen = n_mc_screen
        self.n_keep = n_keep
        if isinstance(acq_function, WeightedSumAcquisition):
            members, weights = list(acq_function.acq_functions), acq_function.weights
        else:
            members, weights = [acq_function], torch.ones(1)
        self.fast = all(_supports_pool_fast_path(m) for m in members)
        self._call_entries: dict[tuple[int, str], dict[str, Tensor]] = {}
        if not self.fast:
            return
        self._chol = psd_safe_cholesky
        self.members = members
        self.model = members[0].model
        x = self.model.train_inputs[0]
        self.weights = weights.to(dtype=x.dtype, device=x.device)
        caches: list[GPPoolPosteriorCache] = [] if posterior_cache is None else [posterior_cache]
        self.posteriors = []
        for m in members:
            cache = next((c for c in caches if c.matches(m.model)), None)
            if cache is None:
                cache = GPPoolPosteriorCache(m.model)
                caches.append(cache)
            self.posteriors.append(_ExactGPPoolPosterior(m.model, cache))
        if n_mc is None:
            sampler = getattr(members[0], "sampler", None)
            n_mc = int(sampler.sample_shape[0]) if sampler is not None else 512
        self.z = draw_sobol_normal_samples(d=q, n=n_mc, seed=seed).to(x)  # (n_mc, q)

    def _mc_values(self, acq: AcquisitionFunction, mean: Tensor, chol: Tensor, z: Tensor) -> Tensor:
        """qLogEI from means ``(N, s+1)``, Cholesky factors ``(N, s+1, s+1)`` and base samples ``(n_mc, s+1)``."""
        out = []
        for start in range(0, mean.shape[0], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            obj = mean[sl].unsqueeze(0) + torch.einsum("nij,mj->mni", chol[sl], z)  # (n_mc, c, s+1)
            out.append(acq._sample_reduction(acq._q_reduction(acq._sample_forward(obj))))
        return torch.cat(out)

    @staticmethod
    def _best_single_log_ei(acq: AcquisitionFunction, mean: Tensor, cov: Tensor) -> Tensor:
        r"""``maxᵢ log EI(xᵢ)`` over the ``s+1`` points of each row: a lower bound on log qEI.

        Args:
            acq: Member qLogEI (for ``best_f``).
            mean: ``(N, s+1)`` outcome-space means.
            cov: ``(N, s+1, s+1)`` outcome-space covariances.

        Returns:
            ``(N,)`` bound.
        """
        from botorch.acquisition.analytic import _log_ei_helper

        sigma = cov.diagonal(dim1=-2, dim2=-1).clamp_min(1e-24).sqrt()
        u = (mean - acq.best_f) / sigma
        return (_log_ei_helper(u) + sigma.log()).amax(dim=-1)

    @staticmethod
    def _screen_values(acq: AcquisitionFunction, mean: Tensor, chol: Tensor, z: Tensor) -> Tensor:
        """Cheap ranking proxy of qLogEI: hard max over q, ``log E[softplus(max Y − best_f)]``."""
        from botorch.utils.safe_math import log_softplus

        obj = mean.unsqueeze(0) + torch.einsum("nij,mj->mni", chol, z)  # (n_mc, N, s+1)
        li = log_softplus(obj.amax(dim=-1) - acq.best_f, tau=acq.tau_relu)  # (n_mc, N)
        return torch.logsumexp(li, dim=0) - math.log(z.shape[0])

    def _weighted(
        self, parts: list[tuple[Tensor, Tensor]], z: Tensor, rows: Tensor | None, screen: bool = False
    ) -> Tensor:
        total = 0.0
        for w, acq, (mean, chol, single) in zip(self.weights, self.members, parts):
            if rows is not None:
                mean, chol, single = mean[rows], chol[rows], single[rows]
            values = self._screen_values(acq, mean, chol, z) if screen else self._mc_values(acq, mean, chol, z)
            total = total + w * torch.maximum(values, single)
        return total

    @torch.no_grad()
    def batch_values(self, batch: Tensor, pool: Tensor, cache: str = "root") -> tuple[Tensor, Tensor]:
        r"""Acquisition value of ``batch ∪ {x}`` for every pool row ``x``.

        Args:
            batch: ``(s, d)`` points already in the batch (``s`` may be 0).
            pool: ``(N, d)`` candidate rows.
            cache: Where the pool factors are kept: ``"root"`` in the shared
                root cache (pools reused across calls), ``"call"`` in this
                evaluator (pools reused within one optimizer call), ``"none"``.

        Returns:
            ``(values, screen)``, both ``(N,)``: final values (``-inf`` for rows
            dropped by the screen) and screening values of all rows.
        """
        if not self.fast:
            out = []
            for start in range(0, pool.shape[0], self.chunk_size):
                cands = pool[start:start + self.chunk_size].unsqueeze(1)  # (c, 1, d)
                x = torch.cat([batch.unsqueeze(0).expand(cands.shape[0], -1, -1), cands], dim=1)
                out.append(self.acq_function(x))  # (c,)
            values = torch.cat(out)
            return values, values

        pool_t = self.model.transform_inputs(pool)
        batch_t = self.model.transform_inputs(batch)
        key = None if cache == "none" else _pool_key(pool_t)
        root_terms: dict[int, tuple[dict[str, Tensor], dict[str, Tensor] | None]] = {}
        parts = []
        for posterior, posterior_acq in zip(self.posteriors, self.members):
            cid = id(posterior.cache)
            if cid not in root_terms:
                if cache == "call":
                    if (cid, key) not in self._call_entries:
                        self._call_entries[(cid, key)] = posterior.cache.pool(pool_t, key, store=False)
                    entry = self._call_entries[(cid, key)]
                else:
                    entry = posterior.cache.pool(pool_t, key)
                terms = posterior.cache.batch_terms(entry, batch_t) if batch.shape[0] > 0 else None
                root_terms[cid] = (entry, terms)
            mean, cov = posterior.joint(*root_terms[cid])
            parts.append((mean, self._chol(cov), self._best_single_log_ei(posterior_acq, mean, cov)))
        z = self.z[:, : batch.shape[0] + 1]
        if self.n_mc_screen is None or pool.shape[0] <= self.n_keep:
            values = self._weighted(parts, z, None)
            return values, values
        screen = self._weighted(parts, z[: self.n_mc_screen], None, screen=True)
        keep = screen.topk(self.n_keep).indices
        values = torch.full_like(screen, -float("inf"))
        values[keep] = self._weighted(parts, z, keep)
        return values, screen


def _pool_components(
    n_pool: int,
    n_cat: int,
    n_sol: int,
    n_bas: int,
    generator: torch.Generator,
) -> Tensor:
    """``(n_pool, 3)`` long tensor of random ``(cat, sol, bas)`` indices."""
    return torch.stack(
        [torch.randint(n, (n_pool,), generator=generator) for n in (n_cat, n_sol, n_bas)], dim=-1
    )


def _match_component_rows(rows: Tensor, table: Tensor) -> Tensor:
    """Index into ``table`` ``(n_t, w)`` of every row of ``rows`` ``(m, w)``; raises if one is missing."""
    hits = torch.isclose(rows.unsqueeze(1), table.unsqueeze(0), atol=1e-12, rtol=0.0).all(dim=-1)
    if not bool(hits.any(dim=1).all()):
        raise ValueError("warm_start rows must use only the available component rows")
    return hits.float().argmax(dim=1)


def _refine_single_points(
    acq_function: AcquisitionFunction,
    x: Tensor,
    bounds: Tensor,
    dims: list[int],
    maxiter: int,
) -> Tensor:
    r"""Optimize the ``dims`` of every row of ``x`` on its single-point value.

    The objective is analytic LogEI (``Σᵢ wᵢ LogEIᵢ`` for a
    :class:`WeightedSumAcquisition`) when every member is a qLogEI on the fast
    path, and ``acq_function`` at ``q = 1`` otherwise. Rows are independent,
    so one L-BFGS-B run optimizes all of them.

    Args:
        acq_function: Acquisition function of the batch.
        x: ``(K, d)`` start rows.
        bounds: ``(2, d)`` bounds.
        dims: Continuous dims to optimize.
        maxiter: L-BFGS-B iterations.

    Returns:
        ``(K, d)`` refined rows (other dims unchanged).
    """
    from botorch.acquisition import LogExpectedImprovement
    from scipy.optimize import minimize as scipy_minimize

    members, weights = (
        (list(acq_function.acq_functions), acq_function.weights)
        if isinstance(acq_function, WeightedSumAcquisition)
        else ([acq_function], torch.ones(1))
    )
    if all(_supports_pool_fast_path(m) for m in members):
        analytic = [LogExpectedImprovement(model=m.model, best_f=m.best_f) for m in members]

        def objective(xx: Tensor) -> Tensor:
            return sum(w * a(xx.unsqueeze(1)) for w, a in zip(weights.to(xx), analytic))
    else:

        def objective(xx: Tensor) -> Tensor:
            return acq_function(xx.unsqueeze(1))

    tkwargs = {"dtype": x.dtype, "device": x.device}
    k = x.shape[0]
    lo, hi = bounds[0, dims].repeat(k), bounds[1, dims].repeat(k)

    def unpack(v: Tensor) -> Tensor:
        out = x.clone()
        out[:, dims] = v.reshape(k, len(dims))
        return out

    def fun(v_np: np.ndarray) -> tuple[float, np.ndarray]:
        v = torch.as_tensor(v_np, **tkwargs).requires_grad_(True)
        loss = -objective(unpack(v)).sum()
        (grad,) = torch.autograd.grad(loss, v)
        return float(loss), grad.detach().cpu().numpy()

    res = scipy_minimize(
        fun,
        x[:, dims].reshape(-1).detach().cpu().numpy(),
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lo.cpu().numpy(), hi.cpu().numpy())),
        options={"maxiter": maxiter},
    )
    return unpack(torch.as_tensor(res.x, **tkwargs)).detach()


def optimize_acqf_mixed_pool_greedy(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    cat_emb_avail: Tensor,
    sol_emb_avail: Tensor,
    bas_emb_avail: Tensor,
    cat_is_salt_flags: Tensor | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    pre_batch: Tensor | None = None,
    warm_start: Tensor | None = None,
    n_pool: int = 1024,
    n_override: int = 256,
    n_local: int = 256,
    n_incumbents: int = 8,
    incumbents: Tensor | None = None,
    local_scale: float = 0.1,
    n_refine: int = 32,
    refine_maxiter: int = 30,
    n_swap_rounds: int = 1,
    polish_maxiter: int = 50,
    n_mc: int | None = None,
    n_mc_screen: int | None = 32,
    n_keep: int = 64,
    posterior_cache: GPPoolPosteriorCache | None = None,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    r"""Greedy batch construction over a shared candidate pool, then polish.

    1. **Pool.** ``n_pool`` rows: ``(cat, sol, bas)`` drawn uniformly from the
       component tables, free continuous dims from a scrambled Sobol sequence,
       ``fixed_dims`` pinned; ``warm_start`` rows and ``n_local`` rows around
       the best training points of the model are appended. The random rows
       depend only on ``seed`` and the tables, so calls that share them share
       those rows (and, with ``posterior_cache``, their GP factors).
    2. **Refine.** The ``n_refine`` rows with the best single-point value get
       their free continuous dims optimized on the single-point acquisition
       and join the pool (it catches narrow high-value regions a random pool
       misses).
    3. **Greedy.** Each step scores ``batch ∪ {x}`` for every pool row ``x``
       and keeps the best. On the grouped dim of ``batch_constraints`` a point
       either opens a new group (its pool value) or joins an open group; for
       joining, the ``n_override`` best rows of the last free-pool screen get
       the group's value on that dim. The partition is reached exactly.
    4. **Swaps.** ``n_swap_rounds`` passes re-pick each free slot with the
       other slots fixed.
    5. **Polish.** L-BFGS-B on ``acq_function`` over the free continuous dims
       with the discrete part and the grouping fixed (one variable per group on
       the grouped dim); kept only if it raises the evaluator's value.

    Steps 3–4 use :class:`_PoolAcquisitionEvaluator` (exact joint posteriors
    and fixed QMC base samples for qLogEI on a single-output exact GP, or a
    :class:`WeightedSumAcquisition` of such; direct calls otherwise).

    Args:
        acq_function: Acquisition function over ``(b, q', d)`` inputs.
        bounds: ``(2, d)`` bounds; the first ``d - d_disc`` dims are continuous.
        q: Batch size including ``pre_batch``.
        cat_emb_avail: ``(n_cat, d_cat)`` available catalyst rows.
        sol_emb_avail: ``(n_sol, d_sol)`` available solvent rows.
        bas_emb_avail: ``(n_bas, d_bas)`` available base rows.
        cat_is_salt_flags: ``(n_cat,)`` bool; the base slice is zeroed where False.
        batch_constraints: At most one continuous dim mapped to an explicit
            partition of ``q`` (e.g. ``{0: (2, 2, 2)}``) or a group cap ``k``.
        fixed_dims: Continuous dims pinned to constants.
        pre_batch: ``(q_pre, d)`` points fixed at the head of the batch; their
            values on the grouped dim count toward the partition.
        warm_start: ``(m, d)`` extra pool rows (e.g. a parent batch); their
            component slices must come from the tables.
        n_pool: Random pool rows.
        n_override: Rows scored per open group when joining it.
        n_local: Pool rows around the ``n_incumbents`` training points with the
            largest targets: free continuous dims perturbed by
            ``local_scale ×`` their range (Gaussian, clipped), discrete part
            the incumbent's (where available) for half of the rows and random
            for the rest. ``0`` disables them.
        n_incumbents: Incumbents the local rows are drawn around.
        incumbents: ``(k, d)`` rows in the candidate layout, best first, whose
            first ``n_incumbents`` replace the training points as incumbents.
            Needed when the model's inputs are not in the candidate layout
            (e.g. an acquisition function that remaps candidates).
        local_scale: Perturbation scale of the local rows.
        n_refine: Pool rows with the best single-point values whose free
            continuous dims are optimized (L-BFGS-B, ``refine_maxiter``
            iterations, see :func:`_refine_single_points`) and appended to the
            pool before the greedy step. ``0`` disables it.
        refine_maxiter: L-BFGS-B iterations of that refinement.
        n_swap_rounds: Swap passes after the greedy construction.
        polish_maxiter: L-BFGS-B iterations; ``0`` skips the polish.
        n_mc: QMC samples of the fast path; defaults to the sampler size of
            ``acq_function`` (512 without one).
        n_mc_screen: Fast path: samples of the screening pass over all rows;
            ``None`` scores every row on all ``n_mc`` samples.
        n_keep: Fast path: rows rescored on all ``n_mc`` samples.
        posterior_cache: Root GP factors shared across calls on models
            conditioned on the root (fantasy trees).
        seed: Seed of the pool and of the fast-path base samples.

    Returns:
        A two-element tuple:

        - **candidates** (*Tensor*): ``(q, d)`` batch, ``pre_batch`` first.
        - **acq_value** (*Tensor*): Scalar ``acq_function`` value of the batch.

    Raises:
        ValueError: If the constraints, ``pre_batch`` or ``warm_start`` are infeasible.
    """
    from scipy.optimize import minimize as scipy_minimize
    from torch.quasirandom import SobolEngine

    tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
    d = bounds.shape[-1]
    d_cat, d_sol, d_bas = cat_emb_avail.shape[1], sol_emb_avail.shape[1], bas_emb_avail.shape[1]
    d_cont = d - d_cat - d_sol - d_bas
    fixed_dims = dict(fixed_dims or {})
    q_pre = 0 if pre_batch is None else pre_batch.shape[0]
    if q - q_pre < 1:
        raise ValueError(f"q ({q}) must exceed the pre_batch size ({q_pre})")

    constraints = dict(batch_constraints or {})
    if len(constraints) > 1:
        raise ValueError("optimize_acqf_mixed_pool_greedy supports one constrained dim")
    group_dim, partition, max_groups = None, None, None
    if constraints:
        group_dim, spec = next(iter(constraints.items()))
        if group_dim >= d_cont or group_dim in fixed_dims:
            raise ValueError(f"Constrained dim {group_dim} must be a free continuous dim")
        if isinstance(spec, Integral) and not isinstance(spec, bool):
            max_groups = int(spec)
        else:
            partition = tuple(sorted(_normalize_partition_spec(spec, q, group_dim), reverse=True))
    free_cont = [i for i in range(d_cont) if i not in fixed_dims and i != group_dim]

    salt = (
        torch.ones(cat_emb_avail.shape[0], dtype=torch.bool)
        if cat_is_salt_flags is None
        else cat_is_salt_flags.to(dtype=torch.bool, device="cpu")
    )

    def assemble(cont: Tensor, idx: Tensor) -> Tensor:
        bas = bas_emb_avail[idx[:, 2]] * salt[idx[:, 0]].to(**tkwargs).unsqueeze(-1)
        return torch.cat([cont, cat_emb_avail[idx[:, 0]], sol_emb_avail[idx[:, 1]], bas], dim=-1)

    generator = torch.Generator().manual_seed(seed)
    idx = _pool_components(n_pool, cat_emb_avail.shape[0], sol_emb_avail.shape[0], bas_emb_avail.shape[0], generator)
    sobol_dims = [i for i in range(d_cont) if i not in fixed_dims]
    cont = torch.empty(n_pool, d_cont, **tkwargs)
    if sobol_dims:
        u = SobolEngine(len(sobol_dims), scramble=True, seed=seed).draw(n_pool).to(**tkwargs)
        cont[:, sobol_dims] = bounds[0, sobol_dims] + (bounds[1, sobol_dims] - bounds[0, sobol_dims]) * u
    for dim, value in fixed_dims.items():
        cont[:, dim] = value
    pool = assemble(cont, idx)  # (n_pool, d)
    if warm_start is not None and warm_start.shape[0] > 0:
        o = d_cont
        cat_idx = _match_component_rows(warm_start[:, o:o + d_cat], cat_emb_avail)
        sol_idx = _match_component_rows(warm_start[:, o + d_cat:o + d_cat + d_sol], sol_emb_avail)
        bas_idx = torch.zeros_like(cat_idx)
        is_salt = salt[cat_idx]
        if bool(is_salt.any()):
            bas_idx[is_salt] = _match_component_rows(warm_start[is_salt, o + d_cat + d_sol:], bas_emb_avail)
        warm_cont = warm_start[:, :d_cont].clone()
        for dim, value in fixed_dims.items():
            warm_cont[:, dim] = value
        warm = assemble(warm_cont, torch.stack([cat_idx, sol_idx, bas_idx], dim=-1))
    else:
        warm = torch.empty(0, d, **tkwargs)
    if n_local > 0:
        if incumbents is None:
            member = acq_function.acq_functions[0] if isinstance(acq_function, WeightedSumAcquisition) else acq_function
            model = member.model
            x_train = model.train_inputs[0]
            if getattr(model, "input_transform", None) is not None:
                x_train = model.input_transform.untransform(x_train)
            top = model.train_targets.topk(min(n_incumbents, x_train.shape[0])).indices
            centres = x_train[top].to(**tkwargs)  # (k, d)
        else:
            if incumbents.ndim != 2 or incumbents.shape[-1] != d or incumbents.shape[0] == 0:
                raise ValueError(f"incumbents must have shape (k >= 1, {d}), got {tuple(incumbents.shape)}")
            centres = incumbents[:n_incumbents].to(**tkwargs)  # (k, d)
        rows = centres[torch.arange(n_local) % centres.shape[0]]  # (n_local, d)
        local_cont = rows[:, :d_cont].clone()
        if sobol_dims:
            span = bounds[1, sobol_dims] - bounds[0, sobol_dims]
            noise = torch.randn(n_local, len(sobol_dims), generator=generator, dtype=bounds.dtype).to(bounds.device)
            local_cont[:, sobol_dims] = torch.minimum(
                torch.maximum(local_cont[:, sobol_dims] + local_scale * span * noise, bounds[0, sobol_dims]),
                bounds[1, sobol_dims],
            )
        for dim, value in fixed_dims.items():
            local_cont[:, dim] = value
        local_idx = _pool_components(n_local, cat_emb_avail.shape[0], sol_emb_avail.shape[0], bas_emb_avail.shape[0], generator)
        keep_discrete = torch.rand(n_local, generator=generator) < 0.5
        o = d_cont
        for col, (lo_c, hi_c, table) in enumerate(
            ((o, o + d_cat, cat_emb_avail), (o + d_cat, o + d_cat + d_sol, sol_emb_avail), (o + d_cat + d_sol, d, bas_emb_avail))
        ):
            hits = torch.isclose(rows[:, lo_c:hi_c].unsqueeze(1), table.unsqueeze(0), atol=1e-12, rtol=0.0).all(-1)
            use = keep_discrete & hits.any(dim=1)
            local_idx[use, col] = hits[use].float().argmax(dim=1)
        warm = torch.cat([warm, assemble(local_cont, local_idx)], dim=0)
    n_random = pool.shape[0]
    pool = torch.cat([pool, warm], dim=0)

    evaluator = _PoolAcquisitionEvaluator(
        acq_function, q=q, n_mc=n_mc, seed=seed, n_mc_screen=n_mc_screen,
        n_keep=n_keep, posterior_cache=posterior_cache,
    )

    batch = pre_batch.clone() if pre_batch is not None else torch.empty(0, d, **tkwargs)
    group_values: list[float] = []
    slot_group: list[int] = []
    if group_dim is not None:
        for row in batch:
            value = float(row[group_dim])
            matches = [g for g, v in enumerate(group_values) if abs(v - value) < 1e-9]
            if matches:
                slot_group.append(matches[0])
            else:
                group_values.append(value)
                slot_group.append(len(group_values) - 1)
        counts = [slot_group.count(g) for g in range(len(group_values))]
        if partition is not None and not (
            len(counts) <= len(partition)
            and all(a <= b for a, b in zip(sorted(counts, reverse=True), partition))
        ):
            raise ValueError(f"pre_batch grouping {counts} cannot complete partition {partition}")

    if n_refine > 0 and sobol_dims:
        empty = torch.empty(0, d, **tkwargs)
        _, screen_random = evaluator.batch_values(empty, pool[:n_random], cache="root")
        screens = [screen_random]
        if pool.shape[0] > n_random:
            screens.append(evaluator.batch_values(empty, pool[n_random:], cache="call")[1])
        top = torch.cat(screens).topk(min(n_refine, pool.shape[0])).indices
        refined = _refine_single_points(
            acq_function, pool[top], bounds, sobol_dims, refine_maxiter
        )
        pool = torch.cat([pool, refined], dim=0)

    ranking = {"screen": None}

    def best_choice(fixed: Tensor, slots: list[int]) -> tuple[float, Tensor, int]:
        """Best next point ``(value, row, group)``; group ``-1`` opens a new one."""
        choices: list[tuple[Tensor | None, int, bool]] = []
        if group_dim is None:
            choices.append((pool, -1, True))
        else:
            counts = [slots.count(g) for g in range(len(group_values))]
            live = [g for g in range(len(group_values)) if counts[g] > 0]
            joinable, can_open = _pool_group_options([counts[g] for g in live], partition, max_groups)
            if can_open or ranking["screen"] is None:
                choices.append((pool, -1, can_open))
            choices.extend((None, live[j], True) for j in joinable)
        best = (-float("inf"), None, -1)
        for cand_pool, g, selectable in choices:
            shared = cand_pool is not None
            if cand_pool is None:
                top = ranking["screen"].topk(min(n_override, pool.shape[0])).indices
                cand_pool = pool[top].clone()
                cand_pool[:, group_dim] = group_values[g]
            if shared:
                values, screen = evaluator.batch_values(fixed, cand_pool[:n_random], cache="root")
                if cand_pool.shape[0] > n_random:
                    v_warm, s_warm = evaluator.batch_values(fixed, cand_pool[n_random:], cache="call")
                    values, screen = torch.cat([values, v_warm]), torch.cat([screen, s_warm])
            else:
                values, screen = evaluator.batch_values(fixed, cand_pool, cache="none")
            if shared:
                ranking["screen"] = screen
            j = int(values.argmax())
            if selectable and float(values[j]) > best[0]:
                best = (float(values[j]), cand_pool[j], g)
        return best

    def place(g: int, row: Tensor) -> int:
        if group_dim is None:
            return -1
        if g >= 0:
            return g
        group_values.append(float(row[group_dim]))
        return len(group_values) - 1

    for _ in range(q - q_pre):
        _, row, g = best_choice(batch, slot_group)
        slot_group.append(place(g, row))
        batch = torch.cat([batch, row.unsqueeze(0)], dim=0)

    for _ in range(n_swap_rounds):
        improved = False
        for i in range(q_pre, q):
            keep = [j for j in range(q) if j != i]
            current = float(evaluator.batch_values(batch[keep], batch[i:i + 1], cache="none")[0])
            slots_wo = [slot_group[j] for j in keep] if group_dim is not None else []
            value, row, g = best_choice(batch[keep], slots_wo)
            if value > current + 1e-9:
                batch = torch.cat([batch[:i], row.unsqueeze(0), batch[i + 1:]], dim=0)
                if group_dim is not None:
                    slot_group[i] = place(g, row)
                improved = True
        if not improved:
            break

    with torch.no_grad():
        value = acq_function(batch.unsqueeze(0)).reshape(())
    free_slots = list(range(q_pre, q))
    groups_free = (
        sorted({slot_group[i] for i in free_slots} - {slot_group[i] for i in range(q_pre)})
        if group_dim is not None else []
    )
    n_point_vars = len(free_slots) * len(free_cont)
    if polish_maxiter > 0 and n_point_vars + len(groups_free) > 0:
        x0 = torch.cat(
            [batch[free_slots][:, free_cont].reshape(-1),
             torch.tensor([group_values[g] for g in groups_free], **tkwargs)]
        )
        lo = torch.cat([bounds[0, free_cont].repeat(len(free_slots)),
                        bounds[0, [group_dim] * len(groups_free)] if groups_free else bounds[0, :0]])
        hi = torch.cat([bounds[1, free_cont].repeat(len(free_slots)),
                        bounds[1, [group_dim] * len(groups_free)] if groups_free else bounds[1, :0]])
        slot_masks = {g: torch.tensor([slot_group[i] == g for i in free_slots]) for g in groups_free}

        def unpack(v: Tensor) -> Tensor:
            pts = batch[free_slots].clone()
            if free_cont:
                pts[:, free_cont] = v[:n_point_vars].reshape(len(free_slots), len(free_cont))
            for k, g in enumerate(groups_free):
                pts[slot_masks[g], group_dim] = v[n_point_vars + k]
            return torch.cat([batch[:q_pre], pts], dim=0)

        def fun(v_np: np.ndarray) -> tuple[float, np.ndarray]:
            v = torch.as_tensor(v_np, **tkwargs).requires_grad_(True)
            loss = -acq_function(unpack(v).unsqueeze(0)).sum()
            (grad,) = torch.autograd.grad(loss, v)
            return float(loss), grad.detach().cpu().numpy()

        res = scipy_minimize(
            fun,
            x0.detach().cpu().numpy(),
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lo.cpu().numpy(), hi.cpu().numpy())),
            options={"maxiter": polish_maxiter},
        )
        polished = unpack(torch.as_tensor(res.x, **tkwargs)).detach()

        def bounded(x: Tensor) -> float:
            return float(evaluator.batch_values(x[:-1], x[-1:], cache="none")[0])

        if bounded(polished) > bounded(batch):
            batch = polished
            with torch.no_grad():
                value = acq_function(batch.unsqueeze(0)).reshape(())
    return batch.detach(), value.detach()
