"""Scope expansion: which locked catalyst to order next.

An ordered catalyst becomes available a fixed number of BO iterations later.
The expanders here score locked catalysts either on the current model
(myopic) or on a tree of fantasy models conditioned on simulated future
batches (fantasized). Every inner batch is built by a batch optimizer from
:mod:`pablo.optimize` (``optimize_acqf_mixed_pool_greedy`` by default); the
fantasized expander shares one :class:`~pablo.optimize.GPPoolPosteriorCache`
across the whole tree, since every node is the root conditioned on its path.
"""

from typing import Callable

import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from pablo.optimize import (
    BatchConstraints,
    GPPoolPosteriorCache,
    WeightedSumAcquisition,
    _build_step_batch_constraints,
    _normalize_batch_constraints,
    optimize_acqf_mixed_pool_greedy,
    release_gp_caches,
)
from pablo.utils.acq_utils import best_observed_f

OPERATING_MODES = ("only_terminal", "averaged", "layer_averaged")


def _expansion_batch_constraints(
    batch_constraints: BatchConstraints | None,
    q_expand: int,
) -> BatchConstraints | None:
    """Partial batch constraints for the ``q_expand`` expansion candidates.

    Explicit partitions are sized for the full batch; the expansion step only
    places ``q_expand`` of its points, so it gets the partial partition (none
    for ``q_expand <= len(partition)``).
    """
    _, explicit_partitions = _normalize_batch_constraints(batch_constraints)
    return _build_step_batch_constraints(batch_constraints, explicit_partitions, q_expand)


def _aggregation_weights(layer_sizes: list[int], operating_mode: str) -> Tensor:
    """Per-node weights of the terminal aggregate, nodes ordered layer by layer."""
    if operating_mode == "only_terminal":
        weights = [0.0] * sum(layer_sizes[:-1]) + [1.0 / layer_sizes[-1]] * layer_sizes[-1]
    elif operating_mode == "averaged":
        weights = [1.0 / sum(layer_sizes)] * sum(layer_sizes)
    else:
        weights = [w for size in layer_sizes for w in [1.0 / (len(layer_sizes) * size)] * size]
    return torch.tensor(weights, dtype=torch.float64)


ComponentTables = tuple[Tensor, Tensor, Tensor, Tensor | None]
"""``(cat_emb, sol_emb, bas_emb, cat_is_salt)`` of one allowed set: ``(n_cat, d_cat)``,
``(n_sol, d_sol)``, ``(n_bas, d_bas)`` rows and ``(n_cat,)`` bool salt flags (``None``:
every catalyst takes a base)."""


def _node_acqf(model: Model, node_mc_samples: int) -> qLogExpectedImprovement:
    """qLogEI of one tree node against its own incumbent (fantasies included)."""
    return qLogExpectedImprovement(
        model=model,
        best_f=best_observed_f(model),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([node_mc_samples])),
    )


def build_fantasy_tree(
    model: Model,
    q_batch: int,
    tables_per_depth: list[ComponentTables],
    bounds: Tensor,
    num_fantasies: list[int],
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    node_mc_samples: int = 128,
    node_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_pool_greedy,
) -> tuple[list[list[Model]], GPPoolPosteriorCache]:
    r"""Fantasy tree of simulated future batches.

    Depth ``j`` simulates the batch ``j + 1`` iterations ahead: every active
    node gets a ``q_batch`` batch from ``node_optimizer`` over
    ``tables_per_depth[j]``, then ``num_fantasies[j]`` children, each the node
    conditioned on one Sobol posterior sample at that batch (no hyperparameter
    refit, so every node is the root conditioned on its path). A child's
    optimizer call gets its parent's batch as ``warm_start``; all calls share
    one root :class:`GPPoolPosteriorCache` and, per depth, one pool seed. Once
    its children exist, a non-root node's GPyTorch prediction caches are
    released (rebuilt on its next posterior call), so the tree does not hold
    one set of dense ``n x n`` caches per node.

    Args:
        model: Fitted single-output GP (the root).
        q_batch: Batch size of every simulated batch.
        tables_per_depth: ``k`` component tables, the set usable at each depth.
        bounds: ``(2, d)`` bounds.
        num_fantasies: Children per node at each depth, length ``k``.
        batch_constraints: Grouped constraints of a ``q_batch`` batch.
        fixed_dims: Continuous dims pinned to constants.
        node_mc_samples: QMC samples of every node's qLogEI.
        node_optimizer: Batch optimizer with the
            :func:`optimize_acqf_mixed_pool_greedy` keyword interface.

    Returns:
        A two-element tuple:

        - **models_by_layer** (*list[list[Model]]*): ``k`` layers; layer ``j``
          holds the ``Π_{i ≤ j} num_fantasies[i]`` models conditioned on the
          batches of depths ``0..j``.
        - **cache** (*GPPoolPosteriorCache*): The root cache shared by the tree.

    Raises:
        ValueError: If ``num_fantasies`` and ``tables_per_depth`` differ in
            length or a fantasy count is not positive.
    """
    if len(num_fantasies) != len(tables_per_depth) or any(nf <= 0 for nf in num_fantasies):
        raise ValueError(
            f"num_fantasies {num_fantasies} must be {len(tables_per_depth)} positive integers"
        )
    cache = GPPoolPosteriorCache(model)
    active: list[tuple[Model, Tensor | None]] = [(model, None)]
    models_by_layer: list[list[Model]] = []
    for j, (cat_emb, sol_emb, bas_emb, cat_is_salt) in enumerate(tables_per_depth):
        children: list[tuple[Model, Tensor]] = []
        for node, parent_batch in active:
            node.eval()
            batch, _ = node_optimizer(
                acq_function=_node_acqf(node, node_mc_samples),
                bounds=bounds,
                q=q_batch,
                cat_emb_avail=cat_emb,
                sol_emb_avail=sol_emb,
                bas_emb_avail=bas_emb,
                cat_is_salt_flags=cat_is_salt,
                batch_constraints=batch_constraints,
                fixed_dims=fixed_dims,
                warm_start=parent_batch,
                posterior_cache=cache,
                seed=j,
            )
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies[j]]))
            with torch.no_grad():
                y_samples = sampler(node.posterior(batch, observation_noise=False))  # (f_j, q, 1)
            for branch in range(num_fantasies[j]):
                child = node.condition_on_observations(X=batch, Y=y_samples[branch])
                child.eval()
                children.append((child, batch))
            if parent_batch is not None:
                release_gp_caches(node)
        active = children
        models_by_layer.append([child for child, _ in children])
    return models_by_layer, cache


def fantasized_criterion(
    models_by_layer: list[list[Model]],
    operating_mode: str,
    node_mc_samples: int = 128,
    release_member_caches: bool = False,
) -> WeightedSumAcquisition:
    r"""Terminal acquisition of a fantasy tree.

    A :class:`WeightedSumAcquisition` of the nodes' qLogEI:

    - ``"only_terminal"``: mean over the deepest layer;
    - ``"averaged"``: mean over all nodes;
    - ``"layer_averaged"``: mean over layers of the per-layer means.

    Args:
        models_by_layer: Tree layers from :func:`build_fantasy_tree`.
        operating_mode: One of ``OPERATING_MODES``.
        node_mc_samples: QMC samples of every member qLogEI.
        release_member_caches: Passed to :class:`WeightedSumAcquisition`: a
            forward call releases each member's GP caches after evaluating it.

    Returns:
        The weighted sum over the nodes with a positive weight.

    Raises:
        ValueError: If ``operating_mode`` is unknown.
    """
    if operating_mode not in OPERATING_MODES:
        raise ValueError(f"operating_mode must be one of {OPERATING_MODES}, got {operating_mode!r}")
    nodes = [m for layer in models_by_layer for m in layer]
    weights = _aggregation_weights([len(layer) for layer in models_by_layer], operating_mode)
    keep = weights > 0
    return WeightedSumAcquisition(
        [_node_acqf(m, node_mc_samples) for m, w in zip(nodes, keep) if w],
        weights[keep],
        release_member_caches=release_member_caches,
    )


def optimize_scope_expansion_fantasized(
    model: Model,
    q_batch: int,
    q_expand: int,
    x_available: Tensor,
    x_ordered: list[Tensor],
    x_unordered: Tensor,
    bounds: Tensor,
    decompose_allowed_set: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]],
    k: int = 4,
    num_fantasies: list[int] | None = None,
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    operating_mode: str = "layer_averaged",
    node_mc_samples: int = 128,
    node_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_pool_greedy,
    expansion_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_pool_greedy,
    completion_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_pool_greedy,
) -> tuple[Tensor, Tensor]:
    r"""Fantasized scope expansion with a sequential two-stage terminal step.

    Depth ``j = 0..k-1`` simulates the batch ``j + 1`` iterations ahead: every
    node model gets a ``q_batch`` batch from ``node_optimizer`` over the set
    available at that depth (``x_available`` plus ``x_ordered[0..j]``), then
    ``num_fantasies[j]``
    children, each the node conditioned on one Sobol posterior sample at that
    batch (no hyperparameter refit, so every node is the root conditioned on
    its path). Each child's optimizer call gets its parent's batch as
    ``warm_start``; all calls share one root :class:`GPPoolPosteriorCache`
    and, per depth, one pool seed.

    The terminal acquisition is a :class:`WeightedSumAcquisition` of the
    nodes' qLogEI according to ``operating_mode``:

    - ``"only_terminal"``: mean over the depth-``k`` nodes;
    - ``"averaged"``: mean over all nodes of depths ``1..k``;
    - ``"layer_averaged"``: mean over depths of the per-depth means.

    The terminal step is the batch ``k + 1`` iterations ahead, the first one
    in which a row ordered now is usable when the dead time is ``k`` batches.
    Stage 1 picks ``q_expand`` expansion points from ``x_unordered`` on it with
    ``expansion_optimizer``; stage 2 completes ``q_batch`` from the terminal
    available set (``x_available`` plus every ``x_ordered`` entry) around them
    (``pre_batch``) with ``completion_optimizer`` and returns its value.
    Stage 2 does not affect the choice.

    Args:
        model: Fitted single-output GP (the root).
        q_batch: Batch size of every simulated batch.
        q_expand: Expansion points picked from ``x_unordered``.
        x_available: ``(n, d_disc)`` discrete rows usable in the next batch.
        x_ordered: At most ``k`` tensors; ``x_ordered[j]`` is ``(n_j, d_disc)``,
            the ordered rows that become usable at depth ``j`` (the batch
            ``j + 1`` iterations ahead) and stay usable afterwards.
        x_unordered: ``(n_unordered, d_disc)`` locked, not-yet-ordered rows.
        bounds: ``(2, d)`` bounds, ``d = d_cont + d_disc``.
        decompose_allowed_set: Splits an ``(n, d_disc)`` allowed set into the
            unique ``(cat_emb, sol_emb, bas_emb)`` component tables.
        k: Number of fantasy depths, ``k ≥ 1``.
        num_fantasies: Children per node at each depth, length ``k``.
        batch_constraints: Grouped constraints of a ``q_batch`` batch, e.g.
            ``{0: (2, 2, 2)}``.
        fixed_dims: Continuous dims pinned to constants.
        operating_mode: One of ``OPERATING_MODES``.
        node_mc_samples: QMC samples of every node's qLogEI (tree batches and
            terminal aggregate).
        node_optimizer: Batch optimizer of the tree nodes, with the
            :func:`optimize_acqf_mixed_pool_greedy` keyword interface.
        expansion_optimizer: Batch optimizer of stage 1.
        completion_optimizer: Batch optimizer of stage 2.

    Returns:
        A two-element tuple:

        - **candidates** (*Tensor*): ``(q_expand, d)`` expansion points.
        - **acq_value** (*Tensor*): Scalar terminal value of the completed batch.

    Raises:
        ValueError: If ``k``, ``q_batch``, ``q_expand``, ``num_fantasies`` or
            ``operating_mode`` are invalid, ``x_ordered`` has more than ``k``
            entries, or ``x_unordered`` is empty.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")
    if q_expand <= 0 or q_expand >= q_batch:
        raise ValueError(f"Need 0 < q_expand < q_batch, got q_expand={q_expand}, q_batch={q_batch}")
    if operating_mode not in OPERATING_MODES:
        raise ValueError(f"operating_mode must be one of {OPERATING_MODES}, got {operating_mode!r}")
    if num_fantasies is None or len(num_fantasies) != k or any(nf <= 0 for nf in num_fantasies):
        raise ValueError(f"num_fantasies must be {k} positive integers, got {num_fantasies}")
    if x_unordered.shape[0] == 0:
        raise ValueError("x_unordered must be non-empty")
    if len(x_ordered) > k:
        raise ValueError(f"x_ordered has {len(x_ordered)} entries, at most k={k} depths can use them")

    allowed_sets: list[Tensor] = []
    accumulated = x_available.clone()
    for j in range(k):
        if j < len(x_ordered):
            accumulated = torch.cat([accumulated, x_ordered[j]], dim=0)
        allowed_sets.append(accumulated.clone())
    terminal_available_set = accumulated

    models_by_layer, cache = build_fantasy_tree(
        model=model,
        q_batch=q_batch,
        tables_per_depth=[(*decompose_allowed_set(allowed), None) for allowed in allowed_sets],
        bounds=bounds,
        num_fantasies=num_fantasies,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        node_mc_samples=node_mc_samples,
        node_optimizer=node_optimizer,
    )
    terminal_acqf = fantasized_criterion(models_by_layer, operating_mode, node_mc_samples).to(bounds)

    cat_unord, sol_unord, bas_unord = decompose_allowed_set(x_unordered)
    expand_candidates, _ = expansion_optimizer(
        acq_function=terminal_acqf,
        bounds=bounds,
        q=q_expand,
        cat_emb_avail=cat_unord,
        sol_emb_avail=sol_unord,
        bas_emb_avail=bas_unord,
        batch_constraints=_expansion_batch_constraints(batch_constraints, q_expand),
        fixed_dims=fixed_dims,
        posterior_cache=cache,
        seed=k,
    )
    cat_term, sol_term, bas_term = decompose_allowed_set(terminal_available_set)
    _, terminal_value = completion_optimizer(
        acq_function=terminal_acqf,
        bounds=bounds,
        q=q_batch,
        cat_emb_avail=cat_term,
        sol_emb_avail=sol_term,
        bas_emb_avail=bas_term,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        pre_batch=expand_candidates,
        posterior_cache=cache,
        seed=k + 1,
    )
    return expand_candidates, terminal_value


def optimize_scope_expansion_myopic(
    model: Model,
    q_expand: int,
    x_unordered: Tensor,
    bounds: Tensor,
    decompose_allowed_set: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]],
    batch_constraints: BatchConstraints | None = None,
    fixed_dims: dict[int, float] | None = None,
    expansion_optimizer: Callable[..., tuple[Tensor, Tensor]] = optimize_acqf_mixed_pool_greedy,
) -> tuple[Tensor, Tensor]:
    r"""Myopic scope expansion: qLogEI of the current model over the locked rows.

    Args:
        model: Fitted single-output GP.
        q_expand: Expansion points picked from ``x_unordered``.
        x_unordered: ``(n_unordered, d_disc)`` locked, not-yet-ordered rows.
        bounds: ``(2, d)`` bounds, ``d = d_cont + d_disc``.
        decompose_allowed_set: Splits an ``(n, d_disc)`` allowed set into the
            unique ``(cat_emb, sol_emb, bas_emb)`` component tables.
        batch_constraints: Grouped constraints of a full batch; the expansion
            step gets the partial partition for ``q_expand`` points.
        fixed_dims: Continuous dims pinned to constants.
        expansion_optimizer: Batch optimizer with the
            :func:`optimize_acqf_mixed_pool_greedy` keyword interface (the same
            one the fantasized expander uses for its terminal stages).

    Returns:
        A two-element tuple:

        - **candidates** (*Tensor*): ``(q_expand, d)`` expansion points.
        - **acq_value** (*Tensor*): Their scalar qLogEI.
    """
    cat_un, sol_un, bas_un = decompose_allowed_set(x_unordered)
    return expansion_optimizer(
        acq_function=qLogExpectedImprovement(model=model, best_f=best_observed_f(model)),
        bounds=bounds,
        q=q_expand,
        cat_emb_avail=cat_un,
        sol_emb_avail=sol_un,
        bas_emb_avail=bas_un,
        batch_constraints=_expansion_batch_constraints(batch_constraints, q_expand),
        fixed_dims=fixed_dims,
        seed=0,
    )
