"""Core BO execution for the run_5 scope-expansion benchmark.

One experiment is a batch BO run on the NHC space of ``nhc_problem`` (q = 6,
temperature partition (2, 2, 2), time at 20 h) against a synthetic yield
landscape. It starts from every recorded experiment of ``nhc_doe.xlsx`` with
the catalysts available there; the other catalysts are locked. Every iteration
the batch generator picks six experiments from the available catalysts, the
measured yields are appended, the featurization-scan model is refit, and one
locked catalyst is ordered: it is dead for the next
``source_a_availability_delay`` batches and usable from the batch after them.

Every iteration, in every operating mode, scores the expansion candidates the
same way: a fantasy tree over the next ``lookahead_horizon`` batches is built
on the current model, and each criterion of ``EXPANSION_CRITERIA`` (myopic
qLogEI and the three tree aggregates) picks its best locked point. The random
expander picks a locked catalyst uniformly, its point being the best one under
the myopic criterion. All five picks are scored under all four criteria, and
the operating mode decides which pick is ordered.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    BASE_FEATURIZATION,
    BASE_KERNEL,
    BATCH_POOL_SIZE,
    BENCHMARK_CLASS_MAP,
    CAT_FEATURIZATION,
    DEFAULT_BATCH_CONSTRAINTS,
    EXPANSION_CRITERIA,
    EXPANSION_POOL_SIZE,
    EXPANSION_REFINE,
    FANTASIZED_OPERATING_MODES,
    FIXED_DIM_INDEX,
    FIXED_DIM_VALUE,
    KERNEL_STRUCTURE,
    LOOKAHEAD_HORIZON,
    MEAN_FUNCTION,
    MYOPIC_OPERATING_MODE,
    N_BO_BATCHES,
    N_CONTINUOUS_DIMS,
    NUM_FANTASIES,
    OUTCOME_TRANSFORM,
    PRIOR_SCHEME,
    Q_BATCH_SIZE,
    Q_EXPAND_SIZE,
    QMC_SAMPLES,
    RANDOM_OPERATING_MODE,
    SOLVENT_FEATURIZATION,
    SOURCE_A_AVAILABILITY_DELAY,
    TREE_POLISH_MAXITER,
    TREE_POOL_SIZE,
    TREE_QMC_SAMPLES,
    TREE_SWAP_ROUNDS,
    get_strategy_for_operating_mode,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.nhc_problem import (
    NHCSpace,
    SyntheticYield,
    fit_surrogate,
    load_nhc_space,
    yield_noise_sd,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.run_state import update_run_state
from pablo.optimize import BatchConstraints, optimize_acqf_mixed_pool_greedy
from pablo.scope_expansion import (
    ComponentTables,
    _expansion_batch_constraints,
    build_fantasy_tree,
    fantasized_criterion,
)
from pablo.utils.acq_utils import best_observed_f, get_joint_log_ei

EXPANSION_PICKS = (RANDOM_OPERATING_MODE, *EXPANSION_CRITERIA)
"""Names of the five expansion picks scored every iteration."""


def _release_matured_source_a(
    available_source_a_mask: Tensor,
    pending_unlock_iteration: Tensor,
    current_iteration: int,
) -> list[int]:
    """Release catalysts whose unlock iteration has arrived; returns their indices."""
    matured = torch.where(
        (pending_unlock_iteration >= 0) & (pending_unlock_iteration <= current_iteration)
    )[0]
    if matured.numel() == 0:
        return []
    available_source_a_mask[matured] = True
    pending_unlock_iteration[matured] = -1
    return [int(idx) for idx in matured.tolist()]


def _pick_random_unordered_source(
    available_source_a_mask: Tensor,
    pending_unlock_iteration: Tensor,
    seed: int,
) -> int:
    """Pick one locked, not-yet-ordered catalyst index uniformly at random."""
    unordered = torch.where((~available_source_a_mask) & (pending_unlock_iteration < 0))[0]
    if unordered.numel() == 0:
        raise ValueError("No unordered source-A indices left to pick from")
    generator = torch.Generator()
    generator.manual_seed(seed)
    pick = torch.randint(unordered.numel(), (1,), generator=generator).item()
    return int(unordered[pick].item())


def pool_greedy_settings() -> dict[str, dict[str, int]]:
    """Keyword settings of ``optimize_acqf_mixed_pool_greedy`` per role.

    Returns:
        Dict from role to keyword arguments: ``"batch"`` for the main q-batch,
        ``"node"`` for the fantasy-tree batches and ``"expansion"`` for every
        expansion pick.
    """
    return {
        "batch": {"n_pool": BATCH_POOL_SIZE},
        "node": {
            "n_pool": TREE_POOL_SIZE,
            "n_swap_rounds": TREE_SWAP_ROUNDS,
            "polish_maxiter": TREE_POLISH_MAXITER,
        },
        "expansion": {
            "n_pool": EXPANSION_POOL_SIZE,
            "n_refine": EXPANSION_REFINE,
            "n_swap_rounds": 0,
            "polish_maxiter": 0,
        },
    }


def _tables(space: NHCSpace, cat_mask: Tensor) -> ComponentTables:
    """Component tables of the catalysts in ``cat_mask`` ``(n_cat,)`` with every solvent and base."""
    return space.cat_table[cat_mask], space.sol_table, space.bas_table, space.cat_is_salt[cat_mask]


def _catalyst_index(point: Tensor, space: NHCSpace) -> int:
    """``cat_table`` row of the catalyst block of ``point`` ``(TOTAL_DIM,)``."""
    block = point[N_CONTINUOUS_DIMS:N_CONTINUOUS_DIMS + space.cat_table.shape[1]]
    matches = torch.where((space.cat_table == block.unsqueeze(0)).all(dim=1))[0]
    if matches.numel() != 1:
        raise ValueError(f"Point matches {matches.numel()} catalyst rows, expected 1")
    return int(matches.item())


def _batch_acquisition(model: SingleTaskGP, qmc_samples: int) -> qLogExpectedImprovement:
    """qLogEI of the batch generator against the best measured yield (model scale)."""
    return qLogExpectedImprovement(
        model=model,
        best_f=best_observed_f(model),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([qmc_samples])),
    )


def _expansion_step(
    model: SingleTaskGP,
    space: NHCSpace,
    available: Tensor,
    pending_unlock_iteration: Tensor,
    iteration: int,
    q: int,
    q_expand: int,
    lookahead_horizon: int,
    num_fantasies: list[int],
    batch_constraints: BatchConstraints,
    fixed_dims: dict[int, float],
    seed: int,
    optimizers: dict[str, functools.partial],
) -> tuple[dict[str, int], dict[str, dict[str, object]], dict[str, dict[str, float]], dict[str, float]]:
    """Every expander's pick for this iteration, scored under every criterion.

    Args:
        model: Model fitted on all measurements so far (the tree root).
        space: The NHC space.
        available: ``(n_cat,)`` bool, catalysts usable in the next batch.
        pending_unlock_iteration: ``(n_cat,)`` unlock iteration of ordered
            catalysts, −1 for the others.
        iteration: Index of the batch just evaluated.
        q: Batch size of the simulated batches.
        q_expand: Points per pick (1).
        lookahead_horizon: Tree depth ``k``; depth ``j`` simulates batch
            ``iteration + j + 1`` over the catalysts usable by then.
        num_fantasies: Children per node at each depth.
        batch_constraints: Grouped constraints of a ``q`` batch.
        fixed_dims: Pinned continuous dims.
        seed: Seed of the expansion pools and of the random expander's draw.
        optimizers: Pool-greedy partials by role.

    Returns:
        A four-element tuple:

        - **indices**: ``EXPANSION_PICKS`` name → ``cat_table`` row of its catalyst.
        - **picks**: ``EXPANSION_PICKS`` name → ``{"catalyst_index", "catalyst_id",
          "point"}``.
        - **scores**: pick name → criterion name → value of the pick's point
          (the criterion's forward value, all picks in one call per criterion).
        - **timing**: wall seconds of ``"tree"``, ``"picks"`` and ``"scores"``.

    Raises:
        ValueError: If ``q_expand != 1``.
    """
    if q_expand != 1:
        raise ValueError(f"Cross-scoring scores single points; q_expand must be 1, got {q_expand}")
    t0 = time.perf_counter()
    tables_per_depth = [
        _tables(
            space,
            available | ((pending_unlock_iteration >= 0) & (pending_unlock_iteration <= iteration + j + 1)),
        )
        for j in range(lookahead_horizon)
    ]
    models_by_layer, cache = build_fantasy_tree(
        model=model,
        q_batch=q,
        tables_per_depth=tables_per_depth,
        bounds=space.bounds,
        num_fantasies=num_fantasies,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        node_mc_samples=TREE_QMC_SAMPLES,
        node_optimizer=optimizers["node"],
    )
    criteria = {MYOPIC_OPERATING_MODE: qLogExpectedImprovement(model=model, best_f=best_observed_f(model))}
    for mode in FANTASIZED_OPERATING_MODES:
        criteria[mode] = fantasized_criterion(
            models_by_layer, mode, TREE_QMC_SAMPLES, release_member_caches=True
        ).to(space.bounds)
    t1 = time.perf_counter()

    unordered = (~available) & (pending_unlock_iteration < 0)
    cat_un, sol_un, bas_un, salt_un = _tables(space, unordered)
    expansion_constraints = _expansion_batch_constraints(batch_constraints, q_expand)
    points: dict[str, Tensor] = {}
    for name, acqf in criteria.items():
        candidate, _ = optimizers["expansion"](
            acq_function=acqf,
            bounds=space.bounds,
            q=q_expand,
            cat_emb_avail=cat_un,
            sol_emb_avail=sol_un,
            bas_emb_avail=bas_un,
            cat_is_salt_flags=salt_un,
            batch_constraints=expansion_constraints,
            fixed_dims=fixed_dims,
            posterior_cache=cache,
            seed=seed,
        )
        points[name] = candidate
    random_idx = _pick_random_unordered_source(available, pending_unlock_iteration, seed)
    only_random = torch.zeros_like(unordered)
    only_random[random_idx] = True
    cat_r, sol_r, bas_r, salt_r = _tables(space, only_random)
    points[RANDOM_OPERATING_MODE], _ = optimizers["expansion"](
        acq_function=criteria[MYOPIC_OPERATING_MODE],
        bounds=space.bounds,
        q=q_expand,
        cat_emb_avail=cat_r,
        sol_emb_avail=sol_r,
        bas_emb_avail=bas_r,
        cat_is_salt_flags=salt_r,
        batch_constraints=expansion_constraints,
        fixed_dims=fixed_dims,
        posterior_cache=cache,
        seed=seed,
    )
    t2 = time.perf_counter()

    indices: dict[str, int] = {}
    picks: dict[str, dict[str, object]] = {}
    for name in EXPANSION_PICKS:
        point = points[name][0]  # (d,)
        idx = _catalyst_index(point, space)
        if not bool(unordered[idx]):
            raise ValueError(f"Pick {name!r} chose catalyst row {idx}, which is not locked and unordered")
        indices[name] = idx
        picks[name] = {"catalyst_index": idx, "catalyst_id": space.cat_ids[idx], "point": point.tolist()}
    stacked = torch.stack([points[name] for name in EXPANSION_PICKS])  # (n_picks, 1, d)
    with torch.no_grad():
        values = {c: acqf(stacked) for c, acqf in criteria.items()}  # criterion -> (n_picks,)
    scores = {
        name: {c: float(v[i].item()) for c, v in values.items()} for i, name in enumerate(EXPANSION_PICKS)
    }
    timing = {"tree": t1 - t0, "picks": t2 - t1, "scores": time.perf_counter() - t2}
    return indices, picks, scores, timing


def run_bo_experiment(
    benchmark_name: str,
    seed: int,
    operating_mode: str,
    q: int = Q_BATCH_SIZE,
    q_expand: int = Q_EXPAND_SIZE,
    n_eval: int = N_BO_BATCHES,
    source_a_availability_delay: int = SOURCE_A_AVAILABILITY_DELAY,
    lookahead_horizon: int = LOOKAHEAD_HORIZON,
    num_fantasies: list[int] | None = None,
    batch_constraints: BatchConstraints | None = None,
    qmc_samples: int = QMC_SAMPLES,
    space: NHCSpace | None = None,
    run_state_path: Path | None = None,
    run_state_combination_index: int | None = None,
    log_label: str | None = None,
) -> dict[str, object]:
    """Execute one BO experiment with delayed catalyst availability.

    Pool seeds and the random expander's draw depend on ``seed``, the benchmark
    and the iteration, not on the operating mode, so the modes of one seed share
    their random-number stream.

    Args:
        benchmark_name: Key from ``BENCHMARK_CLASS_MAP``.
        seed: Landscape and noise seed; the same seed gives the same landscape,
            start-set measurements and random-number stream in every mode.
        operating_mode: One of ``ALL_OPERATING_MODES``; its pick is ordered.
        q: Batch size.
        q_expand: Catalysts ordered per iteration (1).
        n_eval: Number of BO iterations.
        source_a_availability_delay: Dead time in batches: a catalyst ordered
            after iteration ``t``'s batch cannot be used in batches
            ``t + 1 .. t + delay`` and is usable from batch ``t + delay + 1``.
        lookahead_horizon: Fantasy-tree depth.
        num_fantasies: Fantasies per depth; defaults to ``NUM_FANTASIES``.
        batch_constraints: Grouped constraints of the batch and the tree
            batches; defaults to ``DEFAULT_BATCH_CONSTRAINTS``.
        qmc_samples: QMC samples of the batch qLogEI.
        space: The NHC space; loaded from ``nhc_doe.xlsx`` when ``None``.
        run_state_path: Optional run-state file for queue monitoring.
        run_state_combination_index: Index written into the run state.
        log_label: When given, one line per iteration with this label is printed.

    Returns:
        Result dictionary (JSON-serializable) with the optimization trace.
        Yields are on the synthetic [0, 1] scale: ``best_fs`` is the best
        noise-free yield measured so far after each iteration (the best
        attainable is 1, ``global_max``).

    Raises:
        ValueError: If ``n_eval`` or ``lookahead_horizon`` is not positive,
            ``q_expand != 1``, ``source_a_availability_delay`` is negative,
            ``benchmark_name`` or ``operating_mode`` is unknown, or
            ``len(num_fantasies) != lookahead_horizon``.
    """
    if n_eval <= 0:
        raise ValueError("n_eval must be positive")
    if q_expand != 1:
        raise ValueError(f"q_expand must be 1, got {q_expand}")
    if source_a_availability_delay < 0:
        raise ValueError("source_a_availability_delay must be non-negative")
    if lookahead_horizon <= 0:
        raise ValueError("lookahead_horizon must be positive")
    if benchmark_name not in BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    strategy = get_strategy_for_operating_mode(operating_mode)
    active_num_fantasies = list(NUM_FANTASIES) if num_fantasies is None else list(num_fantasies)
    if len(active_num_fantasies) != lookahead_horizon:
        raise ValueError(
            f"num_fantasies length must equal lookahead_horizon ({lookahead_horizon}), "
            f"got {len(active_num_fantasies)}"
        )
    active_batch_constraints: BatchConstraints = (
        dict(DEFAULT_BATCH_CONSTRAINTS) if batch_constraints is None else dict(batch_constraints)
    )

    stream_seed = 1_000_003 * seed + list(BENCHMARK_CLASS_MAP).index(benchmark_name)
    torch.manual_seed(stream_seed)
    noise_generator = torch.Generator().manual_seed(stream_seed)

    space = load_nhc_space() if space is None else space
    landscape = SyntheticYield(benchmark_name, seed, space)
    fixed_dims = {FIXED_DIM_INDEX: FIXED_DIM_VALUE}
    available = space.initial_available.clone()
    pending_unlock_iteration = torch.full_like(available, -1, dtype=torch.long)

    train_x = space.start_x.clone()
    train_y = landscape.observe(train_x, noise_generator)
    true_y = landscape.yield_true(train_x)
    start_y_observed = train_y.squeeze(-1).tolist()
    start_y_true = true_y.squeeze(-1).tolist()

    optimizers = {
        role: functools.partial(optimize_acqf_mixed_pool_greedy, **kwargs)
        for role, kwargs in pool_greedy_settings().items()
    }
    t_fit = time.perf_counter()
    model = fit_surrogate(train_x, train_y, space.feature_dims)
    initial_fit_time = time.perf_counter() - t_fit

    def _emit_phase_status(completed: int, phase: str) -> None:
        if run_state_path is not None and run_state_combination_index is not None:
            update_run_state(
                state_path=run_state_path,
                combination_index=run_state_combination_index,
                total_iterations=n_eval,
                status="running",
                current_iteration=completed,
                best_f=float(true_y.max().item()),
                active_phase=phase,
            )

    _emit_phase_status(0, "ready_for_iteration_1")

    best_fs: list[float] = []
    best_observed: list[float] = []
    candidates_list: list[list[list[float]]] = []
    y_observed_list: list[list[float]] = []
    y_true_list: list[list[float]] = []
    max_acq_values_list: list[float] = []
    selected_source_a_indices: list[int | None] = []
    expansion_picks: list[dict[str, dict[str, object]] | None] = []
    expansion_scores: list[dict[str, dict[str, float]] | None] = []
    released_source_a_by_iteration: list[list[int]] = []
    available_source_counts_before_selection: list[int] = []
    pending_source_counts_before_selection: list[int] = []
    n_candidates_using_released: list[int] = []
    wall_times_acqf_opt: list[float] = []
    wall_times_model_fit: list[float] = []
    wall_times_expander: list[float] = []
    wall_times_expander_parts: list[dict[str, float] | None] = []

    for iteration in range(n_eval):
        _emit_phase_status(iteration, f"iteration_{iteration + 1}_acq_opt_start")
        released_now = _release_matured_source_a(available, pending_unlock_iteration, iteration)
        released_source_a_by_iteration.append(released_now)
        available_source_counts_before_selection.append(int(available.sum().item()))
        pending_source_counts_before_selection.append(int((pending_unlock_iteration >= 0).sum().item()))

        t_start = time.perf_counter()
        cat_av, sol_av, bas_av, salt_av = _tables(space, available)
        chosen, _ = optimizers["batch"](
            acq_function=_batch_acquisition(model, qmc_samples),
            bounds=space.bounds,
            q=q,
            seed=stream_seed + iteration,
            cat_emb_avail=cat_av,
            sol_emb_avail=sol_av,
            bas_emb_avail=bas_av,
            cat_is_salt_flags=salt_av,
            batch_constraints=active_batch_constraints,
            fixed_dims=fixed_dims,
        )
        max_acq_values_list.append(float(get_joint_log_ei(model, best_observed_f(model), chosen)))
        wall_times_acqf_opt.append(time.perf_counter() - t_start)
        n_candidates_using_released.append(
            sum(_catalyst_index(row, space) in released_now for row in chosen)
        )

        _emit_phase_status(iteration, f"iteration_{iteration + 1}_objective_evaluation")
        y_new = landscape.observe(chosen, noise_generator)
        y_new_true = landscape.yield_true(chosen)
        train_x = torch.cat([train_x, chosen])
        train_y = torch.cat([train_y, y_new])
        true_y = torch.cat([true_y, y_new_true])

        _emit_phase_status(iteration, f"iteration_{iteration + 1}_model_refit")
        t_start = time.perf_counter()
        model = fit_surrogate(train_x, train_y, space.feature_dims)
        wall_times_model_fit.append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        selected: int | None = None
        picks, scores, parts = None, None, None
        if bool(((~available) & (pending_unlock_iteration < 0)).any()):
            _emit_phase_status(iteration, f"iteration_{iteration + 1}_expand_optimizer_start")
            indices, picks, scores, parts = _expansion_step(
                model=model,
                space=space,
                available=available,
                pending_unlock_iteration=pending_unlock_iteration,
                iteration=iteration,
                q=q,
                q_expand=q_expand,
                lookahead_horizon=lookahead_horizon,
                num_fantasies=active_num_fantasies,
                batch_constraints=active_batch_constraints,
                fixed_dims=fixed_dims,
                seed=stream_seed + iteration,
                optimizers=optimizers,
            )
            selected = indices[operating_mode]
            pending_unlock_iteration[selected] = iteration + source_a_availability_delay + 1
        wall_times_expander.append(time.perf_counter() - t_start)
        wall_times_expander_parts.append(parts)

        selected_source_a_indices.append(selected)
        expansion_picks.append(picks)
        expansion_scores.append(scores)
        best_fs.append(float(true_y.max().item()))
        best_observed.append(float(train_y.max().item()))
        candidates_list.append(chosen.tolist())
        y_observed_list.append(y_new.squeeze(-1).tolist())
        y_true_list.append(y_new_true.squeeze(-1).tolist())
        _emit_phase_status(iteration + 1, f"iteration_{iteration + 1}_complete")
        if log_label is not None:
            print(
                f"{log_label} iteration {iteration + 1}/{n_eval}: best_f={best_fs[-1]:.4f} "
                f"fit={wall_times_model_fit[-1]:.1f}s batch={wall_times_acqf_opt[-1]:.1f}s "
                f"expander={wall_times_expander[-1]:.1f}s",
                flush=True,
            )

    return {
        "benchmark": benchmark_name,
        "seed": seed,
        "strategy": strategy,
        "operating_mode": operating_mode,
        "model": {
            "cat_featurization": CAT_FEATURIZATION,
            "solvent_featurization": SOLVENT_FEATURIZATION,
            "base_featurization": BASE_FEATURIZATION,
            "base_kernel": BASE_KERNEL,
            "outcome_transform": OUTCOME_TRANSFORM,
            "prior_scheme": PRIOR_SCHEME,
            "kernel_structure": KERNEL_STRUCTURE,
            "mean_function": MEAN_FUNCTION,
        },
        "feature_dims": space.feature_dims,
        "bounds": space.bounds.tolist(),
        "q": q,
        "q_expand": q_expand,
        "n_eval": n_eval,
        "batch_constraints": {str(k): list(v) if not isinstance(v, int) else v for k, v in active_batch_constraints.items()},
        "fixed_dims": {str(k): v for k, v in fixed_dims.items()},
        "optimizer_settings": pool_greedy_settings(),
        "source_a_availability_delay": source_a_availability_delay,
        "lookahead_horizon": lookahead_horizon,
        "num_fantasies": active_num_fantasies,
        "catalyst_ids": space.cat_ids,
        "initial_available_catalyst_indices": torch.where(space.initial_available)[0].tolist(),
        "landscape": {
            "f_min": landscape.f_min,
            "f_max": landscape.f_max,
            "x_max": landscape.x_max[0].tolist(),
            "noise_sd": yield_noise_sd(),
            "flip": landscape.flip.tolist(),
            "offset": landscape.offset.tolist(),
        },
        "global_max": 1.0,
        "n_start": int(space.start_x.shape[0]),
        "start_replicate_id": space.start_replicate_id.tolist(),
        "start_y_observed": start_y_observed,
        "start_y_true": start_y_true,
        "best_fs": best_fs,
        "best_observed": best_observed,
        "candidates_list": candidates_list,
        "y_observed_list": y_observed_list,
        "y_true_list": y_true_list,
        "max_acq_values_list": max_acq_values_list,
        "selected_source_a_indices": selected_source_a_indices,
        "expansion_picks": expansion_picks,
        "expansion_scores": expansion_scores,
        "released_source_a_by_iteration": released_source_a_by_iteration,
        "n_candidates_using_released": n_candidates_using_released,
        "available_source_counts_before_selection": available_source_counts_before_selection,
        "pending_source_counts_before_selection": pending_source_counts_before_selection,
        "initial_fit_time": initial_fit_time,
        "wall_times_acqf_opt": wall_times_acqf_opt,
        "wall_times_model_fit": wall_times_model_fit,
        "wall_times_expander": wall_times_expander,
        "wall_times_expander_parts": wall_times_expander_parts,
    }
