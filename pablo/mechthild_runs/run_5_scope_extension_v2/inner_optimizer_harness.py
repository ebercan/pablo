"""Measurement harness for the inner batch optimizer of the scope expanders.

Four stages, each a subcommand that reads from and writes into ``--out_dir``:

``snapshots``
    One run_5-style BO trajectory per benchmark (random expander, GA batch)
    and the model state at iterations whose training size is ≈ 2·D, ≈ 250 and
    ≈ 550.
``nodes``
    Fantasy nodes at depths 0–4 derived from every snapshot the way the
    fantasized expander derives them: the GA picks the node batch on the
    parent model over the set available at that depth, the child is the parent
    conditioned on one Sobol posterior sample at that batch.
``score``
    Every optimizer on every node, scored by joint qLogEI under one fixed set
    of QMC base samples, normalized as ``(AF − AF_random) / (AF_ref − AF_random)``.
``report``
    Reads every ``scores_<benchmark>*.json``, drops duplicate records and
    prints the markdown summary tables.

Example (repo root)::

    python -m pablo.mechthild_runs.run_5_scope_extension_v2.inner_optimizer_harness snapshots --out_dir <dir>
    python -m pablo.mechthild_runs.run_5_scope_extension_v2.inner_optimizer_harness nodes --out_dir <dir>
    python -m pablo.mechthild_runs.run_5_scope_extension_v2.inner_optimizer_harness score --out_dir <dir>
    python -m pablo.mechthild_runs.run_5_scope_extension_v2.inner_optimizer_harness report --out_dir <dir>
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Callable

import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler

from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    LOOKAHEAD_HORIZON,
    Q_BATCH_SIZE,
    SOURCE_A_AVAILABILITY_DELAY,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.latent_benchmark import (
    DEFAULT_BATCH_CONSTRAINTS,
    FIXED_DIM_FRACTION,
    FIXED_DIM_INDEX,
    INITIAL_AVAILABLE_SOURCE_A_COUNT,
    N_STARTING_POINTS,
    _fit_model,
    _initialize_source_a_state,
    build_benchmark,
    build_x_allowed_from_sources,
    generate_dependent_choice_sources,
    generate_starting_set,
)
from pablo.mechthild_runs.run_5_scope_extension_v2.run_bo_experiment import (
    _pick_random_unordered_source,
    _release_matured_source_a,
)
from pablo.optimize import (
    GPPoolPosteriorCache,
    optimize_acqf_mixed_canonical_GA,
    optimize_acqf_mixed_GA,
    optimize_acqf_mixed_pool_greedy,
)
from pablo.utils.acq_utils import best_observed_f

HARNESS_BENCHMARKS = ("Ackley", "Gallagher101", "LunacekBiRastrigin")
HARNESS_SEEDS = {"seed_discrete": 42, "seed_starting": 19, "seed_initial_available": 1}
SNAPSHOT_ITERATIONS = (0, 33, 83)
GA_N_EVALS = 15000
GA_POP_SIZE = 100
FANTASY_PATHS = 2
SNAPSHOT_QMC_SAMPLES = 128
SCORE_QMC_SAMPLES = 1024
SCORE_QMC_SEED = 12345
N_RANDOM_BATCHES = 64
N_OPTIMIZER_SEEDS = 3
REFERENCE_OPTIMIZER = "ga_tree_4x"
TREE_QMC_SAMPLES = 128


def _problem(benchmark_name: str) -> dict[str, object]:
    """Bounds, pinned dims and component tables of one harness benchmark."""
    benchmark = build_benchmark(benchmark_name)
    bounds = benchmark.bounds().to(dtype=torch.double)
    lo, hi = bounds[0, FIXED_DIM_INDEX].item(), bounds[1, FIXED_DIM_INDEX].item()
    fixed_dims = {FIXED_DIM_INDEX: lo + FIXED_DIM_FRACTION * (hi - lo)}
    source_a, source_b, source_c = generate_dependent_choice_sources(
        bounds=bounds, seed_discrete=HARNESS_SEEDS["seed_discrete"]
    )
    return {
        "benchmark": benchmark,
        "bounds": bounds,
        "fixed_dims": fixed_dims,
        "source_a": source_a,
        "source_b": source_b,
        "source_c": source_c,
    }


def run_snapshot_trajectory(benchmark_name: str) -> list[dict[str, object]]:
    """Run one run_5-style trajectory and snapshot it at ``SNAPSHOT_ITERATIONS``.

    The batch comes from ``optimize_acqf_mixed_GA`` (run_5 budget, QMC samples
    ``SNAPSHOT_QMC_SAMPLES``), the expander orders a uniformly random locked
    catalyst every iteration.

    Args:
        benchmark_name: One of ``HARNESS_BENCHMARKS``.

    Returns:
        One dict per snapshot with ``train_x`` ``(n, D)``, ``train_y`` ``(n, 1)``,
        ``available_mask`` ``(n_a,)``, ``pending_unlock_iteration`` ``(n_a,)``,
        ``iteration`` and ``main_batch`` ``(q, D)`` (the batch chosen on the
        snapshot model).
    """
    problem = _problem(benchmark_name)
    benchmark, bounds, fixed_dims = problem["benchmark"], problem["bounds"], problem["fixed_dims"]
    source_a, source_b, source_c = problem["source_a"], problem["source_b"], problem["source_c"]
    seed_s, seed_i = HARNESS_SEEDS["seed_starting"], HARNESS_SEEDS["seed_initial_available"]
    torch.manual_seed(HARNESS_SEEDS["seed_discrete"] * 1_000_003 + seed_s * 1_009 + seed_i)

    x_allowed = build_x_allowed_from_sources(source_a, source_b, source_c)
    n_a = source_a.shape[0]
    source_a_index_per_row = torch.arange(n_a).repeat_interleave(source_b.shape[0] * source_c.shape[0])
    available_mask, pending, _ = _initialize_source_a_state(
        n_a, INITIAL_AVAILABLE_SOURCE_A_COUNT, seed_i
    )
    train_x = generate_starting_set(
        bounds, x_allowed[available_mask[source_a_index_per_row]], seed_s, N_STARTING_POINTS, fixed_dims
    )
    train_y = benchmark.value(train_x).to(torch.double).reshape(-1, 1)

    snapshots: list[dict[str, object]] = []
    for iteration in range(max(SNAPSHOT_ITERATIONS) + 1):
        _release_matured_source_a(available_mask, pending, iteration)
        model = _fit_model(train_x, train_y, bounds)
        acqf = qLogExpectedImprovement(
            model=model,
            best_f=train_y.max(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([SNAPSHOT_QMC_SAMPLES])),
        )
        batch, _ = optimize_acqf_mixed_GA(
            acq_function=acqf,
            bounds=bounds,
            q=Q_BATCH_SIZE,
            n_evals_ga=GA_N_EVALS,
            pop_size=GA_POP_SIZE,
            cat_emb_avail=source_a[available_mask],
            sol_emb_avail=source_b,
            bas_emb_avail=source_c,
            batch_constraints=DEFAULT_BATCH_CONSTRAINTS,
            fixed_dims=fixed_dims,
        )
        if iteration in SNAPSHOT_ITERATIONS:
            snapshots.append(
                {
                    "benchmark": benchmark_name,
                    "iteration": iteration,
                    "train_x": train_x.clone(),
                    "train_y": train_y.clone(),
                    "available_mask": available_mask.clone(),
                    "pending_unlock_iteration": pending.clone(),
                    "main_batch": batch.clone(),
                }
            )
            print(f"{benchmark_name}: snapshot at iteration {iteration}, n={train_x.shape[0]}", flush=True)
        train_x = torch.cat([train_x, batch], dim=0)
        train_y = torch.cat([train_y, benchmark.value(batch).to(torch.double).reshape(-1, 1)], dim=0)
        picked = _pick_random_unordered_source(available_mask, pending, seed=seed_i * 1_000_003 + iteration)
        pending[picked] = iteration + SOURCE_A_AVAILABILITY_DELAY
    return snapshots


def build_node_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    path: list[tuple[torch.Tensor, torch.Tensor]],
) -> SingleTaskGP:
    """Rebuild a fantasy-node model from its root data, hyperparameters and path.

    Args:
        train_x: ``(n, D)`` root training inputs.
        train_y: ``(n, 1)`` root training targets.
        bounds: ``(2, D)`` input bounds.
        state_dict: Fitted root-model state (hyperparameters and transform buffers).
        path: ``(X_j, Y_j)`` pairs, ``(q, D)`` and ``(q, 1)``, conditioned on in order.

    Returns:
        The node model in eval mode.
    """
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    model.load_state_dict(state_dict)
    model.eval()
    for x_batch, y_batch in path:
        model.posterior(x_batch)
        model = model.condition_on_observations(X=x_batch, Y=y_batch)
        model.eval()
    return model


def derive_nodes(snapshot: dict[str, object]) -> list[dict[str, object]]:
    """Derive the root node and ``FANTASY_PATHS`` fantasy paths of depth 1–4.

    Args:
        snapshot: One entry of :func:`run_snapshot_trajectory`.

    Returns:
        One dict per node with the root data, the fitted root ``state_dict``,
        the fantasy ``path``, the catalyst indices available at that depth
        (``cat_indices``), the node ``depth`` and ``parent_batch`` ``(q, D)``
        (the batch the node was conditioned on; the root's main batch at depth 0).
    """
    problem = _problem(str(snapshot["benchmark"]))
    bounds, fixed_dims = problem["bounds"], problem["fixed_dims"]
    source_a, source_b, source_c = problem["source_a"], problem["source_b"], problem["source_c"]
    train_x, train_y = snapshot["train_x"], snapshot["train_y"]
    root = _fit_model(train_x, train_y, bounds)
    state_dict = {k: v.clone() for k, v in root.state_dict().items()}
    iteration = int(snapshot["iteration"])
    available_mask = snapshot["available_mask"]
    pending = snapshot["pending_unlock_iteration"]

    def cat_indices_at(depth: int) -> torch.Tensor:
        unlocked = (pending >= 0) & (pending <= iteration + depth)
        return torch.where(available_mask | unlocked)[0]

    base = {
        "benchmark": snapshot["benchmark"],
        "iteration": iteration,
        "train_x": train_x,
        "train_y": train_y,
        "state_dict": state_dict,
    }
    nodes = [
        {**base, "path": [], "depth": 0, "fantasy_path": -1,
         "cat_indices": cat_indices_at(0), "parent_batch": snapshot["main_batch"]}
    ]
    for path_idx in range(FANTASY_PATHS):
        path: list[tuple[torch.Tensor, torch.Tensor]] = []
        model = build_node_model(train_x, train_y, bounds, state_dict, path)
        for depth in range(1, LOOKAHEAD_HORIZON + 1):
            cats = cat_indices_at(depth - 1)
            acqf = qLogExpectedImprovement(model=model, best_f=best_observed_f(model))
            batch, _ = optimize_acqf_mixed_GA(
                acq_function=acqf,
                bounds=bounds,
                q=Q_BATCH_SIZE,
                n_evals_ga=GA_N_EVALS,
                pop_size=GA_POP_SIZE,
                cat_emb_avail=source_a[cats],
                sol_emb_avail=source_b,
                bas_emb_avail=source_c,
                batch_constraints=DEFAULT_BATCH_CONSTRAINTS,
                fixed_dims=fixed_dims,
                seed=1 + path_idx * 100 + depth,
            )
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=path_idx * 100 + depth)
            with torch.no_grad():
                y_sample = sampler(model.posterior(batch))[0]
            path = path + [(batch.detach(), y_sample.detach())]
            model = model.condition_on_observations(X=batch, Y=y_sample)
            model.eval()
            nodes.append(
                {**base, "path": list(path), "depth": depth, "fantasy_path": path_idx,
                 "cat_indices": cat_indices_at(depth), "parent_batch": batch.detach()}
            )
    return nodes


def random_feasible_batch(
    bounds: torch.Tensor,
    fixed_dims: dict[int, float],
    cat_emb: torch.Tensor,
    sol_emb: torch.Tensor,
    bas_emb: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """One uniformly random ``(q, D)`` batch with the run_5 temperature partition.

    Args:
        bounds: ``(2, D)`` input bounds.
        fixed_dims: Pinned continuous dims.
        cat_emb: ``(n_cat, 18)`` available catalyst rows.
        sol_emb: ``(n_sol, 2)`` solvent rows.
        bas_emb: ``(n_bas, 2)`` base rows.
        generator: Random generator.

    Returns:
        ``(Q_BATCH_SIZE, D)`` batch.
    """
    d_cont = bounds.shape[-1] - cat_emb.shape[1] - sol_emb.shape[1] - bas_emb.shape[1]
    lo, hi = bounds[0, :d_cont], bounds[1, :d_cont]
    cont = lo + (hi - lo) * torch.rand(Q_BATCH_SIZE, d_cont, dtype=bounds.dtype, generator=generator)
    partition = DEFAULT_BATCH_CONSTRAINTS[0]
    group_temps = cont[: len(partition), 0].clone()
    cont[:, 0] = torch.repeat_interleave(group_temps, torch.tensor(partition))
    for dim, value in fixed_dims.items():
        cont[:, dim] = value
    picks = [torch.randint(t.shape[0], (Q_BATCH_SIZE,), generator=generator) for t in (cat_emb, sol_emb, bas_emb)]
    return torch.cat([cont, cat_emb[picks[0]], sol_emb[picks[1]], bas_emb[picks[2]]], dim=-1)


def optimizer_registry() -> dict[str, tuple[Callable[..., tuple[torch.Tensor, torch.Tensor]], bool]]:
    """Optimizers under test: ``name -> (callable, is_pool_greedy)``.

    Every callable takes the ``optimize_acqf_mixed_GA`` component-mode keywords
    plus ``seed``. Pool-greedy variants also get the node's parent batch as
    ``warm_start`` and a root :class:`GPPoolPosteriorCache`, as in the tree.

    Returns:
        Mapping from optimizer name to ``(callable, is_pool_greedy)``; each
        callable returns ``(batch, acq_value)`` with ``batch`` of shape
        ``q x d``.
    """
    ga = functools.partial(optimize_acqf_mixed_GA, n_evals_ga=GA_N_EVALS, pop_size=GA_POP_SIZE)
    pool = optimize_acqf_mixed_pool_greedy
    return {
        "ga_tree": (ga, False),
        REFERENCE_OPTIMIZER: (
            functools.partial(optimize_acqf_mixed_GA, n_evals_ga=4 * GA_N_EVALS, pop_size=4 * GA_POP_SIZE),
            False,
        ),
        "canonical_ga_tree": (
            functools.partial(optimize_acqf_mixed_canonical_GA, n_evals_ga=GA_N_EVALS, pop_size=GA_POP_SIZE),
            False,
        ),
        "pool_1024": (functools.partial(pool, n_pool=1024, n_mc=128), True),
        "pool_1024_noswap": (functools.partial(pool, n_pool=1024, n_mc=128, n_swap_rounds=0), True),
        "pool_1024_noswap_polish20": (
            functools.partial(pool, n_pool=1024, n_mc=128, n_swap_rounds=0, polish_maxiter=20),
            True,
        ),
        "pool_1024_nopolish": (functools.partial(pool, n_pool=1024, n_mc=128, polish_maxiter=0), True),
        "pool_512_noswap_polish20": (
            functools.partial(pool, n_pool=512, n_mc=128, n_swap_rounds=0, polish_maxiter=20),
            True,
        ),
        "pool_4096": (functools.partial(pool, n_pool=4096, n_mc=128), True),
        "pool_4096_noswap_polish20": (
            functools.partial(pool, n_pool=4096, n_swap_rounds=0, polish_maxiter=20), True,
        ),
        "pool_4096_noswap_polish20_ov128": (
            functools.partial(pool, n_pool=4096, n_swap_rounds=0, polish_maxiter=20, n_override=128), True,
        ),
        "pool_2048_noswap_polish20": (
            functools.partial(pool, n_pool=2048, n_swap_rounds=0, polish_maxiter=20), True,
        ),
        "pool_4096_polish20": (functools.partial(pool, n_pool=4096, polish_maxiter=20), True),
        "pg_tree_4096": (functools.partial(pool, n_pool=4096, n_swap_rounds=0, polish_maxiter=20), True),
        "pg_tree_1024": (functools.partial(pool, n_pool=1024, n_swap_rounds=0, polish_maxiter=20), True),
        "pg_batch_4096": (functools.partial(pool, n_pool=4096), True),
    }


def score_node(node: dict[str, object], optimizers: list[str]) -> list[dict[str, object]]:
    """Run every optimizer on one node and score its batch.

    The score is joint qLogEI with ``SCORE_QMC_SAMPLES`` fixed Sobol base samples
    (seed ``SCORE_QMC_SEED``), identical for every optimizer, clipped to the
    analytic bounds ``maxᵢ log EIᵢ ≤ log qEI ≤ log Σᵢ EIᵢ``: in deep tails no QMC
    sample reaches the improvement region and the MC estimate is meaningless. Each optimizer
    works on its own ``qLogExpectedImprovement``: GA variants use the BoTorch
    default sampler, pool-greedy variants a ``TREE_QMC_SAMPLES``-sample
    (128) Sobol sampler, as the expander tree builds them.

    Args:
        node: One entry of :func:`derive_nodes`.
        optimizers: Names from :func:`optimizer_registry`.

    Returns:
        One record per (optimizer, seed) with ``af`` and ``seconds``, plus one
        ``random`` record holding the median over ``N_RANDOM_BATCHES`` batches.
    """
    problem = _problem(str(node["benchmark"]))
    bounds, fixed_dims = problem["bounds"], problem["fixed_dims"]
    cat_emb = problem["source_a"][node["cat_indices"]]
    sol_emb, bas_emb = problem["source_b"], problem["source_c"]
    model = build_node_model(node["train_x"], node["train_y"], bounds, node["state_dict"], node["path"])
    root = build_node_model(node["train_x"], node["train_y"], bounds, node["state_dict"], [])
    best_f = best_observed_f(model)
    scorer = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([SCORE_QMC_SAMPLES]), seed=SCORE_QMC_SEED),
    )
    meta = {
        "benchmark": node["benchmark"],
        "iteration": node["iteration"],
        "n_train": int(node["train_x"].shape[0]) + 6 * int(node["depth"]),
        "depth": node["depth"],
        "fantasy_path": node["fantasy_path"],
    }

    single = LogExpectedImprovement(model=model, best_f=best_f)

    def score(batch: torch.Tensor) -> float:
        with torch.no_grad():
            mc = scorer(batch.unsqueeze(0)).reshape(())
            singles = single(batch.unsqueeze(1))  # (q,)
            return float(torch.minimum(torch.maximum(mc, singles.max()), torch.logsumexp(singles, dim=0)))

    generator = torch.Generator().manual_seed(0)
    random_scores = torch.tensor(
        [score(random_feasible_batch(bounds, fixed_dims, cat_emb, sol_emb, bas_emb, generator))
         for _ in range(N_RANDOM_BATCHES)]
    )
    records = [{**meta, "optimizer": "random", "seed": -1, "af": float(random_scores.median()), "seconds": 0.0}]
    registry = optimizer_registry()
    for name in optimizers:
        optimizer, is_pool = registry[name]
        n_seeds = 1 if name == "ga_ref" else N_OPTIMIZER_SEEDS
        for seed in range(n_seeds):
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([TREE_QMC_SAMPLES])) if is_pool else None
            acqf = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
            extra = (
                {"warm_start": node["parent_batch"], "posterior_cache": GPPoolPosteriorCache(root)}
                if is_pool else {}
            )
            t0 = time.perf_counter()
            batch, _ = optimizer(
                acq_function=acqf,
                bounds=bounds,
                q=Q_BATCH_SIZE,
                cat_emb_avail=cat_emb,
                sol_emb_avail=sol_emb,
                bas_emb_avail=bas_emb,
                batch_constraints=DEFAULT_BATCH_CONSTRAINTS,
                fixed_dims=fixed_dims,
                seed=seed + 1,
                **extra,
            )
            seconds = time.perf_counter() - t0
            records.append({**meta, "optimizer": name, "seed": seed, "af": score(batch), "seconds": seconds})
    return records


def summarize(records: list[dict[str, object]]) -> str:
    """Markdown tables of the normalized score and wall time.

    Normalized score per record: ``(AF − AF_random) / (AF_ref − AF_random)`` on
    its node, ``AF_ref`` the best ``REFERENCE_OPTIMIZER`` run there. Rows: optimizer; columns: median and 10th percentile of the
    normalized score, median seconds, and the share of runs with
    ``AF ≥ AF(ga_tree)`` (same node, same seed) — over all nodes, per
    training size, per depth and per benchmark.

    Args:
        records: Score records as returned by :func:`score_node`.

    Returns:
        The markdown tables, one per grouping, as a single string.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    node_cols = ["benchmark", "iteration", "depth", "fantasy_path"]
    anchors = df[df.optimizer.isin(["random", REFERENCE_OPTIMIZER, "ga_tree"])]
    rnd = anchors[anchors.optimizer == "random"].set_index(node_cols)["af"].rename("af_random")
    ref = anchors[anchors.optimizer == REFERENCE_OPTIMIZER].groupby(node_cols)["af"].max().rename("af_ref")
    ga = anchors[anchors.optimizer == "ga_tree"].set_index(node_cols + ["seed"])["af"].rename("af_ga")
    df = df.join(rnd, on=node_cols).join(ref, on=node_cols).join(ga, on=node_cols + ["seed"])
    df = df.dropna(subset=["af_random", "af_ref"])
    df["norm"] = (df.af - df.af_random) / (df.af_ref - df.af_random)
    df["beats_ga"] = df.af >= df.af_ga - 1e-9
    df = df[df.optimizer != "random"]
    df["size"] = df.n_train.map(lambda n: "≈2·D" if n < 150 else ("≈250" if n < 450 else "≈550"))
    lines = []
    for keys, title in ((["optimizer"], "all nodes"), (["optimizer", "size"], "by training size"), (["optimizer", "depth"], "by depth"), (["optimizer", "benchmark"], "by benchmark")):
        g = df.groupby(keys)
        table = pd.DataFrame(
            {
                "median": g.norm.median(),
                "p10": g.norm.quantile(0.1),
                "s_median": g.seconds.median(),
                "af≥ga_tree": g.beats_ga.mean(),
                "runs": g.norm.size(),
            }
        ).round(3)
        lines.append(f"### Normalized score, {title}\n\n" + table.to_string() + "\n")
    return "\n".join(lines)


def _cmd_score(out_dir: Path, benchmarks: list[str], optimizers: list[str], tag: str) -> None:
    for name in benchmarks:
        records = []
        for i, node in enumerate(torch.load(out_dir / f"nodes_{name}.pt", weights_only=False)):
            t0 = time.perf_counter()
            records.extend(score_node(node, optimizers))
            print(f"{name}: node {i} (depth {node['depth']}, it {node['iteration']}) in {time.perf_counter() - t0:.0f} s", flush=True)
            (out_dir / f"scores_{name}{tag}.json").write_text(json.dumps(records))


def _cmd_report(out_dir: Path, benchmarks: list[str]) -> None:
    records = []
    for name in benchmarks:
        for path in sorted(out_dir.glob(f"scores_{name}*.json")):
            records.extend(json.loads(path.read_text()))
    unique = {
        (r["benchmark"], r["iteration"], r["depth"], r["fantasy_path"], r["optimizer"], r["seed"]): r
        for r in records
    }
    print(summarize(list(unique.values())))


def _cmd_snapshots(out_dir: Path, benchmarks: list[str]) -> None:
    for name in benchmarks:
        t0 = time.perf_counter()
        snaps = run_snapshot_trajectory(name)
        torch.save(snaps, out_dir / f"snapshots_{name}.pt")
        print(f"{name}: {len(snaps)} snapshots in {time.perf_counter() - t0:.0f} s", flush=True)


def _cmd_nodes(out_dir: Path, benchmarks: list[str]) -> None:
    for name in benchmarks:
        t0 = time.perf_counter()
        nodes = []
        for snapshot in torch.load(out_dir / f"snapshots_{name}.pt", weights_only=False):
            nodes.extend(derive_nodes(snapshot))
        torch.save(nodes, out_dir / f"nodes_{name}.pt")
        print(f"{name}: {len(nodes)} nodes in {time.perf_counter() - t0:.0f} s", flush=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["snapshots", "nodes", "score", "report"])
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(HARNESS_BENCHMARKS))
    parser.add_argument("--optimizers", nargs="+", default=list(optimizer_registry()))
    parser.add_argument("--tag", default="", help="Suffix of the score file, for extra passes")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.stage == "snapshots":
        _cmd_snapshots(args.out_dir, args.benchmarks)
    elif args.stage == "nodes":
        _cmd_nodes(args.out_dir, args.benchmarks)
    elif args.stage == "score":
        _cmd_score(args.out_dir, args.benchmarks, args.optimizers, args.tag)
    else:
        _cmd_report(args.out_dir, args.benchmarks)


if __name__ == "__main__":
    main()
