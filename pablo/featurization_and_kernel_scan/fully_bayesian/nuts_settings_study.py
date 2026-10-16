"""NUTS-settings convergence and predictive study on cross_1 rank-1 configuration.

Evaluates four NUTS MCMC configurations (A through D) across three representative
held-out reaction conditions (high, median, and low raw yield). Compares posterior
predictive mixture distributions against maximum a posteriori (MAP) baseline fits.

CLI::

    python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study generate
    python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study run --task <ID> [--warmup W --samples N]
    python -m pablo.featurization_and_kernel_scan.fully_bayesian.nuts_settings_study aggregate [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import gpytorch
import pandas as pd
import torch
from torch import Tensor

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.models import (
    AcyloinCondensationModel,
    ChainResult,
    FullyBayesianScanGP,
    NutsSettings,
)
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import _fit_multistart
from pablo.featurization_and_kernel_scan.results_analysis.rebuild_predictive import build_cell
from pablo.utils.feature_assembly import CellFeatures
from pablo.utils.metrics import crps_sample_based
from pablo.utils.outcome_transforms import inverse

CONFIG: dict[str, str] = {
    "cat_feat": "grover_pca",
    "solv_feat": "Hansen",
    "base_feat": "grover_pca",
    "kernel_structure": "KS1",
    "base_kernel": "matern52",
    "model_class": "st_y",
    "mean_function": "constant",
    "outcome_transform": "probit",
    "prior_scheme": "P2",
}
"""Rank-1 configuration from cross_1 used for the NUTS settings study."""

SETTINGS: dict[str, NutsSettings] = {
    "A": NutsSettings(num_chains=1, warmup_steps=512, num_samples=256, thinning=16),
    "B": NutsSettings(num_chains=2, warmup_steps=512, num_samples=256, thinning=4),
    "C": NutsSettings(num_chains=4, warmup_steps=512, num_samples=256, thinning=8),
    "D": NutsSettings(num_chains=4, warmup_steps=1000, num_samples=1000, thinning=8),
}
"""NUTS MCMC budget settings A through D to evaluate."""

SETTING_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
"""Evaluation ordering of the NUTS settings."""

BASE_SEED: int = 0
"""Base random seed for chain RNG initialization."""

STUDY_DIR: Path = paths.PKG_ROOT / "runs" / "nuts_settings"
"""Directory holding study inputs, chain samples, and summary tables."""

TABLE_COLUMNS: list[str] = [
    "point_rank",
    "row",
    "y_true",
    "setting",
    "chains_present",
    "wall_time_s_max_chain",
    "wall_time_s_sum",
    "evals_per_iteration",
    "divergences_total",
    "nonpd_evaluations",
    "rhat_max",
    "ess_min",
    "draws_kept",
    "mean_probit",
    "sd_probit",
    "crps_raw",
    "map_mean_probit",
    "map_sd_probit",
    "map_crps_raw",
]
"""Table columns for the study summary outputs."""


def select_held_out_points(cell: CellFeatures) -> list[dict[str, int | float]]:
    """Select the three deterministic held-out evaluation points from the cell.

    Picks the argmax (rank 0, high), lower median by sorted position (rank 1, median),
    and argmin (rank 2, low) of ``cell.y_raw``, breaking ties by lowest row index.

    Args:
        cell: Assembled cell features containing ``cell.y_raw`` of shape (N, 1).

    Returns:
        List of three dicts with keys "point_rank", "row" and "y_true" (raw yield fraction).
    """
    y = cell.y_raw.squeeze(-1).tolist()
    n = len(y)

    max_y = max(y)
    argmax_row = min(i for i in range(n) if y[i] == max_y)

    min_y = min(y)
    argmin_row = min(i for i in range(n) if y[i] == min_y)

    sorted_rows = sorted(range(n), key=lambda i: (y[i], i))
    median_row = sorted_rows[(n - 1) // 2]

    return [
        {"point_rank": 0, "row": argmax_row, "y_true": float(y[argmax_row])},
        {"point_rank": 1, "row": median_row, "y_true": float(y[median_row])},
        {"point_rank": 2, "row": argmin_row, "y_true": float(y[argmin_row])},
    ]


def generate_inputs() -> list[dict[str, object]]:
    """Build cell, select 3 held-out points, and write inputs.json.

    Returns:
        The generated task list of 33 entries.
    """
    cell = build_cell(CONFIG, CONFIG)
    points = select_held_out_points(cell)
    tasks: list[dict[str, object]] = []
    task_id = 0

    for pt in points:
        point_rank = int(pt["point_rank"])
        row = int(pt["row"])
        y_val = float(pt["y_true"])
        for setting_idx, setting_name in enumerate(SETTING_ORDER):
            settings_obj = SETTINGS[setting_name]
            for chain_idx in range(settings_obj.num_chains):
                seed = BASE_SEED + 1000 * point_rank + 100 * setting_idx + chain_idx
                tasks.append({
                    "task": task_id,
                    "point_rank": point_rank,
                    "row": row,
                    "y_true": y_val,
                    "setting": setting_name,
                    "chain": chain_idx,
                    "seed": seed,
                })
                task_id += 1

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    inputs_path = STUDY_DIR / "inputs.json"
    inputs_path.write_text(json.dumps(tasks, indent=2) + "\n")
    print(f"Generated {len(tasks)} tasks and wrote {inputs_path}")
    return tasks


def run_task(
    task_id: int,
    warmup_override: int | None = None,
    samples_override: int | None = None,
) -> None:
    """Run one MCMC chain for the specified task and save chain results.

    Args:
        task_id: Task index I from inputs.json.
        warmup_override: Optional warmup step override (for smoke testing).
        samples_override: Optional post-warmup sample override (for smoke testing).
    """
    torch.set_num_threads(1)

    is_smoke = warmup_override is not None or samples_override is not None
    chains_dir = STUDY_DIR / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if is_smoke else ""
    final_path = chains_dir / f"chain_{task_id}{suffix}.pt"

    if final_path.exists():
        print(f"Chain output {final_path} already exists, skipping.")
        return

    inputs_path = STUDY_DIR / "inputs.json"
    tasks = json.loads(inputs_path.read_text())
    entry = tasks[task_id]

    cell = build_cell(CONFIG, CONFIG)
    held_out_row = int(entry["row"])
    train_idx = [i for i in range(cell.X.shape[0]) if i != held_out_row]

    model = FullyBayesianScanGP.from_config(CONFIG, cell, train_idx)

    base_settings = SETTINGS[str(entry["setting"])]
    warmup = warmup_override if warmup_override is not None else base_settings.warmup_steps
    samples = samples_override if samples_override is not None else base_settings.num_samples
    settings = NutsSettings(
        num_chains=base_settings.num_chains,
        warmup_steps=warmup,
        num_samples=samples,
        thinning=base_settings.thinning,
        max_tree_depth=base_settings.max_tree_depth,
    )

    seed = int(entry["seed"])
    chain_result = model.run_chain(settings, seed)

    data = {
        "task": entry["task"],
        "point_rank": entry["point_rank"],
        "row": entry["row"],
        "y_true": entry["y_true"],
        "setting": entry["setting"],
        "chain": entry["chain"],
        "seed": entry["seed"],
        "num_chains": settings.num_chains,
        "warmup_steps": settings.warmup_steps,
        "num_samples": settings.num_samples,
        "thinning": settings.thinning,
        "max_tree_depth": settings.max_tree_depth,
        "samples": chain_result.samples,
        "divergences": chain_result.divergences,
        "nonpd_evaluations": chain_result.nonpd_evaluations,
        "potential_evaluations": chain_result.potential_evaluations,
        "wall_time_s": chain_result.wall_time_s,
    }

    tmp_path = chains_dir / f".tmp_chain_{task_id}{suffix}.pt"
    torch.save(data, tmp_path)
    os.replace(tmp_path, final_path)
    print(f"Saved chain {final_path} (wall time {chain_result.wall_time_s:.2f}s).")


def fit_map_baseline(
    cell: CellFeatures,
    held_out_row: int,
    generator_seed: int = 0,
) -> tuple[float, float, float]:
    """Fit MAP model on the fold and compute probit mean/SD and raw CRPS.

    Args:
        cell: Assembled cell features containing feature matrix X of shape
            (N, D) and raw yield tensor y_raw of shape (N, 1).
        held_out_row: Index of the single held-out test row.
        generator_seed: RNG seed for drawing predictive mixture samples.

    Returns:
        Tuple of (map_mean_probit, map_sd_probit, map_crps_raw).
    """
    train_idx = [i for i in range(cell.X.shape[0]) if i != held_out_row]
    map_model = AcyloinCondensationModel(
        cell_features=cell,
        train_idx=train_idx,
        kernel_structure=CONFIG["kernel_structure"],
        base_kernel_type=CONFIG["base_kernel"],
        outcome_transform=CONFIG["outcome_transform"],
        lengthscale_scale=config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme=CONFIG["prior_scheme"],
        mean_function=CONFIG["mean_function"],
    ).st_y(None)

    with gpytorch.settings.cholesky_jitter(double_value=config.CHOLESKY_JITTER_F64):
        _fit_multistart(map_model, cell.X[train_idx], config.N_RESTARTS, config.MAX_ITER)

    map_model.eval()
    with torch.no_grad():
        test_x = cell.X[held_out_row : held_out_row + 1]
        post = map_model.posterior(test_x, observation_noise=True)
        m_map = post.mean.squeeze()
        s_map = post.variance.squeeze().sqrt()

        gen = torch.Generator().manual_seed(generator_seed)
        eps = torch.randn(2000, generator=gen, dtype=m_map.dtype, device=m_map.device)
        z_samples = m_map + s_map * eps
        y_samples = inverse(z_samples.unsqueeze(1), CONFIG["outcome_transform"])
        y_true_tensor = cell.y_raw[held_out_row : held_out_row + 1]
        map_crps = float(crps_sample_based(y_true_tensor, y_samples).item())

    return float(m_map.item()), float(s_map.item()), map_crps


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a GitHub-flavored Markdown table without external libraries.

    Args:
        df: DataFrame to format into Markdown.

    Returns:
        Formatted markdown table string.
    """
    headers = list(df.columns)
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        row_strs = []
        for col in headers:
            val = row[col]
            if pd.isna(val):
                row_strs.append("")
            elif col in ("point_rank", "row", "chains_present", "divergences_total", "nonpd_evaluations", "draws_kept"):
                row_strs.append(str(int(val)))
            elif isinstance(val, float):
                row_strs.append(f"{val:.4f}")
            else:
                row_strs.append(str(val))
        lines.append("| " + " | ".join(row_strs) + " |")

    return "\n".join(lines) + "\n"


def aggregate_results(smoke: bool = False) -> pd.DataFrame:
    """Aggregate completed MCMC chain results and output summary tables.

    Args:
        smoke: Whether to aggregate from smoke output files (_smoke suffix).

    Returns:
        Summary DataFrame matching table.parquet.
    """
    torch.set_num_threads(1)

    inputs_path = STUDY_DIR / "inputs.json"
    tasks = json.loads(inputs_path.read_text())
    cell = build_cell(CONFIG, CONFIG)

    chains_dir = STUDY_DIR / "chains"
    suffix = "_smoke" if smoke else ""

    map_cache: dict[int, tuple[float, float, float]] = {}

    def get_map(row_idx: int) -> tuple[float, float, float]:
        if row_idx not in map_cache:
            map_cache[row_idx] = fit_map_baseline(cell, row_idx)
        return map_cache[row_idx]

    rows: list[dict[str, object]] = []

    unique_ranks = sorted({int(t["point_rank"]) for t in tasks})

    for point_rank in unique_ranks:
        rank_tasks = [t for t in tasks if int(t["point_rank"]) == point_rank]
        held_out_row = int(rank_tasks[0]["row"])
        y_true_val = float(rank_tasks[0]["y_true"])

        for setting_name in SETTING_ORDER:
            group_tasks = [t for t in rank_tasks if str(t["setting"]) == setting_name]
            group_tasks.sort(key=lambda t: int(t["chain"]))

            missing_tasks = [
                t["task"]
                for t in group_tasks
                if not (chains_dir / f"chain_{t['task']}{suffix}.pt").exists()
            ]

            if missing_tasks:
                present_count = len(group_tasks) - len(missing_tasks)
                print(
                    f"Point rank {point_rank} (row {held_out_row}), setting {setting_name}: "
                    f"missing {len(missing_tasks)} chains for tasks {missing_tasks}"
                )
                rows.append({
                    "point_rank": point_rank,
                    "row": held_out_row,
                    "y_true": y_true_val,
                    "setting": setting_name,
                    "chains_present": present_count,
                    "wall_time_s_max_chain": float("nan"),
                    "wall_time_s_sum": float("nan"),
                    "evals_per_iteration": float("nan"),
                    "divergences_total": float("nan"),
                    "nonpd_evaluations": float("nan"),
                    "rhat_max": float("nan"),
                    "ess_min": float("nan"),
                    "draws_kept": float("nan"),
                    "mean_probit": float("nan"),
                    "sd_probit": float("nan"),
                    "crps_raw": float("nan"),
                    "map_mean_probit": float("nan"),
                    "map_sd_probit": float("nan"),
                    "map_crps_raw": float("nan"),
                })
                continue

            loaded_data = [
                torch.load(chains_dir / f"chain_{t['task']}{suffix}.pt", weights_only=False)
                for t in group_tasks
            ]

            chain_results = [
                ChainResult(
                    seed=int(d["seed"]),
                    samples={k: v for k, v in d["samples"].items()},
                    divergences=int(d["divergences"]),
                    nonpd_evaluations=int(d["nonpd_evaluations"]),
                    potential_evaluations=int(d["potential_evaluations"]),
                    wall_time_s=float(d["wall_time_s"]),
                )
                for d in loaded_data
            ]

            first_entry = loaded_data[0]
            settings_used = NutsSettings(
                num_chains=int(first_entry["num_chains"]),
                warmup_steps=int(first_entry["warmup_steps"]),
                num_samples=int(first_entry["num_samples"]),
                thinning=int(first_entry["thinning"]),
                max_tree_depth=int(first_entry["max_tree_depth"]),
            )

            train_idx = [i for i in range(cell.X.shape[0]) if i != held_out_row]
            fb_model = FullyBayesianScanGP.from_config(CONFIG, cell, train_idx)
            diagnostics = fb_model.load_chains(chain_results, settings_used, strict=False)

            test_x = cell.X[held_out_row : held_out_row + 1]
            m_draws, s_draws = fb_model.predictive_components(test_x)
            m_vec = m_draws.squeeze(-1)
            s_vec = s_draws.squeeze(-1)

            mean_probit = float(m_vec.mean().item())
            var_probit = float((s_vec.pow(2).mean() + (m_vec - mean_probit).pow(2).mean()).item())
            sd_probit = math.sqrt(var_probit)

            s_kept = int(m_vec.shape[0])
            gen = torch.Generator().manual_seed(0)
            idx = torch.randint(0, s_kept, (2000,), generator=gen)
            eps = torch.randn(2000, generator=gen, dtype=m_vec.dtype, device=m_vec.device)
            z_samples = m_vec[idx] + s_vec[idx] * eps
            y_samples = inverse(z_samples.unsqueeze(1), CONFIG["outcome_transform"])
            y_true_tensor = cell.y_raw[held_out_row : held_out_row + 1]
            crps_raw = float(crps_sample_based(y_true_tensor, y_samples).item())

            wall_time_s_max_chain = float(max(c.wall_time_s for c in chain_results))
            wall_time_s_sum = float(sum(c.wall_time_s for c in chain_results))
            total_evals = diagnostics.potential_evaluations
            n_chains = len(chain_results)
            evals_per_iter = float(
                total_evals / (n_chains * (settings_used.warmup_steps + settings_used.num_samples))
            )

            map_mean_probit, map_sd_probit, map_crps_raw = get_map(held_out_row)

            rows.append({
                "point_rank": point_rank,
                "row": held_out_row,
                "y_true": y_true_val,
                "setting": setting_name,
                "chains_present": len(loaded_data),
                "wall_time_s_max_chain": wall_time_s_max_chain,
                "wall_time_s_sum": wall_time_s_sum,
                "evals_per_iteration": evals_per_iter,
                "divergences_total": int(sum(diagnostics.divergences_per_chain)),
                "nonpd_evaluations": int(diagnostics.nonpd_evaluations),
                "rhat_max": float(diagnostics.rhat_max),
                "ess_min": float(diagnostics.ess_min),
                "draws_kept": int(diagnostics.num_draws_kept),
                "mean_probit": mean_probit,
                "sd_probit": sd_probit,
                "crps_raw": crps_raw,
                "map_mean_probit": map_mean_probit,
                "map_sd_probit": map_sd_probit,
                "map_crps_raw": map_crps_raw,
            })

    df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = STUDY_DIR / "table.parquet"
    md_path = STUDY_DIR / "table.md"
    df.to_parquet(parquet_path, index=False)
    md_content = dataframe_to_markdown(df)
    md_path.write_text(md_content)

    print(f"Wrote {parquet_path} and {md_path}")
    print(md_content)
    return df


def main() -> None:
    """CLI entry point for the NUTS settings study."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "generate",
        help="Generate inputs.json with the 33 study tasks.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Execute one MCMC chain task.",
    )
    run_parser.add_argument(
        "--task",
        type=int,
        required=True,
        help="Task ID index (0 to 32) from inputs.json.",
    )
    run_parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup steps override for smoke testing.",
    )
    run_parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Post-warmup sample count override for smoke testing.",
    )

    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate completed chain runs and write summary tables.",
    )
    agg_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load _smoke chain output files instead of production outputs.",
    )

    args = parser.parse_args()
    if args.command == "generate":
        generate_inputs()
    elif args.command == "run":
        run_task(
            task_id=args.task,
            warmup_override=args.warmup,
            samples_override=args.samples,
        )
    elif args.command == "aggregate":
        aggregate_results(smoke=args.smoke)


if __name__ == "__main__":
    main()
