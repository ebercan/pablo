"""Merge per-cell result JSONs into a flat aggregate parquet for run_3.

Writes one row per fold, carrying the outer and inner cell configuration,
including ``hyperprior``, ``hp_ls_scale``, ``hp_ls_floor`` and
``hp_dirichlet_conc``.

Usage:
    python -m pablo.mechthild_runs.run_3_hyperprior_scan.concat_results [--input-dir DIR] [--output-path FILE] [--strict]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_INPUT_DIR = ROOT / "data" / "results" / "cells"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "results" / "hp_scan_aggregate.parquet"
INPUTS_PATH = ROOT / "data" / "inputs.json"


def _flatten(payload: dict) -> list[dict]:
    outer_id = payload["outer_cell_id"]
    outer = payload["outer_config"]
    rows: list[dict] = []
    for inner_block in payload["inner_cells"]:
        inner = inner_block["inner_config"]
        for fold in inner_block["folds"]:
            n_test = fold["n_test"]
            nlpd_pp = fold["nlpd_per_point"]
            crps_pp = fold["crps_per_point"]
            base_pp = fold["baseline_crps_per_point"]
            rows.append({
                "outer_cell_id": outer_id,
                "inner_id": inner_block["inner_id"],
                "hyperprior": outer["hyperprior"],
                "hp_ls_scale": outer["hyperprior_params"]["ls_scale"],
                "hp_ls_floor": outer["hyperprior_params"]["ls_floor"],
                "hp_dirichlet_conc": outer["hyperprior_params"]["dirichlet_conc"],
                "cat_feat": outer["cat_feat"],
                "solv_feat": outer["solv_feat"],
                "base_feat": outer["base_feat"],
                "kernel_structure": outer["kernel_structure"],
                "outcome_transform": inner["outcome_transform"],
                "model_class": inner["model_class"],
                "base_kernel": inner["base_kernel"],
                "mean_function": inner["mean_function"],
                "split_type": fold["split_type"],
                "repeat": fold["repeat"],
                "fold": fold["fold"],
                "seed": fold.get("seed"),
                "n_train": fold["n_train"],
                "n_test": n_test,
                "nlpd_per_point": nlpd_pp,
                "crps_per_point": crps_pp,
                "baseline_crps_per_point": base_pp,
                "nlpd_mean": sum(nlpd_pp) / n_test if nlpd_pp else float("nan"),
                "crps_mean": sum(crps_pp) / n_test if crps_pp else float("nan"),
                "baseline_crps_mean": sum(base_pp) / n_test if base_pp else float("nan"),
                "crpss": fold["crpss"],
                "q2": fold["q2"],
                "mae": fold["mae"],
                "rmse": fold["rmse"],
                "calibration_ks": fold["calibration_ks"],
                "wall_time_s": fold["wall_time_s"],
                "status": fold.get("status", "ok"),
                "left_out_solvent": fold.get("left_out_solvent"),
                "left_out_type": fold.get("left_out_type"),
                "left_out_scaffold": fold.get("left_out_scaffold"),
                "left_out_replicate_id": fold.get("left_out_replicate_id"),
                "left_out_idx": fold.get("left_out_idx"),
            })
    return rows


def concat(input_dir: Path, output_path: Path, strict: bool) -> Path:
    """Flatten every cell result JSON into one parquet and print a CRPSS summary.

    Cells listed in ``data/inputs.json`` without a result file are reported,
    or raise when ``strict`` is set. The printed summary is the mean CRPSS of
    ok folds by hyperprior × cat_feat × model_class × outcome_transform.

    Args:
        input_dir: Folder holding ``result_<id>.json`` files.
        output_path: Parquet file to write.
        strict: If True, raise when any cell from inputs.json is missing.

    Returns:
        ``output_path``.

    Raises:
        AssertionError: If ``input_dir`` holds no result JSONs.
        FileNotFoundError: If ``strict`` is True and cells are missing.
    """
    json_paths = sorted(input_dir.glob("result_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    assert json_paths, f"No result JSONs found under {input_dir}"

    expected = set(int(k) for k in json.loads(INPUTS_PATH.read_text()).keys())
    found = set(int(p.stem.split("_")[1]) for p in json_paths)
    missing = expected - found
    if missing:
        msg = f"{len(missing)} cells missing: {sorted(missing)}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"[concat] WARNING — {msg}")

    rows: list[dict] = []
    for p in json_paths:
        rows.extend(_flatten(json.loads(p.read_text())))

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n_ok = (df["status"] == "ok").sum()
    n_fail = (df["status"] == "failed").sum()
    print(f"[concat] {len(json_paths)} cells, {len(df)} rows ({n_ok} ok, {n_fail} failed) -> {output_path}")
    print("\nMean CRPSS by hyperprior × cat_feat × model_class × outcome_transform:")
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        pivot = ok.groupby(["hyperprior", "cat_feat", "model_class", "outcome_transform"])["crpss"].mean()
        print(pivot.round(4).to_string())
    return output_path


def main() -> None:
    """Parse the command line and run ``concat``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    concat(args.input_dir, args.output_path, args.strict)


if __name__ == "__main__":
    main()
