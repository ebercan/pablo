"""Merge per-outer-cell result JSONs into a flat aggregate parquet.

Reads every ``result_<outer_id>.json`` under ``results/cells/`` and emits
one row per ``(outer_cell, inner_cell, split_type, repeat, fold)``. Per-point
arrays are preserved as list-typed parquet columns so paired Wilcoxon tests
remain possible. Outer-axis fields (cat, solv, kernel structure, model class,
outcome transform, base kernel, mean function) and the inner-axis fields (base
featurization, prior scheme) are flattened into the row.

Every fold is a row. ``status`` is ``"ok"`` or ``"failed"``; a failed fold
carries its ``error`` string, NaN scalar metrics and empty per-point arrays, so a
consumer must filter on ``status`` explicitly and can count failures rather than
infer them. A fold record with any other status is an error.

``--run <name>`` (default ``elimination``) reads the run's ``cells/`` and
``inputs.json`` and writes its ``scan_aggregate.parquet`` (``paths.run_paths``);
``--input-dir`` and ``--output-path`` override the folders, never the
``inputs.json`` the cell ids are checked against.

Usage::

    python -m pablo.featurization_and_kernel_scan.cross_validation.concat_results [options]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config

ROOT = paths.PKG_ROOT


FOLD_STATUSES: frozenset[str] = frozenset({"ok", "failed"})
"""The two outcomes a fold record can have."""


def _per_point_mean(values: list[float], n_test: int, ok: bool) -> float:
    return sum(values) / n_test if ok else float("nan")


def _flatten(payload: dict) -> tuple[list[dict], int]:
    """Flatten one cell's result JSON into fold rows.

    Returns:
        ``(rows, n_failed)``: one row per fold record, and how many are failed.

    Raises:
        ValueError: If a fold record carries a status other than ok / failed.
    """
    outer_id = payload["outer_cell_id"]
    outer = payload["outer_config"]
    rows: list[dict] = []
    n_failed = 0
    for inner_block in payload["inner_cells"]:
        inner = inner_block["inner_config"]
        for fold in inner_block["folds"]:
            status = fold.get("status", "ok")
            if status not in FOLD_STATUSES:
                raise ValueError(
                    f"outer_cell {outer_id} inner {inner_block['inner_id']} "
                    f"{fold['split_type']}/r{fold['repeat']}/f{fold['fold']}: "
                    f"unknown fold status {status!r}"
                )
            ok = status == "ok"
            n_failed += not ok
            n_test = fold["n_test"]
            rows.append({
                "outer_cell_id": outer_id,
                "inner_id": inner_block["inner_id"],
                "cat_feat": outer["cat_feat"],
                "solv_feat": outer["solv_feat"],
                "base_feat": inner["base_feat"],
                "prior_scheme": inner_block.get(
                    "prior_scheme", inner.get("prior_scheme")
                ),
                "kernel_structure": outer["kernel_structure"],
                "outcome_transform": outer["outcome_transform"],
                "model_class": outer["model_class"],
                "base_kernel": outer["base_kernel"],
                "mean_function": outer["mean_function"],
                "append_form_onehot_to_cat": outer.get("append_form_onehot_to_cat", False),
                "split_type": fold["split_type"],
                "repeat": fold["repeat"],
                "fold": fold["fold"],
                "status": status,
                "error": fold.get("error"),
                "seed": fold.get("seed"),
                "n_train": fold["n_train"],
                "n_test": n_test,
                # --- yield metrics (raw fraction scale) ---
                "nlpd_per_point": fold["nlpd_per_point"],
                "crps_per_point": fold["crps_per_point"],
                "baseline_crps_per_point": fold["baseline_crps_per_point"],
                "baseline_nlpd_per_point": fold.get("baseline_nlpd_per_point", []),
                "mu_pred_per_point": fold.get("mu_pred_per_point", []),
                "var_pred_per_point": fold.get("var_pred_per_point", []),
                "nlpd_mean": _per_point_mean(fold["nlpd_per_point"], n_test, ok),
                "crps_mean": _per_point_mean(fold["crps_per_point"], n_test, ok),
                "baseline_crps_mean": _per_point_mean(fold["baseline_crps_per_point"], n_test, ok),
                "crpss": fold["crpss"],
                "msll": fold.get("msll", float("nan")),
                "q2": fold["q2"],
                "mae": fold["mae"],
                "rmse": fold["rmse"],
                "calibration_ks": fold["calibration_ks"],
                # --- selectivity metrics (empty/NaN for yield-only classes) ---
                "nlpd_per_point_sel": fold.get("nlpd_per_point_sel", []),
                "crps_per_point_sel": fold.get("crps_per_point_sel", []),
                "baseline_crps_per_point_sel": fold.get("baseline_crps_per_point_sel", []),
                "baseline_nlpd_per_point_sel": fold.get("baseline_nlpd_per_point_sel", []),
                "mu_pred_per_point_sel": fold.get("mu_pred_per_point_sel", []),
                "var_pred_per_point_sel": fold.get("var_pred_per_point_sel", []),
                "crpss_sel": fold.get("crpss_sel", float("nan")),
                "msll_sel": fold.get("msll_sel", float("nan")),
                "q2_sel": fold.get("q2_sel", float("nan")),
                "mae_sel": fold.get("mae_sel", float("nan")),
                "rmse_sel": fold.get("rmse_sel", float("nan")),
                "calibration_ks_sel": fold.get("calibration_ks_sel", float("nan")),
                # --- posterior-mean correlation between the two modeled targets ---
                "posterior_mean_pearson": fold.get("posterior_mean_pearson", float("nan")),
                "posterior_mean_spearman": fold.get("posterior_mean_spearman", float("nan")),
                "wall_time_s": fold["wall_time_s"],
                "left_out_solvent": fold.get("left_out_solvent"),
                "left_out_type": fold.get("left_out_type"),
                "left_out_scaffold": fold.get("left_out_scaffold"),
                "left_out_replicate_id": fold.get("left_out_replicate_id"),
                "left_out_idx": fold.get("left_out_idx"),
            })
    if n_failed > 0:
        print(f"[_flatten] outer_cell {outer_id}: {n_failed} failed folds")
    return rows, n_failed


def _check_completeness(json_paths: list[Path], inputs: dict) -> None:
    fold_total = config.FOLD_TOTAL
    incomplete = []
    for p in json_paths:
        payload = json.loads(p.read_text())
        cell_id = payload["outer_cell_id"]
        expected_inners = len(inputs[str(cell_id)]["inner_cells"])
        for inner_block in payload["inner_cells"]:
            n_recorded = len({
                (f["split_type"], f["repeat"], f["fold"]) for f in inner_block["folds"]
            })
            if n_recorded < fold_total:
                incomplete.append(
                    f"cell {cell_id} inner {inner_block['inner_id']}: "
                    f"{n_recorded}/{fold_total} folds recorded (ok or failed)"
                )
        if len(payload["inner_cells"]) < expected_inners:
            incomplete.append(
                f"cell {cell_id}: {len(payload['inner_cells'])}/{expected_inners} "
                f"inner cells present"
            )
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} incomplete inner cells:\n  " + "\n  ".join(incomplete)
        )


def concat(
    input_dir: Path,
    output_path: Path,
    strict: bool,
    require_complete: bool,
    inputs_path: Path = paths.INPUTS_JSON,
) -> Path:
    """Write the fold-level aggregate of one run's result cells.

    Args:
        input_dir: Folder holding ``result_<outer_id>.json``.
        output_path: Aggregate parquet to write.
        strict: Error if a cell of ``inputs_path`` has no result file.
        require_complete: Also error if a fold is absent (neither ok nor failed).
        inputs_path: The run's ``inputs.json``; cell ids are checked against it.

    Returns:
        ``output_path``.

    Raises:
        FileNotFoundError: If expected cells from ``inputs_path`` are missing
            under ``input_dir`` when ``strict`` or ``require_complete`` is True.
        RuntimeError: If ``require_complete`` is True and any inner cell has
            fewer than ``config.FOLD_TOTAL`` recorded folds.
    """
    json_paths = sorted(input_dir.glob("result_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    assert json_paths, f"No result JSONs found under {input_dir}"

    inputs = json.loads(inputs_path.read_text())
    expected = set(int(k) for k in inputs.keys())
    found = set(int(p.stem.split("_")[1]) for p in json_paths)
    missing = expected - found
    if missing:
        msg = (f"{len(missing)} expected cells missing under {input_dir}: "
               f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        if strict or require_complete:
            raise FileNotFoundError(msg)
        print(f"[concat] WARNING - {msg}")

    if require_complete:
        _check_completeness(json_paths, inputs)

    rows: list[dict] = []
    total_failed = 0
    for p in json_paths:
        new_rows, n_failed = _flatten(json.loads(p.read_text()))
        rows.extend(new_rows)
        total_failed += n_failed

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[concat] {len(json_paths)} cells, {len(df)} rows -> {output_path}")
    print(f"[concat] {total_failed} failed folds (kept as status='failed' rows)")
    ok = df[df["status"] == "ok"]
    print("\nNLPD mean by kernel x transform (ok folds):")
    print(ok.groupby(["kernel_structure", "outcome_transform"])["nlpd_mean"].mean().round(4))
    return output_path


def main() -> None:
    """Parse CLI arguments and run aggregate concatenation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=sorted(config.RUN_GRIDS), default=config.DEFAULT_RUN,
                        help="Run whose cells, inputs.json and aggregate path to use.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Result cells folder; the run's cells/ by default.")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Aggregate parquet; the run's scan_aggregate.parquet by default.")
    parser.add_argument("--strict", action="store_true",
                        help="Error if any cell from inputs.json is missing.")
    parser.add_argument("--require-complete", action="store_true",
                        help="Error if any fold is absent (neither ok nor failed).")
    args = parser.parse_args()
    run = paths.run_paths(args.run)
    concat(
        args.input_dir or run.cells_dir,
        args.output_path or run.aggregate_parquet,
        args.strict,
        args.require_complete,
        inputs_path=run.inputs_json,
    )


if __name__ == "__main__":
    main()
