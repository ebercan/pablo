"""Enumerate the run_3 hyperprior scan grid and write data/inputs.json.

Outer axis: hyperprior_config(4) × cat_feat(2) = 8 outer cells.
Inner axis: outcome_transform(2) × model_class(2) × mean_function(2) = 8 configs.
Fixed: KS4, matern52, solvent=physical, base=structured.

Usage:
    python -m pablo.mechthild_runs.run_3_hyperprior_scan.generate_inputs
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

from pablo.mechthild_runs.run_3_hyperprior_scan.config import (
    BASE_KERNEL,
    CAT_FEATURIZATIONS,
    BASE_FEATURIZATION,
    HYPERPRIOR_CONFIGS,
    KERNEL_STRUCTURE,
    MEAN_FUNCTIONS,
    MODEL_CLASSES,
    N_INNER_CONFIGS,
    OUTCOME_TRANSFORMS,
    SOLV_FEATURIZATION,
)


def _inner_configs() -> list[dict]:
    cells = []
    inner_id = 0
    for ot, mc, mf in itertools.product(
        OUTCOME_TRANSFORMS, MODEL_CLASSES, MEAN_FUNCTIONS,
    ):
        cells.append({
            "inner_id": inner_id,
            "outcome_transform": ot,
            "model_class": mc,
            "base_kernel": BASE_KERNEL,
            "mean_function": mf,
        })
        inner_id += 1
    return cells


def enumerate_cells() -> dict[str, dict]:
    """Enumerate the outer cells of the hyperprior scan grid.

    Returns:
        Outer cell id (as a string) → cell dict with the hyperprior name and
        parameters, the catalyst, solvent and base featurizations, the kernel
        structure, the form one-hot flag and the shared list of inner configs.
    """
    inner = _inner_configs()
    assert len(inner) == N_INNER_CONFIGS

    cells: dict[str, dict] = {}
    outer_id = 0
    for hp in HYPERPRIOR_CONFIGS:
        for cat, append_form in CAT_FEATURIZATIONS:
            cells[str(outer_id)] = {
                "outer_cell_id": outer_id,
                "hyperprior": hp.name,
                "hyperprior_params": {
                    "ls_scale": hp.ls_scale,
                    "ls_floor": hp.ls_floor,
                    "dirichlet_conc": hp.dirichlet_conc,
                },
                "cat_feat": cat,
                "solv_feat": SOLV_FEATURIZATION,
                "base_feat": BASE_FEATURIZATION,
                "kernel_structure": KERNEL_STRUCTURE,
                "append_form_onehot_to_cat": append_form,
                "inner_cells": inner,
            }
            outer_id += 1
    return cells


def main() -> None:
    """Write the grid to ``data/inputs.json`` next to this script and print its size."""
    cells = enumerate_cells()
    n_outer = len(HYPERPRIOR_CONFIGS) * len(CAT_FEATURIZATIONS)
    assert len(cells) == n_outer

    total_configs = sum(len(c["inner_cells"]) for c in cells.values())

    out_path = Path(__file__).parent / "data" / "inputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cells, indent=2))

    print(f"Wrote {len(cells)} outer cells -> {out_path}")
    print(f"  outer: hyperprior={len(HYPERPRIOR_CONFIGS)} × cat={len(CAT_FEATURIZATIONS)} = {n_outer}")
    print(f"  inner: outcome={len(OUTCOME_TRANSFORMS)} × model={len(MODEL_CLASSES)} × mean={len(MEAN_FUNCTIONS)} = {N_INNER_CONFIGS}")
    print(f"  total model configs: {total_configs}")


if __name__ == "__main__":
    main()
