"""Scan every fitted fold for ARD lengthscales pinned at the ``GreaterThan`` floor.

Each fold in the cell JSONs under ``results/cells/`` stores the raw (pre-constraint)
lengthscale vector under a ``*.raw_lengthscale`` hyperparameter key. The actual
lengthscale applied by the model is ``softplus(raw) + hp_floor`` (the
``GreaterThan(hp_floor)`` transform in ``kernels.py``). A dimension is
"at the floor" when ``softplus(raw) ≲ 0``, i.e. the optimizer drove that ARD
lengthscale down to the collapse-preventing lower bound.

This script flattens one row per fold with the per-fold floor statistics plus the
config/split coordinates and the fold CRPSS, so the floor-hit rate can be
cross-tabulated against predictive quality without re-reading the 2 GB of JSON,
and one row per (fold, lengthscale scalar) with its floor flag. It prints the
floor-hit rate broken down by axis, prior scheme, model class and catalyst
featurization.

Outputs:
  - ``--output-path`` (default ``results/lengthscale_floor_scan.parquet``):
    one row per fold.
  - ``results/lengthscale_floor_axis_scan.parquet``: one row per fold and
    lengthscale scalar. Always written to this path.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.results_analysis.scan_lengthscale_floor [--input-dir DIR] [--output-path PATH]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.kernels import BLOCKS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "results" / "cells"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "lengthscale_floor_scan.parquet"
AXIS_OUTPUT_PATH = ROOT / "results" / "lengthscale_floor_axis_scan.parquet"

FLOOR_TOL = 1e-3

def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -50.0, 30.0))))

_LENGTHSCALE_KEY = re.compile(
    r"^(?:(?P<prefix>.*)\.)?(?P<container>block_kernels|kernels)"
    r"\.(?P<idx>\d+)(?:\.base_kernel)?\.raw_lengthscale$"
)
"""Matches one block kernel's lengthscale parameter.

``KS1`` nests its four block kernels under a ``ProductKernel`` (``...kernels.N``)
while ``KS2``-``KS4`` hold theirs on ``DirichletAdditiveKernel.block_kernels``
(``...block_kernels.N``); a ``MaskedKernel`` inserts one ``.base_kernel`` level
for the base block. Both container names must be matched: a regex covering only
one of them reports every fold of the other three structures as carrying no
lengthscales at all, which the caller cannot tell apart from a clean fit.
"""


def _axis_names(idx: int, size: int, n_kernels: int) -> list[str]:
    """Name each scalar of one block kernel's lengthscale parameter.

    Under ``kernels.build_partition`` each structure carries exactly one kernel
    per entry of ``BLOCKS``, in that order, so the parameter index is the block.
    Anything else is a partition this function does not know how to label, and
    the index is reported verbatim rather than guessed at.

    Args:
        idx: Position of the block kernel within its container.
        size: Number of lengthscale scalars the kernel carries.
        n_kernels: Number of block kernels found under the same container.

    Returns:
        One axis label per scalar.
    """
    stem = BLOCKS[idx] if n_kernels == len(BLOCKS) and idx < len(BLOCKS) else f"kernel{idx}"
    return [stem] if size == 1 else [f"{stem}_{d}" for d in range(size)]


def _parse_axes(
    hyperparams: dict, hp_floor: float, model_class: str
) -> tuple[float | None, int, int, list[dict]]:
    """Reconstruct applied lengthscales from one fold's raw parameters.

    The applied value is ``softplus(raw) + hp_floor`` — the ``GreaterThan``
    transform installed by ``kernels.make_kernel_from_config``. A dimension
    counts as floored when that lands within ``FLOOR_TOL`` of ``hp_floor``.

    Args:
        hyperparams: The fold's recorded ``name -> raw value`` parameter dict.
        hp_floor: The ``config.LENGTHSCALE_FLOOR`` the run was fit under.
        model_class: Outer-cell model class; ``mapsaas_*`` carries per-component
            SAAS lengthscales rather than the block partition.

    Returns:
        ``(min applied lengthscale, n floored scalars, n scalars, axis rows)``,
        or ``(None, 0, 0, [])`` when the fold records no lengthscale at all.
    """
    mins: list[float] = []
    n_floored = 0
    n_total = 0
    axis_rows: list[dict] = []

    def _consume(value, names: list[str]) -> None:
        nonlocal n_floored, n_total
        applied = _softplus(np.atleast_1d(np.asarray(value, dtype=float))) + hp_floor
        floored = applied <= hp_floor + FLOOR_TOL
        mins.append(float(applied.min()))
        n_floored += int(floored.sum())
        n_total += int(applied.size)
        for name, is_floored in zip(names, floored):
            axis_rows.append({"axis": name, "is_floored": bool(is_floored)})

    if model_class.startswith("mapsaas"):
        for key, value in hyperparams.items():
            if "raw_lengthscale" not in key:
                continue
            size = np.atleast_1d(np.asarray(value, dtype=float)).size
            _consume(value, [f"mapsaas_{i}" for i in range(size)])
        if not mins:
            return None, 0, 0, []
        return min(mins), n_floored, n_total, axis_rows

    containers: dict[tuple[str, str], dict[int, object]] = {}
    for key, value in hyperparams.items():
        match = _LENGTHSCALE_KEY.match(key)
        if match is None:
            continue
        container = (match.group("prefix") or "", match.group("container"))
        containers.setdefault(container, {})[int(match.group("idx"))] = value

    for kernels_dict in containers.values():
        n_kernels = len(kernels_dict)
        for idx, value in sorted(kernels_dict.items()):
            size = np.atleast_1d(np.asarray(value, dtype=float)).size
            _consume(value, _axis_names(idx, size, n_kernels))

    if not mins:
        return None, 0, 0, []
    return min(mins), n_floored, n_total, axis_rows


def _floor_from_run_meta(run_meta: dict) -> float | None:
    """The lengthscale floor a result was fit under.

    The floor is a single global constant applied to every fit, recorded per run
    as ``run_meta.lengthscale_floor``. Archived results that lack that key carry
    it per entry of ``run_meta.hyperprior_sets``; those entries must agree on
    one value.

    Args:
        run_meta: The result JSON's ``run_meta`` block.

    Returns:
        The floor, or ``None`` if the run applied none.

    Raises:
        ValueError: If the ``hyperprior_sets`` entries record different floors.
    """
    if "lengthscale_floor" in run_meta:
        return run_meta["lengthscale_floor"]
    floors = {
        entry.get("lengthscale_floor")
        for entry in run_meta.get("hyperprior_sets", {}).values()
    } - {None}
    if len(floors) > 1:
        raise ValueError(
            f"Archived run records several different lengthscale floors {floors}; "
            "it cannot be scanned as one floor setting."
        )
    return floors.pop() if floors else None


def scan(input_dir: Path, output_path: Path) -> Path:
    """Scan fitted cell JSONs for ARD lengthscales pinned at the floor.

    Folds with a status other than ``ok`` and runs fit without a floor are
    skipped. Writes the fold-level table to ``output_path`` and the per-axis
    table to ``AXIS_OUTPUT_PATH``, then prints the floor-hit breakdowns.

    Args:
        input_dir: Directory holding the cell result JSONs (``result_*.json``).
        output_path: Destination of the fold-level parquet.

    Returns:
        ``output_path``.

    Raises:
        AssertionError: If ``input_dir`` holds no result JSON, or an ``ok``
            fold records no parseable lengthscale.
    """
    json_paths = sorted(
        input_dir.glob("result_*.json"), key=lambda p: int(p.stem.split("_")[1])
    )
    assert json_paths, f"No result JSONs found under {input_dir}"

    rows: list[dict] = []
    axis_data: list[dict] = []
    
    for path in json_paths:
        payload = json.loads(path.read_text())
        outer = payload["outer_config"]
        run_meta = payload.get("run_meta", {})
        hp_floor = _floor_from_run_meta(run_meta)

        for inner_block in payload["inner_cells"]:
            icfg = inner_block["inner_config"]
            
            for fold in inner_block["folds"]:
                if fold.get("status", "ok") != "ok":
                    continue
                
                if hp_floor is None:
                    continue  # this run was fit without a floor

                stats = _parse_axes(fold["hyperparams"], hp_floor, outer["model_class"])
                assert stats[0] is not None, (
                    f"{path.name}: fold records no parseable lengthscale for "
                    f"{outer['kernel_structure']}/{outer['model_class']}. A silent "
                    "skip here would be reported as zero floor hits."
                )

                min_ls, n_floored, n_total, axis_rows = stats
                
                fold_row = {
                    "split_type": fold["split_type"],
                    "kernel_structure": outer["kernel_structure"],
                    "model_class": outer["model_class"],
                    "outcome_transform": outer["outcome_transform"],
                    "cat_feat": outer["cat_feat"],
                    "min_lengthscale": min_ls,
                    "n_floored_dims": n_floored,
                    "n_total_dims": n_total,
                    "floored": n_floored > 0,
                    "crpss": fold["crpss"],
                }
                rows.append(fold_row)
                
                for ar in axis_rows:
                    axis_data.append({
                        "split_type": fold["split_type"],
                        "model_class": outer["model_class"],
                        "cat_feat": outer["cat_feat"],
                        "prior_scheme": icfg.get("prior_scheme"),
                        "axis": ar["axis"],
                        "is_floored": ar["is_floored"]
                    })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    axis_df = pd.DataFrame(axis_data)
    axis_df.to_parquet(AXIS_OUTPUT_PATH, index=False)
    
    print(f"=== Lengthscale Floor Analysis ===")
    overall_floored_pct = 100.0 * axis_df["is_floored"].mean()
    print(f"Overall % of (fold x ARD-dimension) pairs pinned at the floor: {overall_floored_pct:.2f}%")
    
    # Exclude mapsaas from structural breakdown
    structural_df = axis_df[~axis_df["axis"].str.startswith("mapsaas")]
    
    print("\n--- Breakdown by Sub-block/Axis ---")
    axis_gb = structural_df.groupby("axis")["is_floored"].mean() * 100.0
    for ax, val in axis_gb.sort_values(ascending=False).items():
        print(f"  {ax}: {val:.2f}%")
        
    print("\n--- Breakdown by Hyperprior Set ---")
    hp_gb = structural_df.groupby("prior_scheme", dropna=False)["is_floored"].mean() * 100.0
    for hp, val in hp_gb.sort_values(ascending=False).items():
        print(f"  {hp}: {val:.2f}%")
        
    print("\n--- Breakdown by Model Class ---")
    mc_gb = structural_df.groupby("model_class")["is_floored"].mean() * 100.0
    for mc, val in mc_gb.sort_values(ascending=False).items():
        print(f"  {mc}: {val:.2f}%")
        
    print("\n--- Breakdown by Catalyst Featurization ---")
    cat_gb = structural_df.groupby("cat_feat")["is_floored"].mean() * 100.0
    for cat, val in cat_gb.sort_values(ascending=False).items():
        print(f"  {cat}: {val:.2f}%")
        
    # Call out highest floor hits
    top_axes = axis_gb.sort_values(ascending=False).head(3)
    print(f"\n[!] Most floored axes: {', '.join(top_axes.index.tolist())} (these are driving the collapse)")

    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    scan(args.input_dir, args.output_path)

if __name__ == "__main__":
    main()
