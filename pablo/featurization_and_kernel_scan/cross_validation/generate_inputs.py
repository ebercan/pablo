"""Enumerate one run's grid and write its ``inputs.json``.

``--run <name>`` picks the grid from ``config.RUN_GRIDS`` and the output from
``paths.run_paths(name).inputs_json``: ``elimination`` (the default) writes
``data/grid/inputs.json``, every other run ``runs/<name>/inputs.json``.
Nothing here hardcodes an axis or a level — to change a grid, edit its entry.

**Elimination grid: a union of one-axis sweeps.** Per catalyst level the
configurations are the baseline plus, for every swept axis, the baseline with
that axis moved to each of its other levels. The baseline is part of every sweep
and is enumerated once. A configuration belongs to sweep ``A`` when it differs
from the baseline on ``A`` alone (``sweep_membership``); the baseline belongs to
all of them. Cells are level-major in ``cat_feats`` order, then baseline first
and the sweeps in ``sweeps`` order.

**Crossed grid: the cartesian product.** Every catalyst level with every
combination of the levels of ``CrossedGrid.axes``. Outer-cell ids are assigned
by projected cost, most expensive first (``order_by_cost``), so workers, which
claim the lowest pending id, start the longest cells first.

**Listed grid: named configurations.** Every catalyst level with every
configuration of ``ListedGrid.configs``, ids assigned by projected cost as for a
crossed grid.

**Anchored grid: anchors crossed with variant axes.** Every anchor of
``AnchoredGrid.anchors`` is a full configuration with its own catalyst level;
every combination of the ``AnchoredGrid.variants`` levels replaces those axes in
it. Ids are assigned by projected cost.

**One outer cell per configuration.** Every cell carries exactly one inner cell,
so axes on the inner record (base block, prior scheme) still give one worker per
configuration.

**The catalyst axis is ``cat_feat``.** Its levels are featurization directory
names, so ``grover`` and ``grover_pca`` are two levels of one axis. The
``config.CATALYST_SETS`` entry and the form-one-hot flag are derived from the
name (``features.dim_reduction.source_name``).

``inputs.json`` schema::

    {
      "0": {
        "outer_cell_id": 0,
        "cat_feat": "clamp",            # the featurization directory to load
        "solv_feat": "OHE",
        "kernel_structure": "KS1",
        "model_class": "st_y",
        "outcome_transform": "raw",
        "base_kernel": "matern52",
        "mean_function": "constant",
        "append_form_onehot_to_cat": false,
        "inner_cells": [
          {"inner_id": 0, "base_feat": "OHE", "prior_scheme": "P1"}
        ]
      },
      "1": {...}, ...
    }

Every other prior value — the LogNormal σ and location shift, the lengthscale
floor, the replicate-derived noise prior, the outputscale Gamma, the mean-module
σ — is a single global constant in ``config.py``, not a grid axis.

CLI::

    python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs
    python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --dry-run
    python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --cat-set fingerprint
    python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --run cross_1
    python -m pablo.featurization_and_kernel_scan.cross_validation.generate_inputs --run monroe_m3

Every invocation overwrites the run's ``inputs.json``, so a narrowed
``--cat-set`` clobbers the full grid; finish with the default.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core import config
from pablo.featurization_and_kernel_scan.core.config import (
    CatalystSpec,
    AnchoredGrid,
    CrossedGrid,
    EliminationGrid,
    ListedGrid,
)

Grid = EliminationGrid | CrossedGrid | ListedGrid | AnchoredGrid
from pablo.featurization_and_kernel_scan.features.dim_reduction import (
    is_reduced,
    source_name,
)

OUTER_AXES: tuple[str, ...] = (
    "solv_feat",
    "kernel_structure",
    "model_class",
    "outcome_transform",
    "base_kernel",
    "mean_function",
)
"""Axes stored on the outer-cell record, in ``inputs.json`` key order."""

INNER_AXES: tuple[str, ...] = ("base_feat", "prior_scheme")
"""Axes stored on the inner-cell record."""


def catalyst_spec(cat_feat: str) -> CatalystSpec:
    """The ``config.CATALYST_SETS`` entry a catalyst level resolves to.

    Args:
        cat_feat: A catalyst level, e.g. ``"grover"`` or ``"grover_pca"``.

    Returns:
        The entry of the level's full-dimensional source.

    Raises:
        ValueError: If the source is not in ``config.CATALYST_SETS["all"]``.
    """
    name = source_name(cat_feat)
    for spec in config.CATALYST_SETS["all"]:
        if spec.name == name:
            return spec
    raise ValueError(f"{cat_feat!r} resolves to {name!r}, which is not in CATALYST_SETS['all']")


def select_cat_feats(
    grid: Grid, cat_set: str | None
) -> tuple[str, ...]:
    """The grid's catalyst levels, optionally narrowed to one catalyst family.

    Args:
        grid: The grid being enumerated.
        cat_set: A key of ``config.CATALYST_SETS``, or ``None`` for every level.

    Returns:
        The selected levels, in the grid's order.
    """
    if cat_set is None:
        return grid.cat_feats
    family = {spec.name for spec in config.CATALYST_SETS[cat_set]}
    return tuple(c for c in grid.cat_feats if source_name(c) in family)


def check_cat_feats(cat_feats: Sequence[str]) -> None:
    """Fail before submission on a catalyst level that cannot be loaded.

    Args:
        cat_feats: Catalyst levels, e.g. ``("grover", "grover_pca")``.

    Raises:
        ValueError: If a level does not resolve to a catalyst featurization, or
            names a ``<name>_pca/`` directory that has not been built.
    """
    for cat_feat in cat_feats:
        spec = catalyst_spec(cat_feat)
        if is_reduced(cat_feat) and not (paths.FEAT_DIR / cat_feat / "descriptors.parquet").exists():
            raise ValueError(
                f"{cat_feat} has not been built. Run "
                "`python -m pablo.featurization_and_kernel_scan.features.__main__ "
                f"build {cat_feat}` or features.dim_reduction.build_all([{spec.name!r}])."
            )


def configurations(grid: Grid) -> list[dict[str, str]]:
    """The non-catalyst configurations every catalyst level is fitted under.

    Args:
        grid: The grid being enumerated.

    Returns:
        One dict per configuration over every non-catalyst axis. Elimination
        grid: the baseline first, then each swept axis's non-baseline levels in
        sweep order. Crossed grid: the cartesian product of ``grid.axes``, the
        last axis varying fastest. Listed grid: ``grid.configs`` in order.
        Anchored grid: every configuration with ``cat_feat`` included,
        anchor-major (``AnchoredGrid.configs``). No two dicts are equal.
    """
    if isinstance(grid, AnchoredGrid):
        return [dict(cfg) for cfg in grid.configs]
    if isinstance(grid, ListedGrid):
        return [dict(cfg) for cfg in grid.configs]
    if isinstance(grid, CrossedGrid):
        return [
            dict(zip(grid.axes, levels))
            for levels in itertools.product(*grid.axes.values())
        ]
    configs = [dict(grid.baseline)]
    seen = {tuple(sorted(grid.baseline.items()))}
    for axis, levels in grid.sweeps.items():
        for level in levels:
            cfg = {**grid.baseline, axis: level}
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(cfg)
    return configs


def sweep_membership(cfg: dict[str, str], grid: EliminationGrid) -> tuple[str, ...]:
    """The sweeps a configuration belongs to.

    Args:
        cfg: A configuration over the keys of ``grid.baseline``, e.g. one
            outer cell's axes merged with its inner cell's.

    Returns:
        Every swept axis for the baseline; the one axis on which it differs
        from the baseline otherwise.

    Raises:
        ValueError: If the configuration differs from the baseline on an
            unswept axis or on more than one axis.
    """
    moved = tuple(axis for axis, level in grid.baseline.items() if cfg[axis] != level)
    if not moved:
        return tuple(grid.sweeps)
    if len(moved) > 1 or moved[0] not in grid.sweeps:
        raise ValueError(f"configuration {cfg} is in no sweep (moved axes: {moved})")
    return moved


def enumerate_cells(
    cat_feats: Sequence[str], grid: Grid
) -> dict[str, dict]:
    """Build the outer-cell map: one cell per (catalyst level, configuration).

    Args:
        cat_feats: The catalyst levels to enumerate, normally ``grid.cat_feats``.
        grid: The grid being enumerated.

    Returns:
        ``outer_cell_id`` (as a string) to the cell record, level-major; for an
        anchored grid, anchor-major over the anchors whose level is in
        ``cat_feats``.

    Raises:
        ValueError: If a level cannot be loaded (``check_cat_feats``).
    """
    check_cat_feats(cat_feats)
    configs = configurations(grid)
    if isinstance(grid, AnchoredGrid):
        pairs = [(cfg["cat_feat"], cfg) for cfg in configs if cfg["cat_feat"] in cat_feats]
    else:
        pairs = [(cat_feat, cfg) for cat_feat in cat_feats for cfg in configs]
    cells: dict[str, dict] = {}
    for cat_feat, cfg in pairs:
        outer_id = len(cells)
        cells[str(outer_id)] = {
            "outer_cell_id": outer_id,
            "cat_feat": cat_feat,
            **{axis: cfg[axis] for axis in OUTER_AXES},
            "append_form_onehot_to_cat": catalyst_spec(cat_feat).append_form_onehot,
            "inner_cells": [{"inner_id": 0, **{axis: cfg[axis] for axis in INNER_AXES}}],
        }
    return cells


def order_by_cost(cells: dict[str, dict]) -> dict[str, dict]:
    """Renumber outer cells by projected cost, most expensive first.

    Args:
        cells: Outer-cell map, as ``enumerate_cells`` returns it.

    Returns:
        The same records with ``outer_cell_id`` reassigned 0..n-1 in
        non-increasing ``scheduler._per_cell_seconds`` order; equal costs keep
        their enumeration order.
    """
    from pablo.featurization_and_kernel_scan.cross_validation.scheduler import (
        _per_cell_seconds,
    )

    records = sorted(
        (cell for _, cell in sorted(cells.items(), key=lambda kv: int(kv[0]))),
        key=_per_cell_seconds,
        reverse=True,
    )
    return {
        str(new_id): {**cell, "outer_cell_id": new_id}
        for new_id, cell in enumerate(records)
    }


def build_run_cells(
    grid: Grid, cat_set: str | None = None
) -> dict[str, dict]:
    """The outer-cell map ``main`` writes for a grid.

    Args:
        grid: A value of ``config.RUN_GRIDS``.
        cat_set: Optional ``config.CATALYST_SETS`` key narrowing the catalyst axis.

    Returns:
        Elimination grid: cells in enumeration order. Crossed and listed grids:
        cells renumbered by ``order_by_cost``.
    """
    cells = enumerate_cells(select_cat_feats(grid, cat_set), grid)
    if not isinstance(grid, EliminationGrid):
        cells = order_by_cost(cells)
    return cells


def _print_summary(
    grid: Grid, cat_feats: Sequence[str], n_configs: int
) -> None:
    """Print the axes and the configuration count of an enumerated grid."""
    print(f"  cat_feat levels: {len(cat_feats)} ({sum(map(is_reduced, cat_feats))} PCA-reduced)")
    if isinstance(grid, AnchoredGrid):
        for i, anchor in enumerate(grid.anchors, start=1):
            print(f"  anchor {i:>2}: " + "  ".join(f"{k}={v}" for k, v in anchor.items()))
        for axis, levels in grid.variants.items():
            print(f"  variant {axis:<18} {len(levels)} levels {levels}")
        print(f"  configurations: {n_configs} (anchors × variant combinations)")
        return
    if isinstance(grid, ListedGrid):
        for i, cfg in enumerate(grid.configs, start=1):
            print(f"  config {i:>2}: " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
        print(f"  configurations per level: {n_configs} (listed)")
        return
    if isinstance(grid, CrossedGrid):
        for axis, levels in grid.axes.items():
            print(f"  axis {axis:<18} {len(levels)} levels {levels}")
        print(
            f"  configurations per level: {n_configs} "
            f"(product of {' × '.join(str(len(v)) for v in grid.axes.values())})"
        )
        return
    print(f"  baseline: {grid.baseline}")
    for axis, levels in grid.sweeps.items():
        print(
            f"  sweep {axis:<18} {len(levels)} levels {levels} "
            f"→ {len(cat_feats) * len(levels)} configurations"
        )
    swept = sum(len(levels) for levels in grid.sweeps.values())
    print(
        f"  configurations per level: {n_configs} unique "
        f"({swept} over the sweeps, baseline counted once)"
    )


def main() -> None:
    """Enumerate the selected run's grid and write its ``inputs.json``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cat-set", choices=sorted(config.CATALYST_SETS), default=None,
        help="Narrow the grid to the levels of one catalyst family. Clobbers inputs.json.",
    )
    parser.add_argument("--run", choices=sorted(config.RUN_GRIDS), default=config.DEFAULT_RUN,
                        help="Run whose grid to enumerate and whose inputs.json to write.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args()

    grid = config.RUN_GRIDS[args.run]
    cat_feats = select_cat_feats(grid, args.cat_set)
    cells = build_run_cells(grid, args.cat_set)
    configs = configurations(grid)
    expected = (
        sum(cfg["cat_feat"] in cat_feats for cfg in configs)
        if isinstance(grid, AnchoredGrid) else len(cat_feats) * len(configs)
    )
    assert len(cells) == expected, f"Expected {expected} outer cells, got {len(cells)}"

    print(f"run={args.run}  cat-set={args.cat_set or 'all levels'}  — {grid.note}")
    _print_summary(grid, cat_feats, len(configs))
    print(
        f"  total: {len(cells)} outer cells (one configuration each) × "
        f"{config.FOLD_TOTAL} folds = {len(cells) * config.FOLD_TOTAL:,} fits"
    )

    if args.dry_run:
        print("[dry-run] nothing written")
        return
    out_path = paths.run_paths(args.run).inputs_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cells, indent=2))
    print(f"Wrote {len(cells)} outer cells -> {out_path}")


if __name__ == "__main__":
    main()
