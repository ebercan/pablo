"""Rewrite the xTB HOMO, LUMO and gap columns from the relaxed-geometry block of the saved logs.

xTB prints a full property block for the *input* geometry before an ``--opt``
run starts. ``xtb_homo_eV``, ``xtb_lumo_eV`` and ``xtb_gap_eV`` must come from
the block after the final ``GEOMETRY OPTIMIZATION CONVERGED``, which is what
``_parse_xtb_output`` reads. ``xtb_dipole_D`` and
``xtb_polarizability_angst3`` are not printed in the input-geometry block, so
they serve as the control: they must come out bit-identical.

This module re-parses the saved logs under ``paths.XTB_OUTPUTS_DIR`` with
``_parse_xtb_output`` and, on ``--write``, rewrites
``results_checkpoint.parquet`` and patches the three columns in place in the
assembled featurizations. It never runs xTB and never touches
``paths.XTB_INPUTS_DIR``. Default is a dry run that only reports.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.features.interpretable.repair_xtb_scalars
    python -m pablo.featurization_and_kernel_scan.features.interpretable.repair_xtb_scalars --write
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
    XTB_CHECKPOINT,
    _parse_xtb_output,
)

AFFECTED_COLS = ("xtb_homo_eV", "xtb_lumo_eV", "xtb_gap_eV")
CONTROL_COLS = ("xtb_dipole_D", "xtb_polarizability_angst3")
FEATURIZATIONS = ("interpretable", "interpretable_light", "interpretable_dG")


def reparse_logs() -> pd.DataFrame:
    """Re-parse every saved xTB log from its final converged optimization block.

    Returns:
        ``(n_catalysts, 5)`` frame indexed by ``catalyst_id`` over the five xTB
        scalar columns, one row per ``C<id>.txt`` under ``XTB_OUTPUTS_DIR``.
    """
    rows: dict[int, dict[str, float]] = {}
    for log in sorted(paths.XTB_OUTPUTS_DIR.glob("C*.txt")):
        cid = int(log.stem[1:])
        parsed = _parse_xtb_output(log.read_text())
        assert parsed is not None, f"{log}: no converged optimisation block."
        rows[cid] = parsed
    assert rows, f"No xTB logs found under {paths.XTB_OUTPUTS_DIR}."
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "catalyst_id"
    return df


def _report(label: str, old: pd.DataFrame, new: pd.DataFrame, cols: tuple[str, ...]) -> None:
    print(f"\n{label}")
    print(f"  {'column':28s} {'n_changed':>9s} {'MAE':>9s} {'max|d|':>9s} {'old range':>20s} {'new range':>20s}")
    for c in cols:
        if c not in old.columns:
            continue
        d = (new[c] - old[c]).abs()
        print(
            f"  {c:28s} {int((d > 1e-9).sum()):9d} {d.mean():9.4f} {d.max():9.4f}"
            f"   [{old[c].min():7.3f},{old[c].max():7.3f}]   [{new[c].min():7.3f},{new[c].max():7.3f}]"
        )


def repair(write: bool) -> None:
    """Regenerate the checkpoint and patch the assembled featurizations.

    Args:
        write: When ``False`` only report what would change. When ``True``
            rewrite ``XTB_CHECKPOINT`` and the affected columns of every
            featurization in ``FEATURIZATIONS`` that carries them.
    """
    new = reparse_logs()
    assert XTB_CHECKPOINT.exists(), f"Missing {XTB_CHECKPOINT}."
    old = pd.read_parquet(XTB_CHECKPOINT)
    old.index = old.index.astype(int)
    assert set(old.index) == set(new.index), (
        f"Checkpoint covers {len(old)} catalysts, logs cover {len(new)}; "
        f"symmetric difference {sorted(set(old.index) ^ set(new.index))}."
    )
    new = new.reindex(old.index)[list(old.columns)]

    for c in CONTROL_COLS:
        assert np.allclose(old[c], new[c], atol=0.0, rtol=0.0), (
            f"Control column {c} moved; the repair touched more than the parser bug."
        )
    print(f"control columns {CONTROL_COLS}: bit-identical across {len(old)} catalysts.")
    _report(f"checkpoint {XTB_CHECKPOINT.name}", old, new, AFFECTED_COLS)

    patches: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for name in FEATURIZATIONS:
        path = paths.FEAT_DIR / name / "descriptors.parquet"
        assert path.exists(), f"Missing {path}."
        feat = pd.read_parquet(path)
        cols = [c for c in AFFECTED_COLS if c in feat.columns]
        assert cols, f"{name} carries none of {AFFECTED_COLS}; nothing to patch."
        aligned = new.reindex(feat.index.astype(int))
        assert not aligned[cols].isna().any().any(), f"{name}: catalysts missing from logs."
        patched = feat.copy()
        for c in cols:
            patched[c] = aligned[c].to_numpy()
        assert list(patched.columns) == list(feat.columns)
        _report(f"{name}/descriptors.parquet", feat, patched, tuple(cols))
        patches.append((name, path, patched))

    if not write:
        print("\ndry run: nothing written. Re-run with --write to apply.")
        return

    new.to_parquet(XTB_CHECKPOINT)
    print(f"\nwrote {XTB_CHECKPOINT}")
    for name, path, patched in patches:
        patched.to_parquet(path)
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="Apply the repair. Default is a dry run.")
    args = parser.parse_args()
    repair(write=args.write)


if __name__ == "__main__":
    main()
