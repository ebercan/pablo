"""Build ``data/libraries/experiments.parquet`` from the design spreadsheet.

``nhc_doe.xlsx`` is the authoritative record and is read-only here; this module
is the one place that turns its ``Experimental Design`` sheet into the tidy
table the scan consumes. Solvent and base names are canonicalized against the
component libraries so a hand-typed variant cannot silently become a new
component, and replicate groups are detected from the conditions rather than
maintained by hand (see :mod:`core.replicates`).

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.build_experiments
    python -m pablo.featurization_and_kernel_scan.build_experiments --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan import paths
from pablo.utils.catalyst_form_classifier import classify_library
from pablo.featurization_and_kernel_scan.core.replicates import (
    ReplicateNoiseEvidence,
    cross_check_against_comments,
    detect_replicate_groups,
    noise_evidence_by_scale,
    normalize_label,
)

DESIGN_SHEET = "Experimental Design"

EXCEL_TO_COLUMN: dict[str, str] = {
    "Batch": "batch",
    "Catalyst": "catalyst_id",
    "Solvent": "solvent",
    "Base": "base",
    "Formaldehyde Ratio in Eq.": "fa_ratio",
    "Catalyst Loading in Molar Ratio": "loading",
    "Temperature": "temperature",
    "Time": "time",
    "Yield": "yield_pct",
    "Conversion": "conversion",
    "Selectivity": "selectivity",
}
"""Design-sheet header to tidy column name."""

NO_BASE_LABEL = "no"
"""The design sheet's marker for a run with no external base."""

BASE_REQUIRING_FORM = "salt"
"""The only pre-catalyst form that needs an external base to be activated.

``base_active`` is a property of the *catalyst*, not of what was pipetted: CO2
adducts, bicarbonate salts and free NHCs self-activate, so their base block is
zeroed and the base dimension switched off even on the one run where a base was
added anyway. Deriving the flag from the base column instead would turn that run
into the only non-salt row carrying a live base, which no other row can be
compared against."""

COLUMN_ORDER: list[str] = [
    "batch", "catalyst_id", "solvent", "base", "fa_ratio", "loading",
    "temperature", "time", "yield_pct", "conversion", "selectivity",
    "replicate_id", "base_active",
]


def _canonicalize(values: pd.Series, canonical: list[str], what: str) -> pd.Series:
    """Map hand-typed component names onto their library spelling.

    Args:
        values: The sheet column.
        canonical: Library spellings to match against.
        what: Component name, for the error message.

    Returns:
        The column rewritten in library spelling.

    Raises:
        ValueError: If a value matches no library entry.
    """
    lookup = {normalize_label(name): name for name in canonical}
    out, unknown = [], []
    for value in values:
        key = normalize_label(value)
        if key not in lookup:
            unknown.append(str(value))
            out.append(value)
        else:
            out.append(lookup[key])
    if unknown:
        raise ValueError(
            f"{what} values in the design sheet match no library entry: "
            f"{sorted(set(unknown))}. Known: {sorted(canonical)}"
        )
    return pd.Series(out, index=values.index)


def _base_active(catalyst_ids: pd.Series, libraries_dir: Path) -> pd.Series:
    """Whether each experiment's catalyst needs an external base.

    Args:
        catalyst_ids: The experiments' catalyst ids.
        libraries_dir: Directory holding ``catalysts.parquet``.

    Returns:
        Int64 series, 1 where the pre-catalyst form is a true ionic salt.
    """
    catalysts = pd.read_parquet(libraries_dir / "catalysts.parquet")
    form_map = classify_library(catalysts["id"].tolist(), catalysts["smiles"].tolist())
    return catalyst_ids.map(
        lambda cid: int(form_map[cid] == BASE_REQUIRING_FORM)
    ).astype(np.int64)


def build_experiments_table(excel_path: Path, libraries_dir: Path) -> pd.DataFrame:
    """Read the design sheet into the tidy experiments table.

    Rows without a recorded yield are dropped: they are suggested-but-unrun
    candidates written back by the batch pipeline, not observations.

    Args:
        excel_path: Path to ``nhc_doe.xlsx``.
        libraries_dir: Directory holding ``catalysts.parquet``, ``solvents.parquet``
            and ``bases.parquet``.

    Returns:
        The table in ``COLUMN_ORDER``, indexed 0..n-1 in sheet order.

    Raises:
        ValueError: If a required header is absent or a component is unknown.
    """
    raw = pd.read_excel(excel_path, sheet_name=DESIGN_SHEET)
    missing = [c for c in EXCEL_TO_COLUMN if c not in raw.columns]
    if missing:
        raise ValueError(f"{excel_path} sheet {DESIGN_SHEET!r} is missing columns {missing}")

    comments = raw["Comments"] if "Comments" in raw.columns else pd.Series([""] * len(raw))
    keep = raw["Yield"].notna() & raw["Catalyst"].notna()
    raw, comments = raw[keep].reset_index(drop=True), comments[keep].reset_index(drop=True)

    df = raw[list(EXCEL_TO_COLUMN)].rename(columns=EXCEL_TO_COLUMN)

    solvents = pd.read_parquet(libraries_dir / "solvents.parquet")
    bases = pd.read_parquet(libraries_dir / "bases.parquet")
    df["solvent"] = _canonicalize(df["solvent"], solvents["name"].tolist(), "Solvent")
    df["base"] = _canonicalize(
        df["base"], bases["comment"].tolist() + [NO_BASE_LABEL], "Base"
    )

    for column in ("batch", "catalyst_id", "temperature", "time", "yield_pct", "conversion"):
        df[column] = df[column].astype(np.int64)
    for column in ("fa_ratio", "loading", "selectivity"):
        df[column] = df[column].astype(float)

    group_ids = detect_replicate_groups(df)
    cross_check_against_comments(df, group_ids, comments)
    df["replicate_id"] = group_ids
    df["base_active"] = _base_active(df["catalyst_id"], libraries_dir)
    return df[COLUMN_ORDER]


def write_noise_estimate(
    df: pd.DataFrame,
    evidence: dict[tuple[str, str], ReplicateNoiseEvidence],
    out_path: Path,
) -> None:
    """Record the replicate groups and the noise priors they imply.

    A readable companion to ``experiments.parquet``: the priors are recomputed
    live at import in ``core.config``, so this file is documentation, never an
    input. Writing it from the same call that writes the table is what keeps it
    from drifting into describing a set of replicate groups that no longer
    exists.

    Args:
        df: The experiments table.
        evidence: Pooled replicate evidence for that table per
            ``(target, outcome_transform)``, from
            :func:`core.replicates.noise_evidence_by_scale`.
        out_path: Destination JSON.
    """
    groups = []
    for gid in sorted(set(df["replicate_id"]) - {-1}):
        rows = df[df["replicate_id"] == gid]
        y = rows["yield_pct"].to_numpy() / 100.0
        groups.append({
            "group_id": int(gid),
            "n": int(len(rows)),
            "catalyst_id": int(rows["catalyst_id"].iloc[0]),
            "solvent": str(rows["solvent"].iloc[0]),
            "base": str(rows["base"].iloc[0]),
            "batches": rows["batch"].tolist(),
            "yields_pct": rows["yield_pct"].tolist(),
            "conversions_pct": rows["conversion"].tolist(),
            "selectivities_pct": rows["selectivity"].tolist(),
            "var_raw": float(np.var(y, ddof=1)),
        })
    yield_raw = evidence[("Y", "raw")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_experiments": int(len(df)),
        "groups": groups,
        "pooled_dof": yield_raw.dof,
        "inverse_gamma_alpha": yield_raw.alpha,
        "prior_cv": yield_raw.prior_cv,
        "pooled_sd_yield_pct": float(100.0 * np.sqrt(yield_raw.variance)),
        "by_scale": {
            f"{target}/{transform}": {
                "sum_squares": ev.sum_squares,
                "variance": ev.variance,
                "sd": float(np.sqrt(ev.variance)),
                "inverse_gamma_beta": ev.beta,
                "prior_mean_variance": ev.prior_mean,
            }
            for (target, transform), ev in evidence.items()
        },
        "note": (
            "InverseGamma(nu/2, S/2) per target (Y yield, C conversion, S "
            "selectivity) and outcome transform: the conjugate posterior for the "
            "noise variance under a Jeffreys reference prior, given the "
            "within-replicate sum of squares S pooled over nu degrees of freedom on "
            "the scale the GP is trained on. Model builders divide beta by the "
            "variance of the training targets on that scale."
        ),
    }, indent=2))


def main() -> None:
    """Rebuild ``experiments.parquet`` and report the replicate evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=paths.PKG_ROOT.parent / "nhc_doe.xlsx")
    parser.add_argument("--out", type=Path, default=paths.EXPERIMENTS_PARQUET)
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args()

    df = build_experiments_table(args.excel, args.out.parent)
    evidence = noise_evidence_by_scale(df)

    print(f"{len(df)} experiments from {args.excel}")
    print("Replicate evidence per target / outcome transform:")
    for (target, transform), ev in evidence.items():
        print(f"  {target}/{transform}: {ev.summary()}")
    for gid in sorted(set(df['replicate_id']) - {-1}):
        rows = df[df["replicate_id"] == gid]
        print(
            f"  group {gid}: cat {rows['catalyst_id'].iloc[0]} / {rows['solvent'].iloc[0]} / "
            f"{rows['base'].iloc[0]} -> yields {rows['yield_pct'].tolist()}"
        )

    if args.dry_run:
        print("[dry-run] nothing written")
        return
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")
    write_noise_estimate(df, evidence, paths.NOISE_ESTIMATE_JSON)
    print(f"Wrote {paths.NOISE_ESTIMATE_JSON}")


if __name__ == "__main__":
    main()
