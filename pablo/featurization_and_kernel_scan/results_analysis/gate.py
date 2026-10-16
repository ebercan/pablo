"""Stage gates for a staged scan — which arms of an axis survive to the next stage.

A single gem score, bootstrapped over the folds it is built from, is too noisy
to pick one winning configuration out of thousands on 69 experiments, and a
significance test eliminates next to nothing. The gate therefore keeps every arm
that could plausibly be best, and rests on two choices.

**Gate on the axis marginal, not on cells.** When a stage settles one axis, every
arm of that axis is scored pooled over the nuisance levels it shares folds with
(catalysts, solvents, ...). Five catalysts times two solvents is ten paired
comparisons per arm on identical folds, not one.

**Gate on families that can carry a median.** The gate gem of an arm is the min
over ``GATE_FAMILIES`` of the per-family median of a fold-level metric (CRPSS by
default). ``GATE_FAMILIES`` are the three families with enough folds to estimate
a median: ``primary`` (60), ``leave_one_out`` (60) and
``leave_one_solvent_out`` (7). ``leave_one_type_out`` has two folds and
``leave_one_scaffold_out`` three, so their medians bootstrap to noise;
``leave_one_replicate_out`` has five and is the noisiest family (SD 0.31). The
gate gem is distinct from the ``gem`` of ``gem_scores``, which is the min of
pooled CRPSS over ``primary`` and ``leave_one_out``; ``gem_scores`` reports all
six families.

The gate is three filters, applied in order:

* **R1 calibration veto** — median MSLL ≥ 0 is dropped outright. A model no
  better calibrated than a constant baseline will mislead the acquisition
  function however sharp its point predictions are. This is a correctness
  requirement, not a noise judgement.
* **R2 completeness veto** — an arm with fewer ok folds than the fullest arm is
  dropped. A failed fit must never be indistinguishable from a successful one.
* **R3 stability set** — bootstrap the folds, re-rank, and carry every arm that
  lands in the top ``top_m`` in at least ``min_probability`` of resamples. This
  is an explicit "could plausibly be best" filter, not a significance test.
  At least ``min_carry`` arms that pass R1 and R2 are always carried.

Run from the repo root::

    python -m pablo.featurization_and_kernel_scan.results_analysis.gate --axis cat_feat
    python -m pablo.featurization_and_kernel_scan.results_analysis.gate --axis mean_function outcome_transform --top-m 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import (
    CONFIG_COLUMNS,
    SPLIT_FAMILIES,
    ok_folds,
)

GATE_FAMILIES: tuple[str, ...] = (
    "primary",
    "leave_one_out",
    "leave_one_solvent_out",
)
"""Families a config may be eliminated on. The other three are reported only."""

MSLL_VETO_THRESHOLD: float = 0.0
"""Median MSLL at or above which an arm is dropped regardless of its CRPSS."""

DEFAULT_N_BOOT: int = 4000
DEFAULT_MIN_PROBABILITY: float = 0.10
DEFAULT_AGGREGATE = paths.SCAN_AGGREGATE_PARQUET

FOLD_KEY_COLUMNS: tuple[str, ...] = ("split_type", "repeat", "fold")


def _fold_keys(df: pd.DataFrame) -> pd.Series:
    """Identify each CV fold uniquely within its split family."""
    return (
        df["split_type"].astype(str)
        + "|" + df["repeat"].astype(str)
        + "|" + df["fold"].astype(str)
    )


def _arm_labels(df: pd.DataFrame, axis_columns: list[str]) -> pd.Series:
    """Join the axis columns into one printable arm label."""
    return df[axis_columns].astype(str).agg(" · ".join, axis=1)


def _crossed_nuisance_columns(df: pd.DataFrame, axis_columns: list[str]) -> list[str]:
    """Config columns that vary independently of the axis and can be pooled over.

    A column carrying only one value in the whole frame contributes nothing, and
    one that is functionally determined by the axis — a single value per arm — is
    a relabelling of the axis rather than a factor crossed with it. Including
    either would make the arm × nuisance × fold design incomplete by construction.

    Args:
        df: Fold-level aggregate.
        axis_columns: Columns naming the axis being settled.

    Returns:
        The nuisance columns to pool over, in ``CONFIG_COLUMNS`` order.
    """
    candidates = [
        c for c in CONFIG_COLUMNS if c in df.columns and c not in axis_columns
    ]
    varying = [c for c in candidates if df[c].nunique(dropna=False) > 1]
    if not varying:
        return []
    determined = (
        df.groupby(axis_columns, dropna=False)[varying].nunique(dropna=False).max() <= 1
    )
    return [c for c in varying if not determined[c]]


def _family_matrices(
    df: pd.DataFrame, axis_columns: list[str], families: tuple[str, ...], metric: str
) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    """Build one complete observation matrix per family.

    For each family the matrix has shape ``(n_observations, n_arms)``, where an
    observation is one (nuisance level × fold) pair, so every arm is scored on
    exactly the same folds and the bootstrap below is paired.

    Args:
        df: Fold-level aggregate restricted to the gate families.
        axis_columns: Columns naming the axis being settled.
        families: Split families to include.
        metric: Fold-level metric column.

    Returns:
        ``(arms, matrices, fold_row_groups)`` — the arm labels in matrix column
        order; per family the ``(n_obs, n_arms)`` value matrix; and per family a
        list of row-index arrays, one per distinct fold.

    Raises:
        ValueError: If the arm × nuisance × fold design is incomplete, which
            would make the paired comparison compare different folds.
    """
    nuisance = _crossed_nuisance_columns(df, axis_columns)
    work = df.assign(
        _arm=_arm_labels(df, axis_columns),
        _fold=_fold_keys(df),
        _nuisance=(
            df[nuisance].astype(str).agg(" · ".join, axis=1) if nuisance else "-"
        ),
    )
    arms = sorted(work["_arm"].unique())
    matrices: dict[str, np.ndarray] = {}
    fold_row_groups: dict[str, list[np.ndarray]] = {}

    for family in families:
        sub = work[work["split_type"] == family]
        table = sub.pivot_table(
            index=["_nuisance", "_fold"], columns="_arm", values=metric, aggfunc="mean"
        )
        table = table.reindex(columns=arms)
        n_missing = int(table.isna().sum().sum())
        if n_missing:
            raise ValueError(
                f"family {family!r}: {n_missing} missing (arm × nuisance × fold) "
                f"observations out of {table.size}, with nuisance factors "
                f"{nuisance}. The gate needs a complete design so every arm is "
                "scored on identical folds; run the missing cells, or move the "
                "unbalanced factor into --axis."
            )
        matrices[family] = table.to_numpy(dtype=float)
        fold_of_row = table.index.get_level_values("_fold").to_numpy()
        fold_row_groups[family] = [
            np.flatnonzero(fold_of_row == f) for f in pd.unique(fold_of_row)
        ]
    return arms, matrices, fold_row_groups


def _gem_from_rows(
    matrices: dict[str, np.ndarray],
    families: tuple[str, ...],
    rows: dict[str, np.ndarray],
) -> np.ndarray:
    """Gem per arm: min over families of the median over the selected rows."""
    per_family = np.stack(
        [np.median(matrices[f][rows[f], :], axis=0) for f in families]
    )
    return per_family.min(axis=0)


def bootstrap_stability(
    df: pd.DataFrame,
    axis_columns: list[str],
    *,
    families: tuple[str, ...] = GATE_FAMILIES,
    metric: str = "crpss",
    n_boot: int = DEFAULT_N_BOOT,
    top_m: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap the folds and report how often each arm of an axis could win.

    Folds are resampled with replacement within each family, so family sizes are
    preserved and every arm is re-scored on the same resampled folds.

    Args:
        df: Fold-level aggregate, one row per fold.
        axis_columns: Columns naming the axis being settled.
        families: Families the gem is minimised over.
        metric: Fold-level metric column, higher-is-better.
        n_boot: Bootstrap resamples.
        top_m: Rank an arm has to reach to count as a win.
        seed: RNG seed.

    Returns:
        One row per arm: observed gem, bootstrap SD, ``p_top_m`` (fraction of
        resamples in which the arm ranks within ``top_m``) and
        ``p_worse_than_leader``, sorted by observed gem descending.
    """
    present = tuple(f for f in families if (df["split_type"] == f).any())
    if not present:
        raise ValueError(
            f"none of the gate families {families} appear in split_type; got "
            f"{sorted(df['split_type'].unique())}"
        )
    arms, matrices, fold_row_groups = _family_matrices(
        df[df["split_type"].isin(present)], axis_columns, present, metric
    )
    all_rows = {f: np.arange(matrices[f].shape[0]) for f in present}
    observed = _gem_from_rows(matrices, present, all_rows)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(arms)), dtype=float)
    for b in range(n_boot):
        rows = {}
        for f in present:
            groups = fold_row_groups[f]
            picked = rng.integers(0, len(groups), len(groups))
            rows[f] = np.concatenate([groups[i] for i in picked])
        draws[b] = _gem_from_rows(matrices, present, rows)

    threshold = np.partition(draws, -top_m, axis=1)[:, -top_m][:, None]
    leader = int(np.argmax(observed))
    return pd.DataFrame(
        {
            "arm": arms,
            "gem_gate": observed,
            "boot_sd": draws.std(axis=0),
            "p_top_m": (draws >= threshold).mean(axis=0),
            "p_worse_than_leader": (draws[:, leader][:, None] > draws).mean(axis=0),
        }
    ).sort_values("gem_gate", ascending=False, ignore_index=True)


def calibration_veto(
    df: pd.DataFrame, axis_columns: list[str], *, threshold: float = MSLL_VETO_THRESHOLD
) -> pd.DataFrame:
    """Median MSLL per arm, and whether it fails the calibration veto (R1).

    Args:
        df: Fold-level aggregate, one row per fold, with an ``msll`` column.
        axis_columns: Columns naming the axis being settled.
        threshold: Median MSLL at or above which an arm is vetoed.

    Returns:
        One row per arm with columns ``arm``, ``msll_median`` and the boolean
        ``vetoed_calibration``.
    """
    work = df.assign(_arm=_arm_labels(df, axis_columns))
    msll = work.groupby("_arm")["msll"].median().rename("msll_median")
    return (
        msll.reset_index()
        .rename(columns={"_arm": "arm"})
        .assign(vetoed_calibration=lambda t: t["msll_median"] >= threshold)
    )


def completeness_veto(df: pd.DataFrame, axis_columns: list[str]) -> pd.DataFrame:
    """Fold count per arm, and whether it is short of the fullest arm (R2).

    Args:
        df: Fold-level aggregate, one row per fold.
        axis_columns: Columns naming the axis being settled.

    Returns:
        One row per arm with columns ``arm``, ``n_folds`` (rows of ``df`` for
        that arm) and the boolean ``vetoed_incomplete``, true when ``n_folds``
        is below the largest ``n_folds`` of any arm.
    """
    work = df.assign(_arm=_arm_labels(df, axis_columns))
    counts = work.groupby("_arm").size().rename("n_folds")
    return (
        counts.reset_index()
        .rename(columns={"_arm": "arm"})
        .assign(vetoed_incomplete=lambda t: t["n_folds"] < int(t["n_folds"].max()))
    )


def apply_gate(
    df: pd.DataFrame,
    axis_columns: list[str],
    *,
    families: tuple[str, ...] = GATE_FAMILIES,
    metric: str = "crpss",
    n_boot: int = DEFAULT_N_BOOT,
    top_m: int = 1,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    min_carry: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Run R1, R2 and R3 over one axis and mark which arms are carried forward.

    Args:
        df: Fold-level aggregate for the stage just completed.
        axis_columns: Columns naming the axis being settled.
        families: Families the gate gem is minimised over.
        metric: Fold-level metric column, higher-is-better.
        n_boot: Bootstrap resamples for R3.
        top_m: Rank an arm has to reach in a resample to count as a win.
        min_probability: R3 carry threshold on ``p_top_m``.
        min_carry: Never carry fewer than this many arms, so a stage cannot
            collapse onto a single arm that the noise does not support.
        seed: RNG seed.

    Returns:
        One row per arm with the gate columns and a boolean ``carried``, sorted
        by observed gem descending.
    """
    stability = bootstrap_stability(
        df, axis_columns, families=families, metric=metric,
        n_boot=n_boot, top_m=top_m, seed=seed,
    )
    out = (
        stability
        .merge(calibration_veto(df, axis_columns), on="arm", how="left")
        .merge(completeness_veto(df, axis_columns), on="arm", how="left")
    )
    survives_vetoes = ~(out["vetoed_calibration"] | out["vetoed_incomplete"])
    carried = survives_vetoes & (out["p_top_m"] >= min_probability)
    if int(carried.sum()) < min_carry:
        eligible = out.index[survives_vetoes]
        top_up = out.loc[eligible].sort_values("gem_gate", ascending=False).index[:min_carry]
        carried = carried | out.index.isin(top_up)
    out["carried"] = carried
    return out.sort_values("gem_gate", ascending=False, ignore_index=True)


def main() -> None:
    """Gate one axis of a completed stage and print the carry set."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument(
        "--axis", nargs="+", required=True,
        help="Config columns naming the axis being settled, e.g. --axis mean_function",
    )
    parser.add_argument("--metric", default="crpss")
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--min-probability", type=float, default=DEFAULT_MIN_PROBABILITY)
    parser.add_argument("--min-carry", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Optional parquet path.")
    args = parser.parse_args()

    df = ok_folds(pd.read_parquet(args.aggregate))
    missing = [c for c in args.axis if c not in df.columns]
    if missing:
        raise ValueError(f"axis columns absent from the aggregate: {missing}")

    gated = apply_gate(
        df, list(args.axis), metric=args.metric, n_boot=args.n_boot,
        top_m=args.top_m, min_probability=args.min_probability,
        min_carry=args.min_carry, seed=args.seed,
    )
    reported_only = [f for f in SPLIT_FAMILIES if f not in GATE_FAMILIES]
    print(f"axis={args.axis}  metric={args.metric}  top_m={args.top_m}  n_boot={args.n_boot}")
    print(f"gate families : {list(GATE_FAMILIES)}")
    print(f"reported only : {reported_only}")
    print(gated.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    carried = gated.loc[gated["carried"], "arm"].tolist()
    print(f"\ncarried forward ({len(carried)}/{len(gated)}): {carried}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        gated.to_parquet(args.out, index=False)
        print(f"Wrote {len(gated)} rows -> {args.out}")


if __name__ == "__main__":
    main()
