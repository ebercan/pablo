"""Replicate-group detection and the replicate-anchored noise prior.

Replicate groups are experiments run at the *same* condition. They are the only
direct measurement of experimental repeatability in the study, so they are what
the GP's observation-noise prior is anchored to. Nothing here is hard-coded to a
particular set of groups: the groups are detected from the condition columns, so
adding replicates to the design sheet widens the evidence and — by construction —
tightens the prior.

Two facts make exact grouping wrong. Conditions are recorded to the precision
they were dialled in at (a repeat of a 2.993 eq run is logged as 2.990 eq), and
component names are typed by hand (``1,4-Dioxane`` and ``1.4-dioxane`` are the
same solvent). ``detect_replicate_groups`` therefore matches on normalized
labels and per-column numeric tolerances, and unions rows transitively so a
chain of near-equal rows forms one group.

The prior itself is the conjugate posterior for a variance under a Jeffreys
reference prior. Pooling the within-group sums of squares gives ``S`` over ``ν``
degrees of freedom, and

    sigma^2 | replicates  ~  InverseGamma(alpha = nu/2, beta = S/2)

so the location is the pooled replicate variance and the spread is fixed by how
much replicate evidence exists — the coefficient of variation is sqrt(2/(nu-4)),
which falls as groups are added and is never chosen by hand.

A GP's noise term is homoskedastic on the scale the GP is fitted on: its target
(yield, conversion or selectivity) after the outcome transform. ``S`` is
therefore pooled on that same scale, one evidence object per (target, transform)
pair (:func:`noise_evidence_by_scale`). The model builders divide ``beta`` by
the variance of the training targets on that scale, which is what moves the
prior onto the standardized scale the GP is fitted on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pablo.utils.outcome_transforms import OUTCOME_TRANSFORM_NAMES, forward

CONDITION_LABEL_COLUMNS: tuple[str, ...] = ("catalyst_id", "solvent", "base")
"""Condition columns compared as exact, normalized labels."""

NOISE_TARGET_COLUMNS: dict[str, str] = {
    "Y": "yield_pct",
    "C": "conversion",
    "S": "selectivity",
}
"""Experiments-table column, in percent, behind each modelled target label."""

CONDITION_NUMERIC_TOLERANCES: dict[str, float] = {
    "fa_ratio": 0.01,
    "loading": 0.01,
    "temperature": 0.01,
    "time": 0.0,
}
"""Relative tolerance per numeric condition column.

``fa_ratio`` and ``loading`` are dialled in to three or four decimals and a
repeat is logged at the precision of its own weighing, so 2.990 eq against
2.993 eq (0.1%) is the same condition. ``temperature`` is a setpoint with
block-to-block drift of well under a degree. ``time`` is an integer hour count
and must match exactly.
"""

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize_label(value: object) -> str:
    """Canonicalize a hand-typed component name for comparison.

    Case, punctuation and whitespace carry no meaning in these columns, so they
    are stripped: ``"1,4-Dioxane"``, ``"1.4-dioxane"`` and ``"1,4 Dioxane"`` all
    map to ``"14dioxane"``.

    Args:
        value: A cell from a label column.

    Returns:
        The normalized label.
    """
    return _NON_ALPHANUMERIC.sub("", str(value).strip().lower())


def _numerically_equal(left: pd.Series, right: pd.Series) -> bool:
    """Whether two rows agree on every numeric condition within tolerance."""
    for column, rtol in CONDITION_NUMERIC_TOLERANCES.items():
        a, b = float(left[column]), float(right[column])
        if not np.isclose(a, b, rtol=rtol, atol=0.0):
            return False
    return True


def detect_replicate_groups(experiments: pd.DataFrame) -> np.ndarray:
    """Assign a replicate-group id to every experiment.

    Rows are bucketed by their normalized label columns, then unioned within a
    bucket whenever every numeric condition agrees within
    ``CONDITION_NUMERIC_TOLERANCES``. Union is transitive, so a chain of
    near-equal rows forms a single group. Group ids are assigned in order of
    first appearance; a condition run only once gets ``-1``.

    Args:
        experiments: One row per experiment, carrying ``CONDITION_LABEL_COLUMNS``
            and the keys of ``CONDITION_NUMERIC_TOLERANCES``.

    Returns:
        ``(n,)`` int64 array of group ids, ``-1`` where the condition is unique.

    Raises:
        ValueError: If a required condition column is absent.
    """
    required = set(CONDITION_LABEL_COLUMNS) | set(CONDITION_NUMERIC_TOLERANCES)
    missing = sorted(required - set(experiments.columns))
    if missing:
        raise ValueError(f"experiments is missing condition columns {missing}")

    n = len(experiments)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    buckets: dict[tuple[str, ...], list[int]] = {}
    for position, (_, row) in enumerate(experiments.iterrows()):
        key = tuple(normalize_label(row[c]) for c in CONDITION_LABEL_COLUMNS)
        buckets.setdefault(key, []).append(position)

    rows = [row for _, row in experiments.iterrows()]
    for members in buckets.values():
        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1:]:
                if _numerically_equal(rows[a], rows[b]):
                    union(a, b)

    members_by_root: dict[int, list[int]] = {}
    for position in range(n):
        members_by_root.setdefault(find(position), []).append(position)

    group_ids = np.full(n, -1, dtype=np.int64)
    next_id = 0
    for root in sorted(members_by_root):
        members = members_by_root[root]
        if len(members) < 2:
            continue
        group_ids[members] = next_id
        next_id += 1
    return group_ids


def cross_check_against_comments(
    experiments: pd.DataFrame, group_ids: np.ndarray, comments: pd.Series
) -> None:
    """Verify detected groups against the design sheet's ``"<row> repeated"`` notes.

    The design sheet annotates a repeat with the 1-based spreadsheet row of the
    experiment it repeats. That is an independent record of the same fact the
    condition matching derives, so it is checked rather than trusted: a repeat
    whose note points outside its own detected group means the tolerances or the
    label normalization are wrong, and silently keeping either answer would put
    a mis-estimated noise prior into every fit.

    Args:
        experiments: The experiment rows, positionally aligned with ``group_ids``.
        group_ids: Output of :func:`detect_replicate_groups`.
        comments: The sheet's comment column, positionally aligned, whose index
            is the 0-based sheet row.

    Raises:
        ValueError: If an annotated repeat is not grouped with its referenced row.
    """
    header_offset = 2
    problems: list[str] = []
    for position, note in enumerate(comments.tolist()):
        match = re.search(r"(\d+)\s+repeated", str(note))
        if match is None:
            continue
        referenced = int(match.group(1)) - header_offset
        if not 0 <= referenced < len(experiments):
            problems.append(f"row {position}: note points to sheet row {match.group(1)}, out of range")
            continue
        if group_ids[position] < 0 or group_ids[position] != group_ids[referenced]:
            problems.append(
                f"row {position} is annotated as repeating row {referenced} but was "
                f"grouped as {group_ids[position]} vs {group_ids[referenced]}"
            )
    if problems:
        raise ValueError(
            "Replicate detection disagrees with the design sheet's repeat notes:\n  "
            + "\n  ".join(problems)
        )


@dataclass(frozen=True)
class ReplicateNoiseEvidence:
    """Pooled within-replicate variance and the InverseGamma it implies.

    Every variance-valued field is in squared units of the scale the values were
    pooled on (see :func:`noise_evidence_by_scale`).

    Attributes:
        n_groups: Number of replicate groups with at least two members.
        n_experiments: Experiments belonging to those groups.
        dof: Pooled degrees of freedom, ``sum(n_i - 1)``.
        sum_squares: Pooled within-group sum of squares.
        variance: ``sum_squares / dof`` — the pooled repeatability variance.
        alpha: ``dof / 2``, the InverseGamma concentration.
        beta: ``sum_squares / 2``, the InverseGamma rate.
    """

    n_groups: int
    n_experiments: int
    dof: int
    sum_squares: float
    variance: float
    alpha: float
    beta: float

    @property
    def prior_mean(self) -> float:
        """Prior mean ``beta / (alpha - 1)``; requires ``dof > 2``."""
        return self.beta / (self.alpha - 1.0)

    @property
    def prior_cv(self) -> float:
        """Prior coefficient of variation ``sqrt(2 / (dof - 4))``; requires ``dof > 4``."""
        return float(np.sqrt(2.0 / (self.dof - 4.0)))

    def summary(self) -> str:
        """One-line human-readable summary in units of the pooled scale."""
        return (
            f"{self.n_groups} replicate groups / {self.n_experiments} experiments, "
            f"dof={self.dof}, sd={np.sqrt(self.variance):.4g}, "
            f"InvGamma(alpha={self.alpha:.2f}, beta={self.beta:.3e}), "
            f"prior mean sd={np.sqrt(self.prior_mean):.4g}, CV={self.prior_cv:.2f}"
        )


MIN_REPLICATE_DOF: int = 5
"""Smallest pooled degrees of freedom that yields a usable prior.

``alpha = dof/2`` must exceed 2 for the InverseGamma to have finite variance,
so ``dof >= 5``. Below that the replicates do not pin the noise scale at all and
the study should not pretend otherwise.
"""


def estimate_noise_evidence(
    values: np.ndarray, group_ids: np.ndarray
) -> ReplicateNoiseEvidence:
    """Pool within-replicate variance into an InverseGamma noise prior.

    Under a Jeffreys reference prior ``p(sigma^2) ∝ 1/sigma^2``, observing a
    pooled within-group sum of squares ``S`` over ``nu`` degrees of freedom gives
    the posterior ``InverseGamma(nu/2, S/2)``. Using it as the GP's noise prior
    means the location is the measured repeatability and the spread is set by how
    much replicate evidence exists — ``CV = sqrt(2/(nu-4))`` — rather than chosen.

    Args:
        values: ``(n,)`` observed target values on the scale the noise prior is
            for — the scale the GP is fitted on.
        group_ids: ``(n,)`` replicate-group ids, ``-1`` for non-replicates.

    Returns:
        The pooled evidence and the InverseGamma parameters it implies, in
        squared units of ``values``.

    Raises:
        ValueError: If a replicate-group member is NaN or infinite, or if the
            pooled degrees of freedom fall below ``MIN_REPLICATE_DOF``.
    """
    values = np.asarray(values, dtype=float)
    group_ids = np.asarray(group_ids, dtype=np.int64)

    in_groups = group_ids >= 0
    bad = np.flatnonzero(in_groups & ~np.isfinite(values))
    if bad.size:
        raise ValueError(
            f"Replicate-group rows {bad.tolist()} have non-finite values "
            f"{values[bad].tolist()}; their within-group spread is undefined."
        )

    sum_squares = 0.0
    dof = 0
    n_groups = 0
    n_experiments = 0
    for gid in sorted(set(group_ids.tolist()) - {-1}):
        members = values[group_ids == gid]
        if len(members) < 2:
            continue
        sum_squares += float(((members - members.mean()) ** 2).sum())
        dof += len(members) - 1
        n_groups += 1
        n_experiments += len(members)

    if dof < MIN_REPLICATE_DOF:
        raise ValueError(
            f"Replicate evidence is too thin for a noise prior: dof={dof} "
            f"(need >= {MIN_REPLICATE_DOF}) from {n_groups} groups."
        )

    return ReplicateNoiseEvidence(
        n_groups=n_groups,
        n_experiments=n_experiments,
        dof=dof,
        sum_squares=sum_squares,
        variance=sum_squares / dof,
        alpha=dof / 2.0,
        beta=sum_squares / 2.0,
    )


def noise_evidence_by_scale(
    experiments: pd.DataFrame,
) -> dict[tuple[str, str], ReplicateNoiseEvidence]:
    """Replicate noise evidence for every (target, outcome transform) pair.

    Each target column is converted from percent to a fraction and mapped with
    :func:`pablo.utils.outcome_transforms.forward` — the same values the GP is
    trained on — before the within-group sums of squares are pooled.

    Args:
        experiments: One row per experiment, carrying ``replicate_id`` and the
            columns of :data:`NOISE_TARGET_COLUMNS` in percent.

    Returns:
        ``{(target, transform): evidence}`` over the targets of
        :data:`NOISE_TARGET_COLUMNS` and every name in
        :data:`pablo.utils.outcome_transforms.OUTCOME_TRANSFORM_NAMES`.

    Raises:
        ValueError: If a required column is absent, or as
            :func:`estimate_noise_evidence` for any pair.
    """
    required = {"replicate_id"} | set(NOISE_TARGET_COLUMNS.values())
    missing = sorted(required - set(experiments.columns))
    if missing:
        raise ValueError(f"experiments is missing columns {missing}")

    group_ids = experiments["replicate_id"].to_numpy(dtype=np.int64)
    evidence: dict[tuple[str, str], ReplicateNoiseEvidence] = {}
    for target, column in NOISE_TARGET_COLUMNS.items():
        fraction = torch.tensor(
            experiments[column].to_numpy(dtype=np.float64) / 100.0, dtype=torch.float64
        )
        for name in OUTCOME_TRANSFORM_NAMES:
            evidence[(target, name)] = estimate_noise_evidence(
                forward(fraction, name).numpy(), group_ids
            )
    return evidence


def replicate_anchor_values(values: torch.Tensor, group_ids: np.ndarray) -> torch.Tensor:
    """Value each row's measurement variance is evaluated at: its group mean for replicates.

    Args:
        values: ``(n,)`` raw fractions of one target (yield, conversion, ...).
        group_ids: ``(n,)`` replicate-group ids, ``-1`` for singletons.

    Returns:
        ``(n,)`` tensor: the replicate-group mean for rows in a group, the row's
        own value otherwise.
    """
    out = values.clone()
    for gid in sorted(set(group_ids.tolist()) - {-1}):
        mask = torch.as_tensor(group_ids == gid)
        out[mask] = values[mask].mean()
    return out
