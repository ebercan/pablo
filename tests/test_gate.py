"""Stage gates: paired fold bootstrap and the two vetoes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan.results_analysis.gate import (
    GATE_FAMILIES,
    _crossed_nuisance_columns,
    apply_gate,
    bootstrap_stability,
    calibration_veto,
    completeness_veto,
)
from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import SPLIT_FAMILIES

FAMILY_SIZES = {"primary": 20, "leave_one_out": 20, "leave_one_solvent_out": 7}


def _frame(arm_offsets: dict[str, float], *, msll: dict[str, float] | None = None,
           nuisance: tuple[str, ...] = ("a", "b"), seed: int = 0) -> pd.DataFrame:
    """Synthetic fold-level aggregate: each arm sits at a known CRPSS offset."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, n_folds in FAMILY_SIZES.items():
        for fold in range(n_folds):
            for nuis in nuisance:
                for arm, offset in arm_offsets.items():
                    rows.append({
                        "split_type": family,
                        "repeat": 0,
                        "fold": fold,
                        "mean_function": arm,
                        "solv_feat": nuis,
                        "cat_feat": "stub",
                        "crpss": offset + rng.normal(0.0, 0.01),
                        "msll": (msll or {}).get(arm, -0.2),
                    })
    return pd.DataFrame(rows)


def test_gate_families_are_real_split_families() -> None:
    """The gate cannot name a family the aggregate will never contain."""
    assert set(GATE_FAMILIES) <= set(SPLIT_FAMILIES)
    assert "leave_one_type_out" not in GATE_FAMILIES
    assert "leave_one_scaffold_out" not in GATE_FAMILIES


def test_nuisance_excludes_columns_determined_by_the_axis() -> None:
    """A column fixed by the axis is a relabelling of it, not a crossed factor."""
    df = _frame({"m1": 0.2, "m2": 0.1})
    df["base_kernel"] = df["mean_function"].map({"m1": "rbf", "m2": "tanimoto"})
    crossed = _crossed_nuisance_columns(df, ["mean_function"])
    assert "solv_feat" in crossed
    assert "base_kernel" not in crossed
    assert "cat_feat" not in crossed


def test_a_clearly_better_arm_wins_almost_every_resample() -> None:
    """A separation far larger than the fold noise is resolved by the bootstrap."""
    out = bootstrap_stability(_frame({"good": 0.40, "bad": 0.05}), ["mean_function"], n_boot=400)
    good = out.set_index("arm").loc["good"]
    assert good["gem_gate"] > out.set_index("arm").loc["bad", "gem_gate"]
    assert good["p_top_m"] > 0.99
    assert out.set_index("arm").loc["bad", "p_worse_than_leader"] > 0.99


def test_arms_inside_the_noise_both_stay_plausible() -> None:
    """Arms separated by less than fold noise are both carried, not split on rank."""
    out = apply_gate(_frame({"m1": 0.200, "m2": 0.199}), ["mean_function"], n_boot=400)
    assert out["carried"].all()
    assert out["p_top_m"].min() > 0.10


def test_top_m_widens_the_carry_set() -> None:
    """Asking for the top three admits arms that never rank first."""
    offsets = {"a": 0.40, "b": 0.30, "c": 0.20, "d": 0.10}
    narrow = bootstrap_stability(_frame(offsets), ["mean_function"], n_boot=400, top_m=1)
    wide = bootstrap_stability(_frame(offsets), ["mean_function"], n_boot=400, top_m=3)
    assert (wide["p_top_m"] >= narrow["p_top_m"]).all()
    assert (wide["p_top_m"] > 0.5).sum() > (narrow["p_top_m"] > 0.5).sum()


def test_calibration_veto_drops_an_overconfident_arm() -> None:
    """A top-ranked arm with median MSLL >= 0 is not carried however good its CRPSS."""
    df = _frame({"sharp": 0.50, "honest": 0.10}, msll={"sharp": 0.15, "honest": -0.25})
    veto = calibration_veto(df, ["mean_function"]).set_index("arm")
    assert bool(veto.loc["sharp", "vetoed_calibration"])
    assert not bool(veto.loc["honest", "vetoed_calibration"])
    gated = apply_gate(df, ["mean_function"], n_boot=200, min_carry=1).set_index("arm")
    assert not bool(gated.loc["sharp", "carried"])
    assert bool(gated.loc["honest", "carried"])


def test_completeness_veto_drops_an_arm_missing_folds() -> None:
    """An arm short of folds is flagged rather than scored as if it were whole."""
    df = _frame({"whole": 0.3, "short": 0.3})
    df = df.drop(df.index[(df["mean_function"] == "short")][:5])
    veto = completeness_veto(df, ["mean_function"]).set_index("arm")
    assert bool(veto.loc["short", "vetoed_incomplete"])
    assert not bool(veto.loc["whole", "vetoed_incomplete"])


def test_incomplete_design_raises_rather_than_comparing_different_folds() -> None:
    """An unbalanced arm x nuisance x fold design fails loudly."""
    df = _frame({"m1": 0.3, "m2": 0.3})
    df = df[~((df["mean_function"] == "m1") & (df["solv_feat"] == "b"))]
    with pytest.raises(ValueError, match="complete design"):
        bootstrap_stability(df, ["mean_function"], n_boot=10)


def test_min_carry_prevents_collapsing_onto_one_arm() -> None:
    """The gate never returns a single survivor when the floor says two."""
    df = _frame({"a": 0.60, "b": 0.10, "c": 0.05})
    gated = apply_gate(df, ["mean_function"], n_boot=200, min_carry=2)
    assert int(gated["carried"].sum()) >= 2


def test_gate_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs at the same seed agree exactly, so a gate decision is reproducible."""
    df = _frame({"a": 0.3, "b": 0.25})
    first = bootstrap_stability(df, ["mean_function"], n_boot=200, seed=7)
    second = bootstrap_stability(df, ["mean_function"], n_boot=200, seed=7)
    pd.testing.assert_frame_equal(first, second)
