"""Pooled CRPSS, the failure flags, the gem score, and fold rows in the aggregate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan.results_analysis.gem_scores import (
    GEM_STUDIES,
    SPLIT_FAMILIES,
    compute_gem_scores,
)
from pablo.utils.metrics import pooled_crps_skill_score


# ── pooled CRPSS, flags and the gem ──────────────────────────────────────────


FOLDS_PER_STUDY = {
    "primary": 60,
    "leave_one_type_out": 2,
    "leave_one_scaffold_out": 3,
    "leave_one_solvent_out": 7,
    "leave_one_out": 60,
    "leave_one_replicate_out": 5,
}


def _expected(folds_per_study: dict[str, int] = FOLDS_PER_STUDY) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"split_type": s, "repeat": 0, "fold": f}
            for s, n in folds_per_study.items()
            for f in range(n)
        ]
    )


def _config_rows(
    cat_feat: str,
    crpss_by_study: dict[str, float] | None = None,
    n_failed_by_study: dict[str, int] | None = None,
    folds_per_study: dict[str, int] = FOLDS_PER_STUDY,
) -> list[dict]:
    """One-point folds with baseline CRPS 0.1 and a per-study fold CRPSS."""
    rows = []
    for study, n in folds_per_study.items():
        crpss = (crpss_by_study or {}).get(study, 0.2)
        n_failed = (n_failed_by_study or {}).get(study, 0)
        for fold in range(n):
            ok = fold >= n_failed
            rows.append({
                "cat_feat": cat_feat,
                "split_type": study,
                "repeat": 0,
                "fold": fold,
                "status": "ok" if ok else "failed",
                "crps_per_point": [0.1 * (1.0 - crpss)] if ok else [],
                "baseline_crps_per_point": [0.1] if ok else [],
                "crpss": crpss if ok else float("nan"),
                "msll": -0.1 if ok else float("nan"),
                "mae": 0.05 if ok else float("nan"),
                "rmse": 0.07 if ok else float("nan"),
                "q2": 0.3 if ok else float("nan"),
                "calibration_ks": 0.1 if ok else float("nan"),
            })
    return rows


def test_pooled_crpss_reproduces_the_worked_check():
    """Folds with model CRPS 0.05/0.08/0.30 against 0.10 each pool to -0.433."""
    assert pooled_crps_skill_score([0.05, 0.08, 0.30], [0.10, 0.10, 0.10]) == pytest.approx(
        1.0 - 0.43 / 0.30
    )
    assert pooled_crps_skill_score([0.05, 0.08, 0.30], [0.10] * 3) == pytest.approx(-0.4333, abs=1e-4)
    per_fold = [1.0 - m / 0.10 for m in (0.05, 0.08, 0.30)]
    assert float(np.median(per_fold)) == pytest.approx(0.20)


def test_gem_scores_pool_rather_than_take_the_median():
    """The same three folds, as a study inside compute_gem_scores, pool to -0.433."""
    studies = {s: (3 if s == "primary" else 1) for s in SPLIT_FAMILIES}
    rows = _config_rows("c", folds_per_study=studies)
    for row, model in zip([r for r in rows if r["split_type"] == "primary"], (0.05, 0.08, 0.30)):
        row["crps_per_point"] = [model]
    scores = compute_gem_scores(pd.DataFrame(rows), _expected(studies)).set_index("cat_feat")
    assert scores.loc["c", "crpss_pooled_primary"] == pytest.approx(-0.4333, abs=1e-4)
    assert scores.loc["c", "gem"] == pytest.approx(-0.4333, abs=1e-4)
    assert scores.loc["c", "gem_study"] == "primary"


@pytest.mark.parametrize(
    ("study", "n_failed", "ineligible"),
    [
        ("primary", 3, False),
        ("primary", 4, True),
        ("leave_one_solvent_out", 1, True),
    ],
)
def test_ineligible_threshold_is_five_percent_of_the_study(study, n_failed, ineligible):
    """60 folds tolerate 3 failures, not 4; 7 folds tolerate none."""
    rows = _config_rows("c", n_failed_by_study={study: n_failed})
    scores = compute_gem_scores(pd.DataFrame(rows), _expected()).set_index("cat_feat")
    assert scores.loc["c", f"n_failed_{study}"] == n_failed
    assert scores.loc["c", f"n_ok_{study}"] == FOLDS_PER_STUDY[study] - n_failed
    assert bool(scores.loc["c", f"ineligible_{study}"]) is ineligible
    assert bool(scores.loc["c", "ineligible"]) is ineligible


def test_any_failure_makes_a_config_not_production_ready():
    """One failed fold anywhere flags the config, and it is still scored and ranked."""
    rows = _config_rows("one_failure", n_failed_by_study={"leave_one_scaffold_out": 1})
    rows += _config_rows("clean")
    scores = compute_gem_scores(pd.DataFrame(rows), _expected()).set_index("cat_feat")
    assert bool(scores.loc["one_failure", "not_production_ready_leave_one_scaffold_out"])
    assert bool(scores.loc["one_failure", "not_production_ready"])
    assert not bool(scores.loc["one_failure", "not_production_ready_primary"])
    assert not bool(scores.loc["clean", "not_production_ready"])
    assert scores.loc["one_failure", "gem"] == pytest.approx(0.2)


def test_gem_ignores_the_four_reporting_studies():
    """A config terrible on solvent but good on primary and LOO keeps a good gem."""
    rows = _config_rows(
        "bad_solvent",
        crpss_by_study={"primary": 0.5, "leave_one_out": 0.4, "leave_one_solvent_out": -3.0},
    )
    rows += _config_rows("uniform", crpss_by_study={s: 0.1 for s in SPLIT_FAMILIES})
    scores = compute_gem_scores(pd.DataFrame(rows), _expected()).set_index("cat_feat")
    assert scores.loc["bad_solvent", "crpss_pooled_leave_one_solvent_out"] == pytest.approx(-3.0)
    assert scores.loc["bad_solvent", "gem"] == pytest.approx(0.4)
    assert scores.loc["bad_solvent", "gem_study"] == "leave_one_out"
    assert scores.index[0] == "bad_solvent"
    assert set(GEM_STUDIES) == {"primary", "leave_one_out"}


def test_failed_folds_are_dropped_from_the_pool_not_scored_as_zero():
    """Pooled CRPSS runs over the ok folds only."""
    rows = _config_rows("c", n_failed_by_study={"primary": 2})
    scores = compute_gem_scores(pd.DataFrame(rows), _expected()).set_index("cat_feat")
    assert scores.loc["c", "crpss_pooled_primary"] == pytest.approx(0.2)


def test_absent_fold_is_an_error_not_a_failure():
    """A fold neither ok nor failed stops the reduction."""
    rows = [
        r for r in _config_rows("c")
        if not (r["split_type"] == "leave_one_out" and r["fold"] == 0)
    ]
    with pytest.raises(ValueError, match="absent"):
        compute_gem_scores(pd.DataFrame(rows), _expected())


def test_legacy_aggregate_without_status_reads_as_all_ok():
    """An aggregate written before failed rows were kept scores with n_failed = 0."""
    df = pd.DataFrame(_config_rows("c")).drop(columns="status")
    scores = compute_gem_scores(df, _expected()).set_index("cat_feat")
    assert int(scores.loc["c", [f"n_failed_{s}" for s in SPLIT_FAMILIES]].sum()) == 0
    assert scores.loc["c", "gem"] == pytest.approx(0.2)


def test_other_metrics_stay_per_study_medians():
    """MSLL is reported as the median over ok folds."""
    rows = _config_rows("c")
    primary = [r for r in rows if r["split_type"] == "primary"]
    primary[0]["msll"] = 5000.0
    scores = compute_gem_scores(pd.DataFrame(rows), _expected()).set_index("cat_feat")
    assert scores.loc["c", "msll_median_primary"] == pytest.approx(-0.1)


def _fold_record(fold: int, status: str) -> dict:
    ok = status == "ok"
    nan = float("nan")
    return {
        "split_type": "primary", "repeat": 0, "fold": fold, "status": status,
        "n_train": 60, "n_test": 1, "wall_time_s": 1.0 if ok else 0.0,
        "nlpd_per_point": [0.5] if ok else [],
        "crps_per_point": [0.05] if ok else [],
        "baseline_crps_per_point": [0.1] if ok else [],
        "crpss": 0.5 if ok else nan, "msll": -0.1 if ok else nan, "q2": 0.3 if ok else nan,
        "mae": 0.05 if ok else nan, "rmse": 0.05 if ok else nan,
        "calibration_ks": 0.2 if ok else nan,
        **({} if ok else {"error": "NoiseBelowJitterError: fitted noise variance 2.00e-04"}),
    }


def _payload(statuses: list[str]) -> dict:
    return {
        "outer_cell_id": 0,
        "outer_config": {
            "cat_feat": "grover_pca", "solv_feat": "OHE", "kernel_structure": "KS1",
            "outcome_transform": "raw", "model_class": "st_y", "base_kernel": "matern52",
            "mean_function": "constant",
        },
        "inner_cells": [{
            "inner_id": 0,
            "inner_config": {"inner_id": 0, "base_feat": "OHE", "prior_scheme": "P1"},
            "folds": [_fold_record(i, s) for i, s in enumerate(statuses)],
        }],
    }


def test_concat_keeps_failed_folds_as_rows():
    """A failed fold reaches the aggregate with its status, error and NaN metrics."""
    from pablo.featurization_and_kernel_scan.cross_validation.concat_results import _flatten

    rows, n_failed = _flatten(_payload(["ok", "failed", "ok"]))
    assert n_failed == 1
    assert [r["status"] for r in rows] == ["ok", "failed", "ok"]
    failed = rows[1]
    assert failed["error"].startswith("NoiseBelowJitterError")
    assert np.isnan(failed["crpss"]) and np.isnan(failed["crps_mean"])
    assert rows[0]["crps_mean"] == pytest.approx(0.05)


def test_concat_rejects_an_unknown_fold_status():
    """Only ok and failed are fold outcomes."""
    from pablo.featurization_and_kernel_scan.cross_validation.concat_results import _flatten

    with pytest.raises(ValueError, match="unknown fold status"):
        _flatten(_payload(["ok", "running"]))
