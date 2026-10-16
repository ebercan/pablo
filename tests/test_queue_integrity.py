"""Queue init: stranded cells are requeued, and a foreign result is refused."""

from __future__ import annotations

import json

import pytest

from pablo.featurization_and_kernel_scan.cross_validation import scheduler


def _cell_config(cell_id: int, cat_feat: str = "mordred") -> dict:
    return {
        "outer_cell_id": cell_id,
        "cat_feat": cat_feat,
        "solv_feat": "OHE",
        "kernel_structure": "KS1",
        "model_class": "st_y",
        "outcome_transform": "raw",
        "base_kernel": "rbf",
        "mean_function": "constant",
        "inner_cells": [{"inner_id": 0, "base_feat": "OHE", "prior_scheme": "P1"}],
    }


def _result_payload(cell_id: int, n_ok: int, cat_feat: str = "mordred") -> dict:
    folds = [
        {"split_type": "primary", "repeat": 0, "fold": i, "status": "ok"}
        for i in range(n_ok)
    ]
    return {
        "outer_cell_id": cell_id,
        "outer_config": _cell_config(cell_id, cat_feat),
        "inner_cells": [{
            "inner_id": 0,
            "inner_config": {"inner_id": 0, "base_feat": "OHE", "prior_scheme": "P1"},
            "folds": folds,
        }],
    }


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    """Point the scheduler's module-level paths at a temporary run namespace."""
    queue_dir = tmp_path / "queue"
    results_dir = tmp_path / "cells"
    inputs_path = tmp_path / "inputs.json"
    for sub in ("pending", "running", "done"):
        (queue_dir / sub).mkdir(parents=True)
    (queue_dir / "run_state").mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(scheduler, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(scheduler, "RUN_STATE_DIR", queue_dir / "run_state")
    monkeypatch.setattr(scheduler, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(scheduler, "INPUTS_PATH", inputs_path)
    monkeypatch.setattr(scheduler, "FOLD_TOTAL", 10)
    return {"queue": queue_dir, "results": results_dir, "inputs": inputs_path}


def _write_inputs(queue, n_cells: int, cat_feat: str = "mordred") -> None:
    queue["inputs"].write_text(
        json.dumps({str(i): _cell_config(i, cat_feat) for i in range(n_cells)})
    )


def test_fresh_queue_creates_one_pending_lock_per_cell(queue) -> None:
    """A namespace with no results queues every cell exactly once."""
    _write_inputs(queue, 3)
    scheduler.init_queue()
    assert sorted(p.name for p in (queue["queue"] / "pending").glob("*.lock")) == [
        "job_0.lock", "job_1.lock", "job_2.lock"
    ]


def test_completed_cell_is_marked_done(queue) -> None:
    """A cell with every fold ok gets a done lock and no pending work."""
    _write_inputs(queue, 1)
    (queue["results"] / "result_0.json").write_text(json.dumps(_result_payload(0, 10)))
    scheduler.init_queue()
    assert (queue["queue"] / "done" / "job_0.lock").exists()
    assert not list((queue["queue"] / "pending").glob("*.lock"))


def test_stranded_cell_is_requeued_rather_than_skipped(queue) -> None:
    """A running lock with no result and no heartbeat is moved back to pending.

    A worker killed before its first checkpoint leaves exactly this state;
    ``init_queue`` must re-queue the cell so it is not stranded for the sweep.
    """
    _write_inputs(queue, 1)
    (queue["queue"] / "running" / "job_0.lock").touch()
    scheduler.init_queue()
    assert (queue["queue"] / "pending" / "job_0.lock").exists()
    assert not (queue["queue"] / "running" / "job_0.lock").exists()


def test_live_worker_keeps_its_cell(queue) -> None:
    """A fresh heartbeat means a worker is mid-cell; init must not take it away."""
    import time

    _write_inputs(queue, 1)
    (queue["queue"] / "running" / "job_0.lock").touch()
    (queue["queue"] / "run_state" / "job_0.json").write_text(
        json.dumps({"outer_cell_id": 0, "status": "running", "updated_at_epoch": time.time()})
    )
    scheduler.init_queue()
    assert (queue["queue"] / "running" / "job_0.lock").exists()
    assert not (queue["queue"] / "pending" / "job_0.lock").exists()


def test_partial_cell_is_queued_for_resume(queue) -> None:
    """A checkpointed but unfinished cell comes back as pending."""
    _write_inputs(queue, 1)
    (queue["results"] / "result_0.json").write_text(json.dumps(_result_payload(0, 4)))
    scheduler.init_queue()
    assert (queue["queue"] / "pending" / "job_0.lock").exists()


def test_result_from_a_different_grid_is_refused(queue) -> None:
    """A previous grid's results sitting in the live namespace must fail loudly.

    results/cells and data/grid are one namespace keyed by integer cell id, so
    enumerating a new grid over a finished one would otherwise credit the old
    cell 0 to the new cell 0 and never fit it.
    """
    _write_inputs(queue, 1, cat_feat="grover")
    (queue["results"] / "result_0.json").write_text(
        json.dumps(_result_payload(0, 10, cat_feat="mordred"))
    )
    with pytest.raises(ValueError, match="different configuration"):
        scheduler.init_queue()


def test_result_from_another_prior_scheme_is_refused(queue) -> None:
    """Prior-sweep cells share every outer key; a P2 result at a P1 cell id still fails loudly."""
    _write_inputs(queue, 1, cat_feat="grover")
    payload = _result_payload(0, 10, cat_feat="grover")
    payload["inner_cells"][0]["inner_config"]["prior_scheme"] = "P2"
    (queue["results"] / "result_0.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prior_scheme"):
        scheduler.init_queue()


def test_matching_result_is_accepted(queue) -> None:
    """The guard compares configurations, it does not reject every stored result."""
    _write_inputs(queue, 1, cat_feat="grover")
    (queue["results"] / "result_0.json").write_text(
        json.dumps(_result_payload(0, 10, cat_feat="grover"))
    )
    scheduler.init_queue()
    assert (queue["queue"] / "done" / "job_0.lock").exists()
