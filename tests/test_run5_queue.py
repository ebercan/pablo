"""File-lock queue of run_5: init, one-job workers, stale requeue and result checks."""

import json
import subprocess

import pytest

from pablo.mechthild_runs.run_5_scope_extension_v2 import mechthild_scheduler_cpu as sched
from pablo.mechthild_runs.run_5_scope_extension_v2.run_state import (
    check_existing_result,
    get_run_state_path,
    load_run_state,
    update_run_state,
)

INPUTS = {
    str(i): {"operating_mode": mode, "strategy": strategy, "n_eval": 3, "label": f"job{i}"}
    for i, (mode, strategy) in enumerate(
        [("random", "random"), ("myopic", "myopic"), ("random", "random"), ("averaged", "fantasized")]
    )
}


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """A random-mode queue over a four-combination inputs table, with job runs faked by exit code."""
    results = tmp_path / "results_random"
    results.mkdir()
    monkeypatch.setattr(sched, "load_inputs", lambda: INPUTS)
    monkeypatch.setattr(sched, "results_dir", lambda mode: results)
    exit_codes: list[int] = []
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "mv":
            return real_run(cmd, **kwargs)
        return subprocess.CompletedProcess(cmd, exit_codes.pop(0))

    monkeypatch.setattr(sched.subprocess, "run", fake_run)
    return {"path": tmp_path / "queue", "results": results, "exit_codes": exit_codes}


def _locks(queue_path, sub):
    return sorted(p.name for p in (queue_path / sub).glob("*.lock"))


def test_init_creates_pending_locks_of_the_mode_only(queue):
    """init writes one pending lock and one state per combination of the mode."""
    sched.initialize_queue(queue["path"], "random")
    assert _locks(queue["path"], "pending") == ["job_0.lock", "job_2.lock"]
    assert load_run_state(get_run_state_path(queue["path"], 0), 0, 3)["status"] == "pending_waiting"


def test_worker_runs_one_job_and_records_success_and_failure(queue):
    """A worker runs exactly one job; success moves the lock to done, failure back to pending."""
    sched.initialize_queue(queue["path"], "random")
    queue["exit_codes"].extend([0, 1])
    sched.run_worker(queue["path"], "random")
    assert _locks(queue["path"], "done") == ["job_0.lock"]
    assert _locks(queue["path"], "pending") == ["job_2.lock"]
    sched.run_worker(queue["path"], "random")
    assert _locks(queue["path"], "pending") == ["job_2.lock"]
    state = load_run_state(get_run_state_path(queue["path"], 2), 2, 3)
    assert state["status"] == "pending_failed" and state["attempts"] == 1 and state["last_return_code"] == 1
    sched.run_worker(queue["path"], "myopic")
    assert queue["exit_codes"] == []


def test_init_rejects_done_lock_without_result(queue):
    """A done lock left from an earlier queue, with no result file, stops init."""
    (queue["path"] / "done").mkdir(parents=True)
    (queue["path"] / "done" / "job_0.lock").touch()
    with pytest.raises(ValueError, match="no result file"):
        sched.initialize_queue(queue["path"], "random")


def test_requeue_stale_returns_jobs_of_dead_workers(queue, monkeypatch):
    """A running job whose SLURM job is gone returns to pending; a live one stays."""
    sched.initialize_queue(queue["path"], "random")
    for idx, job_id in ((0, "111"), (2, "222")):
        (queue["path"] / "pending" / f"job_{idx}.lock").rename(queue["path"] / "running" / f"job_{idx}.lock")
        update_run_state(get_run_state_path(queue["path"], idx), idx, 3, status="running", slurm_job_id=job_id)
    monkeypatch.setattr(sched, "squeue_lines", lambda fields: ["222"])
    sched.requeue_stale(queue["path"], "random")
    assert _locks(queue["path"], "pending") == ["job_0.lock"]
    assert _locks(queue["path"], "running") == ["job_2.lock"]


def test_check_existing_result_rejects_shortened_and_foreign_results(tmp_path):
    """Only a full run of the same combination counts as finished."""
    path = tmp_path / "result_0.json"
    full = {"n_eval_override": None, "num_fantasies_override": None}
    path.write_text(json.dumps({"combination": INPUTS["0"], "execution_options": full}))
    check_existing_result(path, INPUTS["0"])
    with pytest.raises(ValueError, match="not for the current combination"):
        check_existing_result(path, INPUTS["2"])
    path.write_text(json.dumps({"combination": INPUTS["0"], "execution_options": {**full, "n_eval_override": 1}}))
    with pytest.raises(ValueError, match="shortened run"):
        check_existing_result(path, INPUTS["0"])
