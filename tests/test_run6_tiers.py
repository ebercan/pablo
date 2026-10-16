"""Tests for run_6's tier blocks: index decoding, tier-filtered claiming and per-tier SLURM settings."""

import pytest

from pablo.mechthild_runs.run_6_af_optimizer_af_value import config, scheduler, submit_mechthild
from pablo.mechthild_runs.run_6_af_optimizer_af_value.main_variant import decode_worker


def test_first_block_keeps_the_original_layout():
    """Indices 0-29 decode as optimizer → tier → rep over tiers 0-4, the layout of the finished results."""
    for idx in range(30):
        opt_idx, rem = divmod(idx, 5 * config.N_REPS)
        tier, rep = divmod(rem, config.N_REPS)
        assert decode_worker(idx) == (config.OPTIMIZERS[opt_idx], tier, rep)


def test_every_unit_has_exactly_one_index():
    """The 42 indices cover every (optimizer, tier, rep) once, and the index range is enforced."""
    units = [decode_worker(i) for i in range(config.N_WORKERS)]
    expected = {(o, t, r) for o in config.OPTIMIZERS for t in range(config.N_TIERS) for r in range(config.N_REPS)}
    assert config.N_WORKERS == 42 and set(units) == expected and len(units) == len(expected)
    assert [decode_worker(i)[1] for i in range(30, 42)] == [5, 5, 5, 6, 6, 6] * 2
    with pytest.raises(ValueError):
        decode_worker(config.N_WORKERS)


def test_worker_claims_only_its_tiers(tmp_path, monkeypatch):
    """A worker started for tier 6 skips lower-indexed tier-5 jobs and claims the first tier-6 job."""
    pending, running = tmp_path / "pending", tmp_path / "running"
    pending.mkdir()
    running.mkdir()
    for idx in (30, 31, 33, 40):
        (pending / f"job_{idx}.lock").touch()
    monkeypatch.setattr(scheduler, "PENDING", pending)
    monkeypatch.setattr(scheduler, "RUNNING", running)
    assert scheduler.pending_ids({6}) == [33, 40]
    assert scheduler._claim_next({6}) == 33
    assert (running / "job_33.lock").exists() and not (pending / "job_33.lock").exists()
    assert scheduler._claim_next(None) == 30


def test_mixed_slurm_settings_are_rejected():
    """Tiers with different partitions, limits or memory cannot share one array."""
    assert submit_mechthild.slurm_settings([5]) == ("short", "01:45:00", "24G")
    assert submit_mechthild.slurm_settings([6]) == ("medium", "03:30:00", "40G")
    assert submit_mechthild.slurm_settings([0, 1, 2, 3, 4]) == ("short", "00:45:00", "16G")
    with pytest.raises(ValueError):
        submit_mechthild.slurm_settings([5, 6])
