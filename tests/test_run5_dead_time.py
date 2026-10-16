"""Dead time of an ordered catalyst and the cross-scored expansion record of the run_5 BO loop."""

import pytest

from pablo.mechthild_runs.run_5_scope_extension_v2.config import EXPANSION_CRITERIA
from pablo.mechthild_runs.run_5_scope_extension_v2.nhc_problem import load_nhc_space
from pablo.mechthild_runs.run_5_scope_extension_v2.run_bo_experiment import (
    EXPANSION_PICKS,
    run_bo_experiment,
)

DELAY = 2


@pytest.fixture(scope="module")
def random_run():
    """A short random-mode run with a one-fantasy tree."""
    return run_bo_experiment(
        "Ackley", seed=0, operating_mode="random",
        n_eval=DELAY + 3, source_a_availability_delay=DELAY, num_fantasies=[1, 1, 1, 1],
        space=load_nhc_space(),
    )


def test_ordered_catalyst_is_dead_for_delay_batches(random_run):
    """A catalyst ordered after batch t is released at the start of batch t + delay + 1."""
    ordered = random_run["selected_source_a_indices"]
    released = random_run["released_source_a_by_iteration"]
    for t in range(2):
        assert released[t + DELAY + 1] == [ordered[t]]
    assert all(released[i] == [] for i in range(DELAY + 1))


def test_every_pick_is_scored_under_every_criterion(random_run):
    """Each iteration records five locked picks, scored under the four criteria, and orders its mode's pick."""
    available = set(random_run["initial_available_catalyst_indices"])
    for t, (picks, scores) in enumerate(zip(random_run["expansion_picks"], random_run["expansion_scores"])):
        assert set(picks) == set(EXPANSION_PICKS) == set(scores)
        assert all(set(row) == set(EXPANSION_CRITERIA) for row in scores.values())
        assert all(p["catalyst_index"] not in available for p in picks.values())
        assert random_run["selected_source_a_indices"][t] == picks["random"]["catalyst_index"]


def test_start_set_is_the_recorded_design(random_run):
    """The run starts from every recorded experiment, replicates included."""
    space = load_nhc_space()
    assert random_run["n_start"] == space.start_x.shape[0]
    assert (space.start_replicate_id >= 0).sum() > 0
    assert all(0.0 <= y <= 1.0 for y in random_run["start_y_observed"])
