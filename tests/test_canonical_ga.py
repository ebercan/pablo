"""Tests for the symmetry-aware canonical GA optimizers in pablo.optimize."""

import itertools

import numpy as np
import pytest
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from pymoo.core.population import Population
from pymoo.core.variable import Integer, Real

from pablo.optimize import (
    CanonicalMixedAcqFunProblem,
    _build_canonical_mating,
    CanonicalRepair,
    MixedAcqFunProblem,
    _build_canonical_layout,
    _ga_vector_to_candidates,
    optimize_acqf_mixed_canonical_GA,
)

D_CAT, D_SOL, D_BAS = 5, 3, 2
N_CAT, N_SOL, N_BAS = 7, 4, 3

# qLogEI is a set function in expectation only; its QMC estimate shifts under
# relabeling. Over the 48 relabelings of one physical batch the measured range
# is 0.017 log-units for a GA-optimized batch and 0.13 for a random one, at
# botorch's default 512 QMC samples. The bound below only has to catch a decode
# that produces a genuinely different batch.
MC_RELABEL_TOL = 0.5


def _row_multiset(candidates):
    """Rows of a ``(1, q, d)`` decode as a sorted list, for order-free comparison."""
    return sorted(tuple(row.tolist()) for row in candidates.squeeze(0))


def _make_problem(q=6, pre=0, batch_constraints=None, fixed_dims=None, seed=0):
    """Build a matched (legacy, canonical) problem pair over a small random GP."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if batch_constraints is None:
        batch_constraints = {0: (2, 2, 2)}
    if fixed_dims is None:
        fixed_dims = {3: 0.5}
    d = 4 + D_CAT + D_SOL + D_BAS
    train_x = torch.rand(12, d, dtype=torch.float64)
    train_y = torch.rand(12, 1, dtype=torch.float64)
    gp = SingleTaskGP(train_x, train_y)
    gp.eval()
    acqf = qLogExpectedImprovement(gp, best_f=train_y.max())
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(torch.float64)
    kwargs = dict(
        q=q,
        cat_emb_avail=torch.rand(N_CAT, D_CAT, dtype=torch.float64),
        sol_emb_avail=torch.rand(N_SOL, D_SOL, dtype=torch.float64),
        bas_emb_avail=torch.rand(N_BAS, D_BAS, dtype=torch.float64),
        cat_is_salt_flags=torch.tensor([True, False, True, True, False, True, False]),
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        pre_batch=torch.rand(pre, d, dtype=torch.float64) if pre else None,
    )
    legacy = MixedAcqFunProblem(acqf, bounds, **kwargs)
    canonical = CanonicalMixedAcqFunProblem(acqf, bounds, **kwargs)
    return legacy, canonical, acqf, bounds


def _sample_genome(vars_dict, rng):
    genome = {}
    for key, var in vars_dict.items():
        if isinstance(var, Real):
            genome[key] = rng.uniform(*var.bounds)
        elif isinstance(var, Integer):
            genome[key] = int(rng.integers(var.bounds[0], var.bounds[1] + 1))
        else:
            genome[key] = int(rng.choice(var.options))
    return genome


def _relabel(genome, layout, block_perm, within_swaps):
    """Apply one element of the batch symmetry group to a genome."""
    out = dict(genome)
    nvc = layout.n_var_per_cand
    slots = [
        [genome[c * nvc + j + 1] for j in range(nvc)] for c in range(layout.q)
    ]
    grouped = [
        genome[layout.grouped_offset + j + 1] for j in range(layout.n_grouped_vars)
    ]
    for b, members in enumerate(layout.blocks):
        source = layout.blocks[block_perm[b]]
        order = list(reversed(source)) if within_swaps[b] else list(source)
        for target, src in zip(members, order):
            for j, value in enumerate(slots[src]):
                out[target * nvc + j + 1] = value
        for r, start in enumerate(layout.grouped_starts):
            out[layout.grouped_offset + start + layout.block_group_idx[b][r] + 1] = (
                grouped[start + layout.block_group_idx[block_perm[b]][r]]
            )
    return out


def test_symmetry_group_order_matches_partition():
    """The derived group order is 2!^3 * 3! = 48 for (2,2,2) and q! when unconstrained."""
    _, canonical, _, _ = _make_problem(batch_constraints={0: (2, 2, 2)})
    assert canonical.layout.blocks == [[0, 1], [2, 3], [4, 5]]
    assert canonical.layout.group_order == 48

    _, unconstrained, _, _ = _make_problem(
        q=5, batch_constraints={}, fixed_dims={3: 0.5}
    )
    assert unconstrained.layout.blocks == [[0, 1, 2, 3, 4]]
    assert unconstrained.layout.group_order == 120

    _, uneven, _, _ = _make_problem(batch_constraints={0: (3, 2, 1)})
    assert uneven.layout.blocks == [[0, 1, 2], [3, 4], [5]]
    assert uneven.layout.group_order == 12


def test_vectorized_decode_matches_legacy_path():
    """The batched decoder reproduces the per-individual reconstruction exactly."""
    rng = np.random.default_rng(0)
    for pre, q in ((0, 6), (2, 8)):
        legacy, canonical, _, _ = _make_problem(q=q, pre=pre)
        population = [_sample_genome(canonical.vars, rng) for _ in range(8)]
        got = canonical.decode_population(population)
        expected = []
        for genome in population:
            cand = _ga_vector_to_candidates(
                flat_vals=genome,
                q=legacy.q,
                d=legacy.d,
                free_dims=legacy.free_dims,
                grouped_dims=legacy.grouped_dims,
                fixed_dims=legacy.fixed_dims_dict,
                partition_map=legacy.partition_map,
                cat_emb_avail=legacy.cat_emb_avail,
                sol_emb_avail=legacy.sol_emb_avail,
                bas_emb_avail=legacy.bas_emb_avail,
                cat_is_salt_flags=legacy.cat_is_salt_flags,
            )
            if legacy.pre_batch is not None:
                cand = torch.cat([legacy.pre_batch, cand], dim=0)
            expected.append(cand)
        assert torch.equal(got, torch.stack(expected))


def test_relabelings_collapse_to_one_canonical_form():
    """All 48 relabelings of a batch canonicalize to the same genome."""
    rng = np.random.default_rng(1)
    _, canonical, _, _ = _make_problem()
    layout = canonical.layout
    genome = _sample_genome(canonical.vars, rng)
    reference = layout.canonicalize(dict(genome), canonical.salt_flags)

    forms = set()
    for perm in itertools.permutations(range(3)):
        for swaps in itertools.product([False, True], repeat=3):
            relabeled = _relabel(genome, layout, perm, swaps)
            got = layout.canonicalize(relabeled, canonical.salt_flags)
            forms.add(tuple(got[k] for k in sorted(got)))
            assert got == reference
    assert len(forms) == 1


def test_relabeling_preserves_the_batch_but_not_the_mc_estimate():
    """Relabeling yields the same batch; qLogEI is invariant only in expectation.

    The joint posterior over a permuted batch is exactly the permuted posterior,
    but the QMC sampler couples its base samples to candidate positions, so the
    MC estimate moves under relabeling. This is why a GA that is free to
    relabel can maximize sampler noise instead of the acquisition.
    """
    rng = np.random.default_rng(2)
    _, canonical, acqf, _ = _make_problem()
    layout = canonical.layout
    genome = _sample_genome(canonical.vars, rng)
    reference = _row_multiset(canonical.decode_population([genome]))

    values = []
    for perm in itertools.permutations(range(3)):
        for swaps in itertools.product([False, True], repeat=3):
            relabeled = _relabel(genome, layout, perm, swaps)
            decoded = canonical.decode_population([relabeled])
            assert _row_multiset(decoded) == reference
            with torch.no_grad():
                values.append(acqf(decoded).item())
    assert np.ptp(values) < MC_RELABEL_TOL


def test_canonicalization_preserves_the_batch():
    """Canonicalizing permutes rows without changing the batch or its AF value."""
    rng = np.random.default_rng(3)
    _, canonical, acqf, _ = _make_problem()
    genome = _sample_genome(canonical.vars, rng)
    before = canonical.decode_population([genome])
    after = canonical.decode_population(
        [canonical.layout.canonicalize(dict(genome), canonical.salt_flags)]
    )
    assert _row_multiset(before) == _row_multiset(after)
    with torch.no_grad():
        assert acqf(before).item() == pytest.approx(
            acqf(after).item(), abs=MC_RELABEL_TOL
        )


def test_canonicalization_is_idempotent():
    """Canonicalizing a canonical genome is a no-op."""
    rng = np.random.default_rng(4)
    _, canonical, _, _ = _make_problem()
    genome = canonical.layout.canonicalize(
        _sample_genome(canonical.vars, rng), canonical.salt_flags
    )
    assert canonical.layout.canonicalize(dict(genome), canonical.salt_flags) == genome


def test_non_salt_catalysts_get_their_base_index_collapsed():
    """A non-salt catalyst has a zeroed base slice, so its base index is a don't-care."""
    rng = np.random.default_rng(5)
    _, canonical, _, _ = _make_problem()
    layout = canonical.layout
    genome = _sample_genome(canonical.vars, rng)
    for cand in range(layout.q):
        genome[cand * layout.n_var_per_cand + layout.cat_local + 1] = 1
        genome[cand * layout.n_var_per_cand + layout.bas_local + 1] = 2
    canonical.layout.canonicalize(genome, canonical.salt_flags)
    for cand in range(layout.q):
        assert genome[cand * layout.n_var_per_cand + layout.bas_local + 1] == 0


def test_repair_canonicalizes_every_individual():
    """CanonicalRepair leaves the whole population in canonical form."""
    rng = np.random.default_rng(6)
    _, canonical, _, _ = _make_problem()
    repair = CanonicalRepair(canonical.layout, canonical.salt_flags)
    population = np.array(
        [_sample_genome(canonical.vars, rng) for _ in range(10)], dtype=object
    )
    repair._do(canonical, population)
    for genome in population:
        assert (
            canonical.layout.canonicalize(dict(genome), canonical.salt_flags) == genome
        )


def test_layout_without_grouped_dims_is_fully_exchangeable():
    """With no batch constraint the whole batch is one block."""
    layout = _build_canonical_layout(
        q=4,
        free_dims=[0, 1, 2],
        grouped_dims={},
        partition_map={},
        d_continuous=3,
        use_component_mode=True,
    )
    assert layout.blocks == [[0, 1, 2, 3]]
    assert layout.group_order == 24


def test_optimizer_returns_a_feasible_batch():
    """End-to-end smoke test: the canonical GA honours the partition and pre-batch."""
    _, canonical, acqf, bounds = _make_problem(q=8, pre=2)
    pre_batch = canonical.pre_batch
    candidates, acq_value = optimize_acqf_mixed_canonical_GA(
        acq_function=acqf,
        bounds=bounds,
        q=8,
        n_evals_ga=300,
        pop_size=30,
        cat_emb_avail=canonical.cat_emb_avail,
        sol_emb_avail=canonical.sol_emb_avail,
        bas_emb_avail=canonical.bas_emb_avail,
        cat_is_salt_flags=canonical.cat_is_salt_flags,
        batch_constraints={0: (2, 2, 2)},
        fixed_dims={3: 0.5},
        pre_batch=pre_batch,
    )
    assert candidates.shape == (8, bounds.shape[1])
    assert torch.equal(candidates[:2], pre_batch)
    assert torch.isfinite(acq_value)
    free_temps = candidates[2:, 0]
    assert len(torch.unique(free_temps)) <= 3
    assert candidates[:, 3].eq(0.5).all() or torch.equal(candidates[:2], pre_batch)


def test_discrete_mating_never_invents_an_index():
    """Uniform crossover keeps discrete genes value-preserving; SBX would not.

    Catalyst, solvent and base identifiers are unordered, so an offspring index
    must come from one of its parents. Pymoo's default SBX on ``Integer``
    variables interpolates between them instead.
    """
    rng = np.random.default_rng(7)
    _, canonical, _, _ = _make_problem()
    layout = canonical.layout
    parents = []
    for value in (0, N_CAT - 1):
        genome = _sample_genome(canonical.vars, rng)
        for cand in range(layout.q):
            base = cand * layout.n_var_per_cand
            genome[base + layout.cat_local + 1] = value
        parents.append(genome)

    mating = _build_canonical_mating(repair=None, discrete_mutation_prob=0.0)
    population = Population.new(X=parents * 25)
    offspring = mating.do(
        canonical, population, 50, random_state=np.random.default_rng(8)
    )

    allowed = {0, N_CAT - 1}
    seen = set()
    for individual in offspring:
        for cand in range(layout.q):
            base = cand * layout.n_var_per_cand
            seen.add(int(individual.X[base + layout.cat_local + 1]))
    assert seen <= allowed, f"mating invented catalyst indices {seen - allowed}"
    assert seen == allowed, "crossover never mixed the parents"
