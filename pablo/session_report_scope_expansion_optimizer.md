# Session report — new inner optimizer for scope expansion (run_5)

Branch `claude/blissful-cerf-nwmxqo` (repo `ebercan/pablo`), base `5518375`, last commit `2cb22c2`.
Nothing is running. run_5 itself was **not** run (only smoke jobs of 1–2 iterations and an
optimizer comparison on sampled tree nodes); the full sweep is for Mechthild.

---

## 1. What `n` means in this report

`n` = number of training points of the GP that the optimizer works on.

- In a run_5 job: n = 52 (starting design, 2·D with D = 26) + 6 per finished BO iteration.
  So n ≈ 52 is iteration 0, n ≈ 250 is iteration 33, n ≈ 550 is iteration 83 (the run ends at n = 652).
- Inside the fantasy tree a node at depth j additionally carries 6·j fantasized points (up to +24).

`n` matters for two reasons: cost grows with n (kernel solves are O(n²) per candidate), and the
acquisition landscape changes with n. At large n the incumbent is good and almost all
improvement probability sits in far tails, u = (μ − best_f)/σ ≈ −5 … −8.

---

## 2. Work done, in order

| commit | what |
|---|---|
| `441459d` | Lookup-alternating optimizer removed from `optimize.py` (class, `_greedy_group_maps`, `optimize_acqf_mixed_lookup_alternating`, 2 imports); `tests/test_lookup_alternating.py` removed (`git rm`). run_5 switched temporarily to the batch GA; starting set 69 → 52 (= 2·26); `inputs.json` regenerated; `verify_results.py` partition check made order-free. Two latent bugs fixed that the lookup optimizer had hidden: the q = 1 expansion calls received the full (2,2,2) partition. |
| `79c3df5` | Scope-expansion code moved out of `optimize.py`; `best_observed_f` → `pablo/utils/acq_utils.py`. |
| `e846afa` | Module named `pablo/scope_expansion.py`, expanders renamed `optimize_scope_expansion_fantasized` / `_myopic`. Deleted after your approval: sequential-canonical GA, the two `*_canonical` expander wrappers, `PlanAheadNHC`, `utils/nhc_utils.py`, `utils/starting_set_generation.py`, `utils/canonical_ga_benchmark.py`, stale README block. |
| `f88cc5f` | New optimizer `optimize_acqf_mixed_pool_greedy` + measurement harness `run_5_scope_extension_v2/inner_optimizer_harness.py` + tests. |
| `474d06e` | Expanders rewired onto the new optimizer; root posterior cache shared across the fantasy tree; terminal aggregate as `WeightedSumAcquisition`. |
| `581533a` | Single-point refinement and analytic lower bound (fixes for failures found by the harness); run_5 batch + all expander roles on the new optimizer, settings in `config.py`. |
| `bc75e06` | Harness score clipped to analytic bounds (metric fix, §5); lighter settings for the terminal stages. |
| `2cb22c2` | run_5 README (optimizer, comparison, runtimes), SLURM time limits. |

Net diff vs base: 21 files, +3104 / −5247 lines.

Tests (`python -m pytest tests/`): **359 passed, 8 skipped, 3 failed** — the same 3 as the
baseline, all needing git-ignored featscan data (`test_archived_runs_are_self_describing`,
`test_archive_manifests_match_their_cells`, `test_canonical_data_paths_resolve`).
12 new tests (`tests/test_pool_greedy.py`, `tests/test_scope_expansion.py`).
Error-handling audit: 6 hits, all pre-existing in `pablo/results_analysis/`; none added.

---

## 3. Where the optimizer sits in run_5

Every run_5 iteration does:

1. fit the GP on all real data (n points);
2. **batch step** — choose the q = 6 experiments from the available catalysts (qLogEI, (2,2,2)
   temperature partition, time pinned);
3. **expander step** — choose one locked catalyst to order:
   - `random`: uniform, no optimizer;
   - `myopic`: q = 1 qLogEI of the current model over locked catalysts;
   - `fantasized`: tree `[12, 4, 2, 1]` → **157 tree nodes**, each needing its own q = 6 batch on
     a fantasy model; then *stage 1* picks the expansion point on the aggregated acquisition of
     the nodes (`operating_mode`), *stage 2* completes a batch around it (reported value only).

All of these optimization calls now use the same optimizer, `optimize_acqf_mixed_pool_greedy`
(`pablo/optimize.py`), with per-role settings. The GA was far too slow for 157 calls per iteration
(≈ 9–11 s each ⇒ ≈ 25 min per iteration).

---

## 4. The new algorithm: pool-greedy

### 4.1 Steps for one q = 6 batch

**Step 1 — candidate pool.** A finite set of full candidate points (rows):

- `n_pool` random rows (4096 in the batch and tree roles): catalyst, solvent and base drawn
  uniformly from the available component tables (base slice zeroed for non-salt catalysts);
  free continuous dims from a scrambled Sobol sequence; time pinned.
- warm-start rows: the parent node's batch (in the tree).
- `n_local` = 256 local rows: the 8 training points with the largest targets (fantasy points
  included), continuous dims perturbed by Gaussian noise with sd 0.1 × range (clipped); discrete
  part kept for half of them, random for the other half.

**Step 2 — refine.** The 32 rows with the best *single-point* value get their free continuous dims
optimized by L-BFGS-B on analytic LogEI (discrete part fixed) and are added to the pool. This finds
narrow high-EI regions that random rows miss.

**Step 3 — greedy construction (the greedy part).** For s = 0 … 5:

  x_{s+1} = argmax over pool rows x of  V(B_s ∪ {x}),  B_{s+1} = B_s ∪ {x_{s+1}}

where V is the batch value (§4.3). Temperature grouping: a new point either **opens** a group (keeps
its own temperature) or **joins** an open group; for joining, the 256 best-ranked rows get the
group's temperature written into dim 0 and are scored. Opening/joining is allowed only while the
group counts can still complete the partition, so the result is exactly (2,2,2).

**Step 4 — swaps (batch role only, 1 pass).** Each slot is removed and re-picked from the pool with
the other 5 fixed; kept if the value rises.

**Step 5 — polish.** L-BFGS-B on qLogEI over the free continuous dims of the whole batch, discrete
choices and grouping fixed (one variable per temperature group, so grouped points move together).
Kept only if the value rises.

### 4.2 How greedy is it?

- **Greedy across batch slots, exhaustive within a slot.** Each slot is chosen once, as the best
  completion of the slots already chosen, by scoring *every* pool row. Earlier slots are never
  revisited in the tree setting (no swaps); the batch setting revisits each slot once (1 swap pass).
- **Greedy in the group decision.** Whether point s opens a new temperature group or joins one is
  decided by which gives the higher value at that step; it is not re-planned later.
- **Not greedy in the continuous dims.** Steps 2 and 5 are gradient optimization: per point before
  the greedy step, jointly over the whole batch after it.
- **Quality guarantee.** On a fixed finite pool, expected improvement E[maxᵢ (Yᵢ − f)⁺] is a
  monotone submodular set function, and sequential greedy is within (1 − 1/e) ≈ 63 % of the best
  6-subset *of the pool* (Nemhauser–Wolsey). This bound does not strictly carry over here: the
  optimizer maximizes log EI, the temperature overwrite changes the ground set during the greedy
  pass, and the pool is itself a sample of the space. In practice greedy qEI is typically close to
  the joint optimum, which the harness numbers (§6) are consistent with.
- **One-step (myopic) in time.** The optimizer only maximizes the current acquisition of the model
  it is given. All planning ahead is done by the fantasized expander, which calls it on fantasy
  models.

### 4.3 The batch value V

For one model:

  V(B) = max( MC qLogEI(B),  maxᵢ LogEI(xᵢ) )

- MC qLogEI uses BoTorch's own reductions on fixed QMC base samples (common random numbers across
  candidates).
- maxᵢ LogEI(xᵢ) is analytic and exact. Since qEI(B) ≥ maxᵢ EI(xᵢ), it is a valid lower bound on
  log qEI. It coincides with the MC value wherever MC resolves the tail, and it stays informative
  where no QMC sample reaches the improvement region: there the smoothed MC estimator floors near
  −51 regardless of the batch, even with 8192 samples.

For the fantasized stage 1 the value is `Σₖ wₖ Vₖ` over the tree nodes, with the `operating_mode`
weights (`only_terminal` / `averaged` / `layer_averaged`, same definitions as before).

### 4.4 Why it is fast

- **Pool-cached posteriors.** For an exact GP the optimizer computes W = L⁻¹ K(X, pool) once per pool
  (L L^T = K + σ²I). A greedy step then needs only the (N × s) cross-covariance to the current batch
  plus N small (s+1)×(s+1) Cholesky factors. The GA recomputes a full (s+1)-point posterior for
  every individual it evaluates. The cached path reproduces BoTorch's `qLogExpectedImprovement` to
  1e-13 (tested).
- **Screening.** All rows are first ranked on 32 QMC samples with a cheap proxy
  (log E[softplus(max Y − best_f)]); only the best 64 get the exact value on all samples.
- **Shared root inside the fantasy tree.** Every tree node is the root GP conditioned on f ≤ 24
  fantasy points with the same hyperparameters. `GPPoolPosteriorCache` keeps the root's L₀ and W₀ per
  pool; a node adds only a rank-f correction:
  mean μ₀ + W₁^T β₁, variance σ₀² − ‖W₁‖², cross-covariance C₀ − W₁ₚ^T W₁ᵦ.
  That is O(f·N) per node instead of O(n·N). Nodes at the same depth share one random pool, so W₀ is
  computed once per depth.

### 4.5 Settings per role (`run_5/config.py`, `pool_greedy_settings()`)

| role | pool | refine | swaps | polish iters | QMC samples |
|---|---|---|---|---|---|
| batch | 4096 | 32 | 1 | 50 | 1024 |
| tree node | 4096 | 32 | 0 | 20 | 128 |
| stage 1 / myopic | 4096 | 8 | 0 | 0 | 128 / 512 |
| stage 2 (reported value only) | 512 | 0 | 0 | 0 | 128 |

The tree nodes use 128 QMC samples for their qLogEI (the old code used BoTorch's default 512).

---

## 5. How it was tested against the GA

**Harness** (`inner_optimizer_harness.py`; stages `snapshots` → `nodes` → `score` → `report`):

- One short run_5-style trajectory per benchmark (Ackley, Gallagher101, LunacekBiRastrigin; random
  expander, GA batch), snapshots at n ≈ 52, 250, 550.
- From each snapshot: the root node plus 2 fantasy paths of depth 1–4, built the way the expander
  builds them ⇒ 27 nodes per benchmark, **81 nodes**.
- On every node: GA 15000/100 and the pool-greedy settings, 3 seeds each, plus 64 random feasible
  batches.

**Metric fix.** The first comparison used plain MC qLogEI (1024 samples) as the score, and pool-greedy
looked much worse at n ≈ 550. Inspection showed the GA's batches there scoring *above* their own
union bound log Σᵢ EIᵢ: in the deep-tail regime the GA optimizes into MC artefacts. The final score is

  score = min( max(MC qLogEI, maxᵢ log EIᵢ),  log Σᵢ EIᵢ )

i.e. MC clipped to the analytic lower and upper bounds of log qEI. It equals MC wherever MC is
consistent.

Two optimizer failures were also found by the harness and fixed before the final run: a random pool
missing narrow high-EI regions (→ refine step) and a flat MC landscape in deep tails (→ lower bound
in V).

---

## 6. Results (81 nodes × 3 seeds = 243 runs per optimizer)

Δ = score(pool-greedy) − score(GA, same node and seed), in nats of log qEI.

| optimizer | n ≈ 52: Δ median / share ≥ GA | n ≈ 250 | n ≈ 550 | all: Δ median / p10 / share ≥ GA |
|---|---|---|---|---|
| pool-greedy, tree setting | −0.04 / 21 % | +0.50 / 85 % | +17.3 / 95 % | +0.35 / −0.17 / 67 % |
| pool-greedy, batch setting | −0.05 / 24 % | +0.51 / 84 % | +17.3 / 94 % | +0.34 / −0.18 / 67 % |
| pool-greedy, pool 1024 | −0.08 / 16 % | +0.42 / 78 % | +17.8 / 90 % | +0.25 / −0.35 / 61 % |

Per call: pool-greedy **0.4–0.9 s**, GA 15000/100 **9–11 s** (uncontended; up to 53 s while the
machine was shared).

Reading: pool-greedy is marginally behind the GA at the start of a run (≈ 0.04 nats, i.e. ≈ 4 % in
EI), ahead from mid-run on, and far ahead late in the run, where the GA chases MC noise. It is
≈ 20× faster per call.

---

## 7. Runtime and the run_5 budget

Local, one CPU, per BO iteration:

| n | GP fit | batch | myopic expander | fantasized expander (tree / stage 1 / stage 2) |
|---|---|---|---|---|
| 52 | 1.8 s | 1.0 s | 0.2 s | 101 s (60 / 28 / 10) |
| 250 | 1.8 s | 0.8 s | 0.2 s | 96 s (61 / 22 / 12) |
| 550 | 12.9 s | 0.8 s | 0.3 s | 129 s (92 / 25 / 9) |

- one fantasized job (100 iterations) ≈ 3.2 h; one random or myopic job ≈ 0.2 h;
- full sweep (324 fantasized + 216 baseline jobs) ≈ 1100 CPU-h;
- on 300 single-CPU workers ≈ 6.5 h (the fantasized jobs need two waves), on ≥ 324 workers ≈ 3.5 h;
- SLURM limits set to 1 h (random, myopic) and 8 h (fantasized), ≈ 2× the local estimate.

Mechthild's per-core speed relative to this machine is unknown; probe one fantasized job first.

---

## 8. Decisions I took that you may want to revisit

1. **Main batch on pool-greedy too**, not the GA — same optimizer everywhere; one line in
   `pool_greedy_settings()` to change.
2. **Partition check** (`verify_results.py`): two temperature groups may land on the same value
   (the polish can push both to a bound); the check accepts merged groups ((4,2) or (6) for
   (2,2,2)). If identical group temperatures should be forbidden, the optimizer needs a
   distinctness constraint.
3. **Stage 2 of the fantasized expander** only produces the reported value
   (`selected_source_a_acq_values`) and runs on a cheap setting; it never affects the choice.
4. **Tree QMC samples 128** (was 512 by default).
5. **`run_1_scope_extension`**: only its import lines were updated (to keep the repo-structure
   import test green); its calls use the old expander keywords and would fail at runtime.
   Other legacy runs are untouched.
6. **Environment**: `pyarrow` installed into the venv to run the tests (not in `requirements.txt`);
   `map4==1.1.3` has no build for Python 3.11 and was skipped. No requirements file was edited.

## 9. Open

- Probe one fantasized job on Mechthild and adjust `TIME_LIMIT_BY_OPERATING_MODE`.
- The small deficit at n ≈ 52 (−0.04 nats median) could be closed with more polish in the batch
  role if it matters.
- The 6 bare `except` blocks in `pablo/results_analysis/` (pre-existing) were left alone.

## 10. Lines to update in your local `CLAUDE.md` / `AGENTS.md`

- "the Plan-Ahead acquisition lives in `pablo/plan_ahead.py`, with mixed-optimizer wrappers in
  `pablo/optimize.py`" → "Scope expansion (myopic and fantasized expanders) lives in
  `pablo/scope_expansion.py`; `pablo/optimize.py` holds only batch optimizers (pymoo GA, canonical
  GA, sequential GA, continuous refinement, pool-greedy)."
- Add: "Scope-expansion inner optimizer and run_5 batch optimizer: `optimize_acqf_mixed_pool_greedy`;
  settings in `run_5_scope_extension_v2/config.py`; measured with
  `run_5_scope_extension_v2/inner_optimizer_harness.py`."
- Remove any mention of `optimize_acqf_mixed_lookup_alternating`, `PlanAheadNHC`,
  `optimize_plan_ahead_qlogEI_*`, `utils/nhc_utils.py`, `utils/starting_set_generation.py`,
  `utils/canonical_ga_benchmark.py`.
