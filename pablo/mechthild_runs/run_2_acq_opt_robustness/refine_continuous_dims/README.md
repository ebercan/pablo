# Refine Continuous Dims — `optimize_acqf_continuous` baseline test

## Purpose

Isolated benchmark for [optimize_acqf_continuous](../../../optimize.py#L1281). Takes each of the 487 q=6 batches from the parent's [ranked_results_merged.json](../ranked_results_merged.json) (originally produced by `optimize_acqf_mixed_GA`), freezes the discrete slice (Catalyst / Solvent / Base), and re-optimizes only the continuous dims (Temperature partition + Formaldehyde Ratio + Catalyst Loading; Time pinned to 20 h).

**Pass criterion:** `new_af ≥ old_af_eval` for (essentially) every batch. Continuous-only optimization on a fixed discrete assignment is a strictly easier problem than GA's mixed search — anything else indicates a bug.

## Files

| File | Role |
|---|---|
| [config.py](config.py) | `WORKERS=244`, `BATCHES_PER_WORKER=2`, `TOTAL_BATCHES=487`, partition/mem/time. |
| [run_worker.py](run_worker.py) | Loads model + training data, optimizes 2 batches via `optimize_acqf_continuous`, writes `results/result_XXXX.json`. |
| [submit_mechthild.py](submit_mechthild.py) | SLURM array submission (`0–243`). |
| [merge_results.py](merge_results.py) | Aggregates per-worker JSONs → `../results_merged_optimized.json`; prints improved/worsened/tied + Spearman ρ. |

## Prerequisites on Mechthild

Make sure these are synced before submitting:

- `pablo/trained_main_model.pt`
- `pablo/mechthild_runs/run_2_acq_opt_robustness/ranked_results_merged.json`
- `pablo/nhc_doe.xlsx`

## 1) Local pre-flight sanity check (run this first!)

Last time, the refined AF values came back negative (−1.5, −2.0) while the originals were ≈ 0.65 – 0.85 — a setup bug masquerading as a finished sweep. Always run this before sbatching.

From repo root:

```bash
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.run_worker 0
```

This processes batches 0 and 1 on your laptop (~1 min). For each batch it prints:

```
old_AF_eval=<x.xxxxxx>  new_AF=<y.yyyyyy>  Δ=<+0.xxxxxx>
```

**Pass:** `Δ ≥ 0` and `new_AF` is in the same order of magnitude as `old_AF_eval` (positive, ~0.5 – 1.0).
**Fail:** any negative Δ, or `new_AF` wildly negative — stop and debug. Do **not** submit to Mechthild.

A successful local run writes `results/result_0000.json` — `gio trash` it (or leave it; the worker will skip if the matching ID re-appears).

## 2) Submit array job (Mechthild)

```bash
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.submit_mechthild
# or skip the interactive prompt:
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.submit_mechthild --force
```

Config (from `config.py`):

| Setting | Value |
|---|---|
| Array | `0–243` (244 workers) |
| Batches per worker | 2 (covers 487 with one short final worker) |
| Partition | `medium` (1 day cap) |
| CPUs / task | 1 |
| Memory | 16 G |
| Time limit | `01:00:00` (≫ expected ~2 – 5 min runtime) |

## 3) Monitor

```bash
squeue -u $USER
ls pablo/mechthild_runs/run_2_acq_opt_robustness/refine_continuous_dims/results/ | wc -l   # expect 244
tail -n 50 pablo/mechthild_runs/run_2_acq_opt_robustness/refine_continuous_dims/logs/worker_*.err
```

Cancel if needed:

```bash
scancel -n run3_refine_cont
```

## 4) After SCP — merge locally

SCP the per-worker JSONs into the matching local folder so the structure looks like:

```
refine_continuous_dims/
├── results/
│   ├── result_0000.json
│   ├── result_0001.json
│   └── ... (244 files total)
└── logs/
```

Then, from repo root:

```bash
python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.merge_results
```

This writes `pablo/mechthild_runs/run_2_acq_opt_robustness/results_merged_optimized.json` (batches sorted by `new_af` desc) and prints:

```
RESULTS  (deltas = new_AF - old_AF_evaluated, same model)
  Total batches:          487
  Improved:               XXX  (XX.X %)
  Worsened:               XXX  (XX.X %)
  Tied:                   XXX  (XX.X %)
  Mean old AF (eval):     ...
  Mean new AF:            ...
  Mean AF delta:          ...
  Spearman (eval):        ...
  Spearman (json):        ...
```

**Acceptance:** `Improved` ≈ 100 %, `Mean AF delta > 0`, no large negative `Min AF delta`.

If `merge_results` reports missing batch indices, it lists them and refuses to write the merged JSON — re-submit only the missing worker IDs.

## Per-batch result schema

Each `results/result_XXXX.json` is a list of dicts:

```json
[
  {
    "batch_index": 0,
    "old_af_json": 0.7421,
    "old_af_eval": 0.7385,
    "new_af":      0.8612,
    "refined_batch": [ { "Temperature": ..., "Formaldehyde_Ratio": ..., ... }, ... ]
  },
  ...
]
```

| Field | Meaning |
|---|---|
| `old_af_json` | AF value reported by the original GA run (different model instance — reference only). |
| `old_af_eval` | AF of the original batch re-evaluated with **this** model. The fair baseline. |
| `new_af` | AF after `optimize_acqf_continuous` on the same discrete slice. Should be ≥ `old_af_eval`. |
| `refined_batch` | Decoded q=6 candidates with posterior mean/variance. |
