"""Single HPC worker: load model, re-optimise assigned batches, save result.

Run from the repository root with a worker ID argument::

    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.run_worker <WORKER_ID>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize

from pablo.main import (
    ContinuousBounds,
    DEFAULT_EXCEL_FILE,
    DEVICE,
    DTYPE,
    load_reference_data,
    load_training_data,
    MODEL_SAVE_PATH,
    TEMPERATURE_PARTITION,
)
from pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.config import (
    BATCHES_PER_WORKER,
    RESULTS_DIRNAME,
    TOTAL_BATCHES,
)
from pablo.optimize import optimize_acqf_continuous

torch.set_default_dtype(torch.float64)

# ---- optimisation budget (per batch, per temperature grouping) -------------
NUM_RESTARTS = 2
RAW_SAMPLES = 64


def _model_health_check(model, X_sample, acqf):
    """Check that the model's posterior means vary across candidates.

    Prints the outcome-transform mean and std and the posterior means at
    ``X_sample``.

    Args:
        model: Fitted GP with a ``Standardize`` outcome transform.
        X_sample: Candidates of shape ``q x d``.
        acqf: Acquisition function (unused).

    Returns:
        False if all posterior means are equal after rounding to 3 decimals
        (a prior-only model), True otherwise.
    """
    with torch.no_grad():
        post = model.posterior(X_sample.to(DEVICE))
        means = post.mean.squeeze(-1).cpu()
    unique_means = len(set(round(m, 3) for m in means.tolist()))
    outcome_mean = float(model.outcome_transform.means.item())
    outcome_std = float(model.outcome_transform.stdvs.item())
    print(
        f"  Model health: outcome_transform means={outcome_mean:.3f}  std={outcome_std:.3f}"
    )
    print(
        f"  Posterior means: {[round(m, 2) for m in means.tolist()]}  (unique={unique_means})"
    )
    if unique_means <= 1:
        print("  FATAL: all posterior means identical — model is broken (prior-only).")
        print("  Check that train_inputs are real data, not dummy zeros.")
        return False
    return True


def _load_saved_model(
    model_path: Path,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
) -> SingleTaskGP:
    """Load model state dict into a GP initialised with real training data.

    Creating the model with the actual ``X_train`` / ``Y_train`` ensures
    that ``model.train_inputs`` and ``model.train_targets`` are correct
    for posterior computation.  The saved state dict (kernel hyperparams,
    Normalize / Standardize transforms) is loaded on top.

    Args:
        model_path: Path to the saved state dict.
        X_train: Training inputs of shape ``n x d``.
        Y_train: Training targets of shape ``n x 1``.

    Returns:
        The GP on ``DEVICE`` in eval mode.
    """
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)

    d = X_train.shape[1]
    model = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def _build_x_discrete(batch, cat_map, sol_map, bas_map):
    """Build the discrete part of a batch from decoded candidate dicts.

    Each row is the catalyst, solvent and base embeddings concatenated; a
    ``Base_ID`` below 0 gives a zero base block.

    Args:
        batch: List of ``q`` dicts with ``Catalyst_ID``, ``Solvent_ID`` and
            ``Base_ID``.
        cat_map: Catalyst ID to embedding tensor.
        sol_map: Solvent ID to embedding tensor.
        bas_map: Base ID to embedding tensor.

    Returns:
        Tensor of shape ``q x d_discrete``.
    """
    d_base = next(iter(bas_map.values())).shape[0]
    rows = []
    for cand in batch:
        cat_emb = cat_map[cand["Catalyst_ID"]]
        sol_emb = sol_map[cand["Solvent_ID"]]
        bid = cand["Base_ID"]
        bas_emb = bas_map[bid] if bid >= 0 else torch.zeros(d_base, dtype=DTYPE)
        rows.append(torch.cat([cat_emb, sol_emb, bas_emb]))
    return torch.stack(rows)


def _build_full_candidates(batch, cat_map, sol_map, bas_map, cont_bounds):
    """Rebuild the model-input tensor of a decoded batch for AF re-evaluation.

    Args:
        batch: List of ``q`` decoded candidate dicts with Temperature,
            Formaldehyde_Ratio, Catalyst_Loading, Time and the three IDs.
        cat_map: Catalyst ID to embedding tensor.
        sol_map: Solvent ID to embedding tensor.
        bas_map: Base ID to embedding tensor.
        cont_bounds: ``ContinuousBounds`` used to scale the four continuous
            variables to [0, 1].

    Returns:
        Tensor of shape ``q x (4 + d_discrete)``: scaled continuous columns
        followed by the raw embeddings from ``_build_x_discrete``.
    """
    cb_t = cont_bounds.to_tensor()
    X_disc = _build_x_discrete(batch, cat_map, sol_map, bas_map)
    q = X_disc.shape[0]
    d_cont = 4
    d_disc = X_disc.shape[1]
    X_full = torch.zeros(q, d_cont + d_disc, dtype=DTYPE)
    for i, cand in enumerate(batch):
        X_full[i, 0] = (cand["Temperature"] - cb_t[0, 0].item()) / (
            cb_t[1, 0].item() - cb_t[0, 0].item()
        )
        X_full[i, 1] = (cand["Formaldehyde_Ratio"] - cb_t[0, 1].item()) / (
            cb_t[1, 1].item() - cb_t[0, 1].item()
        )
        X_full[i, 2] = (cand["Catalyst_Loading"] - cb_t[0, 2].item()) / (
            cb_t[1, 2].item() - cb_t[0, 2].item()
        )
        X_full[i, 3] = (cand["Time"] - cb_t[0, 3].item()) / (
            cb_t[1, 3].item() - cb_t[0, 3].item()
        )
    X_full[:, d_cont:] = X_disc
    return X_full


def _decode_refined_batch(candidates, orig_batch, model, cont_bounds):
    cb_t = cont_bounds.to_tensor()
    t_range = (cb_t[1, 0] - cb_t[0, 0]).item()
    f_range = (cb_t[1, 1] - cb_t[0, 1]).item()
    l_range = (cb_t[1, 2] - cb_t[0, 2]).item()
    time_range = (cb_t[1, 3] - cb_t[0, 3]).item()

    with torch.no_grad():
        posterior = model.posterior(candidates.to(DEVICE))
        means = posterior.mean.squeeze(-1).cpu()
        variances = posterior.variance.squeeze(-1).cpu()

    decoded = []
    for i in range(candidates.shape[0]):
        c = candidates[i]
        decoded.append(
            {
                "Temperature": round(c[0].item() * t_range + cb_t[0, 0].item(), 2),
                "Formaldehyde_Ratio": round(
                    c[1].item() * f_range + cb_t[0, 1].item(), 3
                ),
                "Catalyst_Loading": round(c[2].item() * l_range + cb_t[0, 2].item(), 3),
                "Time": round(c[3].item() * time_range + cb_t[0, 3].item(), 1),
                "Catalyst_ID": orig_batch[i]["Catalyst_ID"],
                "Solvent_ID": orig_batch[i]["Solvent_ID"],
                "Base_ID": orig_batch[i]["Base_ID"],
                "Posterior_Mean": round(means[i].item(), 2),
                "Posterior_Var": round(variances[i].item(), 3),
            }
        )
    return decoded


def main(worker_id: int) -> None:
    """Re-optimise the continuous dimensions of this worker's batches.

    Loads the saved GP and ``ranked_results_merged.json``, keeps each batch's
    catalyst/solvent/base choices, and re-optimises qLogEI over the
    continuous dimensions with the temperature partition and Time fixed at
    20 h. Exits with status 1 if ``_model_health_check`` fails. Skips the
    work if the result file already exists.

    Args:
        worker_id: Worker index; it handles batches
            ``[worker_id · BATCHES_PER_WORKER, min((worker_id + 1) · BATCHES_PER_WORKER, TOTAL_BATCHES))``
            and writes ``<RESULTS_DIRNAME>/result_<worker_id:04d>.json``.
    """
    base_dir = Path(__file__).parent
    json_path = base_dir.parent / "ranked_results_merged.json"
    results_dir = base_dir / RESULTS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"result_{worker_id:04d}.json"
    if out_path.exists():
        print(f"Result {out_path.name} already exists. Skipping.")
        return

    batch_start = worker_id * BATCHES_PER_WORKER
    batch_end = min(batch_start + BATCHES_PER_WORKER, TOTAL_BATCHES)

    print(f"Worker {worker_id}: batches [{batch_start}, {batch_end})")

    # ---- setup ------------------------------------------------------------

    print("  Loading reference data ...")
    cat_map, sol_map, bas_map, _cat_is_salt, _cat_bld, _cids, _sids, _bids = (
        load_reference_data(DEFAULT_EXCEL_FILE)
    )

    print("  Loading training data ...")
    cont_bounds = ContinuousBounds()
    X_train, Y_train = load_training_data(
        DEFAULT_EXCEL_FILE, cat_map, sol_map, bas_map, cont_bounds, _cat_is_salt
    )
    best_f = Y_train.max().item()
    d_total = X_train.shape[1]

    print(f"  Loading model (d_total={d_total}) ...")
    model = _load_saved_model(MODEL_SAVE_PATH, X_train, Y_train)

    # ---- model health check (on first batch) ------------------------------
    with open(json_path, "r") as handle:
        data = json.load(handle)
    all_batches = data["batches"]
    old_af_values = data["af_values"]

    acqf = qLogExpectedImprovement(model=model, best_f=best_f)

    test_batch = all_batches[batch_start]
    X_test = _build_full_candidates(test_batch, cat_map, sol_map, bas_map, cont_bounds)
    if not _model_health_check(model, X_test, acqf):
        sys.exit(1)

    full_bounds = torch.zeros(2, d_total, dtype=DTYPE)
    full_bounds[1, :] = 1.0
    cb_t = cont_bounds.to_tensor()
    time_norm_20 = (20.0 - cb_t[0, 3].item()) / (cb_t[1, 3].item() - cb_t[0, 3].item())
    batch_constraints = {0: TEMPERATURE_PARTITION}
    fixed_dims = {3: time_norm_20}

    # ---- load batches & optimise ------------------------------------------

    results: list[dict] = []
    for batch_idx in range(batch_start, batch_end):
        batch = all_batches[batch_idx]
        json_af = old_af_values[batch_idx]  # from original run (different model)

        print(f"  Optimising batch {batch_idx} ...")
        X_orig = _build_full_candidates(batch, cat_map, sol_map, bas_map, cont_bounds)
        with torch.no_grad():
            old_af_eval = float(acqf(X_orig.unsqueeze(0)).item())

        X_disc = _build_x_discrete(batch, cat_map, sol_map, bas_map)
        candidates, acq_value = optimize_acqf_continuous(
            acq_function=acqf,
            bounds=full_bounds,
            X_discrete=X_disc,
            batch_constraints=batch_constraints,
            fixed_dims=fixed_dims,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        new_af = float(acq_value.item())

        refined = _decode_refined_batch(candidates, batch, model, cont_bounds)
        results.append(
            {
                "batch_index": batch_idx,
                "old_af_json": json_af,
                "old_af_eval": old_af_eval,
                "new_af": new_af,
                "refined_batch": refined,
            }
        )
        print(
            f"    old_AF_json={json_af:.6f}  "
            f"old_AF_eval={old_af_eval:.6f}  "
            f"new_AF={new_af:.6f}"
        )

    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"  Wrote {len(results)} results → {out_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <WORKER_ID>", file=sys.stderr)
        sys.exit(1)
    main(int(sys.argv[1]))
