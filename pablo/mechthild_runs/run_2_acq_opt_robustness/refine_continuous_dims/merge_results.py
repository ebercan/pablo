"""Merge per-worker results into a single ``results_merged_optimized.json``.

Run from the repository root after all workers have completed::

    python -m pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.merge_results
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pablo.mechthild_runs.run_2_acq_opt_robustness.refine_continuous_dims.config import (
    RESULTS_DIRNAME,
    TOTAL_BATCHES,
)
from scipy.stats import spearmanr


def main() -> None:
    """Merge the refined batches and print how refinement changed their AF.

    Reads every ``result_*.json`` in ``RESULTS_DIRNAME``. If no file exists or
    fewer than ``TOTAL_BATCHES`` batch results are found, prints the missing
    indices and returns without writing. Otherwise writes
    ``results_merged_optimized.json`` to the parent directory, holding
    ``batches`` and ``af_values`` sorted by refined AF, descending, and prints
    the improved / worsened / tied counts, AF deltas against the unrefined
    batch evaluated under the same model, and Spearman correlations of the
    refined AF with the unrefined AF (same model and original GA run).
    """
    base_dir = Path(__file__).parent
    results_dir = base_dir / RESULTS_DIRNAME

    result_files = sorted(results_dir.glob("result_*.json"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    all_results: list[dict] = []
    for rf in result_files:
        with open(rf, "r") as handle:
            all_results.extend(json.load(handle))

    all_results.sort(key=lambda r: r["batch_index"])
    n_found = len(all_results)
    print(
        f"Found {n_found} / {TOTAL_BATCHES} batch results from {len(result_files)} files"
    )

    if n_found < TOTAL_BATCHES:
        missing = sorted(
            set(range(TOTAL_BATCHES)) - {r["batch_index"] for r in all_results}
        )
        print(f"  Missing batch indices: {missing}")
        print("  Merge skipped: rerun once every batch has a result file.")
        return

    # sort by new AF descending
    sorted_results = sorted(all_results, key=lambda r: r["new_af"], reverse=True)

    out_path = base_dir.parent / "results_merged_optimized.json"
    batches = [r["refined_batch"] for r in sorted_results]
    af_values = [r["new_af"] for r in sorted_results]

    with open(out_path, "w") as handle:
        json.dump({"batches": batches, "af_values": af_values}, handle, indent=2)
    print(f"Saved {len(batches)} batches → {out_path}")

    # ---- analytics --------------------------------------------------------
    # old_af_eval: AF of original (unmodified) batch evaluated with *this* model.
    #   This is the fair baseline — same model, no optimisation.
    # old_af_json: AF from the original GA run (different model instance, reference only).
    old_eval = np.array([r["old_af_eval"] for r in sorted_results])
    old_json = np.array([r["old_af_json"] for r in sorted_results])
    new_af = np.array([r["new_af"] for r in sorted_results])
    deltas = new_af - old_eval

    improved = int((deltas > 1e-12).sum())
    worsened = int((deltas < -1e-12).sum())
    tied = n_found - improved - worsened

    sp_corr, sp_p = spearmanr(old_eval, new_af)
    sp_corr_json, sp_p_json = spearmanr(old_json, new_af)

    print(f"\n{'=' * 60}")
    print("RESULTS  (deltas = new_AF - old_AF_evaluated, same model)")
    print(f"{'=' * 60}")
    print(f"  Total batches:          {n_found}")
    print(f"  Improved:               {improved}  ({100 * improved / n_found:.1f} %)")
    print(f"  Worsened:               {worsened}  ({100 * worsened / n_found:.1f} %)")
    print(f"  Tied:                   {tied}  ({100 * tied / n_found:.1f} %)")
    print(f"  Mean old AF (eval):     {old_eval.mean():.6f}")
    print(f"  Mean old AF (json):     {old_json.mean():.6f}")
    print(f"  Mean new AF:            {new_af.mean():.6f}")
    print(f"  Min  AF delta:          {deltas.min():.8f}")
    print(f"  Max  AF delta:          {deltas.max():.8f}")
    print(f"  Mean AF delta:          {deltas.mean():.8f}")
    print(f"  Spearman (eval):        {sp_corr:.6f}  (p = {sp_p:.6g})")
    print(f"  Spearman (json):        {sp_corr_json:.6f}  (p = {sp_p_json:.6g})")


if __name__ == "__main__":
    main()
