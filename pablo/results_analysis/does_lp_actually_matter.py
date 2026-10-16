"""Evaluate EI and LP-penalized EI on the full combinatorial candidate space.

Trains a SingleTaskGP on the initial experiment results, then evaluates
analytic Expected Improvement on all remaining candidate points of the
n_cat x n_sol x n_base combinatorial space. Then randomly selects 100
candidates as pending points and, for each, evaluates 4 modes of Local
Penalization across all candidates (leave-one-out style penalization).

The values in the JSON are lists with the following order::

    [X_vals_list, ei_scalar, [sl_list], [hl_list], [sg_list], [hg_list]]

Where each inner list has ``N_PENDING`` entries (one per pending point):
    - ``ei_scalar``: base (unpenalized) Expected Improvement value.
    - ``sl_list[i]``: Soft Local LP-penalized EI when pending point i is pending.
    - ``hl_list[i]``: Hard Local LP-penalized EI when pending point i is pending.
    - ``sg_list[i]``: Soft Global LP-penalized EI when pending point i is pending.
    - ``hg_list[i]``: Hard Global LP-penalized EI when pending point i is pending.

The JSON also has a ``"__metadata__"`` key containing the seed, pending
point indices, and their ``(cat, sol, bas)`` combo IDs for reproducibility.

Requires:
    Run ``python -m pablo.results_analysis.apply_pca_to_initial_results`` first to
    generate:
        - ``X_initial.pt``:  ``(n_train, d)`` training features.
        - ``Y_initial.pt``:  ``(n_train, 1)`` training targets.
        - ``X_all_discrete.pt``:  ``(n_total, d)`` full combinatorial features,
          ``n_total = n_cat * n_sol * n_base``.
        - ``combo_ids.pt``:  ``(n_total, 3)`` integer IDs per combo.

Usage:
    python -m pablo.results_analysis.does_lp_actually_matter
"""

import json
from pathlib import Path

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

SEED = 42
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent
DTYPE = torch.float64

BATCH_SIZE_EI = 1024
N_PENDING = 100
EI_PERCENTILE_CUTOFF = 30

GAMMA = 1.0
P_EXP = -5.0


def load_data():
    """Load precomputed tensors from disk.

    Returns:
        Tuple of four tensors:
            - ``X_initial``: ``(n_train, d)`` training features.
            - ``Y_initial``: ``(n_train, 1)`` training targets.
            - ``X_all_discrete``: ``(n_total, d)`` full combinatorial features.
            - ``combo_ids``: ``(n_total, 3)`` integer (cat, sol, bas) IDs.
    """
    X_initial = torch.load(DATA_DIR / "X_initial.pt", weights_only=True)
    Y_initial = torch.load(DATA_DIR / "Y_initial.pt", weights_only=True)
    X_all_discrete = torch.load(DATA_DIR / "X_all_discrete.pt", weights_only=True)
    combo_ids = torch.load(DATA_DIR / "combo_ids.pt", weights_only=True)
    return X_initial, Y_initial, X_all_discrete, combo_ids


def identify_remaining_candidates(X_initial, X_all_discrete):
    """Find indices in X_all_discrete that are NOT present in X_initial.

    Compares rows by L2 distance with a tight tolerance to account for
    floating point representation.

    Args:
        X_initial: ``(n_train, d)`` training feature tensor.
        X_all_discrete: ``(n_total, d)`` full candidate tensor.

    Returns:
        A 1D boolean mask of shape ``(n_total,)`` where True indicates
        a candidate NOT in the training set.
    """
    is_remaining = torch.ones(X_all_discrete.shape[0], dtype=torch.bool)

    for i in range(X_initial.shape[0]):
        dists = torch.norm(X_all_discrete - X_initial[i], dim=1)
        match_mask = dists < 1e-8
        is_remaining &= ~match_mask

    return is_remaining


def train_gp(X_initial, Y_initial):
    """Fit a SingleTaskGP with Normalize + Standardize transforms.

    Args:
        X_initial: ``(n, d)`` training features.
        Y_initial: ``(n, 1)`` training targets.

    Returns:
        A fitted SingleTaskGP model.
    """
    d = X_initial.shape[1]
    model = SingleTaskGP(
        X_initial,
        Y_initial,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def evaluate_acqf_batched(acqf, X_candidates, batch_size=BATCH_SIZE_EI, name="ACQF"):
    """Evaluate an acquisition function on candidates in batches to manage memory.

    Args:
        acqf: An initialized AcquisitionFunction object.
        X_candidates: ``(n, d)`` candidate feature tensor.
        batch_size: Number of candidates to evaluate per batch.
        name: String name of the AcquisitionFunction for logging.

    Returns:
        A 1D tensor of shape ``(n,)`` with acquisition function values for each candidate.
    """
    n = X_candidates.shape[0]
    vals = torch.empty(n, dtype=X_candidates.dtype)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X_candidates[start:end].unsqueeze(1)
        with torch.no_grad():
            vals[start:end] = acqf(X_batch)

        if (start // batch_size) % 10 == 0:
            print(f"  {name} batch {start}..{end} / {n}")

    return vals


def estimate_local_L_fast(model, x_j, lengthscales, n_samples=200):
    """Estimate local Lipschitz constant via random sampling + torch autograd.

    Samples random points in the lengthscale hypercube around ``x_j``,
    computes the posterior mean gradient norm at each via autograd, and
    returns the maximum.

    Args:
        model: A fitted GP model.
        x_j: ``(d,)`` pending point location.
        lengthscales: ``(d,)`` kernel lengthscales.
        n_samples: Number of random sample points for gradient estimation.

    Returns:
        Local Lipschitz constant estimate (float).
    """
    ls = lengthscales
    lb = x_j - ls / 2.0
    ub = x_j + ls / 2.0

    samples = lb + torch.rand(n_samples, x_j.shape[0], dtype=x_j.dtype) * (ub - lb)
    samples = torch.cat([x_j.unsqueeze(0), samples], dim=0)
    samples.requires_grad_(True)

    posterior = model.posterior(samples)
    mu = posterior.mean.squeeze(-1)
    grad = torch.autograd.grad(mu.sum(), samples)[0]
    grad_norms = torch.norm(grad, p=2, dim=1)

    return max(grad_norms.max().item(), 1e-6)


def estimate_global_L_fast(model, search_bounds, n_samples=500):
    """Estimate global Lipschitz constant via random sampling + torch autograd.

    Samples random points uniformly over the search bounds, computes the
    posterior mean gradient norm at each via autograd, and returns the maximum.

    Args:
        model: A fitted GP model.
        search_bounds: ``(2, d)`` bounds tensor ``[lower, upper]``.
        n_samples: Number of random sample points for gradient estimation.

    Returns:
        Global Lipschitz constant estimate (float).
    """
    d = search_bounds.shape[1]
    lb = search_bounds[0]
    ub = search_bounds[1]

    samples = lb + torch.rand(n_samples, d, dtype=search_bounds.dtype) * (ub - lb)
    samples.requires_grad_(True)

    posterior = model.posterior(samples)
    mu = posterior.mean.squeeze(-1)
    grad = torch.autograd.grad(mu.sum(), samples)[0]
    grad_norms = torch.norm(grad, p=2, dim=1)

    return max(grad_norms.max().item(), 1e-6)


def compute_lp_matrix(model, ei_values, X_candidates, X_pending_subset,
                      penalizer_type, l_estimation, search_bounds=None):
    """Compute LP-penalized EI for all candidates with each point in X_pending_subset as pending.

    For each pending point j in X_pending_subset, computes the locally
    penalized EI for all candidates. Uses fast torch-autograd Lipschitz
    estimation (random sampling + batched gradient norms) instead of
    per-point scipy optimization.

    Args:
        model: A fitted GP model.
        ei_values: ``(N,)`` pre-computed base EI values for all candidates.
        X_candidates: ``(N, d)`` candidate feature tensor.
        X_pending_subset: ``(K, d)`` subset of points to use as pending.
        penalizer_type: ``"soft"`` or ``"hard"``.
        l_estimation: ``"local"`` or ``"global"``.
        search_bounds: ``(2, d)`` search space bounds, required for global L.

    Returns:
        ``(K, N)`` tensor of penalized EI values, where row j is the
        penalized EI landscape when pending point j is the sole pending point.
    """
    K = X_pending_subset.shape[0]
    N = X_candidates.shape[0]

    M = model.train_targets.max().item()

    # Get lengthscales for local L estimation
    lengthscales = None
    for module in model.modules():
        if hasattr(module, "lengthscale") and module.lengthscale is not None:
            lengthscales = module.lengthscale.squeeze().detach()
            break

    # Posterior at all pending points (batched)
    with torch.no_grad():
        posterior = model.posterior(X_pending_subset)
        mu = posterior.mean.squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)

    # Lipschitz constant estimation
    if l_estimation == "global":
        L_global = estimate_global_L_fast(model, search_bounds)
        L_vals = [L_global] * K
        print(f"    Global L = {L_global:.6f}")
    else:
        L_vals = []
        for j in range(K):
            L_j = estimate_local_L_fast(model, X_pending_subset[j], lengthscales)
            L_vals.append(L_j)
            if (j + 1) % 25 == 0:
                print(f"    Local L estimated for {j+1}/{K} pending points")

    # R_j = (M - mu_j + gamma * sigma_j) / L_j
    R_vals = torch.tensor(
        [(M - mu[j].item() + GAMMA * sigma[j].item()) / L_vals[j] for j in range(K)],
        dtype=DTYPE,
    )

    # Compute penalized EI: for each pending point, compute penalty for all candidates
    result = torch.empty(K, N, dtype=DTYPE)
    for j in range(K):
        R_j = R_vals[j].item()
        x_j = X_pending_subset[j]
        dist = torch.norm(X_candidates - x_j, dim=1)
        ratio = dist / R_j

        if penalizer_type == "hard":
            ratio_safe = ratio.clamp(min=1e-12)
            alpha = ratio_safe * torch.pow(1.0 + torch.pow(ratio_safe, -P_EXP), 1.0 / P_EXP)
        elif penalizer_type == "soft":
            S_j = R_j / 3.0
            alpha = 0.5 * torch.erfc(
                (R_j - dist) / (torch.sqrt(torch.tensor(2.0, dtype=DTYPE)) * S_j)
            )

        result[j] = ei_values * alpha

    return result


def build_results_dict(X_candidates, combo_ids_remaining, ei_values,
                       sl_matrix, hl_matrix, sg_matrix, hg_matrix):
    """Build the output dictionary keyed by (cat_id, sol_id, bas_id) tuples.

    Args:
        X_candidates: ``(N, d)`` candidate features.
        combo_ids_remaining: ``(N, 3)`` integer IDs for each candidate.
        ei_values: ``(N,)`` base EI values.
        sl_matrix: ``(K, N)`` Soft Local LP matrix.
        hl_matrix: ``(K, N)`` Hard Local LP matrix.
        sg_matrix: ``(K, N)`` Soft Global LP matrix.
        hg_matrix: ``(K, N)`` Hard Global LP matrix.

    Returns:
        Dict mapping ``"(cat_id, sol_id, bas_id)"`` string keys to value
        lists of the form::

            [x_vals, ei_scalar, [sl_K], [hl_K], [sg_K], [hg_K]]
    """
    results = {}
    for i in range(X_candidates.shape[0]):
        c_id, s_id, b_id = combo_ids_remaining[i].tolist()
        key = f"({c_id}, {s_id}, {b_id})"
        x_vals = X_candidates[i].tolist()
        results[key] = [
            x_vals,
            ei_values[i].item(),
            sl_matrix[:, i].tolist(),
            hl_matrix[:, i].tolist(),
            sg_matrix[:, i].tolist(),
            hg_matrix[:, i].tolist(),
        ]
    return results


def generate_data():
    """Train GP, evaluate EI, then compute LP-penalized EI for 100 random pending points."""
    torch.manual_seed(SEED)

    print("Loading data...")
    X_initial, Y_initial, X_all_discrete, combo_ids = load_data()
    print(f"  X_initial: {X_initial.shape}")
    print(f"  Y_initial: {Y_initial.shape}")
    print(f"  X_all_discrete:   {X_all_discrete.shape}")
    print(f"  combo_ids: {combo_ids.shape}")

    print("\nIdentifying remaining candidates...")
    remaining_mask = identify_remaining_candidates(X_initial, X_all_discrete)
    n_remaining = remaining_mask.sum().item()
    n_train = X_initial.shape[0]
    print(f"  Training points: {n_train}")
    print(f"  Total combos:    {X_all_discrete.shape[0]}")
    print(f"  Remaining:       {n_remaining}")
    n_removed = X_all_discrete.shape[0] - n_remaining
    print(f"  Note: removed {n_removed} combinations that match the {n_train} training embeddings.")

    X_candidates = X_all_discrete[remaining_mask]
    combo_ids_remaining = combo_ids[remaining_mask]

    print("\nTraining SingleTaskGP...")
    model = train_gp(X_initial, Y_initial)
    best_f = Y_initial.max().item()
    print(f"  best_f = {best_f}")

    ei_acqf = ExpectedImprovement(model=model, best_f=best_f)
    print("\nEvaluating EI on remaining candidates...")
    ei_values = evaluate_acqf_batched(ei_acqf, X_candidates, name="EI")
    print(f"  EI range: [{ei_values.min().item():.6f}, {ei_values.max().item():.6f}]")
    print(f"  EI mean:  {ei_values.mean().item():.6f}")

    top_k = 10
    top_indices = ei_values.argsort(descending=True)[:top_k]
    print(f"\n  Top {top_k} EI candidates:")
    for rank, idx in enumerate(top_indices):
        c_id, s_id, b_id = combo_ids_remaining[idx].tolist()
        print(f"    {rank+1}. ({c_id}, {s_id}, {b_id})  EI = {ei_values[idx].item():.6f}")

    rng = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_remaining, generator=rng)
    pending_indices = perm[:N_PENDING]
    X_pending = X_candidates[pending_indices]
    pending_combo_ids = combo_ids_remaining[pending_indices]
    print(f"\nRandomly selected {N_PENDING} pending points (seed={SEED})")

    search_bounds = torch.stack([
        torch.zeros(X_initial.shape[1], dtype=DTYPE),
        torch.ones(X_initial.shape[1], dtype=DTYPE),
    ])

    print(f"\n--- Computing LP matrices ({N_PENDING} pending x {n_remaining} candidates) ---")

    print("\n[1/4] Soft Local LP...")
    sl_matrix = compute_lp_matrix(
        model, ei_values, X_candidates, X_pending, "soft", "local"
    )

    print("\n[2/4] Hard Local LP...")
    hl_matrix = compute_lp_matrix(
        model, ei_values, X_candidates, X_pending, "hard", "local"
    )

    print("\n[3/4] Soft Global LP...")
    sg_matrix = compute_lp_matrix(
        model, ei_values, X_candidates, X_pending, "soft", "global", search_bounds
    )

    print("\n[4/4] Hard Global LP...")
    hg_matrix = compute_lp_matrix(
        model, ei_values, X_candidates, X_pending, "hard", "global", search_bounds
    )

    print("\nBuilding results dict...")
    results = build_results_dict(
        X_candidates, combo_ids_remaining,
        ei_values, sl_matrix, hl_matrix, sg_matrix, hg_matrix,
    )

    results["__metadata__"] = {
        "seed": SEED,
        "n_pending": N_PENDING,
        "pending_indices": pending_indices.tolist(),
        "pending_combo_ids": pending_combo_ids.tolist(),
    }

    output_path = OUTPUT_DIR / "ei_values.json"
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results) - 1} candidate entries + metadata to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


def load_results():
    """Load ei_values.json and parse into tensors for analysis.

    Returns:
        Tuple of:
            - ``X``: ``(N, d)`` candidate features.
            - ``ei``: ``(N,)`` base EI values.
            - ``lp_matrices``: dict mapping mode name to ``(K, N)`` tensor.
            - ``metadata``: dict with seed, pending_indices, etc.
    """
    print("Loading ei_values.json...")
    with open(OUTPUT_DIR / "ei_values.json") as f:
        data = json.load(f)

    metadata = data.pop("__metadata__")
    n_pending = metadata["n_pending"]

    keys = list(data.keys())
    N = len(keys)
    print(f"  {N} candidates, {n_pending} pending points")

    d = len(data[keys[0]][0])
    X = torch.zeros(N, d, dtype=DTYPE)
    ei = torch.zeros(N, dtype=DTYPE)
    sl = torch.zeros(n_pending, N, dtype=DTYPE)
    hl = torch.zeros(n_pending, N, dtype=DTYPE)
    sg = torch.zeros(n_pending, N, dtype=DTYPE)
    hg = torch.zeros(n_pending, N, dtype=DTYPE)

    for i, key in enumerate(keys):
        entry = data[key]
        X[i] = torch.tensor(entry[0], dtype=DTYPE)
        ei[i] = entry[1]
        sl[:, i] = torch.tensor(entry[2], dtype=DTYPE)
        hl[:, i] = torch.tensor(entry[3], dtype=DTYPE)
        sg[:, i] = torch.tensor(entry[4], dtype=DTYPE)
        hg[:, i] = torch.tensor(entry[5], dtype=DTYPE)

    lp_matrices = {
        "Soft Local": sl,
        "Hard Local": hl,
        "Soft Global": sg,
        "Hard Global": hg,
    }

    return X, ei, lp_matrices, metadata


def analyze_and_plot(X, ei, lp_matrices, metadata):
    """Generate decay curve plot, top-K overlap plot, and summary table.

    Args:
        X: ``(N, d)`` candidate features.
        ei: ``(N,)`` base EI values.
        lp_matrices: Dict mapping mode name to ``(K, N)`` penalized EI tensor.
        metadata: Dict with ``pending_indices``, ``n_pending``, etc.
    """
    import matplotlib.pyplot as plt

    pending_indices = metadata["pending_indices"]
    n_pending = metadata["n_pending"]
    N = X.shape[0]

    # Filter bottom 30% EI candidates by percentile
    ei_threshold = torch.quantile(ei, EI_PERCENTILE_CUTOFF / 100.0).item()
    valid_mask = ei > ei_threshold
    n_valid = valid_mask.sum().item()
    print(f"\n  EI percentile cutoff: bottom {EI_PERCENTILE_CUTOFF}% (threshold = {ei_threshold:.8f})")
    print(f"  Candidates after filter: {n_valid} / {N}")

    X_valid = X[valid_mask]
    ei_valid = ei[valid_mask]

    # Map original pending indices to filtered indices
    # pending_indices index into the original N candidates
    valid_indices_map = torch.where(valid_mask)[0]

    mode_names = list(lp_matrices.keys())
    mode_colors = {
        "Soft Local": "#2196F3",
        "Hard Local": "#E91E63",
        "Soft Global": "#4CAF50",
        "Hard Global": "#FF9800",
    }

    # ── Plot 1: Penalty Decay Curve ──
    print("\n--- Computing Penalty Decay Curves ---")
    all_ratio_curves = {name: [] for name in mode_names}

    for j_idx in range(n_pending):
        pend_idx = pending_indices[j_idx]
        x_pend = X[pend_idx]

        dists_valid = torch.norm(X_valid - x_pend, dim=1)
        sort_order = dists_valid.argsort()

        for mode_name, lp_mat in lp_matrices.items():
            lp_valid = lp_mat[j_idx][valid_mask]
            ratios = lp_valid / ei_valid
            ratios_sorted = ratios[sort_order]
            all_ratio_curves[mode_name].append(ratios_sorted)

    # Stack: (n_pending, n_valid)
    for mode_name in mode_names:
        all_ratio_curves[mode_name] = torch.stack(all_ratio_curves[mode_name])

    ranks = torch.arange(1, n_valid + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1]})

    linestyles = {
        "Soft Local": "-",
        "Hard Local": "--",
        "Soft Global": "-.",
        "Hard Global": ":",
    }

    # Left panel: full range
    ax = axes[0]
    for mode_name in mode_names:
        curves = all_ratio_curves[mode_name]
        median_curve = curves.median(dim=0).values
        color = mode_colors[mode_name]
        ax.plot(ranks.numpy(), median_curve.numpy(), label=mode_name,
                color=color, linewidth=2, linestyle=linestyles[mode_name])

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.99, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Distance Rank", fontsize=12)
    ax.set_ylabel("Fraction of Original EI", fontsize=12)
    ax.set_title("Full Range (all candidates)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, n_valid)
    ax.grid(True, alpha=0.3)

    # Right panel: zoomed to first 500 ranks
    zoom_n = 500
    ax2 = axes[1]
    for mode_name in mode_names:
        curves = all_ratio_curves[mode_name]
        median_curve = curves.median(dim=0).values[:zoom_n]
        p25 = curves.quantile(0.25, dim=0)[:zoom_n]
        p75 = curves.quantile(0.75, dim=0)[:zoom_n]
        color = mode_colors[mode_name]
        ax2.plot(ranks[:zoom_n].numpy(), median_curve.numpy(), label=mode_name,
                 color=color, linewidth=2, linestyle=linestyles[mode_name])
        ax2.fill_between(ranks[:zoom_n].numpy(), p25.numpy(), p75.numpy(),
                         color=color, alpha=0.1)

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(y=0.99, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
    ax2.set_xlabel("Distance Rank", fontsize=12)
    ax2.set_title(f"Zoomed: First {zoom_n} Nearest Neighbors", fontsize=13)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(0, zoom_n)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("LP Penalty Decay: Penalized EI / Original EI vs. Distance Rank",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    decay_path = OUTPUT_DIR / "penalty_decay_curve.png"
    fig.savefig(decay_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {decay_path}")
    plt.close(fig)

    # ── Plot 2: Top-K Overlap ──
    print("\n--- Computing Top-K Overlap ---")
    K_vals = [1, 5, 10, 25, 50]
    overlap_results = {name: {k: [] for k in K_vals} for name in mode_names}

    for j_idx in range(n_pending):
        for mode_name, lp_mat in lp_matrices.items():
            lp_vals = lp_mat[j_idx]
            for k in K_vals:
                topk_ei = set(ei.argsort(descending=True)[:k].tolist())
                topk_lp = set(lp_vals.argsort(descending=True)[:k].tolist())
                overlap = len(topk_ei & topk_lp) / k
                overlap_results[mode_name][k].append(overlap)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.18
    x_positions = torch.arange(len(K_vals)).float()

    for m_idx, mode_name in enumerate(mode_names):
        means = [torch.tensor(overlap_results[mode_name][k]).mean().item() for k in K_vals]
        offset = (m_idx - 1.5) * bar_width
        bars = ax.bar(x_positions + offset, means, bar_width,
                      label=mode_name, color=mode_colors[mode_name], alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Top-{k}" for k in K_vals], fontsize=11)
    ax.set_ylabel("Fraction of Top-K Preserved After Penalization", fontsize=12)
    ax.set_title("Top-K Candidate Overlap: Unpenalized vs. Penalized EI", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    topk_path = OUTPUT_DIR / "topk_overlap.png"
    fig.savefig(topk_path, dpi=150)
    print(f"  Saved: {topk_path}")
    plt.close(fig)

    # ── Summary Table ──
    print("\n" + "=" * 90)
    print("  SUMMARY TABLE: Local Penalization Impact")
    print("=" * 90)

    header = f"{'Metric':<45s} {'Soft Local':>10s} {'Hard Local':>10s} {'Soft Glob':>10s} {'Hard Glob':>10s}"
    print(header)
    print("-" * 90)

    # Ratio at specific ranks (nearest neighbors)
    for rank_label, rank_idx in [("Ratio @ rank 1 (nearest)", 0),
                                  ("Ratio @ rank 10", 9),
                                  ("Ratio @ rank 100", 99),
                                  ("Ratio @ rank 1000", 999)]:
        vals = []
        for mode_name in mode_names:
            mean_val = all_ratio_curves[mode_name][:, rank_idx].mean().item()
            vals.append(mean_val)
        print(f"  {rank_label:<43s} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f} {vals[3]:10.4f}")

    # % of pending points where argmax changes
    vals = []
    for mode_name, lp_mat in lp_matrices.items():
        argmax_ei = ei.argmax().item()
        changed = sum(1 for j in range(n_pending) if lp_mat[j].argmax().item() != argmax_ei)
        vals.append(changed / n_pending * 100)
    print(f"  {'% pending pts where argmax changes':<43s} {vals[0]:9.1f}% {vals[1]:9.1f}% {vals[2]:9.1f}% {vals[3]:9.1f}%")

    # Mean new rank of original argmax
    vals = []
    for mode_name, lp_mat in lp_matrices.items():
        argmax_ei = ei.argmax().item()
        ranks_list = []
        for j in range(n_pending):
            sorted_lp = lp_mat[j].argsort(descending=True)
            new_rank = (sorted_lp == argmax_ei).nonzero(as_tuple=True)[0].item() + 1
            ranks_list.append(new_rank)
        vals.append(torch.tensor(ranks_list, dtype=torch.float).mean().item())
    print(f"  {'Mean new rank of original argmax':<43s} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:10.1f} {vals[3]:10.1f}")

    # Effective penalty radius: rank at which mean ratio > 0.99
    vals = []
    for mode_name in mode_names:
        mean_curve = all_ratio_curves[mode_name].mean(dim=0)
        above_99 = (mean_curve > 0.99).nonzero(as_tuple=True)[0]
        if len(above_99) > 0:
            vals.append(above_99[0].item() + 1)
        else:
            vals.append(n_valid)
    print(f"  {'Effective penalty radius (rank @ ratio>0.99)':<43s} {vals[0]:10d} {vals[1]:10d} {vals[2]:10d} {vals[3]:10d}")

    # Top-1 overlap (average)
    vals = []
    for mode_name in mode_names:
        vals.append(torch.tensor(overlap_results[mode_name][1]).mean().item() * 100)
    print(f"  {'Top-1 overlap (%)':<43s} {vals[0]:9.1f}% {vals[1]:9.1f}% {vals[2]:9.1f}% {vals[3]:9.1f}%")

    # Top-10 overlap (average)
    vals = []
    for mode_name in mode_names:
        vals.append(torch.tensor(overlap_results[mode_name][10]).mean().item() * 100)
    print(f"  {'Top-10 overlap (%)':<43s} {vals[0]:9.1f}% {vals[1]:9.1f}% {vals[2]:9.1f}% {vals[3]:9.1f}%")

    print("=" * 90)


def main():
    """Generate data if needed, then analyze and plot."""
    output_path = OUTPUT_DIR / "ei_values.json"

    if output_path.exists():
        print(f"Found existing {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
        print("Skipping data generation, jumping to analysis.\n")
    else:
        print(f"{output_path.name} not found. Generating data...\n")
        generate_data()

    X, ei, lp_matrices, metadata = load_results()
    analyze_and_plot(X, ei, lp_matrices, metadata)


if __name__ == "__main__":
    main()

