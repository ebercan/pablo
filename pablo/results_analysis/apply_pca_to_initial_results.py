"""
Precompute PCA-reduced representations for the initial dataset and whole discrete space.

This script computes molecular fingerprints (MAP4) from SMILES strings of the
catalysts, solvents, and bases. It then applies Principal Component Analysis (PCA)
to determine continuous numerical embeddings.

The target dimensionality for the PCA is automatically determined by constructing
an explained variance curve and using the Kneedle algorithm to locate the elbow point.

The script produces the following outputs:
- `X_initial`: A tensor of shape (N_initial, D_total) containing the concatenated
  (catalyst, solvent, base) PCA features for the evaluated experiments.
- `Y_initial`: A tensor of shape (N_initial, 1) containing the yield values.
- `X_all_discrete`: A tensor of shape (N_combinations, D_total) containing the
  features for all valid discrete combinations of the chemical species.
- `combo_ids`: A tensor of shape (N_combinations, 3) mapping each combination to its constituent IDs.

Usage:
    python -m pablo.results_analysis.apply_pca_to_initial_results
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pablo.utils.dim_reduction import apply_pca
from botorch.models.transforms import Normalize
from skfp.fingerprints import MAPFingerprint

DTYPE = torch.float64

OUTPUT_DIR = Path(__file__).parent
EXCEL_FILE = Path(__file__).parent.parent / "nhc_doe.xlsx"


def apply_pca_and_normalize(fingerprints, kneedle_sensitivity=1):
    """Apply PCA to compute continuous molecular embeddings and min-max normalize.

    The number of PCA components is the elbow that the Kneedle algorithm finds on the
    per-component explained variance ratio (sorted, decreasing). If Kneedle finds no
    elbow, it is the smallest number of components reaching 80% cumulative explained
    variance. Fingerprints are standardized before PCA.

    Args:
        fingerprints: ``(n, d_raw)`` numpy array of raw molecular fingerprints.
        kneedle_sensitivity: Sensitivity parameter for the Kneedle algorithm
            used to automatically select the number of PCA components.

    Returns:
        A ``(n, d_pca)`` Tensor of PCA-reduced, [0,1]-normalized features.
    """
    n_samples = fingerprints.shape[0]
    dummy_labels = pd.Series(range(n_samples), name="id")

    df_fingerprints = pd.DataFrame(fingerprints)
    df_fingerprints.insert(0, "id", dummy_labels)

    _, values_tensor, _ = apply_pca(
        df_fingerprints,
        n_components=None,
        kneedle_sensitivity=int(kneedle_sensitivity),
    )

    bounds = torch.stack(
        [values_tensor.min(dim=0).values, values_tensor.max(dim=0).values]
    )
    normalized = Normalize(d=values_tensor.shape[1], bounds=bounds)(values_tensor)
    return normalized


def precompute_and_save(excel_file=None):
    """Load structural datasets, apply PCA to fingerprints, and save the tensors.

    This routine performs the feature engineering end-to-end:
    - Loads the initial yields and the underlying candidate species.
    - Computes fingerprints and derives dense PCA embeddings with auto-selected dimensionality.
    - Concatenates embeddings for the observed data (`X_initial` and `Y_initial`).
    - Constructs the entire candidate space representation (`X_all_discrete`).

    Args:
        excel_file: Path to the NHC dataset Excel workbook. If None, uses defaults.
    """
    if excel_file is None:
        excel_file = EXCEL_FILE

    print(f"Loading dataset from: {excel_file}")

    exp_df = pd.read_excel(excel_file, sheet_name="Experimental Design")

    target_col = "Yield"

    catalyst_col = "Catalyst"
    solvent_col = "Solvent"
    base_col = "Base"

    exp_df = exp_df.dropna(subset=[target_col]).reset_index(drop=True)

    cat_ref = pd.read_excel(excel_file, sheet_name="Catalyst References Achiral")
    sol_ref = pd.read_excel(excel_file, sheet_name="Solvent References")
    bas_ref = pd.read_excel(excel_file, sheet_name="Base References")

    cat_ref = cat_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    sol_ref = sol_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    bas_ref = bas_ref.dropna(subset=["smiles"]).reset_index(drop=True)

    print("Computing fingerprints...")
    cat_fp = np.asarray(
        MAPFingerprint(radius=2).transform(cat_ref["smiles"].tolist()),
        dtype=np.float64,
    )
    sol_fp = np.asarray(
        MAPFingerprint(radius=1).transform(sol_ref["smiles"].tolist()),
        dtype=np.float64,
    )
    bas_fp = np.asarray(
        MAPFingerprint(radius=1).transform(bas_ref["smiles"].tolist()),
        dtype=np.float64,
    )

    print("Applying PCA to catalyst embeddings...")
    cat_emb = apply_pca_and_normalize(cat_fp).to(DTYPE)
    print(f"  Catalyst PCA shape: {cat_emb.shape}")

    print("Applying PCA to solvent embeddings...")
    sol_emb = apply_pca_and_normalize(sol_fp).to(DTYPE)
    print(f"  Solvent PCA shape: {sol_emb.shape}")

    print("Applying PCA to base embeddings...")
    bas_emb = apply_pca_and_normalize(bas_fp).to(DTYPE)
    print(f"  Base PCA shape: {bas_emb.shape}")

    cat_ids = cat_ref["ID"].tolist()
    sol_ids = sol_ref["ID"].tolist()
    bas_ids = bas_ref["ID"].tolist()

    cat_map = {cat_ids[i]: cat_emb[i] for i in range(len(cat_ids))}
    sol_map = {sol_ids[i]: sol_emb[i] for i in range(len(sol_ids))}
    bas_map = {bas_ids[i]: bas_emb[i] for i in range(len(bas_ids))}

    x_cat = torch.stack([cat_map[v] for v in exp_df[catalyst_col]])
    x_sol = torch.stack([sol_map[v] for v in exp_df[solvent_col]])
    x_bas = torch.stack([bas_map[v] for v in exp_df[base_col]])

    X_initial = torch.cat([x_cat, x_sol, x_bas], dim=1).to(DTYPE)
    Y_initial = torch.tensor(exp_df[target_col].to_numpy(), dtype=DTYPE).unsqueeze(-1)

    n_total = len(exp_df)
    assert (
        X_initial.shape[0] == n_total
    ), f"Expected {n_total} datapoints, got {X_initial.shape[0]}"

    sort_order = Y_initial.squeeze(-1).argsort()
    X_initial = X_initial[sort_order]
    Y_initial = Y_initial[sort_order]

    print(f"\nTraining data shapes:")
    print(f"  X_initial: {X_initial.shape}")
    print(f"  Y_initial: {Y_initial.shape}")

    # Build the full combinatorial candidate tensor: all (cat, sol, bas) combos
    print("\nBuilding full combinatorial candidate tensor...")
    n_cats, n_sols, n_bas = len(cat_ids), len(sol_ids), len(bas_ids)
    n_combos = n_cats * n_sols * n_bas
    d = X_initial.shape[1]

    X_all_discrete = torch.empty(n_combos, d, dtype=DTYPE)
    combo_ids = torch.empty(n_combos, 3, dtype=torch.long)

    idx = 0
    for c_id in cat_ids:
        for s_id in sol_ids:
            for b_id in bas_ids:
                X_all_discrete[idx] = torch.cat(
                    [cat_map[c_id], sol_map[s_id], bas_map[b_id]]
                )
                combo_ids[idx] = torch.tensor([c_id, s_id, b_id], dtype=torch.long)
                idx += 1

    assert idx == n_combos, f"Expected {n_combos} combos, built {idx}"

    print(
        f"  X_all_discrete: {X_all_discrete.shape}  ({n_cats} cats x {n_sols} sols x {n_bas} bas)"
    )
    print(f"  combo_ids: {combo_ids.shape}")

    x_path = OUTPUT_DIR / "X_initial.pt"
    y_path = OUTPUT_DIR / "Y_initial.pt"
    x_all_path = OUTPUT_DIR / "X_all_discrete.pt"
    combo_ids_path = OUTPUT_DIR / "combo_ids.pt"

    torch.save(X_initial, x_path)
    torch.save(Y_initial, y_path)
    torch.save(X_all_discrete, x_all_path)
    torch.save(combo_ids, combo_ids_path)

    print(f"\nSaved tensors to:")
    print(f"  {x_path}")
    print(f"  {y_path}")
    print(f"  {x_all_path}")
    print(f"  {combo_ids_path}")


if __name__ == "__main__":
    precompute_and_save()
