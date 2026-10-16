"""Rank featurization × dimensionality-reduction combinations by cross-validated CRPS.

Builds discrete-space features for catalyst, solvent and base from four
representations (MAP, Morgan, RDKit2D, RDKit3D), reduces them with PCA, mutual
information (SelectKBest) or PLS to k ∈ {10, 15} dimensions, fitted once on all
experiments, and appends the four normalized continuous variables. For each
combination a SingleTaskGP is scored by mean Gaussian CRPS under
leave-one-catalyst-out and point-wise leave-one-out splits of
``pablo/nhc_doe.xlsx``. Each split writes a sorted ranking table as a PNG next to
this script.

Usage:
    python -m pablo.results_analysis.combinations_featurization
"""

import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from skfp.fingerprints import MAPFingerprint

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

import pablo.main as pm
from pablo.utils.metrics import crps_gaussian

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-point Gaussian CRPS; wraps ``pablo.utils.metrics.crps_gaussian``.

    Args:
        mu: Predictive mean, shape ``(N,)`` or ``(N, 1)``.
        sigma: Predictive standard deviation, same shape as ``mu``; strictly positive.
        y: Observed value, same shape as ``mu``.

    Returns:
        Per-point CRPS, shape ``(N,)``.
    """
    return crps_gaussian(y, mu, sigma)


def featurize_morgan(smiles_list: List[str], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Compute Morgan bit-vector fingerprints.

    Args:
        smiles_list: SMILES strings.
        radius: Morgan radius.
        n_bits: Fingerprint length.

    Returns:
        Float64 array of shape ``(len(smiles_list), n_bits)``; a SMILES that
        RDKit cannot parse gets an all-zero row.
    """
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None: fps.append(np.zeros(n_bits))
        else: fps.append(np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)))
    return np.asarray(fps, dtype=np.float64)

def featurize_rdkit2d(smiles_list: List[str]) -> np.ndarray:
    """Compute all RDKit 2D descriptors (``Descriptors.descList``).

    Args:
        smiles_list: SMILES strings.

    Returns:
        Float64 array of shape ``(len(smiles_list), len(Descriptors.descList))``.
        Unparseable SMILES give an all-zero row; descriptors that raise, and
        NaN or ±inf values, are set to 0.0.
    """
    desc_funcs = [func for name, func in Descriptors.descList]
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None: fps.append(np.zeros(len(desc_funcs)))
        else:
            vals = []
            for f in desc_funcs:
                try: vals.append(f(mol))
                except: vals.append(0.0)
            fps.append(np.array(vals))
    arr = np.asarray(fps, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def featurize_rdkit3d(smiles_list: List[str]) -> np.ndarray:
    """Compute eight RDKit 3D shape descriptors followed by all RDKit 2D descriptors.

    The 3D block (asphericity, PMI1-3, PBF, NPR1-2, radius of gyration) comes
    from one ETKDGv3 conformer optimized with MMFF.

    Args:
        smiles_list: SMILES strings.

    Returns:
        Float64 array of shape ``(len(smiles_list), 8 + len(Descriptors.descList))``.
        Unparseable SMILES give an all-zero row; if embedding or any descriptor
        in the 3D path fails, the 3D block is zero; failing 2D descriptors, and
        NaN or ±inf values, are set to 0.0.
    """
    desc_funcs = [func for name, func in Descriptors.descList]
    fallback_len = len(desc_funcs)
    n_3d = 8
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(n_3d + fallback_len))
            continue
        try:
            m = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
            if res != 0: raise ValueError
            AllChem.MMFFOptimizeMolecule(m)
            vals = [rdMolDescriptors.CalcAsphericity(m), rdMolDescriptors.CalcPMI1(m), rdMolDescriptors.CalcPMI2(m),
                    rdMolDescriptors.CalcPMI3(m), rdMolDescriptors.CalcPBF(m), rdMolDescriptors.CalcNPR1(m),
                    rdMolDescriptors.CalcNPR2(m), rdMolDescriptors.CalcRadiusOfGyration(m)]
            vals2d = [f(mol) for f in desc_funcs]
            fps.append(np.array(vals + vals2d))
        except Exception:
            vals2d = []
            for f in desc_funcs:
                try: vals2d.append(f(mol))
                except: vals2d.append(0.0)
            fps.append(np.array([0.0]*n_3d + vals2d))
    arr = np.asarray(fps, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def apply_reduction(reducer_name, X_all, Y_all, k=15):
    """Standardize and reduce discrete-space features, fitted on all experiments at once.

    The reducer is fitted before any cross-validation split, so held-out points
    contribute to the reduction.

    Args:
        reducer_name: ``"PCA"``, ``"MutualInfo"`` (SelectKBest on mutual
            information) or ``"PLS"``.
        X_all: Raw discrete features, NumPy array of shape ``n x d_raw``.
        Y_all: Yields, NumPy array of shape ``n x 1``.
        k: Requested number of components; capped by ``n`` and ``d_raw`` (PCA),
            ``d_raw`` (MutualInfo) or ``max(1, n - 2)`` and ``d_raw`` (PLS).

    Returns:
        Tensor of shape ``n x n_components`` with dtype ``DTYPE`` on ``DEVICE``.

    Raises:
        ValueError: If ``reducer_name`` is not one of the three above.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    if reducer_name == "PCA":
        n_c = min(k, X_scaled.shape[0], X_scaled.shape[1])
        model = PCA(n_components=n_c)
        x_red = model.fit_transform(X_scaled)
    elif reducer_name == "MutualInfo":
        n_c = min(k, X_scaled.shape[1])
        model = SelectKBest(mutual_info_regression, k=n_c)
        x_red = model.fit_transform(X_scaled, Y_all.flatten())
    elif reducer_name == "PLS":
        n_samples = X_scaled.shape[0]
        n_c = min(k, max(1, n_samples - 2), X_scaled.shape[1])
        if n_c < 1: n_c = 1 
        model = PLSRegression(n_components=n_c)
        x_red, _ = model.fit_transform(X_scaled, Y_all.flatten())
    else:
        raise ValueError(f"Unknown reducer: {reducer_name}")
        
    return torch.tensor(x_red, dtype=DTYPE).to(DEVICE)

def load_data(excel_path):
    cat_ref = pd.read_excel(excel_path, sheet_name="Catalyst References Achiral").dropna(subset=["smiles"]).reset_index(drop=True)
    sol_ref = pd.read_excel(excel_path, sheet_name="Solvent References").dropna(subset=["smiles"]).reset_index(drop=True)
    bas_ref = pd.read_excel(excel_path, sheet_name="Base References").dropna(subset=["smiles"]).reset_index(drop=True)
    exp_df = pd.read_excel(excel_path, sheet_name="Experimental Design").dropna(
        subset=["Yield", "Temperature", "Formaldehyde Ratio in Eq.", "Catalyst Loading in Molar Ratio", "Time"]
    ).reset_index(drop=True)
    return cat_ref, sol_ref, bas_ref, exp_df

def build_dataset_raw(exp_df, cat_ref, sol_ref, bas_ref, cat_raw, sol_raw, bas_raw, sol_name_to_id, bas_name_to_id):
    cat_ids = cat_ref["ID"].tolist()
    sol_ids = sol_ref["ID"].tolist()
    bas_ids = bas_ref["ID"].tolist()
    
    cat_map = {cat_ids[i]: cat_raw[i] for i in range(len(cat_ids))}
    sol_map = {sol_ids[i]: sol_raw[i] for i in range(len(sol_ids))}
    bas_map = {bas_ids[i]: bas_raw[i] for i in range(len(bas_ids))}
    
    no_base_fallback_id = 0 if 0 in bas_map else sorted(bas_map.keys())[0]
    base_text_fallback_ids = {"no": no_base_fallback_id, "none": no_base_fallback_id, "n/a": no_base_fallback_id, "na": no_base_fallback_id, "-": no_base_fallback_id}

    solvent_exp_ids = [pm._resolve_component_id(v, sol_map, sol_name_to_id, "Solvent") for v in exp_df["Solvent"].tolist()]
    base_exp_ids = [pm._resolve_component_id(v, bas_map, bas_name_to_id, "Base", fallback_text_to_id=base_text_fallback_ids) for v in exp_df["Base"].tolist()]
    cat_exp_ids = exp_df["Catalyst"].tolist()
    
    x_cat = np.stack([cat_map[v] for v in cat_exp_ids])
    x_sol = np.stack([sol_map[v] for v in solvent_exp_ids])
    x_bas = np.stack([bas_map[v] for v in base_exp_ids])
    
    X_disc = np.concatenate([x_cat, x_sol, x_bas], axis=1)
    
    cont_bounds = pm.ContinuousBounds().to_tensor()
    temperature = torch.tensor(pd.to_numeric(exp_df["Temperature"]).values, dtype=DTYPE)
    form_ratio = torch.tensor(pd.to_numeric(exp_df["Formaldehyde Ratio in Eq."]).values, dtype=DTYPE)
    cat_loading = torch.tensor(pd.to_numeric(exp_df["Catalyst Loading in Molar Ratio"]).values, dtype=DTYPE)
    time = torch.tensor(pd.to_numeric(exp_df["Time"]).values, dtype=DTYPE)

    temp_norm = (temperature - cont_bounds[0, 0]) / (cont_bounds[1, 0] - cont_bounds[0, 0])
    form_norm = (form_ratio - cont_bounds[0, 1]) / (cont_bounds[1, 1] - cont_bounds[0, 1])
    load_norm = (cat_loading - cont_bounds[0, 2]) / (cont_bounds[1, 2] - cont_bounds[0, 2])
    time_norm = (time - cont_bounds[0, 3]) / (cont_bounds[1, 3] - cont_bounds[0, 3])

    X_continuous = torch.stack([temp_norm, form_norm, load_norm, time_norm], dim=1).to(DEVICE)
    Y = torch.tensor(exp_df["Yield"].values, dtype=DTYPE).unsqueeze(-1).to(DEVICE)
    
    return X_continuous, X_disc, Y, cat_exp_ids

def run_validation(masks, title, out_filename, combinations_data, Y):
    """Score every combination by mean Gaussian CRPS over the given splits and plot the ranking.

    For each combination and each split, a SingleTaskGP (Normalize input,
    Standardize outcome) is fitted on the training rows and CRPS is computed for
    every held-out row. The sorted table is printed and saved as a PNG next to
    this script.

    Args:
        masks: Boolean NumPy arrays of shape ``(n,)``, one per split; True marks
            the held-out rows.
        title: Figure title.
        out_filename: PNG filename, written in this script's directory.
        combinations_data: Dicts with keys ``"base"``, ``"reducer"``, ``"k"`` and
            ``"X_full"`` (tensor of shape ``n x (4 + n_components)``).
        Y: Yields, tensor of shape ``n x 1``.
    """
    results = []
    total_combos = len(combinations_data)
    
    for idx, combo in enumerate(combinations_data):
        X_full = combo["X_full"]
        
        combo_crps_list = []
        for test_mask in masks:
            train_mask = ~test_mask
            X_tr, X_te = X_full[train_mask], X_full[test_mask]
            Y_tr, Y_te = Y[train_mask], Y[test_mask]
            
            # Same GP as main.py
            model = SingleTaskGP(
                train_X=X_tr,
                train_Y=Y_tr,
                input_transform=Normalize(d=X_full.shape[-1]),
                outcome_transform=Standardize(m=1),
            ).to(DEVICE)
            
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(DEVICE)
            
            try: fit_gpytorch_mll(mll)
            except: pass
            
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(X_te)
                mu = posterior.mean.squeeze(-1)
                var = posterior.variance.squeeze(-1)
                
                if mu.dim() == 0:
                    mu, var = mu.unsqueeze(0), var.unsqueeze(0)
                    y_true = Y_te.squeeze(-1)
                    if y_true.dim() == 0: y_true = y_true.unsqueeze(0)
                else:
                    y_true = Y_te.squeeze(-1)
                    
                for j in range(len(mu)):
                    c = gaussian_crps(mu[j], var[j].sqrt(), y_true[j])
                    combo_crps_list.append(c.item())
                    
        avg_crps = np.mean(combo_crps_list) if len(combo_crps_list) > 0 else float('nan')
        results.append({
            "Featurization": combo["base"],
            "Reduction": combo["reducer"],
            "Target Dim (k)": combo["k"],
            "CRPS": avg_crps
        })
        
    df_res = pd.DataFrame(results)
    df_sorted = df_res.sort_values("CRPS").reset_index(drop=True)
    
    print(f"\n=== Final Sorted Results: {title} ===")
    print(df_sorted.to_string(index=False))

    # --- Draw A Clean Table ---
    fig, ax = plt.subplots(figsize=(8, len(df_sorted)*0.4 + 1.0))
    ax.axis('off'); ax.axis('tight')
    
    col_labels = df_sorted.columns.tolist()
    table_data = []
    for i in range(len(df_sorted)):
        row = [
            str(df_sorted.iloc[i]["Featurization"]),
            str(df_sorted.iloc[i]["Reduction"]),
            str(df_sorted.iloc[i]["Target Dim (k)"]),
            f"{df_sorted.iloc[i]['CRPS']:.3f}"
        ]
        table_data.append(row)
        
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Highlight winner row
    for c_idx in range(len(col_labels)):
        cell = table[1, c_idx]
        cell.get_text().set_weight('bold')
        cell.get_text().set_color('red')

    plt.title(title, pad=20, fontsize=14, weight='bold')
    out_path = Path(__file__).resolve().parent / out_filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved ranking table to {out_path}")

def main():
    excel_file = Path(__file__).resolve().parent.parent / "nhc_doe.xlsx"
    print(f"Loading data from {excel_file}")
    cat_ref, sol_ref, bas_ref, exp_df = load_data(excel_file)
    sol_id_to_name, bas_id_to_name, sol_name_to_id, bas_name_to_id = pm._load_solvent_base_name_maps(excel_file)
    
    bases = ["MAP", "Morgan", "RDKit2D", "RDKit3D"]
    reducers = ["PCA", "MutualInfo", "PLS"]
    k_list = [10, 15]  # Will test dimensions
    
    raw_feat = {}
    print("Computing Raw Featurizations...")
    print("  -> MAP")
    raw_feat["MAP"] = {"cat": np.asarray(MAPFingerprint(radius=2).transform(cat_ref["smiles"].tolist()), dtype=np.float64), "sol": np.asarray(MAPFingerprint(radius=1).transform(sol_ref["smiles"].tolist()), dtype=np.float64), "bas": np.asarray(MAPFingerprint(radius=1).transform(bas_ref["smiles"].tolist()), dtype=np.float64)}
    print("  -> Morgan")
    raw_feat["Morgan"] = {"cat": featurize_morgan(cat_ref["smiles"].tolist(), 2), "sol": featurize_morgan(sol_ref["smiles"].tolist(), 1), "bas": featurize_morgan(bas_ref["smiles"].tolist(), 1)}
    print("  -> RDKit2D")
    raw_feat["RDKit2D"] = {"cat": featurize_rdkit2d(cat_ref["smiles"].tolist()), "sol": featurize_rdkit2d(sol_ref["smiles"].tolist()), "bas": featurize_rdkit2d(bas_ref["smiles"].tolist())}
    print("  -> RDKit3D")
    raw_feat["RDKit3D"] = {"cat": featurize_rdkit3d(cat_ref["smiles"].tolist()), "sol": featurize_rdkit2d(sol_ref["smiles"].tolist()), "bas": featurize_rdkit2d(bas_ref["smiles"].tolist())}
    
    datasets = {}
    for b in bases:
        X_cont, X_disc, Y, cat_exp_ids = build_dataset_raw(exp_df, cat_ref, sol_ref, bas_ref, raw_feat[b]["cat"], raw_feat[b]["sol"], raw_feat[b]["bas"], sol_name_to_id, bas_name_to_id)
        datasets[b] = X_disc
    
    # Pre-compute all dimensionality reductions mapping GLOBALLY exactly as requested
    print("\nPre-computing global dimensionality reductions across ALL points...")
    combinations_data = []
    for b in bases:
        for r in reducers:
            for k in k_list:
                X_reduced = apply_reduction(r, datasets[b], Y.cpu().numpy(), k=k)
                X_full = torch.cat([X_cont, X_reduced], dim=1)
                combinations_data.append({"base": b, "reducer": r, "k": k, "X_full": X_full})

    cat_exp_ids = np.array(cat_exp_ids)
    unique_cats = np.unique(cat_exp_ids)
    n_points = len(cat_exp_ids)
    
    print("\n" + "="*50)
    print(" 1. LEAVE-ONE-CATALYST-OUT VALIDATION (GLOBAL FEAT) ")
    print("="*50)
    cat_masks = [cat_exp_ids == c_id for c_id in unique_cats]
    run_validation(cat_masks, "Catalyst-wise LOO: Sorted Rank", "leave_catalyst_combinations_ranking.png", combinations_data, Y)

    print("\n" + "="*50)
    print(" 2. POINT-WISE LEAVE-ONE-OUT VALIDATION (GLOBAL FEAT) ")
    print("="*50)
    pt_masks = [np.arange(n_points) == i for i in range(n_points)]
    run_validation(pt_masks, "Point-wise LOO: Sorted Rank", "leave_point_combinations_ranking.png", combinations_data, Y)

if __name__ == "__main__":
    main()
