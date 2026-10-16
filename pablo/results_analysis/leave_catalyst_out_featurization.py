"""
Leave-one-catalyst-out cross-validation for different featurization strategies.
Metric: Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions.

Usage:
    python -m pablo.results_analysis.leave_catalyst_out_featurization
"""

import math
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

import pablo.main as pm
from pablo.utils.metrics import crps_gaussian

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-point Gaussian CRPS with arguments in (mu, sigma, y) order.

    Delegates to ``pablo.utils.metrics.crps_gaussian``.

    Args:
        mu: Predictive means, shape `n` or `n x 1`.
        sigma: Predictive standard deviations, shape `n` or `n x 1`, strictly positive.
        y: Observed targets, shape `n` or `n x 1`.

    Returns:
        Per-point CRPS, shape `n`.
    """
    return crps_gaussian(y, mu, sigma)


def apply_pca_local(fingerprints: np.ndarray, kneedle_sensitivity: int = 1) -> torch.Tensor:
    """Reduce raw features by PCA and min-max scale each component to [0, 1].

    The number of components is chosen by the Kneedle elbow in
    ``pablo.utils.dim_reduction.apply_pca``. A component with range below 1e-8 is
    shifted to its minimum and not rescaled.

    Args:
        fingerprints: Raw feature matrix, shape `n x d_raw`.
        kneedle_sensitivity: Kneedle sensitivity ``S`` for the elbow selection.

    Returns:
        Normalized PCA embedding, shape `n x d_pca`, dtype ``DTYPE``.
    """
    from pablo.utils.dim_reduction import apply_pca
    n_samples = fingerprints.shape[0]
    dummy_labels = pd.Series(range(n_samples), name="id")
    df = pd.DataFrame(fingerprints)
    df.insert(0, "id", dummy_labels)
    _, values_tensor, _ = apply_pca(df, n_components=None, kneedle_sensitivity=int(kneedle_sensitivity))
    
    bounds = torch.stack([values_tensor.min(dim=0).values, values_tensor.max(dim=0).values])
    eps = 1e-8
    range_vals = bounds[1] - bounds[0]
    range_vals = torch.where(range_vals < eps, torch.ones_like(range_vals), range_vals)
    normalized = (values_tensor - bounds[0]) / range_vals
    return normalized.to(DTYPE)

def featurize_map(smiles_list: List[str], radius: int) -> torch.Tensor:
    """MAP fingerprints (scikit-fingerprints ``MAPFingerprint``) reduced by ``apply_pca_local``.

    Args:
        smiles_list: SMILES strings, length `n`.
        radius: MAP fingerprint radius.

    Returns:
        Embedding, shape `n x d_pca`, dtype ``DTYPE``.
    """
    fp = np.asarray(MAPFingerprint(radius=radius).transform(smiles_list), dtype=np.float64)
    return apply_pca_local(fp)

def featurize_morgan(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    """Morgan bit-vector fingerprints reduced by ``apply_pca_local``.

    A SMILES that RDKit cannot parse gets an all-zero bit vector.

    Args:
        smiles_list: SMILES strings, length `n`.
        radius: Morgan radius.
        n_bits: Bit-vector length.

    Returns:
        Embedding, shape `n x d_pca`, dtype ``DTYPE``.
    """
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(n_bits))
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
    fp_array = np.asarray(fps, dtype=np.float64)
    return apply_pca_local(fp_array)

def featurize_rdkit2d(smiles_list: List[str]) -> torch.Tensor:
    """All RDKit 2D descriptors (``Descriptors.descList``) reduced by ``apply_pca_local``.

    An unparseable SMILES gets an all-zero row; a descriptor that raises, and any
    NaN or ±inf value, is set to 0.

    Args:
        smiles_list: SMILES strings, length `n`.

    Returns:
        Embedding, shape `n x d_pca`, dtype ``DTYPE``.
    """
    fps = []
    desc_funcs = [func for name, func in Descriptors.descList]
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(len(desc_funcs)))
        else:
            vals = []
            for func in desc_funcs:
                try:
                    v = func(mol)
                except Exception:
                    v = 0.0
                vals.append(v)
            fps.append(np.array(vals))
    fp_array = np.asarray(fps, dtype=np.float64)
    # Remove NaN/Inf
    fp_array = np.nan_to_num(fp_array, nan=0.0, posinf=0.0, neginf=0.0)
    # PCA to reduce highly correlated descriptors
    return apply_pca_local(fp_array)

def featurize_rdkit3d(smiles_list: List[str]) -> torch.Tensor:
    """Eight 3D shape descriptors plus all RDKit 2D descriptors, reduced by ``apply_pca_local``.

    Each molecule is embedded with ETKDGv3 and MMFF-optimized; the 3D block is
    asphericity, PMI1–3, PBF, NPR1–2 and radius of gyration. If embedding or
    optimization fails, the 3D block is zero and only the 2D block is filled.
    An unparseable SMILES gets an all-zero row; NaN and ±inf are set to 0.

    Args:
        smiles_list: SMILES strings, length `n`.

    Returns:
        Embedding, shape `n x d_pca`, dtype ``DTYPE``.
    """
    fps = []
    # 2D fallback length (we use 2D descriptors as fallback if 3D fails)
    desc_funcs = [func for name, func in Descriptors.descList]
    fallback_len = len(desc_funcs)
    # 3D features: Asphericity, PMI1/2/3, PBF, NPR1/2, RadiusOfGyration
    n_3d = 8
    
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(n_3d + fallback_len))
            continue
            
        try:
            m = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
            if res != 0:
                raise ValueError("Embedding failed")
            AllChem.MMFFOptimizeMolecule(m)
            
            asphericity = rdMolDescriptors.CalcAsphericity(m)
            pmi1 = rdMolDescriptors.CalcPMI1(m)
            pmi2 = rdMolDescriptors.CalcPMI2(m)
            pmi3 = rdMolDescriptors.CalcPMI3(m)
            pbf = rdMolDescriptors.CalcPBF(m)
            npr1 = rdMolDescriptors.CalcNPR1(m)
            npr2 = rdMolDescriptors.CalcNPR2(m)
            gyr = rdMolDescriptors.CalcRadiusOfGyration(m)
            
            # Combine 3D descriptors with some basic 2D structural props for completeness
            # we'll just append zeroes for 2D to keep shape consistent, 
            # actually let's just append the 2D descriptors to the 3D ones.
            vals2d = [func(mol) for func in desc_funcs]
            fps.append(np.array([asphericity, pmi1, pmi2, pmi3, pbf, npr1, npr2, gyr] + vals2d))
        except Exception as e:
            # Fallback to 2D
            vals2d = []
            for func in desc_funcs:
                try:
                    v = func(mol)
                except Exception:
                    v = 0.0
                vals2d.append(v)
            fps.append(np.array([0.0] * n_3d + vals2d))
            
    fp_array = np.asarray(fps, dtype=np.float64)
    fp_array = np.nan_to_num(fp_array, nan=0.0, posinf=0.0, neginf=0.0)
    return apply_pca_local(fp_array)

# --- End Featurization ---

def load_data(excel_path: Path):
    """Load the reference sheets and the experimental design from the NHC workbook.

    Reference rows without a SMILES and design rows missing yield or any of the four
    continuous conditions are dropped.

    Args:
        excel_path: Path to ``nhc_doe.xlsx``.

    Returns:
        Tuple ``(cat_ref, sol_ref, bas_ref, exp_df)``: the "Catalyst References
        Achiral", "Solvent References" and "Base References" sheets, and the
        "Experimental Design" sheet, each as a DataFrame with a reset index.
    """
    cat_ref = pd.read_excel(excel_path, sheet_name="Catalyst References Achiral").dropna(subset=["smiles"]).reset_index(drop=True)
    sol_ref = pd.read_excel(excel_path, sheet_name="Solvent References").dropna(subset=["smiles"]).reset_index(drop=True)
    bas_ref = pd.read_excel(excel_path, sheet_name="Base References").dropna(subset=["smiles"]).reset_index(drop=True)

    exp_df = pd.read_excel(excel_path, sheet_name="Experimental Design").dropna(
        subset=["Yield", "Temperature", "Formaldehyde Ratio in Eq.", "Catalyst Loading in Molar Ratio", "Time"]
    ).reset_index(drop=True)
    
    return cat_ref, sol_ref, bas_ref, exp_df

def build_dataset_for_strategy(
    exp_df: pd.DataFrame,
    cat_ref: pd.DataFrame, sol_ref: pd.DataFrame, bas_ref: pd.DataFrame,
    cat_emb: torch.Tensor, sol_emb: torch.Tensor, bas_emb: torch.Tensor,
    sol_name_to_id: dict, bas_name_to_id: dict, bas_map_keys: list
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Build the GP training inputs and targets for one featurization strategy.

    Continuous columns (temperature, formaldehyde ratio, catalyst loading, time) are
    min-max normalized with ``pablo.main.ContinuousBounds``. Solvent and base entries
    are resolved to reference IDs with ``pablo.main._resolve_component_id``; a
    "no"/"none"/"n/a"/"na"/"-" base maps to base ID 0 (or the smallest base ID if 0
    is absent). Base embeddings are not zeroed for non-salt catalysts.

    Args:
        exp_df: Experimental design, `n` rows.
        cat_ref: Catalyst reference sheet; row order matches ``cat_emb``.
        sol_ref: Solvent reference sheet; row order matches ``sol_emb``.
        bas_ref: Base reference sheet; row order matches ``bas_emb``.
        cat_emb: Catalyst embeddings, shape `n_cat x d_cat`.
        sol_emb: Solvent embeddings, shape `n_sol x d_sol`.
        bas_emb: Base embeddings, shape `n_bas x d_bas`.
        sol_name_to_id: Solvent name to reference ID.
        bas_name_to_id: Base name to reference ID.
        bas_map_keys: Unused.

    Returns:
        Tuple ``(X, Y, cat_exp_ids)``: ``X`` of shape `n x (4 + d_cat + d_sol + d_bas)`
        with columns ``[continuous | catalyst | solvent | base]``, ``Y`` (yield) of
        shape `n x 1`, and the catalyst ID of each row.
    """
    
    cat_ids = cat_ref["ID"].tolist()
    sol_ids = sol_ref["ID"].tolist()
    bas_ids = bas_ref["ID"].tolist()
    
    cat_map = {cat_ids[i]: cat_emb[i] for i in range(len(cat_ids))}
    sol_map = {sol_ids[i]: sol_emb[i] for i in range(len(sol_ids))}
    bas_map = {bas_ids[i]: bas_emb[i] for i in range(len(bas_ids))}
    
    no_base_fallback_id = 0 if 0 in bas_map else sorted(bas_map.keys())[0]
    base_text_fallback_ids = {
        "no": no_base_fallback_id, "none": no_base_fallback_id, "n/a": no_base_fallback_id,
        "na": no_base_fallback_id, "-": no_base_fallback_id,
    }

    solvent_exp_ids = [
        pm._resolve_component_id(v, sol_map, sol_name_to_id, "Solvent")
        for v in exp_df["Solvent"].tolist()
    ]
    base_exp_ids = [
        pm._resolve_component_id(v, bas_map, bas_name_to_id, "Base", fallback_text_to_id=base_text_fallback_ids)
        for v in exp_df["Base"].tolist()
    ]
    
    cat_exp_ids = exp_df["Catalyst"].tolist()
    
    x_cat = torch.stack([cat_map[v] for v in cat_exp_ids])
    x_sol = torch.stack([sol_map[v] for v in solvent_exp_ids])
    x_bas = torch.stack([bas_map[v] for v in base_exp_ids])
    
    cont_bounds = pm.ContinuousBounds().to_tensor()
    temperature = torch.tensor(pd.to_numeric(exp_df["Temperature"]).values, dtype=DTYPE)
    form_ratio = torch.tensor(pd.to_numeric(exp_df["Formaldehyde Ratio in Eq."]).values, dtype=DTYPE)
    cat_loading = torch.tensor(pd.to_numeric(exp_df["Catalyst Loading in Molar Ratio"]).values, dtype=DTYPE)
    time = torch.tensor(pd.to_numeric(exp_df["Time"]).values, dtype=DTYPE)

    temp_norm = (temperature - cont_bounds[0, 0]) / (cont_bounds[1, 0] - cont_bounds[0, 0])
    form_norm = (form_ratio - cont_bounds[0, 1]) / (cont_bounds[1, 1] - cont_bounds[0, 1])
    load_norm = (cat_loading - cont_bounds[0, 2]) / (cont_bounds[1, 2] - cont_bounds[0, 2])
    time_norm = (time - cont_bounds[0, 3]) / (cont_bounds[1, 3] - cont_bounds[0, 3])

    X_continuous = torch.stack([temp_norm, form_norm, load_norm, time_norm], dim=1)
    X = torch.cat([X_continuous, x_cat, x_sol, x_bas], dim=1).to(DTYPE)
    Y = torch.tensor(exp_df["Yield"].values, dtype=DTYPE).unsqueeze(-1)
    
    return X, Y, cat_exp_ids

def main():
    excel_file = Path(__file__).resolve().parent.parent / "nhc_doe.xlsx"
    print(f"Loading data from {excel_file}")
    cat_ref, sol_ref, bas_ref, exp_df = load_data(excel_file)
    
    sol_id_to_name, bas_id_to_name, sol_name_to_id, bas_name_to_id = pm._load_solvent_base_name_maps(excel_file)
    
    strategies = ["MAP_PCA", "Morgan_PCA", "RDKit2D_PCA", "RDKit3D_PCA"]
    
    # Pre-compute embeddings for all strategies
    embeddings = {}
    print("Computing Featurizations...")
    
    # 1. MAP
    print("  -> MAP_PCA")
    embeddings["MAP_PCA"] = {
        "cat": featurize_map(cat_ref["smiles"].tolist(), radius=2),
        "sol": featurize_map(sol_ref["smiles"].tolist(), radius=1),
        "bas": featurize_map(bas_ref["smiles"].tolist(), radius=1)
    }
    
    # 2. Morgan
    print("  -> Morgan_PCA")
    embeddings["Morgan_PCA"] = {
        "cat": featurize_morgan(cat_ref["smiles"].tolist(), radius=2),
        "sol": featurize_morgan(sol_ref["smiles"].tolist(), radius=1),
        "bas": featurize_morgan(bas_ref["smiles"].tolist(), radius=1)
    }
    
    # 3. RDKit2D
    print("  -> RDKit2D_PCA")
    embeddings["RDKit2D_PCA"] = {
        "cat": featurize_rdkit2d(cat_ref["smiles"].tolist()),
        "sol": featurize_rdkit2d(sol_ref["smiles"].tolist()),
        "bas": featurize_rdkit2d(bas_ref["smiles"].tolist())
    }
    
    # 4. RDKit3D
    print("  -> RDKit3D_PCA")
    embeddings["RDKit3D_PCA"] = {
        "cat": featurize_rdkit3d(cat_ref["smiles"].tolist()),
        "sol": featurize_rdkit2d(sol_ref["smiles"].tolist()), # sol/base use 2d
        "bas": featurize_rdkit2d(bas_ref["smiles"].tolist())
    }

    # Build datasets
    datasets = {}
    cat_exp_ids = None
    for name in strategies:
        X, Y, c_ids = build_dataset_for_strategy(
            exp_df, cat_ref, sol_ref, bas_ref,
            embeddings[name]["cat"], embeddings[name]["sol"], embeddings[name]["bas"],
            sol_name_to_id, bas_name_to_id, bas_ref["ID"].tolist()
        )
        datasets[name] = (X, Y)
        cat_exp_ids = c_ids # Should be the same for all

    cat_exp_ids = np.array(cat_exp_ids)
    unique_cats = np.unique(cat_exp_ids)
    n_points = len(cat_exp_ids)
    
    print(f"\nTotal points: {n_points}")
    print(f"Unique catalysts in dataset: {len(unique_cats)}")
    
    def run_validation(masks, labels, title, out_filename):
        results_dict = {strat: [] for strat in strategies}
        
        for label, test_mask in zip(labels, masks):
            train_mask = ~test_mask
            print(f"Holding out {label} (n={test_mask.sum()})")
            
            for name in strategies:
                X, Y = datasets[name]
                X_train, Y_train = X[train_mask], Y[train_mask]
                X_test, Y_test = X[test_mask], Y[test_mask]
                
                num_dims = X_train.shape[-1]
                
                model = SingleTaskGP(
                    train_X=X_train,
                    train_Y=Y_train,
                    input_transform=Normalize(d=num_dims),
                    outcome_transform=Standardize(m=1),
                )
                
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                
                model.eval()
                with torch.no_grad():
                    posterior = model.posterior(X_test)
                    mu = posterior.mean.squeeze(-1)
                    var = posterior.variance.squeeze(-1)
                    
                    crps_vals = []
                    if mu.dim() == 0:
                        mu = mu.unsqueeze(0)
                        var = var.unsqueeze(0)
                        y_true = Y_test.squeeze(-1)
                        if y_true.dim() == 0:
                            y_true = y_true.unsqueeze(0)
                    else:
                        y_true = Y_test.squeeze(-1)
                        
                    for j in range(len(mu)):
                        c = gaussian_crps(mu[j], var[j].sqrt(), y_true[j])
                        crps_vals.append(c.item())
                        
                    avg_crps = np.mean(crps_vals)
                results_dict[name].append(avg_crps)
                
        df = pd.DataFrame(results_dict)
        df.insert(0, "Label", labels)
        
        averages = {s: df[s].mean() for s in strategies}
        best_strat = min(averages, key=averages.get)

        print(f"\n--- Final CRPS Comparison ({title}) ---")
        for s in strategies:
            print(f"{s:>15}: {averages[s]:.4f} {'(Winner)' if s == best_strat else ''}")

        fig, ax = plt.subplots(figsize=(12, max(8, len(labels)*0.4)))
        ax.axis('off')
        ax.axis('tight')

        col_labels = ["Held-Out"] + strategies
        table_data = []
        
        for i in range(len(df)):
            row = [str(df['Label'][i])]
            for s in strategies:
                row.append(f"{df[s][i]:.2f}")
            table_data.append(row)
            
        avg_row = ["Average"]
        for s in strategies:
            avg_row.append(f"{averages[s]:.3f}")
        table_data.append(avg_row)

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        n_rows = len(table_data)
        for c_idx, s in enumerate(strategies):
            if s == best_strat:
                cell = table[n_rows, c_idx + 1]
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('red')

        plt.title(title, pad=20, fontsize=14, weight='bold')
        
        out_path = Path(__file__).resolve().parent / out_filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved table to {out_path}")
        
    print("\n" + "="*50)
    print(" 1. LEAVE-ONE-CATALYST-OUT VALIDATION ")
    print("="*50)
    cat_masks = [cat_exp_ids == c_id for c_id in unique_cats]
    cat_labels = [f"ID {c_id}" for c_id in unique_cats]
    run_validation(cat_masks, cat_labels, "Leave-One-Catalyst-Out CRPS Scores", "leave_catalyst_out_crps_table.png")

    print("\n" + "="*50)
    print(" 2. POINT-WISE LEAVE-ONE-OUT VALIDATION ")
    print("="*50)
    pt_masks = []
    pt_labels = []
    for i in range(n_points):
        m = np.zeros(n_points, dtype=bool)
        m[i] = True
        pt_masks.append(m)
        pt_labels.append(f"Pt {i+1} (Cat {cat_exp_ids[i]})")
        
    run_validation(pt_masks, pt_labels, "Point-wise Leave-One-Out CRPS Scores", "leave_point_out_crps_table.png")

if __name__ == "__main__":
    main()
