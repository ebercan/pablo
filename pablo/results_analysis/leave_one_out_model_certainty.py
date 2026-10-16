"""
Leave-one-out cross-validation with SingleTaskGP on the 47 initial points.
Metric: Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions.

Reads ``X_initial.pt`` and ``Y_initial.pt`` from this directory and writes the
per-point CRPS table to ``leave_one_out_crps_table.png`` next to them.

Usage:
    python -m pablo.results_analysis.leave_one_out_model_certainty
"""

import math
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RQKernel, LinearKernel, Kernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from pablo.utils.metrics import crps_gaussian


class ContinuousTanimotoKernel(Kernel):
    """
    Continuous adaptation of the Tanimoto kernel.
    K(x, y) = (x \\cdot y) / (||x||^2 + ||y||^2 - x \\cdot y)
    """
    is_stationary = False
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        """Evaluate the continuous Tanimoto kernel.

        Args:
            x1: ``... x n1 x d`` inputs.
            x2: ``... x n2 x d`` inputs.
            diag: If True, return the diagonal, assuming ``x1`` and ``x2`` are
                the same points; it is all ones.
            **params: Ignored.

        Returns:
            ``... x n1`` ones if ``diag``, else the ``... x n1 x n2`` kernel
            matrix, with the denominator clamped at 1e-8.
        """
        if diag:
            return torch.ones(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        
        cross_prod = torch.matmul(x1, x2.transpose(-2, -1))
        denominator = x1_norm + x2_norm.transpose(-2, -1) - cross_prod
        
        return cross_prod / denominator.clamp_min(1e-8)


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-point CRPS of a Gaussian predictive distribution N(mu, sigma²).

    Wraps ``pablo.utils.metrics.crps_gaussian`` with the arguments in
    (mu, sigma, y) order.

    Args:
        mu: Predictive means, ``n``, ``n x 1`` or a scalar.
        sigma: Predictive standard deviations, same number of elements as ``mu``.
        y: Observed values, same number of elements as ``mu``.

    Returns:
        Per-point CRPS, flattened to shape ``n``.
    """
    return crps_gaussian(y, mu, sigma)


def main() -> None:
    """Run leave-one-out CV over five SingleTaskGP kernels and save a CRPS table.

    Each of the n points is held out in turn; the default SingleTaskGP kernel,
    Matérn-5/2, rational quadratic, linear and continuous Tanimoto models are
    fitted on the rest and scored by CRPS on the held-out point. Prints the
    per-point and average CRPS and saves the table, sorted by observed yield,
    to ``leave_one_out_crps_table.png`` in this directory.
    """
    data_dir = Path(__file__).resolve().parent
    # Refactored file names
    X = torch.load(data_dir / "X_initial.pt", weights_only=True)  # (47, 30) float64
    y = torch.load(
        data_dir / "Y_initial.pt", weights_only=True
    )  # (47, 1) float64

    n = X.shape[0]
    num_dims = X.shape[-1]
    
    # Store aggregated CRPS for table
    results_dict = {"Base": [], "Matern": [], "RQ": [], "Linear": [], "Tanimoto": []}
    y_true_list = []

    for i in range(n):
        mask = torch.arange(n) != i
        X_train = X[mask]
        y_train = y[mask]

        models = {}

        # 1. Baseline (Default SingleTaskGP setup: RBF + LogNormalPrior ARD)
        models["Base"] = SingleTaskGP(
            train_X=X_train,
            train_Y=y_train,
            input_transform=Normalize(d=num_dims),
            outcome_transform=Standardize(m=1),
        )

        # Reusable Vanilla Prior components for lengthscales to ensure stable ARD
        loc_ard = math.sqrt(2.0) + math.log(num_dims) * 0.5
        prior_ard = LogNormalPrior(loc=loc_ard, scale=math.sqrt(3.0))
        constraint_ard = GreaterThan(0.025, transform=None, initial_value=prior_ard.mode)
        
        # 2. Matern (Nu=2.5, ARD + LogNormalPrior)
        models["Matern"] = SingleTaskGP(
            train_X=X_train,
            train_Y=y_train,
            input_transform=Normalize(d=num_dims),
            outcome_transform=Standardize(m=1),
            covar_module=ScaleKernel(MaternKernel(
                nu=2.5,
                ard_num_dims=num_dims,
                lengthscale_prior=prior_ard,
                lengthscale_constraint=constraint_ard
            )),
        )

        # 3. Rational Quadratic (ARD + LogNormalPrior)
        models["RQ"] = SingleTaskGP(
            train_X=X_train,
            train_Y=y_train,
            input_transform=Normalize(d=num_dims),
            outcome_transform=Standardize(m=1),
            covar_module=ScaleKernel(RQKernel(
                ard_num_dims=num_dims,
                lengthscale_prior=prior_ard,
                lengthscale_constraint=constraint_ard
            )),
        )

        # 4. Linear Kernel
        models["Linear"] = SingleTaskGP(
            train_X=X_train,
            train_Y=y_train,
            input_transform=Normalize(d=num_dims),
            outcome_transform=Standardize(m=1),
            covar_module=ScaleKernel(LinearKernel(num_dimensions=num_dims)),
        )

        # 5. Tanimoto Kernel
        models["Tanimoto"] = SingleTaskGP(
            train_X=X_train,
            train_Y=y_train,
            input_transform=Normalize(d=num_dims),
            outcome_transform=Standardize(m=1),
            covar_module=ScaleKernel(ContinuousTanimotoKernel()),
        )

        crps_dict = {}
        X_test = X[i].unsqueeze(0)
        
        for name, model in models.items():
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            model.eval()
            posterior = model.posterior(X_test)
            mu = posterior.mean.squeeze()
            var = posterior.variance.squeeze()
            
            crps_val = gaussian_crps(mu, var.sqrt(), y[i].squeeze())
            crps_dict[name] = crps_val.item()
            results_dict[name].append(crps_val.item())

        y_true_list.append(y[i].item())

        print(
            f"Idx {i:2d} | true: {y[i].item():5.1f} | "
            f"Base: {crps_dict['Base']:.3f} | "
            f"Matern: {crps_dict['Matern']:.3f} | "
            f"RQ: {crps_dict['RQ']:.3f} | "
            f"Linear: {crps_dict['Linear']:.3f} | "
            f"Tanimoto: {crps_dict['Tanimoto']:.3f}"
        )

    # Convert results to DataFrame for easier sorting and plotting
    df = pd.DataFrame(results_dict)
    df['y_true'] = y_true_list
    
    # Sort by true Y values in increasing order
    df = df.sort_values(by='y_true').reset_index(drop=True)

    kernel_names = ["Base", "Matern", "RQ", "Linear", "Tanimoto"]
    averages = {k: df[k].mean() for k in kernel_names}
    best_kernel = min(averages, key=averages.get)

    print("\n--- Final CRPS Comparison (Average) ---")
    for name in kernel_names:
        print(f"{name:>15}: {averages[name]:.4f} {'(Winner)' if name == best_kernel else ''}")

    # Generate Matplotlib Table (Transposed)
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.axis('off')
    ax.axis('tight')

    col_labels = ["Point", "Obs Y"] + kernel_names
    table_data = []
    
    # Data Rows
    for i in range(n):
        row = [f"Pt {i+1}", f"{df['y_true'][i]:.1f}"]
        for k in kernel_names:
            row.append(f"{df[k][i]:.2f}")
        table_data.append(row)
        
    # Average Row
    avg_row = ["Average", "-"]
    for k in kernel_names:
        avg_row.append(f"{averages[k]:.3f}")
    table_data.append(avg_row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style the table: Make the winner's average cell bold
    for c_idx, k in enumerate(kernel_names):
        if k == best_kernel:
            # Table has a header row at index 0. 
            # Data rows are 1..n. Average row is index n + 1.
            # Columns: Point=0, Obs Y=1, Kernels start at 2
            cell = table[n + 1, c_idx + 2]
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('red')

    plt.title("LeaveOneOut Experiment CRPS Scores", pad=20, fontsize=14, weight='bold')
    
    out_path = data_dir / "leave_one_out_crps_table.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved table to {out_path}")

if __name__ == "__main__":
    main()
