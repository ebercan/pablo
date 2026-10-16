"""Pearson correlation between Conversion and Selectivity, raw and residual.

Fits independent SingleTaskGP models to Conversion and Selectivity from the
Experimental Design sheet, then computes the Pearson r between the two
posterior-mean residual vectors.

Usage::

    python -m pablo.results_analysis.conversion_selectivity_pearson
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from pablo.main import (
    _load_solvent_base_name_maps,
    _resolve_component_id,
    load_reference_data,
)


EXCEL_FILE = Path(__file__).resolve().parents[1] / "nhc_doe.xlsx"
SHEET_NAME = "Experimental Design"

DTYPE = torch.float64

DEDUP_KEY_COLS = [
    "Catalyst",
    "Solvent",
    "Base",
    "Temperature",
    "Formaldehyde Ratio in Eq.",
    "Catalyst Loading in Molar Ratio",
    "Time",
]


def main() -> None:
    """Compute raw and GP-residual Pearson r between Conversion and Selectivity."""
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

    if "Conversion" not in df.columns:
        raise ValueError("Column 'Conversion' not found in Experimental Design")

    conv = pd.to_numeric(df["Conversion"], errors="coerce")
    sel = (
        pd.to_numeric(df["Selectivity"], errors="coerce")
        if "Selectivity" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    if sel.isna().all():
        if "Yield" not in df.columns:
            raise ValueError(
                "Column 'Selectivity' missing/unusable and no 'Yield' column to rebuild it"
            )
        yld = pd.to_numeric(df["Yield"], errors="coerce")
        sel = np.where(conv == 0, 0.0, (yld / conv) * 100.0)

    df = df.assign(Conversion=conv, Selectivity=sel)
    df = df.dropna(subset=DEDUP_KEY_COLS + ["Conversion", "Selectivity"]).reset_index(
        drop=True
    )

    grouped = df.groupby(DEDUP_KEY_COLS, dropna=False)
    averaged = grouped[["Conversion", "Selectivity"]].mean().reset_index()

    conv_vals = averaged["Conversion"].to_numpy(dtype=float)
    sel_vals = averaged["Selectivity"].to_numpy(dtype=float)

    r = float(np.corrcoef(conv_vals, sel_vals)[0, 1])
    print(f"pearson_r: {r:.6f}")

    cat_map, sol_map, bas_map, _, _, _, _, _ = load_reference_data(EXCEL_FILE)
    _, _, sol_name_to_id, bas_name_to_id = _load_solvent_base_name_maps(EXCEL_FILE)

    no_base_fallback_id = 0 if 0 in bas_map else sorted(bas_map.keys())[0]
    base_text_fallback_ids = {
        "no": no_base_fallback_id,
        "none": no_base_fallback_id,
        "n/a": no_base_fallback_id,
        "na": no_base_fallback_id,
        "-": no_base_fallback_id,
    }

    solvent_ids = [
        _resolve_component_id(v, sol_map, sol_name_to_id, "Solvent")
        for v in averaged["Solvent"].tolist()
    ]
    base_ids = [
        _resolve_component_id(
            v,
            bas_map,
            bas_name_to_id,
            "Base",
            fallback_text_to_id=base_text_fallback_ids,
        )
        for v in averaged["Base"].tolist()
    ]

    x_cat = torch.stack([cat_map[v] for v in averaged["Catalyst"]])
    x_sol = torch.stack([sol_map[v] for v in solvent_ids])
    x_bas = torch.stack([bas_map[v] for v in base_ids])

    temperature = torch.tensor(
        pd.to_numeric(averaged["Temperature"]).values, dtype=DTYPE
    )
    form_ratio = torch.tensor(
        pd.to_numeric(averaged["Formaldehyde Ratio in Eq."]).values, dtype=DTYPE
    )
    cat_loading = torch.tensor(
        pd.to_numeric(averaged["Catalyst Loading in Molar Ratio"]).values,
        dtype=DTYPE,
    )
    time = torch.tensor(pd.to_numeric(averaged["Time"]).values, dtype=DTYPE)

    x_cont = torch.stack([temperature, form_ratio, cat_loading, time], dim=1)
    x_train = torch.cat([x_cont, x_cat, x_sol, x_bas], dim=1).to(DTYPE)

    y_conv = torch.tensor(conv_vals, dtype=DTYPE).unsqueeze(-1)
    y_sel = torch.tensor(sel_vals, dtype=DTYPE).unsqueeze(-1)

    def fit_model(train_y: torch.Tensor) -> SingleTaskGP:
        model = SingleTaskGP(
            train_X=x_train,
            train_Y=train_y,
            input_transform=Normalize(d=x_train.shape[1]),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    model_conv = fit_model(y_conv)
    model_sel = fit_model(y_sel)

    with torch.no_grad():
        conv_pred = model_conv.posterior(x_train).mean
        sel_pred = model_sel.posterior(x_train).mean

    conv_residuals = (conv_pred - y_conv).squeeze(-1).cpu().numpy()
    sel_residuals = (sel_pred - y_sel).squeeze(-1).cpu().numpy()

    res_r = float(np.corrcoef(conv_residuals, sel_residuals)[0, 1])
    print(f"pearson_r_residuals: {res_r:.6f}")


if __name__ == "__main__":
    main()
