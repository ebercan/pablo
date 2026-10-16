"""
Candidate batch generation for NHC optimization using Bayesian Optimization.

This module generates a q-batch of candidate experiments by training a GP model
on initial experimental data and optimizing an acquisition function over the
mixed continuous-discrete search space. Discrete choices are encoded via
raw, un-reduced molecular featurizations.

The candidate structure is:
    [Temperature, Formaldehyde_Ratio, Catalyst_Loading, Time, discrete_dims...]

where the four continuous columns are min-max normalized to [0, 1] and
discrete_dims are the raw featurization blocks for (catalyst, solvent, base).

Usage:
    python -m pablo.main
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    LogNoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from openpyxl import load_workbook
from rdkit import Chem
from torch import Tensor

from pablo.optimize import (
    optimize_acqf_continuous,
    optimize_acqf_mixed_GA,
    optimize_acqf_mixed_sequential_GA,
)
from pablo.utils.runtime_estimator import calibrate, estimate_runtime, format_eta

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE_PARTITION = (2, 2, 2)
GA_SETTINGS = (20000, 500)  # (n_evals_ga, pop_size)
GA_OPTIMIZATION_REPEATS = 1
OPTIMIZE_SEQUENTIAL = False  # False: batch GA + final refine | True: sequential GA
LBFGS_SETTINGS = (20, 1024)  # (num_restarts, raw_samples) for continuous refinement
MODEL_SAVE_PATH = Path(__file__).parent / "trained_main_model.pt"
EXTEND_SCOPE = False

# Featurization selectors. Each must name a subdirectory under
# FEATURIZATIONS_ROOT containing a descriptors.parquet.
# Canonical options (mirrors pablo/featurization_and_kernel_scan/cross_validation/generate_inputs.py):
#   CAT_FEATURES     : "map4", "ecfp4", "clamp", "mordred", "mol2vec",
#                      "chembert2", "molformer_xl_precat", "molformer_xl_carbene",
#                      "unimol", "grover", "soap", "interpretable"
#   SOLVENT_FEATURES : "map", "KT", "Hansen", "Gutmann", "OHE"
#   BASE_FEATURES    : "map", "thermo", "steric", "OHE"
CAT_FEATURES = "grover"
SOLVENT_FEATURES = "map"
BASE_FEATURES = "map"

FEATURIZATIONS_ROOT = (
        Path(__file__).parent / "featurization_and_kernel_scan" / "data" / "featurizations"
)

CAT_FEATURES_REQUIRING_FORM_ONEHOT = frozenset({"molformer_xl_carbene", "unimol"})

DEFAULT_EXCEL_FILE = Path(__file__).parent / "nhc_doe.xlsx"


class ContinuousBounds:
    """Container for continuous variable bounds.

    Stores bounds for Temperature, Formaldehyde Ratio, Catalyst Loading, and Time.
    All bounds are stored as (lower, upper) tuples.

    Args:
        temperature: Tuple of (min, max) for temperature in °C.
        formaldehyde_ratio: Tuple of (min, max) for formaldehyde equivalents.
        catalyst_loading: Tuple of (min, max) for catalyst molar ratio.
        time: Tuple of (min, max) for reaction time in hours.
    """

    def __init__(
            self,
            temperature: Tuple[float, float] = (25.0, 100.0),
            formaldehyde_ratio: Tuple[float, float] = (1.0, 3.0),
            catalyst_loading: Tuple[float, float] = (0.5, 10.0),
            time: Tuple[float, float] = (16.0, 20.0),
    ):
        self.temperature = temperature
        self.formaldehyde_ratio = formaldehyde_ratio
        self.catalyst_loading = catalyst_loading
        self.time = time

    def to_tensor(self) -> Tensor:
        """Convert bounds to a (2, 4) tensor for continuous dimensions.

        Returns:
            Tensor of shape (2, 4) with lower bounds in row 0 and upper in row 1.
        """
        lower = [
            self.temperature[0],
            self.formaldehyde_ratio[0],
            self.catalyst_loading[0],
            self.time[0],
        ]
        upper = [
            self.temperature[1],
            self.formaldehyde_ratio[1],
            self.catalyst_loading[1],
            self.time[1],
        ]
        return torch.tensor([lower, upper], dtype=DTYPE)


_FORM_ONEHOT_COLUMNS = (
    "form_salt",
    "form_free_nhc",
    "form_co2_adduct",
    "form_bicarbonate",
)


def _resolve_featurization_dir(component: str, featurization_name: str) -> Path:
    """Resolve the on-disk directory for a (component, featurization) pair.
        Returns:
            Path: Return value of _resolve_featurization_dir.

    Returns:
        Path: Return value of _resolve_featurization_dir.
    """
    if component == "cat":
        return FEATURIZATIONS_ROOT / featurization_name
    if component == "solvent":
        return FEATURIZATIONS_ROOT / f"solvent_{featurization_name}"
    if component == "base":
        return FEATURIZATIONS_ROOT / f"base_{featurization_name}"
    raise ValueError(
        f"Unknown component {component!r}; expected 'cat', 'solvent', or 'base'."
    )


def load_featurization_from_parquet(
        component: str,
        featurization_name: str,
        ids: List[int],
        append_form_onehot: bool = False,
) -> Tensor:
    """Load a precomputed featurization and min-max-normalize to [0, 1].

    Reads ``descriptors.parquet`` from the per-featurization directory under
    ``FEATURIZATIONS_ROOT``. The parquet is indexed by ``catalyst_id`` /
    ``solvent_id`` / ``base_id``.

    Args:
        component: One of ``"cat"``, ``"solvent"``, ``"base"``.
        featurization_name: Featurization subdirectory name (without the
            ``solvent_`` / ``base_`` prefix). E.g. ``"map4"``, ``"physical"``.
        ids: Row order to reindex by. Output row ``i`` corresponds to ``ids[i]``.
        append_form_onehot: When ``True`` (only meaningful for ``component='cat'``),
            concatenate the 4-column form one-hot from
            ``interpretable/descriptors.parquet`` after the catalyst block.

    Returns:
        A ``(len(ids), d)`` Tensor of [0, 1]-normalized features.

    Raises:
        FileNotFoundError: If ``descriptors.parquet`` is missing from the
            featurization directory, or if ``append_form_onehot`` is True and
            ``interpretable/descriptors.parquet`` is missing.
        ValueError: If the descriptors or the form one-hot lack rows for any
            requested ID.
    """
    cell_dir = _resolve_featurization_dir(component, featurization_name)
    parquet_path = cell_dir / "descriptors.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No descriptors.parquet under {cell_dir}")

    df = pd.read_parquet(parquet_path).select_dtypes(include="number")
    block = df.reindex(ids)
    missing_mask = block.isna().any(axis=1)
    if missing_mask.any():
        missing_ids = [ids[i] for i, m in enumerate(missing_mask.tolist()) if m]
        raise ValueError(
            f"Featurization at {parquet_path} is missing rows for {component} "
            f"IDs: {missing_ids}"
        )
    values = block.to_numpy(dtype=np.float64)

    if append_form_onehot:
        form_path = FEATURIZATIONS_ROOT / "interpretable" / "descriptors.parquet"
        if not form_path.exists():
            raise FileNotFoundError(
                f"append_form_onehot=True requires {form_path}, which was not found."
            )
        form_df = pd.read_parquet(form_path)[list(_FORM_ONEHOT_COLUMNS)]
        form_block = form_df.reindex(ids)
        form_missing = form_block.isna().any(axis=1)
        if form_missing.any():
            missing_ids = [ids[i] for i, m in enumerate(form_missing.tolist()) if m]
            raise ValueError(
                f"Form one-hot at {form_path} is missing rows for catalyst "
                f"IDs: {missing_ids}"
            )
        values = np.concatenate([values, form_block.to_numpy(dtype=np.float64)], axis=1)

    values_tensor = torch.tensor(values, dtype=DTYPE)
    lower = values_tensor.min(dim=0).values
    upper = values_tensor.max(dim=0).values
    eps = 1e-8
    range_vals = upper - lower
    range_vals = torch.where(range_vals < eps, torch.ones_like(range_vals), range_vals)
    return (values_tensor - lower) / range_vals


_BICARBONATE_SMARTS = Chem.MolFromSmarts("[OH1]C(=O)[O-]")


def requires_external_base(smiles: str) -> bool:
    """Classify whether an NHC catalyst SMILES requires an external base.

    Returns True (needs base) only for true ionic salts — a cationic NHC ring
    paired with a discrete counterion that is not bicarbonate.

    The three classes that do NOT require base all return False:
    - CO₂ adducts: zwitterionic single molecule with covalent C2-carboxylate.
    - Bicarbonate salts: counterion is HCO₃⁻ (self-activating via deprotonation).
    - Free NHCs: explicit divalent carbene carbon, single connected molecule.

    Args:
        smiles: SMILES string for the catalyst.

    Returns:
        True if an external base is required, False otherwise.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    frags = Chem.GetMolFrags(mol, asMols=True)

    if len(frags) == 1:
        # Single connected molecule: free NHC, CO₂ adduct, or zwitterionic NHC.
        return False

    # Multi-fragment: bicarbonate counterion self-activates — no external base.
    if any(frag.HasSubstructMatch(_BICARBONATE_SMARTS) for frag in frags):
        return False

    # Remaining multi-fragment species are true ionic salts.
    return True


def load_reference_data(
        excel_file: Path,
        cat_features: str = CAT_FEATURES,
        solvent_features: str = SOLVENT_FEATURES,
        base_features: str = BASE_FEATURES,
) -> Tuple[
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, bool],
    Dict[int, str],
    List[int],
    List[int],
    List[int],
]:
    """Load and encode molecular references from Excel.

    Reads catalyst, solvent, and base reference sheets, loads the raw
    featurization blocks from disk, and returns embedding dictionaries keyed by ID.

    Args:
        excel_file: Path to the NHC dataset Excel workbook.
        cat_features: Catalyst featurization directory name.
        solvent_features: Solvent featurization, resolving to ``solvent_<name>/``.
        base_features: Base featurization, resolving to ``base_<name>/``.

    Returns:
        A tuple of eight elements:
            - cat_map: Dict mapping catalyst ID to embedding tensor.
            - sol_map: Dict mapping solvent ID to embedding tensor.
            - bas_map: Dict mapping base ID to embedding tensor.
            - cat_is_salt: Dict mapping catalyst ID to whether an external base
                is required (True = ionic salt; False = CO₂ adduct, bicarbonate
                salt, or free NHC).
            - cat_bld: Dict mapping catalyst ID to BLD number, or '-'.
            - cat_ids: List of catalyst IDs.
            - sol_ids: List of solvent IDs.
            - bas_ids: List of base IDs.
    """
    cat_ref = pd.read_excel(excel_file, sheet_name="Catalyst References Achiral")
    sol_ref = pd.read_excel(excel_file, sheet_name="Solvent References")
    bas_ref = pd.read_excel(excel_file, sheet_name="Base References")

    cat_ref = cat_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    sol_ref = sol_ref.dropna(subset=["smiles"]).reset_index(drop=True)
    bas_ref = bas_ref.dropna(subset=["smiles"]).reset_index(drop=True)

    cat_ids = cat_ref["ID"].tolist()
    sol_ids = sol_ref["ID"].tolist()
    bas_ids = bas_ref["ID"].tolist()
    cat_smiles = cat_ref["smiles"].astype(str).tolist()

    cat_emb = load_featurization_from_parquet(
        "cat",
        cat_features,
        cat_ids,
        append_form_onehot=cat_features in CAT_FEATURES_REQUIRING_FORM_ONEHOT,
    )
    sol_emb = load_featurization_from_parquet("solvent", solvent_features, sol_ids)
    bas_emb = load_featurization_from_parquet("base", base_features, bas_ids)

    cat_map = {cat_ids[i]: cat_emb[i] for i in range(len(cat_ids))}
    sol_map = {sol_ids[i]: sol_emb[i] for i in range(len(sol_ids))}
    bas_map = {bas_ids[i]: bas_emb[i] for i in range(len(bas_ids))}
    cat_is_salt = {
        cat_ids[i]: requires_external_base(cat_smiles[i]) for i in range(len(cat_ids))
    }
    cat_bld = _extract_cat_bld_from_column_l(cat_ref, cat_ids)

    return cat_map, sol_map, bas_map, cat_is_salt, cat_bld, cat_ids, sol_ids, bas_ids


def _coerce_reference_id(raw_value, column_name: str, sheet_name: str) -> int:
    """Convert a reference-sheet identifier cell to an integer ID.
        Returns:
            int: Return value of _coerce_reference_id.

    Returns:
        int: Return value of _coerce_reference_id.
    """
    if pd.isna(raw_value):
        raise ValueError(
            f"Missing value in '{column_name}' column of '{sheet_name}' sheet."
        )

    if isinstance(raw_value, (int, np.integer)):
        return int(raw_value)

    if isinstance(raw_value, (float, np.floating)):
        if float(raw_value).is_integer():
            return int(raw_value)
        raise ValueError(f"Non-integer ID value {raw_value!r} in '{sheet_name}' sheet.")

    text_value = str(raw_value).strip()
    if text_value == "":
        raise ValueError(
            f"Empty string in '{column_name}' column of '{sheet_name}' sheet."
        )
    if text_value.lstrip("-").isdigit():
        return int(text_value)

    raise ValueError(f"Invalid ID value {raw_value!r} in '{sheet_name}' sheet.")


def _load_solvent_base_name_maps(
        excel_file: Path,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[str, int], Dict[str, int]]:
    """Load solvent names and base labels plus reverse lookup maps.

    Base labels are sourced from the ``Comment`` column when available,
    falling back to ``Name`` (and then ID string) if needed.

    Args:
        excel_file: Path to the NHC dataset Excel workbook.

    Returns:
        A tuple ``(sol_id_to_name, bas_id_to_name, sol_name_to_id,
        bas_name_to_id)``: the first two map IDs to display names, the last
        two map casefolded names back to IDs.

    Raises:
        ValueError: If the ``Solvent References`` sheet lacks ``ID`` or
            ``Name`` columns, or the ``Base References`` sheet lacks ``ID`` or
            ``Comment`` columns.
    """
    sol_ref = pd.read_excel(excel_file, sheet_name="Solvent References")
    bas_ref = pd.read_excel(excel_file, sheet_name="Base References")

    if "ID" not in sol_ref.columns or "Name" not in sol_ref.columns:
        raise ValueError(
            "Sheet 'Solvent References' must contain 'ID' and 'Name' columns."
        )
    if "ID" not in bas_ref.columns or "Comment" not in bas_ref.columns:
        raise ValueError(
            "Sheet 'Base References' must contain 'ID' and 'Comment' columns."
        )

    sol_id_to_name: Dict[int, str] = {}
    bas_id_to_name: Dict[int, str] = {}
    sol_name_to_id: Dict[str, int] = {}
    bas_name_to_id: Dict[str, int] = {}

    for raw_id, raw_name in zip(sol_ref["ID"], sol_ref["Name"]):
        if pd.isna(raw_id):
            continue
        solvent_id = _coerce_reference_id(raw_id, "ID", "Solvent References")
        solvent_name = (
            str(raw_name).strip() if not pd.isna(raw_name) else str(solvent_id)
        )
        if solvent_name == "":
            solvent_name = str(solvent_id)
        sol_id_to_name[solvent_id] = solvent_name
        sol_name_to_id[solvent_name.casefold()] = solvent_id

    base_names = bas_ref["Name"] if "Name" in bas_ref.columns else [None] * len(bas_ref)
    for raw_id, raw_name, raw_comment in zip(
            bas_ref["ID"],
            base_names,
            bas_ref["Comment"],
    ):
        if pd.isna(raw_id):
            continue
        base_id = _coerce_reference_id(raw_id, "ID", "Base References")
        base_name = str(raw_name).strip() if not pd.isna(raw_name) else ""
        base_comment = str(raw_comment).strip() if not pd.isna(raw_comment) else ""

        if base_comment != "":
            base_label = base_comment
        elif base_name != "":
            base_label = base_name
        else:
            base_label = str(base_id)

        bas_id_to_name[base_id] = base_label
        bas_name_to_id[base_label.casefold()] = base_id
        if base_name != "":
            bas_name_to_id[base_name.casefold()] = base_id
        if base_comment != "":
            bas_name_to_id[base_comment.casefold()] = base_id

    return sol_id_to_name, bas_id_to_name, sol_name_to_id, bas_name_to_id


_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")
_NO_BASE_LABELS = frozenset({"no", "none", "n/a", "na", "-"})


def _match_punctuation_insensitive(
        name_key: str, name_to_id: Dict[str, int], column_name: str
) -> Optional[int]:
    """Match a hand-typed name to a reference name ignoring punctuation and spaces.

    ``"1.4-dioxane"`` and ``"1,4-Dioxane"`` both reduce to ``"14dioxane"``.

    Args:
        name_key: Casefolded cell text.
        name_to_id: Casefolded reference name to ID.
        column_name: Column name, for the error message.

    Returns:
        The matching ID, or ``None`` when no reference name reduces to the same key.

    Raises:
        ValueError: If reference names with different IDs reduce to that key.
    """
    stripped = _NON_ALPHANUMERIC.sub("", name_key)
    matches = {i for name, i in name_to_id.items() if _NON_ALPHANUMERIC.sub("", name) == stripped}
    if len(matches) > 1:
        raise ValueError(
            f"Value {name_key!r} in column '{column_name}' matches several IDs {sorted(matches)} "
            "once punctuation is ignored."
        )
    return matches.pop() if matches else None


def _resolve_component_id(
        raw_value,
        id_to_embedding: Dict[int, Tensor],
        name_to_id: Dict[str, int],
        column_name: str,
        fallback_text_to_id: Optional[Dict[str, int]] = None,
) -> int:
    """Resolve an Experimental Design component value to its integer ID.

    ``fallback_text_to_id`` can be used for known textual sentinels
    (for example, "no" in the Base column).

        Returns:
            int: Return value of _resolve_component_id.

    Returns:
        int: Return value of _resolve_component_id.
    """
    if pd.isna(raw_value):
        raise ValueError(
            f"Experimental Design column '{column_name}' contains an empty value."
        )

    resolved_id: Optional[int] = None

    if isinstance(raw_value, (int, np.integer)):
        resolved_id = int(raw_value)
    elif isinstance(raw_value, (float, np.floating)) and float(raw_value).is_integer():
        resolved_id = int(raw_value)
    else:
        text_value = str(raw_value).strip()
        if text_value == "":
            raise ValueError(
                f"Experimental Design column '{column_name}' contains an empty string."
            )

        if text_value.lstrip("-").isdigit():
            resolved_id = int(text_value)
        else:
            name_key = text_value.casefold()
            if name_key in name_to_id:
                resolved_id = name_to_id[name_key]
            elif fallback_text_to_id is not None and name_key in fallback_text_to_id:
                resolved_id = fallback_text_to_id[name_key]
            else:
                resolved_id = _match_punctuation_insensitive(name_key, name_to_id, column_name)

    if resolved_id is None:
        raise ValueError(
            f"Could not resolve value {raw_value!r} in column '{column_name}' to an ID."
        )

    if resolved_id not in id_to_embedding:
        raise ValueError(
            f"Value {raw_value!r} in column '{column_name}' resolved to ID {resolved_id}, "
            "but this ID has no embedding."
        )

    return resolved_id


def _format_bld_value(raw_value) -> str:
    """Normalize a BLD cell value into a printable string or '-'.
        Returns:
            str: Return value of _format_bld_value.

    Returns:
        str: Return value of _format_bld_value.
    """
    if pd.isna(raw_value):
        return "-"

    if isinstance(raw_value, float) and raw_value.is_integer():
        return str(int(raw_value))

    text_value = str(raw_value).strip()
    if text_value == "":
        return "-"
    if text_value.lower() == "nan":
        return "-"

    if text_value.endswith(".0") and text_value[:-2].isdigit():
        return text_value[:-2]

    return text_value


def _extract_cat_bld_from_column_l(
        cat_ref: pd.DataFrame,
        cat_ids: List[int],
) -> Dict[int, str]:
    """Extract catalyst BLD values from column L (12th column) of cat_ref.
        Returns:
            Dict[int, str]: Return value of _extract_cat_bld_from_column_l.

    Returns:
        Dict[int, str]: Return value of _extract_cat_bld_from_column_l.
    """
    if cat_ref.shape[1] < 12:
        return {c_id: "-" for c_id in cat_ids}

    bld_values = cat_ref.iloc[:, 11].tolist()
    return {cat_ids[i]: _format_bld_value(bld_values[i]) for i in range(len(cat_ids))}


def load_available_agents(
        excel_file: Path,
        batch_column: str = "Next Batch",
) -> Tuple[List[int], List[int], List[int]]:
    """Load currently available reaction agents from Excel.

    Reads the availability sheets and filters to agents marked 'y' for the
    specified batch column.

    Args:
        excel_file: Path to the NHC dataset Excel workbook.
        batch_column: Column name indicating availability ('Next Batch',
            'Subsequent Batch', or 'Following Batch').

    Returns:
        A tuple of three lists:
            - available_catalysts: List of available catalyst IDs.
            - available_solvents: List of available solvent IDs.
            - available_bases: List of available base IDs.
    """
    cat_avail = pd.read_excel(excel_file, sheet_name="Available Catalysts")
    sol_avail = pd.read_excel(excel_file, sheet_name="Available Solvents")
    bas_avail = pd.read_excel(excel_file, sheet_name="Available Bases")

    available_catalysts = cat_avail.loc[
        cat_avail[batch_column] == "y", "Catalyst"
    ].tolist()
    available_solvents = sol_avail.loc[
        sol_avail[batch_column] == "y", "Solvent"
    ].tolist()
    available_bases = bas_avail.loc[bas_avail[batch_column] == "y", "Base"].tolist()

    return available_catalysts, available_solvents, available_bases


def load_unordered_catalysts(
        excel_file: Path,
        valid_cat_ids: List[int],
) -> List[int]:
    """Load catalysts that are not available and not already ordered.

    Unordered catalysts are those that either:
    1. Are marked 'n' in all three availability columns, OR
    2. Are not listed in the availability sheet at all

    Args:
        excel_file: Path to the NHC dataset Excel workbook.
        valid_cat_ids: List of catalyst IDs that have valid embeddings.

    Returns:
        List of catalyst IDs that are completely unordered and have embeddings.
    """
    cat_avail = pd.read_excel(excel_file, sheet_name="Available Catalysts")

    explicitly_unordered_mask = (
            (cat_avail["Next Batch"] == "n")
            & (cat_avail["Subsequent Batch"] == "n")
            & (cat_avail["Following Batch"] == "n")
    )
    explicitly_unordered = set(
        cat_avail.loc[explicitly_unordered_mask, "Catalyst"].tolist()
    )

    listed_ids = set(cat_avail["Catalyst"].tolist())
    valid_set = set(valid_cat_ids)

    not_listed = valid_set - listed_ids

    all_unordered = explicitly_unordered | not_listed

    return sorted([c_id for c_id in all_unordered if c_id in valid_set])


def build_unordered_discrete_space(
        cat_map: Dict[int, Tensor],
        sol_map: Dict[int, Tensor],
        bas_map: Dict[int, Tensor],
        cat_is_salt: Dict[int, bool],
        unordered_catalysts: List[int],
        all_solvent_ids: List[int],
        all_base_ids: List[int],
) -> Tuple[Tensor, Tensor]:
    """Build the unordered discrete search space for scope expansion.

    Constructs the Cartesian product of unordered catalysts with ALL solvents
    and bases. The base embedding slice is set to zero for catalysts whose
    ``cat_is_salt`` flag is False (free NHCs, CO₂ adducts, bicarbonate salts),
    and their base duplicates are collapsed to one row per
    (catalyst, solvent).

    Args:
        cat_map: Dict mapping catalyst ID to embedding tensor.
        sol_map: Dict mapping solvent ID to embedding tensor.
        bas_map: Dict mapping base ID to embedding tensor.
        cat_is_salt: Dict mapping catalyst ID to salt flag.
        unordered_catalysts: List of unordered catalyst IDs.
        all_solvent_ids: List of all solvent IDs.
        all_base_ids: List of all base IDs.

    Returns:
        A tuple of two tensors:
            - X_unordered: (n_unique, d_discrete) tensor of discrete embeddings.
            - combo_ids: (n_unique, 3) tensor of (cat_id, sol_id, bas_id) tuples.
    """
    if len(unordered_catalysts) == 0:
        sample_cat = list(cat_map.values())[0]
        sample_sol = list(sol_map.values())[0]
        sample_bas = list(bas_map.values())[0]
        d_discrete = sample_cat.shape[0] + sample_sol.shape[0] + sample_bas.shape[0]
        return (
            torch.empty(0, d_discrete, dtype=DTYPE),
            torch.empty(0, 3, dtype=torch.long),
        )

    sample_bas = bas_map[all_base_ids[0]]
    d_base = sample_bas.shape[0]
    base_zeros = torch.zeros(d_base, dtype=DTYPE)

    seen_combo_keys = set()
    unique_rows: List[Tensor] = []
    unique_combo_ids: List[Tensor] = []

    for c_id in unordered_catalysts:
        for s_id in all_solvent_ids:
            for b_id in all_base_ids:
                is_salt = cat_is_salt[c_id]
                dedupe_key = (c_id, s_id, b_id) if is_salt else (c_id, s_id, -1)
                if dedupe_key in seen_combo_keys:
                    continue

                seen_combo_keys.add(dedupe_key)
                row = torch.cat([cat_map[c_id], sol_map[s_id], bas_map[b_id]])
                if not is_salt:
                    row[-d_base:] = base_zeros

                unique_rows.append(row)
                unique_combo_ids.append(
                    torch.tensor([c_id, s_id, b_id], dtype=torch.long)
                )

    X_unordered = torch.stack(unique_rows, dim=0)
    combo_ids = torch.stack(unique_combo_ids, dim=0)
    return X_unordered, combo_ids


def build_discrete_space(
        cat_map: Dict[int, Tensor],
        sol_map: Dict[int, Tensor],
        bas_map: Dict[int, Tensor],
        cat_is_salt: Dict[int, bool],
        available_catalysts: List[int],
        available_solvents: List[int],
        available_bases: List[int],
) -> Tuple[Tensor, Tensor]:
    """Build the discrete search space from available agents.

    Constructs the Cartesian product of available catalysts, solvents, and bases.
    The base embedding slice is set to zero for catalysts whose
    ``cat_is_salt`` flag is False (free NHCs, CO₂ adducts, bicarbonate salts),
    and their base duplicates are collapsed to one row per
    (catalyst, solvent).

    Args:
        cat_map: Dict mapping catalyst ID to embedding tensor.
        sol_map: Dict mapping solvent ID to embedding tensor.
        bas_map: Dict mapping base ID to embedding tensor.
        cat_is_salt: Dict mapping catalyst ID to salt flag.
        available_catalysts: List of available catalyst IDs.
        available_solvents: List of available solvent IDs.
        available_bases: List of available base IDs.

    Returns:
        A tuple of two tensors:
            - X_discrete: (n_unique, d_discrete) tensor of discrete embeddings.
            - combo_ids: (n_unique, 3) tensor of (cat_id, sol_id, bas_id) tuples.
    """
    sample_bas = bas_map[available_bases[0]]
    d_base = sample_bas.shape[0]
    base_zeros = torch.zeros(d_base, dtype=DTYPE)

    seen_combo_keys = set()
    unique_rows: List[Tensor] = []
    unique_combo_ids: List[Tensor] = []

    for c_id in available_catalysts:
        for s_id in available_solvents:
            for b_id in available_bases:
                is_salt = cat_is_salt[c_id]
                dedupe_key = (c_id, s_id, b_id) if is_salt else (c_id, s_id, -1)
                if dedupe_key in seen_combo_keys:
                    continue

                seen_combo_keys.add(dedupe_key)
                row = torch.cat([cat_map[c_id], sol_map[s_id], bas_map[b_id]])
                if not is_salt:
                    row[-d_base:] = base_zeros

                unique_rows.append(row)
                unique_combo_ids.append(
                    torch.tensor([c_id, s_id, b_id], dtype=torch.long)
                )

    X_discrete = torch.stack(unique_rows, dim=0)
    combo_ids = torch.stack(unique_combo_ids, dim=0)
    return X_discrete, combo_ids


def load_training_data(
        excel_file: Path,
        cat_map: Dict[int, Tensor],
        sol_map: Dict[int, Tensor],
        bas_map: Dict[int, Tensor],
        continuous_bounds: ContinuousBounds,
        cat_is_salt: Dict[int, bool],
) -> Tuple[Tensor, Tensor]:
    """Load and encode training data from the Experimental Design sheet.

    Reads experimental results, encodes molecular components via their raw
    featurization blocks, and normalizes continuous features. Rows whose
    catalyst needs no external base carry an all-zero base slice, the encoding
    ``build_discrete_space`` and the optimizers give such candidates.

    Args:
        excel_file: Path to the NHC dataset Excel workbook.
        cat_map: Dict mapping catalyst ID to embedding tensor.
        sol_map: Dict mapping solvent ID to embedding tensor.
        bas_map: Dict mapping base ID to embedding tensor.
        continuous_bounds: Bounds for continuous variables.
        cat_is_salt: Dict mapping catalyst ID to whether it needs an external base.

    Returns:
        A tuple of two tensors:
            - X_train: (n, d) tensor of input features.
            - Y_train: (n, 1) tensor of target values (Yield).

    Raises:
        ValueError: If a row with a base-requiring catalyst lists no base.
    """
    exp_df = pd.read_excel(excel_file, sheet_name="Experimental Design")
    exp_df = exp_df.dropna(
        subset=[
            "Yield",
            "Temperature",
            "Formaldehyde Ratio in Eq.",
            "Catalyst Loading in Molar Ratio",
            "Time",
        ]
    ).reset_index(drop=True)

    _, _, sol_name_to_id, bas_name_to_id = _load_solvent_base_name_maps(excel_file)

    solvent_ids = [
        _resolve_component_id(v, sol_map, sol_name_to_id, "Solvent")
        for v in exp_df["Solvent"].tolist()
    ]

    base_zeros = torch.zeros_like(next(iter(bas_map.values())))
    base_rows: List[Tensor] = []
    for row_idx, (cat_id, base_value) in enumerate(zip(exp_df["Catalyst"], exp_df["Base"])):
        if not cat_is_salt[cat_id]:
            base_rows.append(base_zeros)
            continue
        if pd.isna(base_value) or str(base_value).strip().casefold() in _NO_BASE_LABELS:
            raise ValueError(
                f"Experimental Design row {row_idx}: catalyst {cat_id} needs an external base "
                f"but the Base column reads {base_value!r}."
            )
        base_rows.append(
            bas_map[_resolve_component_id(base_value, bas_map, bas_name_to_id, "Base")]
        )

    x_cat = torch.stack([cat_map[v] for v in exp_df["Catalyst"]])
    x_sol = torch.stack([sol_map[v] for v in solvent_ids])
    x_bas = torch.stack(base_rows)

    cont_bounds = continuous_bounds.to_tensor()
    temperature = torch.tensor(pd.to_numeric(exp_df["Temperature"]).values, dtype=DTYPE)
    form_ratio = torch.tensor(
        pd.to_numeric(exp_df["Formaldehyde Ratio in Eq."]).values, dtype=DTYPE
    )
    cat_loading = torch.tensor(
        pd.to_numeric(exp_df["Catalyst Loading in Molar Ratio"]).values, dtype=DTYPE
    )
    time = torch.tensor(pd.to_numeric(exp_df["Time"]).values, dtype=DTYPE)

    temp_norm = (temperature - cont_bounds[0, 0]) / (
            cont_bounds[1, 0] - cont_bounds[0, 0]
    )
    form_norm = (form_ratio - cont_bounds[0, 1]) / (
            cont_bounds[1, 1] - cont_bounds[0, 1]
    )
    load_norm = (cat_loading - cont_bounds[0, 2]) / (
            cont_bounds[1, 2] - cont_bounds[0, 2]
    )
    time_norm = (time - cont_bounds[0, 3]) / (cont_bounds[1, 3] - cont_bounds[0, 3])

    X_continuous = torch.stack([temp_norm, form_norm, load_norm, time_norm], dim=1)

    X_train = torch.cat([X_continuous, x_cat, x_sol, x_bas], dim=1).to(DTYPE)
    Y_train = torch.tensor(exp_df["Yield"].values, dtype=DTYPE).unsqueeze(-1)

    return X_train, Y_train


def train_gp_model(X_train: Tensor, Y_train: Tensor) -> SingleTaskGP:
    """Train a SingleTaskGP model on the training data.

    Uses input normalization and outcome standardization transforms.

    Args:
        X_train: (n, d) tensor of input features.
        Y_train: (n, 1) tensor of target values.

    Returns:
        A fitted SingleTaskGP model.
    """
    X_train = X_train.to(DEVICE)
    Y_train = Y_train.to(DEVICE)

    model = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        input_transform=Normalize(d=X_train.shape[1]),
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def _print_runtime_estimate(
        acqf: AcquisitionFunction,
        bounds: Tensor,
        cat_emb_avail: Tensor,
        sol_emb_avail: Tensor,
        bas_emb_avail: Tensor,
        cat_is_salt_flags: Tensor,
        q: int,
        batch_constraints: Optional[Dict[int, int | Tuple[int, ...]]],
        fixed_dims: Dict[int, float],
        sequential: bool,
        refine: bool,
        label: str,
) -> None:
    """Calibrate on the real model and print a runtime ETA for one optimization.

    Args:
        acqf: Acquisition function to be optimized.
        bounds: (2, d) bounds tensor.
        cat_emb_avail: (n_cat, d_cat) available-catalyst embeddings.
        sol_emb_avail: (n_sol, d_sol) available-solvent embeddings.
        bas_emb_avail: (n_bas, d_bas) available-base embeddings.
        cat_is_salt_flags: (n_cat,) salt flags.
        q: Batch size.
        batch_constraints: Constraints used for the GA calibration.
        fixed_dims: Fixed-dimension map.
        sequential: Whether the sequential optimizer is used.
        refine: Whether a continuous refine runs (False for scope expansion).
        label: Human-readable label for the printout.
    """
    print(f"  Calibrating runtime estimate ({label})...", flush=True)
    calib = calibrate(
        acqf=acqf,
        bounds=bounds,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        q=q,
        pop=GA_SETTINGS[1],
        raw_samples=LBFGS_SETTINGS[1],
        fixed_dims=fixed_dims,
        batch_constraints=batch_constraints,
        calibrate_refine=refine,
    )
    est = estimate_runtime(
        calib,
        n_evals=GA_SETTINGS[0],
        q=q,
        partition=TEMPERATURE_PARTITION,
        num_restarts=LBFGS_SETTINGS[0],
        repeats=GA_OPTIMIZATION_REPEATS,
        sequential=sequential,
        refine=refine,
    )
    print(format_eta(label, est))


def _optimize_with_repeats(
        acqf: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        cat_emb_avail: Tensor,
        sol_emb_avail: Tensor,
        bas_emb_avail: Tensor,
        cat_is_salt_flags: Tensor,
        batch_constraints: Optional[Dict[int, int | Tuple[int, ...]]],
        fixed_dims: Dict[int, float],
        sequential_and_refine: bool = False,
) -> Tuple[Tensor, float, List[float]]:
    """Run the mixed optimizer ``N`` times and keep the best AF result.

    When ``sequential_and_refine`` is False (default), each repeat runs the
    batch optimizer ``optimize_acqf_mixed_GA`` only, preserving legacy behavior.

    When ``sequential_and_refine`` is True, the optimizer is selected by the
    module-level ``OPTIMIZE_SEQUENTIAL`` flag:

    - ``OPTIMIZE_SEQUENTIAL=True``: each repeat runs
      ``optimize_acqf_mixed_sequential_GA``, which builds the batch greedily and
      refines all continuous dims internally.
    - ``OPTIMIZE_SEQUENTIAL=False``: each repeat runs ``optimize_acqf_mixed_GA``
      followed by a final ``optimize_acqf_continuous`` pass that re-optimizes the
      continuous dims of the chosen discrete batch (enumerating the temperature
      groupings implied by ``batch_constraints``).

    GA settings are read from ``GA_SETTINGS`` and L-BFGS-B refinement settings
    from ``LBFGS_SETTINGS``.

    Args:
        acqf: Acquisition function to optimize.
        bounds: (2, d) tensor of lower/upper bounds.
        q: Batch size for optimization.
        cat_emb_avail: (n_cat, d_cat) available-catalyst embeddings.
        sol_emb_avail: (n_sol, d_sol) available-solvent embeddings.
        bas_emb_avail: (n_bas, d_bas) available-base embeddings.
        cat_is_salt_flags: (n_cat,) bool tensor; when False at index
            ``c`` the base slice is zeroed for candidates choosing that
            catalyst.
        batch_constraints: Optional batch constraints for grouped dimensions.
        fixed_dims: Dimensions fixed to constants during optimization.
        sequential_and_refine: When True, honor ``OPTIMIZE_SEQUENTIAL`` and apply
            a final continuous refinement to the batch path. When False, run the
            plain batch GA (used by scope expansion).

    Returns:
        A tuple of:
            - best_candidates: ``q x d`` candidate tensor with the best AF
              value.
            - best_acq_value: Best acquisition value as float.
            - acq_values: List of AF values across all repeats.

    Raises:
        RuntimeError: If no repeat produced candidates.
    """
    acq_values: List[float] = []
    best_candidates: Optional[Tensor] = None
    best_acq_value: Optional[float] = None

    for _ in range(GA_OPTIMIZATION_REPEATS):
        if sequential_and_refine and OPTIMIZE_SEQUENTIAL:
            run_candidates, run_acq_value = optimize_acqf_mixed_sequential_GA(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                n_evals_ga=GA_SETTINGS[0],
                pop_size=GA_SETTINGS[1],
                cat_emb_avail=cat_emb_avail,
                sol_emb_avail=sol_emb_avail,
                bas_emb_avail=bas_emb_avail,
                cat_is_salt_flags=cat_is_salt_flags,
                batch_constraints=batch_constraints,
                fixed_dims=fixed_dims,
                num_restarts=LBFGS_SETTINGS[0],
                raw_samples=LBFGS_SETTINGS[1],
            )
        else:
            run_candidates, run_acq_value = optimize_acqf_mixed_GA(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                n_evals_ga=GA_SETTINGS[0],
                pop_size=GA_SETTINGS[1],
                cat_emb_avail=cat_emb_avail,
                sol_emb_avail=sol_emb_avail,
                bas_emb_avail=bas_emb_avail,
                cat_is_salt_flags=cat_is_salt_flags,
                batch_constraints=batch_constraints,
                fixed_dims=fixed_dims,
            )
            if sequential_and_refine:
                d_discrete = (
                        cat_emb_avail.shape[1]
                        + sol_emb_avail.shape[1]
                        + bas_emb_avail.shape[1]
                )
                d_continuous = bounds.shape[1] - d_discrete
                run_candidates, run_acq_value = optimize_acqf_continuous(
                    acq_function=acqf,
                    bounds=bounds,
                    X_discrete=run_candidates[:, d_continuous:],
                    batch_constraints=batch_constraints,
                    fixed_dims=fixed_dims,
                    num_restarts=LBFGS_SETTINGS[0],
                    raw_samples=LBFGS_SETTINGS[1],
                )
        run_value = float(run_acq_value.item())
        acq_values.append(run_value)

        if best_acq_value is None or run_value > best_acq_value:
            best_acq_value = run_value
            best_candidates = run_candidates

    if best_candidates is None or best_acq_value is None:
        raise RuntimeError(
            "optimize_acqf_mixed_GA did not produce any candidates "
            f"across {GA_OPTIMIZATION_REPEATS} repeat(s)."
        )

    return best_candidates, best_acq_value, acq_values


def decode_candidates(
        candidates: Tensor,
        combo_ids: Tensor,
        X_discrete: Tensor,
        cat_is_salt: Dict[int, bool],
        cat_bld: Dict[int, str],
        continuous_bounds: ContinuousBounds,
        model: SingleTaskGP,
        d_continuous: int = 4,
        sol_id_to_name: Optional[Dict[int, str]] = None,
        bas_id_to_name: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """Decode optimized candidates back to interpretable values.

    Maps normalized continuous values back to original scale and matches
    discrete embeddings to their corresponding IDs. Also computes posterior
    mean and variance for each candidate.

    Args:
        candidates: (q, d) tensor of candidate points.
        combo_ids: (n_combos, 3) tensor of (cat_id, sol_id, bas_id).
        X_discrete: (n_combos, d_discrete) tensor of discrete embeddings.
        cat_is_salt: Dict mapping catalyst ID to salt flag.
        cat_bld: Dict mapping catalyst ID to BLD number.
        continuous_bounds: Bounds for continuous variables.
        model: Fitted GP model for posterior predictions.
        d_continuous: Number of continuous dimensions.
        sol_id_to_name: Optional dict mapping solvent IDs to names.
        bas_id_to_name: Optional dict mapping base IDs to display labels.

    Returns:
        A list of dicts, each containing decoded candidate values plus
        posterior mean and variance.
    """
    cont_bounds = continuous_bounds.to_tensor()
    results = []

    with torch.no_grad():
        posterior = model.posterior(candidates.to(DEVICE))
        means = posterior.mean.squeeze(-1).cpu()
        variances = posterior.variance.squeeze(-1).cpu()

    for i in range(candidates.shape[0]):
        cand = candidates[i]

        temp = (
                cand[0].item() * (cont_bounds[1, 0] - cont_bounds[0, 0]).item()
                + cont_bounds[0, 0].item()
        )
        form = (
                cand[1].item() * (cont_bounds[1, 1] - cont_bounds[0, 1]).item()
                + cont_bounds[0, 1].item()
        )
        load = (
                cand[2].item() * (cont_bounds[1, 2] - cont_bounds[0, 2]).item()
                + cont_bounds[0, 2].item()
        )
        time = (
                cand[3].item() * (cont_bounds[1, 3] - cont_bounds[0, 3]).item()
                + cont_bounds[0, 3].item()
        )

        discrete_part = cand[d_continuous:]
        distances = torch.norm(X_discrete - discrete_part, dim=1)
        closest_idx = torch.argmin(distances).item()
        cat_id, sol_id, bas_id = combo_ids[closest_idx].tolist()
        workbook_base_id = bas_id
        if not cat_is_salt[cat_id]:
            bas_id = -1
        bld_number = cat_bld.get(cat_id, "-")
        if bld_number == "":
            bld_number = "-"
        solvent_name = (
            sol_id_to_name.get(sol_id, str(sol_id))
            if sol_id_to_name is not None
            else str(sol_id)
        )
        base_label = (
            bas_id_to_name.get(workbook_base_id, str(workbook_base_id))
            if bas_id_to_name is not None
            else str(workbook_base_id)
        )

        results.append(
            {
                "Temperature": round(temp, 2),
                "Formaldehyde_Ratio": round(form, 3),
                "Catalyst_Loading": round(load, 3),
                "Time": round(time, 1),
                "Catalyst_ID": cat_id,
                "Catalyst_BLD": bld_number,
                "Solvent_ID": sol_id,
                "Solvent_Name": solvent_name,
                "Base_ID": bas_id,
                "Base_ID_Workbook": workbook_base_id,
                "Base_Name": base_label,
                "Posterior_Mean": round(means[i].item(), 2),
                "Posterior_Var": round(variances[i].item(), 3),
            }
        )

    return results


def generate_batch(
        excel_file: Optional[Path] = None,
        q_batch: int = 6,
        max_temperature_values: int = 3,
        continuous_bounds: Optional[ContinuousBounds] = None,
        batch_column: str = "Next Batch",
        save_model: bool = False,
) -> Tuple[Tensor, List[Dict]]:
    """Generate a q-batch of candidate experiments.

    Trains a GP model on initial experimental data, then optimizes
    qLogNoisyExpectedImprovement using the training inputs as the
    X_baseline over the mixed continuous-discrete search space with
    exact temperature partitioning.

    Args:
        excel_file: Path to NHC dataset Excel. Defaults to package location.
        q_batch: Number of candidates to generate.
        max_temperature_values: Retained compatibility knob. Must be ``3``
            because temperature partitioning is fixed to ``(2, 2, 2)``.
        continuous_bounds: Bounds for continuous variables.
        batch_column: Which availability column to use.
        save_model: If True, persist the trained model state dict.

    Returns:
        A tuple of:
            - candidates: (q, d) tensor of optimized candidates.
            - decoded: List of dicts with decoded candidate values.

    Raises:
        ValueError: If ``q_batch`` is not compatible with the fixed
            temperature partition ``(2, 2, 2)``.
        ValueError: If ``max_temperature_values`` is not ``3``.
    """
    if excel_file is None:
        excel_file = DEFAULT_EXCEL_FILE
    if continuous_bounds is None:
        continuous_bounds = ContinuousBounds()

    if q_batch != sum(TEMPERATURE_PARTITION):
        raise ValueError(
            "Temperature partition (2, 2, 2) was designed for q = 6; "
            f"got q_batch={q_batch}"
        )
    if max_temperature_values != len(TEMPERATURE_PARTITION):
        raise ValueError(
            "Temperature partition is fixed to (2, 2, 2), so "
            f"max_temperature_values must be 3; got {max_temperature_values}"
        )

    print(f"Loading data from: {excel_file}")
    cat_map, sol_map, bas_map, cat_is_salt, cat_bld, cat_ids, sol_ids, bas_ids = (
        load_reference_data(excel_file)
    )
    sol_id_to_name, bas_id_to_name, _, _ = _load_solvent_base_name_maps(excel_file)
    print(
        f"  Loaded {len(cat_ids)} catalysts, {len(sol_ids)} solvents, {len(bas_ids)} bases"
    )
    print(
        f"  Featurizations: cat={CAT_FEATURES}, solvent={SOLVENT_FEATURES}, base={BASE_FEATURES}"
    )

    available_cats, available_sols, available_bases = load_available_agents(
        excel_file, batch_column
    )
    print(
        f"  Available: {len(available_cats)} catalysts, {len(available_sols)} solvents, {len(available_bases)} bases"
    )

    n_salt = sum(1 for c_id in available_cats if cat_is_salt.get(c_id, False))
    n_nonsalt = len(available_cats) - n_salt
    n_solvents = len(available_sols)
    n_bases = len(available_bases)
    x_allowed_rows_expected = (n_nonsalt * n_solvents) + (n_salt * n_solvents * n_bases)
    print(f"  Catalyst split: {n_salt} salt, {n_nonsalt} non-salt")
    print(
        "  X_allowed rows formula: "
        f"({n_nonsalt} * {n_solvents}) + ({n_salt} * {n_solvents} * {n_bases}) "
        f"= {x_allowed_rows_expected}"
    )

    X_discrete, combo_ids = build_discrete_space(
        cat_map,
        sol_map,
        bas_map,
        cat_is_salt,
        available_cats,
        available_sols,
        available_bases,
    )
    n_discrete_combos = X_discrete.shape[0]
    d_discrete = X_discrete.shape[1]
    print(
        f"  Discrete search space (X_allowed): {n_discrete_combos} rows ({d_discrete} embedding dims)"
    )

    cat_emb_avail = torch.stack([cat_map[c] for c in available_cats]).to(DTYPE)
    sol_emb_avail = torch.stack([sol_map[s] for s in available_sols]).to(DTYPE)
    bas_emb_avail = torch.stack([bas_map[b] for b in available_bases]).to(DTYPE)
    cat_is_salt_flags = torch.tensor(
        [cat_is_salt[c] for c in available_cats], dtype=torch.bool
    )

    X_train, Y_train = load_training_data(
        excel_file, cat_map, sol_map, bas_map, continuous_bounds, cat_is_salt
    )
    d_continuous = 4
    d_total = d_continuous + d_discrete
    print(f"  Training data: {X_train.shape[0]} experiments, {d_total} total dims")

    print("Training GP model...")
    model = train_gp_model(X_train, Y_train)
    best_f = Y_train.max().item()
    print(f"  Best observed yield: {best_f:.2f}")

    if save_model:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  Model state dict saved to {MODEL_SAVE_PATH}")

    full_bounds = torch.zeros(2, d_total, dtype=DTYPE)
    full_bounds[0, :d_continuous] = 0.0
    full_bounds[1, :d_continuous] = 1.0
    full_bounds[0, d_continuous:] = 0.0
    full_bounds[1, d_continuous:] = 1.0

    batch_constraints = {0: TEMPERATURE_PARTITION}

    # Fix Time=20: dim 3 is Time, normalized to 1.0 with bounds (16, 20).
    cont_bounds = continuous_bounds.to_tensor()
    time_norm_20 = (20.0 - cont_bounds[0, 3].item()) / (
            cont_bounds[1, 3].item() - cont_bounds[0, 3].item()
    )
    fixed_dims = {3: time_norm_20}

    optimizer_mode = "sequential GA" if OPTIMIZE_SEQUENTIAL else "batch GA + refine"
    print(
        f"Optimizing acquisition function (q={q_batch}, temp_partition={TEMPERATURE_PARTITION}, time=20)..."
    )
    print(
        f"  Optimizer: {optimizer_mode}; GA settings: "
        f"n_evals={GA_SETTINGS[0]}, pop_size={GA_SETTINGS[1]}, repeats={GA_OPTIMIZATION_REPEATS}; "
        f"L-BFGS settings: num_restarts={LBFGS_SETTINGS[0]}, raw_samples={LBFGS_SETTINGS[1]}"
    )
    # acqf = qLogExpectedImprovement(model=model, best_f=best_f)
    acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=X_train.to(DEVICE))
    _print_runtime_estimate(
        acqf=acqf,
        bounds=full_bounds,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        q=q_batch,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        sequential=OPTIMIZE_SEQUENTIAL,
        refine=True,
        label="batch generation",
    )
    candidates, best_acq_value, acq_values = _optimize_with_repeats(
        acqf=acqf,
        bounds=full_bounds,
        q=q_batch,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        batch_constraints=batch_constraints,
        fixed_dims=fixed_dims,
        sequential_and_refine=True,
    )

    acq_values_str = ", ".join(f"{value:.3f}" for value in acq_values)
    print(f"  Acquisition values: [{acq_values_str}]")
    print(f"  Best acquisition value: {best_acq_value:.4f}")

    decoded = decode_candidates(
        candidates,
        combo_ids,
        X_discrete,
        cat_is_salt,
        cat_bld,
        continuous_bounds,
        model,
        d_continuous,
        sol_id_to_name=sol_id_to_name,
        bas_id_to_name=bas_id_to_name,
    )

    return candidates, decoded


def print_candidates(decoded: List[Dict]) -> None:
    """Print decoded candidates in a formatted table.

    Args:
        decoded: List of dicts with decoded candidate values.
    """
    print("\n" + "=" * 134)
    print("GENERATED CANDIDATES")
    print("=" * 134)
    print(
        f"{'#':<3} {'Temp (°C)':<10} {'Form. Ratio':<12} {'Cat. Load':<10} {'Time':<8} "
        f"{'Cat ID':<8} {'BLD':<12} {'Sol ID':<8} {'Base ID':<8} {'Mean':<8} {'Var':<8}"
    )
    print("-" * 134)

    for i, cand in enumerate(decoded, 1):
        print(
            f"{i:<3} "
            f"{cand['Temperature']:<10.2f} "
            f"{cand['Formaldehyde_Ratio']:<12.3f} "
            f"{cand['Catalyst_Loading']:<10.3f} "
            f"{cand['Time']:<8.1f} "
            f"{cand['Catalyst_ID']:<8} "
            f"{cand['Catalyst_BLD']:<12} "
            f"{cand['Solvent_ID']:<8} "
            f"{cand['Base_ID']:<8} "
            f"{cand['Posterior_Mean']:<8.2f} "
            f"{cand['Posterior_Var']:<8.3f}"
        )

    temps = [c["Temperature"] for c in decoded]
    unique_temps = len(set([round(t, 1) for t in temps]))
    print("-" * 134)
    print(f"Unique temperature values: {unique_temps}")
    print("=" * 134)


def _is_empty_excel_value(value) -> bool:
    """Return True when an Excel cell value should be considered empty.
        Returns:
            bool: Return value of _is_empty_excel_value.

    Returns:
        bool: Return value of _is_empty_excel_value.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _row_has_any_value(
        ws, row_idx: int, max_col: int, skip_col: Optional[int] = None
) -> bool:
    """Return True when a row contains any non-empty value.

    Args:
        ws: Worksheet to inspect.
        row_idx: 1-based row index to inspect.
        max_col: Maximum column index to scan.
        skip_col: Optional column index to ignore while scanning.

    Returns:
        True if any non-empty value is found in the row.
    """
    for col_idx in range(1, max_col + 1):
        if skip_col is not None and col_idx == skip_col:
            continue
        if not _is_empty_excel_value(ws.cell(row_idx, col_idx).value):
            return True
    return False


def _collect_design_rows(ws) -> Tuple[List[int], List[int]]:
    """Collect populated and invalid Experimental Design rows.

    Column B (Catalyst) is treated as the sentinel for a populated row. Rows
    with an empty column B are ignored unless other columns contain data.

    Returns:
        A tuple of (populated_rows, invalid_rows), where invalid_rows have
        empty column B but contain other non-empty values.
    """
    max_col = ws.max_column
    populated_rows: List[int] = []
    invalid_rows: List[int] = []

    for row_idx in range(2, ws.max_row + 1):
        catalyst_val = ws.cell(row_idx, 2).value
        if _is_empty_excel_value(catalyst_val):
            if _row_has_any_value(ws, row_idx, max_col, skip_col=2):
                invalid_rows.append(row_idx)
            continue
        populated_rows.append(row_idx)

    return populated_rows, invalid_rows


def _next_batch_value(ws, row_indices: Optional[List[int]] = None) -> int:
    """Infer the next batch index from column A values in Experimental Design.

    Args:
        ws: Worksheet to inspect.
        row_indices: Optional list of row indices to consider. When None,
            all rows in the worksheet are scanned.

    Returns:
        The next batch integer to assign.
    """
    numeric_values: List[int] = []
    if row_indices is None:
        row_indices = list(range(2, ws.max_row + 1))

    for row_idx in row_indices:
        batch_value = ws.cell(row_idx, 1).value
        if batch_value is None:
            continue
        if isinstance(batch_value, (int, float)):
            numeric_values.append(int(batch_value))
            continue
        text_value = str(batch_value).strip()
        if text_value == "":
            continue
        if text_value.isdigit() or (
                text_value.startswith("-") and text_value[1:].isdigit()
        ):
            numeric_values.append(int(text_value))

    if len(numeric_values) == 0:
        return 0
    return max(numeric_values) + 1


def _write_suggested_row(
        ws, row_idx: int, candidate: Dict, batch_value: Optional[int]
) -> None:
    """Write a single suggested candidate row into Experimental Design.

    ``candidate["Time"]`` is treated as optional for backward compatibility
    with legacy decoded payloads that did not include a time entry.
    """
    if batch_value is not None:
        ws.cell(row_idx, 1).value = batch_value
    ws.cell(row_idx, 2).value = candidate["Catalyst_ID"]
    ws.cell(row_idx, 3).value = candidate.get("Solvent_Name", candidate["Solvent_ID"])
    ws.cell(row_idx, 4).value = candidate.get(
        "Base_Name", candidate.get("Base_ID_Workbook", candidate["Base_ID"])
    )
    ws.cell(row_idx, 5).value = candidate["Formaldehyde_Ratio"]
    ws.cell(row_idx, 6).value = candidate["Catalyst_Loading"]
    ws.cell(row_idx, 7).value = candidate["Temperature"]
    ws.cell(row_idx, 8).value = candidate.get("Time")
    ws.cell(row_idx, 9).value = None
    ws.cell(row_idx, 10).value = None
    ws.cell(row_idx, 11).value = None


def update_experimental_design_with_batch(
        excel_file: Path,
        decoded_batch: List[Dict],
        q_batch: int,
) -> None:
    """Append or update suggested q-batch rows in Experimental Design.

    Rules:
    - If result-less row count is 0, append q rows.
    - If result-less row count is exactly q, update those rows.
    - Otherwise, emit warning and do not edit workbook.
    - Rows are counted only when column B (Catalyst) is non-empty. Rows with
      empty column B but other data are treated as invalid.

    Args:
        excel_file: Path to the NHC dataset Excel workbook; modified in place.
        decoded_batch: Decoded candidate dicts to write, one per row.
        q_batch: Expected batch size; the workbook is left untouched if
            ``len(decoded_batch)`` differs.

    Raises:
        ValueError: If the workbook has no ``Experimental Design`` sheet.
    """
    wb = load_workbook(excel_file)
    if "Experimental Design" not in wb.sheetnames:
        raise ValueError("Sheet 'Experimental Design' not found")

    ws = wb["Experimental Design"]

    if len(decoded_batch) != q_batch:
        print(
            "WARNING: Generated batch size does not match q "
            f"({len(decoded_batch)} vs {q_batch}). Not editing workbook."
        )
        return

    populated_rows, invalid_rows = _collect_design_rows(ws)
    if invalid_rows:
        print(
            "WARNING: Experimental Design rows with empty Catalyst (column B) "
            "but other data present at rows "
            f"{invalid_rows}. Please fix the Excel sheet. Not editing workbook."
        )
        return

    resultless_rows: List[int] = []
    for row_idx in populated_rows:
        yield_val = ws.cell(row_idx, 9).value
        if _is_empty_excel_value(yield_val):
            resultless_rows.append(row_idx)

    n_resultless = len(resultless_rows)
    print(
        "  Experimental Design rows without results "
        f"(Yield empty, column B present): {n_resultless}"
    )

    last_populated_row = populated_rows[-1] if populated_rows else None
    if last_populated_row is not None:
        last_yield = ws.cell(last_populated_row, 9).value
        print(f"  Last populated row Yield: {last_yield}")

    if n_resultless == 0:
        next_batch = _next_batch_value(ws, populated_rows)
        start_row = (last_populated_row or 1) + 1
        for idx, candidate in enumerate(decoded_batch):
            _write_suggested_row(
                ws=ws,
                row_idx=start_row + idx,
                candidate=candidate,
                batch_value=next_batch,
            )

        wb.save(excel_file)
        print(
            f"  Appended {q_batch} suggested rows to Experimental Design "
            f"(batch={next_batch})."
        )
        return

    if n_resultless == q_batch:
        for row_idx, candidate in zip(resultless_rows, decoded_batch):
            _write_suggested_row(
                ws=ws,
                row_idx=row_idx,
                candidate=candidate,
                batch_value=None,
            )

        wb.save(excel_file)
        print(f"  Updated existing {q_batch} result-less rows with suggested batch.")
        return

    print(
        "WARNING: Number of result-less rows does not match q "
        f"({n_resultless} vs {q_batch}). Not editing workbook."
    )


def generate_scope_expansion(
        excel_file: Optional[Path] = None,
        continuous_bounds: Optional[ContinuousBounds] = None,
) -> Tuple[Tensor, Dict]:
    """Generate a single scope expansion candidate from unordered catalysts.

    Selects one catalyst from the unordered set (marked 'n' in all availability
    columns) using LogNoisyExpectedImprovement. This represents ordering a new
    catalyst that is not currently available or already on order.

    Args:
        excel_file: Path to NHC dataset Excel. Defaults to package location.
        continuous_bounds: Bounds for continuous variables.

    Returns:
        A tuple of:
            - candidate: (1, d) tensor of the expansion candidate.
            - decoded: Dict with decoded candidate values.

    Raises:
        ValueError: If no unordered catalysts are available.
    """
    if excel_file is None:
        excel_file = DEFAULT_EXCEL_FILE
    if continuous_bounds is None:
        continuous_bounds = ContinuousBounds()

    print(f"Loading data from: {excel_file}")
    cat_map, sol_map, bas_map, cat_is_salt, cat_bld, cat_ids, sol_ids, bas_ids = (
        load_reference_data(excel_file)
    )
    sol_id_to_name, bas_id_to_name, _, _ = _load_solvent_base_name_maps(excel_file)
    print(
        f"  Loaded {len(cat_ids)} catalysts, {len(sol_ids)} solvents, {len(bas_ids)} bases"
    )
    print(
        f"  Featurizations: cat={CAT_FEATURES}, solvent={SOLVENT_FEATURES}, base={BASE_FEATURES}"
    )

    unordered_cats = load_unordered_catalysts(excel_file, cat_ids)
    print(f"  Unordered catalysts (n,n,n): {len(unordered_cats)}")

    if len(unordered_cats) == 0:
        raise ValueError("No unordered catalysts available for scope expansion")

    n_salt = sum(1 for c_id in unordered_cats if cat_is_salt.get(c_id, False))
    n_nonsalt = len(unordered_cats) - n_salt
    n_solvents = len(sol_ids)
    n_bases = len(bas_ids)
    x_allowed_rows_expected = (n_nonsalt * n_solvents) + (n_salt * n_solvents * n_bases)
    print(f"  Catalyst split: {n_salt} salt, {n_nonsalt} non-salt")
    print(
        "  X_allowed rows formula: "
        f"({n_nonsalt} * {n_solvents}) + ({n_salt} * {n_solvents} * {n_bases}) "
        f"= {x_allowed_rows_expected}"
    )

    X_unordered, combo_ids_unordered = build_unordered_discrete_space(
        cat_map,
        sol_map,
        bas_map,
        cat_is_salt,
        unordered_cats,
        sol_ids,
        bas_ids,
    )
    n_unordered_combos = X_unordered.shape[0]
    d_discrete = X_unordered.shape[1]
    print(
        f"  Unordered search space (X_allowed): {n_unordered_combos} rows ({d_discrete} embedding dims)"
    )

    cat_emb_avail = torch.stack([cat_map[c] for c in unordered_cats]).to(DTYPE)
    sol_emb_avail = torch.stack([sol_map[s] for s in sol_ids]).to(DTYPE)
    bas_emb_avail = torch.stack([bas_map[b] for b in bas_ids]).to(DTYPE)
    cat_is_salt_flags = torch.tensor(
        [cat_is_salt[c] for c in unordered_cats], dtype=torch.bool
    )

    X_train, Y_train = load_training_data(
        excel_file, cat_map, sol_map, bas_map, continuous_bounds, cat_is_salt
    )
    d_continuous = 4
    d_total = d_continuous + d_discrete
    print(f"  Training data: {X_train.shape[0]} experiments, {d_total} total dims")

    print("Training GP model...")
    model = train_gp_model(X_train, Y_train)
    best_f = Y_train.max().item()
    print(f"  Best observed yield: {best_f:.2f}")

    full_bounds = torch.zeros(2, d_total, dtype=DTYPE)
    full_bounds[0, :d_continuous] = 0.0
    full_bounds[1, :d_continuous] = 1.0
    full_bounds[0, d_continuous:] = 0.0
    full_bounds[1, d_continuous:] = 1.0

    # Fix Time=20: dim 3 is Time, normalized to 1.0 with bounds (16, 20).
    cont_bounds = continuous_bounds.to_tensor()
    time_norm_20 = (20.0 - cont_bounds[0, 3].item()) / (
            cont_bounds[1, 3].item() - cont_bounds[0, 3].item()
    )
    fixed_dims = {3: time_norm_20}

    print("Optimizing LogEI for scope expansion (q=1, time=20)...")
    print(
        "  GA settings: "
        f"n_evals={GA_SETTINGS[0]}, pop_size={GA_SETTINGS[1]}, repeats={GA_OPTIMIZATION_REPEATS}"
    )
    acqf = LogNoisyExpectedImprovement(model=model, best_f=best_f)

    _print_runtime_estimate(
        acqf=acqf,
        bounds=full_bounds,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        q=1,
        batch_constraints=None,
        fixed_dims=fixed_dims,
        sequential=False,
        refine=False,
        label="scope expansion",
    )

    candidate, best_acq_value, acq_values = _optimize_with_repeats(
        acqf=acqf,
        bounds=full_bounds,
        q=1,
        cat_emb_avail=cat_emb_avail,
        sol_emb_avail=sol_emb_avail,
        bas_emb_avail=bas_emb_avail,
        cat_is_salt_flags=cat_is_salt_flags,
        batch_constraints=None,
        fixed_dims=fixed_dims,
    )

    acq_values_str = ", ".join(f"{value:.3f}" for value in acq_values)
    print(f"  Acquisition values: [{acq_values_str}]")
    print(f"  Best acquisition value: {best_acq_value:.4f}")

    decoded_list = decode_candidates(
        candidate,
        combo_ids_unordered,
        X_unordered,
        cat_is_salt,
        cat_bld,
        continuous_bounds,
        model,
        d_continuous,
        sol_id_to_name=sol_id_to_name,
        bas_id_to_name=bas_id_to_name,
    )
    decoded = decoded_list[0]

    return candidate, decoded


def print_expansion_candidate(decoded: Dict) -> None:
    """Print the scope expansion candidate.

    Args:
        decoded: Dict with decoded candidate values.
    """
    print("\n" + "=" * 134)
    print("SCOPE EXPANSION CANDIDATE (from unordered catalysts)")
    print("=" * 134)
    print(
        f"{'Temp (°C)':<10} {'Form. Ratio':<12} {'Cat. Load':<10} {'Time':<8} "
        f"{'Cat ID':<8} {'BLD':<12} {'Sol ID':<8} {'Base ID':<8} {'Mean':<8} {'Var':<8}"
    )
    print("-" * 134)
    print(
        f"{decoded['Temperature']:<10.2f} "
        f"{decoded['Formaldehyde_Ratio']:<12.3f} "
        f"{decoded['Catalyst_Loading']:<10.3f} "
        f"{decoded['Time']:<8.1f} "
        f"{decoded['Catalyst_ID']:<8} "
        f"{decoded['Catalyst_BLD']:<12} "
        f"{decoded['Solvent_ID']:<8} "
        f"{decoded['Base_ID']:<8} "
        f"{decoded['Posterior_Mean']:<8.2f} "
        f"{decoded['Posterior_Var']:<8.3f}"
    )
    print("=" * 134)
    print(
        f">> Recommended catalyst to order: {decoded['Catalyst_ID']} "
        f"(BLD: {decoded['Catalyst_BLD']})"
    )
    print("=" * 134)


def main():
    """Main entry point for batch candidate generation and scope expansion."""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING BATCH CANDIDATES (from available reagents)")
    print("=" * 80 + "\n")

    q_batch = 6
    candidates, decoded = generate_batch(
        q_batch=q_batch,
        max_temperature_values=3,
        save_model=True,
    )
    print_candidates(decoded)
    update_experimental_design_with_batch(DEFAULT_EXCEL_FILE, decoded, q_batch)

    if EXTEND_SCOPE:
        print("\n" + "=" * 80)
        print("STEP 2: SCOPE EXPANSION (selecting catalyst to order)")
        print("=" * 80 + "\n")

        expansion_candidate, expansion_decoded = generate_scope_expansion()
        print_expansion_candidate(expansion_decoded)
    else:
        print("\nSkipping scope expansion because EXTEND_SCOPE=False.")


if __name__ == "__main__":
    main()
