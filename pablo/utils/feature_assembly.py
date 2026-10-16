"""Assemble a per-cell `(X, y, dims)` tensor from per-component feature parquets.

The output `CellFeatures` object is what the featurization scan's CV stages
consume: [fit_eval](../featurization_and_kernel_scan/cross_validation/fit_eval.py),
the [CV scheduler](../featurization_and_kernel_scan/cross_validation/scheduler.py),
and the kernel and model builders in
[core/](../featurization_and_kernel_scan/core/).

X-column layout, left to right:

    [cat_features ... | solv_features ... | base_features ... | base_active | cont_features ... | form_id]

`feature_dims` reports the column ranges for cat / solv / base / cont (excluding the
single `base_active` and `form_id` columns). `active_flag_dim` is the index of the
`base_active` column and `form_id_dim` the index of the integer pre-catalyst-form
column.

Conventions:
    - All featurization parquets are indexed by `catalyst_id` / `solvent_id` / `base_id`.
    - `experiments.parquet` keys solvents by `solvent_library.name` and bases by
      `base_library.comment` (NOT `name`). Every row with `base_active == 0` has its
      base feature block filled with zeros: the rows with `base == "no"`, and the
      non-salt catalyst runs a base was added to anyway. The zero block is the
      "none" base location (no base block carries an integer index column, so 0 means
      "no base"); KS2-KS4 additionally zero-mask their base terms via the
      `base_active` flag.
    - Continuous features are derived directly from `experiments.parquet` columns
      (`fa_ratio`, `loading`, `temperature`) using the library bounds in
      [data/featurizations/continuous/metadata.json](../featurization_and_kernel_scan/data/featurizations/continuous/metadata.json).
      The deduplicated 59-row `continuous/descriptors.parquet` is intentionally
      not consumed here — it's a Phase-1 artifact that doesn't align with
      experiments-row indexing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

DTYPE = torch.float64
LOGIT_EPS = 1e-3

FORM_ID_ORDER = ("salt", "free_nhc", "co2_adduct", "bicarbonate")
N_FORMS = len(FORM_ID_ORDER)

FORM_ONEHOT_COLUMNS = tuple(f"form_{label}" for label in FORM_ID_ORDER)
"""Pre-catalyst-form one-hot columns of ``interpretable/descriptors.parquet``."""


@dataclass
class CellFeatures:
    """Assembled feature matrix and indices for one (cat, solv, base) cell.

    Attributes:
        X: (N, D_total) — feature matrix, float64.
        y_raw: (N, 1) — yield as fraction in [0, 1].
        y_logit: (N, 1) — logit((y/100 + ε)/(1+2ε)) of yield_pct, ε = 1e-3.
        y_conversion_raw: (N, 1) — conversion as fraction in [0, 1]. First task
            of the (C, S) LCM-Q=2 multitask model.
        y_conversion_logit: (N, 1) — logit-transformed conversion, ε = 1e-3.
        y_selectivity_raw: (N, 1) — selectivity as fraction in [0, 1]. Second
            task of the (C, S) LCM-Q=2 multitask model. Yield metrics are
            derived as yield = C × S via sample-based product of the joint
            posterior.
        y_selectivity_logit: (N, 1) — logit-transformed selectivity, ε = 1e-3.
        feature_dims: dict with keys "cat", "solv", "base", "cont"; each maps to
            the list of column indices into X. Excludes `active_flag_dim`.
        active_flag_dim: column index of the `base_active` indicator.
        catalyst_ids: (N,) np.int64 — for catalyst-grouped CV.
        replicate_id: (N,) np.int64 — −1 for non-replicated rows.
        pre_cat_form: (N,) object/str — one of {"salt", "free_nhc",
            "co2_adduct", "bicarbonate"}; used for stratified CV.
        feature_names: list[str] of length D_total — for hyperparameter readability.
        form_id_dim: column index of the integer pre-catalyst-form ID
            (0..N_FORMS-1, mapping `("salt", "free_nhc", "co2_adduct",
            "bicarbonate")`). Appended after the continuous block. Kernels
            ignore it (excluded from `feature_dims`); `TypeLinearMean` reads it.
            The model factory must pass `indices=non_form_dims` to
            `Normalize(...)` so the integer is left intact.
        partition: Optional lengthscale partition with keys "cat", "solv",
            "base", "cont" — each a list of column-index groups, one shared
            lengthscale per group. `None` until populated by
            `kernels.build_partition`; the kernel builders call that same
            function themselves when it is `None`, so leaving it unset yields
            the default recipe (one shared lengthscale per discrete block, full
            ARD on the continuous block).
        cat_library: (L, d_cat) — the catalyst block of every catalyst in the
            featurization's table (the 161-catalyst library), built exactly as
            the rows of X; read by the KS7 kernel. `None` when not assembled.
    """

    X: Tensor
    y_raw: Tensor
    y_logit: Tensor
    y_conversion_raw: Tensor
    y_conversion_logit: Tensor
    y_selectivity_raw: Tensor
    y_selectivity_logit: Tensor
    feature_dims: dict[str, list[int]]
    active_flag_dim: int
    catalyst_ids: np.ndarray
    replicate_id: np.ndarray
    pre_cat_form: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    form_id_dim: int = -1
    partition: Optional[dict[str, list[list[int]]]] = None
    cat_library: Optional[Tensor] = None


def _load_numeric_descriptors(path: Path) -> pd.DataFrame:
    """Read a parquet, keep only numeric columns. Index is preserved.
        Returns:
            pd.DataFrame: Return value of _load_numeric_descriptors.
    
    Returns:
        pd.DataFrame: Return value of _load_numeric_descriptors.
"""
    df = pd.read_parquet(path)
    return df.select_dtypes(include="number").copy()


def _resolve_catalyst_parquet(featurizations_root: Path, name: str) -> Path:
    """Locate a featurization's delivered feature matrix.

    Every featurization ships its raw matrix as `descriptors.parquet`.

    Returns:
        Path: Path to the featurization's `descriptors.parquet`.
    """
    return _resolve_feature_parquet(featurizations_root / name)


def _resolve_feature_parquet(cell_dir: Path) -> Path:
    """Resolve a featurization directory to its raw ``descriptors.parquet``.

    Returns:
        Path: Path to the featurization's ``descriptors.parquet``.
    """
    p = cell_dir / "descriptors.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No descriptors.parquet under {cell_dir}")
    return p


def _build_continuous(
    experiments_df: pd.DataFrame, featurizations_root: Path
) -> tuple[np.ndarray, list[str]]:
    """Apply library-bounds [-1, 1] map to fa_ratio / loading / temperature.
        Returns:
            tuple[np.ndarray, list[str]]: Return value of _build_continuous.
    
    Returns:
        tuple[np.ndarray, list[str]]: Return value of _build_continuous.
"""
    meta = json.loads(
        (featurizations_root / "continuous" / "metadata.json").read_text()
    )
    bounds = meta["bounds"]
    raw_cols = meta["raw_columns"]
    out = np.empty((len(experiments_df), len(raw_cols)), dtype=np.float64)
    names = []
    for j, col in enumerate(raw_cols):
        lo, hi = bounds[col]
        out[:, j] = 2.0 * (experiments_df[col].to_numpy(dtype=np.float64) - lo) / (hi - lo) - 1.0
        names.append(f"{col}_std")
    return out, names


def _build_solv_block(
    experiments_df: pd.DataFrame,
    solv_featurization: str,
    featurizations_root: Path,
    libraries_root: Path,
) -> tuple[np.ndarray, list[str]]:
    feat_path = _resolve_feature_parquet(featurizations_root / f"solvent_{solv_featurization}")
    feats = _load_numeric_descriptors(feat_path)
    solv_lib = pd.read_parquet(libraries_root / "solvents.parquet")
    name_to_id = dict(zip(solv_lib["name"], solv_lib["id"]))
    ids = experiments_df["solvent"].map(name_to_id)
    if ids.isna().any():
        unknown = experiments_df.loc[ids.isna(), "solvent"].unique().tolist()
        raise KeyError(f"Solvents in experiments not in solvent_library: {unknown}")
    block = feats.reindex(ids.to_numpy()).to_numpy(dtype=np.float64)
    return block, [f"solv__{c}" for c in feats.columns]


def _build_base_block(
    experiments_df: pd.DataFrame,
    base_featurization: str,
    featurizations_root: Path,
    libraries_root: Path,
) -> tuple[np.ndarray, list[str]]:
    """Base feature block, zero on every row whose catalyst needs no base.

    A row is zeroed when ``base_active == 0``, which ``build_experiments`` derives
    from the catalyst's pre-catalyst form. That covers ``base == "no"`` and the
    non-salt runs a base was pipetted into anyway: the base is chemically inert
    there, so the row sits at the "none" location like every other non-salt row.

    Args:
        experiments_df: Experiment table with ``base`` and ``base_active``.
        base_featurization: Resolves to ``base_<name>/``.
        featurizations_root: ``data/featurizations/`` root.
        libraries_root: Folder holding ``bases.parquet``.

    Returns:
        ``(block, names)``: the ``(n_experiments, d_base)`` block and its column
        names.

    Raises:
        KeyError: If an active row names a base absent from ``bases.parquet``.
        ValueError: If a row has no base but ``base_active == 1``.
    """
    feat_path = _resolve_feature_parquet(featurizations_root / f"base_{base_featurization}")
    feats = _load_numeric_descriptors(feat_path)
    base_lib = pd.read_parquet(libraries_root / "bases.parquet")
    comment_to_id = dict(zip(base_lib["comment"], base_lib["id"]))

    n = len(experiments_df)
    block = np.zeros((n, feats.shape[1]), dtype=np.float64)
    active = experiments_df["base_active"].to_numpy()
    for i, b in enumerate(experiments_df["base"].to_numpy()):
        if active[i] == 0:
            continue
        if b == "no":
            raise ValueError(f"row {i} has base_active == 1 but no base")
        if b not in comment_to_id:
            raise KeyError(f"Base {b!r} not in base_library.comment column.")
        block[i] = feats.loc[comment_to_id[b]].to_numpy(dtype=np.float64)
    return block, [f"base__{c}" for c in feats.columns]


def _build_cat_block(
    experiments_df: pd.DataFrame,
    cat_featurization: str,
    featurizations_root: Path,
    append_form_onehot_to_cat: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Catalyst block of the experiments and of the whole catalyst table.

    Returns:
        ``(block, names, library)``: the ``(N, d_cat)`` block of the experiment
        rows, its column names, and the ``(L, d_cat)`` block of every catalyst in
        the featurization's table, built the same way.

    Raises:
        ValueError: If an experiment's catalyst or a library catalyst has no
            complete row.
    """
    feat_path = _resolve_catalyst_parquet(featurizations_root, cat_featurization)
    feats = _load_numeric_descriptors(feat_path)
    block = feats.reindex(experiments_df["catalyst_id"].to_numpy()).to_numpy(dtype=np.float64)
    library = feats.to_numpy(dtype=np.float64)
    names = [f"cat__{c}" for c in feats.columns]

    if append_form_onehot_to_cat:
        form_df = _form_onehot_table(featurizations_root)
        form_block = form_df.reindex(experiments_df["catalyst_id"].to_numpy()).to_numpy(dtype=np.float64)
        block = np.concatenate([block, form_block], axis=1)
        library = np.concatenate(
            [library, form_df.reindex(feats.index).to_numpy(dtype=np.float64)], axis=1
        )
        names += [f"cat__{c}" for c in form_df.columns]

    if np.isnan(block).any():
        bad = experiments_df.loc[np.isnan(block).any(axis=1), "catalyst_id"].unique().tolist()
        raise ValueError(f"Catalyst featurization missing rows for catalyst_ids: {bad}")
    if np.isnan(library).any():
        bad = feats.index[np.isnan(library).any(axis=1)].tolist()
        raise ValueError(f"Catalyst library rows incomplete for catalyst_ids: {bad}")
    return block, names, library


def _form_onehot_table(featurizations_root: Path) -> pd.DataFrame:
    """The catalyst-indexed pre-catalyst-form one-hot block.

    Args:
        featurizations_root: ``data/featurizations/`` root.

    Returns:
        ``(161, 4)`` frame over ``FORM_ONEHOT_COLUMNS``, indexed by catalyst id.

    Raises:
        ValueError: If the columns are absent, or a row is not exactly one-hot.
    """
    path = featurizations_root / "interpretable" / "descriptors.parquet"
    df = pd.read_parquet(path)
    missing = [c for c in FORM_ONEHOT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing the pre-catalyst-form one-hot columns {missing}. "
            "Rebuild it with features.interpretable.assemble."
        )
    block = df[list(FORM_ONEHOT_COLUMNS)]
    row_sums = block.to_numpy(dtype=np.int64).sum(axis=1)
    if not (row_sums == 1).all():
        bad = df.index[row_sums != 1].tolist()
        raise ValueError(f"Form one-hot is not one-hot for catalyst_ids {bad}")
    return block


def _pre_cat_form(
    experiments_df: pd.DataFrame, featurizations_root: Path
) -> np.ndarray:
    block = _form_onehot_table(featurizations_root)
    form_series = pd.Series(
        np.array(FORM_ID_ORDER)[block.to_numpy(dtype=np.int64).argmax(axis=1)],
        index=block.index,
    )
    return form_series.reindex(experiments_df["catalyst_id"].to_numpy()).to_numpy()


def assemble_features(
    cat_featurization: str,
    solv_featurization: str,
    base_featurization: str,
    experiments_path: Path | str,
    featurizations_root: Path | str,
    libraries_root: Optional[Path | str] = None,
    append_form_onehot_to_cat: bool = False,
) -> CellFeatures:
    """Load and assemble a CellFeatures object for one (cat, solv, base) cell.

    Args:
        cat_featurization: subdirectory name under `featurizations_root`,
            e.g. "map4", "ecfp4", "mol2vec", "mordred", "interpretable",
            "interpretable_light", or any future foundation-model variant.
        solv_featurization: solvent featurization name; resolves to
            `featurizations_root / "solvent_<solv_featurization>"`.
        base_featurization: base featurization name; resolves to
            `featurizations_root / "base_<base_featurization>"`.
        experiments_path: path to experiments.parquet.
        featurizations_root: path to data/featurizations/ root.
        libraries_root: where to find solvents.parquet / bases.parquet.
            Defaults to `experiments_path.parent`.
        append_form_onehot_to_cat: if True, concatenate the 4-column form
            one-hot from interpretable to the catalyst block. Use this for
            decoupled foundation-model featurizations (free-carbene SMILES + form).

    Returns:
        CellFeatures populated and ready for indexing by CV splits.
    """
    experiments_path = Path(experiments_path)
    featurizations_root = Path(featurizations_root)
    libraries_root = (
        Path(libraries_root) if libraries_root is not None else experiments_path.parent
    )

    experiments_df = pd.read_parquet(experiments_path)
    n_raw = len(experiments_df)
    experiments_df = (
        experiments_df.dropna(subset=["yield_pct"]).reset_index(drop=True)
    )
    if (n_dropped := n_raw - len(experiments_df)) > 0:
        print(f"[assemble] dropped {n_dropped} rows with NaN yield_pct (kept {len(experiments_df)}).")

    cat_block, cat_names, cat_library = _build_cat_block(
        experiments_df, cat_featurization, featurizations_root, append_form_onehot_to_cat
    )
    solv_block, solv_names = _build_solv_block(
        experiments_df, solv_featurization, featurizations_root, libraries_root
    )
    base_block, base_names = _build_base_block(
        experiments_df, base_featurization, featurizations_root, libraries_root
    )
    cont_block, cont_names = _build_continuous(experiments_df, featurizations_root)

    base_active = experiments_df["base_active"].to_numpy(dtype=np.float64).reshape(-1, 1)

    pre_cat_form = _pre_cat_form(experiments_df, featurizations_root)
    form_id_lookup = {f: i for i, f in enumerate(FORM_ID_ORDER)}
    try:
        form_ids = np.array([form_id_lookup[f] for f in pre_cat_form], dtype=np.float64)
    except KeyError as e:
        raise ValueError(
            f"pre_cat_form contains a label outside FORM_ID_ORDER={FORM_ID_ORDER}: {e}"
        )
    form_id_col = form_ids.reshape(-1, 1)

    blocks = [cat_block, solv_block, base_block, base_active, cont_block, form_id_col]
    X_np = np.concatenate(blocks, axis=1)
    feature_names = (
        cat_names + solv_names + base_names + ["base_active"] + cont_names + ["form_id"]
    )

    d_cat = cat_block.shape[1]
    d_solv = solv_block.shape[1]
    d_base = base_block.shape[1]
    d_cont = cont_block.shape[1]
    feature_dims = {
        "cat": list(range(0, d_cat)),
        "solv": list(range(d_cat, d_cat + d_solv)),
        "base": list(range(d_cat + d_solv, d_cat + d_solv + d_base)),
        "cont": list(
            range(d_cat + d_solv + d_base + 1, d_cat + d_solv + d_base + 1 + d_cont)
        ),
    }
    active_flag_dim = d_cat + d_solv + d_base
    form_id_dim = d_cat + d_solv + d_base + 1 + d_cont

    def _to_frac_and_logit(col_pct: np.ndarray) -> tuple[Tensor, Tensor]:
        frac = col_pct / 100.0
        logit_np = np.log((frac + LOGIT_EPS) / (1.0 - frac + LOGIT_EPS))
        return (
            torch.tensor(frac, dtype=DTYPE).unsqueeze(-1),
            torch.tensor(logit_np, dtype=DTYPE).unsqueeze(-1),
        )

    y_raw, y_logit = _to_frac_and_logit(
        experiments_df["yield_pct"].to_numpy(dtype=np.float64)
    )
    y_conversion_raw, y_conversion_logit = _to_frac_and_logit(
        experiments_df["conversion"].to_numpy(dtype=np.float64)
    )
    y_selectivity_raw, y_selectivity_logit = _to_frac_and_logit(
        experiments_df["selectivity"].to_numpy(dtype=np.float64)
    )

    return CellFeatures(
        X=torch.tensor(X_np, dtype=DTYPE),
        y_raw=y_raw,
        y_logit=y_logit,
        y_conversion_raw=y_conversion_raw,
        y_conversion_logit=y_conversion_logit,
        y_selectivity_raw=y_selectivity_raw,
        y_selectivity_logit=y_selectivity_logit,
        feature_dims=feature_dims,
        active_flag_dim=active_flag_dim,
        catalyst_ids=experiments_df["catalyst_id"].to_numpy(dtype=np.int64),
        replicate_id=experiments_df["replicate_id"].to_numpy(dtype=np.int64),
        pre_cat_form=pre_cat_form,
        feature_names=feature_names,
        form_id_dim=form_id_dim,
        cat_library=torch.tensor(cat_library, dtype=DTYPE),
    )
