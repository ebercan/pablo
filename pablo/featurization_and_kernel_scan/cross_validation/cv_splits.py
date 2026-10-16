"""Generate and serialise CV split indices for the featurization × kernel scan.

Six split blocks, all indexing into the row-order of ``experiments.parquet``
(69 rows; no NaN yields in the current table):

1. **Primary** — 10× repeated stratified catalyst-grouped 6-fold CV. Stratify by
   ``(pre_cat_form × yield_tercile)``. 60 folds.
2. **Leave-one-pre-cat-type-out** — one fold per pre-cat form (2 folds;
   "salt" and "co2_adduct" excluded due to >10 test points).
3. **Leave-one-scaffold-out** — one fold per NHC ring type (3 folds:
   imidazolylidene, triazolinylidene, thiazolylidene; "other" has no
   tested catalysts, "imidazolinylidene" excluded due to >10 test points).
4. **Leave-one-solvent-out** — one fold per unique solvent in the experiment
   table (7 folds; "THF" excluded due to >10 test points).
5. **Leave-one-replicate-out** — one fold per ``replicate_id`` ≥ 0
   (5 folds — four triplicates + one duplicate).
6. **Leave-one-out** — 60 folds. Non-replicate experiments get standard
   single-point holdout (55 folds). Replicate groups are held out as a unit
   (5 folds) so that no fold ever has one replicate in test and another in
   training.

Total per cell: **137 folds** (``config.FOLD_TOTAL``). Precomputed indices
live in ``data/grid/cv_splits.json`` (``paths.CV_SPLITS_JSON``) and are
SHA-256-hashed into every result file so that paired Wilcoxon-style
comparisons across cells remain valid.

Run with::

    python -m pablo.featurization_and_kernel_scan.cross_validation.cv_splits
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pablo.featurization_and_kernel_scan import paths
from pablo.utils.catalyst_form_classifier import (
    classify_library,
)


def generate_primary_cv_splits(
    experiments_df: pd.DataFrame,
    catalyst_library_df: pd.DataFrame,
    n_splits: int = 6,
    n_repeats: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Build the primary block: repeated stratified catalyst-grouped K-fold CV.

    Catalysts, not experiments, are assigned to folds, so all experiments of a
    catalyst share one test fold. Each catalyst gets the stratum
    ``<pre_cat_form>_<yield_tercile>``: the form from ``classify_library`` on
    its SMILES, the tercile from its mean ``yield_pct`` over the experiment
    table. Per repeat, catalysts are shuffled, sorted by stratum and dealt
    round-robin into ``n_splits`` folds.

    Args:
        experiments_df: Experiment table with ``catalyst_id`` and
            ``yield_pct`` columns and a 0-based ``RangeIndex``; returned
            indices are its index labels.
        catalyst_library_df: Catalyst table with ``id`` and ``smiles`` columns.
        n_splits: Number of folds per repeat.
        n_repeats: Number of repeats.
        seed: Seed of the ``np.random.RandomState`` used for all shuffles.

    Returns:
        ``n_repeats`` blocks ``{"split_type": "primary", "repeat": r,
        "folds": [...]}``, each holding ``n_splits`` folds
        ``{"train": [...], "test": [...], "left_out_catalysts": [...]}``.
    """
    rng = np.random.RandomState(seed)

    cat_ids = catalyst_library_df["id"].tolist()
    smiles_list = catalyst_library_df["smiles"].tolist()
    form_map = classify_library(cat_ids, smiles_list)

    cat_stats = experiments_df.groupby("catalyst_id")["yield_pct"].mean().reset_index()
    cat_stats["form"] = cat_stats["catalyst_id"].map(form_map)

    terciles = pd.qcut(cat_stats["yield_pct"], 3, labels=["low", "mid", "high"])
    cat_stats["tercile"] = terciles
    cat_stats["stratum"] = (
        cat_stats["form"].astype(str) + "_" + cat_stats["tercile"].astype(str)
    )

    splits = []

    for r in range(n_repeats):
        folds: dict[int, list] = {i: [] for i in range(n_splits)}

        # Shuffle within each stratum, then round-robin into folds
        shuffled = cat_stats.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        shuffled = shuffled.sort_values("stratum")

        fold_idx = 0
        for _, row in shuffled.iterrows():
            folds[fold_idx].append(int(row["catalyst_id"]))
            fold_idx = (fold_idx + 1) % n_splits

        fold_splits = []
        for i in range(n_splits):
            test_cats = set(folds[i])
            train_idx = (
                experiments_df.index[~experiments_df["catalyst_id"].isin(test_cats)]
                .tolist()
            )
            test_idx = (
                experiments_df.index[experiments_df["catalyst_id"].isin(test_cats)]
                .tolist()
            )
            fold_splits.append({
                "train": train_idx,
                "test": test_idx,
                "left_out_catalysts": list(test_cats)
            })

        splits.append({
            "split_type": "primary",
            "repeat": r,
            "folds": fold_splits
        })

    return splits


def generate_type_cv_splits(exp_df: pd.DataFrame, cat_df: pd.DataFrame) -> list[dict]:
    """Build the leave-one-pre-cat-type-out block.

    Classifies every catalyst with ``classify_library`` and makes one fold per
    pre-catalyst form present in the experiment table, except ``"salt"`` and
    ``"co2_adduct"``, which would each put more than 10 experiments in test.

    Args:
        exp_df: Experiment table with a ``catalyst_id`` column and a 0-based
            ``RangeIndex``.
        cat_df: Catalyst table with ``id`` and ``smiles`` columns.

    Returns:
        One block ``{"split_type": "leave_one_type_out", "repeat": 0,
        "folds": [...]}`` whose folds are
        ``{"train": [...], "test": [...], "left_out_type": form}``.
    """
    cat_ids = cat_df["id"].tolist()
    smiles_list = cat_df["smiles"].tolist()
    form_map = classify_library(cat_ids, smiles_list)

    exp_df_forms = exp_df["catalyst_id"].map(form_map)
    forms = exp_df_forms.unique()

    splits = []
    for form in forms:
        if str(form) in ["salt", "co2_adduct"]:
            continue
        test_idx = exp_df.index[exp_df_forms == form].tolist()
        train_idx = exp_df.index[exp_df_forms != form].tolist()
        splits.append({"train": train_idx, "test": test_idx, "left_out_type": str(form)})

    return [{"split_type": "leave_one_type_out", "repeat": 0, "folds": splits}]


def generate_scaffold_cv_splits(
    exp_df: pd.DataFrame, featurizations_root: Path
) -> list[dict]:
    """Build the leave-one-scaffold-out block from the interpretable ring one-hots.

    Reads the one-hot columns ``ring_<label>`` for the five ring types
    (imidazolylidene / imidazolinylidene / triazolinylidene / thiazolylidene /
    other) from ``interpretable/descriptors.parquet``, takes the argmax per
    catalyst, joins on ``catalyst_id`` and makes one fold per ring type present
    in the experiment table, except ``"imidazolinylidene"``, which would put
    more than 10 experiments in test.

    Args:
        exp_df: Experiment table with a ``catalyst_id`` column and a 0-based
            ``RangeIndex``.
        featurizations_root: Directory holding the featurizations
            (``paths.FEAT_DIR``); the descriptor table is indexed by catalyst id.

    Returns:
        One block ``{"split_type": "leave_one_scaffold_out", "repeat": 0,
        "folds": [...]}`` whose folds are
        ``{"train": [...], "test": [...], "left_out_scaffold": ring_type}``.

    Raises:
        KeyError: If any catalyst in ``exp_df`` has no row in the interpretable
            descriptor table.
    """
    interp_df = pd.read_parquet(
        featurizations_root / "interpretable" / "descriptors.parquet"
    )
    ring_labels = ["imidazolylidene", "imidazolinylidene", "triazolinylidene",
                   "thiazolylidene", "other"]
    ring_cols = [f"ring_{label}" for label in ring_labels]
    ring_by_cat = pd.Series(
        np.array(ring_labels)[interp_df[ring_cols].to_numpy().argmax(axis=1)],
        index=interp_df.index,
    ).to_dict()
    exp_scaffolds = exp_df["catalyst_id"].map(ring_by_cat)

    if exp_scaffolds.isna().any():
        missing = exp_df.loc[exp_scaffolds.isna(), "catalyst_id"].unique().tolist()
        raise KeyError(
            f"ring_type missing for catalyst_ids: {missing}. "
            "Regenerate interpretable/descriptors.parquet."
        )

    splits = []
    for scaffold in exp_scaffolds.unique():
        if str(scaffold) == "imidazolinylidene":
            continue
        test_idx = exp_df.index[exp_scaffolds == scaffold].tolist()
        train_idx = exp_df.index[exp_scaffolds != scaffold].tolist()
        splits.append({
            "train": train_idx,
            "test": test_idx,
            "left_out_scaffold": str(scaffold),
        })

    return [{"split_type": "leave_one_scaffold_out", "repeat": 0, "folds": splits}]


def generate_solvent_cv_splits(exp_df: pd.DataFrame) -> list[dict]:
    """Build the leave-one-solvent-out block.

    One fold per solvent in the experiment table, except ``"THF"``, which would
    put more than 10 experiments in test.

    Args:
        exp_df: Experiment table with a ``solvent`` column and a 0-based
            ``RangeIndex``.

    Returns:
        One block ``{"split_type": "leave_one_solvent_out", "repeat": 0,
        "folds": [...]}`` whose folds are
        ``{"train": [...], "test": [...], "left_out_solvent": solvent}``.
    """
    solvents = exp_df["solvent"].unique()
    splits = []
    for solv in solvents:
        if str(solv) == "THF":
            continue
        test_idx = exp_df.index[exp_df["solvent"] == solv].tolist()
        train_idx = exp_df.index[exp_df["solvent"] != solv].tolist()
        splits.append({"train": train_idx, "test": test_idx, "left_out_solvent": str(solv)})

    return [{"split_type": "leave_one_solvent_out", "repeat": 0, "folds": splits}]


def generate_loo_cv_splits(exp_df: pd.DataFrame) -> list[dict]:
    """Build the leave-one-out block, holding out replicate groups as a unit.

    Rows with ``replicate_id`` = -1 each get a single-point fold. All rows of a
    replicate group (``replicate_id`` ≥ 0) form one fold, emitted at the
    group's first row, so no replicate is in train while another is in test.

    Args:
        exp_df: Experiment table with a ``replicate_id`` column and a 0-based
            ``RangeIndex``.

    Returns:
        One block ``{"split_type": "leave_one_out", "repeat": 0,
        "folds": [...]}`` whose folds are ``{"train": [...], "test": [...],
        "left_out_idx": i}``, with ``i`` the first row of the held-out unit.
    """
    replicate_ids = exp_df["replicate_id"]
    rep_groups: dict[int, list[int]] = {}
    for idx, rid in replicate_ids.items():
        if rid >= 0:
            rep_groups.setdefault(int(rid), []).append(int(idx))

    seen_rep_ids: set[int] = set()
    splits = []
    all_idx = set(range(len(exp_df)))

    for i in range(len(exp_df)):
        rid = int(replicate_ids.iloc[i])
        if rid >= 0:
            if rid in seen_rep_ids:
                continue
            seen_rep_ids.add(rid)
            test_idx = rep_groups[rid]
        else:
            test_idx = [i]
        train_idx = sorted(all_idx - set(test_idx))
        splits.append({"train": train_idx, "test": test_idx, "left_out_idx": i})

    return [{"split_type": "leave_one_out", "repeat": 0, "folds": splits}]


def generate_lot_cv_splits(exp_df: pd.DataFrame) -> list[dict]:
    """Build the leave-one-replicate-out block.

    One fold per replicate group (``replicate_id`` ≥ 0); the test set is the
    whole group and the training set is every other row, including the
    non-replicate rows.

    Args:
        exp_df: Experiment table with a ``replicate_id`` column (-1 for
            non-replicates) and a 0-based ``RangeIndex``.

    Returns:
        One block ``{"split_type": "leave_one_replicate_out", "repeat": 0,
        "folds": [...]}`` whose folds are ``{"train": [...], "test": [...],
        "left_out_replicate_id": rep_id}``.
    """
    splits = []
    replicate_groups = exp_df["replicate_id"].unique()
    for rep_id in replicate_groups:
        if rep_id == -1:
            continue
        test_idx = exp_df.index[exp_df["replicate_id"] == rep_id].tolist()
        train_idx = exp_df.index[exp_df["replicate_id"] != rep_id].tolist()
        splits.append({"train": train_idx, "test": test_idx, "left_out_replicate_id": int(rep_id)})

    return [{"split_type": "leave_one_replicate_out", "repeat": 0, "folds": splits}]


def generate_all_cv_splits(libraries_dir: Path, output_path: Path) -> None:
    """Write every CV split block for the scan to ``output_path``.

    Args:
        libraries_dir: Directory holding ``experiments.parquet`` and
            ``catalysts.parquet`` (``paths.LIB_DIR``).
        output_path: Destination JSON (``paths.CV_SPLITS_JSON``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_df = pd.read_parquet(libraries_dir / "experiments.parquet").reset_index(drop=True)
    cat_df = pd.read_parquet(libraries_dir / "catalysts.parquet")
    featurizations_root = paths.FEAT_DIR

    splits = []
    splits.extend(generate_primary_cv_splits(exp_df, cat_df))
    splits.extend(generate_type_cv_splits(exp_df, cat_df))
    splits.extend(generate_scaffold_cv_splits(exp_df, featurizations_root))
    splits.extend(generate_solvent_cv_splits(exp_df))
    splits.extend(generate_lot_cv_splits(exp_df))
    splits.extend(generate_loo_cv_splits(exp_df))

    output_path.write_text(json.dumps(splits, indent=2))

    total_folds = sum(len(r["folds"]) for r in splits)
    print(f"Wrote {len(splits)} split blocks containing {total_folds} total folds → {output_path}")


if __name__ == "__main__":
    generate_all_cv_splits(paths.LIB_DIR, paths.CV_SPLITS_JSON)
    print("CV splits generated.")
