"""Assemblers for the four interpretable catalyst featurizations.

Each function writes one featurization's ``descriptors.parquet`` from the
intermediate caches that ``xtb_pipeline`` and ``descriptors`` populate:

    interpretable_steric  10-dim permutation-invariant N/C substituent aggregates
    interpretable_light   21-dim structural indices + LP Sterimol + xTB scalars
    interpretable         30-dim the above + quadrant %Vbur + C2 electronics
    interpretable_dG      30-dim as above with the C2 deprotonation free energy

No dimensionality reduction is applied; the written matrix is the deliverable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
    HARTREE_TO_KCAL,
    RING_TYPE_LABELS,
    STRUCTURAL_FEATURE_COLS,
    XTB_CHECKPOINT,
    classify_ring_type,
    collect_per_catalyst_jsons,
    has_unsaturated_backbone,
    load_intermediate,
    rdkit_scalar_descriptors,
    ring_type_to_index,
)
from pablo.utils.catalyst_form_classifier import (
    FORM_LABELS,
    classify_library,
    form_to_index,
)

def generate_structural() -> None:
    """Write structural descriptors to ``_intermediate/structural_onehots/``.

    Per catalyst: the pre-catalyst form and ring type as labels and as integer
    ids (``cat_form_id``, ``ring_type_id``), ``backbone_saturated``, and the
    RDKit scalar descriptors. The integer ids are expanded to one-hot columns
    at assembly time by ``expand_structural_categoricals``.
    """
    out_dir = paths.INTERMEDIATE_DIR / "structural_onehots"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_df = pd.read_parquet(paths.CATALYSTS_PARQUET)
    exp_df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    cat_ids = cat_df["id"].tolist()
    smiles_list = cat_df["smiles"].tolist()
    tested_ids = set(exp_df["catalyst_id"].unique())

    print("Classifying pre-catalyst forms ...")
    form_map = classify_library(cat_ids, smiles_list, tested_ids=tested_ids)

    rows = []
    for cid, smi in zip(cat_ids, smiles_list):
        form = form_map[cid]
        ring_type = classify_ring_type(smi)
        rows.append(
            {
                "catalyst_id": cid,
                "form": form,
                "ring_type": ring_type,
                "cat_form_id": form_to_index(form),
                "ring_type_id": ring_type_to_index(ring_type),
                "backbone_saturated": 0 if has_unsaturated_backbone(smi) else 1,
                **rdkit_scalar_descriptors(smi),
            }
        )
    df = pd.DataFrame(rows).set_index("catalyst_id")

    print(f"\nForm distribution:")
    for form in (*FORM_LABELS, "unknown"):
        print(f"  {form:<15}: {(df['form'] == form).sum()}")
    print(f"Ring type distribution:")
    for rt in RING_TYPE_LABELS:
        print(f"  {rt:<25}: {(df['ring_type'] == rt).sum()}")

    df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_molecules": len(df),
                "feature_cols": [
                    c for c in df.columns if c not in ("form", "ring_type")
                ],
                "note": "Structural-only stage. Consumed by --assemble-light / --assemble-full.",
            },
            indent=2,
        )
    )
    print(f"\n✓ Structural descriptors → {out_dir.relative_to(paths.PKG_ROOT)}")


def tally_topology() -> None:
    """Print ring-topology counts across the catalyst library.

    Breaks the 161-catalyst library out by tested (appears in
    experiments.parquet) vs scope-only, per ring type, and lists the tested
    catalyst ids of each ring type.
    """
    cat_df = pd.read_parquet(paths.CATALYSTS_PARQUET)
    exp_df = pd.read_parquet(paths.EXPERIMENTS_PARQUET)
    tested_ids = set(exp_df["catalyst_id"].unique())

    rows = []
    for cid, smi in zip(cat_df["id"].tolist(), cat_df["smiles"].tolist()):
        rows.append(
            {
                "catalyst_id": cid,
                "ring_type": classify_ring_type(smi),
                "tested": cid in tested_ids,
            }
        )
    df = pd.DataFrame(rows)

    n_total = len(df)
    n_tested = int(df["tested"].sum())
    n_scope = n_total - n_tested

    print(f"\nCatalyst library ring-topology tally")
    print(f"  total catalysts : {n_total}")
    print(f"  tested (in DoE) : {n_tested}")
    print(f"  scope-only      : {n_scope}")
    print()
    header = f"  {'ring_type':<22} {'tested':>8} {'scope-only':>12} {'total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rt in RING_TYPE_LABELS:
        mask = df["ring_type"] == rt
        n_t = int((mask & df["tested"]).sum())
        n_s = int((mask & ~df["tested"]).sum())
        print(f"  {rt:<22} {n_t:>8} {n_s:>12} {n_t + n_s:>8}")
    print()

    tested_ids_per_type = (
        df[df["tested"]].groupby("ring_type")["catalyst_id"].apply(sorted).to_dict()
    )
    print("Tested catalyst IDs by ring type:")
    for rt in RING_TYPE_LABELS:
        ids = tested_ids_per_type.get(rt, [])
        if ids:
            print(f"  {rt:<22} {ids}")
    print()


# ─── Featurization assembly column contracts ────────────────────────────────

FORM_ONEHOT_COLS = [f"form_{label}" for label in FORM_LABELS]
"""One-hot expansion of ``cat_form_id``: salt, free_nhc, co2_adduct, bicarbonate."""

RING_ONEHOT_COLS = [f"ring_{label}" for label in RING_TYPE_LABELS]
"""One-hot expansion of ``ring_type_id``: the five NHC ring topologies."""


def expand_structural_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the two integer category columns with their one-hot expansions.

    ``cat_form_id`` and ``ring_type_id`` are unordered categories stored as
    contiguous integers. Every kernel in the scan measures a real-valued
    distance over the raw feature columns — there is no Hamming/categorical
    route — so an integer ``ring_type_id`` would put "thiazolylidene = 3"
    three times as far from "imidazolylidene = 0" as "imidazolinylidene = 1"
    is. One-hot columns put every category at the same distance from every
    other.

    Args:
        df: Catalyst-indexed frame carrying ``cat_form_id`` and
            ``ring_type_id`` as integer columns.

    Returns:
        A copy of ``df`` with those two columns replaced, at their original
        positions, by ``FORM_ONEHOT_COLS`` and ``RING_ONEHOT_COLS``. ``df``
        is not modified.

    Raises:
        ValueError: If either id column is absent or carries an out-of-range
            value.
    """
    out = df.copy()
    for id_col, labels, names in (
        ("cat_form_id", FORM_LABELS, FORM_ONEHOT_COLS),
        ("ring_type_id", RING_TYPE_LABELS, RING_ONEHOT_COLS),
    ):
        if id_col not in out.columns:
            raise ValueError(f"{id_col!r} absent; cannot expand to one-hot.")
        ids = out[id_col].to_numpy(dtype=np.int64)
        if ids.min() < 0 or ids.max() >= len(labels):
            raise ValueError(
                f"{id_col} carries values outside [0, {len(labels) - 1}]: "
                f"{sorted(set(ids.tolist()))}"
            )
        onehot = np.zeros((len(ids), len(labels)), dtype=np.int64)
        onehot[np.arange(len(ids)), ids] = 1
        position = out.columns.get_loc(id_col)
        out = out.drop(columns=[id_col])
        for offset, name in enumerate(names):
            out.insert(position + offset, name, onehot[:, offset])
    return out


_LIGHT_COLS = [
    *FORM_ONEHOT_COLS,
    *RING_ONEHOT_COLS,
    "backbone_saturated",
    "lp_L",
    "lp_B1",
    "lp_B5",
    "vbur_scalar",
    "vbur_asymmetry",
    "xtb_homo_eV",
    "xtb_lumo_eV",
    "xtb_gap_eV",
    "xtb_dipole_D",
    "mw",
    "heavy_atom_count",
]
assert len(_LIGHT_COLS) == 21

_FULL_COLS = [
    *FORM_ONEHOT_COLS,
    *RING_ONEHOT_COLS,
    "backbone_saturated",
    "lp_L",
    "lp_B1",
    "lp_B5",
    "vbur_q1",
    "vbur_q2",
    "vbur_q3",
    "vbur_q4",
    "fukui_plus_c2",
    "fukui_minus_c2",
    "homo_weight_c2",
    "mulliken_charge_c2",
    "delta_e_h_plus",
    "delta_e_bh3",
    "xtb_homo_eV",
    "xtb_lumo_eV",
    "xtb_gap_eV",
    "xtb_dipole_D",
    "xtb_polarizability_angst3",
    "mw",
    "heavy_atom_count",
]
assert len(_FULL_COLS) == 30

_DG_COLS = [c for c in _FULL_COLS if c != "delta_e_h_plus"] + ["xtb_deltaG_deprot"]
"""``interpretable_dG`` column set: ``interpretable`` with ``delta_e_h_plus``
*replaced* by the xTB C2 deprotonation free energy, not joined by it.

The two are the same quantity at different levels of care — ``delta_e_h_plus``
is an electronic protonation energy read off a single carbene-derived geometry,
``xtb_deltaG_deprot`` is a Boltzmann free energy over independently embedded
conformer ensembles with RRHO thermal terms — and they correlate at r = -0.96
across the library. Carrying both would put two near-collinear columns in the
same GP block and, worse, would turn the head-to-head against ``interpretable``
into "does ΔG add anything on top of ΔE", which a 0.96 correlation answers
"no" almost by construction. Swapping instead makes the two featurizations
differ in exactly one column and asks the question actually worth answering:
whether the conformer ensemble and the thermal correction are worth their
cost."""



def assemble_steric() -> None:
    """Write ``data/featurizations/interpretable_steric/`` from the aggregate cache."""
    src = paths.INTERMEDIATE_DIR / "n_c_aggregates.parquet"
    assert src.exists(), f"Missing intermediate: {src}. Run --n-c-aggregates first."
    df = pd.read_parquet(src)
    assert df.isna().sum().sum() == 0, "NaN in N/C-aggregate cache."
    assert df.shape[1] == 10, f"Expected 10 cols, got {df.shape[1]}."

    out_dir = paths.FEAT_DIR / "interpretable_steric"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_molecules": len(df),
                "n_features": int(df.shape[1]),
                "feature_cols": list(df.columns),
                "descriptors_file": "descriptors.parquet",
                "method": (
                    "Permutation-invariant aggregates over ring-N and ring-C "
                    "exocyclic substituents. Each block = "
                    "(card, max L, sum L, max B5, min B1). Empty multiset → 0. "
                    "Topology-agnostic; well-defined on all ring types."
                ),
            },
            indent=2,
        )
    )
    print(
        f"✓ interpretable_steric ({df.shape[1]} cols × {len(df)} rows) → {out_dir.relative_to(paths.PKG_ROOT)}"
    )


def assemble_light() -> None:
    """Write ``data/featurizations/interpretable_light/`` (21 dims).

    Combines structural one-hots, LP-axis Sterimol, scalar + asymmetry
    %Vbur (derived from quadrant %Vbur), molecule-global xTB scalars
    (HOMO, LUMO, gap, dipole), and RDKit trivial scalars.
    """
    struct_path = paths.INTERMEDIATE_DIR / "structural_onehots" / "descriptors.parquet"
    assert struct_path.exists(), f"Missing: {struct_path}. Run --structural first."
    struct_df = pd.read_parquet(struct_path)

    for c in STRUCTURAL_FEATURE_COLS:
        assert c in struct_df.columns, f"Missing column {c!r} in structural parquet."
    struct_df = expand_structural_categoricals(struct_df[STRUCTURAL_FEATURE_COLS])

    lp_path = paths.INTERMEDIATE_DIR / "lp_sterimol.parquet"
    assert lp_path.exists(), f"Missing: {lp_path}. Run --lp-sterimol first."
    lp_df = pd.read_parquet(lp_path)

    vbur_path = paths.INTERMEDIATE_DIR / "vbur_quadrants.parquet"
    assert vbur_path.exists(), f"Missing: {vbur_path}. Run --vbur-quadrants first."
    vbur_df = pd.read_parquet(vbur_path)
    vbur_derived = pd.DataFrame(index=vbur_df.index)
    q_vals = vbur_df[["vbur_q1", "vbur_q2", "vbur_q3", "vbur_q4"]]
    vbur_derived["vbur_scalar"] = q_vals.mean(axis=1)
    vbur_derived["vbur_asymmetry"] = q_vals.max(axis=1) - q_vals.min(axis=1)

    assert XTB_CHECKPOINT.exists(), f"Missing: {XTB_CHECKPOINT}. Run --xtb first."
    xtb_df = pd.read_parquet(XTB_CHECKPOINT)
    xtb_light_cols = ["xtb_homo_eV", "xtb_lumo_eV", "xtb_gap_eV", "xtb_dipole_D"]
    xtb_df = xtb_df[xtb_light_cols]

    df = struct_df.join(lp_df).join(vbur_derived).join(xtb_df)
    df = df[_LIGHT_COLS]

    assert df.shape[1] == 21, f"Expected 21 cols, got {df.shape[1]}."
    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        nan_cols = {c: df[c].isna().sum() for c in df.columns if df[c].isna().any()}
        missing_cats = df[df.isna().any(axis=1)].index.tolist()
        assert False, (
            f"{n_nan} NaN values in assembled interpretable_light table.\n"
            f"  Columns with NaN (col: count): {nan_cols}\n"
            f"  Catalyst IDs with NaN ({len(missing_cats)} total): {missing_cats}\n"
            f"  Check that --structural, --lp-sterimol, --vbur-quadrants, --xtb "
            f"completed for all 161 catalysts."
        )

    out_dir = paths.FEAT_DIR / "interpretable_light"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_molecules": len(df),
                "n_features": int(df.shape[1]),
                "feature_cols": list(df.columns),
                "descriptors_file": "descriptors.parquet",
                "method": (
                    "Structural one-hots + LP-axis Sterimol + scalar %Vbur + "
                    "%Vbur asymmetry + molecule-global xTB scalars (HOMO, LUMO, "
                    "gap, dipole) + RDKit trivial scalars. No carbene-localised "
                    "electronics."
                ),
            },
            indent=2,
        )
    )
    print(
        f"✓ interpretable_light ({df.shape[1]} cols × {len(df)} rows) → {out_dir.relative_to(paths.PKG_ROOT)}"
    )


def _build_full_frame() -> pd.DataFrame:
    """Assemble the 30-column interpretable descriptor table for ``assemble_full``.

    Joins the structural descriptors, LP-axis Sterimol, quadrant %Vbur, xTB
    scalars and the per-catalyst probe JSONs from ``_intermediate/``.

    Returns:
        pd.DataFrame: Catalyst-indexed table with exactly ``_FULL_COLS``.
    """
    struct_path = paths.INTERMEDIATE_DIR / "structural_onehots" / "descriptors.parquet"
    assert struct_path.exists(), f"Missing: {struct_path}. Run --structural first."
    struct_df = pd.read_parquet(struct_path)

    for c in STRUCTURAL_FEATURE_COLS:
        assert c in struct_df.columns, f"Missing column {c!r} in structural parquet."
    struct_df = expand_structural_categoricals(struct_df[STRUCTURAL_FEATURE_COLS])

    lp_path = paths.INTERMEDIATE_DIR / "lp_sterimol.parquet"
    assert lp_path.exists(), f"Missing: {lp_path}. Run --lp-sterimol first."
    lp_df = pd.read_parquet(lp_path)

    vbur_path = paths.INTERMEDIATE_DIR / "vbur_quadrants.parquet"
    assert vbur_path.exists(), f"Missing: {vbur_path}. Run --vbur-quadrants first."
    vbur_df = pd.read_parquet(vbur_path)

    assert XTB_CHECKPOINT.exists(), f"Missing: {XTB_CHECKPOINT}. Run --xtb first."
    xtb_df = pd.read_parquet(XTB_CHECKPOINT)

    xtb_extra = collect_per_catalyst_jsons("xtb_extra")
    assert len(xtb_extra) > 0, "No xtb_extra JSONs found. Run --xtb-extra first."
    extra_df = pd.DataFrame.from_dict(xtb_extra, orient="index")
    extra_df.index.name = "catalyst_id"

    h_plus = collect_per_catalyst_jsons("probe_h_plus")
    assert len(h_plus) > 0, "No probe_h_plus JSONs found. Run --probe-h-plus first."
    h_plus_df = pd.DataFrame.from_dict(h_plus, orient="index")[["delta_e_h_plus"]]
    h_plus_df.index.name = "catalyst_id"

    bh3_raw = collect_per_catalyst_jsons("probe_bh3")
    assert len(bh3_raw) > 0, "No probe_bh3 JSONs found. Run --probe-bh3 first."

    ref_path = paths.INTERMEDIATE_DIR / "bh3_reference.json"
    assert ref_path.exists(), f"Missing: {ref_path}. Run --bh3-reference first."
    e_bh3 = json.loads(ref_path.read_text())["e_bh3_hartree"]

    bh3_delta = {}
    for cid, d in bh3_raw.items():
        delta = (
            d["e_adduct_bh3_hartree"] - d["e_carbene_hartree"] - e_bh3
        ) * HARTREE_TO_KCAL
        bh3_delta[cid] = {"delta_e_bh3": delta}
    bh3_df = pd.DataFrame.from_dict(bh3_delta, orient="index")
    bh3_df.index.name = "catalyst_id"

    df = (
        struct_df.join(lp_df)
        .join(vbur_df)
        .join(extra_df)
        .join(h_plus_df)
        .join(bh3_df)
        .join(xtb_df)
    )
    df = df[_FULL_COLS]

    assert df.shape[1] == 30, f"Expected 30 cols, got {df.shape[1]}."
    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        nan_cols = {c: df[c].isna().sum() for c in df.columns if df[c].isna().any()}
        missing_cats = df[df.isna().any(axis=1)].index.tolist()
        assert False, (
            f"{n_nan} NaN values in assembled interpretable table.\n"
            f"  Columns with NaN (col: count): {nan_cols}\n"
            f"  Catalyst IDs with NaN ({len(missing_cats)} total): {missing_cats}\n"
            f"  Check that --xtb-extra, --probe-h-plus, --probe-bh3 completed "
            f"for all 161 catalysts."
        )
    return df


def assemble_full() -> None:
    """Write ``data/featurizations/interpretable/`` (30 dims).

    Combines structural one-hots, LP-axis Sterimol, quadrant %Vbur,
    carbene-localised electronics (Fukui, HOMO weight, Mulliken charge,
    probe-binding ΔE), xTB molecular scalars, and RDKit trivial scalars.
    """
    df = _build_full_frame()

    out_dir = paths.FEAT_DIR / "interpretable"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_molecules": len(df),
                "n_features": int(df.shape[1]),
                "feature_cols": list(df.columns),
                "descriptors_file": "descriptors.parquet",
                "method": (
                    "Structural one-hots + LP-axis Sterimol + quadrant %Vbur + "
                    "carbene-localised electronics (Fukui, HOMO weight, Mulliken "
                    "charge, probe-binding ΔE) + xTB molecular scalars + RDKit "
                    "trivial scalars."
                ),
            },
            indent=2,
        )
    )
    print(
        f"✓ interpretable ({df.shape[1]} cols × {len(df)} rows) → {out_dir.relative_to(paths.PKG_ROOT)}"
    )


def assemble_dG() -> None:
    """Write ``data/featurizations/interpretable_dG/`` (30 dims).

    The interpretable block with ``delta_e_h_plus`` swapped for
    ``xtb_deltaG_deprot``, the xTB C2 deprotonation free energy (see
    ``_DG_COLS`` for why it is a swap and not an addition). Catalysts whose ΔG
    run failed QC are reported rather than imputed, so the assembly stops
    instead of silently filling a column that is only meaningful as a relative
    quantity across the library.

    Catalysts that were not deprotonated from the ordinary azolium(+1) →
    carbene(0) ladder are printed as a warning block. Their values are correct
    first-deprotonation free energies but carry the Coulomb work of a different
    charge state, which is worth tens of kcal/mol and is not basicity.

    The 29 shared columns are read from the assembled ``interpretable`` table
    rather than re-derived from ``_intermediate/``. That guarantees the two
    featurizations are identical wherever they overlap, and it means this step
    does not depend on the Stage-2 per-catalyst JSONs still being on disk —
    those are large, disposable, and routinely absent once a sweep has been
    assembled.
    """
    src = paths.FEAT_DIR / "interpretable" / "descriptors.parquet"
    assert src.exists(), f"Missing {src}. Run --assemble-full first."
    df = pd.read_parquet(src)
    assert list(df.columns) == _FULL_COLS, (
        f"{src} has columns {list(df.columns)}, expected {_FULL_COLS}. "
        "Re-run --assemble-full."
    )

    dg_raw = collect_per_catalyst_jsons("deltaG_deprot")
    assert len(dg_raw) > 0, "No deltaG_deprot JSONs found. Run --deltaG-deprot first."

    failed = {cid: d["failures"] for cid, d in dg_raw.items() if d["xtb_deltaG_deprot"] is None}
    dg_df = pd.DataFrame.from_dict(
        {
            cid: {"xtb_deltaG_deprot": d["xtb_deltaG_deprot"]}
            for cid, d in dg_raw.items()
            if d["xtb_deltaG_deprot"] is not None
        },
        orient="index",
    )
    dg_df.index.name = "catalyst_id"

    df = df.join(dg_df)
    df = df[_DG_COLS]

    assert df.shape[1] == 30, f"Expected 30 cols, got {df.shape[1]}."
    missing = df[df["xtb_deltaG_deprot"].isna()].index.tolist()
    if missing:
        assert False, (
            f"xtb_deltaG_deprot missing for {len(missing)} catalysts: {missing}\n"
            f"  QC failures ({len(failed)}): { {k: v for k, v in failed.items()} }\n"
            f"  Catalysts with no cached JSON: {sorted(set(missing) - set(dg_raw))}\n"
            f"  Values are logged, never imputed — rerun --deltaG-deprot --cat-id <id> "
            f"for the missing catalysts."
        )

    # Derived here, not read from the JSON: the flag was added to the writer
    # after the first full library run, so cached JSONs predate the key while
    # every one of them records the two species charges it is computed from.
    nonstandard = sorted(
        cid
        for cid, d in dg_raw.items()
        if (d["azolium"]["charge"], d["carbene"]["charge"]) != (1, 0)
    )
    if nonstandard:
        print(
            f"  [charge state] {len(nonstandard)} catalysts were not deprotonated "
            f"from azolium(+1) → carbene(0):"
        )
        for cid in nonstandard:
            d = dg_raw[cid]
            print(
                f"    C{cid}: {d['azolium']['charge']:+d} → {d['carbene']['charge']:+d}, "
                f"ΔG = {d['xtb_deltaG_deprot']:.1f} kcal/mol"
            )
        print(
            "    These are correct first-deprotonation free energies, but they sit "
            "on a different charge ladder and are shifted by Coulomb work, not basicity."
        )

    seeded = sorted(
        cid
        for cid, d in dg_raw.items()
        if d["azolium"].get("conformer_source") != "etkdgv3"
        or d["carbene"].get("conformer_source") != "etkdgv3"
    )
    if seeded:
        print(
            f"  [conformer source] {len(seeded)} catalysts did not use the standard "
            f"ETKDG ensemble: {seeded}. The descriptor is only meaningful as a "
            f"relative quantity, so a per-catalyst protocol difference is a caveat "
            f"on those rows."
        )

    out_dir = paths.FEAT_DIR / "interpretable_dG"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "descriptors.parquet")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_molecules": len(df),
                "n_features": int(df.shape[1]),
                "feature_cols": list(df.columns),
                "descriptors_file": "descriptors.parquet",
                "method": (
                    "The interpretable block with delta_e_h_plus replaced by "
                    "xtb_deltaG_deprot = G(carbene) − G(azolium) at "
                    "GFN2/ALPB(toluene), kcal/mol. Both species independently "
                    "embedded (ETKDGv3, MMFF-pruned), --ohess on every conformer, "
                    "Boltzmann-weighted over distinct minima at 298.15 K; the "
                    "free-proton term is dropped as a library-wide constant. The "
                    "swap is deliberate: the two columns are the same quantity at "
                    "different levels of care and correlate at r = -0.96, so this "
                    "featurization differs from interpretable in exactly one column. "
                    "NOTE: only xtb_deltaG_deprot is conformer-ensemble averaged — "
                    "the other 22 columns come from the single stored xTB geometry."
                ),
                "charge_state_nonstandard": nonstandard,
                "nonstandard_conformer_source": seeded,
            },
            indent=2,
        )
    )
    print(
        f"✓ interpretable_dG ({df.shape[1]} cols × {len(df)} rows) → {out_dir.relative_to(paths.PKG_ROOT)}"
    )
