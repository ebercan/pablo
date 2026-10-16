"""xTB log parsing takes values from the converged geometry, and the stored
scalars agree with a fresh re-parse of the saved logs.

The logs under ``paths.XTB_OUTPUTS_DIR`` are git-ignored, so these tests skip
on a checkout that does not carry them.
"""

import re

import pandas as pd
import pytest

from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.features.interpretable.canonicalize_duplicates import (
    FREE_CARBENE_COLS,
    duplicate_groups,
)
from pablo.featurization_and_kernel_scan.features.interpretable.repair_xtb_scalars import (
    AFFECTED_COLS,
    FEATURIZATIONS,
    reparse_logs,
)
from pablo.featurization_and_kernel_scan.features.interpretable.xtb_pipeline import (
    XTB_CHECKPOINT,
    _parse_xtb_output,
)

_GAP = re.compile(r"HOMO-LUMO [Gg][Aa][Pp]\s+([0-9.\-]+)\s+eV")
_HOMO = re.compile(r"([0-9.\-]+)\s+\(HOMO\)")
_LUMO = re.compile(r"([0-9.\-]+)\s+\(LUMO\)")
_MARK = "GEOMETRY OPTIMIZATION CONVERGED"


def _logs() -> list:
    logs = sorted(paths.XTB_OUTPUTS_DIR.glob("C*.txt"))
    if not logs:
        pytest.skip(f"no xTB logs under {paths.XTB_OUTPUTS_DIR} (git-ignored)")
    return logs


def test_parser_reads_the_converged_block_not_the_input_geometry():
    """On a log with several property blocks, HOMO/LUMO/gap come from the last one."""
    n_multi = 0
    for log in _logs():
        text = log.read_text()
        gaps = [float(v) for v in _GAP.findall(text)]
        homos = [float(v) for v in _HOMO.findall(text)]
        lumos = [float(v) for v in _LUMO.findall(text)]
        parsed = _parse_xtb_output(text)
        assert parsed is not None, log
        assert parsed["xtb_gap_eV"] == pytest.approx(gaps[-1], abs=1e-9), log
        assert parsed["xtb_homo_eV"] == pytest.approx(homos[-1], abs=1e-9), log
        assert parsed["xtb_lumo_eV"] == pytest.approx(lumos[-1], abs=1e-9), log
        if len(gaps) > 1 and abs(gaps[0] - gaps[-1]) > 1e-6:
            n_multi += 1
            assert parsed["xtb_gap_eV"] != pytest.approx(gaps[0], abs=1e-9), log
    assert n_multi > 0, "no log exercised the multi-block case; the guard is vacuous"


def test_parser_only_sees_text_after_the_final_convergence_marker():
    """Every parsed value is reproducible from the tail of the log alone."""
    for log in _logs():
        text = log.read_text()
        full = _parse_xtb_output(text)
        tail = _parse_xtb_output(text[text.rfind(_MARK) :])
        assert full == tail, log


def test_dipole_is_the_dipole_line_not_the_quadrupole_line():
    """The dipole regex must not fall through to the six-number quadrupole ``full:`` line."""
    for log in _logs():
        text = log.read_text()
        tail = text[text.rfind(_MARK) :]
        m = re.search(r"molecular dipole:.*?full:\s+\S+\s+\S+\s+\S+\s+([0-9.]+)", tail, re.S)
        assert m is not None, log
        parsed = _parse_xtb_output(text)
        assert parsed["xtb_dipole_D"] == pytest.approx(float(m.group(1)), abs=1e-9), log
        assert parsed["xtb_dipole_D"] < 20.0, log


def test_canonical_catalysts_match_a_fresh_reparse():
    """For one entry per free carbene, the checkpoint is exactly what the parser gives on its log.

    Non-canonical entries carry their canonical entry's values instead — see
    ``canonicalize_duplicates`` — so only the canonical rows pin the parser.
    """
    _logs()
    assert XTB_CHECKPOINT.exists()
    stored = pd.read_parquet(XTB_CHECKPOINT)
    stored.index = stored.index.astype(int)
    fresh = reparse_logs().reindex(stored.index)[list(stored.columns)]
    canonical = sorted(duplicate_groups())
    pd.testing.assert_frame_equal(
        stored.loc[canonical], fresh.loc[canonical], check_exact=False, atol=1e-9, rtol=0.0
    )


def test_duplicate_free_carbenes_share_their_scalars():
    """Entries that are the same free carbene carry identical xTB scalars."""
    _logs()
    stored = pd.read_parquet(XTB_CHECKPOINT)
    stored.index = stored.index.astype(int)
    groups = {c: ids for c, ids in duplicate_groups().items() if len(ids) > 1}
    assert groups, "no duplicate groups in the library; this guard would be vacuous"
    for canonical, ids in groups.items():
        spread = (stored.loc[ids].max() - stored.loc[ids].min()).abs()
        assert (spread == 0).all(), (canonical, ids, spread[spread > 0].to_dict())


@pytest.mark.parametrize("name", FEATURIZATIONS)
def test_assembled_featurizations_carry_the_checkpoint_scalars(name):
    """Each assembled table's xTB scalar columns equal the checkpoint row for the same catalyst."""
    _logs()
    feat = pd.read_parquet(paths.FEAT_DIR / name / "descriptors.parquet")
    cols = [c for c in AFFECTED_COLS if c in feat.columns]
    assert cols, name
    ckpt = pd.read_parquet(XTB_CHECKPOINT)
    ckpt.index = ckpt.index.astype(int)
    aligned = ckpt.reindex(feat.index.astype(int))
    for c in cols:
        assert (feat[c].to_numpy() == pytest.approx(aligned[c].to_numpy(), abs=1e-9)), (name, c)


@pytest.mark.parametrize("name", FEATURIZATIONS)
def test_free_carbene_descriptors_agree_within_a_duplicate_group(name):
    """Salt forms of one free carbene carry identical free-carbene descriptors."""
    df = pd.read_parquet(paths.FEAT_DIR / name / "descriptors.parquet")
    cols = [c for c in FREE_CARBENE_COLS if c in df.columns]
    assert cols, name
    groups = {c: ids for c, ids in duplicate_groups().items() if len(ids) > 1}
    assert groups
    for canonical, ids in groups.items():
        spread = (df.loc[ids, cols].max() - df.loc[ids, cols].min()).abs()
        assert (spread == 0).all(), (name, canonical, ids, spread[spread > 0].to_dict())


@pytest.mark.parametrize("name", FEATURIZATIONS)
def test_form_dependent_descriptors_still_vary_within_a_group(name):
    """Canonicalization must not have flattened the columns that distinguish salt forms."""
    df = pd.read_parquet(paths.FEAT_DIR / name / "descriptors.parquet")
    groups = {c: ids for c, ids in duplicate_groups().items() if len(ids) > 1}
    varies = sum(
        1 for ids in groups.values() if df.loc[ids, "mw"].nunique() > 1
    )
    assert varies > 0, f"{name}: mw is constant in every duplicate group; forms were flattened"
