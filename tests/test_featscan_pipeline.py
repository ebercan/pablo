"""End-to-end pipeline invariants for the featurization and kernel scan.

A frozen Mechthild run imports, every archived run's MANIFEST.json matches its
cell files, no module writes the hand-maintained report, and a real fold fits
with finite metrics under every kernel structure.
"""

from __future__ import annotations

import importlib

import pytest

from pablo.featurization_and_kernel_scan import paths

def test_frozen_mechthild_runs_still_import():
    """run_4's variant driver (``run_4_af_opt_strategy.main_variant``) imports."""
    importlib.import_module("pablo.mechthild_runs.run_4_af_opt_strategy.main_variant")


def test_archived_runs_are_self_describing():
    """Every archived run carries a MANIFEST.json naming its layout."""
    import json

    runs = [d for d in sorted(paths.ARCHIVE_DIR.iterdir()) if d.is_dir()]
    assert runs, "no archived runs found"
    for run in runs:
        manifest = run / "MANIFEST.json"
        assert manifest.exists(), f"{run.name} has no MANIFEST.json"
        meta = json.loads(manifest.read_text())
        for key in ("run", "purpose", "frozen", "layout", "reproduce"):
            assert key in meta, f"{run.name} MANIFEST missing {key}"
        assert meta["run"] == run.name


def test_archive_manifests_match_their_cells():
    """Each MANIFEST's cell count, vocabulary and result fields match its cell JSONs.

    The vocabulary of ``kernel_structure``, ``solv_feat``, ``base_feat`` and
    ``outcome_transform`` and the ``result_fields`` markers are re-read from the
    cell files and compared with what the manifest states.
    """
    import json
    import re

    fields = ("kernel_structure", "solv_feat", "base_feat", "outcome_transform")
    for run in sorted(d for d in paths.ARCHIVE_DIR.iterdir() if d.is_dir()):
        meta = json.loads((run / "MANIFEST.json").read_text())
        cells = sorted((run / "cells").glob("result_*.json")) if (run / "cells").is_dir() else []

        assert meta["n_cell_results"] == len(cells), (
            f"{run.name}: manifest says {meta['n_cell_results']} cells, found {len(cells)}"
        )
        if not cells:
            assert not meta.get("vocabulary"), (
                f"{run.name}: manifest claims a vocabulary but has no cells to support it"
            )
            continue

        found = {f: set() for f in fields}
        markers = set()
        for cell in cells:
            text = cell.read_text()
            for name in fields:
                found[name].update(re.findall(rf'"{name}":\s*"([^"]*)"', text))
            for marker in ("subblock_group_sizes", "partition_group_sizes"):
                if marker in text:
                    markers.add(marker)

        claimed = meta.get("vocabulary", {})
        for name, values in found.items():
            if not values:
                continue
            assert sorted(values) == claimed.get(name), (
                f"{run.name}: manifest {name}={claimed.get(name)} but cells contain "
                f"{sorted(values)}"
            )
        assert sorted(markers) == meta.get("result_fields"), (
            f"{run.name}: manifest result_fields={meta.get('result_fields')} but cells "
            f"contain {sorted(markers)}"
        )


def test_nothing_writes_the_hand_maintained_report():
    """No module in the package references featurization_kernel_scan_report.html.

    That report is written and edited by hand, so no script may write to its path.
    """
    offenders = []
    for py in sorted(paths.PKG_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        if "featurization_kernel_scan_report.html" in py.read_text():
            offenders.append(py.relative_to(paths.PKG_ROOT).as_posix())
    assert not offenders, f"modules referencing the hand-maintained report: {offenders}"


@pytest.mark.parametrize("ks", ["KS1", "KS2", "KS3", "KS4"])
def test_end_to_end_single_fold_fit(ks):
    """Assemble real features, fit one fold per structure, get finite metrics."""
    from pablo.featurization_and_kernel_scan.core.kernels import build_partition
    from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import fit_eval
    from pablo.utils.feature_assembly import assemble_features

    cell = assemble_features(
        cat_featurization="interpretable_steric",
        solv_featurization="KT",
        base_featurization="thermo",
        experiments_path=paths.EXPERIMENTS_PARQUET,
        featurizations_root=paths.FEAT_DIR,
        libraries_root=paths.LIB_DIR,
    )
    cell.partition = build_partition(cell.feature_dims)
    n = cell.X.shape[0]
    result = fit_eval(
        cell,
        train_idx=list(range(n - 8)),
        test_idx=list(range(n - 8, n)),
        kernel_structure=ks,
        outcome_transform="raw",
        model_class="st_y",
        base_kernel="matern52",
        mean_function="type_linear",
        n_restarts=1,
        max_iter=40,
    )
    assert result.status == "ok", result
    assert result.rmse is not None and result.rmse > 0


def test_mapsaas_kernels_do_not_see_the_form_id_column():
    """MAP-SAAS drops the form-ID column before its additive SAAS kernels."""
    import torch

    from pablo.featurization_and_kernel_scan.core.models import make_mapsaas_gp

    g = torch.Generator().manual_seed(0)
    x = torch.rand(15, 6, generator=g, dtype=torch.float64)
    x[:, 5] = torch.randint(0, 4, (15,), generator=g).double()
    y = torch.rand(15, 1, generator=g, dtype=torch.float64)
    model = make_mapsaas_gp(x, y, form_id_dim=5)
    assert model.transform_inputs(x).shape[-1] == 5
    for kernel in model.covar_module.kernels if hasattr(model.covar_module, "kernels") else [model.covar_module]:
        base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
        assert base.ard_num_dims == 5
    moved = x.clone()
    moved[:, 5] = 3.0 - moved[:, 5]
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model.posterior(x).mean, model.posterior(moved).mean)
