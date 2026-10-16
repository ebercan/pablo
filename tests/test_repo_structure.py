"""Structural invariants of the whole ``pablo`` package.

Every module must import, every declared path must resolve, no module may be a thin
re-export stub, every cross-module import and module-attribute patch must name a real
attribute, every global name must resolve, and the interpretable-feature import graph
must stay acyclic.

Scope: the import, stub, cross-import, patch and undefined-global checks walk **all**
of ``pablo``, not just ``featurization_and_kernel_scan``, because modules outside
featscan can break import just as well.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pkgutil
from pathlib import Path

import pytest

import pablo
import pablo.featurization_and_kernel_scan as FKS
from pablo.featurization_and_kernel_scan import paths

PKG = FKS.__name__
ROOT_PKG = pablo.__name__
PABLO_ROOT = Path(pablo.__file__).resolve().parent

# venv_wrappers run inside isolated per-model virtualenvs and import packages that
# are deliberately absent from the main environment. Nothing else may be skipped:
# an entry here is a module admitted to be broken.
SKIP_PREFIXES = ("pablo.featurization_and_kernel_scan.features.venv_wrappers",)


def _module_names() -> list[str]:
    """Every importable module in ``pablo``, minus the venv-side wrappers."""
    names = []
    for mod in pkgutil.walk_packages(pablo.__path__, ROOT_PKG + "."):
        if mod.name.startswith(SKIP_PREFIXES):
            continue
        names.append(mod.name)
    return sorted(names)


def _source_files() -> list[Path]:
    """Every source file in ``pablo`` that the invariants apply to."""
    return [
        py for py in sorted(PABLO_ROOT.rglob("*.py"))
        if "__pycache__" not in py.parts and "venv_wrappers" not in py.parts
    ]


@pytest.mark.parametrize("module_name", _module_names())
def test_every_module_imports(module_name):
    """No module anywhere in ``pablo`` raises on import."""
    importlib.import_module(module_name)


def test_no_reexport_stubs():
    """No module exists only to forward to another one.

    Re-export stubs left at old import paths look harmless and are not: the stub and
    its target are different module objects with different ``__file__``, and
    ``scheduler`` hashes ``Path(config.__file__)`` into every result JSON as run
    provenance. A stub therefore lets a run record the hash of a forwarding file
    instead of the config it actually used.
    """
    offenders = []
    for py in _source_files():
        text = py.read_text()
        tree = ast.parse(text)
        star = any(
            isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
            for n in ast.walk(tree)
        )
        forwarder = any(
            isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
            for n in tree.body
        )
        if star or forwarder:
            offenders.append(f"{py.relative_to(PABLO_ROOT)} (star={star}, getattr={forwarder})")
    assert not offenders, f"re-export stubs: {offenders}"


def test_cross_module_imports_resolve():
    """Every ``from pablo.X import name`` names something ``pablo.X`` defines.

    Checked statically over all source files, so every unresolved name is reported
    in one run instead of one ImportError at a time.
    """
    offenders = []
    for py in _source_files():
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            mod = node.module or ""
            if not mod.startswith("pablo."):
                continue
            if mod.startswith(SKIP_PREFIXES):
                continue
            try:
                target = importlib.import_module(mod)
            except Exception as exc:
                offenders.append(f"{py.relative_to(PABLO_ROOT)}: cannot import {mod} ({exc.__class__.__name__})")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if hasattr(target, alias.name):
                    continue
                # `from pkg import submodule` is valid even though the attribute does
                # not exist until the submodule is imported.
                try:
                    importlib.import_module(f"{mod}.{alias.name}")
                except ImportError:
                    offenders.append(
                        f"{py.relative_to(PABLO_ROOT)}: {mod} has no {alias.name!r}"
                    )
    assert not offenders, "unresolved imports:\n  " + "\n  ".join(offenders)


def test_expected_packages_exist():
    """The restructured subpackages are present and are real packages."""
    for sub in ("core", "features", "features.interpretable", "cross_validation",
                "results_analysis"):
        mod = importlib.import_module(f"{PKG}.{sub}")
        assert mod.__doc__, f"{sub} has no module docstring"


def test_canonical_data_paths_resolve():
    """Every committed data path points at something that exists.

    ``INPUTS_JSON`` is deliberately absent from this list: it is regenerable output of
    ``generate_inputs``, not a committed artifact, and requiring it here would force a
    stale grid to be kept on disk just to keep the suite green.
    """
    must_exist = [
        paths.PKG_ROOT, paths.REPO_ROOT, paths.DATA_DIR, paths.LIB_DIR,
        paths.FEAT_DIR, paths.INTERMEDIATE_DIR, paths.XTB_DIR,
        paths.XTB_INPUTS_DIR, paths.GRID_DIR, paths.QUEUE_DIR,
        paths.CATALYSTS_PARQUET, paths.SOLVENTS_PARQUET, paths.BASES_PARQUET,
        paths.EXPERIMENTS_PARQUET, paths.CV_SPLITS_JSON,
        paths.ARCHIVE_DIR,
    ]
    missing = [str(p) for p in must_exist if not p.exists()]
    assert not missing, f"missing paths: {missing}"


def test_generated_grid_is_absent_or_current():
    """If a grid file exists at all, its kernel structures must be buildable.

    A stale ``inputs.json`` crashes the scan at ``KernelStructure.build`` on the first
    fit rather than at load time, so a wrong grid is worse than no grid.
    """
    if not paths.INPUTS_JSON.exists():
        pytest.skip("no generated grid; run generate_inputs to create one")

    import json

    from pablo.featurization_and_kernel_scan.core import config

    cells = json.loads(paths.INPUTS_JSON.read_text())
    found = {c["kernel_structure"] for c in cells.values()}
    unknown = sorted(found - set(config.KERNEL_STRUCTURES))
    assert not unknown, (
        f"inputs.json names structures that cannot be built: {unknown}. "
        "Regenerate it: python -m "
        "pablo.featurization_and_kernel_scan.cross_validation.generate_inputs"
    )


def test_no_legacy_data_layout_left_behind():
    """The pre-restructure data directories are gone, not duplicated.

    Checked by literal path rather than by a ``LEGACY_*`` constant: those constants
    were removed with the fallbacks that used them, and hardcoding the old layout here
    is what makes the check independent of the code it is checking.
    """
    legacy = [
        paths.DATA_DIR / "xtb_inputs",
        paths.DATA_DIR / "xtb_outputs",
        paths.DATA_DIR / "queue",
        paths.PKG_ROOT / "results_archived",
        paths.DATA_DIR / "experiments.parquet",
        paths.DATA_DIR / "catalysts.parquet",
        paths.DATA_DIR / "inputs.json",
    ]
    present = [str(p) for p in legacy if p.exists()]
    assert not present, f"legacy paths still present: {present}"


def test_module_attribute_patches_target_real_names():
    """A ``module.KNOB = value`` patch must name a knob the module defines.

    Assigning an attribute that does not exist is silent: Python attaches it, nothing
    reads it, and the run uses the module's default while appearing to use the
    override (e.g. setting ``main.GA_N_EVALS`` when ``main`` reads ``GA_SETTINGS``).
    """
    offenders = []
    for py in _source_files():
        tree = ast.parse(py.read_text())
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("pablo.") and a.asname:
                        aliases[a.asname] = a.name
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pablo"):
                for a in node.names:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
        if not aliases:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
                    continue
                mod_name = aliases.get(target.value.id)
                # Only config-style knobs; lowercase attributes are ordinary state.
                if not mod_name or not target.attr.isupper():
                    continue
                try:
                    mod = importlib.import_module(mod_name)
                except ImportError:
                    continue
                if not hasattr(mod, target.attr):
                    offenders.append(
                        f"{py.relative_to(PABLO_ROOT)}:{node.lineno}: "
                        f"{mod_name} has no {target.attr!r} — this patch is silently ignored"
                    )
    assert not offenders, "dead attribute patches:\n  " + "\n  ".join(offenders)


def test_package_source_has_no_undefined_globals():
    """Every module resolves its global names — catches a bad file split.

    A module sliced out of a larger one can import cleanly while still
    referencing a name that stayed behind; that only fails when the function is
    called. This check is static, so it catches it at test time.
    """
    offenders = {}
    for py in _source_files():
        tree = ast.parse(py.read_text())
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.alias):
                defined.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        used = {
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        missing = used - defined - set(dir(builtins)) - {"__file__", "__doc__", "__name__"}
        if missing:
            offenders[py.relative_to(PABLO_ROOT).as_posix()] = sorted(missing)
    assert not offenders, f"undefined globals: {offenders}"


def test_interpretable_import_graph_is_acyclic():
    """xtb_pipeline must not import its own consumers.

    The import order is xtb_pipeline → descriptors → assemble; xtb_pipeline importing
    ``descriptors`` or ``assemble`` would create a cycle.
    """
    pipeline = (paths.PKG_ROOT / "features" / "interpretable" / "xtb_pipeline.py").read_text()
    for consumer in ("descriptors", "assemble"):
        assert f"import {consumer}" not in pipeline, (
            f"xtb_pipeline imports {consumer} — the import cycle is back"
        )


def test_no_module_reimports_the_old_generator_modules():
    """Featscan package code must not import the ``generate_*_features`` module names.

    None of these modules exist in the package, so any such import fails.
    """
    banned = (
        "generate_descriptor_features",
        "generate_foundation_features",
        "generate_gnn_features",
        "generate_interpretable_features",
    )
    offenders = []
    for py in sorted(paths.PKG_ROOT.rglob("*.py")):
        if py.name in {f"{b}.py" for b in banned} or "__pycache__" in py.parts:
            continue
        text = py.read_text()
        for name in banned:
            if f"import {name}" in text or f"from .{name}" in text:
                offenders.append(f"{py.name} -> {name}")
    assert not offenders, offenders
