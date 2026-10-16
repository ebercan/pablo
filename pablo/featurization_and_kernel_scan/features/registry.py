"""The featurizer registry — one entry per buildable featurization step.

Each step declares the execution environment it needs in the ``runtime``
field:

    local  CPU only, no prerequisites beyond the installed requirements
    xtb    needs the ``xtb`` binary on PATH and the optimised geometries
    gpu    needs an H100 node; ``features.foundation.submit`` sends it to SLURM
    venv   shells out to a per-model virtualenv on the Desktop

``requires`` names steps that must have run first, so the CLI can refuse a step
whose inputs do not exist yet instead of failing deep inside a subprocess.

Adding a featurizer means adding one ``Featurizer`` entry. Nothing else in the
package enumerates featurizers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pablo.featurization_and_kernel_scan import paths

RUNTIMES: tuple[str, ...] = ("local", "xtb", "gpu", "venv")
FAMILIES: tuple[str, ...] = (
    "fingerprint",
    "component",
    "interpretable",
    "foundation",
    "gnn",
)


@dataclass(frozen=True)
class Featurizer:
    """One buildable step of the featurization pipeline.

    Attributes:
        name: CLI name, unique across the registry.
        family: One of ``FAMILIES``; groups the listing only.
        runtime: One of ``RUNTIMES``; what the step needs to run.
        build: Zero-argument callable performing the step.
        outputs: Paths the step writes, relative to ``paths.FEAT_DIR`` or
            ``paths.DATA_DIR``: featurization directory names, or cache paths
            such as ``"_intermediate/lp_sterimol.parquet"``. The CLI treats a
            step as done when all of them exist. Empty only for report-only
            steps that write nothing (``tally_topology``); such a step never
            counts as done.
        requires: Names of steps that must run first.
        note: One-line description shown by ``--list``.
        build_one: Optional single-catalyst builder taking a catalyst id. Set on
            the steps the Mechthild CPU array fans out one task per catalyst;
            ``None`` means the step only runs over the whole library.
    """

    name: str
    family: str
    runtime: str
    build: Callable[[], None]
    outputs: tuple[str, ...] = ()
    requires: tuple[str, ...] = field(default=())
    note: str = ""
    build_one: Callable[[int], None] | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}")
        if self.runtime not in RUNTIMES:
            raise ValueError(f"{self.name}: unknown runtime {self.runtime!r}")


def _fingerprints() -> list[Featurizer]:
    from pablo.featurization_and_kernel_scan.features import fingerprints as fp

    def mol2vec() -> None:
        model_path = paths.WEIGHTS_DIR / "mol2vec_weights.pkl"
        assert model_path.exists(), (
            f"Mol2vec weights not found at {model_path}. Download "
            "mol2vec_weights.pkl into data/weights/."
        )
        fp.run_mol2vec(model_path)

    return [
        Featurizer("map4", "fingerprint", "local", fp.run_map4, ("map4",),
                   note="MAP4 fingerprints, radius 2 (skfp)"),
        Featurizer("ecfp4", "fingerprint", "local", fp.run_ecfp4, ("ecfp4",),
                   note="ECFP4 count fingerprints, Morgan radius 2 (RDKit)"),
        Featurizer("mordred", "fingerprint", "local", fp.run_mordred, ("mordred",),
                   note="2D Mordred descriptors + 5-step preprocessing"),
        Featurizer("mol2vec", "fingerprint", "local", mol2vec, ("mol2vec",),
                   note="pretrained Word2Vec embeddings, L2-normalised"),
        Featurizer("soap", "fingerprint", "xtb", fp.generate_soap, ("soap",),
                   requires=("xtb",),
                   note="SOAP power spectrum on xTB free-carbene geometries"),
    ]


def _components() -> list[Featurizer]:
    from pablo.featurization_and_kernel_scan.features import components as comp

    return [
        Featurizer(
            "solvent_base", "component", "local", comp.run_solvent_base,
            ("solvent_KT", "solvent_Hansen", "solvent_Gutmann", "solvent_OHE",
             "base_thermo", "base_steric", "base_OHE", "continuous"),
            note="all solvent/base descriptor blocks + continuous bounds",
        ),
        Featurizer("solvent_map", "component", "local", comp.run_solvent_map,
                   ("solvent_map",), note="MAP2 fingerprints over the 14 solvents"),
        Featurizer("base_map", "component", "local", comp.run_base_map,
                   ("base_map",), note="MAP2 fingerprints over the 8 bases"),
        Featurizer("component_soap", "component", "xtb", comp.run_component_soap,
                   ("solvent_soap", "base_soap"),
                   note="SOAP on xTB geometries of the solvents and bases"),
        Featurizer("component_pca", "component", "local", comp.run_component_pca,
                   tuple(f"{n}_pca" for n in comp.COMPONENT_PCA_SOURCES),
                   requires=("grover_components", "component_soap"),
                   note="PCA-knee reductions of the solvent/base GROVER and SOAP"),
    ]


def _interpretable() -> list[Featurizer]:
    from pablo.featurization_and_kernel_scan.features.interpretable import (
        assemble as asm,
    )
    from pablo.featurization_and_kernel_scan.features.interpretable import (
        descriptors as desc,
    )
    from pablo.featurization_and_kernel_scan.features.interpretable import xtb_pipeline

    return [
        Featurizer("structural", "interpretable", "local", asm.generate_structural,
                   ("_intermediate/structural_onehots",),
                   note="RDKit-only structural one-hots"),
        Featurizer("xtb", "interpretable", "xtb", xtb_pipeline.run_all_xtb,
                   ("xtb/results_checkpoint.parquet",),
                   note="optimise every catalyst geometry with xTB (the prerequisite)"),
        Featurizer("tally_topology", "interpretable", "local", asm.tally_topology,
                   note="report the library's ring-type / form breakdown"),
        Featurizer("n_c_aggregates", "interpretable", "xtb", desc.generate_nc_aggregates,
                   ("_intermediate/n_c_aggregates.parquet",), requires=("xtb",),
                   note="N/C substituent Sterimol aggregates"),
        Featurizer("lp_sterimol", "interpretable", "xtb", desc.generate_lp_sterimol,
                   ("_intermediate/lp_sterimol.parquet",), requires=("xtb",),
                   note="LP-axis whole-ligand Sterimol"),
        Featurizer("vbur_quadrants", "interpretable", "xtb", desc.generate_vbur_quadrants,
                   ("_intermediate/vbur_quadrants.parquet",), requires=("xtb",),
                   note="quadrant-resolved %Vbur"),
        Featurizer("xtb_extra", "interpretable", "xtb", desc.generate_xtb_extra,
                   ("_intermediate/xtb_extra",), requires=("xtb",),
                   note="Fukui / HOMO weight / Mulliken charge at C2",
                   build_one=desc.generate_xtb_extra),
        Featurizer("probe_h_plus", "interpretable", "xtb", desc.generate_probe_h_plus,
                   ("_intermediate/probe_h_plus",), requires=("xtb",),
                   note="H+ probe-binding energy",
                   build_one=desc.generate_probe_h_plus),
        Featurizer("probe_bh3", "interpretable", "xtb", desc.generate_probe_bh3,
                   ("_intermediate/probe_bh3",), requires=("xtb",),
                   note="BH3 probe-binding energy",
                   build_one=desc.generate_probe_bh3),
        Featurizer("bh3_reference", "interpretable", "xtb", desc.generate_bh3_reference,
                   ("_intermediate/bh3_reference.json",), requires=("xtb",),
                   note="free BH3 reference energy"),
        Featurizer("deltaG_deprot", "interpretable", "xtb", desc.generate_deltaG_deprot,
                   ("_intermediate/deltaG_deprot",), requires=("xtb",),
                   note="C2 deprotonation free energy, GFN2/ALPB",
                   build_one=desc.generate_deltaG_deprot),
        Featurizer("interpretable_steric", "interpretable", "local", asm.assemble_steric,
                   ("interpretable_steric",), requires=("n_c_aggregates",),
                   note="assemble the 10-dim steric featurization"),
        Featurizer("interpretable_light", "interpretable", "local", asm.assemble_light,
                   ("interpretable_light",),
                   requires=("structural", "lp_sterimol", "xtb"),
                   note="assemble the 14-dim light featurization"),
        Featurizer("interpretable", "interpretable", "local", asm.assemble_full,
                   ("interpretable",),
                   requires=("structural", "lp_sterimol", "vbur_quadrants", "xtb_extra"),
                   note="assemble the 23-dim full featurization"),
        Featurizer("interpretable_dG", "interpretable", "local", asm.assemble_dG,
                   ("interpretable_dG",), requires=("deltaG_deprot",),
                   note="the 23-dim featurization with the deprotonation free energy"),
    ]


def _foundation() -> list[Featurizer]:
    from pablo.featurization_and_kernel_scan.features import foundation as fnd

    notes = {
        "molformer_xl_precat": "MolFormer-XL on pre-catalyst SMILES",
        "molformer_xl_carbene": "MolFormer-XL on free-carbene SMILES + form one-hot",
        "chembert2": "ChemBERTa-2, DeepChem/ChemBERTa-77M-MTR",
        "clamp": "CLAMP molecular MLP encoder, 768-d",
        "unimol": "UniMol2 1.1B on free-carbene 3D conformers",
        "grover": "GROVERlarge fingerprints via CLI subprocess",
        "grover_components": "GROVERlarge on the solvent and base libraries",
    }
    outputs = {"grover_components": ("solvent_grover", "base_grover")}
    return [
        Featurizer(name, "foundation", "gpu",
                   (lambda n=name: fnd.run_model(n)), outputs.get(name, (name,)),
                   note=notes[name])
        for name in fnd.VALID_MODELS
    ]


def _gnn() -> list[Featurizer]:
    from pablo.featurization_and_kernel_scan.features import gnn

    return [
        Featurizer(name, "gnn", "venv",
                   (lambda n=name: gnn.run_model(n)), (name,),
                   note=f"{name.rsplit('_', 1)[0]} in its own Desktop virtualenv")
        for name in gnn.VALID_MODELS
    ]


def build_registry() -> dict[str, Featurizer]:
    """Assemble the registry, importing each family lazily.

    Families are imported inside the builders so that a missing optional
    dependency (``skfp``, ``dscribe``, a GNN venv) only breaks the family that
    needs it, and ``--list`` still works for the rest.

    Returns:
        Mapping of featurizer name to its ``Featurizer`` entry.

    Raises:
        ValueError: If two families declare the same name.
    """
    entries: list[Featurizer] = []
    for loader in (_fingerprints, _components, _interpretable, _foundation, _gnn):
        entries.extend(loader())

    registry: dict[str, Featurizer] = {}
    for entry in entries:
        if entry.name in registry:
            raise ValueError(f"Duplicate featurizer name {entry.name!r}")
        registry[entry.name] = entry
    return registry
