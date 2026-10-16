"""Featurization code for the catalyst, solvent, and base blocks.

    registry.py      name -> Featurizer, the only enumeration of featurizers
    __main__.py      the single CLI for every step
    common.py        shared library loading / matrix writing / normalisation
    fingerprints.py  map4, ecfp4, mordred, mol2vec, soap
    components.py    solvent_*, base_*, continuous
    dim_reduction.py frozen PCA-knee reductions (<name>_pca) of any block
    interpretable/   xtb_pipeline -> descriptors -> assemble
    foundation.py    six pretrained encoders (GPU)
    gnn.py           twelve graph models (per-model Desktop virtualenvs)
    venv_wrappers/   the scripts those virtualenvs execute; never imported here

Run ``python -m pablo.featurization_and_kernel_scan.features --list`` for the
registry, including which outputs already exist.
"""

from pablo.featurization_and_kernel_scan.features.registry import (
    Featurizer,
    build_registry,
)

__all__ = ["Featurizer", "build_registry"]
