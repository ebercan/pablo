"""Modeling core: replicate noise evidence, configuration, priors, kernels, and the model factory.

    replicates.py  replicate-group detection and the replicate-anchored noise prior
    config.py      every grid axis, prior value, and SLURM setting
    priors.py      the LogNormal / Gamma / Dirichlet prior constructors
    kernels.py     base kernels, the partial-ARD layer, and KS1-KS4
    models.py      mean modules and the AcyloinCondensationModel factory

Dependency order is replicates -> config -> priors -> kernels -> models; nothing
here imports from ``features`` or ``cross_validation``.
"""
