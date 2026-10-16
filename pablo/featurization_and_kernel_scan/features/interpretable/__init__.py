"""Interpretable and xTB-derived catalyst featurizations.

Pipeline modules, in dependency order (the import graph is acyclic):

    xtb_pipeline  geometry generation, xTB execution, intermediate caches
    descriptors   Sterimol / %Vbur / electronics / ΔG_deprot blocks, built on
                  the pipeline
    assemble      the four ``interpretable*`` featurization assemblers

Maintenance scripts that patch the caches and assembled tables in place:

    canonicalize_duplicates  make free-carbene descriptors identical across the
                             salt forms of one carbene
    repair_xtb_scalars       re-parse saved xTB logs for the relaxed-geometry
                             HOMO / LUMO / gap scalars
"""
