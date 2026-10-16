"""Central configuration for the featurization x kernel x model cross-validation study.

This module is the single source of truth for:

  1. Grid axes — ``ELIMINATION_GRID`` holds the catalyst levels, the baseline
     level of every other axis and the one-axis sweeps of the elimination run;
     ``CROSSED_GRID_1`` the full cross of the ``cross_1`` run;
     ``MONROE_M3_GRID`` the listed configurations of the ``monroe_m3`` run.
     ``RUN_GRIDS`` names every run.
  2. Hyper-hyperparameters — the priors (InverseGamma noise, Gamma outputscale,
     LogNormal lengthscale, Dirichlet additive-weights, Normal mean-module),
     their parameters, and the initial outputscale. These are global constants,
     not grid axes: the only prior axis is ``PRIOR_SCHEMES``, which varies how
     the Hvarfner ``log√D`` location resolves and how many lengthscales it
     applies to. The lengthscale floor is not a prior at all — it is a
     constraint applied to every fit.
  3. Optimization settings — L-BFGS restarts, max iterations, Cholesky jitter.
  4. SLURM submission settings for Mechthild.

No function in the codebase should hold defaults for any of these values.
All call sites import from here. Retuning the study means editing this file
and nothing else.

The InverseGamma noise prior is computed live at import from the replicate
groups detected in ``data/libraries/experiments.parquet`` — both α and β, per
target and outcome scale; see ``NOISE_EVIDENCE``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pablo.utils.outcome_transforms import OUTCOME_TRANSFORM_NAMES
from pablo.featurization_and_kernel_scan import paths
from pablo.featurization_and_kernel_scan.core.replicates import (
    ReplicateNoiseEvidence,
    noise_evidence_by_scale,
)


DEFAULT_BASE_KERNELS: tuple[str, ...] = ("matern52",)
"""Base kernels for a catalyst featurization that does not declare its own.

Every dense real-valued featurization uses this. ``kernels.BASE_KERNELS`` also
holds ``rbf``, which a crossed grid selects through its ``base_kernel`` axis,
and ``tanimoto``/``minmax``, which are reserved for the fingerprint
featurizations (``FINGERPRINT_BASE_KERNELS``). There are no ``*_id`` index columns left on the
grid — every unordered category (pre-catalyst form, ring type, solvent, base) is
one-hot encoded in its stored featurization and measured by the block's own base
kernel, so no Hamming/``CategoricalKernel`` route exists. Vanilla uses RBF
independently."""

FINGERPRINT_BASE_KERNELS: tuple[str, ...] = ("tanimoto", "minmax", "matern52")
"""Base kernels for the fingerprint featurizations.

``matern52`` is the within-featurization control: without it, a Tanimoto win
could not be attributed to the kernel rather than to the fingerprint, because
the only alternative comparison (fingerprint + Tanimoto vs. embedding +
Matern) varies both at once."""


@dataclass(frozen=True)
class CatalystSpec:
    """One catalyst featurization and the base kernels it is scanned with.

    Carrying ``base_kernels`` on the entry itself — rather than in a side table
    keyed by name — is what keeps the grid from silently becoming a full
    ``catalyst x base_kernel`` cartesian product: a featurization cannot be
    added without declaring which kernels it pairs with.

    Args:
        name: Directory under ``data/featurizations/`` holding
            ``descriptors.parquet``.
        append_form_onehot: Whether the pre-catalyst-form one-hot column is
            appended to the catalyst block.
        base_kernels: Base kernel types this featurization is scanned with.
    """

    name: str
    append_form_onehot: bool
    base_kernels: tuple[str, ...] = DEFAULT_BASE_KERNELS


CATALYST_SETS: dict[str, list[CatalystSpec]] = {
    "main": [
        CatalystSpec("clamp", False),
        CatalystSpec("mordred", False),
        CatalystSpec("mol2vec", False),
        CatalystSpec("chembert2", False),
        CatalystSpec("molformer_xl_precat", False),
        CatalystSpec("molformer_xl_carbene", True),
        CatalystSpec("unimol", True),
        CatalystSpec("grover", False),
        CatalystSpec("soap", False),
        CatalystSpec("minimol_precat", False),
        CatalystSpec("minimol_carbene", True),
        CatalystSpec("scage_precat", False),
        CatalystSpec("scage_carbene", True),
        CatalystSpec("graphormer_precat", False),
        CatalystSpec("graphormer_carbene", True),
        CatalystSpec("infomax3d_precat", False),
        CatalystSpec("infomax3d_carbene", True),
        CatalystSpec("frad_precat", False),
        CatalystSpec("frad_carbene", True),
        CatalystSpec("transformerm_precat", False),
        CatalystSpec("transformerm_carbene", True),
        CatalystSpec("monroe_precat", False),
    ],
    "interpretable": [
        CatalystSpec("interpretable_steric", False),
        CatalystSpec("interpretable_light", False),
        CatalystSpec("interpretable", False),
        CatalystSpec("interpretable_dG", False),
    ],
    "fingerprint": [
        CatalystSpec("ecfp4", False, FINGERPRINT_BASE_KERNELS),
        CatalystSpec("map4", False, FINGERPRINT_BASE_KERNELS),
    ],
}
"""Catalyst featurizations, grouped into families.

A grid's catalyst level resolves to its entry here by name, a ``<name>_pca``
level to the entry of ``<name>``; ``generate_inputs --cat-set`` narrows a grid to
the levels of one family. The "all" set is main + interpretable + fingerprint.

``interpretable_dG`` is ``interpretable`` with ``delta_e_h_plus`` *replaced* by
the xTB C2-deprotonation free energy ``xtb_deltaG_deprot`` — a swap, not an
addition. The two columns are the same quantity at different levels of care
(single-geometry electronic protonation energy vs. Boltzmann free energy over
conformer ensembles) and correlate at r = -0.96, so the pair differs in exactly
one column and the scan measures head-to-head whether the conformer ensemble and
the thermal correction earn their cost.

The ``fingerprint`` set is the *only* one carrying ``tanimoto``/``minmax``.
Those two kernels compare multisets of substructures via overlap-over-union, so
they need a fingerprint to be meaningful: ``ecfp4`` is a 2048-bit count
fingerprint and ``map4`` is 2048-bit binary. On the other 25 featurizations —
all dense real-valued embeddings — they are both unmotivated and near-redundant
(Gram rank-correlation with RBF >= 0.99 on 16 of 25), and they are basis
dependent: rotating an embedding into its own PCA basis, which preserves all
information, moves the MinMax Gram to r = -0.45 against itself.

Under the ``block_shared`` and ``legacy`` partitions (``kernels.build_partition``)
the catalyst block carries one shared lengthscale. A uniform rescaling cancels in
both ratio kernels, which makes that lengthscale inert for the fingerprint cells —
only the outputscale is learned on that block. This is the standard
fixed-Tanimoto GP baseline and is intentional. Under ``full_ard`` each column has
its own lengthscale and only their ratios matter.
"""
CATALYST_SETS["all"] = (
    CATALYST_SETS["main"]
    + CATALYST_SETS["interpretable"]
    + CATALYST_SETS["fingerprint"]
)


SOLVENT_FEATURIZATIONS: list[str] = ["KT", "Hansen", "Gutmann", "OHE", "grover_pca", "soap_pca"]
"""Solvent featurization blocks. The elimination grid pins ``OHE``.

Each name resolves to ``solvent_<name>/``. Rather than one pooled physical
table, the solvent is described by three competing accounts of what makes it
matter plus a control, so the grid can report which account the GP needs:

- ``KT`` (3 dims) — Kamlet-Taft alpha / beta / pi*.
- ``Hansen`` (4 dims) — Hansen dD / dP / dH plus logP.
- ``Gutmann`` (2 dims) — Gutmann donor number and relative permittivity.
- ``OHE`` (14 dims) — one-hot identity; the no-descriptor control.
- ``grover_pca`` (8 dims) — PCA-knee of GROVERlarge fingerprints.
- ``soap_pca`` (6 dims) — PCA-knee of SOAP on GFN2-xTB geometries.

The two ``_pca`` levels are reduced from ``solvent_grover`` / ``solvent_soap``,
fit over the 14-solvent library (``features.dim_reduction``).

``solvent_map`` (MAP4 fingerprints) remains generatable and is what
``pablo.main`` uses in production, but it is off this grid."""

BASE_FEATURIZATIONS: list[str] = ["thermo", "steric", "OHE", "grover_pca", "soap_pca"]
"""Base featurization blocks. The elimination grid pins ``OHE``.

Each name resolves to ``base_<name>/``:

- ``thermo`` (3 dims) — proton affinity plus an organic/inorganic one-hot.
- ``steric`` (3 dims) — Sterimol L / B1 / B5 around the basic atom.
- ``OHE`` (8 dims) — one-hot identity; the no-descriptor control.
- ``grover_pca`` (4 dims) — PCA-knee of GROVERlarge fingerprints.
- ``soap_pca`` (4 dims) — PCA of SOAP on GFN2-xTB ion-pair geometries; Kneedle
  finds no knee, so it keeps the 80% variance fallback.

Both ``_pca`` levels featurize the full salt, counter-cation included, and are
fit over the 8-base library. PC scores are centred, so the zero-filled no-base
rows sit at the base centroid rather than at an unoccupied point.

``base_map`` remains generatable but is off-grid."""

OUTCOME_TRANSFORMS: list[str] = list(OUTCOME_TRANSFORM_NAMES)
"""Every implemented outcome transform. ``raw`` is yield as a fraction in [0, 1]
and is the elimination-grid baseline; all four are swept."""

KERNEL_STRUCTURES: list[str] = ["KS1", "KS2", "KS3", "KS4", "KS5", "KS6", "KS7"]
"""Every implemented kernel structure (see ``KernelStructure`` in
kernels.py). Every cat/solv/base featurization is delivered raw and un-reduced,
so all four structures share one partition recipe (``kernels.build_partition``):
one shared lengthscale over each discrete block's raw feature columns, and full
ARD over the continuous block.

All four combine the same four block kernels ``k_cat, k_solv, k_base, k_cont``:

**KS1** — ``ScaleKernel`` over the product k_cat · k_solv · k_base · k_cont.

**KS2** — Dirichlet-weighted additive mixture of the 4 single-block kernels.

**KS3** — KS2's 4 singles plus the 6 pairwise products, 10 terms.

**KS4** — KS3's 10 terms plus the full 4-way product, 11 terms.

**KS5** — KS1 with the catalyst factor ``rho + (1 - rho) k_cat``,
``rho ~ Beta(KS5_FLOOR_BETA)``.

**KS6** — the KS1 product plus ``s_+² (k_solv + k_base + k_cont) / 3``, the
block kernels shared with the product, ``s_+ ~ HalfNormal(KS6_MAIN_SCALE_HALFNORMAL)``.

**KS7** — ``[s_g² + s_d² k̃_cat] · k_solv · k_base · k_cont``, with ``k̃_cat`` the
catalyst kernel centred over the 161-catalyst library and
``s_g ~ HalfNormal(KS7_SHARED_SCALE_HALFNORMAL)``.

KS2–KS4 put their weights on a softmax simplex under a symmetric Dirichlet prior
with a single global Gamma-priored outputscale (``DirichletAdditiveKernel``);
per-term outputscales would be redundant with the weights.

No-base handling: KS1 is unmasked — the zero-filled base rows sit at the "none"
location (no base block carries an integer index column, so an all-zero row is a
point no real base occupies), and inside a pure product a zero mask would
annihilate the whole covariance row. KS2–KS4 zero-mask the base block via
``base_active``, so every base-containing term vanishes for no-base rows. KS5–KS7
are unmasked, as KS1.

The flat-ARD ``Vanilla`` remains buildable in models.py but off-grid."""


MODEL_CLASSES: list[str] = [
    "st_y",
    "indep_cs",
    "indep_ys",
    "indep_yc",
    "icm_cs",
    "icm_ys",
    "icm_yc",
    "lcm_cs",
    "lcm_ys",
    "lcm_yc",
    "mapsaas_y",
    "mapsaas_cs",
    "mapsaas_ys",
    "mapsaas_yc",
    "logsum_cy",
    "condprod_cy",
]
"""Every implemented model class; the elimination grid pins ``st_y``. See
``AcyloinCondensationModel`` in models.py — one method per entry. Fourteen classes
span four families × the {yield-only, (C,S), (Y,S), (Y,C)} target combinations,
and two build ``Y = C·S`` into the model (``core/product_models.py``):

  ``st_y`` / ``mapsaas_y``    single-task on yield (yield only).
  ``indep_*``                 two independent ``SingleTaskGP``s.
  ``icm_*``                   ``KroneckerMultiTaskGP`` ICM, full rank (rank=2 = num_tasks).
  ``lcm_*``                   true LCM (``LCMKernel``, Q=2 latent groups, rank=2 each).
  ``mapsaas_*``               ``AdditiveMapSaasSingleTaskGP`` (single / independent pair).
  ``logsum_cy``               log C and log Y = log C + log S, LMC with fixed mixing;
                              outcome transform ``raw`` only.
  ``condprod_cy``             Y = C·s(x) + noise with the measured C, plus a probit
                              conversion GP; outcome transform ``probit`` only.

Yield is direct (``*_y``, ``*_ys``, ``*_yc``), marginal (ICM/LCM task), or the
sample-based product ``Y = C·S`` (``*_cs``) of noise-free conversion and
selectivity draws plus one yield measurement noise. Selectivity is direct/marginal where
modeled, or the sample-based ratio ``S = Y/C`` for ``*_yc``. The ICM/LCM task
covariance is an unconstrained ``IndexKernel`` (cross-task correlation free in
sign — positive or negative, no bias)."""

MEAN_FUNCTIONS: list[str] = [
    "constant",
    "block_linear",
    "type_linear",
    "type_quadlog",
    "cond_linear",
    "cond_trend",
    "form_offset",
]
"""Every implemented mean function, the keys of ``models._MEAN_BUILDERS``.

``constant`` is the gpytorch-default ``ConstantMean`` and the elimination-grid
baseline. ``block_linear`` is one linear (a, b) term per feature block, ``type_linear`` one
global continuous (a, b) plus four per-pre-cat-form discrete (a, b) pairs, and
``type_quadlog`` a partially pooled temperature-quadratic / log-loading shape per
pre-cat form (see models.py).

``cond_linear`` is ``block_linear`` without the catalyst slopes: one intercept,
slopes on the solvent, base and continuous blocks and a base intercept, the base
terms multiplied by ``base_active``. ``cond_trend`` puts the same coefficients
under ``Normal(0, s² D)`` and integrates them out, which leaves a constant mean
plus the kernel term ``s² h(x)ᵀ D h(x')``. ``form_offset`` is a constant mean plus
an integrated per-pre-catalyst-form offset, the kernel term ``tau² 1[form = form']``.
The kernel terms come from ``models.build_mean_kernel_term``.

``Vanilla`` ignores the mean module (always ConstantMean); it is off-grid."""


def _load_noise_evidence() -> dict[tuple[str, str], ReplicateNoiseEvidence]:
    """Pool the replicate groups of ``experiments.parquet`` into noise evidence.

    Returns:
        ``{(target, outcome_transform): evidence}`` for the targets ``"Y"``,
        ``"C"``, ``"S"`` and every outcome transform.
    """
    return noise_evidence_by_scale(pd.read_parquet(paths.EXPERIMENTS_PARQUET))


NOISE_EVIDENCE: dict[tuple[str, str], ReplicateNoiseEvidence] = _load_noise_evidence()
"""InverseGamma observation-noise prior per ``(target, outcome_transform)``.

Each entry pools the within-replicate sums of squares of one target (``"Y"``
yield, ``"C"`` conversion, ``"S"`` selectivity) on one outcome scale — the values
a GP of that target and transform is trained on — and states
``InverseGamma(α = ν/2, β = S/2)`` in squared units of that scale. The model
builders divide β by the variance of the training targets, which puts the prior
on the standardized scale the GP is fitted on.

Not a stance. Under a Jeffreys reference prior on σ², pooled replicate evidence
carrying ``ν`` degrees of freedom gives exactly ``InverseGamma(ν/2, S/2)``, so
how sharply the prior is stated is fixed by how much repeatability data exists —
``CV = √(2/(ν−4))`` — and tightens on its own as replicate groups accumulate
instead of being picked to look weakly informative. Every entry shares ν.

The pair (α, β) is one object: β alone is meaningless, and overriding α without
re-deriving β silently moves the prior mean ``β/(α−1)`` off the measured
repeatability. A stance that wants to budget for model misspecification on top of
repeatability multiplies β, which moves the mean while leaving the data-derived
tightness intact.

Computed live at import from the replicate groups ``core.replicates`` detects in
``data/libraries/experiments.parquet``. Adding replicate experiments to the
design sheet and rebuilding that table is the only way to move it."""


LENGTHSCALE_LOGNORMAL_LOC_SHIFT: float = 1.4142135623730951
"""Additive offset applied to the LogNormal lengthscale prior location
log(√D). Biases toward longer lengthscales (Hvarfner et al., 2024)."""

LENGTHSCALE_LOGNORMAL_SCALE: float = 2.0
"""Standard deviation σ of the LogNormal lengthscale prior.

This is the Hvarfner et al. (2024) dimensionality-scaled prior and the only
lengthscale prior the study uses; ``PRIOR_SCHEMES`` varies how ``D`` is resolved
and how many lengthscales it applies to, never σ."""

@dataclass(frozen=True)
class PriorScheme:
    """One lengthscale-prior scheme swept on the CV grid.

    A scheme fixes two things together, because they are only meaningful as a
    pair: how many lengthscales the kernel gets (the partition granularity) and
    which ``D`` the Hvarfner ``log sqrt(D)`` prior location resolves to. Stating
    ``log sqrt(D_total)`` over per-block lengthscales, or ``log sqrt(D_block)``
    over a flat ARD vector, would put the prior mode somewhere neither reference
    intends.

    Attributes:
        partition_mode: A value of ``kernels.PARTITION_MODES``.
        lengthscale_dim_mode: A value of ``priors.LENGTHSCALE_DIM_MODES``.
        description: What the scheme is, for the results tables.
    """

    partition_mode: str
    lengthscale_dim_mode: str
    description: str


PRIOR_SCHEMES: dict[str, PriorScheme] = {
    "P1": PriorScheme(
        partition_mode="full_ard",
        lengthscale_dim_mode="total",
        description="Global Hvarfner: full ARD, log sqrt(D_total) location.",
    ),
    "P2": PriorScheme(
        partition_mode="full_ard",
        lengthscale_dim_mode="block",
        description="Block-wise Hvarfner: full ARD, log sqrt(D_block) location.",
    ),
    "P3": PriorScheme(
        partition_mode="block_shared",
        lengthscale_dim_mode="block",
        description="No ARD: one shared lengthscale per block, log sqrt(D_block).",
    ),
}
"""Lengthscale-prior schemes enumerated by the CV grid.

**P1** is the BoTorch default stance (Hvarfner et al., 2024): one lengthscale per
input dimension, all of them under a single LogNormal centred on
``log sqrt(D_total) + shift`` where ``D_total`` is the full design-matrix width.

**P2** keeps full ARD but resolves ``D`` per block, so a 7425-column ``soap``
catalyst block and a 4-column continuous block no longer share a prior mode set
by their sum.

**P3** turns ARD off: one lengthscale per block, still at ``log sqrt(D_block)``.
It is the baseline-performance case, and the only one of the three in which the
partition-and-interpolate degeneracy is structurally unreachable for a product
kernel — no single coordinate can be shrunk on its own.

P1 and P2 fit one lengthscale per column, so on a full-dimensional catalyst
block they fit thousands of them against tens of experiments. That is the regime
the collapse investigation characterised; pairing them with a reduced catalyst
block is what keeps them identifiable."""

DEFAULT_PRIOR_SCHEME: str = "P1"
"""Scheme a new grid starts from when it does not say otherwise.

This is a grid-authoring default only. An inner cell that names no
``prior_scheme`` is **not** silently assigned this one: the scheduler falls back
to the ``"legacy"`` partition and ``LENGTHSCALE_DIM_MODE``, which is what the
archived sweeps were fit under and is none of P1-P3. Re-labelling an old run as
P3 because it happens to be the nearest of the three would be worse than leaving
it unlabelled."""


LENGTHSCALE_DIM_MODE: str = "block"
"""How the LogNormal lengthscale prior location resolves ``D`` in ``log√D``.

``"block"`` uses the width of the block the kernel acts on, ``"total"`` the width
of the whole design matrix (see ``priors.LENGTHSCALE_DIM_MODES``). ``"block"``
is what the P2/P3 prior schemes resolve to and ``"total"`` what P1 does; read by
``kernels.make_kernel_from_config`` when no scheme and no explicit mode is passed."""

LENGTHSCALE_FLOOR: float | None = 0.05
"""Lower bound of the ``GreaterThan`` constraint on every base-kernel lengthscale,
or ``None`` for the default ``Positive`` constraint.

Applied to every fit, under every prior scheme. It is **not** part of the prior:
the prior says where a lengthscale is expected to sit, the floor says where the
optimizer may not go. A lengthscale driven to zero makes the kernel partition the
training points into near-interpolating blocks, which is a numerically
pathological optimum rather than a belief about the surface, so the guard belongs
on the constraint and not in the LogNormal.

0.05 is low enough to be inert wherever the degeneracy is structurally
unreachable — under block-shared lengthscales the floored and unfloored fits
coincide — and high enough to remove the pathological tail. Recorded per run in
``run_meta.lengthscale_floor`` and reported per fold by
``results_analysis/scan_lengthscale_floor.py``."""

TASK_COVAR_VAR_FLOOR: float = 1e-6
"""Lower bound for the IndexKernel ``var`` parameter in multitask models.
Replaces the default ``Positive()`` (bare Softplus) with ``GreaterThan(floor)``.
Prevents Softplus underflow to exactly 0.0 at extreme negative raw_var, which
triggers ``InverseGamma.log_prob(0)`` → ValueError."""


ALPHA_GAMMA_OUTPUTSCALE: float = 3.0
"""Concentration α of the Gamma prior on the kernel outputscale."""

BETA_GAMMA_OUTPUTSCALE: float = 4.0
"""Rate β of the Gamma prior on the kernel outputscale. With
``ALPHA_GAMMA_OUTPUTSCALE`` = 3.0 the prior mean is α/β = 0.75, matching
``OUTPUTSCALE_INIT``."""

OUTPUTSCALE_INIT: float = 0.75
"""Initial value of the kernel outputscale (the KS1 ``ScaleKernel`` and the global
``total_outputscale`` of ``DirichletAdditiveKernel`` for KS2–KS4)."""


KS5_FLOOR_BETA: tuple[float, float] = (1.0, 3.0)
"""Beta(a, b) prior on the KS5 catalyst correlation floor ``rho``. The mode is 0
(KS1) with finite density there, the mean 0.25."""

KS5_FLOOR_INIT: float = 0.25
"""Initial ``rho``: the prior mean."""

KS6_MAIN_SCALE_HALFNORMAL: float = 0.5
"""Scale of the half-normal prior on the KS6 main-effect standard deviation
``s_+``. The prior mean of ``s_+²`` is 0.25 against 0.75 for the product."""

KS6_MAIN_SCALE_INIT: float = 0.4
"""Initial ``s_+``: the prior mean, ``0.5 · sqrt(2/pi)``. A start at 0 would sit on
the saddle where the gradient of ``s_+²`` vanishes."""

KS7_SHARED_SCALE_HALFNORMAL: float = 0.5
"""Scale of the half-normal prior on the KS7 shared-response standard deviation ``s_g``."""

KS7_SHARED_SCALE_INIT: float = 0.4
"""Initial ``s_g``: the prior mean."""

COND_TREND_SCALE_HALFNORMAL: float = 1.0
"""Scale of the half-normal prior on the ``cond_trend`` coefficient scale ``s``. At
``s = 1`` the integrated coefficients have the ``cond_linear`` priors."""

COND_TREND_SCALE_INIT: float = 0.8
"""Initial ``s``: the prior mean, ``sqrt(2/pi)``."""

FORM_OFFSET_SCALE_HALFNORMAL: float = 0.5
"""Scale of the half-normal prior on the ``form_offset`` standard deviation ``tau``."""

FORM_OFFSET_SCALE_INIT: float = 0.4
"""Initial ``tau``: the prior mean."""

CONCENTRATION_DIRICHLET: float = 2.0
"""Concentration of the symmetric Dirichlet prior on the softmax-transformed
additive kernel weights. At 1.0 the density is flat, its gradient is zero and
the weights are unregularized; 2.0 shrinks them gently toward equal weights.
Must stay ≥ 1: in the hyper_prior_scan study conc=0.5 made every fit fail."""


MEAN_MODULE_NORMAL_LOC: float = 0.0
"""Location μ of the Normal prior on every ``BlockLinearMean`` /
``TypeLinearMean`` slope and intercept parameter."""

MEAN_MODULE_NORMAL_SCALE: float = 1.0
"""Standard deviation σ of the Normal prior on every ``BlockLinearMean`` /
``TypeLinearMean`` slope and intercept parameter."""

MEAN_TYPE_DEVIATION_FRAC: float = 0.3
"""Partial-pooling strength of ``TypeQuadLogMean``.

The three shape coefficients (temperature quadratic, temperature linear, log
loading) are a global value plus a per-pre-cat-form deviation; the deviation's
Normal prior is this fraction of the global prior's σ. The 69-experiment table
pins 47 rows at a single (80 °C, 5.25 eq) screening point, and only the
``co2_adduct`` form spans a temperature range wide enough to identify a
curvature of its own, so the deviations are kept tight enough that a form with
no temperature or loading variation falls back onto the global curve."""

N_PRECATALYST_FORM_TYPES: int = 4
"""Number of pre-catalyst-form groups for ``TypeLinearMean``: salt,
free_nhc, co2_adduct, bicarbonate."""


TASK_COVAR_FACTOR_NORMAL_LOC: float = 0.0
"""Location μ of the Normal prior on the IndexKernel's ``covar_factor`` for
ICM/LCM Kronecker multitask models."""

TASK_COVAR_FACTOR_NORMAL_SCALE: float = 1.0
"""Standard deviation σ of the Normal prior on the IndexKernel's
``covar_factor`` for ICM/LCM Kronecker multitask models."""


MAPSAAS_NUM_TAUS: int = 4
"""Number of additive Matérn-5/2 components in ``AdditiveMapSaasSingleTaskGP``
(the ``num_taus`` argument). Each component carries its own SAAS sparsity level
and outputscale; MAP inference learns all of them."""

CONVERSION_DIVISION_FLOOR: float = 1e-2
"""Lower clamp on conversion samples when deriving selectivity as ``S = Y / C``
for the ``*_yc`` models. Guards the division against near-zero conversion; the
resulting selectivity is then clipped to ``[0, 1]``."""

LOGSUM_ZERO_YIELD: float = 1e-3
"""Raw yield at which ``logsum_cy`` models a row whose recorded yield is 0.

A recorded 0 is a true zero, which has no logarithm. 0.1% is the value every
bounded outcome transform already maps a zero yield to (``LOGIT_EPS``)."""

LOGSUM_ZERO_NOISE_ANCHOR: float = 1e-2
"""Raw yield at which the log-scale noise of a zero-yield row is evaluated in
``logsum_cy``: 1%, the smallest non-zero yield the design sheet records, so a zero
row is as precise on the log scale as a 1% row (``σ_Y / 0.01``)."""

LOGSUM_NOISE_MULTIPLIER_FLOOR: float = 1e-4
"""Lower bound of the ``logsum_cy`` noise multipliers, each a multiple of its
target's pooled raw replicate variance."""


CAT_LEVELS: tuple[str, ...] = (
    "clamp", "clamp_pca",
    "mordred", "mordred_pca",
    "mol2vec", "mol2vec_pca",
    "chembert2", "chembert2_pca",
    "molformer_xl_precat", "molformer_xl_precat_pca",
    "molformer_xl_carbene", "molformer_xl_carbene_pca",
    "unimol", "unimol_pca",
    "grover", "grover_pca",
    "soap", "soap_pca",
    "minimol_precat", "minimol_precat_pca",
    "minimol_carbene", "minimol_carbene_pca",
    "scage_precat", "scage_precat_pca",
    "scage_carbene", "scage_carbene_pca",
    "graphormer_precat", "graphormer_precat_pca",
    "graphormer_carbene", "graphormer_carbene_pca",
    "infomax3d_precat", "infomax3d_precat_pca",
    "infomax3d_carbene", "infomax3d_carbene_pca",
    "frad_precat", "frad_precat_pca",
    "frad_carbene", "frad_carbene_pca",
    "transformerm_precat", "transformerm_precat_pca",
    "transformerm_carbene", "transformerm_carbene_pca",
    "interpretable_steric",
    "interpretable_light",
    "interpretable",
    "interpretable_dG",
    "ecfp4", "ecfp4_pca",
    "map4", "map4_pca",
)
"""Catalyst levels: every featurization, full-dimensional and PCA-reduced.

The interpretable featurizations have no ``_pca`` level. They exist so that a
fitted lengthscale or slope maps onto a named chemical quantity — ``vbur_q2``,
``fukui_plus_c2``, ``xtb_gap_eV`` — and a rotation into principal components
destroys exactly that, for no gain: at 10-30 columns against 69 experiments there
is no dimensionality problem to solve, and the knee discards ~30% of the variance
to take ``interpretable`` from 23 columns to 5."""


@dataclass(frozen=True)
class EliminationGrid:
    """The CV grid: one baseline plus one-axis sweeps, per catalyst level.

    Every configuration is either the baseline or the baseline with exactly one
    swept axis moved to another of its levels. The grid is therefore the union
    of the sweeps, not their cartesian product, and the baseline configuration —
    which every sweep contains — is enumerated once.

    Attributes:
        cat_feats: Levels of the catalyst axis, in enumeration order. Each is a
            directory under ``data/featurizations/``; a ``<name>_pca`` level is
            the stored PCA reduction of ``<name>`` (``features.dim_reduction``).
            Everything else about a level — its ``CATALYST_SETS`` entry and its
            form-one-hot flag — is derived from the name.
        baseline: The level of every non-catalyst axis, keyed by its
            ``inputs.json`` name: ``solv_feat``, ``base_feat``, ``base_kernel``,
            ``kernel_structure``, ``mean_function``, ``outcome_transform``,
            ``prior_scheme``, ``model_class``.
        sweeps: The axes varied one at a time, each mapped to all of its levels
            in reporting order. Every level tuple contains the baseline level.
        note: One line on what the grid is for.
    """

    cat_feats: tuple[str, ...]
    baseline: dict[str, str]
    sweeps: dict[str, tuple[str, ...]]
    note: str

    def __post_init__(self) -> None:
        for axis, levels in self.sweeps.items():
            if axis not in self.baseline:
                raise ValueError(f"swept axis {axis!r} has no baseline level")
            if self.baseline[axis] not in levels:
                raise ValueError(
                    f"sweep {axis!r} levels {levels} omit the baseline "
                    f"{self.baseline[axis]!r}"
                )
            if len(set(levels)) != len(levels):
                raise ValueError(f"sweep {axis!r} repeats a level: {levels}")


ELIMINATION_GRID: EliminationGrid = EliminationGrid(
    cat_feats=CAT_LEVELS,
    baseline={
        "solv_feat": "OHE",
        "base_feat": "OHE",
        "base_kernel": "matern52",
        "kernel_structure": "KS1",
        "mean_function": "constant",
        "outcome_transform": "raw",
        "prior_scheme": "P1",
        "model_class": "st_y",
    },
    sweeps={
        "kernel_structure": ("KS1", "KS2", "KS3", "KS4"),
        "prior_scheme": ("P1", "P2", "P3"),
        "outcome_transform": ("raw", "logit", "arcsin_sqrt", "probit"),
        "mean_function": ("constant", "block_linear", "type_linear", "type_quadlog"),
    },
    note=(
        "Elimination phase: every catalyst level at one baseline, plus four "
        "one-axis sweeps (kernel structure, prior scheme, outcome transform, "
        "mean function). A level survives if its gem clears the threshold in "
        "every configuration."
    ),
)
"""The grid ``generate_inputs`` enumerates.

50 catalyst levels × 12 configurations (1 baseline + 3 + 2 + 3 + 3 non-baseline
levels) = 600 configurations, one outer cell each. Solvent block, base block and
base kernel stay at the baseline throughout; ``matern52`` is the base kernel of
every level, the full-dimensional fingerprints included."""


@dataclass(frozen=True)
class CrossedGrid:
    """A CV grid that is the full cartesian product of its axes.

    Every combination of one catalyst level with one level of every other axis is
    a configuration. An axis held fixed is an axis with a single level.

    Attributes:
        cat_feats: Levels of the catalyst axis, in enumeration order; resolved as
            in ``EliminationGrid``.
        axes: Every non-catalyst axis, keyed by its ``inputs.json`` name, mapped
            to its levels in enumeration order. The keys are those of
            ``ELIMINATION_GRID.baseline``.
        note: One line on what the grid is for.
    """

    cat_feats: tuple[str, ...]
    axes: dict[str, tuple[str, ...]]
    note: str

    def __post_init__(self) -> None:
        if not self.cat_feats or len(set(self.cat_feats)) != len(self.cat_feats):
            raise ValueError(f"cat_feats must be non-empty and distinct: {self.cat_feats}")
        expected = set(ELIMINATION_GRID.baseline)
        if set(self.axes) != expected:
            raise ValueError(
                f"axes {sorted(self.axes)} must be exactly {sorted(expected)}"
            )
        for axis, levels in self.axes.items():
            if not levels or len(set(levels)) != len(levels):
                raise ValueError(f"axis {axis!r} levels must be non-empty and distinct: {levels}")


CROSSED_GRID_1: CrossedGrid = CrossedGrid(
    cat_feats=(
        "grover_pca",
        "soap_pca",
        "interpretable_steric",
        "unimol",
        "minimol_precat_pca",
        "scage_carbene",
        "transformerm_precat_pca",
    ),
    axes={
        "solv_feat": ("KT", "Hansen", "Gutmann", "OHE", "grover_pca", "soap_pca"),
        "base_feat": ("thermo", "steric", "OHE", "grover_pca", "soap_pca"),
        "base_kernel": ("rbf", "matern52"),
        "kernel_structure": ("KS1",),
        "mean_function": ("constant",),
        "outcome_transform": ("raw", "probit", "arcsin_sqrt"),
        "prior_scheme": ("P1", "P2"),
        "model_class": ("st_y",),
    },
    note=(
        "cross_1: seven catalyst levels crossed with every solvent block, base "
        "block, base kernel, three outcome transforms and two prior schemes at "
        "KS1 / constant mean / st_y."
    ),
)
"""7 catalyst levels × 6 solvent × 5 base × 2 base kernels × 3 transforms × 2 prior
schemes = 2,520 configurations, one outer cell each."""


@dataclass(frozen=True)
class ListedGrid:
    """A CV grid of explicitly listed configurations, each fitted with every catalyst level.

    Attributes:
        cat_feats: Levels of the catalyst axis, in enumeration order; resolved as
            in ``EliminationGrid``.
        configs: The non-catalyst configurations, each over exactly the keys of
            ``ELIMINATION_GRID.baseline``, in enumeration order.
        note: One line on what the grid is for.
    """

    cat_feats: tuple[str, ...]
    configs: tuple[dict[str, str], ...]
    note: str

    def __post_init__(self) -> None:
        if not self.cat_feats or len(set(self.cat_feats)) != len(self.cat_feats):
            raise ValueError(f"cat_feats must be non-empty and distinct: {self.cat_feats}")
        if not self.configs:
            raise ValueError("configs must be non-empty")
        expected = set(ELIMINATION_GRID.baseline)
        keys = set()
        for cfg in self.configs:
            if set(cfg) != expected:
                raise ValueError(f"config {cfg} must have exactly the keys {sorted(expected)}")
            key = tuple(sorted(cfg.items()))
            if key in keys:
                raise ValueError(f"config {cfg} is listed twice")
            keys.add(key)

    @property
    def axes(self) -> dict[str, tuple[str, ...]]:
        """Every non-catalyst axis mapped to the levels the listed configurations use."""
        return {
            axis: tuple(dict.fromkeys(cfg[axis] for cfg in self.configs))
            for axis in ELIMINATION_GRID.baseline
        }


def _m3_config(
    solv_feat: str, base_feat: str, base_kernel: str, outcome_transform: str, prior_scheme: str
) -> dict[str, str]:
    """One ``monroe_m3`` configuration at KS1 / constant mean / ``st_y``."""
    return {
        "solv_feat": solv_feat,
        "base_feat": base_feat,
        "base_kernel": base_kernel,
        "kernel_structure": "KS1",
        "mean_function": "constant",
        "outcome_transform": outcome_transform,
        "prior_scheme": prior_scheme,
        "model_class": "st_y",
    }


MONROE_M3_GRID: ListedGrid = ListedGrid(
    cat_feats=("grover_pca", "monroe_precat_pca", "monroe_precat_pca8"),
    configs=(
        _m3_config("Hansen", "grover_pca", "matern52", "probit", "P2"),
        _m3_config("Hansen", "steric", "matern52", "arcsin_sqrt", "P2"),
        _m3_config("Hansen", "OHE", "matern52", "probit", "P2"),
        _m3_config("grover_pca", "grover_pca", "matern52", "probit", "P2"),
        _m3_config("Hansen", "steric", "matern52", "probit", "P2"),
        _m3_config("Gutmann", "OHE", "matern52", "arcsin_sqrt", "P1"),
        _m3_config("Gutmann", "steric", "rbf", "arcsin_sqrt", "P1"),
        _m3_config("Gutmann", "steric", "matern52", "arcsin_sqrt", "P1"),
        _m3_config("Gutmann", "steric", "matern52", "arcsin_sqrt", "P2"),
        _m3_config("Gutmann", "steric", "matern52", "probit", "P1"),
    ),
    note=(
        "monroe_m3: the ten best cross_1 configurations by gem, each with the "
        "grover_pca catalyst block (control), Monroe at its knee (15 PCs) and "
        "Monroe at 8 PCs."
    ),
)
"""3 catalyst levels × 10 configurations = 30 configurations, one outer cell each.

The configurations are cross_1 gem ranks 1-10 (``runs/cross_1/gem_scores.parquet``,
all ``grover_pca``; rank 11 scores 0.0011 below rank 10), in rank order. The
``grover_pca`` cells are refitted rather than read from cross_1, so all three
catalyst levels share one code version and one noise prior."""


@dataclass(frozen=True)
class AnchoredGrid:
    """A CV grid of anchor configurations, each crossed with variant axes.

    Every anchor is a full configuration, catalyst level included; every
    combination of the variant axes' levels replaces those axes in each anchor.

    Attributes:
        anchors: Full configurations over ``cat_feat`` and the keys of
            ``ELIMINATION_GRID.baseline``, in enumeration order.
        variants: Axes, each a key of ``ELIMINATION_GRID.baseline``, mapped to
            their levels in enumeration order; the cartesian product of the
            levels is applied to every anchor, the last axis varying fastest.
        note: One line on what the grid is for.
    """

    anchors: tuple[dict[str, str], ...]
    variants: dict[str, tuple[str, ...]]
    note: str

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("anchors must be non-empty")
        expected = {"cat_feat", *ELIMINATION_GRID.baseline}
        seen = set()
        for anchor in self.anchors:
            if set(anchor) != expected:
                raise ValueError(f"anchor {anchor} must have exactly the keys {sorted(expected)}")
            key = tuple(sorted(anchor.items()))
            if key in seen:
                raise ValueError(f"anchor {anchor} is listed twice")
            seen.add(key)
        if not self.variants:
            raise ValueError("variants must be non-empty")
        for axis, levels in self.variants.items():
            if axis not in ELIMINATION_GRID.baseline:
                raise ValueError(f"variant axis {axis!r} is not a configuration axis")
            if not levels or len(set(levels)) != len(levels):
                raise ValueError(f"variant {axis!r} levels must be non-empty and distinct: {levels}")

    @property
    def cat_feats(self) -> tuple[str, ...]:
        """The anchors' catalyst levels, in anchor order, each once."""
        return tuple(dict.fromkeys(anchor["cat_feat"] for anchor in self.anchors))

    @property
    def configs(self) -> tuple[dict[str, str], ...]:
        """Every configuration, ``cat_feat`` included: anchor-major, variants crossed."""
        combos = list(itertools.product(*self.variants.values()))
        return tuple(
            {**anchor, **dict(zip(self.variants, combo))}
            for anchor in self.anchors for combo in combos
        )

    @property
    def axes(self) -> dict[str, tuple[str, ...]]:
        """Every non-catalyst axis mapped to the levels its configurations use."""
        return {
            axis: tuple(dict.fromkeys(cfg[axis] for cfg in self.configs))
            for axis in ELIMINATION_GRID.baseline
        }


def _km1_anchor(
    cat_feat: str,
    solv_feat: str,
    base_feat: str,
    base_kernel: str,
    outcome_transform: str,
    prior_scheme: str,
) -> dict[str, str]:
    """One ``kernel_mean_1`` anchor at KS1 / constant mean / ``st_y``."""
    return {
        "cat_feat": cat_feat,
        "solv_feat": solv_feat,
        "base_feat": base_feat,
        "base_kernel": base_kernel,
        "kernel_structure": "KS1",
        "mean_function": "constant",
        "outcome_transform": outcome_transform,
        "prior_scheme": prior_scheme,
        "model_class": "st_y",
    }


KERNEL_MEAN_1_GRID: AnchoredGrid = AnchoredGrid(
    anchors=(
        _km1_anchor("grover_pca", "Hansen", "grover_pca", "matern52", "probit", "P2"),
        _km1_anchor("soap_pca", "Gutmann", "grover_pca", "rbf", "probit", "P2"),
        _km1_anchor("minimol_precat_pca", "grover_pca", "OHE", "rbf", "arcsin_sqrt", "P2"),
        _km1_anchor("transformerm_precat_pca", "grover_pca", "OHE", "rbf", "probit", "P2"),
        _km1_anchor("scage_carbene", "Gutmann", "thermo", "rbf", "probit", "P1"),
        _km1_anchor("interpretable_steric", "Gutmann", "thermo", "rbf", "probit", "P2"),
    ),
    variants={
        "kernel_structure": ("KS1", "KS2", "KS5", "KS6"),
        "mean_function": ("constant", "cond_linear", "cond_trend", "form_offset"),
    },
    note=(
        "kernel_mean_1: the best cross_1 configuration of six catalyst levels "
        "(unimol excluded), crossed with KS1, KS2, KS5, KS6 and the constant, "
        "cond_linear, cond_trend and form_offset means."
    ),
)
"""6 anchors × 4 kernel structures × 4 mean functions = 96 configurations, one
outer cell each, 13,152 fits.

Each anchor is the highest-gem configuration of its catalyst level in
``runs/cross_1/gem_scores.parquet`` (gem +0.278 to +0.425); the seventh
``cross_1`` level, ``unimol``, is not an anchor. KS7 is implemented but not on
this grid. The KS1 / constant cell of each anchor refits it under the current
code and noise prior, and is the paired control the other 15 cells of that
anchor are compared with."""


def _p1_config(model_class: str, outcome_transform: str) -> dict[str, str]:
    """One ``product_1`` configuration at the cross_1 rank-1 setting."""
    return {
        "solv_feat": "Hansen",
        "base_feat": "grover_pca",
        "base_kernel": "matern52",
        "kernel_structure": "KS1",
        "mean_function": "constant",
        "outcome_transform": outcome_transform,
        "prior_scheme": "P2",
        "model_class": model_class,
    }


PRODUCT_1_GRID: ListedGrid = ListedGrid(
    cat_feats=("grover_pca",),
    configs=(
        _p1_config("st_y", "probit"),
        _p1_config("st_y", "raw"),
        _p1_config("lcm_cs", "raw"),
        _p1_config("lcm_cs", "probit"),
        _p1_config("logsum_cy", "raw"),
        _p1_config("condprod_cy", "probit"),
    ),
    note=(
        "product_1: the product-structured classes logsum_cy and condprod_cy against "
        "st_y and lcm_cs at the cross_1 rank-1 setting."
    ),
)
"""1 catalyst level × 6 configurations = 6 configurations, one outer cell each.

The setting is cross_1 gem rank 1: ``grover_pca`` catalyst, Hansen solvent,
``grover_pca`` base, Matérn-5/2, KS1, constant mean, P2. ``st_y`` on probit is the
production model; ``st_y`` raw and ``lcm_cs`` on raw and probit are the
yield-only and the sampled-product references. ``logsum_cy`` runs on ``raw`` and
``condprod_cy`` on ``probit``, the only transforms each accepts."""


RUN_GRIDS: dict[str, EliminationGrid | CrossedGrid | ListedGrid | AnchoredGrid] = {
    paths.ELIMINATION_RUN: ELIMINATION_GRID,
    "cross_1": CROSSED_GRID_1,
    "monroe_m3": MONROE_M3_GRID,
    "kernel_mean_1": KERNEL_MEAN_1_GRID,
    "product_1": PRODUCT_1_GRID,
}
"""Every run the CV tools can target with ``--run``, mapped to the grid it
enumerates. A run's folders come from ``paths.run_paths`` under the same name."""

DEFAULT_RUN: str = paths.ELIMINATION_RUN
"""Run a CV tool targets when ``--run`` is not given."""


N_RESTARTS: int = 5
"""Number of independent L-BFGS restarts used when fitting each GP."""

MAX_ITER: int = 2000
"""Maximum L-BFGS iterations per restart."""

RESTART_JITTER_SD: float = 0.1
"""Every restart after the first starts from the initial hyperparameters plus
N(0, RESTART_JITTER_SD²) on each raw (unconstrained) parameter."""

FLOOR_PLATEAU_RAW: float = -10.0
"""A raw parameter below this value under a ``GreaterThan(lb)`` softplus constraint
sits within 4.5e-5 of ``lb`` with gradient ≈ 0: L-BFGS cannot leave it."""

CHOLESKY_JITTER_F64: float = 1e-6
"""Cholesky jitter applied during GP fitting at float64 precision."""

JITTER_THRESHOLD: float = 1e-3
"""If Cholesky jitter escalates to this level or above during fitting, the
fold is treated as a numerical failure and skipped."""

MIN_NOISE_TO_JITTER_RATIO: float = 10.0
"""A fold fails when the fitted likelihood noise variance is below this multiple
of the final Cholesky jitter, both on the standardized scale the GP is fitted on.
Below it the diagonal the posterior relies on is jitter rather than learned noise."""


FB_MAX_DIVERGENCES: int = 0
"""Largest number of post-warmup NUTS divergences, summed over chains, that a
fully Bayesian fit may have before ``FullyBayesianScanGP.check_diagnostics``
raises. A potential evaluation whose covariance is not positive definite has log
density -inf and counts as a divergence when it happens after warmup."""

FB_RHAT_MAX: float = 1.01
"""Largest split-R̂ over every scalar hyperparameter of a fully Bayesian fit."""

FB_ESS_MIN: float = 100.0
"""Smallest bulk effective sample size over every scalar hyperparameter of a fully
Bayesian fit, computed on all post-warmup draws before thinning."""


FOLD_TOTAL: int = 137
"""Total CV folds per inner config: 10×6 (primary stratified) + 2 (type) + 3
(scaffold) + 7 (solvent) + 5 (replicate) + 60 (LOO).

Both replicate-dependent counts follow the groups ``core.replicates`` detects,
so they move when replicate experiments are added. At the current 69 experiments
there are 5 groups (four triplets and one doublet, 14 experiments): one
leave-one-replicate-out fold each, and LOO emits 60 folds rather than 69 because
those 14 collapse into 5 leave-group-out folds — no fold may put one replicate in
test while another trains."""


SLURM_PARTITION: str = "medium"
"""Mechthild SLURM partition for CV worker submission. ``medium`` caps the wall
time at 24 h; every submission passes ``--time`` explicitly
(``SLURM_TIME_LIMIT`` / ``SLURM_SINGLE_CELL_TIME_LIMIT``), just under that cap."""

SLURM_CPUS_PER_TASK: int = 1
"""CPU cores allocated per worker task."""

SLURM_MEMORY: str = "4G"
"""Memory request per worker task. The worker loads the raw, full-width
``descriptors.parquet`` of its cat/solv/base blocks; at 69 experiments the
assembled design matrix stays under 4 MB even for the 7425-dim ``soap`` block.
The binding cost is the ``UnitMinMaxKernel`` pairwise path, which materializes
``(n1, n2, d)`` — ~80 MB on a 2048-bit fingerprint block and ~280 MB at
``soap``'s 7425-column width (n1 = n2 = 69, float64). 4G keeps ≥10× headroom
over that peak."""

SLURM_TIME_LIMIT: str = "23:59:00"
"""Wall-clock limit per draining worker, which runs cells until the queue is
empty. Fold-level resume covers an overrun: ``scheduler init`` requeues the cell
and the next worker continues from its last finished fold."""

SLURM_SINGLE_CELL_TIME_LIMIT: str = "23:59:00"
"""Wall-clock limit per one-cell task (``submit_mechthild --single``). The
slowest cell of the elimination grid is ``soap`` under a ``*_linear`` or
``type_quadlog`` mean, projected by ``scheduler estimate-runtime``."""

SLURM_EMAIL: str = "enisbaris@mpi-magdeburg.mpg.de"
"""Email for SLURM notifications."""

SLURM_MAX_CONCURRENT_WORKERS: int = 300
"""Throttle on simultaneously-running CV worker jobs. Applied two ways: as the
SLURM array ``%N`` cap (``--array=1-K%N``, enforced by the SLURM controller) and
as the submitter's ceiling on active plus pending workers. It is a
cluster-politeness guard, **not** a correctness requirement — the file-based queue
lets any number of workers atomically claim cells without collision.

The 600-cell elimination grid therefore runs as two rounds of 300 one-cell tasks:
``submit_mechthild submit --single`` fills the 300 slots, and ``autoscale
--single`` hands each slot a round-1 task frees to the next pending cell. The
cluster's own QOS (ASSOCMAX/QOSMAX) can cap below this; the submitter detects that
throttle (``_capacity_snapshot``)."""

SLURM_POLL_INTERVAL_SECONDS: int = 30
"""Sleep interval between submission/queue-state polls during autoscale."""
