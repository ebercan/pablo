"""Frozen PCA-knee reductions of the catalyst, solvent and base featurizations.

A reduced featurization is written to its own directory, ``<name>_pca/``, and
from then on is just another entry on its axis (``grover_pca`` on the catalyst
axis, ``solvent_grover_pca`` on the solvent axis). Fitting it once and
storing the result — rather than reducing inside each CV worker — is what makes
the reduction auditable: the retained variance, the knee, and the per-component
explained-variance ratios are recorded in ``metadata.json`` next to the reduced
descriptors, and every fold of every cell provably sees the same embedding.

**The fit is frozen, full-catalog, and X-only.** It runs over the whole
library — all 161 catalysts, not the ~25 that appear in the experiments, and
all 14 solvents or 8 bases — and it never sees a yield. So it leaks no outcome information into a fold, which is what would
invalidate the CV estimate; it does use the test catalysts' *features*, which is
the standard transductive-featurization convention this study already follows
for every other block and is the only way a reduction can be shared across folds
at all.

The component count comes from the Kneedle knee of the explained-variance curve.
The knee is a per-featurization quantity, not a constant information budget: it
retains 44% of the variance on ``ecfp4`` and 94% on ``infomax3d``. Both numbers
are recorded in ``metadata.json`` so a comparison along the reduction axis can
be read with that in mind.

A reduction to a fixed component count ``k`` is stored as ``<name>_pca<k>/``
(``monroe_precat_pca8``). It is the same frozen fit truncated at ``k`` instead
of at the knee, so its columns are the first ``k`` columns of ``<name>_pca/``
whenever ``k`` does not exceed the knee.
"""

from __future__ import annotations

import json
import re

import kneed
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pablo.featurization_and_kernel_scan import paths

PCA_SUFFIX = "_pca"
"""Directory suffix marking a reduced featurization."""

_REDUCED_NAME = re.compile(rf"^(?P<source>.+){PCA_SUFFIX}(?P<k>\d*)$")
"""``<source>_pca`` (knee) or ``<source>_pca<k>`` (fixed component count)."""

KNEEDLE_SENSITIVITY: int = 1
"""Kneedle ``S``. Higher is more conservative (fewer, later knees)."""

RANDOM_STATE: int = 42
"""Seed for the PCA solver, so a rebuild reproduces the stored matrix."""

VARIANCE_FALLBACK: float = 0.80
"""Cumulative variance target used when Kneedle finds no knee at all."""


def reduced_name(featurization: str, n_components: int | None = None) -> str:
    """The directory name of a featurization's PCA reduction.

    Args:
        featurization: The full-dimensional featurization.
        n_components: A fixed component count, or ``None`` for the knee.

    Returns:
        ``<featurization>_pca`` for the knee, ``<featurization>_pca<k>`` otherwise.
    """
    return f"{featurization}{PCA_SUFFIX}{'' if n_components is None else n_components}"


def is_reduced(featurization: str) -> bool:
    """Whether a featurization name refers to a stored PCA reduction."""
    return _REDUCED_NAME.match(featurization) is not None


def source_name(featurization: str) -> str:
    """The full-dimensional featurization a name refers to (itself if unreduced)."""
    match = _REDUCED_NAME.match(featurization)
    return featurization if match is None else match.group("source")


def build_pca_featurization(
    featurization: str, extra_meta: dict | None = None, n_components: int | None = None
) -> dict:
    """Fit and store the frozen PCA reduction of one featurization.

    Args:
        featurization: Directory name under ``paths.FEAT_DIR`` holding the
            full-dimensional ``descriptors.parquet``, indexed by catalyst,
            solvent or base id.
        extra_meta: Additional ``metadata.json`` keys.
        n_components: Keep this many components and write ``<name>_pca<k>/``;
            ``None`` keeps the Kneedle knee and writes ``<name>_pca/``.

    Returns:
        The metadata written alongside the reduced descriptors.

    Raises:
        FileNotFoundError: If the source featurization has no descriptors.
        ValueError: If the source is already a reduction, carries NaN, or has
            fewer than ``n_components`` components.
    """
    if is_reduced(featurization):
        raise ValueError(f"{featurization!r} is already a reduction; refusing to nest.")

    source_dir = paths.FEAT_DIR / featurization
    source_path = source_dir / "descriptors.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"No descriptors.parquet under {source_dir}")

    df = pd.read_parquet(source_path)
    X = df.to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(f"{source_path} carries non-finite values; cannot PCA it.")

    scaler = StandardScaler().fit(X)
    pca = PCA(random_state=RANDOM_STATE).fit(scaler.transform(X))
    explained = np.sort(pca.explained_variance_ratio_)[::-1]
    cumulative = np.cumsum(explained)

    knee = kneed.KneeLocator(
        range(1, len(explained) + 1),
        explained.tolist(),
        S=KNEEDLE_SENSITIVITY,
        curve="convex",
        direction="decreasing",
    ).knee
    used_fallback = knee is None
    if used_fallback:
        knee = int(np.argmax(cumulative >= VARIANCE_FALLBACK)) + 1
    fixed_count = n_components
    if fixed_count is None:
        n_components = int(knee)
    elif not 1 <= fixed_count <= len(explained):
        raise ValueError(
            f"{featurization} has {len(explained)} components; cannot keep {fixed_count}."
        )
    else:
        n_components = fixed_count

    embedding = pca.transform(scaler.transform(X))[:, :n_components]
    columns = [f"{featurization}_pc{i}" for i in range(n_components)]
    reduced = pd.DataFrame(embedding, index=df.index, columns=columns)

    out_dir = paths.FEAT_DIR / reduced_name(featurization, fixed_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(out_dir / "descriptors.parquet")

    metadata = {
        "n_molecules": int(reduced.shape[0]),
        "n_features": n_components,
        "feature_cols": columns,
        "descriptors_file": "descriptors.parquet",
        "method": (
            "Frozen PCA of the standardized full-dimensional featurization, fit "
            f"once over the whole {df.shape[0]}-member {df.index.name} library "
            "with no access to any outcome. "
            + (
                "Component count from the Kneedle knee of the explained-variance curve."
                if fixed_count is None
                else f"Component count fixed at {fixed_count}; the knee is recorded alongside."
            )
        ),
        "source_featurization": featurization,
        "source_n_features": int(df.shape[1]),
        **({} if fixed_count is None else {"knee": int(knee)}),
        "kneedle_sensitivity": KNEEDLE_SENSITIVITY,
        "knee_from_fallback": bool(used_fallback),
        "variance_retained": float(cumulative[n_components - 1]),
        "n_components_for_90pct": int(np.argmax(cumulative >= 0.90)) + 1,
        "n_components_for_95pct": int(np.argmax(cumulative >= 0.95)) + 1,
        "explained_variance_ratio": [float(v) for v in explained[:n_components]],
        "random_state": RANDOM_STATE,
        **(extra_meta or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def build_all(featurizations: list[str]) -> pd.DataFrame:
    """Build every named reduction and return a summary table.

    Args:
        featurizations: Full-dimensional featurization directory names.

    Returns:
        One row per reduction: source width, knee, and variance retained.
    """
    rows = []
    for name in featurizations:
        meta = build_pca_featurization(name)
        rows.append({
            "featurization": name,
            "raw_dim": meta["source_n_features"],
            "knee": meta["n_features"],
            "variance_retained": meta["variance_retained"],
            "d90": meta["n_components_for_90pct"],
            "d95": meta["n_components_for_95pct"],
        })
        print(
            f"✓ {reduced_name(name):28s} {meta['source_n_features']:5d} -> "
            f"{meta['n_features']:3d} pc ({100 * meta['variance_retained']:.1f}% var)"
        )
    return pd.DataFrame(rows)
