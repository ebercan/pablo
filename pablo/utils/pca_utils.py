"""PCA pipeline utilities for featurization.

Provides a PCAPipeline dataclass that wraps a StandardScaler + PCA fit on the
full 161-molecule library.  The fitted objects are serialised to disk and loaded
during CV without refitting.

Critical invariant:
    PCA + scaler are fit ONCE on all 161 library molecules — never inside a CV
    fold.  CV folds only use continuous-feature scalers fitted on the training
    split.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class _RemappingUnpickler(pickle.Unpickler):
    """Unpickler that remaps legacy module paths for PCAPipeline."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if name == "PCAPipeline" and module != __name__:
            module = __name__
        return super().find_class(module, name)


@dataclass
class PCAPipeline:
    """Fitted StandardScaler + PCA pipeline for a molecular library.

    Attributes:
        scaler: Fitted StandardScaler (on the 161-molecule library).
        pca: Fitted PCA (all components, on the standardized library).
        n_components: Number of components selected by kneedle (or fallback).
        explained_variance_ratios: Per-component explained variance ratios.
        kneedle_sensitivity: Sensitivity used when selecting n_components.
        metadata: Dimensionality, molecule count, pipeline label and total
            explained variance.
    """

    scaler: StandardScaler
    pca: PCA
    n_components: int
    explained_variance_ratios: list[float]
    kneedle_sensitivity: float = 1.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        kneedle_sensitivity: float = 1.0,
        label: str = "",
        force_n_components: Optional[int] = None,
    ) -> "PCAPipeline":
        """Fit StandardScaler then PCA on the library matrix X.

        Args:
            X: (n_molecules, d_raw) float64 array of raw descriptors/fingerprints.
            kneedle_sensitivity: Sensitivity for KneeLocator (lower → more components).
            label: Optional label for progress messages.
            force_n_components: Optional hard override for target dimensionality.

        Returns:
            Fitted PCAPipeline.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float64))

        pca = PCA(random_state=42)
        pca.fit(X_scaled)

        var_ratios = pca.explained_variance_ratio_.tolist()
        # Sort descending — PCA already returns in descending order but be explicit.
        var_sorted = sorted(var_ratios, reverse=True)

        if force_n_components is not None:
            n_comp = force_n_components
        else:
            kneedle = KneeLocator(
                range(1, len(var_sorted) + 1),
                var_sorted,
                S=kneedle_sensitivity,
                curve="convex",
                direction="decreasing",
            )
            n_comp = kneedle.knee

            if n_comp is None:
                cumulative = np.cumsum(var_sorted)
                n_comp = int(np.argmax(cumulative >= 0.80)) + 1

        prefix = f"[{label}] " if label else ""
        total_var = sum(var_sorted[:n_comp])
        print(
            f"{prefix}PCA: {X.shape[1]} → {n_comp} components  "
            f"({total_var:.1%} explained variance)"
        )

        return cls(
            scaler=scaler,
            pca=pca,
            n_components=n_comp,
            explained_variance_ratios=var_sorted,
            kneedle_sensitivity=kneedle_sensitivity,
            metadata={
                "n_input_dims": int(X.shape[1]),
                "n_molecules": int(X.shape[0]),
                "label": label,
                "total_explained_variance": float(total_var),
            },
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler + PCA to new data.

        Args:
            X: (n, d_raw) float64 array.  Must have same d_raw as the fit data.

        Returns:
            (n, n_components) float64 array of PCA embeddings.
        """
        X_scaled = self.scaler.transform(X.astype(np.float64))
        return self.pca.transform(X_scaled)[:, : self.n_components]

    def save(self, path: Path) -> None:
        """Serialise to a pickle file.

        Args:
            path: Destination path (created including parents).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "PCAPipeline":
        """Load a previously saved PCAPipeline from disk.

        Args:
            path: Path to the pickle file.

        Returns:
            Loaded PCAPipeline instance.
        """
        with open(path, "rb") as fh:
            obj = _RemappingUnpickler(fh).load()
        assert isinstance(obj, cls), f"Unexpected type in {path}: {type(obj)}"
        return obj
