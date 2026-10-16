"""Dimensionality reduction utilities (PCA, UMAP) for featurized chemical data.

Module: pablo.utils.dim_reduction
"""

import kneed
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import LabelEncoder, StandardScaler


def excel_to_df(file_path, sheet_name=None):
    """Load one Excel sheet into a DataFrame with the openpyxl engine.

    Args:
        file_path: Path to the ``.xlsx`` file.
        sheet_name: Sheet name or index; the first sheet when None.

    Returns:
        The sheet as a DataFrame.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    target_sheet = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(file_path, sheet_name=target_sheet, engine="openpyxl")
    print(f"Data Loaded. Shape: {df.shape}")
    return df


def apply_umap(
    df,
    n_components=10,
    n_neighbors=20,
    min_dist=0.1,
):
    """Standardize the features and embed them with UMAP.

    Args:
        df: ``N`` rows; first column holds the labels, the remaining ``D``
            columns the numeric features.
        n_components: Embedding dimension ``d``.
        n_neighbors: UMAP neighbourhood size, also used for the
            trustworthiness score.
        min_dist: UMAP minimum distance between embedded points.

    Returns:
        Tuple ``(labels_tensor, values_tensor, trustworthiness_score)``:
        - labels_tensor: ``N x 1`` long tensor of label-encoded first-column
          values.
        - values_tensor: ``N x d`` float64 tensor of the embedding.
        - trustworthiness_score: float in [0, 1]; higher means local
          neighbourhoods are better preserved.
    """

    raw_labels = df.iloc[:, 0].values
    X_raw = df.iloc[:, 1:].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long).unsqueeze(1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(
        f"Running UMAP (neighbors={n_neighbors}, min_dist={min_dist}, d={n_components})..."
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=42,
        init="spectral",
    )

    embedding = reducer.fit_transform(X_scaled)

    score = trustworthiness(X_scaled, embedding, n_neighbors=n_neighbors)

    values_tensor = torch.from_numpy(embedding).to(torch.float64)

    return labels_tensor, values_tensor, score


def apply_pca(df, n_components=None, kneedle_sensitivity: int = 1.0):
    """Standardize the features and project them onto their leading PCs.

    Args:
        df: ``N`` rows; first column holds integer ids, the remaining ``D``
            columns the numeric features.
        n_components: Number of PCs ``d`` to keep. When None, ``d`` is the knee
            of the explained-variance curve (Kneedle); if no knee is found, the
            smallest ``d`` reaching 80 % cumulative explained variance.
        kneedle_sensitivity: Kneedle sensitivity ``S``.

    Returns:
        Tuple ``(labels, values_tensor, variance_list)``:
        - labels: ``N`` int64 tensor of the first-column ids, not re-encoded.
        - values_tensor: ``N x d`` float64 tensor of PC scores.
        - variance_list: explained-variance ratio of every PC (all
          ``min(N, D)``, not only the kept ``d``), in decreasing order.

    Raises:
        ValueError: If ``n_components`` exceeds the number of available PCs.
    """
    raw_labels = torch.tensor(df.iloc[:, 0].to_numpy(copy=True), dtype=torch.int64)
    X_raw = df.iloc[:, 1:].values
    X_scaled = StandardScaler().fit_transform(X_raw)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    variance_list = sorted(pca_full.explained_variance_ratio_.tolist(), reverse=True)

    if n_components is None:
        kneedle = kneed.KneeLocator(
            range(1, len(variance_list) + 1),
            variance_list,
            S=kneedle_sensitivity,
            curve="convex",
            direction="decreasing",
        )
        n_components = kneedle.knee
        if n_components is None:
            cumulative = np.cumsum(variance_list)
            n_components = int(np.argmax(cumulative >= 0.8)) + 1

    max_components = pca_full.n_components_
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds available PCA components "
            f"({max_components})"
        )
    print(f"Running PCA (d={n_components})...")
    embedding = pca_full.transform(X_scaled)[:, :n_components]
    values_tensor = torch.from_numpy(embedding).to(torch.float64)

    return raw_labels, values_tensor, variance_list
