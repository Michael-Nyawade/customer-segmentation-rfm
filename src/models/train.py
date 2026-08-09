"""Model training utilities: scaling, k-selection diagnostics, and KMeans fitting."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def scale_rfm_features(rfm: pd.DataFrame) -> tuple:
    """Scale Recency, Frequency, Monetary for clustering.

    Parameters
    ----------
    rfm : pd.DataFrame
        Must contain Recency, Frequency, Monetary columns.

    Returns
    -------
    tuple
        (scaled_features: np.ndarray, fitted_scaler: StandardScaler)
    """
    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def compute_elbow_inertia(scaled_features, k_range=range(1, 11), random_state=42) -> dict:
    """Compute KMeans inertia for a range of k values (Elbow method).

    Returns
    -------
    dict
        Mapping of k -> inertia.
    """
    inertia = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(scaled_features)
        inertia[k] = kmeans.inertia_
    return inertia


def compute_silhouette_scores(scaled_features, k_range=range(2, 11), random_state=42) -> dict:
    """Compute silhouette scores for a range of k values.

    Returns
    -------
    dict
        Mapping of k -> silhouette score.
    """
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(scaled_features)
        scores[k] = silhouette_score(scaled_features, labels)
    return scores