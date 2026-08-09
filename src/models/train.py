""""Model training utilities: scaling, k-selection diagnostics, and KMeans fitting."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import joblib

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


def fit_kmeans(scaled_features, n_clusters: int, random_state: int = 42) -> KMeans:
    """Fit the final KMeans model on scaled RFM features.

    Parameters
    ----------
    scaled_features : np.ndarray
        Output of scale_rfm_features.
    n_clusters : int
        Number of clusters to fit.
    random_state : int
        For reproducibility.

    Returns
    -------
    KMeans
        The fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(scaled_features)
    return kmeans


def save_model_artifacts(kmeans: KMeans, scaler: StandardScaler, models_dir: str) -> None:
    """Persist the fitted KMeans model and scaler to disk.

    Parameters
    ----------
    kmeans : KMeans
        Fitted model from fit_kmeans.
    scaler : StandardScaler
        Fitted scaler from scale_rfm_features.
    models_dir : str
        Directory to save artifacts into (created if it doesn't exist).
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(kmeans, models_path / "kmeans_model.joblib")
    joblib.dump(scaler, models_path / "scaler.joblib")


def load_model_artifacts(models_dir: str) -> tuple:
    """Load the persisted KMeans model and scaler from disk.

    Parameters
    ----------
    models_dir : str
        Directory containing kmeans_model.joblib and scaler.joblib.

    Returns
    -------
    tuple
        (kmeans: KMeans, scaler: StandardScaler)
    """
    models_path = Path(models_dir)
    kmeans_file = models_path / "kmeans_model.joblib"
    scaler_file = models_path / "scaler.joblib"

    if not kmeans_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {models_path}. "
            "Train and save the model first."
        )

    kmeans = joblib.load(kmeans_file)
    scaler = joblib.load(scaler_file)
    return kmeans, scaler
