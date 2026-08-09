"""Model training utilities: scaling, k-selection diagnostics, and KMeans fitting."""

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


# Cluster profiling and labeling
CLUSTER_LABELS_BY_RANK = ["VIP", "Loyal High-Spender", "Mid-Value", "At Risk"]


def profile_clusters(rfm: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Compute mean R/F/M and size per cluster.

    Parameters
    ----------
    rfm : pd.DataFrame
        Must contain Recency, Frequency, Monetary, and cluster_col.
    cluster_col : str
        Column containing cluster assignments.

    Returns
    -------
    pd.DataFrame
        Indexed by cluster id, with Recency, Frequency, Monetary means and Count.
    """
    summary = rfm.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean().round(1)
    summary["Count"] = rfm[cluster_col].value_counts()
    return summary


def label_clusters_by_value(cluster_summary: pd.DataFrame) -> dict:
    """Derive a cluster id -> descriptive label mapping, ranked by mean Monetary.

    Clusters are ranked by mean Monetary (descending) and assigned labels
    from CLUSTER_LABELS_BY_RANK in order. This avoids hardcoding a cluster
    index -> label mapping, since KMeans cluster indices are arbitrary and
    can differ between runs even with a fixed random_state.

    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Output of profile_clusters. Must have exactly len(CLUSTER_LABELS_BY_RANK)
        rows for the label set to apply cleanly.

    Returns
    -------
    dict
        Mapping of cluster id -> descriptive label.
    """
    if len(cluster_summary) != len(CLUSTER_LABELS_BY_RANK):
        raise ValueError(
            f"Expected {len(CLUSTER_LABELS_BY_RANK)} clusters to apply named "
            f"labels, got {len(cluster_summary)}. Update CLUSTER_LABELS_BY_RANK "
            "or the n_clusters config value to match."
        )

    ranked_cluster_ids = cluster_summary.sort_values("Monetary", ascending=False).index
    return dict(zip(ranked_cluster_ids, CLUSTER_LABELS_BY_RANK))
