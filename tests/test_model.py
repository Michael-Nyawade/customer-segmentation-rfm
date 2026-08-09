"""Tests for src.models.train."""

import numpy as np
import pandas as pd
import tempfile
import pytest

from src.models.train import (
    compute_elbow_inertia,
    compute_silhouette_scores,
    scale_rfm_features,
    fit_kmeans,
    load_model_artifacts,
    save_model_artifacts,
)

def _make_rfm_df():
    # Two well-separated groups so clustering behavior is predictable.
    return pd.DataFrame({
        "Recency": [5, 6, 4, 300, 310, 290],
        "Frequency": [20, 22, 19, 1, 2, 1],
        "Monetary": [5000, 5200, 4800, 100, 150, 120],
    }, index=[f"cust_{i}" for i in range(6)])

def test_scale_rfm_features_returns_zero_mean_unit_variance():
    rfm = _make_rfm_df()
    scaled, scaler = scale_rfm_features(rfm)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-8)

def test_scale_rfm_features_returns_fitted_scaler():
    rfm = _make_rfm_df()
    _, scaler = scale_rfm_features(rfm)
    assert hasattr(scaler, "mean_")
    assert len(scaler.mean_) == 3

def test_compute_elbow_inertia_decreases_with_k():
    rfm = _make_rfm_df()
    scaled, _ = scale_rfm_features(rfm)
    inertia = compute_elbow_inertia(scaled, k_range=range(1, 5), random_state=42)
    values = [inertia[k] for k in sorted(inertia)]
    # Inertia should be non-increasing as k grows.
    assert all(earlier >= later for earlier, later in zip(values, values[1:]))

def test_compute_silhouette_scores_identifies_two_clear_clusters():
    rfm = _make_rfm_df()
    scaled, _ = scale_rfm_features(rfm)
    scores = compute_silhouette_scores(scaled, k_range=range(2, 4), random_state=42)
    # With two obviously separated groups, k=2 should score highest.
    assert scores[2] == max(scores.values())

def test_fit_kmeans_returns_correct_number_of_clusters():
    rfm = _make_rfm_df()
    scaled, _ = scale_rfm_features(rfm)
    kmeans = fit_kmeans(scaled, n_clusters=2, random_state=42)
    assert kmeans.n_clusters == 2
    assert len(set(kmeans.labels_)) == 2

def test_save_and_load_model_artifacts_roundtrip():
    rfm = _make_rfm_df()
    scaled, scaler = scale_rfm_features(rfm)
    kmeans = fit_kmeans(scaled, n_clusters=2, random_state=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_model_artifacts(kmeans, scaler, tmp_dir)
        loaded_kmeans, loaded_scaler = load_model_artifacts(tmp_dir)

        assert (kmeans.predict(scaled) == loaded_kmeans.predict(scaled)).all()
        assert np.allclose(scaler.mean_, loaded_scaler.mean_)

def test_load_model_artifacts_raises_when_missing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FileNotFoundError):
            load_model_artifacts(tmp_dir)


# Cluster profiling and labeling tests

from src.models.train import CLUSTER_LABELS_BY_RANK, label_clusters_by_value, profile_clusters

def _make_labeled_rfm_df(n_clusters=3):
    if n_clusters == 3:
        return pd.DataFrame({
            "Recency": [5, 5, 40, 40, 300, 300],
            "Frequency": [50, 48, 10, 12, 1, 2],
            "Monetary": [100000, 95000, 5000, 4800, 200, 250],
            "Cluster": [0, 0, 1, 1, 2, 2],
        })
    raise ValueError("Only n_clusters=3 fixture defined")

def test_profile_clusters_computes_means_and_counts():
    rfm = _make_labeled_rfm_df()
    summary = profile_clusters(rfm)
    assert list(summary.index) == [0, 1, 2]
    assert summary.loc[0, "Count"] == 2
    assert summary.loc[0, "Monetary"] > summary.loc[1, "Monetary"] > summary.loc[2, "Monetary"]

def test_label_clusters_by_value_raises_on_wrong_cluster_count():
    # Fixture has 3 clusters; CLUSTER_LABELS_BY_RANK has 4 entries -> mismatch.
    rfm = _make_labeled_rfm_df()
    summary = profile_clusters(rfm)
    with pytest.raises(ValueError):
        label_clusters_by_value(summary)

def test_label_clusters_by_value_correct_mapping_with_four_clusters():
    # 4 clusters with distinct Monetary ranks and arbitrary, non-sequential
    # cluster ids -- checks the mapping is rank-based, not index-based.
    rfm = pd.DataFrame({
        "Recency": [5, 15, 40, 300],
        "Frequency": [80, 20, 4, 1],
        "Monetary": [120000, 12000, 1500, 400],
        "Cluster": [10, 20, 30, 40],
    })
    summary = profile_clusters(rfm)
    labels = label_clusters_by_value(summary)
    assert labels[10] == "VIP"
    assert labels[20] == "Loyal High-Spender"
    assert labels[30] == "Mid-Value"
    assert labels[40] == "At Risk"
