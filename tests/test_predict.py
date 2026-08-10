"""Tests for src.models.predict."""

import tempfile

import pandas as pd

from src.models.predict import predict_from_rfm, predict_from_transactions
from src.models.train import fit_kmeans, save_model_artifacts, scale_rfm_features


def _make_training_rfm():
    # Two well-separated groups so KMeans clustering is stable and predictable.
    return pd.DataFrame({
        "Recency": [5, 6, 4, 300, 310, 290],
        "Frequency": [20, 22, 19, 1, 2, 1],
        "Monetary": [5000, 5200, 4800, 100, 150, 120],
    }, index=[f"cust_{i}" for i in range(6)])


def _train_and_save(tmp_dir):
    rfm = _make_training_rfm()
    scaled, scaler = scale_rfm_features(rfm)
    kmeans = fit_kmeans(scaled, n_clusters=2, random_state=42)
    cluster_labels = {0: "Active", 1: "Inactive"}
    save_model_artifacts(kmeans, scaler, cluster_labels, tmp_dir)
    return cluster_labels


def test_predict_from_rfm_adds_cluster_and_label_columns():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _train_and_save(tmp_dir)

        new_customer = pd.DataFrame({
            "Recency": [3],
            "Frequency": [25],
            "Monetary": [5500],
        }, index=["new_cust_1"])

        result = predict_from_rfm(new_customer, tmp_dir)
        assert "Cluster" in result.columns
        assert "Cluster_Label" in result.columns
        assert result.loc["new_cust_1", "Cluster_Label"] in {"Active", "Inactive"}


def test_predict_from_rfm_uses_saved_labels_not_reprofiled_ones():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cluster_labels = _train_and_save(tmp_dir)

        # A single new customer -- too small a batch to re-profile meaningfully,
        # which is exactly why we persist and reuse training-time labels.
        new_customer = pd.DataFrame({
            "Recency": [2],
            "Frequency": [30],
            "Monetary": [6000],
        }, index=["new_cust_2"])

        result = predict_from_rfm(new_customer, tmp_dir)
        assigned_label = result.loc["new_cust_2", "Cluster_Label"]
        assert assigned_label in cluster_labels.values()


def test_predict_from_transactions_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _train_and_save(tmp_dir)

        new_transactions = pd.DataFrame({
            "InvoiceNo": ["9001", "9002"],
            "CustomerID": [999.0, 999.0],
            "InvoiceDate": ["12/01/2011 10:00", "12/05/2011 11:00"],
            "Quantity": [5, 3],
            "UnitPrice": [10.0, 20.0],
        })

        result = predict_from_transactions(new_transactions, tmp_dir)
        assert "Cluster_Label" in result.columns
        assert len(result) == 1  # one unique customer