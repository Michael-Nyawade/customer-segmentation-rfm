"""Prediction utilities: score new customer data using saved model artifacts."""

import pandas as pd

from src.data.preprocess import clean_transactions
from src.features.feature_engineering import compute_rfm
from src.models.train import load_model_artifacts


def predict_from_rfm(rfm: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """Assign clusters and descriptive labels to already-computed RFM data.

    Uses the cluster label mapping saved at training time, so labels stay
    consistent regardless of how large or representative the new batch is.

    Parameters
    ----------
    rfm : pd.DataFrame
        Must contain Recency, Frequency, Monetary columns. Can be new,
        unseen customer data -- does not need to match the training set.
    models_dir : str
        Directory containing the saved model artifacts.

    Returns
    -------
    pd.DataFrame
        rfm with Cluster and Cluster_Label columns added.
    """
    kmeans, scaler, cluster_labels = load_model_artifacts(models_dir)

    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaled = scaler.transform(features)  # transform only -- never re-fit on new data

    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.predict(scaled)
    rfm["Cluster_Label"] = rfm["Cluster"].map(cluster_labels)

    return rfm


def predict_from_transactions(
    transactions: pd.DataFrame, models_dir: str, reference_date_offset_days: int = 1
) -> pd.DataFrame:
    """Convenience wrapper: clean raw transactions, compute RFM, then predict.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction rows for new customers (same schema as the
        original dataset).
    models_dir : str
        Directory containing the saved model artifacts.
    reference_date_offset_days : int
        Passed through to compute_rfm.

    Returns
    -------
    pd.DataFrame
        RFM data with Cluster and Cluster_Label columns added.
    """
    clean = clean_transactions(transactions)
    rfm = compute_rfm(clean, reference_date_offset_days)
    return predict_from_rfm(rfm, models_dir)