"""RFM feature engineering: aggregation, scoring, and rule-based segmentation."""

import pandas as pd


def compute_rfm(df: pd.DataFrame, reference_date_offset_days: int = 1) -> pd.DataFrame:
    """Aggregate cleaned transactions into per-customer RFM metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions (output of clean_transactions), must contain
        CustomerID, InvoiceDate, InvoiceNo, TotalPrice.
    reference_date_offset_days : int
        Days after the latest transaction to use as the Recency reference point.

    Returns
    -------
    pd.DataFrame
        Indexed by CustomerID, with Recency, Frequency, Monetary columns.
    """
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=reference_date_offset_days)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum",
    })

    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary",
    }, inplace=True)

    return rfm


def score_rfm(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Add quintile-based R/F/M scores and a combined RFM_Score string.

    Each metric is rank-transformed before binning (method='first', which
    breaks ties by row order) so that pd.qcut always finds n_bins distinct
    bin edges, regardless of how skewed or tied the underlying values are.

    Recency is scored so that lower Recency (more recent) yields a higher
    score. Frequency and Monetary are scored so that higher values yield
    a higher score.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of compute_rfm.
    n_bins : int
        Number of quantile bins to use (5 => scores 1-5).

    Returns
    -------
    pd.DataFrame
        rfm with R_Score, F_Score, M_Score, and RFM_Score columns added.
    """
    rfm = rfm.copy()

    descending_labels = list(range(n_bins, 0, -1))  # e.g. [5, 4, 3, 2, 1]
    ascending_labels = list(range(1, n_bins + 1))    # e.g. [1, 2, 3, 4, 5]

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), n_bins, labels=descending_labels
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), n_bins, labels=ascending_labels
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), n_bins, labels=ascending_labels
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )

    return rfm


def _segment_customer(row: pd.Series) -> str:
    """Assign a rule-based segment label from a single RFM-scored row."""
    if row["RFM_Score"] == "555":
        return "Champion"
    elif row["R_Score"] >= 4 and row["F_Score"] >= 4:
        return "Loyal Customer"
    elif row["R_Score"] >= 4 and row["M_Score"] >= 4:
        return "Big Spender"
    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "At Risk"
    elif row["R_Score"] == 1:
        return "Lost"
    else:
        return "Others"


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add a rule-based Segment column based on R/F/M scores.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of score_rfm (must contain R_Score, F_Score, M_Score, RFM_Score).

    Returns
    -------
    pd.DataFrame
        rfm with a Segment column added.
    """
    rfm = rfm.copy()
    rfm["Segment"] = rfm.apply(_segment_customer, axis=1)
    return rfm