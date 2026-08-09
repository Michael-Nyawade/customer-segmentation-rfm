"""Tests for src.features.feature_engineering."""

import pandas as pd

from src.features.feature_engineering import assign_segments, compute_rfm, score_rfm


def _make_clean_df():
    # Three customers with clearly distinct RFM profiles.
    return pd.DataFrame({
        "CustomerID": ["A", "A", "B", "C"],
        "InvoiceNo": ["1001", "1002", "1003", "1004"],
        "InvoiceDate": pd.to_datetime([
            "2011-12-01",
            "2011-12-05",
            "2011-11-01",
            "2011-12-09",
        ]),
        "TotalPrice": [100.0, 50.0, 10.0, 500.0],
    })


def test_compute_rfm_shape_and_columns():
    df = _make_clean_df()
    rfm = compute_rfm(df, reference_date_offset_days=1)
    assert set(rfm.columns) == {"Recency", "Frequency", "Monetary"}
    assert len(rfm) == 3  # 3 unique customers


def test_compute_rfm_frequency_counts_unique_invoices():
    df = _make_clean_df()
    rfm = compute_rfm(df, reference_date_offset_days=1)
    assert rfm.loc["A", "Frequency"] == 2
    assert rfm.loc["B", "Frequency"] == 1
    assert rfm.loc["C", "Frequency"] == 1


def test_compute_rfm_monetary_sums_total_price():
    df = _make_clean_df()
    rfm = compute_rfm(df, reference_date_offset_days=1)
    assert rfm.loc["A", "Monetary"] == 150.0
    assert rfm.loc["C", "Monetary"] == 500.0


def test_score_rfm_produces_scores_in_valid_range():
    df = _make_clean_df()
    rfm = compute_rfm(df, reference_date_offset_days=1)
    scored = score_rfm(rfm, n_bins=3)
    for col in ["R_Score", "F_Score", "M_Score"]:
        assert scored[col].between(1, 3).all()


def test_score_rfm_recency_lower_is_better():
    df = _make_clean_df()
    rfm = compute_rfm(df, reference_date_offset_days=1)
    scored = score_rfm(rfm, n_bins=3)
    # Customer C purchased most recently (2011-12-09) -> should have the
    # highest R_Score among the three.
    most_recent_customer = rfm["Recency"].idxmin()
    assert scored.loc[most_recent_customer, "R_Score"] == scored["R_Score"].max()


def test_assign_segments_adds_segment_column():
    df = _make_clean_df()
    rfm = compute_rfm(df, reference_date_offset_days=1)
    scored = score_rfm(rfm, n_bins=3)
    segmented = assign_segments(scored)
    assert "Segment" in segmented.columns
    assert segmented["Segment"].notnull().all()


def test_assign_segments_handles_skewed_data_without_qcut_error():
    # Regression test: many tied/near-duplicate Monetary values, which would
    # break raw-value pd.qcut but should work fine with rank-transformed qcut.
    df = pd.DataFrame({
        "CustomerID": [f"cust_{i}" for i in range(20)],
        "InvoiceNo": [f"inv_{i}" for i in range(20)],
        "InvoiceDate": pd.to_datetime(["2011-12-01"] * 20),
        "TotalPrice": [10.0] * 18 + [500.0, 1000.0],  # heavy ties
    })
    rfm = compute_rfm(df, reference_date_offset_days=1)
    scored = score_rfm(rfm, n_bins=5)  # should not raise
    segmented = assign_segments(scored)
    assert len(segmented) == 20