"""Tests for src.data.preprocess."""

import pandas as pd

from src.data.preprocess import clean_transactions


def _make_raw_df():
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536366", "C536367", "536368", "536369"],
        "StockCode": ["A1", "A2", "A3", "A4", "A5"],
        "Description": ["Item A", "Item B", "Item C", "Item D", "Item E"],
        "Quantity": [6, -2, 3, 0, 4],
        "InvoiceDate": [
            "12/01/2010 08:26",
            "12/01/2010 08:30",
            "12/01/2010 08:35",
            "12/01/2010 08:40",
            "12/01/2010 08:45",
        ],
        "UnitPrice": [2.55, 3.39, 5.00, 4.00, 1.50],
        "CustomerID": [17850.0, 17850.0, 17851.0, None, 17852.0],
        "Country": ["United Kingdom"] * 5,
    })


def test_clean_transactions_drops_missing_customer_id():
    df = _make_raw_df()
    cleaned = clean_transactions(df)
    assert cleaned["CustomerID"].isnull().sum() == 0


def test_clean_transactions_removes_cancelled_orders():
    df = _make_raw_df()
    cleaned = clean_transactions(df)
    assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()


def test_clean_transactions_removes_non_positive_quantity_and_price():
    df = _make_raw_df()
    cleaned = clean_transactions(df)
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["UnitPrice"] > 0).all()


def test_clean_transactions_computes_total_price():
    df = _make_raw_df()
    cleaned = clean_transactions(df)
    expected = cleaned["Quantity"] * cleaned["UnitPrice"]
    pd.testing.assert_series_equal(cleaned["TotalPrice"], expected, check_names=False)


def test_clean_transactions_only_valid_rows_survive():
    # Of the 5 rows, only row0 (536365) and row4 (536369) pass every filter:
    # not cancelled, positive qty/price, has CustomerID.
    df = _make_raw_df()
    cleaned = clean_transactions(df)
    assert set(cleaned["InvoiceNo"]) == {"536365", "536369"}