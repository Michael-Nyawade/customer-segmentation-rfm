"""Cleaning and preprocessing for the raw retail transactions data."""

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction data ready for RFM feature engineering.

    Steps:
    - Drop rows with missing CustomerID
    - Cast CustomerID to string
    - Parse InvoiceDate to datetime
    - Remove cancelled orders (InvoiceNo starting with 'C')
    - Remove non-positive Quantity or UnitPrice
    - Derive TotalPrice = Quantity * UnitPrice

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions DataFrame, as returned by load_raw_data.

    Returns
    -------
    pd.DataFrame
        Cleaned transactions, ready for RFM aggregation.
    """
    df = df.copy()

    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df