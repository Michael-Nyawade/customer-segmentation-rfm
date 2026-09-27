"""Vercel serverless function: predict a customer's segment."""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.predict import predict_from_rfm, predict_from_transactions

MODELS_DIR = str(PROJECT_ROOT / "models")

CLUSTER_RECOMMENDATIONS = {
    "VIP": (
        "Priority treatment: exclusive offers, loyalty rewards, early access "
        "to sales, and personalized service."
    ),
    "Loyal High-Spender": (
        "Strengthen loyalty: membership benefits, premium product upsells, "
        "and subscription-style offers."
    ),
    "Mid-Value": (
        "Encourage more frequent purchases: targeted promotions and "
        "limited-time discounts."
    ),
    "At Risk": (
        "Re-engagement: win-back emails, \"we miss you\" offers, and "
        "surveys to understand drop-off."
    ),
}

app = FastAPI()


class RFMInput(BaseModel):
    recency: float
    frequency: float
    monetary: float


class Transaction(BaseModel):
    InvoiceNo: str
    CustomerID: str
    InvoiceDate: str
    Quantity: float
    UnitPrice: float


class TransactionsInput(BaseModel):
    transactions: list[Transaction]


@app.post("/api/predict/rfm")
def predict_rfm(payload: RFMInput):
    rfm_input = pd.DataFrame(
        {
            "Recency": [payload.recency],
            "Frequency": [payload.frequency],
            "Monetary": [payload.monetary],
        },
        index=["customer"],
    )
    try:
        result = predict_from_rfm(rfm_input, MODELS_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {e}")

    label = result.loc["customer", "Cluster_Label"]
    return {
        "cluster_label": label,
        "recommendation": CLUSTER_RECOMMENDATIONS.get(label),
    }


@app.post("/api/predict/transactions")
def predict_transactions(payload: TransactionsInput):
    transactions_df = pd.DataFrame([t.model_dump() for t in payload.transactions])

    try:
        result = predict_from_transactions(transactions_df, MODELS_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {e}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected column: {e}")

    customers = [
        {
            "customer_id": str(idx),
            "recency": row["Recency"],
            "frequency": row["Frequency"],
            "monetary": row["Monetary"],
            "cluster_label": row["Cluster_Label"],
            "recommendation": CLUSTER_RECOMMENDATIONS.get(row["Cluster_Label"]),
        }
        for idx, row in result.iterrows()
    ]
    return {"customers": customers}