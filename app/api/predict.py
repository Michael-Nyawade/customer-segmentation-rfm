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
    "VIP": {
        "profile": (
            "Buys often, spends heavily, and purchased recently - the "
            "profile of your most valuable relationships."
        ),
        "actions": [
            "Assign a dedicated contact for priority service.",
            "Offer early access to new stock and exclusive pricing.",
            "Avoid generic mass-market campaigns - treat outreach as white-glove.",
        ],
        "priority": (
            "Protect this relationship. Losing a VIP account has an "
            "outsized revenue impact given how few accounts reach this tier."
        ),
    },
    "Loyal High-Spender": {
        "profile": (
            "Consistent, high-spending, and still active - a step below "
            "VIP but trending the right direction."
        ),
        "actions": [
            "Introduce a loyalty or membership tier with tangible perks.",
            "Upsell premium or bundled products based on past purchases.",
            "Watch recency closely - early slowdown is the best time to re-engage.",
        ],
        "priority": "Nurture towards VIP status with consistent, personalized attention.",
    },
    "Mid-Value": {
        "profile": (
            "Moderate spend and infrequent purchases - your largest "
            "group, with the most room to grow."
        ),
        "actions": [
            "Send targeted promotions timed to their typical purchase cycle.",
            "Use limited-time offers to increase purchase frequency.",
            "Test bundling or cross-sell campaigns to lift average order value.",
        ],
        "priority": (
            "Highest growth potential by volume - small frequency gains "
            "across this group compound quickly."
        ),
    },
    "At Risk": {
        "profile": (
            "Long period of inactivity and low historical spend - "
            "showing clear signs of disengagement."
        ),
        "actions": [
            "Launch a win-back campaign with a meaningful incentive.",
            "Send a direct \"we miss you\" message rather than a generic newsletter.",
            "Survey lapsed accounts to learn why they stopped purchasing.",
        ],
        "priority": (
            "Act before further drop-off - re-engagement gets harder "
            "the longer recency grows."
        ),
    },
}

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


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