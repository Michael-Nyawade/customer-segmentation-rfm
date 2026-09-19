import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

sys.path.append(str(PROJECT_ROOT))

from src.models.predict import (
    predict_from_rfm,
    predict_from_transactions
)

MODELS_DIR = str(
    PROJECT_ROOT / "models"
)

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


def render_predictor():

    st.subheader(
        "Customer Segment Predictor"
    )

    st.write(
        "Score a new or existing customer using the trained model - either "
        "by entering their RFM values directly, or by uploading their raw "
        "transaction history and letting the app compute RFM for you."
    )

    mode = st.radio(
        "Choose input method",
        [
            "Enter RFM Values",
            "Upload Transactions"
        ],
        horizontal=True
    )

    if mode == "Enter RFM Values":

        col1, col2, col3 = st.columns(3)

        with col1:
            recency = st.number_input(
                "Recency",
                min_value=0,
                value=30,
            )

        with col2:
            frequency = st.number_input(
                "Frequency",
                min_value=1,
                value=5,
            )

        with col3:
            monetary = st.number_input(
                "Monetary",
                min_value=0.0,
                value=500.0,
            )

        if st.button("Predict Segment"):

            rfm_input = pd.DataFrame(
                {
                    "Recency": [recency],
                    "Frequency": [frequency],
                    "Monetary": [monetary],
                },
                index=["customer"]
            )

            try:
                result = predict_from_rfm(
                    rfm_input,
                    MODELS_DIR
                )

                label = result.loc[
                    "customer",
                    "Cluster_Label"
                ]

                st.success(
                    f"Predicted Segment: {label}"
                )

                recommendation = (
                    CLUSTER_RECOMMENDATIONS.get(label)
                )

                if recommendation:
                    st.info(
                        f"**Suggested action:** {recommendation}"
                    )

                st.dataframe(
                    result,
                    use_container_width=True
                )

            except FileNotFoundError as e:
                st.error(
                    f"Model not found: {e}"
                )

    else:

        st.write(
            "Upload a CSV with columns: `InvoiceNo`, `CustomerID`, "
            "`InvoiceDate`, `Quantity`, `UnitPrice` (same schema as the "
            "original dataset). One row per transaction line - a customer "
            "with several past orders should have multiple rows."
        )

        example_df = pd.DataFrame({
            "InvoiceNo": ["536365", "536365", "536390"],
            "CustomerID": [17850, 17850, 17850],
            "InvoiceDate": [
                "12/01/2010 08:26",
                "12/01/2010 08:26",
                "12/03/2010 09:15"
            ],
            "Quantity": [6, 8, 3],
            "UnitPrice": [2.55, 2.75, 5.00],
        })

        st.caption("Example format:")

        st.dataframe(
            example_df,
            hide_index=True
        )

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type="csv"
        )

        if uploaded_file:

            transactions = pd.read_csv(
                uploaded_file,
                encoding="ISO-8859-1"
            )

            st.dataframe(
                transactions.head(),
                use_container_width=True
            )

            if st.button(
                "Generate Predictions"
            ):

                try:
                    result = (
                        predict_from_transactions(
                            transactions,
                            MODELS_DIR
                        )
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.write(
                        "### Suggested Actions by Segment"
                    )

                    for cluster_label in (
                        result["Cluster_Label"].unique()
                    ):
                        recommendation = (
                            CLUSTER_RECOMMENDATIONS.get(
                                cluster_label
                            )
                        )

                        if recommendation:
                            count = (
                                result["Cluster_Label"]
                                == cluster_label
                            ).sum()

                            st.info(
                                f"**{cluster_label}** "
                                f"({count} customer(s)): "
                                f"{recommendation}"
                            )

                except FileNotFoundError as e:
                    st.error(
                        f"Model not found: {e}"
                    )

                except KeyError as e:
                    st.error(
                        f"Missing expected column in uploaded file: {e}"
                    )