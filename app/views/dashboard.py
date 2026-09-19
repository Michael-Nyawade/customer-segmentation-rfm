import plotly.express as px
import streamlit as st

from utils.data_loader import load_dashboard_data
from components.metrics import metric_card


def render_dashboard():

    st.subheader(
        "Customer Segmentation Dashboard"
    )

    st.write(
        "This view summarizes how customers from the training dataset were "
        "segmented, both by K-Means clustering (data-driven groups) and by "
        "rule-based RFM scoring (interpretable, threshold-based groups). "
        "These are two independent segmentations of the same customers, "
        "shown here for comparison."
    )

    try:

        rfm = load_dashboard_data()

    except FileNotFoundError:

        st.error(
            "Dashboard dataset not found."
        )

        return

    avg_spend = rfm["Monetary"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Customers",
            f"{len(rfm):,}"
        )

    with c2:
        metric_card(
            "Segments",
            rfm["Segment"].nunique()
        )

    with c3:
        metric_card(
            "Clusters",
            rfm["Cluster_Label"].nunique()
        )

    with c4:
        metric_card(
            "Average Spend",
            f"${avg_spend:,.0f}"
        )

    col1, col2 = st.columns(2)

    with col1:

        cluster_counts = (
            rfm["Cluster_Label"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = [
            "Cluster",
            "Customers"
        ]

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Customers",
            color="Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        segment_counts = (
            rfm["Segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "Segment",
            "Customers"
        ]

        fig = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            color="Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Cluster Summary"
    )

    cluster_summary = (
        rfm.groupby("Cluster_Label")
        [
            ["Recency",
             "Frequency",
             "Monetary"]
        ]
        .mean()
        .round(1)
    )

    cluster_summary["Count"] = (
        rfm["Cluster_Label"]
        .value_counts()
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )