import streamlit as st

def metric_card(
    label,
    value,
    note=""
):

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-label">
        {label}
        </div>

        <div class="metric-value">
        {value}
        </div>

        <div>
        {note}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )