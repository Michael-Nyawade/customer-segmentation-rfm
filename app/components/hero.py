import streamlit as st

def render_hero():

    st.markdown(
        """
        <div class="hero">

        <h1>Customer Segmentation Intelligence</h1>

        <p>
        Explore customer purchasing behaviour using
        RFM analysis and machine learning clustering.

        Identify valuable customers, uncover hidden
        segments, and generate predictions using a
        trained K-Means model.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )