import streamlit as st

def render_footer():

    st.markdown(
        """
        <div class="footer">

        Customer Segmentation Platform

        <br>

        Streamlit • Scikit-Learn • K-Means • RFM Analysis

        </div>
        """,
        unsafe_allow_html=True
    )