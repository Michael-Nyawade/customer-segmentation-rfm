import streamlit as st


def render_about():

    st.subheader(
        "About This Project"
    )

    st.markdown("""
    ### Business Problem

    Not all customers are equally valuable, and treating them the same
    wastes marketing budget and misses opportunities. This tool identifies
    which customers are high=value, loyal, at risk of churning, or worth
    investing in for growth - so outreach can be targeted rather than
    generic.

    ### How It Works

    Customer purchase history is summarized into three behavioral
    measures for each customer:

    - **Recency** - how recently they last purchased
    - **Frequency** - how often they purchase
    - **Monetary** - how much they've spent in total

    These measures are used in two complementary ways: a rule-based
    scoring system assigns interpretable labels (e.g. *Champion*,
    *At Risk*), and a K-Means clustering model groups customers by
    overall similarity, labeled by average spend (*VIP*,
    *Loyal High-Spender*, *Mid-Value*, *At Risk*).

    ### Dataset

    [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) -
    ~540,000 transactions from a UK-based online retailer (2010-2011).

    ### Technology

    Built with Python, pandas, scikit-learn, and Streamlit.
    """)