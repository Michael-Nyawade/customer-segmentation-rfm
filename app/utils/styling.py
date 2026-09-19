import streamlit as st


def load_css():

    st.markdown("""
    <style>

    .block-container{
        max-width:1400px;
        padding-top:11.2rem;
        padding-bottom:2rem;
    }

    .hero{
        position:fixed;
        top:3.7rem;
        left:0;
        right:0;
        z-index:999;

        padding:1rem 1.5rem;
        margin:0;
        border-radius:0 0 16px 16px;

        display:flex;
        flex-direction:column;
        align-items:center;
        text-align:center;

        background:linear-gradient(135deg, #1E3A8A, #2563EB);
        color:white;
        box-shadow:0 2px 8px rgba(0,0,0,.2);
    }

    .hero h1{
        font-size:1.5rem;
        margin-bottom:.25rem;
        color:white;
    }

    .hero p{
        max-width:850px;
        opacity:.95;
        line-height:1.4;
        font-size:.85rem;
        margin:0;
        color:white;
    }

    .metric-card{
        background:var(--secondary-background-color);
        border-radius:18px;
        padding:1rem;
        border:1px solid rgba(120,120,120,.15);
        margin-bottom:1rem;
    }

    .metric-label{
        font-size:.8rem;
        opacity:.75;
    }

    .metric-value{
        font-size:1.8rem;
        font-weight:700;
    }

    .footer{
        text-align:center;
        opacity:.7;
        margin-top:3rem;
        padding-top:1.5rem;
        border-top:1px solid rgba(120,120,120,.15);
    }

    </style>
    """, unsafe_allow_html=True)