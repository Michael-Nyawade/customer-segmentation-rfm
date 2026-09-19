import streamlit as st
from streamlit_option_menu import option_menu
from utils.styling import load_css
from components.hero import render_hero
from components.footer import render_footer
from views.dashboard import render_dashboard
from views.predictor import render_predictor
from views.about import render_about

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

load_css()
render_hero()

selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Predictor", "About"],
    icons=["bar-chart", "search", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "#EFF6FF",
        },
        "icon": {
            "color": "#1E3A8A",
            "font-size": "16px",
        },
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
            "margin": "0px",
            "color": "#1E3A8A",
            "--hover-color": "#DBEAFE",
        },
        "nav-link-selected": {
            "background-color": "#2563EB",
            "color": "white",
        },
    },
)

if selected == "Dashboard":
    render_dashboard()
elif selected == "Predictor":
    render_predictor()
else:
    render_about()

render_footer()