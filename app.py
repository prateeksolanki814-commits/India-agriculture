"""
app.py
========
Main entry point for AGRI VISION AI. Run with:
    streamlit run app.py

This file renders the Home page and defines the sidebar navigation.
Each other page lives in pages/N_Name.py and is auto-discovered by
Streamlit's multipage app mechanism (native st.Page / st.navigation).
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import APP_NAME, APP_ICON, APP_TAGLINE, INDIAN_STATES, CROPS
from utils.styling import inject_custom_css, render_hero, render_section_header, render_kpi_card
from utils.helpers import get_agri_data, format_inr, format_number

st.set_page_config(
    page_title=f"{APP_NAME} | Smart Agriculture Analytics",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_NAME}")
    st.caption(APP_TAGLINE)
    st.divider()
    st.markdown("""
    **Navigate using the pages above:**
    - 📊 Dashboard
    - 🤖 AI Predictions
    - 📈 Analytics
    - 🗺️ Maps & GIS
    - 💡 Recommendations
    - 🌦️ Weather Intelligence
    - 🚨 Smart Alerts
    - 📄 Reports
    - 💬 AI Assistant
    """)
    st.divider()
    st.caption("Built with Streamlit • scikit-learn • XGBoost • LightGBM • Prophet")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
render_hero()

# ---------------------------------------------------------------------------
# Intro + quick KPIs pulled straight from the live dataset
# ---------------------------------------------------------------------------
df = get_agri_data()
latest_year = df["year"].max()
latest_df = df[df["year"] == latest_year]

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("States Covered", f"{df['state'].nunique()}")
with col2:
    render_kpi_card("Crops Tracked", f"{df['crop'].nunique()}")
with col3:
    render_kpi_card("Total Cultivated Area", format_number(latest_df["area_ha"].sum(), " ha"))
with col4:
    render_kpi_card("Total Production", format_number(latest_df["production_tonnes"].sum(), " t"))

st.write("")
render_section_header("Welcome to AGRI VISION AI")

st.markdown("""
**AGRI VISION AI** is an end-to-end smart-agriculture analytics and prediction
platform built for India's farmers, agribusinesses, researchers, and policy
makers. It combines a decade of state- and district-level agricultural data
with a suite of machine-learning models to turn raw numbers into decisions
you can act on.

Use the sidebar to explore:
""")

feat_col1, feat_col2, feat_col3 = st.columns(3)
with feat_col1:
    st.markdown("""
    #### 📊 Dashboard & Analytics
    State-wise and district-wise statistics, crop production trends,
    rainfall/temperature summaries, and India-wide KPIs at a glance.
    """)
    st.markdown("""
    #### 🗺️ Maps & GIS
    Interactive choropleth and marker maps for crop distribution,
    rainfall, soil quality, and flood/drought risk zones.
    """)
    st.markdown("""
    #### 🚨 Smart Alerts
    Automatic flags for heavy rain, flood, drought, heatwave, pest and
    disease risk, and price swings — generated from live model outputs.
    """)
with feat_col2:
    st.markdown("""
    #### 🤖 AI Predictions
    Sixteen ML-powered predictors: crop yield, crop & fertilizer
    recommendation, market price, sustainability, and four risk models —
    each with a transparent confidence score.
    """)
    st.markdown("""
    #### 💡 Smart Recommendations
    AI-generated guidance on the best crop, sowing/harvest timing,
    fertilizer plan, irrigation, and expected profit for your inputs.
    """)
    st.markdown("""
    #### 📄 Reports
    Export state, district, or crop-level reports as PDF, Excel, or CSV,
    complete with an AI-written summary.
    """)
with feat_col3:
    st.markdown("""
    #### 🌦️ Weather Intelligence
    Prophet-powered rainfall and temperature forecasting per state, with
    trend direction and extreme-weather flags.
    """)
    st.markdown("""
    #### 💬 AI Assistant
    A chatbot that answers agriculture questions using this platform's
    own data — ask about any state, crop, or trend.
    """)
    st.markdown("""
    #### 🌾 15 States, 15 Crops
    Built on Punjab, Maharashtra, Uttar Pradesh, Tamil Nadu, and 11 more
    states covering rice, wheat, cotton, sugarcane, pulses, and more.
    """)

st.divider()
st.caption(
    "Note: this platform runs on a realistic synthetic dataset modeled after "
    "typical Indian agricultural patterns (rainfall bands, crop-soil affinities, "
    "regional yields) rather than a live government feed, so all figures are "
    "illustrative and intended to demonstrate the analytics and ML pipeline."
)
