"""
pages/8_📄_Reports.py
========================
Report generation UI: lets users scope a report (by state, district, or
crop), preview an AI-written summary, and download it as PDF, Excel, or
CSV.
"""

import sys
import os
import datetime
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS
from utils.styling import inject_custom_css, render_hero, render_section_header
from utils.helpers import get_agri_data
from utils.report_generator import (
    generate_ai_summary, dataframe_to_csv_bytes,
    dataframe_to_excel_bytes, dataframe_to_pdf_bytes,
)

st.set_page_config(page_title="Reports | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("📄 Reports", "Generate downloadable state, district, or crop reports with an AI-written summary")

df = get_agri_data()

render_section_header("Report Scope")
report_type = st.radio("Report Type", ["State Report", "District Report", "Crop Report", "Full Dataset"], horizontal=True)

scope_df = df.copy()
scope_label = "All India (Full Dataset)"

if report_type == "State Report":
    state = st.selectbox("Select State", INDIAN_STATES)
    scope_df = df[df["state"] == state]
    scope_label = f"{state} (State Report)"
elif report_type == "District Report":
    state = st.selectbox("Select State", INDIAN_STATES, key="dist_state")
    districts = sorted(df[df["state"] == state]["district"].unique())
    district = st.selectbox("Select District", districts)
    scope_df = df[(df["state"] == state) & (df["district"] == district)]
    scope_label = f"{district}, {state} (District Report)"
elif report_type == "Crop Report":
    crop = st.selectbox("Select Crop", CROPS)
    scope_df = df[df["crop"] == crop]
    scope_label = f"{crop} (Crop Report)"

year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()),
                        (int(df["year"].min()), int(df["year"].max())))
scope_df = scope_df[(scope_df["year"] >= year_range[0]) & (scope_df["year"] <= year_range[1])]

if scope_df.empty:
    st.warning("No data matches this report scope.")
    st.stop()

render_section_header("Preview")
st.dataframe(scope_df.head(20), width="stretch", hide_index=True)
st.caption(f"Showing first 20 of {len(scope_df):,} matching rows.")

ai_summary = generate_ai_summary(scope_df, scope_label)
render_section_header("🤖 AI Summary")
st.markdown(f'<div class="rec-card"><pre style="white-space: pre-wrap; font-family: inherit;">{ai_summary}</pre></div>',
            unsafe_allow_html=True)

render_section_header("Download Report")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
safe_scope = scope_label.split(" (")[0].replace(", ", "_").replace(" ", "_")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("📄 Prepare PDF Report", width="stretch"):
        with st.spinner("Building PDF..."):
            pdf_bytes = dataframe_to_pdf_bytes(scope_df, ai_summary, scope_label)
        st.download_button(
            "⬇️ Download PDF", data=pdf_bytes,
            file_name=f"agrivision_{safe_scope}_{timestamp}.pdf",
            mime="application/pdf", width="stretch",
        )
with c2:
    if st.button("📊 Prepare Excel Report", width="stretch"):
        with st.spinner("Building Excel workbook..."):
            xlsx_bytes = dataframe_to_excel_bytes(scope_df, ai_summary)
        st.download_button(
            "⬇️ Download Excel", data=xlsx_bytes,
            file_name=f"agrivision_{safe_scope}_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
with c3:
    if st.button("📑 Prepare CSV Report", width="stretch"):
        with st.spinner("Building CSV..."):
            csv_bytes = dataframe_to_csv_bytes(scope_df)
        st.download_button(
            "⬇️ Download CSV", data=csv_bytes,
            file_name=f"agrivision_{safe_scope}_{timestamp}.csv",
            mime="text/csv", width="stretch",
        )

st.caption(
    "The AI Summary is generated deterministically from the exact figures in your selected "
    "scope (not a generative model), so every number it states is directly traceable to the data table above."
)
