"""
pages/1_📊_Dashboard.py
=========================
The main analytics dashboard: India-wide overview, state/district
statistics, crop production trends, cultivated area, rainfall/temperature
summary, water usage, and headline AI insights -- all driven by filters
in the sidebar.
"""

import sys
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS, YEARS
from utils.styling import inject_custom_css, render_hero, render_section_header, render_kpi_card
from utils.helpers import get_agri_data, format_inr, format_number, yoy_kpi, apply_plotly_theme

st.set_page_config(page_title="Dashboard | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("📊 Agriculture Dashboard", "India's agriculture at a glance — states, districts, crops, and trends")

df = get_agri_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Dashboard Filters")
year_sel = st.sidebar.selectbox("Year", ["All"] + sorted(df["year"].unique().tolist(), reverse=True))
state_sel = st.sidebar.selectbox("State", ["All"] + INDIAN_STATES)
crop_sel = st.sidebar.selectbox("Crop", ["All"] + CROPS)

filtered = df.copy()
if year_sel != "All":
    filtered = filtered[filtered["year"] == year_sel]
if state_sel != "All":
    filtered = filtered[filtered["state"] == state_sel]
if crop_sel != "All":
    filtered = filtered[filtered["crop"] == crop_sel]

if filtered.empty:
    st.warning("No data matches the selected filters. Try broadening your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
render_section_header("Key Performance Indicators")

total_area, prev_area, area_delta = yoy_kpi(df if state_sel == "All" and crop_sel == "All" else filtered, "area_ha")
total_prod, prev_prod, prod_delta = yoy_kpi(df if state_sel == "All" and crop_sel == "All" else filtered, "production_tonnes")
total_profit, prev_profit, profit_delta = yoy_kpi(df if state_sel == "All" and crop_sel == "All" else filtered, "profit_inr")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Cultivated Area", format_number(filtered["area_ha"].sum(), " ha"),
                     f"{area_delta:+.1f}% YoY", area_delta >= 0)
with k2:
    render_kpi_card("Total Production", format_number(filtered["production_tonnes"].sum(), " t"),
                     f"{prod_delta:+.1f}% YoY", prod_delta >= 0)
with k3:
    render_kpi_card("Avg Yield", f"{filtered['yield_t_per_ha'].mean():.2f} t/ha")
with k4:
    render_kpi_card("Total Profit", format_inr(filtered["profit_inr"].sum()),
                     f"{profit_delta:+.1f}% YoY", profit_delta >= 0)
with k5:
    render_kpi_card("Avg Rainfall", f"{filtered['rainfall_mm'].mean():.0f} mm")

st.write("")
k6, k7, k8, k9 = st.columns(4)
with k6:
    render_kpi_card("Avg Temperature", f"{filtered['avg_temp_c'].mean():.1f} °C")
with k7:
    render_kpi_card("Avg Water Use", f"{filtered['water_use_mm'].mean():.0f} mm")
with k8:
    render_kpi_card("Sustainability Score", f"{filtered['sustainability_score'].mean():.0f} / 100")
with k9:
    render_kpi_card("Districts Covered", f"{filtered['district'].nunique()}")

# ---------------------------------------------------------------------------
# Crop production trend (national/filtered) over years
# ---------------------------------------------------------------------------
render_section_header("Crop Production Trend")

trend = df.copy()
if state_sel != "All":
    trend = trend[trend["state"] == state_sel]
if crop_sel != "All":
    trend = trend[trend["crop"] == crop_sel]
yearly = trend.groupby("year").agg(
    production_tonnes=("production_tonnes", "sum"),
    area_ha=("area_ha", "sum"),
    avg_yield=("yield_t_per_ha", "mean"),
).reset_index()

c1, c2 = st.columns([2, 1])
with c1:
    fig = px.line(yearly, x="year", y="production_tonnes", markers=True,
                   labels={"production_tonnes": "Production (tonnes)", "year": "Year"})
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(apply_plotly_theme(fig, title="Total Production Over Time"), width='stretch')
with c2:
    fig2 = px.line(yearly, x="year", y="avg_yield", markers=True,
                    labels={"avg_yield": "Yield (t/ha)", "year": "Year"})
    fig2.update_traces(line=dict(width=3, color="#F9A825"))
    st.plotly_chart(apply_plotly_theme(fig2, title="Average Yield Over Time"), width='stretch')

# ---------------------------------------------------------------------------
# State-wise statistics
# ---------------------------------------------------------------------------
render_section_header("State-wise Statistics")

state_agg = filtered.groupby("state").agg(
    production_tonnes=("production_tonnes", "sum"),
    area_ha=("area_ha", "sum"),
    avg_yield=("yield_t_per_ha", "mean"),
    profit_inr=("profit_inr", "sum"),
).reset_index().sort_values("production_tonnes", ascending=False)

c3, c4 = st.columns(2)
with c3:
    fig3 = px.bar(state_agg.head(10), x="state", y="production_tonnes",
                   labels={"production_tonnes": "Production (tonnes)", "state": ""})
    st.plotly_chart(apply_plotly_theme(fig3, title="Top 10 States by Production"), width='stretch')
with c4:
    fig4 = px.bar(state_agg.sort_values("avg_yield", ascending=False).head(10),
                   x="state", y="avg_yield", labels={"avg_yield": "Avg Yield (t/ha)", "state": ""})
    fig4.update_traces(marker_color="#1565C0")
    st.plotly_chart(apply_plotly_theme(fig4, title="Top 10 States by Average Yield"), width='stretch')

# ---------------------------------------------------------------------------
# District-wise analytics (only meaningful once a state is chosen, but works generally)
# ---------------------------------------------------------------------------
render_section_header("District-wise Analytics")

district_agg = filtered.groupby(["state", "district"]).agg(
    production_tonnes=("production_tonnes", "sum"),
    avg_yield=("yield_t_per_ha", "mean"),
).reset_index().sort_values("production_tonnes", ascending=False).head(15)

fig5 = px.bar(district_agg, x="district", y="production_tonnes", color="state",
               labels={"production_tonnes": "Production (tonnes)", "district": ""})
st.plotly_chart(apply_plotly_theme(fig5, title="Top 15 Districts by Production"), width='stretch')

# ---------------------------------------------------------------------------
# Crop-mix and weather summary
# ---------------------------------------------------------------------------
render_section_header("Crop Mix & Environmental Summary")

c5, c6 = st.columns(2)
with c5:
    crop_mix = filtered.groupby("crop")["area_ha"].sum().reset_index().sort_values("area_ha", ascending=False)
    fig6 = px.pie(crop_mix, names="crop", values="area_ha", hole=0.45)
    st.plotly_chart(apply_plotly_theme(fig6, title="Cultivated Area Share by Crop"), width='stretch')
with c6:
    weather_summary = filtered.groupby("state").agg(
        rainfall_mm=("rainfall_mm", "mean"), avg_temp_c=("avg_temp_c", "mean")
    ).reset_index()
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(x=weather_summary["state"], y=weather_summary["rainfall_mm"], name="Rainfall (mm)"))
    fig7.add_trace(go.Scatter(x=weather_summary["state"], y=weather_summary["avg_temp_c"] * 20,
                                name="Temp (°C, scaled x20)", mode="lines+markers", yaxis="y"))
    st.plotly_chart(apply_plotly_theme(fig7, title="Rainfall & Temperature by State"), width='stretch')

# ---------------------------------------------------------------------------
# AI Insights (rule-based summary derived from the filtered data itself)
# ---------------------------------------------------------------------------
render_section_header("🤖 AI Insights")

best_state = state_agg.iloc[0]["state"] if len(state_agg) else "N/A"
best_yield_state = state_agg.sort_values("avg_yield", ascending=False).iloc[0]["state"] if len(state_agg) else "N/A"
risk_flag = filtered["drought_risk_score"].mean()

insight_col1, insight_col2 = st.columns(2)
with insight_col1:
    st.info(
        f"📈 **{best_state}** leads in total production for the current filter selection, "
        f"contributing {format_number(state_agg.iloc[0]['production_tonnes'])} tonnes."
    )
    st.info(
        f"🌾 **{best_yield_state}** shows the highest average yield efficiency "
        f"({state_agg.sort_values('avg_yield', ascending=False).iloc[0]['avg_yield']:.2f} t/ha)."
    )
with insight_col2:
    if risk_flag > 55:
        st.warning(f"⚠️ Average drought-risk score is elevated ({risk_flag:.0f}/100) for this selection — consider water-saving irrigation.")
    else:
        st.success(f"✅ Average drought-risk score is within a manageable range ({risk_flag:.0f}/100) for this selection.")
    profit_margin = (filtered["profit_inr"].sum() / filtered["revenue_inr"].sum() * 100) if filtered["revenue_inr"].sum() else 0
    st.success(f"💰 Estimated profit margin across this selection is **{profit_margin:.1f}%** of revenue.")
