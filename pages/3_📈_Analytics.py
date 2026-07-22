"""
pages/3_📈_Analytics.py
=========================
Advanced agriculture analytics: production trends, yield comparison,
state/district rankings, seasonal analysis, water/irrigation analysis,
soil analysis, historical trends vs forecasts, and profit analytics.
"""

import sys
import os
import streamlit as st
import plotly.express as px
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS
from utils.styling import inject_custom_css, render_hero, render_section_header, render_kpi_card
from utils.helpers import get_agri_data, format_inr, format_number, apply_plotly_theme

st.set_page_config(page_title="Analytics | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("📈 Agriculture Analytics", "Deep-dive production, yield, seasonal, and profitability analysis")

df = get_agri_data()

st.sidebar.header("🔎 Analytics Filters")
states_sel = st.sidebar.multiselect("States", INDIAN_STATES, default=INDIAN_STATES[:5])
crops_sel = st.sidebar.multiselect("Crops", CROPS, default=CROPS[:5])

filtered = df.copy()
if states_sel:
    filtered = filtered[filtered["state"].isin(states_sel)]
if crops_sel:
    filtered = filtered[filtered["crop"].isin(crops_sel)]

if filtered.empty:
    st.warning("No data for the current filter combination. Select at least one state and crop.")
    st.stop()

tabs = st.tabs([
    "🌾 Production & Yield", "🏆 Rankings", "📅 Seasonal Analysis",
    "💧 Water & Irrigation", "🧫 Soil Analysis", "💰 Profit Analytics", "📊 Growth Index",
])

# ---------------------------------------------------------------------------
# TAB 1: Production & Yield
# ---------------------------------------------------------------------------
with tabs[0]:
    render_section_header("Crop Production Analytics")
    prod_by_crop_year = filtered.groupby(["year", "crop"])["production_tonnes"].sum().reset_index()
    fig = px.line(prod_by_crop_year, x="year", y="production_tonnes", color="crop", markers=True)
    st.plotly_chart(apply_plotly_theme(fig, title="Production Trend by Crop"), width="stretch")

    render_section_header("Yield Comparison Across Crops")
    yield_compare = filtered.groupby("crop")["yield_t_per_ha"].agg(["mean", "std"]).reset_index()
    yield_compare.columns = ["crop", "avg_yield", "yield_std"]
    fig2 = px.bar(yield_compare.sort_values("avg_yield", ascending=False), x="crop", y="avg_yield",
                   error_y="yield_std", labels={"avg_yield": "Avg Yield (t/ha)", "crop": ""})
    st.plotly_chart(apply_plotly_theme(fig2, title="Average Yield by Crop (± std dev)"), width="stretch")

    render_section_header("Yield Distribution")
    fig3 = px.box(filtered, x="crop", y="yield_t_per_ha", color="crop")
    fig3.update_layout(showlegend=False)
    st.plotly_chart(apply_plotly_theme(fig3, title="Yield Spread by Crop", height=460), width="stretch")

# ---------------------------------------------------------------------------
# TAB 2: Rankings
# ---------------------------------------------------------------------------
with tabs[1]:
    render_section_header("State Rankings")
    state_rank = filtered.groupby("state").agg(
        production_tonnes=("production_tonnes", "sum"),
        avg_yield=("yield_t_per_ha", "mean"),
        profit_inr=("profit_inr", "sum"),
    ).reset_index().sort_values("production_tonnes", ascending=False)
    state_rank.insert(0, "Rank", range(1, len(state_rank) + 1))
    st.dataframe(
        state_rank.style.format({
            "production_tonnes": "{:,.0f}", "avg_yield": "{:.2f}", "profit_inr": "₹{:,.0f}",
        }),
        width="stretch", hide_index=True,
    )

    render_section_header("District Rankings")
    district_rank = filtered.groupby(["state", "district"]).agg(
        production_tonnes=("production_tonnes", "sum"),
        avg_yield=("yield_t_per_ha", "mean"),
    ).reset_index().sort_values("production_tonnes", ascending=False).head(20)
    district_rank.insert(0, "Rank", range(1, len(district_rank) + 1))
    st.dataframe(
        district_rank.style.format({"production_tonnes": "{:,.0f}", "avg_yield": "{:.2f}"}),
        width="stretch", hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 3: Seasonal Analysis
# ---------------------------------------------------------------------------
with tabs[2]:
    render_section_header("Seasonal Production Analysis")
    season_agg = filtered.groupby("season").agg(
        production_tonnes=("production_tonnes", "sum"),
        area_ha=("area_ha", "sum"),
        avg_yield=("yield_t_per_ha", "mean"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig4 = px.pie(season_agg, names="season", values="production_tonnes", hole=0.45)
        st.plotly_chart(apply_plotly_theme(fig4, title="Production Share by Season"), width="stretch")
    with c2:
        fig5 = px.bar(season_agg, x="season", y="avg_yield", labels={"avg_yield": "Avg Yield (t/ha)"})
        st.plotly_chart(apply_plotly_theme(fig5, title="Average Yield by Season"), width="stretch")

    render_section_header("Crop-Season Heatmap")
    pivot = filtered.pivot_table(index="crop", columns="season", values="yield_t_per_ha", aggfunc="mean")
    fig6 = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale="Greens")
    st.plotly_chart(apply_plotly_theme(fig6, title="Avg Yield: Crop × Season", height=500), width="stretch")

# ---------------------------------------------------------------------------
# TAB 4: Water & Irrigation
# ---------------------------------------------------------------------------
with tabs[3]:
    render_section_header("Water Consumption Analysis")
    c1, c2 = st.columns(2)
    with c1:
        water_by_crop = filtered.groupby("crop")["water_use_mm"].mean().reset_index().sort_values("water_use_mm", ascending=False)
        fig7 = px.bar(water_by_crop, x="crop", y="water_use_mm", labels={"water_use_mm": "Water Use (mm)"})
        fig7.update_traces(marker_color="#1565C0")
        st.plotly_chart(apply_plotly_theme(fig7, title="Avg Water Use by Crop"), width="stretch")
    with c2:
        fig8 = px.scatter(filtered.sample(min(500, len(filtered))), x="water_use_mm", y="yield_t_per_ha",
                            color="crop", trendline=None, opacity=0.6)
        st.plotly_chart(apply_plotly_theme(fig8, title="Water Use vs Yield"), width="stretch")

    render_section_header("Irrigation Efficiency (Yield per mm Water)")
    filtered_copy = filtered.copy()
    filtered_copy["water_efficiency"] = filtered_copy["yield_t_per_ha"] / filtered_copy["water_use_mm"]
    eff = filtered_copy.groupby("crop")["water_efficiency"].mean().reset_index().sort_values("water_efficiency", ascending=False)
    fig9 = px.bar(eff, x="crop", y="water_efficiency", labels={"water_efficiency": "t/ha per mm water"})
    fig9.update_traces(marker_color="#2E7D32")
    st.plotly_chart(apply_plotly_theme(fig9, title="Water Use Efficiency by Crop"), width="stretch")

# ---------------------------------------------------------------------------
# TAB 5: Soil Analysis
# ---------------------------------------------------------------------------
with tabs[4]:
    render_section_header("Soil Nutrient Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig10 = px.box(filtered, x="soil_type", y="nitrogen_n", color="soil_type")
        fig10.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig10, title="Nitrogen by Soil Type"), width="stretch")
    with c2:
        fig11 = px.box(filtered, x="soil_type", y="phosphorus_p", color="soil_type")
        fig11.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig11, title="Phosphorus by Soil Type"), width="stretch")
    with c3:
        fig12 = px.box(filtered, x="soil_type", y="potassium_k", color="soil_type")
        fig12.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig12, title="Potassium by Soil Type"), width="stretch")

    render_section_header("Soil pH Distribution")
    fig13 = px.histogram(filtered, x="soil_ph", color="soil_type", nbins=30, opacity=0.7)
    st.plotly_chart(apply_plotly_theme(fig13, title="Soil pH Distribution by Soil Type"), width="stretch")

# ---------------------------------------------------------------------------
# TAB 6: Profit Analytics
# ---------------------------------------------------------------------------
with tabs[5]:
    render_section_header("Profitability Overview")
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Total Revenue", format_inr(filtered["revenue_inr"].sum()))
    with k2:
        render_kpi_card("Total Cost", format_inr(filtered["total_cost_inr"].sum()))
    with k3:
        render_kpi_card("Net Profit", format_inr(filtered["profit_inr"].sum()))

    profit_by_crop = filtered.groupby("crop").agg(
        revenue_inr=("revenue_inr", "sum"), profit_inr=("profit_inr", "sum"),
    ).reset_index().sort_values("profit_inr", ascending=False)
    fig14 = px.bar(profit_by_crop, x="crop", y=["revenue_inr", "profit_inr"], barmode="group",
                    labels={"value": "INR", "crop": "", "variable": ""})
    st.plotly_chart(apply_plotly_theme(fig14, title="Revenue vs Profit by Crop"), width="stretch")

    render_section_header("Profit Margin by State")
    state_profit = filtered.groupby("state").agg(
        revenue_inr=("revenue_inr", "sum"), profit_inr=("profit_inr", "sum"),
    ).reset_index()
    state_profit["margin_pct"] = (state_profit["profit_inr"] / state_profit["revenue_inr"] * 100).round(1)
    fig15 = px.bar(state_profit.sort_values("margin_pct", ascending=False), x="state", y="margin_pct",
                    labels={"margin_pct": "Profit Margin (%)"})
    fig15.update_traces(marker_color="#F9A825")
    st.plotly_chart(apply_plotly_theme(fig15, title="Profit Margin by State"), width="stretch")

# ---------------------------------------------------------------------------
# TAB 7: Agricultural Growth Index
# ---------------------------------------------------------------------------
with tabs[6]:
    render_section_header("Agricultural Growth Index (composite YoY score)")
    st.caption(
        "A simple composite index (production + yield + profit growth, equally weighted) "
        "tracking overall agricultural momentum year over year for the selected filters."
    )
    yearly = filtered.groupby("year").agg(
        production_tonnes=("production_tonnes", "sum"),
        avg_yield=("yield_t_per_ha", "mean"),
        profit_inr=("profit_inr", "sum"),
    ).sort_index().reset_index()

    for col in ["production_tonnes", "avg_yield", "profit_inr"]:
        yearly[f"{col}_growth"] = yearly[col].pct_change() * 100

    yearly["growth_index"] = yearly[[c for c in yearly.columns if c.endswith("_growth")]].mean(axis=1)

    fig16 = px.line(yearly.dropna(), x="year", y="growth_index", markers=True,
                      labels={"growth_index": "Growth Index (%)"})
    fig16.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(apply_plotly_theme(fig16, title="Agricultural Growth Index Over Time"), width="stretch")

    st.dataframe(
        yearly[["year", "production_tonnes", "avg_yield", "profit_inr", "growth_index"]].style.format({
            "production_tonnes": "{:,.0f}", "avg_yield": "{:.2f}",
            "profit_inr": "₹{:,.0f}", "growth_index": "{:.1f}%",
        }),
        width="stretch", hide_index=True,
    )
