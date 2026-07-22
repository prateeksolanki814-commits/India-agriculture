"""
pages/4_🗺️_Maps_GIS.py
=========================
Interactive Folium maps for spatial agriculture analysis: crop
distribution, rainfall, soil quality, irrigation coverage, production
heatmaps, and flood/drought risk zones -- all plotted at the state level
using the centroid coordinates defined in config.py.
"""

import sys
import os
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS, STATE_COORDS
from utils.styling import inject_custom_css, render_hero, render_section_header
from utils.helpers import get_agri_data, format_number

st.set_page_config(page_title="Maps & GIS | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("🗺️ Maps & GIS", "Spatial view of production, rainfall, soil, and risk across India")

df = get_agri_data()
latest_year = df["year"].max()
latest = df[df["year"] == latest_year]

map_type = st.selectbox(
    "Select Map Layer",
    [
        "🌾 Crop Distribution", "🌧️ Rainfall", "🧫 Soil Quality",
        "💧 Irrigation Coverage (Water Use)", "🔥 Production Heatmap",
        "🌊 Flood Risk Zones", "🏜️ Drought Risk Zones",
    ],
)

state_metrics = latest.groupby("state").agg(
    production_tonnes=("production_tonnes", "sum"),
    rainfall_mm=("rainfall_mm", "mean"),
    water_use_mm=("water_use_mm", "mean"),
    soil_ph=("soil_ph", "mean"),
    flood_risk_score=("flood_risk_score", "mean"),
    drought_risk_score=("drought_risk_score", "mean"),
    top_crop=("crop", lambda s: s.value_counts().idxmax()),
).reset_index()
state_metrics["lat"] = state_metrics["state"].map(lambda s: STATE_COORDS[s][0])
state_metrics["lon"] = state_metrics["state"].map(lambda s: STATE_COORDS[s][1])


def risk_color(score):
    if score < 35:
        return "green"
    elif score < 65:
        return "orange"
    return "red"


render_section_header(map_type)

m = folium.Map(location=[22.5, 80], zoom_start=5, tiles="CartoDB positron")

if map_type == "🌾 Crop Distribution":
    for _, row in state_metrics.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=f"<b>{row['state']}</b><br>Dominant crop: {row['top_crop']}<br>"
                  f"Production: {format_number(row['production_tonnes'])} t",
            tooltip=row["state"],
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(m)

elif map_type == "🌧️ Rainfall":
    heat_data = [[row["lat"], row["lon"], row["rainfall_mm"]] for _, row in state_metrics.iterrows()]
    HeatMap(heat_data, radius=45, blur=30, max_zoom=6).add_to(m)
    for _, row in state_metrics.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=6, color="#1565C0", fill=True,
            popup=f"{row['state']}: {row['rainfall_mm']:.0f} mm avg rainfall",
        ).add_to(m)

elif map_type == "🧫 Soil Quality":
    for _, row in state_metrics.iterrows():
        ph_color = "green" if 6.0 <= row["soil_ph"] <= 7.5 else "orange"
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=12, color=ph_color, fill=True, fill_opacity=0.7,
            popup=f"<b>{row['state']}</b><br>Avg Soil pH: {row['soil_ph']:.2f}",
            tooltip=row["state"],
        ).add_to(m)

elif map_type == "💧 Irrigation Coverage (Water Use)":
    for _, row in state_metrics.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=max(6, row["water_use_mm"] / 40),
            color="#1565C0", fill=True, fill_opacity=0.6,
            popup=f"<b>{row['state']}</b><br>Avg Water Use: {row['water_use_mm']:.0f} mm",
            tooltip=row["state"],
        ).add_to(m)

elif map_type == "🔥 Production Heatmap":
    heat_data = [[row["lat"], row["lon"], row["production_tonnes"] / 1000] for _, row in state_metrics.iterrows()]
    HeatMap(heat_data, radius=45, blur=30, max_zoom=6).add_to(m)

elif map_type == "🌊 Flood Risk Zones":
    for _, row in state_metrics.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=14, color=risk_color(row["flood_risk_score"]),
            fill=True, fill_opacity=0.7,
            popup=f"<b>{row['state']}</b><br>Flood Risk Score: {row['flood_risk_score']:.0f}/100",
            tooltip=row["state"],
        ).add_to(m)

elif map_type == "🏜️ Drought Risk Zones":
    for _, row in state_metrics.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=14, color=risk_color(row["drought_risk_score"]),
            fill=True, fill_opacity=0.7,
            popup=f"<b>{row['state']}</b><br>Drought Risk Score: {row['drought_risk_score']:.0f}/100",
            tooltip=row["state"],
        ).add_to(m)

st_folium(m, width=1200, height=560, returned_objects=[])

render_section_header("State Snapshot Table")
display_cols = ["state", "top_crop", "production_tonnes", "rainfall_mm", "water_use_mm",
                 "soil_ph", "flood_risk_score", "drought_risk_score"]
st.dataframe(
    state_metrics[display_cols].sort_values("production_tonnes", ascending=False).style.format({
        "production_tonnes": "{:,.0f}", "rainfall_mm": "{:.0f}", "water_use_mm": "{:.0f}",
        "soil_ph": "{:.2f}", "flood_risk_score": "{:.0f}", "drought_risk_score": "{:.0f}",
    }),
    width="stretch", hide_index=True,
)
