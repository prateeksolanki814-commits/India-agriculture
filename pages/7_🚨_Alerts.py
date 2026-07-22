"""
pages/7_🚨_Alerts.py
======================
Smart Alerts: scans the latest data (and, where relevant, live model
outputs) across all states to surface actionable warnings -- heavy
rain, flood, drought, heatwave, pest/disease risk, and price
swings -- ranked by severity so the most urgent items appear first.
"""

import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES
from utils.styling import inject_custom_css, render_hero, render_section_header, render_badge
from utils.helpers import get_agri_data, get_weather_data, format_inr

st.set_page_config(page_title="Smart Alerts | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("🚨 Smart Alerts", "Automatically generated warnings across weather, risk, and market conditions")

df = get_agri_data()
weather_df = get_weather_data()
latest_year = df["year"].max()
latest = df[df["year"] == latest_year]

severity_filter = st.multiselect("Filter by severity", ["High", "Medium", "Low"], default=["High", "Medium"])

alerts = []  # each: (severity, category, state, message)

# ---------------------------------------------------------------------------
# Weather-based alerts (heavy rain, heatwave) from the daily weather series
# ---------------------------------------------------------------------------
recent_window = weather_df.groupby("state").tail(14)  # last ~2 weeks per state
for state, grp in recent_window.groupby("state"):
    max_rain = grp["rainfall_mm"].max()
    max_temp = grp["temperature_c"].max()
    if max_rain > 45:
        alerts.append(("High", "🌧️ Heavy Rain", state, f"Recent daily rainfall peaked at {max_rain:.0f} mm — risk of waterlogging."))
    elif max_rain > 25:
        alerts.append(("Medium", "🌧️ Heavy Rain", state, f"Elevated rainfall observed ({max_rain:.0f} mm) — monitor field drainage."))

    if max_temp > 40:
        alerts.append(("High", "🔥 Heatwave", state, f"Temperatures reached {max_temp:.1f}°C — heat stress risk for standing crops."))
    elif max_temp > 36:
        alerts.append(("Medium", "🔥 Heatwave", state, f"Temperatures reached {max_temp:.1f}°C — monitor for heat stress."))

# ---------------------------------------------------------------------------
# Risk-score based alerts (flood/drought/pest/disease) from latest agri data
# ---------------------------------------------------------------------------
state_risk = latest.groupby("state").agg(
    flood_risk_score=("flood_risk_score", "mean"),
    drought_risk_score=("drought_risk_score", "mean"),
    pest_risk_score=("pest_risk_score", "mean"),
    disease_risk_score=("disease_risk_score", "mean"),
).reset_index()

for _, row in state_risk.iterrows():
    if row["flood_risk_score"] >= 65:
        alerts.append(("High", "🌊 Flood Risk", row["state"], f"Flood risk score at {row['flood_risk_score']:.0f}/100 — high alert."))
    elif row["flood_risk_score"] >= 45:
        alerts.append(("Medium", "🌊 Flood Risk", row["state"], f"Flood risk score at {row['flood_risk_score']:.0f}/100 — monitor closely."))

    if row["drought_risk_score"] >= 65:
        alerts.append(("High", "🏜️ Drought", row["state"], f"Drought risk score at {row['drought_risk_score']:.0f}/100 — water conservation advised."))
    elif row["drought_risk_score"] >= 45:
        alerts.append(("Medium", "🏜️ Drought", row["state"], f"Drought risk score at {row['drought_risk_score']:.0f}/100 — monitor irrigation needs."))

    if row["pest_risk_score"] >= 65:
        alerts.append(("High", "🐛 Pest Attack", row["state"], f"Pest risk score at {row['pest_risk_score']:.0f}/100 — inspect crops closely."))

    if row["disease_risk_score"] >= 65:
        alerts.append(("High", "🦠 Disease Risk", row["state"], f"Disease risk score at {row['disease_risk_score']:.0f}/100 — consider preventive treatment."))

# ---------------------------------------------------------------------------
# Price-based alerts (large YoY swings per crop)
# ---------------------------------------------------------------------------
price_by_year = df.groupby(["crop", "year"])["price_per_quintal_inr"].mean().reset_index()
for crop, grp in price_by_year.groupby("crop"):
    grp = grp.sort_values("year")
    if len(grp) < 2:
        continue
    latest_price, prev_price = grp.iloc[-1]["price_per_quintal_inr"], grp.iloc[-2]["price_per_quintal_inr"]
    pct = (latest_price - prev_price) / prev_price * 100 if prev_price else 0
    if pct <= -10:
        alerts.append(("High", "📉 Price Drop", crop, f"Price fell {abs(pct):.1f}% YoY to ₹{latest_price:.0f}/quintal."))
    elif pct >= 15:
        alerts.append(("Medium", "📈 Price Increase", crop, f"Price rose {pct:.1f}% YoY to ₹{latest_price:.0f}/quintal."))

# ---------------------------------------------------------------------------
# Water scarcity alerts (very low rainfall vs water use)
# ---------------------------------------------------------------------------
water_check = latest.groupby("state").agg(
    rainfall_mm=("rainfall_mm", "mean"), water_use_mm=("water_use_mm", "mean"),
).reset_index()
for _, row in water_check.iterrows():
    if row["water_use_mm"] > row["rainfall_mm"] * 1.3:
        alerts.append(("Medium", "💧 Water Scarcity", row["state"],
                        f"Water use ({row['water_use_mm']:.0f} mm) significantly exceeds rainfall "
                        f"({row['rainfall_mm']:.0f} mm) — groundwater dependency risk."))

# ---------------------------------------------------------------------------
# Render alerts, sorted by severity
# ---------------------------------------------------------------------------
severity_order = {"High": 0, "Medium": 1, "Low": 2}
alerts_df = pd.DataFrame(alerts, columns=["severity", "category", "location", "message"])
alerts_df = alerts_df[alerts_df["severity"].isin(severity_filter)]
alerts_df["sort_key"] = alerts_df["severity"].map(severity_order)
alerts_df = alerts_df.sort_values("sort_key").drop(columns="sort_key")

render_section_header(f"Active Alerts ({len(alerts_df)})")

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("🔴 High Severity", int((alerts_df["severity"] == "High").sum()))
with k2:
    st.metric("🟠 Medium Severity", int((alerts_df["severity"] == "Medium").sum()))
with k3:
    st.metric("🟢 Low Severity", int((alerts_df["severity"] == "Low").sum()))

if alerts_df.empty:
    st.success("✅ No alerts match the current filter — conditions look stable.")
else:
    css_class = {"High": "alert-high", "Medium": "alert-medium", "Low": "alert-low"}
    for _, row in alerts_df.iterrows():
        st.markdown(f"""
        <div class="alert-card {css_class[row['severity']]}">
            <b>{row['category']}</b> — {row['location']} &nbsp;
            {render_badge(row['severity'], {'High': 'red', 'Medium': 'orange', 'Low': 'green'}[row['severity']])}
            <br>{row['message']}
        </div>
        """, unsafe_allow_html=True)

st.caption(
    "Alerts are generated automatically from the latest data and model outputs using threshold "
    "rules (e.g. risk score ≥ 65 = High). Thresholds are illustrative and should be tuned against "
    "real regional baselines for production use."
)
