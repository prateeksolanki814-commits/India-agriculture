"""
pages/6_🌦️_Weather.py
========================
Weather Intelligence: current/latest weather snapshot, Prophet-based
rainfall & temperature forecasts, humidity/wind summaries, climate
trends over the historical record, and extreme-weather alert flags.
"""

import sys
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES
from utils.styling import inject_custom_css, render_hero, render_section_header, render_kpi_card, render_badge
from utils.helpers import get_weather_data, apply_plotly_theme

st.set_page_config(page_title="Weather Intelligence | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("🌦️ Weather Intelligence", "Current conditions, forecasts, and climate trends for smarter farming decisions")

weather_df = get_weather_data()

state = st.selectbox("Select State", INDIAN_STATES)
days_ahead = st.slider("Forecast Horizon (days)", 7, 60, 30)

state_weather = weather_df[weather_df["state"] == state].sort_values("date")
latest_row = state_weather.iloc[-1]

# ---------------------------------------------------------------------------
# Current weather snapshot
# ---------------------------------------------------------------------------
render_section_header(f"Current Weather Snapshot — {state}")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Rainfall (latest day)", f"{latest_row['rainfall_mm']:.1f} mm")
with k2:
    render_kpi_card("Temperature", f"{latest_row['temperature_c']:.1f} °C")
with k3:
    render_kpi_card("Humidity", f"{latest_row['humidity_pct']:.0f} %")
with k4:
    render_kpi_card("Wind Speed", f"{latest_row['wind_speed_kmph']:.1f} km/h")

# ---------------------------------------------------------------------------
# Historical climate trend
# ---------------------------------------------------------------------------
render_section_header("Historical Climate Trend (Past Year)")
c1, c2 = st.columns(2)
with c1:
    fig1 = px.line(state_weather, x="date", y="rainfall_mm", labels={"rainfall_mm": "Rainfall (mm)"})
    fig1.update_traces(line=dict(color="#1565C0"))
    st.plotly_chart(apply_plotly_theme(fig1, title="Daily Rainfall"), width="stretch")
with c2:
    fig2 = px.line(state_weather, x="date", y="temperature_c", labels={"temperature_c": "Temperature (°C)"})
    fig2.update_traces(line=dict(color="#EF6C00"))
    st.plotly_chart(apply_plotly_theme(fig2, title="Daily Temperature"), width="stretch")

# ---------------------------------------------------------------------------
# AI Forecast (Prophet)
# ---------------------------------------------------------------------------
render_section_header(f"🤖 AI Forecast — Next {days_ahead} Days")

if st.button("🔮 Generate Forecast", type="primary"):
    from ml.rainfall_forecast_model import forecast_rainfall, forecast_temperature, get_forecast_summary

    with st.spinner("Running Prophet time-series models (first run per state may take a moment)..."):
        rain_fc = forecast_rainfall(state, days_ahead)
        temp_fc = forecast_temperature(state, days_ahead)
        summary = get_forecast_summary(state, days_ahead)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Expected Total Rainfall", f"{summary['total_expected_rainfall_mm']:.0f} mm")
    with k2:
        render_kpi_card("Expected Avg Temperature", f"{summary['avg_expected_temp_c']:.1f} °C")
    with k3:
        trend_icon = "📈" if summary["rainfall_trend"] == "increasing" else "📉"
        render_kpi_card("Rainfall Trend", f"{trend_icon} {summary['rainfall_trend'].title()}")
    with k4:
        trend_icon2 = "📈" if summary["temperature_trend"] == "increasing" else "📉"
        render_kpi_card("Temperature Trend", f"{trend_icon2} {summary['temperature_trend'].title()}")

    c3, c4 = st.columns(2)
    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=rain_fc["date"], y=rain_fc["upper_bound"], line=dict(width=0),
                                    showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=rain_fc["date"], y=rain_fc["lower_bound"], fill="tonexty",
                                    fillcolor="rgba(21,101,192,0.15)", line=dict(width=0),
                                    showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=rain_fc["date"], y=rain_fc["predicted"], mode="lines+markers",
                                    line=dict(color="#1565C0", width=3), name="Predicted Rainfall"))
        st.plotly_chart(apply_plotly_theme(fig3, title="Rainfall Forecast (80% confidence interval)"), width="stretch")
    with c4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=temp_fc["date"], y=temp_fc["upper_bound"], line=dict(width=0),
                                    showlegend=False, hoverinfo="skip"))
        fig4.add_trace(go.Scatter(x=temp_fc["date"], y=temp_fc["lower_bound"], fill="tonexty",
                                    fillcolor="rgba(239,108,0,0.15)", line=dict(width=0),
                                    showlegend=False, hoverinfo="skip"))
        fig4.add_trace(go.Scatter(x=temp_fc["date"], y=temp_fc["predicted"], mode="lines+markers",
                                    line=dict(color="#EF6C00", width=3), name="Predicted Temperature"))
        st.plotly_chart(apply_plotly_theme(fig4, title="Temperature Forecast (80% confidence interval)"), width="stretch")

    # --- Extreme weather alerts derived from the forecast itself ---
    render_section_header("🚨 Extreme Weather Flags (from forecast)")
    max_rain_day = rain_fc.loc[rain_fc["predicted"].idxmax()]
    max_temp_day = temp_fc.loc[temp_fc["predicted"].idxmax()]

    alert_cols = st.columns(2)
    with alert_cols[0]:
        if max_rain_day["predicted"] > 40:
            st.markdown(f"""<div class="alert-card alert-high">
            🌧️ <b>Heavy rain expected</b> around {max_rain_day['date'].strftime('%d %b %Y')}
            (~{max_rain_day['predicted']:.0f} mm)</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-card alert-low">
            ✅ No extreme rainfall spikes flagged in this forecast window.</div>""", unsafe_allow_html=True)
    with alert_cols[1]:
        if max_temp_day["predicted"] > 38:
            st.markdown(f"""<div class="alert-card alert-high">
            🔥 <b>Heatwave risk</b> around {max_temp_day['date'].strftime('%d %b %Y')}
            (~{max_temp_day['predicted']:.1f} °C)</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-card alert-low">
            ✅ No heatwave conditions flagged in this forecast window.</div>""", unsafe_allow_html=True)
else:
    st.info("Click **Generate Forecast** to run the Prophet forecasting models for this state.")
