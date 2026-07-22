"""
utils/helpers.py
==================
Small, reusable utility functions shared across pages:
  - number formatting (Indian numbering system, currency, compact units)
  - cached data-loading wrappers around database/db_manager
  - a consistent Plotly layout template
  - year-over-year delta calculation for KPI cards
"""

import sys
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import THEME, PLOTLY_TEMPLATE_COLORS


# ---------------------------------------------------------------------------
# Cached data loaders (Streamlit caches these across reruns/pages)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading agriculture data...")
def get_agri_data() -> pd.DataFrame:
    from database.db_manager import load_agriculture_df
    return load_agriculture_df()


@st.cache_data(show_spinner="Loading weather data...")
def get_weather_data() -> pd.DataFrame:
    from database.db_manager import load_weather_df
    return load_weather_df()


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
def format_inr(value: float, compact: bool = True) -> str:
    """Formats a number as Indian Rupees, e.g. 12,340,000 -> '₹1.23 Cr'."""
    if value is None or pd.isna(value):
        return "₹0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if compact:
        if value >= 1e7:
            return f"{sign}₹{value / 1e7:.2f} Cr"
        elif value >= 1e5:
            return f"{sign}₹{value / 1e5:.2f} L"
        elif value >= 1e3:
            return f"{sign}₹{value / 1e3:.1f} K"
        return f"{sign}₹{value:.0f}"
    return f"{sign}₹{value:,.0f}"


def format_number(value: float, suffix: str = "") -> str:
    """Compact number formatting, e.g. 1,230,000 -> '1.23M'."""
    if value is None or pd.isna(value):
        return "0"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M{suffix}"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K{suffix}"
    return f"{value:,.1f}{suffix}"


def pct_change(current: float, previous: float) -> float:
    """Safe percentage-change calculation (handles zero/None previous)."""
    if previous in (None, 0) or pd.isna(previous):
        return 0.0
    return ((current - previous) / abs(previous)) * 100


def yoy_kpi(df: pd.DataFrame, value_col: str, agg: str = "sum"):
    """
    Given a dataframe with a 'year' column, returns (latest_value,
    previous_value, pct_delta) for the given metric aggregated by year.
    Used to populate KPI card deltas on the Dashboard.
    """
    by_year = df.groupby("year")[value_col].agg(agg).sort_index()
    if len(by_year) < 2:
        latest = by_year.iloc[-1] if len(by_year) else 0
        return latest, latest, 0.0
    latest, previous = by_year.iloc[-1], by_year.iloc[-2]
    return latest, previous, pct_change(latest, previous)


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def apply_plotly_theme(fig: go.Figure, height: int = 420, title: str = None) -> go.Figure:
    """Applies a consistent look to every chart across the app."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        font=dict(family="Inter, sans-serif", size=13, color=THEME["text_dark"]),
        colorway=PLOTLY_TEMPLATE_COLORS,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#E0E0E0")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2ED", showline=False)
    return fig


def confidence_badge_color(confidence: float) -> str:
    """Maps a 0-1 model confidence score to a semantic color name."""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    return "red"
