"""
utils/styling.py
==================
Injects custom CSS into every Streamlit page so the app looks like a
designed product rather than a default Streamlit app. Also provides small
reusable HTML-snippet builders (KPI cards, badges, section headers) so
pages don't hand-roll markup individually.
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import THEME, APP_NAME, APP_ICON, APP_TAGLINE


def inject_custom_css():
    """Call once at the top of every page (after st.set_page_config)."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .main {{
            background-color: {THEME['background']};
        }}

        h1, h2, h3, h4 {{
            font-family: 'Poppins', sans-serif !important;
            color: {THEME['text_dark']} !important;
        }}

        /* ---------- Top hero banner ---------- */
        .agri-hero {{
            background: linear-gradient(135deg, {THEME['primary']} 0%, {THEME['primary_light']} 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(46, 125, 50, 0.25);
        }}
        .agri-hero h1 {{
            color: white !important;
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
        }}
        .agri-hero p {{
            color: rgba(255,255,255,0.92);
            font-size: 1.02rem;
            margin: 0;
        }}

        /* ---------- KPI cards ---------- */
        .kpi-card {{
            background: {THEME['card_bg']};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border-left: 5px solid {THEME['primary']};
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.82rem;
            color: {THEME['text_muted']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-size: 1.7rem;
            font-weight: 700;
            color: {THEME['text_dark']};
            font-family: 'Poppins', sans-serif;
        }}
        .kpi-delta-up {{
            color: {THEME['success']};
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .kpi-delta-down {{
            color: {THEME['danger']};
            font-size: 0.85rem;
            font-weight: 600;
        }}

        /* ---------- Section header ---------- */
        .section-header {{
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.3rem;
            color: {THEME['text_dark']};
            border-left: 5px solid {THEME['secondary']};
            padding-left: 0.7rem;
            margin: 1.4rem 0 0.8rem 0;
        }}

        /* ---------- Badges ---------- */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .badge-green {{ background: #E6F4EA; color: {THEME['success']}; }}
        .badge-orange {{ background: #FFF3E0; color: {THEME['warning']}; }}
        .badge-red {{ background: #FDECEA; color: {THEME['danger']}; }}
        .badge-blue {{ background: #E3F2FD; color: {THEME['accent']}; }}

        /* ---------- Alert cards ---------- */
        .alert-card {{
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
            border-left: 5px solid;
        }}
        .alert-high {{ background: #FDECEA; border-color: {THEME['danger']}; }}
        .alert-medium {{ background: #FFF3E0; border-color: {THEME['warning']}; }}
        .alert-low {{ background: #E6F4EA; border-color: {THEME['success']}; }}

        /* ---------- Recommendation card ---------- */
        .rec-card {{
            background: white;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            margin-bottom: 0.9rem;
            border-top: 4px solid {THEME['secondary']};
        }}

        /* ---------- Chat bubbles ---------- */
        .chat-user {{
            background: {THEME['accent']};
            color: white;
            padding: 0.7rem 1rem;
            border-radius: 14px 14px 2px 14px;
            margin: 0.4rem 0;
            max-width: 80%;
            margin-left: auto;
        }}
        .chat-bot {{
            background: white;
            color: {THEME['text_dark']};
            padding: 0.7rem 1rem;
            border-radius: 14px 14px 14px 2px;
            margin: 0.4rem 0;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] {{
            background-color: #F0F5EF;
        }}

        /* Hide default Streamlit footer/menu clutter */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def render_hero(title: str = None, subtitle: str = None):
    """Renders the shared gradient hero banner at the top of a page."""
    title = title or f"{APP_ICON} {APP_NAME}"
    subtitle = subtitle or APP_TAGLINE
    st.markdown(f"""
    <div class="agri-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, delta: str = None, delta_positive: bool = True):
    """Renders a single KPI card. Use inside st.columns(...) for a row of KPIs."""
    delta_html = ""
    if delta:
        cls = "kpi-delta-up" if delta_positive else "kpi-delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, color: str = "green"):
    """color in {green, orange, red, blue}"""
    return f'<span class="badge badge-{color}">{text}</span>'


def risk_badge(score: float) -> str:
    """Maps a 0-100 risk score to a colored badge (low/medium/high)."""
    if score < 35:
        return render_badge(f"Low ({score:.0f})", "green")
    elif score < 65:
        return render_badge(f"Medium ({score:.0f})", "orange")
    return render_badge(f"High ({score:.0f})", "red")
