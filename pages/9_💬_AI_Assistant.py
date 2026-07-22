"""
pages/9_💬_AI_Assistant.py
============================
A rule-based, data-grounded chatbot. Rather than calling an external LLM
(no API key is configured in this project), it parses the user's question
for known entities (state/crop/metric keywords) and answers directly from
the platform's own dataset -- so every answer is verifiable and never
hallucinated. This keeps the assistant genuinely useful offline and
demonstrates the "answers using project data" requirement transparently.
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS
from utils.styling import inject_custom_css, render_hero, render_section_header
from utils.helpers import get_agri_data, format_inr, format_number

st.set_page_config(page_title="AI Assistant | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("💬 AI Assistant", "Ask about any state, crop, or trend — answered directly from the platform's data")

df = get_agri_data()
latest_year = df["year"].max()


def _find_entity(text, options):
    text_lower = text.lower()
    for opt in options:
        if opt.lower() in text_lower:
            return opt
    return None


def answer_question(question: str) -> str:
    """
    Deterministic intent router: detects a state and/or crop mentioned in
    the question, plus a handful of metric keywords, and returns a data-
    grounded answer built directly from pandas aggregations.
    """
    q = question.lower()
    state = _find_entity(question, INDIAN_STATES)
    crop = _find_entity(question, CROPS)

    scoped = df.copy()
    scope_desc = "across all of India"
    if state:
        scoped = scoped[scoped["state"] == state]
        scope_desc = f"in {state}"
    if crop:
        scoped = scoped[scoped["crop"] == crop]
        scope_desc += f" for {crop}" if state else f" for {crop} across India"

    if scoped.empty:
        return f"I don't have data matching that combination ({scope_desc}). Try a different state or crop."

    latest_scoped = scoped[scoped["year"] == latest_year]
    active = latest_scoped if not latest_scoped.empty else scoped

    if any(w in q for w in ["yield", "productivity"]):
        val = active["yield_t_per_ha"].mean()
        return f"The average yield {scope_desc} is **{val:.2f} tonnes/hectare** (latest available year: {latest_year})."

    if any(w in q for w in ["production", "how much", "output"]):
        val = active["production_tonnes"].sum()
        return f"Total production {scope_desc} was **{format_number(val, ' tonnes')}** in {latest_year}."

    if "rain" in q:
        val = active["rainfall_mm"].mean()
        return f"The average rainfall {scope_desc} is **{val:.0f} mm** (based on {latest_year} data)."

    if "temperature" in q or "temp" in q or "hot" in q or "cold" in q:
        val = active["avg_temp_c"].mean()
        return f"The average temperature {scope_desc} is **{val:.1f}°C**."

    if ("price" in q) or ("cost" in q and "profit" not in q):
        val = active["price_per_quintal_inr"].mean()
        return f"The average market price {scope_desc} is **₹{val:.0f} per quintal** (latest year: {latest_year})."

    if "profit" in q or "income" in q or "earn" in q:
        val = active["profit_inr"].sum()
        return f"Estimated total profit {scope_desc} is **{format_inr(val)}** for {latest_year}."

    if any(w in q for w in ["best", "top", "highest"]):
        if "yield" in q:
            top = scoped.groupby("crop")["yield_t_per_ha"].mean().idxmax()
            return f"The crop with the highest average yield {scope_desc} is **{top}**."
        top_crop = scoped.groupby("crop")["production_tonnes"].sum().idxmax()
        if not state:
            top_state = scoped.groupby("state")["production_tonnes"].sum().idxmax()
            return f"The top-producing crop {scope_desc} is **{top_crop}**, and the leading state is **{top_state}**."
        return f"The top-producing crop {scope_desc} is **{top_crop}**."

    if "risk" in q or "drought" in q or "flood" in q or "pest" in q or "disease" in q:
        drought = active["drought_risk_score"].mean()
        flood = active["flood_risk_score"].mean()
        pest = active["pest_risk_score"].mean()
        return (f"Risk scores {scope_desc} (0-100 scale): "
                f"Drought **{drought:.0f}**, Flood **{flood:.0f}**, Pest **{pest:.0f}**.")

    if "sustain" in q or "carbon" in q or "eco" in q:
        val = active["sustainability_score"].mean()
        return f"The average sustainability score {scope_desc} is **{val:.0f}/100**."

    prod = active["production_tonnes"].sum()
    yld = active["yield_t_per_ha"].mean()
    return (
        f"Here's a quick overview {scope_desc}: total production of {format_number(prod, ' tonnes')} "
        f"and an average yield of {yld:.2f} t/ha (latest year: {latest_year}). "
        f"Try asking about yield, production, rainfall, price, profit, risk, or 'best crop'."
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("bot", "Hi! I'm the AGRI VISION AI assistant. Ask me things like "
                 "*'What is the average yield of Wheat in Punjab?'* or "
                 "*'What is the drought risk in Rajasthan?'*")
    ]

render_section_header("Chat")

for role, msg in st.session_state.chat_history:
    css_class = "chat-user" if role == "user" else "chat-bot"
    label = "🧑 You" if role == "user" else "🤖 Assistant"
    st.markdown(f'<div class="{css_class}"><b>{label}:</b><br>{msg}</div>', unsafe_allow_html=True)

st.write("")
example_cols = st.columns(4)
examples = [
    "What is the yield of Rice in Punjab?",
    "Best crop in Maharashtra?",
    "Drought risk in Rajasthan?",
    "Profit for Cotton in Gujarat?",
]
clicked_example = None
for col, ex in zip(example_cols, examples):
    if col.button(ex, key=f"ex_{ex}"):
        clicked_example = ex

user_input = st.chat_input("Ask a question about India's agriculture data...")

question = clicked_example or user_input
if question:
    st.session_state.chat_history.append(("user", question))
    response = answer_question(question)
    st.session_state.chat_history.append(("bot", response))
    st.rerun()

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.caption(
    "This assistant answers using rule-based parsing directly over the platform's own dataset — "
    "every figure it states is computed live from the data, not generated freely."
)
