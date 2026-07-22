"""
pages/5_💡_Recommendations.py
===============================
Smart Recommendation Engine: given a farmer's location, crop, and soil
conditions, generates an integrated set of AI-driven recommendations --
best crop, sowing/harvest timing, fertilizer plan, irrigation guidance,
risk mitigation, water-saving tips, expected profit, and relevant
government scheme pointers -- by combining outputs from multiple models.
"""

import sys
import os
import datetime
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS, SEASONS, SOIL_TYPES
from utils.styling import inject_custom_css, render_hero, render_section_header, render_badge
from utils.helpers import get_agri_data, format_inr, confidence_badge_color

st.set_page_config(page_title="Recommendations | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("💡 Smart Recommendation Engine", "One integrated AI-driven plan: crop, timing, fertilizer, irrigation, and profit")

df = get_agri_data()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
render_section_header("Tell us about your field")
c1, c2, c3 = st.columns(3)
with c1:
    state = st.selectbox("State", INDIAN_STATES)
    season = st.selectbox("Season", SEASONS)
with c2:
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.7)
with c3:
    nitrogen_n = st.slider("Nitrogen (N)", 10, 160, 85)
    phosphorus_p = st.slider("Phosphorus (P)", 10, 90, 40)

c4, c5, c6 = st.columns(3)
with c4:
    potassium_k = st.slider("Potassium (K)", 10, 110, 45)
with c5:
    rainfall_mm = st.slider("Expected Rainfall (mm)", 100, 2000, 850)
with c6:
    avg_temp_c = st.slider("Expected Avg Temperature (°C)", 10, 45, 27)

humidity_pct = st.slider("Expected Humidity (%)", 10, 100, 62)
area_ha = st.number_input("Cultivated Area (ha)", 100, 100000, 5000)

# Government scheme pointers -- static reference info, kept separate from
# ML outputs since scheme eligibility is a policy fact, not a prediction.
GOV_SCHEMES = [
    ("PM-KISAN", "Direct income support of ₹6,000/year for landholding farmer families."),
    ("PMFBY (Fasal Bima Yojana)", "Crop insurance covering yield losses from natural calamities, pests, and disease."),
    ("Soil Health Card Scheme", "Free periodic soil testing with crop-wise nutrient recommendations."),
    ("PM Krishi Sinchai Yojana", "Subsidies for micro-irrigation (drip/sprinkler) to improve water-use efficiency."),
    ("Kisan Credit Card (KCC)", "Low-interest short-term credit for seed, fertilizer, and input costs."),
]

if st.button("🚀 Generate My Recommendations", type="primary"):
    from ml.crop_recommendation_model import recommend_crops
    from ml.fertilizer_model import recommend_fertilizer
    from ml.price_prediction_model import predict_profit
    from ml.risk_models import predict_pest_risk, predict_disease_risk, predict_flood_risk, predict_drought_risk
    from ml.sustainability_model import predict_sustainability

    with st.spinner("Combining AI models to build your recommendation plan..."):
        crop_recs = recommend_crops(
            nitrogen_n=nitrogen_n, phosphorus_p=phosphorus_p, potassium_k=potassium_k,
            soil_ph=soil_ph, rainfall_mm=rainfall_mm, avg_temp_c=avg_temp_c,
            humidity_pct=humidity_pct, top_n=3,
        )
        best_crop = crop_recs[0]["crop"]

        fert_rec = recommend_fertilizer(
            crop=best_crop, nitrogen_n=nitrogen_n, phosphorus_p=phosphorus_p,
            potassium_k=potassium_k, soil_ph=soil_ph,
        )

        profit_proj = predict_profit(
            year=datetime.datetime.now().year + 1, state=state, crop=best_crop, season=season,
            production_tonnes=area_ha * 2.5, area_ha=area_ha,
        )

        pest = predict_pest_risk(state, best_crop, humidity_pct, avg_temp_c, rainfall_mm)
        disease = predict_disease_risk(state, best_crop, humidity_pct, avg_temp_c, rainfall_mm, soil_ph)
        flood = predict_flood_risk(state, soil_type, rainfall_mm, rainfall_mm * 0.6)
        drought = predict_drought_risk(state, soil_type, rainfall_mm, avg_temp_c, humidity_pct)

        sustain = predict_sustainability(
            crop=best_crop, nitrogen_n=nitrogen_n, phosphorus_p=phosphorus_p,
            potassium_k=potassium_k, water_use_mm=rainfall_mm * 0.6, area_ha=area_ha,
            rainfall_mm=rainfall_mm,
        )

    # --- Best Crop ---
    render_section_header("🌱 Recommended Crops")
    cols = st.columns(3)
    for i, (rec, col) in enumerate(zip(crop_recs, cols)):
        badge_color = confidence_badge_color(rec["suitability_confidence"] if "suitability_confidence" in rec else rec["confidence"])
        conf = rec.get("confidence", rec.get("suitability_confidence"))
        with col:
            st.markdown(f"""
            <div class="rec-card">
                <h4>{'🥇' if i == 0 else '🥈' if i == 1 else '🥉'} {rec['crop']}</h4>
                {render_badge(f'{conf*100:.1f}% match', badge_color)}
            </div>
            """, unsafe_allow_html=True)

    # --- Sowing / Harvest timing (rule-based on season, transparent logic) ---
    render_section_header("📅 Best Sowing & Harvest Window")
    SEASON_WINDOWS = {
        "Kharif": ("Mid-June to Mid-July (with monsoon onset)", "Late September to November"),
        "Rabi": ("Mid-October to Mid-December", "March to April"),
        "Zaid": ("March to April", "June to July"),
    }
    sow_window, harvest_window = SEASON_WINDOWS[season]
    c1, c2 = st.columns(2)
    c1.info(f"🌱 **Best Sowing Window:** {sow_window}")
    c2.info(f"🌾 **Best Harvest Window:** {harvest_window}")

    # --- Fertilizer plan ---
    render_section_header("🧪 Fertilizer Plan")
    st.markdown(f"""
    <div class="rec-card">
        <b>Recommended Fertilizer:</b> {fert_rec['recommended_fertilizer']}<br>
        N deficit: <b>{fert_rec['nitrogen_deficit_kg_ha']} kg/ha</b> &nbsp;|&nbsp;
        P deficit: <b>{fert_rec['phosphorus_deficit_kg_ha']} kg/ha</b> &nbsp;|&nbsp;
        K deficit: <b>{fert_rec['potassium_deficit_kg_ha']} kg/ha</b><br>
        💡 {fert_rec['soil_ph_note']}
    </div>
    """, unsafe_allow_html=True)

    # --- Irrigation & water-saving ---
    render_section_header("💧 Irrigation & Water-Saving Guidance")
    if drought["risk_level"] == "High":
        st.warning(
            "🏜️ Drought risk is **High**. Prioritize drip irrigation, mulching to retain soil "
            "moisture, and consider drought-tolerant crop varieties for this season."
        )
    elif flood["risk_level"] == "High":
        st.warning(
            "🌊 Flood risk is **High**. Ensure field drainage channels are clear and consider "
            "raised-bed planting to protect against waterlogging."
        )
    else:
        st.success(
            "✅ Water risk is manageable. Standard scheduled irrigation (every 7-10 days depending "
            "on crop stage) should be sufficient — consider drip irrigation to further improve efficiency."
        )

    # --- Risk mitigation ---
    render_section_header("⚠️ Risk Assessment & Mitigation")
    risk_cols = st.columns(4)
    risk_data = [
        ("🐛 Pest Risk", pest["risk_level"]), ("🦠 Disease Risk", disease["risk_level"]),
        ("🌊 Flood Risk", flood["risk_level"]), ("🏜️ Drought Risk", drought["risk_level"]),
    ]
    for col, (label, level) in zip(risk_cols, risk_data):
        color = {"Low": "green", "Medium": "orange", "High": "red"}[level]
        with col:
            st.markdown(f"**{label}**<br>{render_badge(level, color)}", unsafe_allow_html=True)

    # --- Expected profit ---
    render_section_header("💰 Expected Profit Projection")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Price", f"₹{profit_proj['predicted_price_per_quintal_inr']:.0f}/quintal")
    m2.metric("Projected Revenue", format_inr(profit_proj["projected_revenue_inr"]))
    m3.metric("Projected Profit", format_inr(profit_proj["projected_profit_inr"]),
               f"{profit_proj['profit_margin_pct']:.1f}% margin")

    st.markdown(f"""
    <div class="rec-card">
        ♻️ <b>Sustainability Score:</b> {sustain['sustainability_score']}/100 ({sustain['rating']}) &nbsp;|&nbsp;
        <b>Est. Carbon Footprint:</b> {sustain['estimated_carbon_footprint_kg_co2e']} kg CO2e
    </div>
    """, unsafe_allow_html=True)

    # --- Government schemes ---
    render_section_header("🏛️ Relevant Government Scheme Suggestions")
    for name, desc in GOV_SCHEMES:
        st.markdown(f"- **{name}** — {desc}")
    st.caption("Scheme details are illustrative reference information; confirm current eligibility with your local agriculture office.")

else:
    st.info("Fill in your field details above and click **Generate My Recommendations** to see your personalized AI plan.")
