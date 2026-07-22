"""
pages/2_🤖_AI_Predictions.py
==============================
Interactive hub for every ML model in the platform. Each model gets its
own tab with input widgets, a "Predict" button, and a result card showing
the prediction plus a model confidence score.
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_ICON, INDIAN_STATES, CROPS, SEASONS, SOIL_TYPES
from utils.styling import inject_custom_css, render_hero, render_section_header, render_badge
from utils.helpers import confidence_badge_color, format_inr

st.set_page_config(page_title="AI Predictions | AGRI VISION AI", page_icon=APP_ICON, layout="wide")
inject_custom_css()
render_hero("🤖 AI Predictions", "Sixteen ML-powered models for yield, risk, price, and sustainability forecasting")

tabs = st.tabs([
    "🌾 Crop Yield", "🌱 Crop Recommendation", "🧪 Fertilizer",
    "💰 Price & Profit", "🐛 Pest & Disease Risk", "🌊 Flood & Drought Risk",
    "♻️ Sustainability",
])

def confidence_line(confidence: float, model_note: str = ""):
    color = confidence_badge_color(confidence)
    st.markdown(
        f"**Model Confidence:** {render_badge(f'{confidence*100:.1f}%', color)} {model_note}",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TAB 1: Crop Yield Prediction
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Predict expected yield (tonnes/hectare) for given conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", INDIAN_STATES, key="yield_state")
        crop = st.selectbox("Crop", CROPS, key="yield_crop")
        season = st.selectbox("Season", SEASONS, key="yield_season")
    with c2:
        soil_type = st.selectbox("Soil Type", SOIL_TYPES, key="yield_soil")
        rainfall_mm = st.slider("Rainfall (mm)", 100, 2000, 800, key="yield_rain")
        avg_temp_c = st.slider("Avg Temperature (°C)", 10, 45, 26, key="yield_temp")
    with c3:
        humidity_pct = st.slider("Humidity (%)", 10, 100, 60, key="yield_hum")
        nitrogen_n = st.slider("Nitrogen (N)", 10, 160, 90, key="yield_n")
        phosphorus_p = st.slider("Phosphorus (P)", 10, 90, 40, key="yield_p")
    c4, c5 = st.columns(2)
    with c4:
        potassium_k = st.slider("Potassium (K)", 10, 110, 45, key="yield_k")
    with c5:
        soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.6, key="yield_ph")
    water_use_mm = st.slider("Water Use (mm)", 100, 1200, 450, key="yield_water")

    if st.button("🔮 Predict Yield", type="primary", key="btn_yield"):
        from ml.crop_yield_model import predict_yield
        with st.spinner("Running XGBoost yield model..."):
            result = predict_yield(
                state=state, crop=crop, season=season, soil_type=soil_type,
                rainfall_mm=rainfall_mm, avg_temp_c=avg_temp_c, humidity_pct=humidity_pct,
                nitrogen_n=nitrogen_n, phosphorus_p=phosphorus_p, potassium_k=potassium_k,
                soil_ph=soil_ph, water_use_mm=water_use_mm,
            )
        st.markdown(f"""
        <div class="rec-card">
            <h3>🌾 Predicted Yield: {result['predicted_yield_t_per_ha']} t/ha</h3>
        </div>
        """, unsafe_allow_html=True)
        confidence_line(result["confidence"], f"(Model R²: {result['model_r2']}, MAE: {result['model_mae']} t/ha)")

# ---------------------------------------------------------------------------
# TAB 2: Crop Recommendation
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("#### Get the top recommended crops for your soil & climate conditions")
    c1, c2 = st.columns(2)
    with c1:
        rn = st.slider("Nitrogen (N)", 10, 160, 90, key="rec_n")
        rp = st.slider("Phosphorus (P)", 10, 90, 42, key="rec_p")
        rk = st.slider("Potassium (K)", 10, 110, 48, key="rec_k")
        rph = st.slider("Soil pH", 4.5, 8.5, 6.7, key="rec_ph")
    with c2:
        rrain = st.slider("Rainfall (mm)", 100, 2000, 900, key="rec_rain")
        rtemp = st.slider("Avg Temperature (°C)", 10, 45, 26, key="rec_temp")
        rhum = st.slider("Humidity (%)", 10, 100, 65, key="rec_hum")
        top_n = st.number_input("Number of recommendations", 1, 10, 5, key="rec_topn")

    if st.button("🌱 Recommend Crops", type="primary", key="btn_reco"):
        from ml.crop_recommendation_model import recommend_crops
        with st.spinner("Running Random Forest recommendation model..."):
            results = recommend_crops(
                nitrogen_n=rn, phosphorus_p=rp, potassium_k=rk, soil_ph=rph,
                rainfall_mm=rrain, avg_temp_c=rtemp, humidity_pct=rhum, top_n=int(top_n),
            )
        for i, r in enumerate(results, 1):
            color = confidence_badge_color(r["confidence"])
            st.markdown(f"""
            <div class="rec-card">
                <b>#{i} {r['crop']}</b> &nbsp; {render_badge(f"{r['confidence']*100:.1f}% match", color)}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 3: Fertilizer Recommendation
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Get a fertilizer recommendation and nutrient dosage plan")
    c1, c2 = st.columns(2)
    with c1:
        fcrop = st.selectbox("Crop", CROPS, key="fert_crop")
        fn = st.slider("Current Nitrogen (N)", 10, 160, 60, key="fert_n")
    with c2:
        fp = st.slider("Current Phosphorus (P)", 10, 90, 30, key="fert_p")
        fk = st.slider("Current Potassium (K)", 10, 110, 25, key="fert_k")
    fph = st.slider("Soil pH", 4.5, 8.5, 6.9, key="fert_ph")

    if st.button("🧪 Recommend Fertilizer", type="primary", key="btn_fert"):
        from ml.fertilizer_model import recommend_fertilizer
        with st.spinner("Analyzing nutrient deficits..."):
            result = recommend_fertilizer(crop=fcrop, nitrogen_n=fn, phosphorus_p=fp,
                                            potassium_k=fk, soil_ph=fph)
        st.markdown(f"""
        <div class="rec-card">
            <h3>🧪 Recommended: {result['recommended_fertilizer']}</h3>
            <p>N deficit: <b>{result['nitrogen_deficit_kg_ha']} kg/ha</b> &nbsp;|&nbsp;
               P deficit: <b>{result['phosphorus_deficit_kg_ha']} kg/ha</b> &nbsp;|&nbsp;
               K deficit: <b>{result['potassium_deficit_kg_ha']} kg/ha</b></p>
            <p>💡 {result['soil_ph_note']}</p>
        </div>
        """, unsafe_allow_html=True)
        confidence_line(result["confidence"])

# ---------------------------------------------------------------------------
# TAB 4: Price & Profit Prediction
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### Predict market price and project profit")
    c1, c2, c3 = st.columns(3)
    with c1:
        pstate = st.selectbox("State", INDIAN_STATES, key="price_state")
        pcrop = st.selectbox("Crop", CROPS, key="price_crop")
    with c2:
        pseason = st.selectbox("Season", SEASONS, key="price_season")
        pyear = st.number_input("Year", 2024, 2030, 2026, key="price_year")
    with c3:
        pprod = st.number_input("Expected Production (tonnes)", 100, 200000, 40000, key="price_prod")
        parea = st.number_input("Cultivated Area (ha)", 100, 100000, 15000, key="price_area")
    pcost = st.slider("Estimated Cost (INR/ha)", 10000, 100000, 35000, key="price_cost")

    if st.button("💰 Predict Price & Profit", type="primary", key="btn_price"):
        from ml.price_prediction_model import predict_profit
        with st.spinner("Running LightGBM price model..."):
            result = predict_profit(
                year=int(pyear), state=pstate, crop=pcrop, season=pseason,
                production_tonnes=pprod, area_ha=parea, estimated_cost_per_ha=pcost,
            )
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Price", f"₹{result['predicted_price_per_quintal_inr']:.0f}/quintal")
        m2.metric("Projected Revenue", format_inr(result["projected_revenue_inr"]))
        m3.metric("Projected Profit", format_inr(result["projected_profit_inr"]),
                   f"{result['profit_margin_pct']:.1f}% margin")
        confidence_line(result["confidence"])

# ---------------------------------------------------------------------------
# TAB 5: Pest & Disease Risk
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("#### Assess pest and disease risk for a crop")
    c1, c2 = st.columns(2)
    with c1:
        dstate = st.selectbox("State", INDIAN_STATES, key="disease_state")
        dcrop = st.selectbox("Crop", CROPS, key="disease_crop")
        dhum = st.slider("Humidity (%)", 10, 100, 75, key="disease_hum")
    with c2:
        dtemp = st.slider("Avg Temperature (°C)", 10, 45, 27, key="disease_temp")
        drain = st.slider("Rainfall (mm)", 100, 2000, 1000, key="disease_rain")
        dph = st.slider("Soil pH", 4.5, 8.5, 6.4, key="disease_ph")

    if st.button("🐛 Assess Pest & Disease Risk", type="primary", key="btn_disease"):
        from ml.risk_models import predict_pest_risk, predict_disease_risk
        with st.spinner("Running Gradient Boosting risk models..."):
            pest = predict_pest_risk(dstate, dcrop, dhum, dtemp, drain)
            disease = predict_disease_risk(dstate, dcrop, dhum, dtemp, drain, dph)

        rc1, rc2 = st.columns(2)
        with rc1:
            color = {"Low": "green", "Medium": "orange", "High": "red"}[pest["risk_level"]]
            st.markdown(f"""
            <div class="rec-card">
                <h4>🐛 Pest Risk: {render_badge(pest['risk_level'], color)}</h4>
                <p>Confidence: {pest['confidence']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            color2 = {"Low": "green", "Medium": "orange", "High": "red"}[disease["risk_level"]]
            st.markdown(f"""
            <div class="rec-card">
                <h4>🦠 Disease Risk: {render_badge(disease['risk_level'], color2)}</h4>
                <p>Confidence: {disease['confidence']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 6: Flood & Drought Risk
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("#### Assess flood and drought risk for a region")
    c1, c2 = st.columns(2)
    with c1:
        fstate = st.selectbox("State", INDIAN_STATES, key="flood_state")
        fsoil = st.selectbox("Soil Type", SOIL_TYPES, key="flood_soil")
        frain = st.slider("Rainfall (mm)", 100, 2500, 900, key="flood_rain")
    with c2:
        fwater = st.slider("Water Use (mm)", 100, 1200, 500, key="flood_water")
        ftemp = st.slider("Avg Temperature (°C)", 10, 45, 28, key="flood_temp")
        fhum = st.slider("Humidity (%)", 10, 100, 55, key="flood_hum")

    if st.button("🌊 Assess Flood & Drought Risk", type="primary", key="btn_flood"):
        from ml.risk_models import predict_flood_risk, predict_drought_risk
        with st.spinner("Running risk classification models..."):
            flood = predict_flood_risk(fstate, fsoil, frain, fwater)
            drought = predict_drought_risk(fstate, fsoil, frain, ftemp, fhum)

        rc1, rc2 = st.columns(2)
        with rc1:
            color = {"Low": "green", "Medium": "orange", "High": "red"}[flood["risk_level"]]
            st.markdown(f"""
            <div class="rec-card">
                <h4>🌊 Flood Risk: {render_badge(flood['risk_level'], color)}</h4>
                <p>Confidence: {flood['confidence']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            color2 = {"Low": "green", "Medium": "orange", "High": "red"}[drought["risk_level"]]
            st.markdown(f"""
            <div class="rec-card">
                <h4>🏜️ Drought Risk: {render_badge(drought['risk_level'], color2)}</h4>
                <p>Confidence: {drought['confidence']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 7: Sustainability Score
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("#### Estimate sustainability score and carbon footprint")
    c1, c2 = st.columns(2)
    with c1:
        scrop = st.selectbox("Crop", CROPS, key="sus_crop")
        sn = st.slider("Nitrogen Use (N)", 10, 160, 90, key="sus_n")
        sp = st.slider("Phosphorus Use (P)", 10, 90, 42, key="sus_p")
    with c2:
        sk = st.slider("Potassium Use (K)", 10, 110, 45, key="sus_k")
        swater = st.slider("Water Use (mm)", 100, 1200, 450, key="sus_water")
        sarea = st.number_input("Cultivated Area (ha)", 100, 100000, 15000, key="sus_area")
    srain = st.slider("Rainfall (mm)", 100, 2000, 800, key="sus_rain")

    if st.button("♻️ Estimate Sustainability", type="primary", key="btn_sus"):
        from ml.sustainability_model import predict_sustainability
        with st.spinner("Running sustainability & carbon models..."):
            result = predict_sustainability(
                crop=scrop, nitrogen_n=sn, phosphorus_p=sp, potassium_k=sk,
                water_use_mm=swater, area_ha=sarea, rainfall_mm=srain,
            )
        rating_color = {"Excellent": "green", "Good": "green", "Needs Improvement": "orange", "Poor": "red"}
        st.markdown(f"""
        <div class="rec-card">
            <h3>♻️ Sustainability Score: {result['sustainability_score']} / 100
                {render_badge(result['rating'], rating_color[result['rating']])}</h3>
            <p>Estimated Carbon Footprint: <b>{result['estimated_carbon_footprint_kg_co2e']} kg CO2e</b></p>
            <p>Top Influencing Factor: <b>{result['top_influencing_factor'].replace('_', ' ').title()}</b></p>
        </div>
        """, unsafe_allow_html=True)
        confidence_line(result["confidence"], "(Note: sustainability scoring reflects noisy synthetic drivers — treat as directional, not precise)")
