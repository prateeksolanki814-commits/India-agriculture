"""
ml/sustainability_model.py
=============================
Two related outputs bundled together since they share the same inputs:

  - Sustainability Score (0-100, higher = more sustainable) via a
    RandomForestRegressor trained on the dataset's sustainability_score.
  - Carbon Footprint Estimation (kg CO2e/hectare) via a second regressor
    trained on carbon_footprint_kg_co2e.

Both are framed around fertilizer use, water use, and area -- the
practical levers a farmer/policy-maker can act on -- so the model's
feature-importance output doubles as a simple "what's driving your
score" explanation.

Public API:
    train_and_save_model()
    load_model()
    predict_sustainability(...)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILES, RANDOM_SEED

FEATURE_COLUMNS = [
    "crop", "nitrogen_n", "phosphorus_p", "potassium_k",
    "water_use_mm", "area_ha", "rainfall_mm",
]
CATEGORICAL_COLUMNS = ["crop"]


def _encode(df, encoders=None, fit=False):
    df = df.copy()
    encoders = encoders or {}
    for col in CATEGORICAL_COLUMNS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    return df, encoders


def train_and_save_model(df: pd.DataFrame = None) -> dict:
    if df is None:
        from utils.helpers import get_agri_data
        df = get_agri_data()

    data = df[FEATURE_COLUMNS + ["sustainability_score", "carbon_footprint_kg_co2e"]].dropna()
    X_raw = data[FEATURE_COLUMNS]
    X_encoded, encoders = _encode(X_raw, fit=True)

    y_sustain = data["sustainability_score"]
    y_carbon = data["carbon_footprint_kg_co2e"]

    X_train, X_test, ys_train, ys_test, yc_train, yc_test = train_test_split(
        X_encoded, y_sustain, y_carbon, test_size=0.2, random_state=RANDOM_SEED
    )

    sustain_model = RandomForestRegressor(
        n_estimators=250, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1,
    )
    sustain_model.fit(X_train, ys_train)
    sustain_preds = sustain_model.predict(X_test)
    sustain_mae = mean_absolute_error(ys_test, sustain_preds)
    sustain_r2 = r2_score(ys_test, sustain_preds)

    carbon_model = RandomForestRegressor(
        n_estimators=250, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1,
    )
    carbon_model.fit(X_train, yc_train)
    carbon_preds = carbon_model.predict(X_test)
    carbon_mae = mean_absolute_error(yc_test, carbon_preds)
    carbon_r2 = r2_score(yc_test, carbon_preds)

    bundle = {
        "sustain_model": sustain_model, "carbon_model": carbon_model,
        "encoders": encoders, "feature_columns": FEATURE_COLUMNS,
        "sustain_mae": sustain_mae, "sustain_r2": sustain_r2,
        "carbon_mae": carbon_mae, "carbon_r2": carbon_r2,
    }
    joblib.dump(bundle, MODEL_FILES["sustainability"])
    return {
        "sustainability_mae": sustain_mae, "sustainability_r2": sustain_r2,
        "carbon_mae": carbon_mae, "carbon_r2": carbon_r2,
    }


def load_model():
    if not os.path.exists(MODEL_FILES["sustainability"]):
        train_and_save_model()
    return joblib.load(MODEL_FILES["sustainability"])


def predict_sustainability(crop, nitrogen_n, phosphorus_p, potassium_k,
                            water_use_mm, area_ha, rainfall_mm=800) -> dict:
    bundle = load_model()
    sustain_model, carbon_model, encoders = (
        bundle["sustain_model"], bundle["carbon_model"], bundle["encoders"]
    )

    row = pd.DataFrame([{
        "crop": crop, "nitrogen_n": nitrogen_n, "phosphorus_p": phosphorus_p,
        "potassium_k": potassium_k, "water_use_mm": water_use_mm, "area_ha": area_ha,
        "rainfall_mm": rainfall_mm,
    }])
    encoded_row, _ = _encode(row, encoders=encoders, fit=False)
    encoded_row = encoded_row[bundle["feature_columns"]]

    sustain_score = float(np.clip(sustain_model.predict(encoded_row)[0], 0, 100))
    carbon_kg_co2e = float(max(0, carbon_model.predict(encoded_row)[0]))

    # Feature importances double as a plain-language "what's driving this".
    importances = dict(zip(bundle["feature_columns"], sustain_model.feature_importances_))
    top_driver = max(importances, key=importances.get)

    if sustain_score >= 70:
        rating = "Excellent"
    elif sustain_score >= 50:
        rating = "Good"
    elif sustain_score >= 30:
        rating = "Needs Improvement"
    else:
        rating = "Poor"

    return {
        "sustainability_score": round(sustain_score, 1),
        "rating": rating,
        "estimated_carbon_footprint_kg_co2e": round(carbon_kg_co2e, 1),
        "top_influencing_factor": top_driver,
        "confidence": round(float(np.clip(bundle["sustain_r2"], 0.3, 0.95)), 3),
    }


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Training metrics:", metrics)
    result = predict_sustainability(
        crop="Wheat", nitrogen_n=95, phosphorus_p=42, potassium_k=38,
        water_use_mm=420, area_ha=15000, rainfall_mm=650,
    )
    print("Sustainability prediction:", result)
