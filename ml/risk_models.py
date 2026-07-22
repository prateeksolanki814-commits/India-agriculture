"""
ml/risk_models.py
====================
Four related risk-classification models bundled into one file since they
share the same pattern (train a classifier on a discretized risk-score
target, return risk level + probability):

  - Pest Risk Prediction
  - Crop Disease Risk Prediction
  - Flood Risk Prediction
  - Drought Risk Prediction

Each underlying *_risk_score column in the dataset (0-100, continuous) is
bucketed into Low/Medium/High, and a GradientBoostingClassifier learns to
predict the bucket from weather/soil/crop features. This gives a genuine
classification task with interpretable probability outputs, rather than
just re-displaying the synthetic score itself.

Public API:
    train_and_save_all()
    predict_pest_risk(...)
    predict_disease_risk(...)
    predict_flood_risk(...)
    predict_drought_risk(...)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILES, RANDOM_SEED

RISK_BUCKETS = ["Low", "Medium", "High"]


def _bucket_score(score: float) -> str:
    if score < 35:
        return "Low"
    elif score < 65:
        return "Medium"
    return "High"


# Each risk model's own feature set + which score column it targets.
RISK_CONFIGS = {
    "pest_risk": {
        "features": ["state", "crop", "humidity_pct", "avg_temp_c", "rainfall_mm"],
        "categorical": ["state", "crop"],
        "target": "pest_risk_score",
    },
    "disease_risk": {
        "features": ["state", "crop", "humidity_pct", "avg_temp_c", "rainfall_mm", "soil_ph"],
        "categorical": ["state", "crop"],
        "target": "disease_risk_score",
    },
    "flood_risk": {
        "features": ["state", "soil_type", "rainfall_mm", "water_use_mm"],
        "categorical": ["state", "soil_type"],
        "target": "flood_risk_score",
    },
    "drought_risk": {
        "features": ["state", "soil_type", "rainfall_mm", "avg_temp_c", "humidity_pct"],
        "categorical": ["state", "soil_type"],
        "target": "drought_risk_score",
    },
}


def _encode(df, categorical_cols, encoders=None, fit=False):
    df = df.copy()
    encoders = encoders or {}
    for col in categorical_cols:
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


def _train_one(key: str, df: pd.DataFrame) -> dict:
    cfg = RISK_CONFIGS[key]
    data = df[cfg["features"] + [cfg["target"]]].dropna().copy()
    data["risk_bucket"] = data[cfg["target"]].apply(_bucket_score)

    X_raw = data[cfg["features"]]
    X_encoded, encoders = _encode(X_raw, cfg["categorical"], fit=True)

    bucket_encoder = LabelEncoder()
    y = bucket_encoder.fit_transform(data["risk_bucket"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.08, random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    bundle = {
        "model": model, "encoders": encoders, "bucket_encoder": bucket_encoder,
        "feature_columns": cfg["features"], "accuracy": acc,
    }
    joblib.dump(bundle, MODEL_FILES[key])
    return {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}


def train_and_save_all(df: pd.DataFrame = None) -> dict:
    if df is None:
        from utils.helpers import get_agri_data
        df = get_agri_data()
    results = {}
    for key in RISK_CONFIGS:
        results[key] = _train_one(key, df)
    return results


def _load(key: str):
    if not os.path.exists(MODEL_FILES[key]):
        train_and_save_all()
    return joblib.load(MODEL_FILES[key])


def _predict(key: str, input_dict: dict) -> dict:
    cfg = RISK_CONFIGS[key]
    bundle = _load(key)
    model, encoders, bucket_encoder = bundle["model"], bundle["encoders"], bundle["bucket_encoder"]

    row = pd.DataFrame([input_dict])[cfg["features"]]
    encoded_row, _ = _encode(row, cfg["categorical"], encoders=encoders, fit=False)
    encoded_row = encoded_row[bundle["feature_columns"]]

    proba = model.predict_proba(encoded_row)[0]
    pred_idx = int(np.argmax(proba))
    risk_level = bucket_encoder.inverse_transform([pred_idx])[0]

    proba_by_level = {
        bucket_encoder.inverse_transform([i])[0]: round(float(p), 3)
        for i, p in enumerate(proba)
    }

    return {
        "risk_level": risk_level,
        "confidence": round(float(proba[pred_idx]), 3),
        "probability_breakdown": proba_by_level,
        "model_accuracy": round(bundle["accuracy"], 3),
    }


def predict_pest_risk(state, crop, humidity_pct, avg_temp_c, rainfall_mm) -> dict:
    return _predict("pest_risk", {
        "state": state, "crop": crop, "humidity_pct": humidity_pct,
        "avg_temp_c": avg_temp_c, "rainfall_mm": rainfall_mm,
    })


def predict_disease_risk(state, crop, humidity_pct, avg_temp_c, rainfall_mm, soil_ph) -> dict:
    return _predict("disease_risk", {
        "state": state, "crop": crop, "humidity_pct": humidity_pct,
        "avg_temp_c": avg_temp_c, "rainfall_mm": rainfall_mm, "soil_ph": soil_ph,
    })


def predict_flood_risk(state, soil_type, rainfall_mm, water_use_mm) -> dict:
    return _predict("flood_risk", {
        "state": state, "soil_type": soil_type,
        "rainfall_mm": rainfall_mm, "water_use_mm": water_use_mm,
    })


def predict_drought_risk(state, soil_type, rainfall_mm, avg_temp_c, humidity_pct) -> dict:
    return _predict("drought_risk", {
        "state": state, "soil_type": soil_type, "rainfall_mm": rainfall_mm,
        "avg_temp_c": avg_temp_c, "humidity_pct": humidity_pct,
    })


if __name__ == "__main__":
    results = train_and_save_all()
    print("Training results:", results)

    print("\nPest risk:", predict_pest_risk("Punjab", "Rice", 78, 28, 1100))
    print("Disease risk:", predict_disease_risk("Punjab", "Rice", 78, 28, 1100, 6.3))
    print("Flood risk:", predict_flood_risk("West Bengal", "Alluvial", 1800, 900))
    print("Drought risk:", predict_drought_risk("Rajasthan", "Arid", 350, 32, 30))
