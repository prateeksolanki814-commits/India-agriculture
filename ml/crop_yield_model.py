"""
ml/crop_yield_model.py
========================
Predicts crop yield (tonnes/hectare) from agronomic + environmental
inputs using an XGBoost regressor. Trained on the synthetic agriculture
dataset (state, crop, soil, rainfall, temperature, NPK, etc.).

Public API:
    train_and_save_model()  -> trains, evaluates, persists to disk
    load_model()             -> loads persisted model + encoders
    predict_yield(features)  -> dict with prediction + confidence
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILES, RANDOM_SEED

FEATURE_COLUMNS = [
    "state", "crop", "season", "soil_type", "rainfall_mm", "avg_temp_c",
    "humidity_pct", "nitrogen_n", "phosphorus_p", "potassium_k", "soil_ph",
    "water_use_mm",
]
CATEGORICAL_COLUMNS = ["state", "crop", "season", "soil_type"]
TARGET_COLUMN = "yield_t_per_ha"


def _encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    """Label-encodes categorical columns, reusing fitted encoders at inference time."""
    df = df.copy()
    encoders = encoders or {}
    for col in CATEGORICAL_COLUMNS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Map unseen categories to the most common trained class (-1 => 0) safely
            df[col] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    return df, encoders


def train_and_save_model(df: pd.DataFrame = None) -> dict:
    """Trains the XGBoost yield model and saves it + metrics to disk."""
    if df is None:
        from utils.helpers import get_agri_data
        df = get_agri_data()

    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    X_raw, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN]

    X_encoded, encoders = _encode_features(X_raw, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_SEED, objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    bundle = {
        "model": model, "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS, "mae": mae, "r2": r2,
    }
    joblib.dump(bundle, MODEL_FILES["crop_yield"])
    return {"mae": mae, "r2": r2, "n_train": len(X_train), "n_test": len(X_test)}


def load_model():
    if not os.path.exists(MODEL_FILES["crop_yield"]):
        train_and_save_model()
    return joblib.load(MODEL_FILES["crop_yield"])


def predict_yield(state, crop, season, soil_type, rainfall_mm, avg_temp_c,
                   humidity_pct, nitrogen_n, phosphorus_p, potassium_k,
                   soil_ph, water_use_mm) -> dict:
    """
    Predicts yield (t/ha) for a single set of inputs. Returns the point
    prediction plus a confidence score derived from the model's trained R²
    and the spread of individual tree predictions (a simple uncertainty proxy).
    """
    bundle = load_model()
    model, encoders = bundle["model"], bundle["encoders"]

    row = pd.DataFrame([{
        "state": state, "crop": crop, "season": season, "soil_type": soil_type,
        "rainfall_mm": rainfall_mm, "avg_temp_c": avg_temp_c,
        "humidity_pct": humidity_pct, "nitrogen_n": nitrogen_n,
        "phosphorus_p": phosphorus_p, "potassium_k": potassium_k,
        "soil_ph": soil_ph, "water_use_mm": water_use_mm,
    }])
    encoded_row, _ = _encode_features(row, encoders=encoders, fit=False)
    encoded_row = encoded_row[bundle["feature_columns"]]

    # Per-tree prediction spread as a lightweight uncertainty estimate.
    booster = model.get_booster()
    dmat = xgb.DMatrix(encoded_row, feature_names=bundle["feature_columns"])
    leaf_preds = booster.predict(dmat, pred_leaf=False, iteration_range=(0, model.n_estimators))
    point_pred = float(model.predict(encoded_row)[0])

    # Confidence: blend of trained R² and inverse coefficient-of-variation
    # across a small bootstrap of the trees' partial sums.
    per_tree_preds = []
    for i in range(1, model.n_estimators + 1, max(1, model.n_estimators // 20)):
        partial = model.predict(encoded_row, iteration_range=(0, i))
        per_tree_preds.append(partial[0])
    spread = np.std(per_tree_preds) / (abs(point_pred) + 1e-6)
    confidence = float(np.clip(bundle["r2"] * (1 - min(spread, 0.5)), 0.3, 0.97))

    return {
        "predicted_yield_t_per_ha": round(max(0, point_pred), 3),
        "confidence": round(confidence, 3),
        "model_r2": round(bundle["r2"], 3),
        "model_mae": round(bundle["mae"], 3),
    }


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Training metrics:", metrics)
    sample = predict_yield(
        state="Punjab", crop="Wheat", season="Rabi", soil_type="Alluvial",
        rainfall_mm=650, avg_temp_c=24, humidity_pct=55, nitrogen_n=90,
        phosphorus_p=40, potassium_k=45, soil_ph=6.8, water_use_mm=400,
    )
    print("Sample prediction:", sample)
