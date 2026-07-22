"""
ml/price_prediction_model.py
==============================
Predicts market price (INR/quintal) for a crop given year, state, season,
and production volume, using a LightGBM regressor. Also derives a simple
profit projection (expected revenue - estimated cost) from the predicted
price, which the Recommendations page uses for "expected profit" advice.

Public API:
    train_and_save_model()
    load_model()
    predict_price(...)
    predict_profit(...)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILES, RANDOM_SEED

FEATURE_COLUMNS = ["year", "state", "crop", "season", "production_tonnes", "area_ha"]
CATEGORICAL_COLUMNS = ["state", "crop", "season"]
TARGET_COLUMN = "price_per_quintal_inr"


def _encode_features(df, encoders=None, fit=False):
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

    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    X_raw, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN]
    X_encoded, encoders = _encode_features(X_raw, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_SEED, verbosity=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    bundle = {
        "model": model, "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS, "mae": mae, "r2": r2,
    }
    joblib.dump(bundle, MODEL_FILES["price"])
    return {"mae": mae, "r2": r2, "n_train": len(X_train), "n_test": len(X_test)}


def load_model():
    if not os.path.exists(MODEL_FILES["price"]):
        train_and_save_model()
    return joblib.load(MODEL_FILES["price"])


def predict_price(year, state, crop, season, production_tonnes, area_ha) -> dict:
    bundle = load_model()
    model, encoders = bundle["model"], bundle["encoders"]

    row = pd.DataFrame([{
        "year": year, "state": state, "crop": crop, "season": season,
        "production_tonnes": production_tonnes, "area_ha": area_ha,
    }])
    encoded_row, _ = _encode_features(row, encoders=encoders, fit=False)
    encoded_row = encoded_row[bundle["feature_columns"]]

    point_pred = float(model.predict(encoded_row)[0])
    confidence = float(np.clip(bundle["r2"], 0.3, 0.95))

    return {
        "predicted_price_per_quintal_inr": round(max(50, point_pred), 1),
        "confidence": round(confidence, 3),
        "model_r2": round(bundle["r2"], 3),
        "model_mae": round(bundle["mae"], 1),
    }


def predict_profit(year, state, crop, season, production_tonnes, area_ha,
                    estimated_cost_per_ha=35000) -> dict:
    """
    Combines the predicted price with production volume to project revenue,
    cost, and net profit -- feeding the 'Expected Profit' recommendation.
    """
    price_result = predict_price(year, state, crop, season, production_tonnes, area_ha)
    price = price_result["predicted_price_per_quintal_inr"]

    revenue_inr = production_tonnes * 10 * price  # tonnes -> quintals
    total_cost_inr = estimated_cost_per_ha * area_ha
    profit_inr = revenue_inr - total_cost_inr
    profit_margin_pct = (profit_inr / revenue_inr * 100) if revenue_inr > 0 else 0

    return {
        "predicted_price_per_quintal_inr": price,
        "projected_revenue_inr": round(revenue_inr, 0),
        "estimated_cost_inr": round(total_cost_inr, 0),
        "projected_profit_inr": round(profit_inr, 0),
        "profit_margin_pct": round(profit_margin_pct, 1),
        "confidence": price_result["confidence"],
    }


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Training metrics:", metrics)
    price = predict_price(2025, "Punjab", "Wheat", "Rabi", 45000, 15000)
    print("Price prediction:", price)
    profit = predict_profit(2025, "Punjab", "Wheat", "Rabi", 45000, 15000)
    print("Profit projection:", profit)
