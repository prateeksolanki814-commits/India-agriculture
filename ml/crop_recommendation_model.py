"""
ml/crop_recommendation_model.py
==================================
Given soil nutrients (N, P, K, pH) and climate conditions (rainfall,
temperature, humidity), recommends the top-N most suitable crops using a
Random Forest classifier trained on the synthetic dataset.

Approach: for each historical (state, district, crop, year) row, the crop
actually grown there under those conditions is treated as a positive label
for that condition-cluster. A multi-class RandomForest then learns which
crop tends to suit which condition profile, and predict_proba gives us a
ranked list with confidence -- a lightweight but genuine ML approach to
"recommend the best crop for these conditions."
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILES, RANDOM_SEED

FEATURE_COLUMNS = [
    "nitrogen_n", "phosphorus_p", "potassium_k", "soil_ph",
    "rainfall_mm", "avg_temp_c", "humidity_pct",
]
TARGET_COLUMN = "crop"


def train_and_save_model(df: pd.DataFrame = None) -> dict:
    if df is None:
        from utils.helpers import get_agri_data
        df = get_agri_data()

    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    X = data[FEATURE_COLUMNS]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    bundle = {
        "model": model, "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS, "accuracy": acc, "f1_macro": f1,
    }
    joblib.dump(bundle, MODEL_FILES["crop_recommendation"])
    return {"accuracy": acc, "f1_macro": f1, "n_train": len(X_train), "n_test": len(X_test)}


def load_model():
    if not os.path.exists(MODEL_FILES["crop_recommendation"]):
        train_and_save_model()
    return joblib.load(MODEL_FILES["crop_recommendation"])


def recommend_crops(nitrogen_n, phosphorus_p, potassium_k, soil_ph,
                     rainfall_mm, avg_temp_c, humidity_pct, top_n=5) -> list:
    """
    Returns a ranked list of the top_n recommended crops:
    [{"crop": ..., "confidence": ...}, ...] sorted by descending confidence.
    """
    bundle = load_model()
    model, le = bundle["model"], bundle["label_encoder"]

    row = pd.DataFrame([{
        "nitrogen_n": nitrogen_n, "phosphorus_p": phosphorus_p,
        "potassium_k": potassium_k, "soil_ph": soil_ph,
        "rainfall_mm": rainfall_mm, "avg_temp_c": avg_temp_c,
        "humidity_pct": humidity_pct,
    }])[bundle["feature_columns"]]

    proba = model.predict_proba(row)[0]
    ranked_idx = np.argsort(proba)[::-1][:top_n]

    results = []
    for idx in ranked_idx:
        crop_name = le.inverse_transform([idx])[0]
        results.append({
            "crop": crop_name,
            "confidence": round(float(proba[idx]), 3),
        })
    return results


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Training metrics:", metrics)
    recs = recommend_crops(
        nitrogen_n=90, phosphorus_p=42, potassium_k=48, soil_ph=6.7,
        rainfall_mm=900, avg_temp_c=26, humidity_pct=65,
    )
    print("Recommendations:", recs)
