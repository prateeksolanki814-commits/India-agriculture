"""
ml/fertilizer_model.py
========================
Recommends a fertilizer type and dosage adjustment for a given crop and
soil-test reading. Approach:

  1. A RandomForest CLASSIFIER predicts the most suitable fertilizer
     category (Urea, DAP, MOP, NPK Complex, Organic Compost) based on the
     crop's actual NPK deficit relative to its known optimum.
  2. A simple deterministic rule-engine (informed by the same crop optima
     used in the recommendation model) computes the precise kg/ha
     adjustment for N, P, and K -- this is the actionable number a farmer
     needs, and rule-based logic is more trustworthy/explainable here than
     a black-box regression for exact dosage.

This hybrid (ML for "what kind" + explainable rules for "how much") mirrors
how real agri-advisory tools are built.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILES, RANDOM_SEED
from data.data_generator import CROP_OPTIMAL_CONDITIONS

FEATURE_COLUMNS = ["crop", "nitrogen_n", "phosphorus_p", "potassium_k", "soil_ph"]


def _label_fertilizer(row) -> str:
    """
    Deterministic labeling rule used to synthesize a training target:
    whichever nutrient has the largest deficit vs. the crop's optimum
    drives the fertilizer choice. This mirrors real agronomic advice
    (e.g., big N deficit -> Urea; big P deficit -> DAP; big K deficit ->
    MOP; balanced deficit -> NPK Complex; near-optimal -> Organic Compost
    for maintenance).
    """
    opt = CROP_OPTIMAL_CONDITIONS[row["crop"]]
    n_gap = opt["n"] - row["nitrogen_n"]
    p_gap = opt["p"] - row["phosphorus_p"]
    k_gap = opt["k"] - row["potassium_k"]
    gaps = {"N": n_gap, "P": p_gap, "K": k_gap}
    max_gap_nutrient = max(gaps, key=gaps.get)
    max_gap_value = gaps[max_gap_nutrient]

    if max_gap_value < 8:
        return "Organic Compost"
    if max_gap_nutrient == "N":
        return "Urea"
    if max_gap_nutrient == "P":
        return "DAP"
    if max_gap_nutrient == "K":
        return "MOP (Potash)"
    return "NPK Complex"


def train_and_save_model(df: pd.DataFrame = None) -> dict:
    if df is None:
        from utils.helpers import get_agri_data
        df = get_agri_data()

    data = df[["crop", "nitrogen_n", "phosphorus_p", "potassium_k", "soil_ph"]].dropna().copy()
    data["fertilizer_label"] = data.apply(_label_fertilizer, axis=1)

    crop_encoder = LabelEncoder()
    data["crop_encoded"] = crop_encoder.fit_transform(data["crop"])

    X = data[["crop_encoded", "nitrogen_n", "phosphorus_p", "potassium_k", "soil_ph"]]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["fertilizer_label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_SEED,
        n_jobs=-1, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    bundle = {
        "model": model, "crop_encoder": crop_encoder,
        "label_encoder": label_encoder, "accuracy": acc,
    }
    joblib.dump(bundle, MODEL_FILES["fertilizer"])
    return {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}


def load_model():
    if not os.path.exists(MODEL_FILES["fertilizer"]):
        train_and_save_model()
    return joblib.load(MODEL_FILES["fertilizer"])


def recommend_fertilizer(crop, nitrogen_n, phosphorus_p, potassium_k, soil_ph) -> dict:
    """
    Returns the recommended fertilizer type (ML classification) plus
    explicit kg/ha dosage adjustments for N, P, K (rule-based, explainable).
    """
    bundle = load_model()
    model, crop_encoder, label_encoder = (
        bundle["model"], bundle["crop_encoder"], bundle["label_encoder"]
    )

    if crop not in crop_encoder.classes_:
        crop = crop_encoder.classes_[0]  # graceful fallback
    crop_encoded = crop_encoder.transform([crop])[0]

    X = pd.DataFrame([{
        "crop_encoded": crop_encoded, "nitrogen_n": nitrogen_n,
        "phosphorus_p": phosphorus_p, "potassium_k": potassium_k, "soil_ph": soil_ph,
    }])
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    fertilizer_type = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    opt = CROP_OPTIMAL_CONDITIONS.get(crop, {"n": 80, "p": 40, "k": 40, "ph": 6.5})
    n_adjustment = round(max(0, opt["n"] - nitrogen_n), 1)
    p_adjustment = round(max(0, opt["p"] - phosphorus_p), 1)
    k_adjustment = round(max(0, opt["k"] - potassium_k), 1)
    ph_note = (
        "Soil is acidic — consider agricultural lime" if soil_ph < 6.0 else
        "Soil is alkaline — consider gypsum or organic matter" if soil_ph > 7.5 else
        "Soil pH is in the acceptable range"
    )

    return {
        "recommended_fertilizer": fertilizer_type,
        "confidence": round(confidence, 3),
        "nitrogen_deficit_kg_ha": n_adjustment,
        "phosphorus_deficit_kg_ha": p_adjustment,
        "potassium_deficit_kg_ha": k_adjustment,
        "soil_ph_note": ph_note,
    }


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Training metrics:", metrics)
    rec = recommend_fertilizer(
        crop="Wheat", nitrogen_n=60, phosphorus_p=30, potassium_k=25, soil_ph=6.9,
    )
    print("Recommendation:", rec)
