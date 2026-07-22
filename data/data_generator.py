"""
data/data_generator.py
========================
Generates a realistic, internally-consistent synthetic dataset that stands
in for real government agriculture data (e.g. ICAR / Agricultural Census /
IMD sources). Real live feeds are not reachable from this environment, so
values are built from plausible ranges and relationships (e.g. rice yields
higher in high-rainfall alluvial states, cotton favoring black-soil states)
so downstream charts and ML models behave sensibly.

Running this file directly (python data_generator.py) regenerates all CSVs
inside the data/ folder and is also imported by database/db_manager.py to
seed the SQLite database on first run.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INDIAN_STATES, STATE_DISTRICTS, CROPS, SEASONS, SOIL_TYPES, YEARS,
    DATA_DIR, RANDOM_SEED,
)

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Reference tables that encode realistic regional biases.
# These are approximate, illustrative agronomic patterns -- not official
# statistics -- used only to make synthetic data internally consistent.
# ---------------------------------------------------------------------------
STATE_RAINFALL_BASE = {
    # average annual rainfall in mm (rough regional bands)
    "Punjab": 650, "Haryana": 600, "Uttar Pradesh": 900, "Madhya Pradesh": 1100,
    "Maharashtra": 1200, "Rajasthan": 450, "Gujarat": 800, "Bihar": 1200,
    "West Bengal": 1600, "Andhra Pradesh": 950, "Telangana": 900,
    "Karnataka": 1100, "Tamil Nadu": 950, "Odisha": 1450, "Chhattisgarh": 1300,
}

STATE_SOIL_BIAS = {
    "Punjab": "Alluvial", "Haryana": "Alluvial", "Uttar Pradesh": "Alluvial",
    "Madhya Pradesh": "Black", "Maharashtra": "Black", "Rajasthan": "Arid",
    "Gujarat": "Black", "Bihar": "Alluvial", "West Bengal": "Alluvial",
    "Andhra Pradesh": "Red", "Telangana": "Red", "Karnataka": "Red",
    "Tamil Nadu": "Red", "Odisha": "Laterite", "Chhattisgarh": "Red",
}

# Base yield (tonnes/hectare) per crop -- approximate national averages.
CROP_BASE_YIELD = {
    "Rice": 2.7, "Wheat": 3.3, "Maize": 2.9, "Cotton": 0.5, "Sugarcane": 70.0,
    "Soybean": 1.1, "Groundnut": 1.4, "Bajra": 1.2, "Jowar": 1.0,
    "Mustard": 1.3, "Gram": 1.0, "Tur (Arhar)": 0.8, "Potato": 22.0,
    "Onion": 17.0, "Tomato": 24.0,
}

CROP_SEASON = {
    "Rice": "Kharif", "Wheat": "Rabi", "Maize": "Kharif", "Cotton": "Kharif",
    "Sugarcane": "Zaid", "Soybean": "Kharif", "Groundnut": "Kharif",
    "Bajra": "Kharif", "Jowar": "Kharif", "Mustard": "Rabi", "Gram": "Rabi",
    "Tur (Arhar)": "Kharif", "Potato": "Rabi", "Onion": "Rabi", "Tomato": "Zaid",
}

# Base market price (INR per quintal) per crop -- illustrative only.
CROP_BASE_PRICE = {
    "Rice": 2100, "Wheat": 2200, "Maize": 1900, "Cotton": 6500,
    "Sugarcane": 340, "Soybean": 4300, "Groundnut": 5500, "Bajra": 2350,
    "Jowar": 2900, "Mustard": 5400, "Gram": 5200, "Tur (Arhar)": 6900,
    "Potato": 1200, "Onion": 1500, "Tomato": 1400,
}

# Approximate optimal growing conditions per crop (N, P, K in kg/ha, pH,
# temp in C, humidity %, rainfall mm) -- loosely modeled on agronomic
# reference ranges (similar in spirit to the well-known Kaggle crop
# recommendation dataset). These centers are what let a recommendation
# model learn genuine crop <-> condition relationships instead of noise.
CROP_OPTIMAL_CONDITIONS = {
    "Rice":        {"n": 100, "p": 45, "k": 45, "ph": 6.3, "temp": 26, "humidity": 82, "rainfall": 1200},
    "Wheat":       {"n": 100, "p": 50, "k": 40, "ph": 6.8, "temp": 20, "humidity": 55, "rainfall": 650},
    "Maize":       {"n": 90,  "p": 45, "k": 35, "ph": 6.5, "temp": 24, "humidity": 60, "rainfall": 750},
    "Cotton":      {"n": 110, "p": 40, "k": 55, "ph": 7.2, "temp": 28, "humidity": 55, "rainfall": 800},
    "Sugarcane":   {"n": 140, "p": 55, "k": 60, "ph": 6.8, "temp": 27, "humidity": 70, "rainfall": 1400},
    "Soybean":     {"n": 40,  "p": 55, "k": 45, "ph": 6.5, "temp": 25, "humidity": 65, "rainfall": 900},
    "Groundnut":   {"n": 30,  "p": 55, "k": 40, "ph": 6.4, "temp": 27, "humidity": 60, "rainfall": 700},
    "Bajra":       {"n": 55,  "p": 30, "k": 30, "ph": 7.0, "temp": 30, "humidity": 40, "rainfall": 450},
    "Jowar":       {"n": 60,  "p": 30, "k": 30, "ph": 6.8, "temp": 28, "humidity": 45, "rainfall": 550},
    "Mustard":     {"n": 70,  "p": 35, "k": 30, "ph": 6.9, "temp": 18, "humidity": 45, "rainfall": 500},
    "Gram":        {"n": 25,  "p": 55, "k": 35, "ph": 7.1, "temp": 21, "humidity": 45, "rainfall": 500},
    "Tur (Arhar)": {"n": 30,  "p": 50, "k": 35, "ph": 6.6, "temp": 27, "humidity": 55, "rainfall": 800},
    "Potato":      {"n": 120, "p": 60, "k": 100, "ph": 5.8, "temp": 19, "humidity": 75, "rainfall": 600},
    "Onion":       {"n": 90,  "p": 45, "k": 60, "ph": 6.4, "temp": 23, "humidity": 60, "rainfall": 650},
    "Tomato":      {"n": 100, "p": 55, "k": 70, "ph": 6.3, "temp": 24, "humidity": 68, "rainfall": 700},
}

# Which crops each state realistically grows (keeps data plausible; every
# state still gets a reasonable subset rather than all 15 crops).
STATE_CROP_AFFINITY = {
    "Punjab": ["Rice", "Wheat", "Cotton", "Maize", "Mustard"],
    "Haryana": ["Wheat", "Rice", "Cotton", "Mustard", "Bajra"],
    "Uttar Pradesh": ["Wheat", "Rice", "Sugarcane", "Potato", "Maize"],
    "Madhya Pradesh": ["Soybean", "Wheat", "Gram", "Cotton", "Maize"],
    "Maharashtra": ["Cotton", "Soybean", "Sugarcane", "Jowar", "Tur (Arhar)"],
    "Rajasthan": ["Bajra", "Mustard", "Wheat", "Gram", "Groundnut"],
    "Gujarat": ["Cotton", "Groundnut", "Wheat", "Bajra", "Sugarcane"],
    "Bihar": ["Rice", "Wheat", "Maize", "Potato", "Onion"],
    "West Bengal": ["Rice", "Potato", "Jowar", "Maize", "Onion"],
    "Andhra Pradesh": ["Rice", "Cotton", "Groundnut", "Tomato", "Maize"],
    "Telangana": ["Rice", "Cotton", "Maize", "Tur (Arhar)", "Groundnut"],
    "Karnataka": ["Rice", "Maize", "Cotton", "Tur (Arhar)", "Tomato"],
    "Tamil Nadu": ["Rice", "Sugarcane", "Groundnut", "Tomato", "Maize"],
    "Odisha": ["Rice", "Groundnut", "Maize", "Tur (Arhar)", "Onion"],
    "Chhattisgarh": ["Rice", "Maize", "Gram", "Tur (Arhar)", "Soybean"],
}


def _noise(scale=0.08, size=1):
    """Small multiplicative noise helper to avoid perfectly smooth trends."""
    return rng.normal(1.0, scale, size=size)


def generate_agriculture_dataset() -> pd.DataFrame:
    """
    Builds the core fact table: one row per (year, state, district, crop).
    Encodes rainfall, temperature, soil, area, production, yield, price,
    water usage, and derived profit -- all internally consistent.
    """
    rows = []
    for year in YEARS:
        # Mild national upward yield trend over the decade (tech adoption).
        year_index = year - YEARS[0]
        trend_factor = 1 + 0.015 * year_index

        for state in INDIAN_STATES:
            base_rain = STATE_RAINFALL_BASE[state]
            soil = STATE_SOIL_BIAS[state]
            districts = STATE_DISTRICTS[state]

            for district in districts:
                # District-level rainfall/temperature variation around state base.
                rainfall_mm = max(150, base_rain * _noise(0.15)[0])
                avg_temp_c = rng.normal(26.5, 3.2)
                humidity_pct = np.clip(rng.normal(62, 12), 20, 95)

                crops_here = STATE_CROP_AFFINITY[state]
                for crop in crops_here:
                    base_yield = CROP_BASE_YIELD[crop]
                    season = CROP_SEASON[crop]
                    opt = CROP_OPTIMAL_CONDITIONS[crop]

                    # Climate/soil values are sampled AROUND each crop's
                    # agronomic optimum (with noise), blended lightly with
                    # the district's own rainfall/temp reality. This keeps
                    # regional flavor while giving models genuine signal
                    # linking soil/climate conditions to crop suitability.
                    crop_rainfall = 0.5 * rainfall_mm + 0.5 * rng.normal(opt["rainfall"], opt["rainfall"] * 0.12)
                    crop_rainfall = max(150, crop_rainfall)
                    crop_temp = 0.4 * avg_temp_c + 0.6 * rng.normal(opt["temp"], 2.2)
                    crop_humidity = 0.4 * humidity_pct + 0.6 * np.clip(rng.normal(opt["humidity"], 8), 15, 95)

                    n_val = np.clip(rng.normal(opt["n"], 12), 10, 160)
                    p_val = np.clip(rng.normal(opt["p"], 8), 10, 90)
                    k_val = np.clip(rng.normal(opt["k"], 10), 10, 110)
                    ph_val = np.clip(rng.normal(opt["ph"], 0.35), 4.5, 8.5)

                    # Yield responds to how close conditions are to the
                    # crop's optimum (rainfall/temp/nutrient adequacy),
                    # not just generic state rainfall.
                    rain_factor = np.clip(1 - abs(crop_rainfall - opt["rainfall"]) / (opt["rainfall"] * 2), 0.5, 1.15)
                    temp_penalty = np.clip(1 - abs(crop_temp - opt["temp"]) * 0.015, 0.6, 1.05)
                    nutrient_factor = np.clip(
                        1 - (abs(n_val - opt["n"]) / opt["n"] + abs(p_val - opt["p"]) / opt["p"]
                             + abs(k_val - opt["k"]) / opt["k"]) * 0.08,
                        0.6, 1.1
                    )
                    yield_t_per_ha = (
                        base_yield * trend_factor * rain_factor
                        * temp_penalty * nutrient_factor * _noise(0.08)[0]
                    )
                    yield_t_per_ha = max(0.05, yield_t_per_ha)

                    area_ha = max(500, rng.normal(15000, 6000))
                    production_tonnes = yield_t_per_ha * area_ha

                    price_per_quintal = max(
                        200, CROP_BASE_PRICE[crop] * (1 + 0.04 * year_index) * _noise(0.12)[0]
                    )
                    revenue_inr = production_tonnes * 10 * price_per_quintal  # tonnes->quintals
                    cost_per_ha = rng.normal(35000, 9000)
                    total_cost_inr = cost_per_ha * area_ha
                    profit_inr = revenue_inr - total_cost_inr

                    water_use_mm = max(200, crop_rainfall * rng.uniform(0.4, 0.9))
                    rainfall_mm_out = crop_rainfall
                    avg_temp_c_out = crop_temp
                    humidity_pct_out = crop_humidity

                    pest_risk_score = np.clip(
                        rng.normal(35 + (humidity_pct - 60) * 0.4, 15), 0, 100
                    )
                    disease_risk_score = np.clip(
                        rng.normal(30 + (humidity_pct - 60) * 0.3, 14), 0, 100
                    )
                    flood_risk_score = np.clip(
                        rng.normal((rainfall_mm - 800) / 12, 15), 0, 100
                    )
                    drought_risk_score = np.clip(
                        rng.normal((800 - rainfall_mm) / 8, 15), 0, 100
                    )
                    sustainability_score = np.clip(
                        rng.normal(65 - (n_val - 85) * 0.1 - drought_risk_score * 0.1, 10),
                        0, 100
                    )
                    carbon_footprint_kg_co2e = max(
                        50, (n_val * 4.5 + area_ha * 0.02) * _noise(0.1)[0]
                    )

                    rows.append({
                        "year": year, "state": state, "district": district,
                        "crop": crop, "season": season, "soil_type": soil,
                        "area_ha": round(area_ha, 1),
                        "production_tonnes": round(production_tonnes, 1),
                        "yield_t_per_ha": round(yield_t_per_ha, 3),
                        "rainfall_mm": round(rainfall_mm_out, 1),
                        "avg_temp_c": round(avg_temp_c_out, 2),
                        "humidity_pct": round(humidity_pct_out, 1),
                        "price_per_quintal_inr": round(price_per_quintal, 1),
                        "revenue_inr": round(revenue_inr, 0),
                        "total_cost_inr": round(total_cost_inr, 0),
                        "profit_inr": round(profit_inr, 0),
                        "water_use_mm": round(water_use_mm, 1),
                        "nitrogen_n": round(n_val, 1),
                        "phosphorus_p": round(p_val, 1),
                        "potassium_k": round(k_val, 1),
                        "soil_ph": round(ph_val, 2),
                        "pest_risk_score": round(pest_risk_score, 1),
                        "disease_risk_score": round(disease_risk_score, 1),
                        "flood_risk_score": round(flood_risk_score, 1),
                        "drought_risk_score": round(drought_risk_score, 1),
                        "sustainability_score": round(sustainability_score, 1),
                        "carbon_footprint_kg_co2e": round(carbon_footprint_kg_co2e, 1),
                    })

    df = pd.DataFrame(rows)
    return df


def generate_weather_timeseries() -> pd.DataFrame:
    """
    Builds a daily weather time series per state for the most recent year,
    used by the Weather Intelligence page and rainfall/temperature
    forecasting models (Prophet expects a long daily 'ds'/'y' format).
    """
    rows = []
    last_year = YEARS[-1]
    date_range = pd.date_range(f"{last_year}-01-01", f"{last_year}-12-31", freq="D")

    for state in INDIAN_STATES:
        base_rain = STATE_RAINFALL_BASE[state]
        daily_rain_mean = base_rain / 365

        # Simple seasonal monsoon bump (Jun-Sep) using a smooth curve.
        for date in date_range:
            doy = date.dayofyear
            monsoon_boost = np.exp(-((doy - 210) ** 2) / (2 * 55 ** 2)) * 6
            rainfall_today = max(0, rng.exponential(daily_rain_mean * 0.6 + monsoon_boost))

            seasonal_temp = 27 + 6 * np.sin((doy - 80) / 365 * 2 * np.pi)
            temp_today = seasonal_temp + rng.normal(0, 2.0)
            humidity_today = np.clip(50 + monsoon_boost * 4 + rng.normal(0, 6), 20, 98)
            wind_kmph = max(2, rng.normal(12, 4))

            rows.append({
                "date": date, "state": state,
                "rainfall_mm": round(rainfall_today, 2),
                "temperature_c": round(temp_today, 2),
                "humidity_pct": round(humidity_today, 1),
                "wind_speed_kmph": round(wind_kmph, 1),
            })

    return pd.DataFrame(rows)


def save_all_datasets():
    """Generates and writes both datasets to CSV inside data/."""
    agri_df = generate_agriculture_dataset()
    weather_df = generate_weather_timeseries()

    agri_path = os.path.join(DATA_DIR, "agriculture_dataset.csv")
    weather_path = os.path.join(DATA_DIR, "weather_timeseries.csv")

    agri_df.to_csv(agri_path, index=False)
    weather_df.to_csv(weather_path, index=False)

    print(f"Saved {len(agri_df):,} agriculture rows -> {agri_path}")
    print(f"Saved {len(weather_df):,} weather rows   -> {weather_path}")
    return agri_df, weather_df


if __name__ == "__main__":
    save_all_datasets()
