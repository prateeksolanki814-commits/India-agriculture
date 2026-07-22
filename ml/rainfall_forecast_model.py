"""
ml/rainfall_forecast_model.py
================================
Forecasts future daily rainfall AND temperature per state using Facebook
Prophet, which natively handles yearly seasonality (monsoon cycles) far
better than a generic regressor. One lightweight Prophet model is fit
per state per metric, on demand, and cached to disk so repeat calls in
the same session are instant.

Public API:
    forecast_rainfall(state, days_ahead=30) -> DataFrame with yhat/lower/upper
    forecast_temperature(state, days_ahead=30) -> DataFrame with yhat/lower/upper
    get_forecast_summary(state, days_ahead=30) -> dict of headline numbers
"""

import os
import sys
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR

os.makedirs(os.path.join(MODEL_DIR, "prophet_cache"), exist_ok=True)


def _fit_or_load_prophet(state: str, metric_col: str, df: pd.DataFrame):
    """
    Fits (or loads a cached) Prophet model for a given state + metric
    (e.g. 'rainfall_mm' or 'temperature_c'). Caching avoids re-fitting
    15 states x 2 metrics on every single page interaction.
    """
    from prophet import Prophet

    cache_path = os.path.join(MODEL_DIR, "prophet_cache", f"{state}_{metric_col}.pkl")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    state_df = df[df["state"] == state][["date", metric_col]].rename(
        columns={"date": "ds", metric_col: "y"}
    ).sort_values("ds")

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
        changepoint_prior_scale=0.05, interval_width=0.8,
    )
    model.fit(state_df)
    joblib.dump(model, cache_path)
    return model


def _forecast(state: str, metric_col: str, days_ahead: int) -> pd.DataFrame:
    from utils.helpers import get_weather_data
    df = get_weather_data()

    model = _fit_or_load_prophet(state, metric_col, df)
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_ahead).copy()
    result = result.rename(columns={
        "ds": "date", "yhat": "predicted", "yhat_lower": "lower_bound", "yhat_upper": "upper_bound",
    })
    result["predicted"] = result["predicted"].clip(lower=0)
    result["lower_bound"] = result["lower_bound"].clip(lower=0)
    return result.reset_index(drop=True)


def forecast_rainfall(state: str, days_ahead: int = 30) -> pd.DataFrame:
    return _forecast(state, "rainfall_mm", days_ahead)


def forecast_temperature(state: str, days_ahead: int = 30) -> pd.DataFrame:
    return _forecast(state, "temperature_c", days_ahead)


def get_forecast_summary(state: str, days_ahead: int = 30) -> dict:
    """Headline numbers for KPI cards: total expected rainfall, avg temp, trend."""
    rain_fc = forecast_rainfall(state, days_ahead)
    temp_fc = forecast_temperature(state, days_ahead)

    total_rain = float(rain_fc["predicted"].sum())
    avg_temp = float(temp_fc["predicted"].mean())

    # Simple trend: compare first-half vs second-half average.
    half = max(1, days_ahead // 2)
    rain_trend = "increasing" if rain_fc["predicted"].iloc[half:].mean() > rain_fc["predicted"].iloc[:half].mean() else "decreasing"
    temp_trend = "increasing" if temp_fc["predicted"].iloc[half:].mean() > temp_fc["predicted"].iloc[:half].mean() else "decreasing"

    return {
        "state": state, "days_ahead": days_ahead,
        "total_expected_rainfall_mm": round(total_rain, 1),
        "avg_expected_temp_c": round(avg_temp, 1),
        "rainfall_trend": rain_trend,
        "temperature_trend": temp_trend,
    }


if __name__ == "__main__":
    summary = get_forecast_summary("Punjab", days_ahead=14)
    print("Forecast summary:", summary)
