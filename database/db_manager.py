"""
database/db_manager.py
========================
Owns the SQLAlchemy engine/session and provides high-level helper
functions the rest of the app calls (get_engine, seed_database_if_empty,
load_agriculture_df, load_weather_df, get_filtered_data ...).

Design choice: for a Streamlit analytics app, round-tripping every chart
through the DB on every rerun is unnecessary overhead. So db_manager loads
data ONCE into pandas DataFrames (cached via streamlit's cache when called
from within Streamlit) and the DB itself mainly exists to (a) demonstrate
a real persistence layer and (b) allow SQL-style filtered queries when
needed. All pages primarily use the pandas helpers below.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL, DATA_DIR
from database.models import Base, AgricultureRecord, WeatherDaily

_engine = None
_SessionLocal = None


def get_engine():
    """Lazily creates (and caches at module level) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, echo=False, future=True)
        Base.metadata.create_all(_engine)
    return _engine


def get_session():
    """Returns a new SQLAlchemy session bound to the shared engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), future=True)
    return _SessionLocal()


def _table_is_empty(engine, model) -> bool:
    with engine.connect() as conn:
        result = conn.execute(select(model.id).limit(1)).first()
        return result is None


def seed_database_if_empty():
    """
    Ensures the database has data. If empty, generates the synthetic
    dataset (or loads existing CSVs if already generated) and bulk-inserts
    it. Safe to call on every app startup -- it's a no-op once seeded.
    """
    engine = get_engine()

    agri_csv = os.path.join(DATA_DIR, "agriculture_dataset.csv")
    weather_csv = os.path.join(DATA_DIR, "weather_timeseries.csv")

    if not (os.path.exists(agri_csv) and os.path.exists(weather_csv)):
        from data.data_generator import save_all_datasets
        save_all_datasets()

    if _table_is_empty(engine, AgricultureRecord):
        df = pd.read_csv(agri_csv)
        df.to_sql("agriculture_records", engine, if_exists="append", index=False)

    if _table_is_empty(engine, WeatherDaily):
        wdf = pd.read_csv(weather_csv, parse_dates=["date"])
        wdf["date"] = wdf["date"].dt.date  # store as pure date (no time component)
        wdf.to_sql("weather_daily", engine, if_exists="append", index=False)


def load_agriculture_df() -> pd.DataFrame:
    """Loads the full agriculture fact table as a pandas DataFrame."""
    seed_database_if_empty()
    engine = get_engine()
    return pd.read_sql_table("agriculture_records", engine)


def load_weather_df() -> pd.DataFrame:
    """Loads the full daily weather table as a pandas DataFrame."""
    seed_database_if_empty()
    engine = get_engine()
    df = pd.read_sql_table("weather_daily", engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_filtered_data(df: pd.DataFrame, state=None, district=None, crop=None,
                       year=None, season=None) -> pd.DataFrame:
    """
    Generic in-memory filter helper used across dashboard/analytics pages
    so every page doesn't reimplement the same boolean masking logic.
    Any argument left as None (or 'All') is ignored.
    """
    out = df.copy()
    filters = {
        "state": state, "district": district, "crop": crop,
        "year": year, "season": season,
    }
    for col, value in filters.items():
        if value is not None and value != "All":
            out = out[out[col] == value]
    return out


if __name__ == "__main__":
    seed_database_if_empty()
    a = load_agriculture_df()
    w = load_weather_df()
    print(f"Agriculture records in DB: {len(a):,}")
    print(f"Weather records in DB:     {len(w):,}")
