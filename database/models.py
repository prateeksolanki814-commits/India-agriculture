"""
database/models.py
====================
SQLAlchemy ORM model definitions. Two tables:

  1. AgricultureRecord  - the core fact table (state/district/crop/year
     level metrics: area, production, yield, weather, soil, risk scores).
  2. WeatherDaily        - daily weather time series per state, used for
     forecasting models and the Weather Intelligence page.

Kept intentionally simple (flat tables, no joins needed) since this is an
analytics-read-heavy app rather than a transactional system.
"""

from sqlalchemy import Column, Integer, Float, String, Date
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AgricultureRecord(Base):
    """One row = one (year, state, district, crop) observation."""
    __tablename__ = "agriculture_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, index=True, nullable=False)
    state = Column(String(64), index=True, nullable=False)
    district = Column(String(64), index=True, nullable=False)
    crop = Column(String(64), index=True, nullable=False)
    season = Column(String(16))
    soil_type = Column(String(32))

    area_ha = Column(Float)
    production_tonnes = Column(Float)
    yield_t_per_ha = Column(Float)

    rainfall_mm = Column(Float)
    avg_temp_c = Column(Float)
    humidity_pct = Column(Float)

    price_per_quintal_inr = Column(Float)
    revenue_inr = Column(Float)
    total_cost_inr = Column(Float)
    profit_inr = Column(Float)

    water_use_mm = Column(Float)
    nitrogen_n = Column(Float)
    phosphorus_p = Column(Float)
    potassium_k = Column(Float)
    soil_ph = Column(Float)

    pest_risk_score = Column(Float)
    disease_risk_score = Column(Float)
    flood_risk_score = Column(Float)
    drought_risk_score = Column(Float)
    sustainability_score = Column(Float)
    carbon_footprint_kg_co2e = Column(Float)


class WeatherDaily(Base):
    """One row = one (date, state) daily weather observation."""
    __tablename__ = "weather_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, nullable=False)
    state = Column(String(64), index=True, nullable=False)

    rainfall_mm = Column(Float)
    temperature_c = Column(Float)
    humidity_pct = Column(Float)
    wind_speed_kmph = Column(Float)
