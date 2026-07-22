"""
config.py
==========
Central configuration file for AGRI VISION AI.

Holds every constant, path, and static reference list (states, crops,
seasons, color theme, model file locations) that other modules import from.
Keeping these in one place means changing a color or adding a new state
only has to happen here.
"""

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "database")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
REPORTS_DIR = os.path.join(BASE_DIR, "reports_output")

for _dir in (DATA_DIR, MODEL_DIR, REPORTS_DIR):
    os.makedirs(_dir, exist_ok=True)

DATABASE_PATH = os.path.join(DB_DIR, "agrivision.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ---------------------------------------------------------------------------
# APP METADATA
# ---------------------------------------------------------------------------
APP_NAME = "AGRI VISION AI"
APP_TAGLINE = "AI-Powered Smart Agriculture Analytics & Prediction Platform for India"
APP_ICON = "🌾"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# INDIA REFERENCE DATA
# ---------------------------------------------------------------------------
# 15 major agricultural states used to keep the synthetic dataset realistic
# in scale while remaining fast to generate and query.
INDIAN_STATES = [
    "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Maharashtra",
    "Rajasthan", "Gujarat", "Bihar", "West Bengal", "Andhra Pradesh",
    "Telangana", "Karnataka", "Tamil Nadu", "Odisha", "Chhattisgarh",
]

# A handful of representative districts per state (kept short deliberately —
# extend this dict freely without touching any other file).
STATE_DISTRICTS = {
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Bathinda"],
    "Haryana": ["Karnal", "Hisar", "Rohtak", "Panipat"],
    "Uttar Pradesh": ["Lucknow", "Meerut", "Kanpur", "Varanasi"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Rajasthan": ["Jaipur", "Kota", "Jodhpur", "Udaipur"],
    "Gujarat": ["Ahmedabad", "Rajkot", "Surat", "Vadodara"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur"],
    "West Bengal": ["Bardhaman", "Murshidabad", "Nadia", "Hooghly"],
    "Andhra Pradesh": ["Guntur", "Krishna", "Kurnool", "Nellore"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
    "Karnataka": ["Bengaluru Rural", "Mysuru", "Belagavi", "Hubballi"],
    "Tamil Nadu": ["Coimbatore", "Thanjavur", "Madurai", "Salem"],
    "Odisha": ["Cuttack", "Puri", "Sambalpur", "Balasore"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bastar"],
}

# Approximate lat/lon centroids for state-level map plotting.
STATE_COORDS = {
    "Punjab": (31.1471, 75.3412), "Haryana": (29.0588, 76.0856),
    "Uttar Pradesh": (26.8467, 80.9462), "Madhya Pradesh": (23.4733, 77.9470),
    "Maharashtra": (19.7515, 75.7139), "Rajasthan": (27.0238, 74.2179),
    "Gujarat": (22.2587, 71.1924), "Bihar": (25.0961, 85.3131),
    "West Bengal": (22.9868, 87.8550), "Andhra Pradesh": (15.9129, 79.7400),
    "Telangana": (18.1124, 79.0193), "Karnataka": (15.3173, 75.7139),
    "Tamil Nadu": (11.1271, 78.6569), "Odisha": (20.9517, 85.0985),
    "Chhattisgarh": (21.2787, 81.8661),
}

CROPS = [
    "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean",
    "Groundnut", "Bajra", "Jowar", "Mustard", "Gram", "Tur (Arhar)",
    "Potato", "Onion", "Tomato",
]

SEASONS = ["Kharif", "Rabi", "Zaid"]

SOIL_TYPES = ["Alluvial", "Black", "Red", "Laterite", "Arid", "Loamy"]

YEARS = list(range(2015, 2025))  # 10 years of historical data

# ---------------------------------------------------------------------------
# UI THEME (used by utils/styling.py and Plotly charts for a consistent look)
# ---------------------------------------------------------------------------
THEME = {
    "primary": "#2E7D32",       # deep agricultural green
    "primary_light": "#66BB6A",
    "secondary": "#F9A825",     # harvest gold
    "accent": "#1565C0",        # sky blue (water/weather)
    "danger": "#C62828",
    "warning": "#EF6C00",
    "success": "#2E7D32",
    "background": "#F4F7F3",
    "card_bg": "#FFFFFF",
    "text_dark": "#1B2E1F",
    "text_muted": "#5C6B5D",
}

PLOTLY_TEMPLATE_COLORS = [
    THEME["primary"], THEME["secondary"], THEME["accent"],
    THEME["primary_light"], THEME["warning"], THEME["danger"],
]

# ---------------------------------------------------------------------------
# MODEL FILE NAMES (ml/*.py save/load trained models using these constants)
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "crop_yield": os.path.join(MODEL_DIR, "crop_yield_model.pkl"),
    "crop_recommendation": os.path.join(MODEL_DIR, "crop_recommendation_model.pkl"),
    "fertilizer": os.path.join(MODEL_DIR, "fertilizer_model.pkl"),
    "price": os.path.join(MODEL_DIR, "price_model.pkl"),
    "rainfall": os.path.join(MODEL_DIR, "rainfall_model.pkl"),
    "pest_risk": os.path.join(MODEL_DIR, "pest_risk_model.pkl"),
    "disease_risk": os.path.join(MODEL_DIR, "disease_risk_model.pkl"),
    "flood_risk": os.path.join(MODEL_DIR, "flood_risk_model.pkl"),
    "drought_risk": os.path.join(MODEL_DIR, "drought_risk_model.pkl"),
    "sustainability": os.path.join(MODEL_DIR, "sustainability_model.pkl"),
}

RANDOM_SEED = 42
