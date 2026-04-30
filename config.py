"""
config.py
=========
Centralized configuration for Pearls AQI Predictor.
All tuneable parameters, API keys, and paths live here.
Load secrets from a .env file so nothing is committed to git.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables from .env ────────────────────
load_dotenv()

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"

# Ensure directories exist at import time
for _d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Configuration ────────────────────────────────────────
# Sign up at https://openweathermap.org/api for a free key
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_AQI_URL: str  = "https://api.openweathermap.org/data/2.5/air_pollution"

# AQICN (optional fallback)
AQICN_API_KEY: str = os.getenv("AQICN_API_KEY", "demo")
AQICN_BASE_URL: str = "https://api.waqi.info/feed"

# ── City Configuration ───────────────────────────────────────
SUPPORTED_CITIES: list[str] = ["Sukkur"]

DEFAULT_CITY: str = "Sukkur"

# ── Data Configuration ───────────────────────────────────────
RAW_DATA_FILE: Path      = DATA_DIR / "raw_aqi_data.csv"
PROCESSED_DATA_FILE: Path = DATA_DIR / "processed_aqi_data.csv"
SAMPLE_DATA_FILE: Path   = DATA_DIR / "sample_aqi_data.csv"
CACHE_DIR: Path          = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS: int = 3600  # 1 hour

# ── Feature Engineering ──────────────────────────────────────
LAG_PERIODS: list[int]     = [1, 2, 3, 6, 12, 24]   # hours
ROLLING_WINDOWS: list[int] = [3, 7, 14]              # days
TARGET_COLUMN: str         = "aqi"

TIME_FEATURES: list[str] = [
    "hour", "day_of_week", "day_of_month", "month", "is_weekend"
]

# ── Model Configuration ──────────────────────────────────────
RANDOM_SEED: int   = 42
TEST_SIZE: float   = 0.2
CV_FOLDS: int      = 5

BEST_MODEL_PATH: Path         = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH: Path     = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH: Path   = MODELS_DIR / "model_comparison.csv"
SCALER_PATH: Path             = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH: Path      = MODELS_DIR / "feature_names.json"

# ── Prediction Configuration ─────────────────────────────────
FORECAST_DAYS: int     = 3
FORECAST_HOURS: int    = FORECAST_DAYS * 24   # hourly internally, display daily

# ── AQI Category Thresholds (US EPA standard) ────────────────
AQI_CATEGORIES: list[dict] = [
    {"min": 0,   "max": 50,  "label": "Good",                  "color": "#00e400", "emoji": "😊"},
    {"min": 51,  "max": 100, "label": "Moderate",              "color": "#ffff00", "emoji": "😐"},
    {"min": 101, "max": 150, "label": "Unhealthy for Sensitive","color": "#ff7e00", "emoji": "😷"},
    {"min": 151, "max": 200, "label": "Unhealthy",             "color": "#ff0000", "emoji": "🤢"},
    {"min": 201, "max": 300, "label": "Very Unhealthy",        "color": "#8f3f97", "emoji": "🚨"},
    {"min": 301, "max": 500, "label": "Hazardous",             "color": "#7e0023", "emoji": "☠️"},
]

AQI_ALERT_THRESHOLD: int = 150  # Show alert in dashboard if AQI > this

# ── API Server Configuration ─────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Logging ──────────────────────────────────────────────────
LOG_FILE: Path  = LOGS_DIR / "aqi_predictor.log"
LOG_LEVEL: str  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_aqi_category(aqi_value: float) -> dict:
    """Return the AQI category dict for a given AQI value."""
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_value <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]  # Hazardous fallback


# ══════════════════════════════════════════════════════════════
# HOPSWORKS FEATURE STORE + MODEL REGISTRY
# ══════════════════════════════════════════════════════════════

HOPSWORKS_API_KEY:  str = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT:  str = os.getenv("HOPSWORKS_PROJECT", "pearlsAQI")
HOPSWORKS_HOST:     str = os.getenv("HOPSWORKS_HOST",    "eu-west.cloud.hopsworks.ai")

# ── Feature Group ─────────────────────────────────────────────
FEATURE_GROUP_NAME:    str = "aqi_features"
FEATURE_GROUP_VERSION: int = 1

# ── Feature View (used for training data retrieval) ──────────
FEATURE_VIEW_NAME:    str = "aqi_feature_view"
FEATURE_VIEW_VERSION: int = 1

# ── Model Registry ───────────────────────────────────────────
MODEL_NAME:    str = "pearls_aqi_model"
MODEL_VERSION: int = 1

# ── Runtime flag: True only when key is set and not placeholder ─
USE_FEATURE_STORE: bool = (
    bool(HOPSWORKS_API_KEY)
    and HOPSWORKS_API_KEY != "PASTE_YOUR_HOPSWORKS_API_KEY_HERE"
)

