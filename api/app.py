"""
api/app.py
==========
FastAPI backend for the Pearls AQI Predictor.

Endpoints:
  GET /              — API info
  GET /health        — Health check
  GET /predict       — Predict AQI for a city
  GET /cities        — List supported cities
  GET /model/info    — Current best model metadata

Design:
  - Uses async background refresh of data (non-blocking)
  - Response caching (in-memory, TTL-based)
  - Full OpenAPI docs at /docs
  - CORS enabled for Streamlit frontend
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger, load_json, is_cache_valid

logger = get_logger(__name__)

# ── In-memory prediction cache ───────────────────────────────
_prediction_cache: dict = {}
_CACHE_TTL = config.CACHE_TTL_SECONDS


# ══════════════════════════════════════════════════════════════
# APP INITIALISATION
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Production-ready Air Quality Index prediction API. "
        "Predicts AQI for the next 3 days for any supported city "
        "using an ensemble ML model trained on historical + real-time data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

class DayForecast(BaseModel):
    day:       int
    date:      str
    aqi_mean:  float
    aqi_max:   float
    aqi_min:   float
    category:  str
    color:     str
    emoji:     str
    alert:     bool

class PredictionResponse(BaseModel):
    city:                str
    current_aqi:         float
    current_category:    str
    current_color:       str
    current_emoji:       str
    current_alert:       bool
    current_weather:     dict
    current_pollutants:  dict
    daily_predictions:   list[DayForecast]
    model_name:          str
    model_metrics:       dict
    forecast_days:       int
    generated_at:        str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    api_version:  str
    timestamp:    str


# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Validate that the model is available at startup."""
    if not config.BEST_MODEL_PATH.exists():
        logger.warning(
            "⚠️  No trained model found. "
            "Run `python src/train.py` before using the /predict endpoint."
        )
    else:
        logger.info("✅ API started. Model is ready.")


# ══════════════════════════════════════════════════════════════
# HELPER — Run prediction in executor to avoid blocking async loop
# ══════════════════════════════════════════════════════════════

def _run_prediction_sync(city: str, compute_shap: bool) -> dict:
    """Synchronous prediction call (runs in thread pool from async context)."""
    from src.fetch_data import fetch_city_data
    from src.predict import predict_next_days

    weather, aqi, historical_df = fetch_city_data(city, use_cache=True)
    return predict_next_days(
        city, weather, aqi, historical_df, compute_shap=compute_shap
    )


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", summary="API Root")
async def root():
    """Welcome endpoint with API overview."""
    return {
        "name":        "Pearls AQI Predictor API",
        "version":     "1.0.0",
        "description": "Predict Air Quality Index for the next 3 days",
        "endpoints": {
            "/predict":    "GET ?city=<city_name>&shap=true",
            "/cities":     "GET — List supported cities",
            "/health":     "GET — Health check",
            "/model/info": "GET — Current model metadata",
            "/docs":       "GET — OpenAPI documentation",
        },
        "example": "/predict?city=Karachi",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """
    Check API health and model availability.
    Returns 200 if healthy, model_loaded indicates whether a trained model exists.
    """
    model_loaded = config.BEST_MODEL_PATH.exists()
    return HealthResponse(
        status       = "healthy",
        model_loaded = model_loaded,
        api_version  = "1.0.0",
        timestamp    = datetime.utcnow().isoformat(),
    )


@app.get("/cities", summary="List Supported Cities")
async def list_cities():
    """Return all cities supported by the prediction system."""
    return {
        "cities":      config.SUPPORTED_CITIES,
        "default":     config.DEFAULT_CITY,
        "total_count": len(config.SUPPORTED_CITIES),
    }


@app.get("/predict", summary="Predict AQI for a City")
async def predict(
    city: str = Query(
        default=config.DEFAULT_CITY,
        description="City name to predict AQI for",
        example="Karachi",
    ),
    shap: bool = Query(
        default=False,
        description="Include SHAP feature importance in response (slower)",
    ),
):
    """
    Predict AQI for the next 3 days for a given city.

    Returns:
    - **current_aqi**: Current AQI value
    - **daily_predictions**: List of 3 daily forecasts (mean/max/min + category)
    - **model_name**: ML model used for prediction
    - **shap_explanation**: Top contributing features (if shap=true)
    - **current_alert**: True if current AQI > 150 (Unhealthy threshold)
    """
    # ── Validate city ────────────────────────────────────────
    # Allow any city (not just the config list) but warn if unknown
    city_clean = city.strip().title()

    # ── Check in-memory cache ────────────────────────────────
    cache_key = f"{city_clean}_shap{shap}"
    if cache_key in _prediction_cache:
        cached_entry = _prediction_cache[cache_key]
        age = time.time() - cached_entry["_cached_at"]
        if age < _CACHE_TTL:
            logger.info(f"Memory cache hit for '{city_clean}' (age={age:.0f}s)")
            return cached_entry["data"]

    # ── Run prediction ───────────────────────────────────────
    if not config.BEST_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. "
                "Please run `python src/train.py` first."
            ),
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _run_prediction_sync, city_clean, shap
        )

        # Store in memory cache
        _prediction_cache[cache_key] = {
            "_cached_at": time.time(),
            "data":       result,
        }

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as exc:
        logger.error(f"Prediction error for '{city_clean}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


@app.get("/model/info", summary="Current Model Metadata")
async def model_info():
    """
    Return metadata about the currently deployed best model,
    including its performance metrics and training timestamp.
    """
    if not config.MODEL_METADATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Train a model first."
        )

    metadata = load_json(config.MODEL_METADATA_PATH)
    return metadata


@app.get("/model/comparison", summary="All Model Comparison Results")
async def model_comparison():
    """Return the full model comparison table from the last training run."""
    import pandas as pd

    if not config.MODEL_COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model comparison not found. Train models first."
        )

    df = pd.read_csv(config.MODEL_COMPARISON_PATH)
    return {
        "models":      df.to_dict(orient="records"),
        "best_model":  df.iloc[0]["model_name"] if len(df) > 0 else None,
        "ranked_by":   "RMSE (ascending)",
    }


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
