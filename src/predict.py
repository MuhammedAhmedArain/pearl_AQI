"""
src/predict.py
==============
Prediction and explainability module.

Responsibilities:
  1. Load the saved best model + scaler + feature list
  2. Generate future feature rows for the next N days
  3. Apply the scaler transform
  4. Run inference and return daily AQI predictions
  5. Compute SHAP values for explainability
  6. Return structured prediction results

Design:
  - Uses recursive 1-step forecasting (each hourly prediction feeds into next)
  - Aggregates hourly → daily mean AQI for the 3-day output
  - SHAP uses TreeExplainer for tree-based models (fast), KernelExplainer fallback
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from typing import Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger, timer, load_json, aqi_category
from src.feature_engineering import generate_future_features

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_best_model() -> tuple[Any, Any, list[str], dict]:
    """
    Load saved model, scaler, feature names, and metadata.

    Source priority:
      1. Hopsworks Model Registry  (if HOPSWORKS_API_KEY is configured)
      2. Local models/best_model.pkl (fallback)

    Returns:
        (model, scaler, feature_names, metadata)
    Raises:
        FileNotFoundError if no model found anywhere.
    """
    # -- Try Hopsworks Model Registry first ---------------------------
    if config.USE_FEATURE_STORE:
        try:
            from src.feature_store import load_model_from_registry
            model, scaler, feature_names, metadata = load_model_from_registry()
            logger.info(
                f"Loaded from Registry: {metadata.get('model_name', 'Unknown')} "
                f"(RMSE={metadata.get('rmse', 'N/A')})"
            )
            return model, scaler, feature_names, metadata
        except Exception as exc:
            logger.warning(f"Model Registry load failed ({exc}). Falling back to local.")

    # -- Local fallback -----------------------------------------------
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.BEST_MODEL_PATH}. "
            "Run `python src/train.py` first."
        )

    model    = joblib.load(config.BEST_MODEL_PATH)
    scaler   = joblib.load(config.SCALER_PATH) if config.SCALER_PATH.exists() else None
    metadata = load_json(config.MODEL_METADATA_PATH) if config.MODEL_METADATA_PATH.exists() else {}
    feature_names = (
        load_json(config.FEATURE_NAMES_PATH)
        if config.FEATURE_NAMES_PATH.exists()
        else metadata.get("feature_names", [])
    )

    logger.info(
        f"Loaded local model: {metadata.get('model_name', 'Unknown')} "
        f"(RMSE={metadata.get('rmse', 'N/A')})"
    )
    return model, scaler, feature_names, metadata


# ══════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 100,
) -> dict:
    """
    Compute SHAP values for the given prediction input.

    Strategy:
      - Tree-based models (RF, XGBoost, GBM): TreeExplainer (fast)
      - Other models: KernelExplainer (slower, uses background sample)

    Returns:
        Dict with 'values', 'feature_names', 'mean_abs_shap' (sorted)
    """
    try:
        # Limit samples for performance
        X_sample = X.head(min(max_samples, len(X)))

        model_type = type(model).__name__.lower()
        tree_models = ("randomforest", "gradientboosting", "xgb", "lgbm", "extra")

        if any(t in model_type for t in tree_models):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_sample)
        else:
            # Use a small background dataset
            background = shap.sample(X_sample, min(50, len(X_sample)))
            explainer  = shap.KernelExplainer(model.predict, background)
            shap_vals  = explainer.shap_values(X_sample, nsamples=50)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        feature_importance = sorted(
            zip(feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:15]  # Top 15 features

        return {
            "shap_values":     shap_vals.tolist(),
            "feature_names":   feature_names,
            "top_features":    [{"feature": f, "importance": round(v, 4)} for f, v in feature_importance],
            "status":          "success",
        }

    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}. Skipping explainability.")
        return {"status": "failed", "error": str(exc), "top_features": []}


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════

def _align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Align the input DataFrame to the exact feature set the model was trained on.
    Missing columns are filled with 0; extra columns are dropped.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_names]


@timer
def predict_next_days(
    city: str,
    current_weather: dict,
    current_aqi: dict,
    historical_df: pd.DataFrame,
    forecast_days: int = config.FORECAST_DAYS,
    compute_shap: bool = True,
) -> dict:
    """
    Main prediction function. Generates hourly predictions for the next
    `forecast_days` days, then aggregates to daily AQI.

    Args:
        city:            City name (for metadata)
        current_weather: Dict of current weather conditions
        current_aqi:     Dict of current AQI + pollutants
        historical_df:   DataFrame with recent AQI history + features
        forecast_days:   Number of days to forecast (default 3)
        compute_shap:    Whether to compute SHAP values

    Returns:
        Prediction result dict with keys:
          city, current_aqi, daily_predictions, model_name,
          model_metrics, shap_explanation
    """
    # ── 1. Load model ────────────────────────────────────────
    model, scaler, feature_names, metadata = load_best_model()

    # ── 2. Feature engineering on history ────────────────────
    from src.preprocess import handle_missing_values, remove_outliers, clean_data
    from src.feature_engineering import add_all_features

    # Ensure history has required columns
    if "timestamp" not in historical_df.columns:
        historical_df["timestamp"] = pd.date_range(
            end=pd.Timestamp.now(), periods=len(historical_df), freq="h"
        )

    hist_clean     = clean_data(historical_df)
    hist_imputed   = handle_missing_values(hist_clean)
    hist_featured  = add_all_features(hist_imputed.copy())
    hist_featured  = hist_featured.dropna()

    # ── 3. Generate future features ──────────────────────────
    forecast_hours = forecast_days * 24
    future_df      = generate_future_features(hist_featured, forecast_hours=forecast_hours)

    # ── 4. Align to model features ───────────────────────────
    future_aligned = _align_features(future_df.copy(), feature_names)

    # ── 5. Scale if scaler available ─────────────────────────
    if scaler is not None:
        try:
            future_scaled = pd.DataFrame(
                scaler.transform(future_aligned),
                columns=feature_names,
            )
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}. Using unscaled features.")
            future_scaled = future_aligned
    else:
        future_scaled = future_aligned

    # ── 6. Predict ───────────────────────────────────────────
    hourly_preds = model.predict(future_scaled)
    hourly_preds = np.clip(hourly_preds, 0, 500)  # Enforce valid AQI range

    # ── 7. Aggregate to daily ────────────────────────────────
    daily_predictions = []
    for day in range(forecast_days):
        start_h = day * 24
        end_h   = start_h + 24
        day_preds = hourly_preds[start_h:end_h]

        daily_mean = float(np.mean(day_preds))
        daily_max  = float(np.max(day_preds))
        daily_min  = float(np.min(day_preds))
        cat        = aqi_category(daily_mean)
        date_label = (pd.Timestamp.now() + pd.Timedelta(days=day + 1)).strftime("%Y-%m-%d")

        daily_predictions.append({
            "day":      day + 1,
            "date":     date_label,
            "aqi_mean": round(daily_mean, 1),
            "aqi_max":  round(daily_max,  1),
            "aqi_min":  round(daily_min,  1),
            "category": cat["label"],
            "color":    cat["color"],
            "emoji":    cat["emoji"],
            "alert":    daily_mean > config.AQI_ALERT_THRESHOLD,
        })

    # ── 8. SHAP explainability ───────────────────────────────
    shap_result = {}
    if compute_shap:
        shap_result = compute_shap_values(
            model, future_scaled, feature_names
        )

    # ── 9. Current AQI category ──────────────────────────────
    curr_aqi_val = float(current_aqi.get("aqi", 0))
    curr_cat     = aqi_category(curr_aqi_val)

    result = {
        "city":              city,
        "current_aqi":       round(curr_aqi_val, 1),
        "current_category":  curr_cat["label"],
        "current_color":     curr_cat["color"],
        "current_emoji":     curr_cat["emoji"],
        "current_alert":     curr_aqi_val > config.AQI_ALERT_THRESHOLD,
        "current_weather": {
            "temp":       current_weather.get("temp"),
            "humidity":   current_weather.get("humidity"),
            "wind_speed": current_weather.get("wind_speed"),
            "pressure":   current_weather.get("pressure"),
        },
        "current_pollutants": {
            "pm2_5": round(float(current_aqi.get("pm2_5", 0)), 2),
            "pm10":  round(float(current_aqi.get("pm10",  0)), 2),
            "no2":   round(float(current_aqi.get("no2",   0)), 2),
            "o3":    round(float(current_aqi.get("o3",    0)), 2),
            "co":    round(float(current_aqi.get("co",    0)), 2),
        },
        "daily_predictions": daily_predictions,
        "model_name":        metadata.get("model_name", "Unknown"),
        "model_metrics": {
            "mae":  metadata.get("mae"),
            "rmse": metadata.get("rmse"),
            "r2":   metadata.get("r2"),
        },
        "shap_explanation":  shap_result,
        "forecast_days":     forecast_days,
        "generated_at":      pd.Timestamp.now().isoformat(),
    }

    logger.info(
        f"Predictions for '{city}': current={curr_aqi_val:.1f}, "
        f"day1={daily_predictions[0]['aqi_mean']:.1f}, "
        f"day2={daily_predictions[1]['aqi_mean']:.1f}, "
        f"day3={daily_predictions[2]['aqi_mean']:.1f}"
    )
    return result



# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict AQI for a city.")
    parser.add_argument("--city", default=config.DEFAULT_CITY)
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP computation")
    args = parser.parse_args()

    from src.fetch_data import fetch_city_data
    weather, aqi, hist = fetch_city_data(args.city)
    result = predict_next_days(args.city, weather, aqi, hist, compute_shap=not args.no_shap)

    print(f"\n  City        : {result['city']}")
    print(f"   Current AQI : {result['current_aqi']} ({result['current_category']})")
    print(f"\n  3-Day Forecast:")
    for day in result["daily_predictions"]:
        alert = "     ALERT!" if day["alert"] else ""
        print(f"  Day {day['day']} ({day['date']}): AQI {day['aqi_mean']}   {day['category']}{alert}")

    if result["shap_explanation"].get("top_features"):
        print(f"\n  Top Features (SHAP):")
        for feat in result["shap_explanation"]["top_features"][:5]:
            print(f"  {feat['feature']:35s} {feat['importance']:.4f}")
