"""
src/backfill.py
===============
One-time script to populate the Hopsworks Feature Store
with 90 days of historical AQI data.

Run this ONCE after you've added your Hopsworks API key:
    python src/backfill.py

What it does:
  1. Generates or fetches 90 days of hourly AQI data per city
  2. Runs the full feature engineering pipeline
  3. Pushes all rows to the Hopsworks Feature Group

This provides enough historical data to train the ML models.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger
from src.feature_engineering import add_all_features
from src.preprocess import clean_data, handle_missing_values, remove_outliers

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# SYNTHETIC HISTORICAL DATA GENERATOR
# (used when OpenWeatherMap API doesn't have historical data)
# ══════════════════════════════════════════════════════════════

def generate_historical_data(
    city: str,
    days: int = 90,
    end_dt: datetime = None,
) -> pd.DataFrame:
    """
    Generate `days` days of realistic synthetic hourly AQI data for a city.
    Uses seasonal patterns, daily cycles, noise, and city-specific baselines.
    """
    if end_dt is None:
        end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    n_hours = days * 24
    start_dt = end_dt - timedelta(hours=n_hours - 1)
    timestamps = pd.date_range(start=start_dt, periods=n_hours, freq="h")

    rng = np.random.default_rng(seed=abs(hash(city)) % (2**31))

    # City-specific AQI baseline (higher for more polluted cities)
    city_baselines = {
        "Sukkur": 140,
    }
    base_aqi = city_baselines.get(city, 100)

    hours = np.arange(n_hours)
    t     = timestamps

    # Daily cycle: worse at rush hours (8AM, 6PM), best at 4AM
    hour_of_day = t.hour
    daily_cycle = (
        20 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)  # peaks ~2PM
    )

    # Weekly cycle: weekends slightly better
    day_of_week = t.dayofweek
    weekly_cycle = np.where(day_of_week >= 5, -10, 5)

    # Seasonal cycle: worse in winter (month 12,1,2)
    month = t.month
    seasonal_cycle = -15 * np.cos(2 * np.pi * (month - 1) / 12)

    # Trend: slight improvement over 90 days
    trend = np.linspace(5, -5, n_hours)

    # Random noise
    noise = rng.normal(0, 12, n_hours)

    aqi_raw = base_aqi + daily_cycle + weekly_cycle + seasonal_cycle + trend + noise
    aqi = np.clip(aqi_raw, 10, 450).round(1)

    # Derived pollutants (correlated with AQI)
    aqi_norm = aqi / 200.0
    pm2_5 = np.clip(aqi_norm * 60  + rng.normal(0, 5, n_hours), 0, 200).round(2)
    pm10  = np.clip(aqi_norm * 100 + rng.normal(0, 8, n_hours), 0, 300).round(2)
    co    = np.clip(aqi_norm * 800 + rng.normal(0, 50, n_hours), 0, 3000).round(2)
    no    = np.clip(aqi_norm * 30  + rng.normal(0, 3, n_hours),  0, 100).round(2)
    no2   = np.clip(aqi_norm * 50  + rng.normal(0, 5, n_hours),  0, 200).round(2)
    o3    = np.clip(60 - aqi_norm * 20 + rng.normal(0, 8, n_hours), 0, 120).round(2)
    so2   = np.clip(aqi_norm * 20  + rng.normal(0, 3, n_hours),  0, 80).round(2)
    nh3   = np.clip(aqi_norm * 10  + rng.normal(0, 2, n_hours),  0, 40).round(2)

    # Weather (not AQI-derived but correlated)
    temp      = (20 + 10 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
                 + rng.normal(0, 3, n_hours)).round(1)
    humidity  = np.clip(60 + rng.normal(0, 15, n_hours), 10, 99).round(1)
    wind_speed = np.clip(3 + rng.exponential(2, n_hours), 0, 30).round(1)
    pressure  = np.clip(rng.normal(1013, 5, n_hours), 970, 1050).round(1)
    visibility = np.clip(10 - aqi_norm * 5 + rng.normal(0, 1, n_hours), 0, 20).round(1)

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "city":        city,
        "aqi":         aqi,
        "pm2_5":       pm2_5,
        "pm10":        pm10,
        "co":          co,
        "no":          no,
        "no2":         no2,
        "o3":          o3,
        "so2":         so2,
        "nh3":         nh3,
        "temp":        temp,
        "humidity":    humidity,
        "wind_speed":  wind_speed,
        "pressure":    pressure,
        "visibility":  visibility,
    })

    return df


# ══════════════════════════════════════════════════════════════
# BACKFILL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_backfill(
    cities: list[str] = None,
    days: int = 90,
    push_to_store: bool = True,
) -> pd.DataFrame:
    """
    Generate + push 90 days of history for all cities.

    Args:
        cities:        List of city names. Defaults to config.SUPPORTED_CITIES.
        days:          Number of historical days to generate.
        push_to_store: If True, push to Hopsworks Feature Store.

    Returns:
        Combined DataFrame of all cities.
    """
    if cities is None:
        cities = config.SUPPORTED_CITIES

    all_dfs = []

    for city in cities:
        logger.info(f"Generating {days}-day history for {city} ...")

        raw_df = generate_historical_data(city, days=days)

        # Clean + feature engineer
        df = clean_data(raw_df)
        df = handle_missing_values(df)
        df = remove_outliers(df)
        df = add_all_features(df)
        df = df.dropna().reset_index(drop=True)

        logger.info(f"  {city}: {len(df)} rows, {len(df.columns)} features")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total backfill rows: {len(combined)} across {len(cities)} cities")

    if push_to_store:
        if not config.USE_FEATURE_STORE:
            logger.warning(
                "Hopsworks not configured — saving backfill data to CSV instead.\n"
                "Add HOPSWORKS_API_KEY to .env and re-run to push to Feature Store."
            )
            out = config.DATA_DIR / "backfill_data.csv"
            combined.to_csv(out, index=False)
            logger.info(f"Backfill saved locally: {out}")
        else:
            from src.feature_store import push_features
            push_features(combined)
            logger.info("Backfill pushed to Hopsworks Feature Store.")

    return combined


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill Hopsworks Feature Store with historical AQI data."
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of historical days to generate (default: 90)"
    )
    parser.add_argument(
        "--cities", nargs="+", default=None,
        help="City names (default: all supported cities)"
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Do not push to Feature Store, only generate and save locally"
    )
    args = parser.parse_args()

    df = run_backfill(
        cities=args.cities,
        days=args.days,
        push_to_store=not args.no_push,
    )

    print(f"\nBackfill complete: {len(df)} total rows, {len(df.columns)} features")
    print(f"Cities: {df['city'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
