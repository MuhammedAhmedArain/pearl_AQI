"""
src/fetch_data.py
=================
Data acquisition layer.

Responsibilities:
  1. Fetch current AQI + weather data from OpenWeatherMap API
  2. Fetch historical AQI data (last 5 days) for lag feature creation
  3. Cache responses to avoid hitting rate limits
  4. Gracefully fall back to sample data if API is unavailable
  5. Persist raw data to CSV

API Endpoints used:
  - /weather         — current weather (temp, humidity, wind, pressure)
  - /air_pollution   — current + historical AQI breakdown (PM2.5, PM10, O3, NO2, SO2, CO)
  - /air_pollution/history — hourly history (past 5 days)

AQI sub-components returned by OpenWeatherMap:
  co, no, no2, o3, so2, pm2_5, pm10, nh3
"""

import json
import time
import datetime
import pandas as pd
import requests
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger, retry, timer, get_cache_path, is_cache_valid, save_json, load_json

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# GEOCODING — City name → (lat, lon)
# ══════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=1.5, exceptions=(requests.RequestException,))
def get_city_coordinates(city: str) -> tuple[float, float]:
    """
    Resolve a city name to (latitude, longitude) using OpenWeatherMap Geocoding API.

    Returns:
        (lat, lon) tuple.
    Raises:
        ValueError if city cannot be geocoded.
    """
    url    = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": config.OPENWEATHER_API_KEY}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"City '{city}' could not be geocoded.")

    lat, lon = data[0]["lat"], data[0]["lon"]
    logger.info(f"Geocoded '{city}' → lat={lat:.4f}, lon={lon:.4f}")
    return lat, lon


# ══════════════════════════════════════════════════════════════
# CURRENT WEATHER
# ══════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=1.5, exceptions=(requests.RequestException,))
def fetch_current_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather conditions for a location.

    Returns dict with: temp, humidity, wind_speed, pressure, visibility, clouds
    """
    url    = f"{config.OPENWEATHER_BASE_URL}/weather"
    params = {"lat": lat, "lon": lon, "units": "metric", "appid": config.OPENWEATHER_API_KEY}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    weather = {
        "temp":        data["main"]["temp"],
        "feels_like":  data["main"]["feels_like"],
        "humidity":    data["main"]["humidity"],
        "pressure":    data["main"]["pressure"],
        "wind_speed":  data["wind"]["speed"],
        "wind_deg":    data["wind"].get("deg", 0),
        "clouds":      data["clouds"]["all"],
        "visibility":  data.get("visibility", 10000),
        "weather_main": data["weather"][0]["main"],
        "timestamp":   datetime.datetime.utcnow().isoformat(),
    }
    logger.debug(f"Weather fetched: temp={weather['temp']}°C, humidity={weather['humidity']}%")
    return weather


# ══════════════════════════════════════════════════════════════
# CURRENT AQI
# ══════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=1.5, exceptions=(requests.RequestException,))
def fetch_current_aqi(lat: float, lon: float) -> dict:
    """
    Fetch current air quality index and pollutant breakdown.

    OpenWeatherMap returns AQI on a 1-5 scale; we convert to US EPA 0-500 scale.
    Pollutants: co, no, no2, o3, so2, pm2_5, pm10, nh3 (µg/m³)
    """
    params   = {"lat": lat, "lon": lon, "appid": config.OPENWEATHER_API_KEY}
    response = requests.get(config.OPENWEATHER_AQI_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    item       = data["list"][0]
    components = item["components"]

    # Convert OWM 1-5 AQI to approximate US EPA scale
    owm_aqi     = item["main"]["aqi"]
    epa_aqi     = _owm_to_epa_aqi(owm_aqi, components)

    result = {
        "aqi":      epa_aqi,
        "owm_aqi":  owm_aqi,
        "pm2_5":    components.get("pm2_5",  0),
        "pm10":     components.get("pm10",   0),
        "co":       components.get("co",     0),
        "no":       components.get("no",     0),
        "no2":      components.get("no2",    0),
        "o3":       components.get("o3",     0),
        "so2":      components.get("so2",    0),
        "nh3":      components.get("nh3",    0),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    logger.debug(f"AQI fetched: EPA AQI={epa_aqi:.1f}")
    return result


def _owm_to_epa_aqi(owm_aqi: int, components: dict) -> float:
    """
    Approximate conversion from OpenWeatherMap 1-5 scale to US EPA 0-500.
    Uses PM2.5 as the primary driver when available.
    """
    pm25 = components.get("pm2_5", 0)
    # PM2.5 breakpoints (µg/m³) → AQI breakpoints (EPA)
    breakpoints = [
        (0.0,   12.0,   0,  50),
        (12.1,  35.4,   51, 100),
        (35.5,  55.4,  101, 150),
        (55.5, 150.4,  151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
            return round(aqi, 1)

    # Fallback: map OWM 1-5 linearly
    scale = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
    return float(scale.get(owm_aqi, 50))


# ══════════════════════════════════════════════════════════════
# HISTORICAL AQI
# ══════════════════════════════════════════════════════════════

@retry(max_attempts=3, delay=2.0, exceptions=(requests.RequestException,))
def fetch_historical_aqi(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """
    Fetch hourly AQI history for the last `days` days.

    Returns a DataFrame with columns: timestamp, aqi, pm2_5, pm10, co, no2, o3, so2
    """
    end_ts   = int(time.time())
    start_ts = end_ts - days * 86400

    url    = f"{config.OPENWEATHER_AQI_URL}/history"
    params = {
        "lat":   lat,
        "lon":   lon,
        "start": start_ts,
        "end":   end_ts,
        "appid": config.OPENWEATHER_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    records = []
    for item in data.get("list", []):
        comp = item["components"]
        owm_aqi = item["main"]["aqi"]
        records.append({
            "timestamp": datetime.datetime.utcfromtimestamp(item["dt"]).isoformat(),
            "aqi":       _owm_to_epa_aqi(owm_aqi, comp),
            "pm2_5":     comp.get("pm2_5", 0),
            "pm10":      comp.get("pm10",  0),
            "co":        comp.get("co",    0),
            "no":        comp.get("no",    0),
            "no2":       comp.get("no2",   0),
            "o3":        comp.get("o3",    0),
            "so2":       comp.get("so2",   0),
            "nh3":       comp.get("nh3",   0),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    logger.info(f"Historical AQI: {len(df)} hourly records fetched ({days} days)")
    return df


# ══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

@timer
def fetch_city_data(city: str, use_cache: bool = True) -> tuple[dict, dict, pd.DataFrame]:
    """
    Full data fetch pipeline for a city:
      1. Check cache
      2. Geocode city
      3. Fetch current weather + AQI
      4. Fetch 30-day historical AQI
      5. Persist to CSV

    Returns:
        (current_weather, current_aqi, historical_df)
    """
    cache_path = get_cache_path(city, "json")

    # ── Cache hit ────────────────────────────────────────────
    if use_cache and is_cache_valid(cache_path):
        logger.info(f"Cache hit for '{city}' — skipping API calls.")
        cached = load_json(cache_path)
        historical_df = pd.read_csv(config.RAW_DATA_FILE) if config.RAW_DATA_FILE.exists() else _load_sample_data()
        return cached["weather"], cached["aqi"], historical_df

    # ── Live fetch ───────────────────────────────────────────
    try:
        lat, lon       = get_city_coordinates(city)
        current_weather = fetch_current_weather(lat, lon)
        current_aqi     = fetch_current_aqi(lat, lon)
        historical_df   = fetch_historical_aqi(lat, lon, days=30)

        # Persist cache
        save_json({"weather": current_weather, "aqi": current_aqi}, cache_path)

        # Add city + merge weather columns to historical
        historical_df["city"] = city
        _save_historical(historical_df, city)

        return current_weather, current_aqi, historical_df

    except Exception as exc:
        logger.error(f"API fetch failed for '{city}': {exc}. Falling back to sample data.")
        return _fallback_data(city)


def _save_historical(df: pd.DataFrame, city: str) -> None:
    """Append / overwrite raw historical CSV."""
    city_col = df.copy()
    city_col["city"] = city

    if config.RAW_DATA_FILE.exists():
        existing = pd.read_csv(config.RAW_DATA_FILE, parse_dates=["timestamp"])
        combined = pd.concat([existing, city_col], ignore_index=True)
        combined.drop_duplicates(subset=["timestamp", "city"], keep="last", inplace=True)
    else:
        combined = city_col

    combined.to_csv(config.RAW_DATA_FILE, index=False)
    logger.info(f"Raw data saved → {config.RAW_DATA_FILE} ({len(combined)} rows)")


def _load_sample_data() -> pd.DataFrame:
    """Load the bundled sample CSV as fallback."""
    if config.SAMPLE_DATA_FILE.exists():
        return pd.read_csv(config.SAMPLE_DATA_FILE, parse_dates=["timestamp"])
    return pd.DataFrame()


def _fallback_data(city: str) -> tuple[dict, dict, pd.DataFrame]:
    """Return synthetic fallback data when API is unavailable."""
    import numpy as np
    rng   = pd.date_range(end=pd.Timestamp.now(), periods=720, freq="h")
    aqi_vals = np.clip(np.random.normal(80, 30, len(rng)), 10, 400)

    df = pd.DataFrame({
        "timestamp": rng,
        "aqi":       aqi_vals,
        "pm2_5":     aqi_vals * 0.25,
        "pm10":      aqi_vals * 0.4,
        "co":        300 + np.random.normal(0, 50, len(rng)),
        "no":        np.random.exponential(2, len(rng)),
        "no2":       aqi_vals * 0.15,
        "o3":        aqi_vals * 0.2,
        "so2":       aqi_vals * 0.05,
        "nh3":       np.random.exponential(1, len(rng)),
        "city":      city,
    })

    weather = {
        "temp": 28.0, "feels_like": 30.0, "humidity": 65,
        "pressure": 1012, "wind_speed": 3.5, "wind_deg": 180,
        "clouds": 20, "visibility": 8000, "weather_main": "Haze",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    aqi_val = float(aqi_vals[-1])
    aqi     = {
        "aqi": aqi_val, "owm_aqi": 3,
        "pm2_5": aqi_val * 0.25, "pm10": aqi_val * 0.4,
        "co": 300.0, "no": 2.0, "no2": aqi_val * 0.15,
        "o3": aqi_val * 0.2, "so2": aqi_val * 0.05, "nh3": 1.0,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    logger.warning(f"Using synthetic fallback data for '{city}'.")
    return weather, aqi, df


# ══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch AQI data for a city.")
    parser.add_argument("--city", default=config.DEFAULT_CITY, help="City name")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    parser.add_argument("--push-to-store", action="store_true", help="Push results to Feature Store")
    args = parser.parse_args()

    weather, aqi, hist = fetch_city_data(args.city, use_cache=not args.no_cache)

    if args.push_to_store:
        try:
            from src.feature_engineering import add_all_features
            from src.feature_store import push_features

            # Prepare data for push: We need enough history for lag features
            featured_df = add_all_features(hist.copy())
            latest_row  = featured_df.tail(1)

            push_features(latest_row)
            print("🚀 Successfully pushed latest observation to Feature Store.")
        except Exception as e:
            print(f"❌ Failed to push to Feature Store: {e}")
    print(f"\n📍 City     : {args.city}")
    print(f"🌡  Temp     : {weather['temp']}°C")
    print(f"💨 AQI      : {aqi['aqi']}")
    print(f"📊 History  : {len(hist)} rows")
    print(f"PM2.5       : {aqi['pm2_5']} µg/m³")
