import pandas as pd

import config
from api.app import app
from fastapi.testclient import TestClient
from src.feature_engineering import add_all_features
from src.fetch_data import build_latest_feature_payload
from src.utils import aqi_category


def test_aqi_category_boundaries():
    assert aqi_category(25)["label"] == "Good"
    assert aqi_category(100)["label"] == "Moderate"
    assert aqi_category(175)["label"] == "Unhealthy"
    assert aqi_category(350)["label"] == "Hazardous"


def test_feature_engineering_adds_expected_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="h"),
            "city": [config.DEFAULT_CITY] * 30,
            "aqi": range(30),
            "pm2_5": range(30),
            "pm10": range(1, 31),
            "co": [300.0] * 30,
            "no": [1.0] * 30,
            "no2": [10.0] * 30,
            "o3": [20.0] * 30,
            "so2": [2.0] * 30,
            "nh3": [0.5] * 30,
        }
    )

    featured = add_all_features(df)

    for column in [
        "hour",
        "hour_sin",
        "aqi_lag_24",
        "aqi_roll_mean_3d",
        "aqi_diff_1h",
        "pm_ratio",
        "aqi_ewm_24h",
    ]:
        assert column in featured.columns


def test_latest_feature_payload_contains_weather_and_features():
    hist = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=36, freq="h"),
            "city": [config.DEFAULT_CITY] * 36,
            "aqi": [80.0] * 36,
            "pm2_5": [20.0] * 36,
            "pm10": [40.0] * 36,
            "co": [300.0] * 36,
            "no": [1.0] * 36,
            "no2": [10.0] * 36,
            "o3": [20.0] * 36,
            "so2": [2.0] * 36,
            "nh3": [0.5] * 36,
        }
    )
    weather = {
        "temp": 30.0,
        "humidity": 55.0,
        "wind_speed": 3.0,
        "pressure": 1008.0,
        "visibility": 9000.0,
    }
    aqi = {
        "aqi": 90.0,
        "pm2_5": 22.0,
        "pm10": 44.0,
        "co": 320.0,
        "no": 1.0,
        "no2": 11.0,
        "o3": 18.0,
        "so2": 2.0,
        "nh3": 0.5,
    }

    latest = build_latest_feature_payload(config.DEFAULT_CITY, weather, aqi, hist)

    assert len(latest) == 1
    assert latest.iloc[0]["city"] == config.DEFAULT_CITY
    assert latest.iloc[0]["temp"] == 30.0
    assert "aqi_lag_24" in latest.columns
    assert "pm_ratio" in latest.columns


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
