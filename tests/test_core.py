import pandas as pd

import config
from api.app import app
from fastapi.testclient import TestClient
from src.feature_engineering import add_all_features
from src.fetch_data import build_latest_feature_payload
from src.preprocess import (
    FORECAST_TARGET_COLUMN,
    build_forecast_training_frame,
    get_feature_columns,
)
from src.predict import append_current_observation
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


def test_forecast_training_frame_uses_future_aqi_target():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=90, freq="h"),
            "city": [config.DEFAULT_CITY] * 90,
            "aqi": range(90),
            "pm2_5": range(90),
            "pm10": range(1, 91),
            "co": [300.0] * 90,
            "no": [1.0] * 90,
            "no2": [10.0] * 90,
            "o3": [20.0] * 90,
            "so2": [2.0] * 90,
            "nh3": [0.5] * 90,
        }
    )

    featured = add_all_features(df).dropna().reset_index(drop=True)
    forecast = build_forecast_training_frame(featured, forecast_hours=3)
    sample = forecast[forecast["timestamp_offset"] == 3].iloc[0]
    expected_ts = sample["timestamp"] + pd.Timedelta(hours=3)
    expected_aqi = featured.loc[featured["timestamp"] == expected_ts, "aqi"].iloc[0]

    assert sample[FORECAST_TARGET_COLUMN] == expected_aqi


def test_forecast_feature_columns_exclude_target_leakage():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=90, freq="h"),
            "city": [config.DEFAULT_CITY] * 90,
            "aqi": range(90),
            "pm2_5": range(90),
            "pm10": range(1, 91),
            "co": [300.0] * 90,
            "no": [1.0] * 90,
            "no2": [10.0] * 90,
            "o3": [20.0] * 90,
            "so2": [2.0] * 90,
            "nh3": [0.5] * 90,
        }
    )

    forecast = build_forecast_training_frame(
        add_all_features(df).dropna().reset_index(drop=True),
        forecast_hours=3,
    )
    feature_cols = get_feature_columns(forecast, target_col=FORECAST_TARGET_COLUMN)

    assert "timestamp_offset" in feature_cols
    assert "aqi" not in feature_cols
    assert "target_aqi" not in feature_cols
    assert "aqi_diff_1h" not in feature_cols
    assert "aqi_pct_1h" not in feature_cols


def test_append_current_observation_uses_latest_weather_and_aqi():
    hist = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "city": [config.DEFAULT_CITY] * 3,
            "aqi": [70.0, 72.0, 74.0],
            "pm2_5": [20.0, 21.0, 22.0],
            "pm10": [40.0, 41.0, 42.0],
            "temp": [25.0, 26.0, 27.0],
            "humidity": [50.0, 51.0, 52.0],
        }
    )

    appended = append_current_observation(
        hist,
        config.DEFAULT_CITY,
        {"temp": 35.0, "humidity": 65.0},
        {"aqi": 120.0, "pm2_5": 45.0, "pm10": 80.0},
    )

    latest = appended.iloc[-1]
    assert latest["aqi"] == 120.0
    assert latest["pm2_5"] == 45.0
    assert latest["temp"] == 35.0
    assert latest["humidity"] == 65.0


def test_prediction_accepts_string_timestamps_from_cached_csv(monkeypatch):
    from src.predict import predict_next_days

    class DummyModel:
        def predict(self, X):
            return [90.0] * len(X)

    monkeypatch.setattr(
        "src.predict.load_best_model",
        lambda: (
            DummyModel(),
            None,
            ["hour", "aqi_lag_1", "timestamp_offset"],
            {"model_name": "Dummy", "mae": 1.0, "rmse": 1.0, "r2": 1.0},
        ),
    )

    hist = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=36, freq="h").astype(str),
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

    result = predict_next_days(
        config.DEFAULT_CITY,
        weather,
        aqi,
        hist,
        compute_shap=False,
    )

    assert result["model_name"]
    assert len(result["daily_predictions"]) == 3


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
