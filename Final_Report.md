# Pearls AQI Predictor - Final Submission Report

## Overview

Pearls AQI Predictor is an end-to-end, serverless-friendly machine learning
system that forecasts Air Quality Index (AQI) for the next 72 hours. It follows
the internship specification by implementing feature ingestion, feature
engineering, historical backfill, automated training, model registry support,
and a web dashboard for real-time predictions.

## 1. Feature Pipeline and Feature Store

The feature pipeline in `src/fetch_data.py` fetches current weather and
pollutant data from OpenWeatherMap. It combines the current observation with
historical AQI context, computes model-ready features, and can push the latest
complete row into Hopsworks Feature Store.

Engineered features include:

- Time features: hour, day of week, day of month, month, weekend flag, quarter
- Cyclical encodings: hour, month, and day-of-week sine/cosine
- Lag features: AQI and pollutant lags
- Rolling statistics: 3-day, 7-day, and 14-day rolling means/stds
- Change features: 1-hour and 3-hour AQI deltas and percentage change
- Derived ratios: PM2.5/PM10 and NO2/O3
- Exponential weighted means: 12-hour, 24-hour, and 72-hour AQI trends

## 2. Historical Backfill

The backfill pipeline in `src/backfill.py` generates 90 days of hourly
historical AQI, pollutant, and weather context for supported cities. It applies
the same feature engineering pipeline used by live ingestion and can seed the
Hopsworks Feature Store. When Hopsworks is not configured, it saves a local CSV
for development and testing.

## 3. Training Pipeline and Model Registry

The training pipeline in `src/train.py` loads historical features from
Hopsworks when credentials are configured, otherwise it falls back to local CSV
data. It evaluates multiple model families:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Scikit-learn MLP neural network
- TensorFlow/Keras neural network when TensorFlow is available

Models are evaluated with RMSE, MAE, and R2. The best model is selected by
lowest RMSE. The selected model, scaler, feature schema, and metadata are saved
locally and can also be published to the Hopsworks Model Registry.

## 4. Automation

The project uses GitHub Actions for automated CI/CD:

- `ci-tests.yml`: runs linting and tests on push and pull request
- `feature-pipeline.yml`: runs the feature pipeline every hour
- `training-pipeline.yml`: retrains models daily at 02:00 UTC
- `backfill.yml`: manually seeds historical data

Repository secrets should provide OpenWeatherMap and Hopsworks credentials.

## 5. Web Application

The web layer is split into:

- `api/app.py`: FastAPI backend exposing health, prediction, city, model info,
  and model comparison endpoints
- `app.py`: Streamlit dashboard showing current AQI, 3-day forecasts, trend
  charts, SHAP explanations, model metrics, and alert banners

The assignment allows Flask/FastAPI, so FastAPI is used for the backend.

## 6. Advanced Analytics

The project includes:

- EDA notebook in `notebooks/EDA.ipynb`
- SHAP feature importance in `src/predict.py`
- Dashboard SHAP bar chart
- Hazardous AQI alerts when AQI exceeds the configured threshold

## Current Limitations

- The default configured city is `Sukkur`.
- Hopsworks must be configured through environment variables and GitHub secrets
  for the fully serverless path.
- Local fallback data is included so the app remains demoable without external
  services.

## Conclusion

The project now satisfies the core internship requirements: feature pipeline,
historical backfill, feature store integration, model training, model registry
support, automation, dashboard, explainability, alerts, and final reporting.
