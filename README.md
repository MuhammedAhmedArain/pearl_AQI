# Pearls AQI Predictor

End-to-end AQI forecasting system for the next 3 days. The project includes an
hourly feature pipeline, historical backfill, multi-model training, Hopsworks
Feature Store / Model Registry integration, a FastAPI backend, and a Streamlit
dashboard with SHAP explanations and AQI alerts.

## What It Implements

| Internship requirement | Implementation |
| --- | --- |
| Raw weather and pollutant ingestion | `src/fetch_data.py` uses OpenWeatherMap and cache fallback |
| Feature engineering | `src/feature_engineering.py` creates time, lag, rolling, change, ratio, and EWM features |
| Feature Store | `src/feature_store.py` pushes/loads features from Hopsworks when configured |
| Historical backfill | `src/backfill.py` generates 90-day training history and can push to Hopsworks |
| Training pipeline | `src/train.py` trains sklearn, XGBoost, MLP, and TensorFlow/Keras models |
| Model Registry | Best model, scaler, feature schema, and metadata can be saved to Hopsworks |
| CI/CD automation | GitHub Actions run tests, hourly feature ingestion, daily training, and manual backfill |
| Web app | FastAPI backend in `api/app.py`; Streamlit dashboard in `app.py` |
| Explainability | SHAP feature importance in `src/predict.py` and the dashboard |
| Alerts | Current and forecast AQI alerts when AQI exceeds the configured threshold |

## Project Structure

```text
aqi-predictor/
  api/app.py                         FastAPI prediction API
  app.py                             Streamlit dashboard
  config.py                          Central configuration
  src/fetch_data.py                  OpenWeather ingestion and Feature Store payloads
  src/feature_engineering.py         Feature generation
  src/preprocess.py                  Cleaning, splitting, scaling
  src/train.py                       Multi-model training and registry save
  src/predict.py                     Forecasting and SHAP explanations
  src/feature_store.py               Hopsworks Feature Store / Model Registry
  src/backfill.py                    Historical data backfill
  tests/                             Core unit tests
  .github/workflows/                 CI, hourly feature, daily training, backfill workflows
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your own keys. Do not commit real credentials.

## Run Locally

```bash
python src/backfill.py --days 90 --no-push
python src/train.py --no-tune
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
streamlit run app.py
```

Open the dashboard at http://localhost:8501.

## Hopsworks Mode

Set these variables in `.env` and GitHub repository secrets:

```text
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
HOPSWORKS_HOST
```

Then run:

```bash
python src/backfill.py --days 90
python src/fetch_data.py --city Sukkur --push-to-store
python src/train.py --no-tune
```

When Hopsworks is configured, training loads historical data from the Feature
Store and saves the best model to the Model Registry. If Hopsworks is not
configured, the project falls back to local CSV/model artifacts for development.

## GitHub Actions

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `ci-tests.yml` | Push / PR / manual | Lint and run tests |
| `feature-pipeline.yml` | Hourly / manual | Fetch latest AQI/weather and push features |
| `training-pipeline.yml` | Daily / manual | Retrain and publish model metrics |
| `backfill.yml` | Manual | Seed historical features |

## API Endpoints

| Endpoint | Description |
| --- | --- |
| `/health` | API and model health |
| `/cities` | Supported city list |
| `/predict?city=Sukkur&shap=true` | 3-day forecast with optional SHAP |
| `/model/info` | Current model metadata |
| `/model/comparison` | Latest model leaderboard |

## Notes

- Current default city is `Sukkur` in `config.py`.
- FastAPI is used because the assignment allows Flask/FastAPI.
- TensorFlow is included for the GitHub Actions Python 3.11 runtime. Local
  Python 3.13 environments gracefully fall back to a sklearn MLP candidate if
  TensorFlow is unavailable.
