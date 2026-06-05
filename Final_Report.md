# Pearls AQI Predictor - Final Submission Report

## Overview

Pearls AQI Predictor forecasts Air Quality Index (AQI) for the next 72 hours
and displays the result as a 3-day dashboard forecast. The production automation
path uses OpenWeather data ingestion, engineered features, Hopsworks Feature
Store, multi-model training, Hopsworks Model Registry, GitHub Actions, and a
Streamlit dashboard.

Live Streamlit deployment:
[https://pearlaqi.streamlit.app/](https://pearlaqi.streamlit.app/)

Note for evaluators: Streamlit Community Cloud may put the application to sleep
after inactivity. If a wake-up prompt appears, click the button and wait briefly
for the app to restart.

## Compliance Checklist

| Requirement | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Predict AQI for next 3 days / 72 hours | Complete | `config.py`, `src/predict.py` | `FORECAST_DAYS=3`, `FORECAST_HOURS=72`; `predict_next_days()` generates hourly predictions and aggregates them into 3 daily forecasts. |
| Use last 3 months / 90 days for training | Complete | `src/preprocess.py`, `src/feature_store.py` | `load_from_feature_store()` calls `get_training_data(days=90)`. |
| Do not use CSV/local files as production data source | Complete for scheduled production | `.github/workflows/*.yml`, `config.py`, `src/preprocess.py` | GitHub Actions set `REQUIRE_FEATURE_STORE=true`, which forces Feature Store reads/writes and fails instead of falling back. CSV/local files remain development/demo fallback only. |
| Use centralized Feature Store | Complete | `src/feature_store.py`, `.github/workflows/feature-pipeline.yml` | Hopsworks Feature Group `aqi_features` stores engineered feature rows. |
| Feature pipeline fetches new data hourly | Complete | `.github/workflows/feature-pipeline.yml` | Cron schedule `0 * * * *`; runs `python src/fetch_data.py --city "Sukkur" --push-to-store`. |
| Training pipeline runs daily | Complete | `.github/workflows/training-pipeline.yml` | Cron schedule `0 2 * * *`; runs `python src/train.py --no-tune`. |
| Training data retrieved from Feature Store | Complete in production | `src/train.py`, `src/preprocess.py` | Daily workflow sets `REQUIRE_FEATURE_STORE=true`; training source becomes `feature_store`. |
| Multiple ML models trained and compared | Complete | `src/train.py` | Trains Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, MLP, and TensorFlow/fallback neural model. |
| Evaluation uses RMSE, MAE, R2 | Complete | `src/train.py` | `evaluate_model()` computes MAE, RMSE, and R2; comparison CSV stores all three. |
| Best model selected automatically | Complete | `src/train.py` | Results are sorted by RMSE, then MAE; `select_and_save_best_model()` uses the top result. |
| Best model stored in Model Registry | Complete in production | `src/train.py`, `src/feature_store.py` | With Hopsworks configured, `save_model_to_registry()` stores model, scaler, feature names, and metadata. In strict production mode, registry save failure raises an error. |
| Dashboard shows real-time AQI, forecast, charts, alerts, model info | Complete | `app.py`, `api/app.py` | Dashboard displays current AQI, 3-day forecast cards, charts, alert banners, model metrics, and model comparison. |
| GitHub Actions or Airflow automation exists | Complete | `.github/workflows/` | CI tests, hourly feature ingestion, daily training, and manual backfill workflows exist. |

## Production Data and Fallback Policy

Hopsworks Feature Store is the production source of truth for scheduled
ingestion and training. The hourly feature pipeline writes the newest engineered
observation to Hopsworks. The daily training pipeline reads the last 90 days of
training features from Hopsworks.

CSV and local model artifacts are retained only for local development,
debugging, and Streamlit demo resilience. They are not the production source for
the scheduled GitHub Actions pipelines. This is enforced by
`REQUIRE_FEATURE_STORE=true` in the hourly feature and daily training workflows.
When that flag is enabled, Feature Store read/write failures and Model Registry
save/load failures are fatal instead of silently falling back.

## Feature Pipeline

`src/fetch_data.py` fetches current weather and pollutant data from
OpenWeatherMap, combines it with recent AQI history, computes the latest
model-ready feature row, and pushes it to Hopsworks through
`src/feature_store.py`.

The scheduled production feature pipeline is defined in
`.github/workflows/feature-pipeline.yml` and runs every hour. It installs the
Hopsworks Python/Kafka dependencies, verifies the imports, sets
`REQUIRE_FEATURE_STORE=true`, and runs:

```bash
python src/fetch_data.py --city "Sukkur" --push-to-store
```

## Historical Backfill

`src/backfill.py` generates 90 days of hourly AQI, pollutant, and weather
context and applies the same feature engineering logic used by live ingestion.
The manual backfill workflow can seed Hopsworks before automated daily training
is evaluated.

Local CSV output from backfill is a development fallback and audit artifact; it
is not the production data source when strict mode is enabled.

## Training Pipeline and Model Registry

`src/train.py` runs the end-to-end training pipeline. In production, the daily
GitHub Actions workflow sets `REQUIRE_FEATURE_STORE=true`, causing
`run_training_pipeline()` to force `source="feature_store"`. Training therefore
retrieves the last 90 days from Hopsworks through `get_training_data(days=90)`.

The pipeline trains and compares:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Scikit-learn MLP neural network
- TensorFlow/Keras neural network, with sklearn MLP fallback when TensorFlow is
  unavailable

Each model is evaluated using MAE, RMSE, and R2. The best model is selected
automatically by lowest RMSE, with MAE as the secondary sort key. The selected
model, scaler, feature schema, and metadata are saved to the Hopsworks Model
Registry in production. Because strict mode is enabled in the daily workflow,
registry save failure fails the training run.

## Prediction and Dashboard

`src/predict.py` loads the registered Hopsworks model first when Hopsworks
credentials are configured. In strict mode, a Model Registry load failure is
fatal. Outside strict mode, local artifacts are allowed as a development/demo
fallback.

The FastAPI backend in `api/app.py` uses Hopsworks latest feature rows for
inference when Hopsworks credentials are configured. If strict mode is disabled,
it can fall back to live API fetches so the app remains demoable during local
development.

The Streamlit dashboard in `app.py` shows:

- Current AQI and category
- 3-day forecast
- Forecast trend charts
- Hazardous/unhealthy AQI alerts
- Model name, training timestamp, RMSE, MAE, and R2
- Optional SHAP feature importance
- Model comparison leaderboard

The deployed Streamlit app uses the same codebase. Whether the deployed app
uses Hopsworks/Model Registry at runtime depends on the Streamlit Cloud secrets
configured for that deployment. To make the live deployment strictly production
backed, set `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `HOPSWORKS_HOST`, and
`REQUIRE_FEATURE_STORE=true` in Streamlit Cloud secrets.

## Automation

The project uses GitHub Actions:

- `.github/workflows/ci-tests.yml`: linting and tests on push, pull request,
  and manual dispatch
- `.github/workflows/feature-pipeline.yml`: hourly feature ingestion into
  Hopsworks
- `.github/workflows/training-pipeline.yml`: daily retraining from Hopsworks and
  Model Registry publishing
- `.github/workflows/backfill.yml`: manual 90-day historical Feature Store seed

## Advanced Analytics

The project includes:

- EDA notebook in `notebooks/EDA.ipynb`
- SHAP feature importance in `src/predict.py`
- Dashboard SHAP bar chart, enabled on demand
- AQI alerts when current or forecast AQI exceeds configured thresholds

## Audit Conclusion

Project fully satisfies the internship requirements for the scheduled
production pipeline. The earlier report wording needed correction because it
used weak phrases such as "can push" and "can be published" without clearly
separating production strict mode from local/demo fallback behavior.

The main remaining operational requirement is configuration, not code: the
Hopsworks and OpenWeather secrets must be present in GitHub Actions, and the
Streamlit Cloud deployment should also define the Hopsworks secrets plus
`REQUIRE_FEATURE_STORE=true` if evaluators require the live app itself to fail
rather than use local/demo fallback.

## Exact Fixes Needed

No project code fix is required for the scheduled production requirements based
on the audited files. The smallest deployment/configuration fix, if not already
done, is to add these secrets to Streamlit Cloud:

```text
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT=pearlsAQI
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
REQUIRE_FEATURE_STORE=true
```

Without those Streamlit Cloud secrets, the deployed dashboard may still run in
demo-resilient mode using live fetch/local artifacts, while GitHub Actions
remain the strict production MLOps path.
