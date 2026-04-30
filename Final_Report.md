# Pearls AQI Predictor - Final Submission Report

## Overview
The Pearls AQI Predictor is a 100% serverless, end-to-end Machine Learning pipeline designed to forecast the Air Quality Index (AQI) for the next 72 hours. Built strictly following the 10 Pearls internship guidelines, this system automates data fetching, feature engineering, model training, and prediction serving, wrapped in a decoupled frontend-backend architecture.

## 1. Feature Pipeline & Feature Store
The system automatically fetches live meteorological and pollutant data from the **OpenWeather API**. 
- **Feature Engineering**: The raw data undergoes extensive preprocessing. We extract time-based cyclical features (sine/cosine of hour and month) to capture seasonality, calculate rolling statistics (12h, 24h, 72h moving averages), and derive advanced metrics like the `pm2_5` to `pm10` ratio.
- **Feature Store**: Instead of saving to local CSV files, all engineered features are pushed securely to the **Hopsworks Feature Store**. This provides a centralized "single source of truth" that allows decoupling the data pipeline from the training pipeline. A backfill script successfully seeded 90 days of historical data into the store.

## 2. Training Pipeline & Model Registry
The training pipeline pulls historical training batches directly from the Hopsworks Feature Store.
- **Model Experimentation**: The pipeline automatically cross-validates and tests multiple algorithms simultaneously:
  - *Statistical & Tree Models*: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost.
  - *Deep Learning*: A Multi-Layer Perceptron (MLP Neural Network) using Scikit-Learn's `MLPRegressor` architecture.
- **Evaluation**: Each model is evaluated using RMSE, MAE, and R² scores. 
- **Model Registry**: The best-performing model (along with its feature schema and data scalers) is dynamically pushed to the **Hopsworks Model Registry**.

## 3. Automation (CI/CD)
The entire workflow is automated via **GitHub Actions** (`.github/workflows/main.yml`):
- **Hourly Runs**: A cron job triggers the Feature Pipeline every hour to fetch the latest weather/pollution data and push it to Hopsworks.
- **Daily Runs**: A separate cron job triggers the Training Pipeline every night to retrain the models on the newly aggregated data, ensuring the system never suffers from data drift.

## 4. Web Application (FastAPI + Streamlit)
The frontend architecture consists of two decoupled layers:
- **FastAPI Backend**: Located in `api/app.py`, this acts as the prediction engine. It downloads the latest model from the Model Registry, pulls the latest context from the Feature Store, and serves a REST API endpoint `/predict`.
- **Streamlit Frontend**: Located in `app.py`, this serves as the interactive dashboard. It consumes the FastAPI endpoints to display 3-day AQI forecasts.
- **Explainability (SHAP)**: The dashboard integrates SHAP (SHapley Additive exPlanations) to dynamically generate feature importance charts, explaining *why* the model made its prediction (e.g., "High wind speed reduced AQI").
- **Alerts**: The dashboard automatically triggers hazardous warning alerts if the predicted AQI crosses the unhealthy threshold (AQI > 150).

## 5. Exploratory Data Analysis (EDA)
A Jupyter Notebook (`notebooks/EDA.ipynb`) was created to perform initial exploratory data analysis. The notebook evaluates AQI distributions, charts the 24-hour rolling averages, and builds correlation heatmaps between key meteorological factors (like temperature and wind speed) against pollutants (like PM2.5 and Ozone).

## Conclusion
The project successfully implements a scalable, MLOps-driven architecture that is entirely serverless. By utilizing Hopsworks for state management and GitHub Actions for compute scheduling, the Pearls AQI Predictor requires zero manual intervention to maintain its forecasting accuracy.
