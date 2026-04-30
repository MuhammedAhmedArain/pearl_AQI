# 🌬️ Pearls AQI Predictor

> **Production-ready end-to-end ML system** that predicts Air Quality Index (AQI) for the next 3 days for any city, with a Streamlit dashboard, FastAPI backend, SHAP explainability, and GitHub Actions automation.

---

## 📁 Project Structure

```
aqi-predictor/
├── data/
│   ├── sample_aqi_data.csv       ← Bundled example dataset (720 h)
│   ├── raw_aqi_data.csv          ← Fetched live data (auto-generated)
│   └── processed_aqi_data.csv    ← After feature engineering
├── models/
│   ├── best_model.pkl            ← Best trained model (joblib)
│   ├── scaler.pkl                ← RobustScaler
│   ├── feature_names.json        ← Feature list used at training
│   ├── model_metadata.json       ← Metrics + training info
│   └── model_comparison.csv      ← All model results table
├── src/
│   ├── fetch_data.py             ← OpenWeatherMap API fetching + caching
│   ├── preprocess.py             ← Cleaning, imputation, splitting, scaling
│   ├── feature_engineering.py   ← Lag/rolling/time/pollutant features
│   ├── train.py                  ← Multi-model training + best model selection
│   ├── predict.py                ← Forecasting + SHAP explainability
│   └── utils.py                  ← Logger, decorators, helpers
├── api/
│   └── app.py                    ← FastAPI REST API
├── app.py                        ← Streamlit dashboard
├── config.py                     ← All configuration (no hardcoded values)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── .github/workflows/main.yml    ← GitHub Actions automation
```

---

## 🚀 Quick Start (Local)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourname/pearls-aqi-predictor.git
cd pearls-aqi-predictor

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your OpenWeatherMap API key
# Get a free key at: https://openweathermap.org/api
```

### 3. Fetch Data

```bash
python src/fetch_data.py --city Karachi
# Falls back to sample_aqi_data.csv if API key not set
```

### 4. Train Models

```bash
# Full training with hyperparameter tuning (recommended, ~5-10 min)
python src/train.py

# Fast training without tuning (~1 min)
python src/train.py --no-tune
```

**Example output:**
```
======================================================================
           MODEL COMPARISON — Pearls AQI Predictor
======================================================================
Model                           MAE      RMSE        R²
----------------------------------------------------------------------
XGBoost                      4.2310    6.1450    0.9412
Random Forest                5.1230    7.8920    0.9201
Gradient Boosting            5.6780    8.4500    0.9134
Ridge Regression            11.2340   15.6700    0.7820
Linear Regression           11.9800   16.3200    0.7710
----------------------------------------------------------------------

🏆 BEST MODEL: XGBoost | RMSE=6.1450 | MAE=4.2310 | R²=0.9412
```

### 5. Run Prediction (CLI)

```bash
python src/predict.py --city Karachi

# Output:
# 📍 City        : Karachi
# 🌬  Current AQI : 124.5 (Unhealthy for Sensitive Groups)
# 📅 3-Day Forecast:
#   Day 1 (2026-05-01): AQI 118.3 — Unhealthy for Sensitive Groups
#   Day 2 (2026-05-02): AQI 132.7 — Unhealthy for Sensitive Groups
#   Day 3 (2026-05-03): AQI 98.1  — Moderate
```

### 6. Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API overview |
| `/health` | GET | Health check |
| `/predict?city=Karachi` | GET | 3-day AQI prediction |
| `/predict?city=Karachi&shap=true` | GET | With SHAP feature importance |
| `/cities` | GET | List supported cities |
| `/model/info` | GET | Current model metadata |
| `/model/comparison` | GET | All model comparison results |
| `/docs` | GET | Interactive OpenAPI docs |

**Example API Response (`/predict?city=Karachi`):**
```json
{
  "city": "Karachi",
  "current_aqi": 124.5,
  "current_category": "Unhealthy for Sensitive Groups",
  "current_alert": false,
  "current_weather": {
    "temp": 34.2,
    "humidity": 62,
    "wind_speed": 4.1,
    "pressure": 1008
  },
  "current_pollutants": {
    "pm2_5": 34.8,
    "pm10":  52.3,
    "no2":   22.4,
    "o3":    27.5,
    "co":    312.1
  },
  "daily_predictions": [
    {"day": 1, "date": "2026-05-01", "aqi_mean": 118.3, "category": "Unhealthy for Sensitive Groups", "alert": false},
    {"day": 2, "date": "2026-05-02", "aqi_mean": 132.7, "category": "Unhealthy for Sensitive Groups", "alert": false},
    {"day": 3, "date": "2026-05-03", "aqi_mean": 98.1,  "category": "Moderate",                      "alert": false}
  ],
  "model_name": "XGBoost",
  "model_metrics": {"mae": 4.23, "rmse": 6.14, "r2": 0.941}
}
```

### 7. Launch the Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🤖 Models Trained

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Baseline | No regularization |
| Ridge Regression | Regularized | L2 penalty, tuned alpha |
| Random Forest | Ensemble | 200 trees, tuned depth |
| Gradient Boosting | Boosting | sklearn GBM |
| **XGBoost** | **Boosting** | **Typically best performer** |

**Selection criterion:** Lowest RMSE on hold-out test set (temporal split).

---

## 🔧 Features Engineered

| Category | Features |
|----------|----------|
| Time | hour, day_of_week, month, is_weekend, quarter |
| Cyclic | hour_sin/cos, month_sin/cos, dow_sin/cos |
| Lag | aqi_lag_1/2/3/6/12/24h, pm2_5_lag_1, pm10_lag_1 |
| Rolling | aqi_roll_mean/std for 3d, 7d, 14d windows |
| Change | aqi_diff_1h, aqi_diff_3h, aqi_pct_1h |
| EWM | aqi_ewm_12h/24h/72h |
| Ratios | pm_ratio, no2_o3_ratio |

---

## ⚙️ GitHub Actions Automation

| Job | Trigger | Description |
|-----|---------|-------------|
| `test` | Push / PR | Lint + unit tests |
| `fetch-data` | Every hour | Fetch live data for all cities |
| `retrain` | Daily 2 AM UTC | Retrain models, commit new best model |

Add your `OPENWEATHER_API_KEY` to GitHub Secrets for automation.

---

## 📊 AQI Scale (US EPA)

| Range | Category | Health Impact |
|-------|----------|---------------|
| 0–50 | 😊 Good | Air quality satisfactory |
| 51–100 | 😐 Moderate | Acceptable for most |
| 101–150 | 😷 Unhealthy for Sensitive Groups | Concern for sensitive groups |
| 151–200 | 🤢 Unhealthy | Everyone may experience effects |
| 201–300 | 🚨 Very Unhealthy | Health alert |
| 301–500 | ☠️ Hazardous | Emergency conditions |

---

## 🧑‍💻 Development

```bash
# Format code
black src/ api/ config.py app.py

# Lint
flake8 src/ api/ --max-line-length=100

# Run tests
pytest tests/ -v
```

---

## 📄 License

MIT License — see `LICENSE` for details.

---

*Built with ❤️ using Python, Scikit-learn, XGBoost, FastAPI, and Streamlit*
