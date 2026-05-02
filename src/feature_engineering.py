"""
src/feature_engineering.py
==========================
Feature engineering for the AQI time-series prediction problem.

Features created:
  ─ Time-based     : hour, day_of_week, day_of_month, month, is_weekend,
                     hour_sin, hour_cos, month_sin, month_cos (cyclic encoding)
  ─ Lag features   : AQI at t-1, t-2, t-3, t-6, t-12, t-24 hours
  ─ Rolling stats  : rolling mean/std for 3, 7, 14-day windows
  ─ Change rate    : AQI diff over 1h and 3h
  ─ Pollutant lags : pm2_5 and pm10 lag-1
  ─ Weather proxy  : wind_speed_bin (if available)

All features are added IN-PLACE to avoid unnecessary copies.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# TIME-BASED FEATURES
# ══════════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the 'timestamp' column.
    Cyclic encoding (sin/cos) is applied to hour and month so that
    the model understands that hour 23 is close to hour 0.
    """
    df = df.copy()

    if "timestamp" not in df.columns:
        logger.warning("'timestamp' column not found — skipping time features.")
        return df

    ts = pd.to_datetime(df["timestamp"])

    df["hour"]         = ts.dt.hour
    df["day_of_week"]  = ts.dt.dayofweek       # 0=Monday, 6=Sunday
    df["day_of_month"] = ts.dt.day
    df["month"]        = ts.dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = ts.dt.quarter

    # ── Cyclic encoding ──────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.debug("Time features added.")
    return df


# ══════════════════════════════════════════════════════════════
# LAG FEATURES
# ══════════════════════════════════════════════════════════════

def add_lag_features(
    df: pd.DataFrame,
    col: str = config.TARGET_COLUMN,
    lags: list[int] = config.LAG_PERIODS,
) -> pd.DataFrame:
    """
    Create lag features: value at t-n for each n in lags.
    Also adds lag features for key pollutants if present.
    """
    df = df.copy()
    n = len(df)

    for lag in lags:
        # Only add lag column if we have at least `lag` previous rows.
        # Otherwise the column would be all-NaN and later dropna() would remove
        # all rows.
        if lag < n:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            logger.debug(f"Lag feature: {col}_lag_{lag}")
        else:
            logger.debug(
                f"Skipping lag feature {col}_lag_{lag} (lag {lag} >= "
                f"data length {n})"
            )

    # Pollutant lags (lag-1 only to control dimensionality)
    for pollutant in ["pm2_5", "pm10", "no2", "o3"]:
        if pollutant in df.columns and n > 1:
            df[f"{pollutant}_lag_1"] = df[pollutant].shift(1)

    return df


# ══════════════════════════════════════════════════════════════
# ROLLING STATISTICS
# ══════════════════════════════════════════════════════════════

def add_rolling_features(
    df: pd.DataFrame,
    col: str = config.TARGET_COLUMN,
    windows: list[int] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Rolling mean and standard deviation for each window.
    min_periods=1 prevents NaN for early rows.
    Windows are specified in DAYS; since data is hourly, multiply by 24.
    """
    df = df.copy()

    for w in windows:
        w_hours = w * 24
        df[f"{col}_roll_mean_{w}d"] = (
            df[col].shift(1).rolling(window=w_hours, min_periods=1).mean()
        )
        df[f"{col}_roll_std_{w}d"] = (
            df[col].shift(1).rolling(window=w_hours, min_periods=1).std()
            .fillna(0)
        )
        logger.debug(f"Rolling features: {col} × {w}d window")

    return df


# ══════════════════════════════════════════════════════════════
# CHANGE RATE / TREND FEATURES
# ══════════════════════════════════════════════════════════════

def add_change_features(
    df: pd.DataFrame,
    col: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """
    AQI change rate over 1h and 3h windows.
    These capture sudden spikes / rapid improvements.
    """
    df = df.copy()

    df[f"{col}_diff_1h"] = df[col].diff(1)
    df[f"{col}_diff_3h"] = df[col].diff(3)
    df[f"{col}_pct_1h"] = df[col].pct_change(1).replace([np.inf, -np.inf], 0)

    return df


# ══════════════════════════════════════════════════════════════
# POLLUTANT RATIO FEATURES
# ══════════════════════════════════════════════════════════════

def add_pollutant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer cross-pollutant ratios that can be informative:
      - pm2_5 / pm10 ratio (fine vs coarse particle ratio)
      - no2 / o3 ratio (photochemical indicator)
    """
    df = df.copy()

    if "pm2_5" in df.columns and "pm10" in df.columns:
        # Avoid division by zero
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"].replace(0, np.nan)).fillna(1)

    if "no2" in df.columns and "o3" in df.columns:
        df["no2_o3_ratio"] = df["no2"] / (df["o3"].replace(0, np.nan)).fillna(1)

    return df


# ══════════════════════════════════════════════════════════════
# EXPONENTIAL WEIGHTED MEAN
# ══════════════════════════════════════════════════════════════

def add_ewm_features(
    df: pd.DataFrame,
    col: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Exponentially Weighted Mean (EWM) captures recent trends while
    down-weighting older observations — great for AQI which can
    change rapidly.
    """
    df = df.copy()

    df[f"{col}_ewm_12h"] = df[col].shift(1).ewm(span=12, adjust=False).mean()
    df[f"{col}_ewm_24h"] = df[col].shift(1).ewm(span=24, adjust=False).mean()
    df[f"{col}_ewm_72h"] = df[col].shift(1).ewm(span=72, adjust=False).mean()

    return df


# ══════════════════════════════════════════════════════════════
# MAIN COMBINATOR
# ══════════════════════════════════════════════════════════════

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline in order.
    Each step is independently testable and reversible.

    Returns a new DataFrame with all features added.
    """
    logger.info("Starting feature engineering pipeline …")

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = add_pollutant_features(df)
    df = add_ewm_features(df)

    n_features = len(df.columns)
    logger.info(f"Feature engineering complete: {n_features} total columns")
    return df


# ══════════════════════════════════════════════════════════════
# FUTURE FEATURE GENERATION (for prediction)
# ══════════════════════════════════════════════════════════════

def generate_future_features(
    history_df: pd.DataFrame,
    forecast_hours: int = config.FORECAST_HOURS,
) -> pd.DataFrame:
    """
    Generate feature rows for the next `forecast_hours` time steps.
    Uses the historical DataFrame (already feature-engineered) as context.

    Strategy:
      - Known: calendar features (deterministic)
      - Unknown: lag/rolling features → use last known values (recursive forecast)

    Returns a DataFrame with the same feature columns as training data.
    """
    logger.info(f"Generating future features for {forecast_hours} hours …")

    # Start from last known timestamp
    last_row = history_df.iloc[-1]
    last_ts  = pd.Timestamp.now()

    future_rows = []
    last_aqi    = last_row.get("aqi", 80.0)

    for i in range(1, forecast_hours + 1):
        future_ts   = last_ts + pd.Timedelta(hours=i)
        row: dict   = {}

        # ── Calendar features (deterministic) ───────────────
        row["hour"]         = future_ts.hour
        row["day_of_week"]  = future_ts.dayofweek
        row["day_of_month"] = future_ts.day
        row["month"]        = future_ts.month
        row["is_weekend"]   = int(future_ts.dayofweek >= 5)
        row["quarter"]      = future_ts.quarter

        # Cyclic
        row["hour_sin"]  = np.sin(2 * np.pi * row["hour"]  / 24)
        row["hour_cos"]  = np.cos(2 * np.pi * row["hour"]  / 24)
        row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
        row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)
        row["dow_sin"]   = np.sin(2 * np.pi * row["day_of_week"] / 7)
        row["dow_cos"]   = np.cos(2 * np.pi * row["day_of_week"] / 7)

        # ── Lag features (use last known AQI) ────────────────
        for lag in config.LAG_PERIODS:
            row[f"aqi_lag_{lag}"] = last_aqi  # Simplified: use last known

        # ── Rolling features ─────────────────────────────────
        for w in config.ROLLING_WINDOWS:
            col_key = f"aqi_roll_mean_{w}d"
            row[col_key] = history_df[col_key].iloc[-1] if col_key in history_df.columns else last_aqi
            col_key_std = f"aqi_roll_std_{w}d"
            row[col_key_std] = history_df[col_key_std].iloc[-1] if col_key_std in history_df.columns else 0.0

        # ── Change features ───────────────────────────────────
        row["aqi_diff_1h"] = 0.0
        row["aqi_diff_3h"] = 0.0
        row["aqi_pct_1h"]  = 0.0

        # ── Pollutant lags (use last known) ──────────────────
        for p in ["pm2_5", "pm10", "no2", "o3"]:
            col_key = f"{p}_lag_1"
            row[col_key] = history_df[col_key].iloc[-1] if col_key in history_df.columns else 0.0

        # ── Pollutant ratios ─────────────────────────────────
        row["pm_ratio"]      = last_row.get("pm_ratio", 0.6)
        row["no2_o3_ratio"]  = last_row.get("no2_o3_ratio", 0.5)

        # ── EWM ──────────────────────────────────────────────
        row["aqi_ewm_12h"] = history_df.get("aqi_ewm_12h", pd.Series([last_aqi])).iloc[-1]
        row["aqi_ewm_24h"] = history_df.get("aqi_ewm_24h", pd.Series([last_aqi])).iloc[-1]
        row["aqi_ewm_72h"] = history_df.get("aqi_ewm_72h", pd.Series([last_aqi])).iloc[-1]

        # Add raw pollutants (from last known)
        for col in ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]:
            if col in history_df.columns:
                row[col] = history_df[col].iloc[-1]

        row["timestamp_offset"] = i  # Hours into future
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    logger.info(f"Future feature matrix shape: {future_df.shape}")
    return future_df


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd
    sample = pd.read_csv(config.SAMPLE_DATA_FILE, parse_dates=["timestamp"])
    engineered = add_all_features(sample)
    print(f"\n✅ Features: {list(engineered.columns)}")
    print(f"   Shape: {engineered.shape}")
