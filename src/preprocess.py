"""
src/preprocess.py
=================
Data cleaning and preprocessing pipeline.

Steps:
  1. Load data: Hopsworks Feature Store (primary) or CSV (fallback)
  2. Parse + validate timestamps
  3. Handle missing values (forward-fill, interpolation, median imputation)
  4. Remove outliers using IQR method
  5. Encode categorical features
  6. Scale numerical features (fit on train, transform all)
  7. Split into train / test sets
  8. Persist scaler + feature names locally

Data source priority:
  1. Hopsworks Feature Store (if HOPSWORKS_API_KEY is set)
  2. data/raw_aqi_data.csv   (if it exists)
  3. data/sample_aqi_data.csv (built-in fallback)
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.utils import get_logger, timer  # noqa: E402

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════

def load_from_feature_store() -> pd.DataFrame:
    """
    Fetch training data from Hopsworks Feature Store (last 90 days).
    Raises FeatureStoreNotConfigured if not configured.
    """
    from src.feature_store import get_training_data
    return get_training_data(days=90)


def load_raw_data(
    path: Path = config.RAW_DATA_FILE, source: str = "auto"
) -> pd.DataFrame:
    """
    Load AQI data with source priority:
      source="auto":
        1. Hopsworks Feature Store  (if USE_FEATURE_STORE=True)
        2. raw CSV file             (if it exists)
        3. sample CSV file          (built-in fallback)
      source="feature_store" -- force Feature Store
      source="csv"           -- force CSV
    """
    # -- Feature Store path -------------------------------------------
    if (
        source == "feature_store" or
        (source == "auto" and config.USE_FEATURE_STORE)
    ):
        try:
            df = load_from_feature_store()
            logger.info(f"Loaded {len(df)} rows from Hopsworks Feature Store.")
            return df
        except Exception as exc:
            if source == "feature_store":
                raise
            logger.warning(
                f"Feature Store unavailable ({exc}). Falling back to CSV."
            )

    # -- CSV path -----------------------------------------------------
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info(f"Loaded raw data: {len(df)} rows from {path}")
    else:
        logger.warning(f"{path} not found - loading sample data instead.")
        df = pd.read_csv(config.SAMPLE_DATA_FILE, parse_dates=["timestamp"])

    return df


# ══════════════════════════════════════════════════════════════
# CLEANING
# ══════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core cleaning steps:
      - Drop exact duplicates
      - Ensure timestamp is datetime and sorted
      - Drop rows where target (aqi) is NaN
      - Clip AQI to valid 0-500 range
    """
    original_len = len(df)

    # Sort and deduplicate
    subset = (
        ["timestamp", "city"] if "city" in df.columns else ["timestamp"]
    )
    df = df.drop_duplicates(subset=subset)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to only supported cities
    if "city" in df.columns:
        df = df[df["city"].isin(config.SUPPORTED_CITIES)]

    # Drop rows without target
    df = df.dropna(subset=[config.TARGET_COLUMN])

    # Clip AQI to valid range
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].clip(0, 500)

    logger.info(
        f"Cleaning: {original_len} → {len(df)} rows "
        f"(removed {original_len - len(df)})"
    )

    if len(df) == 0:
        raise ValueError(
            f"No data remaining after filtering for cities "
            f"{config.SUPPORTED_CITIES}. Ensure the Feature Store or sample "
            f"CSV contains data for the target city. Run "
            f"`python src/backfill.py` to populate the Feature Store."
        )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-strategy missing value imputation:
      1. Forward-fill (for time-series continuity)
      2. Backward-fill (catch leading NaNs)
      3. Median imputation (for any remaining NaNs in numeric cols)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Step 1 & 2: time-series fill
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Step 3: median imputation fallback
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.debug(f"Median imputed '{col}' with {median_val:.3f}")

    remaining = df.isna().sum().sum()
    logger.info(
        f"Missing value handling complete. Remaining NaNs: {remaining}"
    )
    return df


def remove_outliers(
    df: pd.DataFrame, columns: list[str] = None
) -> pd.DataFrame:
    """
    Remove outliers using IQR method (1.5× IQR fence) for specified columns.
    Default columns: aqi, pm2_5, pm10.
    Outliers are capped (Winsorized), not dropped, to preserve
    time-series structure.
    """
    if columns is None:
        columns = [c for c in ["aqi", "pm2_5", "pm10"] if c in df.columns]

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        clipped = (df[col] < lo) | (df[col] > hi)
        df[col] = df[col].clip(lo, hi)
        logger.debug(
            f"Outlier cap '{col}': [{lo:.2f}, {hi:.2f}], "
            f"{clipped.sum()} values clipped"
        )

    return df


# ══════════════════════════════════════════════════════════════
# SCALING
# ══════════════════════════════════════════════════════════════

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Fit RobustScaler on training data only, then transform both sets.
    RobustScaler is resistant to outliers (uses median + IQR).

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = RobustScaler()
    X_train_sc = X_train.copy()
    X_test_sc = X_test.copy()

    X_train_sc[feature_columns] = scaler.fit_transform(
        X_train[feature_columns]
    )
    X_test_sc[feature_columns] = scaler.transform(X_test[feature_columns])

    # Persist scaler
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Scaler saved → {config.SCALER_PATH}")

    return X_train_sc, X_test_sc, scaler


# ══════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_data(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    test_size: float = config.TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Temporal train/test split (NO shuffle) to respect time ordering.
    The last `test_size` fraction of data is used as the test set.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    feature_cols = [
        c for c in df.columns
        if c not in [target_col, "timestamp", "city"]
    ]

    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        f"Train/test split: {len(X_train)} train rows, {len(X_test)} "
        f"test rows (split at index {split_idx})"
    )
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════

@timer
def run_preprocessing_pipeline(
    raw_path: Path = config.RAW_DATA_FILE,
    save_processed: bool = True,
    source: str = "auto",
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], RobustScaler
]:
    """
    End-to-end preprocessing pipeline:
      load (FS or CSV) -> clean -> impute -> outlier-cap -> engineer -> split

    Args:
        raw_path:       Path to raw CSV (used when FS unavailable)
        save_processed: If True, save processed DataFrame to CSV for debugging
        source:         "auto" | "feature_store" | "csv"
                        "auto" tries Feature Store first, falls back to CSV.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Import here to avoid circular imports with feature_engineering
    from src.feature_engineering import add_all_features

    # 1. Load (Feature Store -> CSV fallback)
    df = load_raw_data(raw_path, source=source)

    # 2. Clean
    df = clean_data(df)

    # 3. Impute
    df = handle_missing_values(df)

    # 4. Cap outliers
    df = remove_outliers(df)

    # 5. Feature engineering (skip if already engineered by Feature Store)
    if "hour" not in df.columns:
        df = add_all_features(df)

    # 6. Drop remaining NaNs created by lag/rolling (head of series)
    df = df.dropna().reset_index(drop=True)

    if save_processed:
        df.to_csv(config.PROCESSED_DATA_FILE, index=False)
        logger.info(f"Processed data saved -> {config.PROCESSED_DATA_FILE}")

    # 7. Train/test split
    X_train, X_test, y_train, y_test = split_data(df)

    # 8. Identify numeric feature columns (exclude target/meta)
    feature_cols = [
        c for c in X_train.columns
        if c not in [config.TARGET_COLUMN, "timestamp", "city"]
    ]

    # 9. Scale
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test, feature_cols
    )

    # 10. Persist feature names (needed for future prediction)
    config.FEATURE_NAMES_PATH.write_text(json.dumps(feature_cols))
    logger.info(
        f"Feature names saved -> {config.FEATURE_NAMES_PATH} "
        f"({len(feature_cols)} features)"
    )

    return X_train_sc, X_test_sc, y_train, y_test, feature_cols, scaler


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, sc = run_preprocessing_pipeline()
    print("\n✅ Preprocessing complete")
    print(f"   Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"   Features ({len(feats)}): {feats[:5]} …")
