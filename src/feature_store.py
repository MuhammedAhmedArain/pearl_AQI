"""
src/feature_store.py
====================
Hopsworks Feature Store + Model Registry integration layer.

Responsibilities:
  1. Connect to Hopsworks (project: pearlsAQI)
  2. Create / get the AQI Feature Group (table)
  3. Push engineered features to the Feature Store
  4. Fetch training data (last N days) from the Feature Store
  5. Fetch latest rows for inference
  6. Save best model + scaler + metadata to Model Registry
  7. Load best model from Model Registry

Design:
  - All functions check config.USE_FEATURE_STORE before connecting.
  - If Hopsworks is not configured, they raise FeatureStoreNotConfigured
    so callers can catch it and fall back to local files gracefully.
  - Lazy connection: Hopsworks client is only created on first use.
"""

import json
import tempfile
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════════════════════════

class FeatureStoreNotConfigured(RuntimeError):
    """Raised when HOPSWORKS_API_KEY is not set or is a placeholder."""
    pass


# ══════════════════════════════════════════════════════════════
# CONNECTION (lazy singleton)
# ══════════════════════════════════════════════════════════════

_project = None
_fs = None
_mr = None


def _connect() -> tuple:
    """
    Connect to Hopsworks and return (project, feature_store, model_registry).
    Uses a module-level singleton so we only login once per process.
    """
    global _project, _fs, _mr

    if not config.USE_FEATURE_STORE:
        raise FeatureStoreNotConfigured(
            "HOPSWORKS_API_KEY is not set. "
            "Add it to your .env file to use the Feature Store."
        )

    if _project is None:
        try:
            import hopsworks
            logger.info(
                f"Connecting to Hopsworks project "
                f"'{config.HOPSWORKS_PROJECT}' ..."
            )
            _project = hopsworks.login(
                host=config.HOPSWORKS_HOST,
                project=config.HOPSWORKS_PROJECT,
                api_key_value=config.HOPSWORKS_API_KEY,
                cert_folder=tempfile.gettempdir(),
            )
            _fs = _project.get_feature_store()
            _mr = _project.get_model_registry()
            logger.info("Connected to Hopsworks successfully.")
        except Exception as exc:
            _project = _fs = _mr = None
            raise RuntimeError(f"Hopsworks connection failed: {exc}") from exc

    return _project, _fs, _mr


def get_feature_store():
    """Return connected Hopsworks feature store handle."""
    _, fs, _ = _connect()
    return fs


def get_model_registry():
    """Return connected Hopsworks model registry handle."""
    _, _, mr = _connect()
    return mr


# ══════════════════════════════════════════════════════════════
# FEATURE GROUP SCHEMA
# ══════════════════════════════════════════════════════════════

# These are the columns that MUST exist in the DataFrame pushed to the FG.
# Any extra engineered features are also accepted (schema is flexible).
REQUIRED_COLUMNS = [
    "city",
    "timestamp",
    "aqi",
    # raw weather
    "temp", "humidity", "wind_speed", "pressure", "visibility",
    # raw pollutants
    "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3",
    # time features
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
]


def _prepare_df_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame is ready for Hopsworks ingestion:
      - 'timestamp' must be datetime (UTC-normalised)
      - All column names must be lowercase with underscores
      - No timezone info (Hopsworks expects naive UTC timestamps)
      - Primary key columns must not be null
    """
    df = df.copy()

    # Normalise column names
    df.columns = [
        c.lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Strip timezone info (make naive UTC)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = (
                df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            )

    # Drop rows missing primary keys
    pk_cols = [c for c in ["city", "timestamp"] if c in df.columns]
    df = df.dropna(subset=pk_cols)

    # Replace inf values
    df = df.replace([float("inf"), float("-inf")], np.nan)

    # Fill remaining NaNs in numeric cols with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    return df


# ══════════════════════════════════════════════════════════════
# FEATURE GROUP — GET OR CREATE
# ══════════════════════════════════════════════════════════════

def get_or_create_feature_group():
    """
    Get the AQI Feature Group from Hopsworks, creating it if it doesn't exist.

    Feature Group:
      Name   : aqi_features
      Version: 1
      PK     : [city, timestamp]
      EventTime: timestamp
    """
    fs = get_feature_store()

    try:
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        if fg is not None:
            logger.info(
                f"Found existing feature group: {config.FEATURE_GROUP_NAME} "
                f"v{config.FEATURE_GROUP_VERSION}"
            )
            return fg
    except Exception:
        pass  # will create below

    logger.info(
        f"Creating feature group: {config.FEATURE_GROUP_NAME} "
        f"v{config.FEATURE_GROUP_VERSION}"
    )

    # We create a minimal schema DataFrame to register the feature group
    # The actual data is inserted separately via fg.insert()
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description=(
            "Hourly AQI observations + engineered features "
            "for Pearls AQI Predictor"
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,          # Offline-only (saves cost)
    )
    logger.info(f"Feature group created: {fg.name} v{fg.version}")
    return fg


# ══════════════════════════════════════════════════════════════
# PUSH FEATURES
# ══════════════════════════════════════════════════════════════

def push_features(df: pd.DataFrame) -> None:
    """
    Insert/upsert engineered features into the Hopsworks Feature Group.

    Args:
        df: DataFrame with at least REQUIRED_COLUMNS. Extra columns accepted.
    """
    logger.info(f"Pushing {len(df)} rows to Feature Store ...")

    fg = get_or_create_feature_group()
    df_ready = _prepare_df_for_hopsworks(df)

    fg.insert(df_ready, write_options={"wait_for_job": False})
    logger.info(
        f"Feature Store: inserted {len(df_ready)} rows into '{fg.name}'."
    )


# ══════════════════════════════════════════════════════════════
# FETCH TRAINING DATA
# ══════════════════════════════════════════════════════════════

def get_training_data(days: int = 90) -> pd.DataFrame:
    """
    Fetch the last `days` days of data from the Feature Store for training.

    Returns a DataFrame with all feature columns + target (aqi).
    Raises FeatureStoreNotConfigured if Hopsworks is not set up.
    """
    fs = get_feature_store()
    logger.info(
        f"Fetching last {days} days of training data from Feature Store ..."
    )

    fg = get_or_create_feature_group()

    # --- Obtain (or re-create) the Feature View ----------------------
    fv = None
    for attempt in range(2):
        try:
            fv = fs.get_feature_view(
                name=config.FEATURE_VIEW_NAME,
                version=config.FEATURE_VIEW_VERSION,
            )
            # Smoke-test: if the view is stale it may return None on queries
            test_df = fv.get_batch_data()
            if test_df is not None and len(test_df) > 0:
                break  # view is healthy
            # View exists but returns no data — delete and recreate
            logger.warning("Feature View returned empty data, recreating ...")
            try:
                fv.delete()
            except Exception:
                pass
            fv = None
        except Exception:
            fv = None

        if fv is None:
            logger.info("Creating fresh Feature View from Feature Group ...")
            fv = fs.get_or_create_feature_view(
                name=config.FEATURE_VIEW_NAME,
                version=config.FEATURE_VIEW_VERSION,
                query=fg.select_all(),
            )

    # --- Fetch data ---------------------------------------------------
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    df = None
    try:
        df, _ = fv.training_data(
            start_time=start_dt,
            end_time=end_dt,
        )
    except Exception:
        pass

    if df is None or len(df) == 0:
        try:
            df = fv.get_batch_data()
        except Exception:
            pass

    if df is None or len(df) == 0:
        raise ValueError(
            "Feature Store returned empty training data. "
            "Run `python src/backfill.py` first to populate the Feature Store."
        )

    logger.info(f"Fetched {len(df)} training rows from Feature Store.")
    return df


# ══════════════════════════════════════════════════════════════
# FETCH LATEST FEATURES (for inference)
# ══════════════════════════════════════════════════════════════

def get_latest_features(city: str, n_rows: int = 48) -> pd.DataFrame:
    """
    Fetch the most recent `n_rows` feature rows for a given city.
    Used by the prediction pipeline instead of live API fetch.

    Returns:
        DataFrame sorted by timestamp ascending, last `n_rows` rows.
    """
    fs = get_feature_store()

    try:
        fv = fs.get_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
        )
        df = fv.get_batch_data()
    except Exception as exc:
        raise RuntimeError(f"Could not fetch latest features: {exc}") from exc

    # Filter to city
    if "city" in df.columns:
        df = df[df["city"] == city]

    # Sort and take last n_rows
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df = df.tail(n_rows).reset_index(drop=True)
    logger.info(f"Fetched {len(df)} recent feature rows for city='{city}'.")
    return df


# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY — SAVE
# ══════════════════════════════════════════════════════════════

def save_model_to_registry(
    model: Any,
    scaler: Any,
    feature_names: list[str],
    metadata: dict,
) -> None:
    """
    Save the best model, scaler, feature names, and metadata to the
    Hopsworks Model Registry.

    Files saved inside the registry entry:
      - best_model.pkl
      - scaler.pkl
      - feature_names.json
      - model_metadata.json
    """
    mr = get_model_registry()
    logger.info(
        f"Saving model '{metadata.get('model_name')}' to Model Registry ..."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Save artefacts to temp dir
        joblib.dump(model, tmp / "best_model.pkl")
        joblib.dump(scaler, tmp / "scaler.pkl")
        (tmp / "feature_names.json").write_text(json.dumps(feature_names))
        (tmp / "model_metadata.json").write_text(json.dumps(metadata))

        # Create / get model in registry
        aqi_model = mr.python.create_model(
            name=config.MODEL_NAME,
            metrics={
                "rmse": metadata.get("rmse", 0),
                "mae": metadata.get("mae", 0),
                "r2": metadata.get("r2", 0),
            },
            description=(
                f"Pearls AQI Predictor — "
                f"{metadata.get('model_name', 'unknown')} | "
                f"RMSE={metadata.get('rmse')} | "
                f"Trained {metadata.get('trained_at', '')}"
            ),
            input_example=None,
            model_schema=None,
        )

        aqi_model.save(str(tmp))

    logger.info(
        f"Model saved to Registry: {config.MODEL_NAME} "
        f"(RMSE={metadata.get('rmse')}, MAE={metadata.get('mae')})"
    )


# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY — LOAD
# ══════════════════════════════════════════════════════════════

def load_model_from_registry() -> tuple[Any, Any, list[str], dict]:
    """
    Download the best model artefacts from Hopsworks Model Registry.

    Returns:
        (model, scaler, feature_names, metadata)
    Raises:
        FeatureStoreNotConfigured — if Hopsworks not configured
        RuntimeError              — if model not found in registry
    """
    mr = get_model_registry()
    logger.info(f"Loading model '{config.MODEL_NAME}' from Model Registry ...")

    try:
        model_entry = mr.get_best_model(
            name=config.MODEL_NAME,
            metric="rmse",
            direction="min",
        )
    except Exception:
        # Fallback: get latest version
        try:
            model_entry = mr.get_model(name=config.MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"No model found in registry under '{config.MODEL_NAME}'. "
                "Run `python src/train.py` first."
            ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = model_entry.download(local_path=tmpdir)
        model_path = Path(model_dir)

        model = joblib.load(model_path / "best_model.pkl")
        scaler_file = model_path / "scaler.pkl"
        scaler = joblib.load(scaler_file) if scaler_file.exists() else None
        fn_file = model_path / "feature_names.json"
        feature_names = (
            json.loads(fn_file.read_text()) if fn_file.exists() else []
        )
        meta_file = model_path / "model_metadata.json"
        metadata = (
            json.loads(meta_file.read_text()) if meta_file.exists() else {}
        )

    logger.info(
        f"Loaded model from Registry: {metadata.get('model_name', 'Unknown')}"
    )
    return model, scaler, feature_names, metadata
