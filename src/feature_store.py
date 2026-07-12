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
import os
import tempfile
import time
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
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# Max retries for transient Hopsworks / Arrow Flight errors
_MAX_RETRIES = int(os.getenv("HOPSWORKS_MAX_RETRIES", "3"))
_BACKOFF_BASE = float(os.getenv("HOPSWORKS_BACKOFF_BASE", "5"))  # seconds

# Exception families we consider transient and worth retrying
_TRANSIENT_NAMES = (
    "FlightUnavailableError",
    "FlightInternalError",
    "FlightTimedOutError",
    "ArrowInvalid",
    "ConnectionError",
    "TimeoutError",
    "ReadTimeoutError",
)


# ══════════════════════════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════════════════════════


class FeatureStoreNotConfigured(RuntimeError):
    """Raised when HOPSWORKS_API_KEY is not set or is a placeholder."""

    pass


# ══════════════════════════════════════════════════════════════
# RETRY HELPER
# ══════════════════════════════════════════════════════════════


def _is_transient(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient Flight / network error."""
    name = type(exc).__name__
    if name in _TRANSIENT_NAMES:
        return True
    # Walk the cause-chain (raise … from …)
    cause = exc.__cause__ or exc.__context__
    while cause:
        if type(cause).__name__ in _TRANSIENT_NAMES:
            return True
        cause = cause.__cause__ or cause.__context__
    # Last resort: check the string repr (catches wrapped gRPC errors)
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "flight", "unavailable", "connect", "timed out",
        "tcp handshaker", "grpc",
    ))


def _retry(fn, *, label: str, max_retries: int = _MAX_RETRIES):
    """
    Call *fn()* up to *max_retries* times with exponential back-off.

    Only retries on errors that look transient (Flight / network).
    All other exceptions propagate immediately.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == max_retries:
                break
            wait = _BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                f"[{label}] attempt {attempt}/{max_retries} failed "
                f"({type(exc).__name__}: {exc}). "
                f"Retrying in {wait:.0f}s …"
            )
            time.sleep(wait)
    raise last_exc


# ══════════════════════════════════════════════════════════════
# ARROW FLIGHT KILL-SWITCH
# ══════════════════════════════════════════════════════════════

_flight_disabled = False


def _disable_flight_client():
    """
    Programmatically disable the hsfs Arrow Flight client for this session.

    When disabled, ``_should_be_used()`` returns False so that:
      * ``fg.read()`` falls back to REST file download (port 443)
        via ``dataset_api.read_content()``.
      * ``is_data_format_supported()`` / ``is_query_supported()``
        return False, preventing any Flight call.

    Called automatically when ``HOPSWORKS_DISABLE_FLIGHT=true``.
    """
    global _flight_disabled
    if _flight_disabled:
        return
    try:
        from hsfs.core import arrow_flight_client as _afc

        inst = _afc.get_instance()
        inst._disabled_for_session = True
        logger.info(
            "Arrow Flight client disabled for this session — "
            "all reads will use REST API (port 443)."
        )
    except Exception as exc:
        logger.debug(f"Could not disable Arrow Flight client: {exc}")
    _flight_disabled = True


def _is_flight_disabled() -> bool:
    """Return True when Flight has been disabled."""
    return _flight_disabled


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
                f"Connecting to Hopsworks project " f"'{config.HOPSWORKS_PROJECT}' ..."
            )
            _project = hopsworks.login(
                host=config.HOPSWORKS_HOST,
                project=config.HOPSWORKS_PROJECT,
                api_key_value=config.HOPSWORKS_API_KEY,
                cert_folder=tempfile.gettempdir(),
            )

            # Disable Arrow Flight if requested before calling get_feature_store()
            # (CI runners can't reach port 5005, pre-setting disables connection attempts)
            if os.getenv("HOPSWORKS_DISABLE_FLIGHT", "").lower() in ("true", "1"):
                try:
                    from hsfs.core import arrow_flight_client as _afc
                    _afc._arrow_flight_instance = _afc.ArrowFlightClient(disabled_for_session=True)
                    logger.info("Arrow Flight client disabled before get_feature_store() to prevent connection attempts.")
                except Exception as exc:
                    logger.debug(f"Could not pre-disable Arrow Flight: {exc}")
                _disable_flight_client()

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
    "temp",
    "humidity",
    "wind_speed",
    "pressure",
    "visibility",
    # raw pollutants
    "pm2_5",
    "pm10",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "nh3",
    # time features
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
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
    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Strip timezone info (make naive UTC)
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

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
            "Hourly AQI observations + engineered features " "for Pearls AQI Predictor"
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,  # Offline-only (saves cost)
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
    logger.info(f"Feature Store: inserted {len(df_ready)} rows into '{fg.name}'.")


# ══════════════════════════════════════════════════════════════
# FETCH TRAINING DATA
# ══════════════════════════════════════════════════════════════


def get_training_data(days: int = 90) -> pd.DataFrame:
    """
    Fetch the last `days` days of data from the Feature Store for training.

    Read strategy (order depends on whether Arrow Flight is available):

    When Flight is *enabled* (local dev, Streamlit cloud):
      1. Feature View .training_data()   — fastest, time-filtered
      2. Feature View .get_batch_data()   — full dump, filter later
      3. Feature Group .read()            — REST file fallback

    When Flight is *disabled* (CI runners where port 5005 is blocked):
      → Feature Group .read()  only  (uses REST download on port 443)

    Returns a DataFrame with all feature columns + target (aqi).
    Raises FeatureStoreNotConfigured if Hopsworks is not set up.
    """
    fs = get_feature_store()
    logger.info(f"Fetching last {days} days of training data from Feature Store ...")

    fg = get_or_create_feature_group()

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    df = None

    # ── When Flight is disabled, use materialized training dataset (REST) ──
    if _is_flight_disabled():
        logger.info("Flight disabled — using REST materialization read path ...")
        fv = _get_or_create_feature_view(fs, fg)
        try:
            # Step A: Check for existing materialized datasets to avoid job creation overhead
            logger.info("Checking for existing materialized training datasets ...")
            tds = fv.get_training_datasets()
            # Filter out IN_MEMORY datasets because they do not have files on disk and require Flight to read
            tds = [td for td in tds if td.training_dataset_type != td.IN_MEMORY]
            if tds:
                latest_td = sorted(tds, key=lambda x: x.version, reverse=True)[0]
                logger.info(f"Loading latest materialized training dataset version {latest_td.version} ...")
                result = fv.get_training_data(training_dataset_version=latest_td.version)
                df = result[0] if isinstance(result, tuple) else result

            # Step B: If none exists, materialize a new one
            if df is None or len(df) == 0:
                logger.info("No existing training dataset found. Materializing new version on cluster ...")
                version, job = fv.create_training_data(
                    start_time=start_dt,
                    end_time=end_dt,
                    description=f"Auto-materialized training dataset for last {days} days",
                    data_format="parquet",
                    write_options={"wait_for_job": True}
                )
                logger.info(f"Materialization complete. Downloading version {version} ...")
                result = fv.get_training_data(training_dataset_version=version)
                df = result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            logger.warning(f"REST-based materialized read failed ({type(exc).__name__}: {exc}).")
    else:
        # ── Flight available — try Feature View first ────────────────
        fv = _get_or_create_feature_view(fs, fg)

        # Strategy 1: FV.training_data()
        try:
            logger.info("Strategy 1: FV.training_data() ...")
            result = _retry(
                lambda: fv.training_data(start_time=start_dt, end_time=end_dt),
                label="FV-training-data",
            )
            if result is not None:
                df = result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            logger.warning(f"Strategy 1 failed ({type(exc).__name__}: {exc}).")

        # Strategy 2: FV.get_batch_data()
        if df is None or len(df) == 0:
            try:
                logger.info("Strategy 2: FV.get_batch_data() ...")
                df = _retry(
                    lambda: fv.get_batch_data(),
                    label="FV-batch-data",
                )
            except Exception as exc:
                logger.warning(f"Strategy 2 failed ({type(exc).__name__}: {exc}).")

        # Strategy 3: Materialize fallback if Flight fails
        if df is None or len(df) == 0:
            logger.info("Strategies 1-2 failed — disabling Flight, using REST materialization ...")
            _disable_flight_client()
            try:
                tds = fv.get_training_datasets()
                tds = [td for td in tds if td.training_dataset_type != td.IN_MEMORY]
                if tds:
                    latest_td = sorted(tds, key=lambda x: x.version, reverse=True)[0]
                    logger.info(f"Loading materialized training dataset version {latest_td.version} ...")
                    result = fv.get_training_data(training_dataset_version=latest_td.version)
                    df = result[0] if isinstance(result, tuple) else result
            except Exception as exc:
                logger.warning(f"Materialization fallback failed ({type(exc).__name__}: {exc}).")
    if df is None or len(df) == 0:
        raise ValueError(
            "Feature Store returned empty training data after all strategies. "
            "Run `python src/backfill.py` first to populate the Feature Store."
        )

    # Time-filter the results if we got a full dump
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        mask = df["timestamp"] >= pd.Timestamp(start_dt)
        if mask.sum() > 0:
            df = df[mask]

    logger.info(f"Fetched {len(df)} training rows from Feature Store.")
    return df


def _get_or_create_feature_view(fs, fg):
    """Obtain (or re-create) the Feature View. Internal helper."""
    fv = None
    for _attempt in range(2):
        try:
            fv = fs.get_feature_view(
                name=config.FEATURE_VIEW_NAME,
                version=config.FEATURE_VIEW_VERSION,
            )
            break
        except Exception:
            fv = None

        if fv is None:
            logger.info("Creating fresh Feature View from Feature Group ...")
            fv = fs.get_or_create_feature_view(
                name=config.FEATURE_VIEW_NAME,
                version=config.FEATURE_VIEW_VERSION,
                query=fg.select_all(),
            )
    return fv


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
    fg = get_or_create_feature_group()

    df = None

    if _is_flight_disabled():
        # Flight is off → try to load latest materialized training dataset
        try:
            fv = _get_or_create_feature_view(fs, fg)
            tds = fv.get_training_datasets()
            tds = [td for td in tds if td.training_dataset_type != td.IN_MEMORY]
            if tds:
                latest_td = sorted(tds, key=lambda x: x.version, reverse=True)[0]
                logger.info(f"Loading latest materialized training dataset version {latest_td.version} for inference ...")
                result = fv.get_training_data(training_dataset_version=latest_td.version)
                df = result[0] if isinstance(result, tuple) else result
            if df is None:
                raise ValueError("No materialized training datasets found.")
        except Exception as exc:
            raise RuntimeError(f"Could not fetch latest features via materialization: {exc}") from exc
    else:
        # Try Feature View first (uses Flight)
        try:
            fv = fs.get_feature_view(
                name=config.FEATURE_VIEW_NAME,
                version=config.FEATURE_VIEW_VERSION,
            )
            df = _retry(
                lambda: fv.get_batch_data(),
                label="latest-fv-batch",
            )
        except Exception as exc:
            logger.warning(f"Feature View read failed ({exc}). Trying materialization fallback ...")

        # Fallback: materialized training dataset read
        if df is None or len(df) == 0:
            _disable_flight_client()
            try:
                fv = _get_or_create_feature_view(fs, fg)
                tds = fv.get_training_datasets()
                tds = [td for td in tds if td.training_dataset_type != td.IN_MEMORY]
                if tds:
                    latest_td = sorted(tds, key=lambda x: x.version, reverse=True)[0]
                    logger.info(f"Loading materialized training dataset version {latest_td.version} for inference ...")
                    result = fv.get_training_data(training_dataset_version=latest_td.version)
                    df = result[0] if isinstance(result, tuple) else result
            except Exception as exc:
                raise RuntimeError(
                    f"Could not fetch latest features: {exc}"
                ) from exc
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

    Retries with exponential back-off on transient Flight / network
    errors so that temporary port-5005 blocks (e.g. GitHub Actions)
    don't crash the pipeline.
    """
    mr = get_model_registry()
    logger.info(f"Saving model '{metadata.get('model_name')}' to Model Registry ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Save artefacts to temp dir
        joblib.dump(model, tmp / "best_model.pkl")
        joblib.dump(scaler, tmp / "scaler.pkl")
        (tmp / "feature_names.json").write_text(json.dumps(feature_names))
        (tmp / "model_metadata.json").write_text(json.dumps(metadata))

        # Create / get model in registry (retry on transient errors)
        aqi_model = _retry(
            lambda: mr.python.create_model(
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
            ),
            label="MR-create-model",
        )

        _retry(
            lambda: aqi_model.save(str(tmp)),
            label="MR-save-model",
        )

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
        # Strategy 1: Get the best model based on RMSE
        model_entry = mr.get_best_model(
            name=config.MODEL_NAME,
            metric="rmse",
            direction="min",
        )
    except Exception:
        # Strategy 2: Fallback to the latest version if no 'best' exists
        try:
            models = mr.get_models(name=config.MODEL_NAME)
            if not models:
                raise RuntimeError(f"No versions found for '{config.MODEL_NAME}'")
            # Sort by version descending and pick the top one
            model_entry = sorted(models, key=lambda m: m.version, reverse=True)[0]
            logger.info(f"Fallback: using latest model version {model_entry.version}")
        except Exception as exc:
            raise RuntimeError(
                f"No model found in registry under '{config.MODEL_NAME}'. "
                "Ensure you have run the training pipeline at least once."
            ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = model_entry.download(local_path=tmpdir)
        model_path = Path(model_dir)

        model = joblib.load(model_path / "best_model.pkl")
        scaler_file = model_path / "scaler.pkl"
        scaler = joblib.load(scaler_file) if scaler_file.exists() else None
        fn_file = model_path / "feature_names.json"
        feature_names = json.loads(fn_file.read_text()) if fn_file.exists() else []
        meta_file = model_path / "model_metadata.json"
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    logger.info(f"Loaded model from Registry: {metadata.get('model_name', 'Unknown')}")
    return model, scaler, feature_names, metadata
