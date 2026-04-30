"""
src/utils.py
============
Shared utility functions used across the entire project:
  - Logger factory
  - File I/O helpers
  - JSON serialisation helpers
  - Timing decorator
  - Retry decorator for flaky API calls
"""

import json
import logging
import logging.handlers
import time
import functools
from pathlib import Path
from typing import Any, Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with:
      - Console handler (INFO)
      - Rotating file handler (DEBUG, max 5 MB × 3 backups)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(config.LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # ── Console handler ──────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ══════════════════════════════════════════════════════════════
# DECORATORS
# ══════════════════════════════════════════════════════════════

def timer(func: Callable) -> Callable:
    """Log execution time of a function."""
    logger = get_logger("timer")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator with exponential back-off.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay in seconds (doubles on each retry).
        exceptions: Tuple of exception types to catch.
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger("retry")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__qualname__} failed after {max_attempts} attempts. "
                            f"Last error: {exc}"
                        )
                        raise
                    logger.warning(
                        f"{func.__qualname__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {current_delay:.1f}s …"
                    )
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential back-off

        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════
# FILE I/O
# ══════════════════════════════════════════════════════════════

def save_json(data: Any, path: Path) -> None:
    """Serialise data to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load and return data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# ══════════════════════════════════════════════════════════════
# AQI HELPERS
# ══════════════════════════════════════════════════════════════

def aqi_category(aqi_value: float) -> dict:
    """Return AQI category metadata dict for a numeric AQI value."""
    return config.get_aqi_category(float(aqi_value))


def format_metrics_table(results: list[dict]) -> str:
    """
    Pretty-print a model comparison table.

    Args:
        results: List of dicts with keys: model_name, mae, rmse, r2

    Returns:
        Formatted ASCII table string.
    """
    header = f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for r in results:
        rows.append(
            f"{r['model_name']:<30} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}"
        )
    rows.append(sep)
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════

def get_cache_path(city: str, suffix: str = "json") -> Path:
    """Return a deterministic cache file path for a city."""
    safe_city = city.lower().replace(" ", "_")
    return config.CACHE_DIR / f"{safe_city}_cache.{suffix}"


def is_cache_valid(path: Path, ttl: int = config.CACHE_TTL_SECONDS) -> bool:
    """Return True if cache file exists and is younger than TTL seconds."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl
