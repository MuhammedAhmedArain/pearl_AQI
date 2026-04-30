"""
src/train.py
============
Multi-model training pipeline with automatic best-model selection.

Pipeline:
  1. Run preprocessing (load → clean → engineer → split → scale)
  2. Define all model candidates with hyperparameter grids
  3. Train each model with k-fold cross-validation
  4. Optionally run RandomizedSearchCV for hyperparameter tuning
  5. Evaluate each model on hold-out test set: MAE, RMSE, R²
  6. Print a formatted comparison table
  7. Select the best model by lowest RMSE
  8. Save best model + metadata using joblib
  9. Persist model comparison CSV

Model candidates:
  ┌────────────────────┬───────────────────────────────────┐
  │ Model              │ Why included                      │
  ├────────────────────┼───────────────────────────────────┤
  │ Linear Regression  │ Baseline                          │
  │ Ridge Regression   │ Regularized baseline              │
  │ Random Forest      │ Strong ensemble, low variance     │
  │ Gradient Boosting  │ Sklearn GBM                       │
  │ XGBoost            │ Fast, high-performance boosting   │
  └────────────────────┴───────────────────────────────────┘
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_logger, timer, format_metrics_table, save_json
from src.preprocess import run_preprocessing_pipeline

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ── Fix random seed for reproducibility ─────────────────────
np.random.seed(config.RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════

def get_model_candidates() -> list[dict]:
    """
    Return a list of model configuration dicts.
    Each dict has:
      - name         : display name
      - model        : sklearn-compatible estimator
      - param_grid   : hyperparameter search space (for RandomizedSearchCV)
      - tune         : whether to run hyperparam search (slower but better)
    """
    return [
        {
            "name":  "Linear Regression",
            "model": LinearRegression(n_jobs=-1),
            "param_grid": None,
            "tune":  False,
        },
        {
            "name":  "Ridge Regression",
            "model": Ridge(random_state=config.RANDOM_SEED),
            "param_grid": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "tune":  True,
        },
        {
            "name":  "Random Forest",
            "model": RandomForestRegressor(
                n_estimators=200,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth":    [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "max_features":      ["sqrt", "log2"],
            },
            "tune":  True,
        },
        {
            "name":  "Gradient Boosting",
            "model": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=config.RANDOM_SEED,
            ),
            "param_grid": {
                "n_estimators":  [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth":     [3, 5, 7],
                "subsample":     [0.7, 0.8, 1.0],
            },
            "tune":  True,
        },
        {
            "name":  "XGBoost",
            "model": xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            ),
            "param_grid": {
                "n_estimators":     [200, 300, 500],
                "learning_rate":    [0.01, 0.05, 0.1],
                "max_depth":        [4, 6, 8],
                "subsample":        [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "reg_alpha":        [0, 0.1, 0.5],
            },
            "tune":  True,
        },
        {
            "name":  "Deep Learning (MLP Neural Net)",
            "model": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size="auto",
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=config.RANDOM_SEED,
            ),
            "param_grid": {
                "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
                "alpha": [0.0001, 0.001, 0.01],
            },
            "tune": True,
        },
    ]


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and R² on the test set.

    Returns:
        Dict with keys: mae, rmse, r2
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def cross_validate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = config.CV_FOLDS,
) -> dict[str, float]:
    """
    Run k-fold cross-validation and return mean/std of CV RMSE.
    Uses negative MSE scoring (sklearn convention) then converts.
    """
    scores = cross_val_score(
        model, X_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    cv_rmse = -scores  # negate to get positive RMSE
    return {
        "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
        "cv_rmse_std":  round(float(cv_rmse.std()),  4),
    }


# ══════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════

def tune_model(
    model: Any,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
) -> Any:
    """
    RandomizedSearchCV with RMSE scoring.
    Returns the best estimator found.
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info(
        f"  Best params: {search.best_params_} | "
        f"CV RMSE: {-search.best_score_:.4f}"
    )
    return search.best_estimator_


# ══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════

@timer
def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    tune_hyperparams: bool = True,
) -> list[dict]:
    """
    Train every model candidate, optionally tune hyperparameters,
    cross-validate, and evaluate on test set.

    Returns:
        List of result dicts (one per model), sorted by RMSE ascending.
    """
    candidates = get_model_candidates()
    results    = []

    logger.info("=" * 60)
    logger.info(f"Training {len(candidates)} models …")
    logger.info("=" * 60)

    for cfg in candidates:
        name = cfg["name"]
        logger.info(f"\n▶ Training: {name}")

        model = cfg["model"]

        # ── Optional hyperparameter tuning ───────────────────
        if tune_hyperparams and cfg["tune"] and cfg["param_grid"]:
            logger.info(f"  Tuning hyperparameters (RandomizedSearchCV) …")
            model = tune_model(model, cfg["param_grid"], X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # ── Cross-validation ─────────────────────────────────
        cv_scores = cross_validate_model(model, X_train, y_train)

        # ── Test set evaluation ──────────────────────────────
        metrics = evaluate_model(model, X_test, y_test)

        result = {
            "model_name":    name,
            "model_object":  model,
            "mae":           metrics["mae"],
            "rmse":          metrics["rmse"],
            "r2":            metrics["r2"],
            "cv_rmse_mean":  cv_scores["cv_rmse_mean"],
            "cv_rmse_std":   cv_scores["cv_rmse_std"],
        }
        results.append(result)

        logger.info(
            f"  ✓ {name}: MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f} | "
            f"CV RMSE={cv_scores['cv_rmse_mean']:.4f}±{cv_scores['cv_rmse_std']:.4f}"
        )

    # Sort by RMSE (primary), then MAE (secondary)
    results.sort(key=lambda r: (r["rmse"], r["mae"]))
    return results


# ══════════════════════════════════════════════════════════════
# BEST MODEL SELECTION & PERSISTENCE
# ══════════════════════════════════════════════════════════════

def select_and_save_best_model(
    results: list[dict],
    feature_names: list[str],
    scaler: Any = None,
) -> dict:
    """
    Pick the model with lowest RMSE and save:
      - Model object  → models/best_model.pkl         (local fallback)
      - Scaler        → models/scaler.pkl
      - Metadata      → models/model_metadata.json
      - Model Registry → Hopsworks (if configured)

    Returns:
        The winning result dict.
    """
    best = results[0]  # Already sorted by RMSE
    logger.info(f"\nBEST MODEL: {best['model_name']}")
    logger.info(f"   RMSE={best['rmse']} | MAE={best['mae']} | R2={best['r2']}")

    # Save model locally (always — acts as fallback)
    joblib.dump(best["model_object"], config.BEST_MODEL_PATH)
    logger.info(f"   Saved locally -> {config.BEST_MODEL_PATH}")

    # Build metadata dict
    metadata = {
        "model_name":     best["model_name"],
        "mae":            best["mae"],
        "rmse":           best["rmse"],
        "r2":             best["r2"],
        "cv_rmse_mean":   best["cv_rmse_mean"],
        "cv_rmse_std":    best["cv_rmse_std"],
        "feature_count":  len(feature_names),
        "feature_names":  feature_names,
        "trained_at":     datetime.datetime.utcnow().isoformat(),
        "config": {
            "test_size":   config.TEST_SIZE,
            "cv_folds":    config.CV_FOLDS,
            "random_seed": config.RANDOM_SEED,
        },
    }
    save_json(metadata, config.MODEL_METADATA_PATH)
    logger.info(f"   Metadata -> {config.MODEL_METADATA_PATH}")

    # Save to Hopsworks Model Registry (if configured)
    if config.USE_FEATURE_STORE:
        try:
            from src.feature_store import save_model_to_registry
            save_model_to_registry(
                model=best["model_object"],
                scaler=scaler,
                feature_names=feature_names,
                metadata=metadata,
            )
            logger.info("   Model saved to Hopsworks Model Registry.")
        except Exception as exc:
            logger.warning(f"   Model Registry save failed (non-fatal): {exc}")
    else:
        logger.info("   Hopsworks not configured - model saved locally only.")

    return best



def save_comparison_table(results: list[dict]) -> None:
    """Save model comparison results to CSV and print to console."""
    rows = [
        {
            "model_name":   r["model_name"],
            "mae":          r["mae"],
            "rmse":         r["rmse"],
            "r2":           r["r2"],
            "cv_rmse_mean": r["cv_rmse_mean"],
            "cv_rmse_std":  r["cv_rmse_std"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Comparison table saved → {config.MODEL_COMPARISON_PATH}")

    # Pretty print
    print("\n" + "=" * 70)
    print("           MODEL COMPARISON — Pearls AQI Predictor")
    print("=" * 70)
    print(format_metrics_table(rows))
    print()


# ══════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════

@timer
def run_training_pipeline(tune_hyperparams: bool = True) -> dict:
    """
    End-to-end training pipeline:
      preprocess → train all models → compare → select best → save

    Args:
        tune_hyperparams: If True, run RandomizedSearchCV for eligible models.

    Returns:
        Metadata dict of the best model.
    """
    logger.info("🚀 Starting Pearls AQI Predictor training pipeline …")

    # 1. Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = run_preprocessing_pipeline()

    # 2. Align feature columns
    feature_cols = [c for c in X_train.columns if c in feature_names]
    X_tr = X_train[feature_cols]
    X_te = X_test[feature_cols]

    # 3. Train all models
    results = train_all_models(X_tr, X_te, y_train, y_test, feature_cols, tune_hyperparams)

    # 4. Save comparison table
    save_comparison_table(results)

    # 5. Select and save best model (local + Model Registry)
    best = select_and_save_best_model(results, feature_cols, scaler=scaler)


    logger.info("\n✅ Training pipeline complete!")
    return {
        "model_name": best["model_name"],
        "mae":        best["mae"],
        "rmse":       best["rmse"],
        "r2":         best["r2"],
    }


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Pearls AQI Predictor models.")
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter tuning (faster but less accurate)"
    )
    args = parser.parse_args()

    result = run_training_pipeline(tune_hyperparams=not args.no_tune)

    print(f"\nBest Model : {result['model_name']}")
    print(f"   MAE    : {result['mae']}")
    print(f"   RMSE   : {result['rmse']}")
    print(f"   R2     : {result['r2']}")
