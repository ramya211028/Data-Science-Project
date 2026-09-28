"""
train_model.py — Train, evaluate, and save ML models for delay prediction.

Models:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. Gradient Boosting Regressor

Run standalone:  python src/train_model.py
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

from src.utils               import BEST_MODEL, FEATURE_COLS, MODEL_CMP, safe_log, ensure_dirs
from src.data_processing     import load_cleaned_data
from src.feature_engineering import build_features


# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ─────────────────────────────────────────────────────────────────────────────
def build_models() -> dict:
    """Return dict of {name: model/pipeline}."""
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame = None) -> tuple[object, pd.DataFrame, list]:
    """
    Full training pipeline.

    Returns
    -------
    best_model  : fitted model
    results_df  : DataFrame with metrics per model
    feature_names : list of feature columns used
    """
    ensure_dirs()

    if df is None:
        df = load_cleaned_data()

    if df.empty:
        raise RuntimeError("Empty dataset — cannot train.")

    X, y, feature_names = build_features(df)

    # ── Train / Test split (random; data spans multiple years non-sequentially) ─
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    safe_log(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    models  = build_models()
    results = []

    for name, model in models.items():
        safe_log(f"Training: {name} …")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results.append({"Model": name, **metrics})
        safe_log(f"  {name} → MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R²={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

    # ── Best model = highest R² ───────────────────────────────────────────────
    best_name  = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    safe_log(f"\nBest model: {best_name}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(best_model,    BEST_MODEL)
    joblib.dump(feature_names, FEATURE_COLS)
    results_df.to_csv(MODEL_CMP, index=False)

    safe_log(f"Saved model   → {BEST_MODEL}")
    safe_log(f"Saved columns → {FEATURE_COLS}")
    safe_log(f"Saved metrics → {MODEL_CMP}")

    return best_model, results_df, feature_names


# ─────────────────────────────────────────────────────────────────────────────
def load_best_model():
    """Load persisted best model and feature list."""
    if not BEST_MODEL.exists() or not FEATURE_COLS.exists():
        safe_log("Model files not found — training now …", "WARN")
        best_model, _, feature_names = train_and_evaluate()
        return best_model, feature_names
    model         = joblib.load(BEST_MODEL)
    feature_names = joblib.load(FEATURE_COLS)
    return model, feature_names


# ─────────────────────────────────────────────────────────────────────────────
def get_feature_importance(model, feature_names: list) -> pd.DataFrame | None:
    """Extract feature importance if the model supports it."""
    # Unwrap Pipeline
    inner = model
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model", model)

    if hasattr(inner, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": inner.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi["Importance"] = fi["Importance"].round(6)
        return fi

    if hasattr(inner, "coef_"):
        coefs = np.abs(inner.coef_)
        fi = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": coefs / coefs.sum(),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi["Importance"] = fi["Importance"].round(6)
        return fi

    return None


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    best, results, feats = train_and_evaluate()
    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))
