"""
feature_engineering.py — Build ML-ready features from the cleaned dataset.

Responsibilities:
  - Select features that do NOT leak the target (Delay_Minutes)
  - Encode categorical variables
  - Return X (feature matrix) and y (target vector)
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.utils import TARGET_COL, EXCLUDE_FROM_ML, safe_log


# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["Weather", "Season", "Traffic_Level", "Weekday", "Time_Period"]
NUMERICAL_FEATURES   = [
    "Hour", "Minute", "Is_Weekend", "Peak_Hour", "Month",
    "Temperature_C", "Rainfall_mm", "Humidity_pct",
    "Traffic_Index", "Distance_km_Proxy",
]

# Optional passenger features (simulated — labelled clearly in UI)
PASSENGER_FEATURES = ["Passenger_Count_Simulated"]


# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   include_passenger: bool = True) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Build the ML feature matrix X and target y.

    Parameters
    ----------
    df                 : cleaned DataFrame
    include_passenger  : include simulated passenger count as a feature

    Returns
    -------
    X : pd.DataFrame  — encoded feature matrix
    y : pd.Series     — target (Delay_Minutes)
    feature_names : list[str]
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # ── Target ────────────────────────────────────────────────────────────────
    y = df[TARGET_COL].copy()

    # ── Numerical features present in dataset ─────────────────────────────────
    num_feats = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if include_passenger:
        num_feats += [c for c in PASSENGER_FEATURES if c in df.columns]

    # ── Categorical features — one-hot encode ─────────────────────────────────
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X_num = df[num_feats].copy().astype(float)
    X_cat = pd.get_dummies(df[cat_feats], drop_first=False, dtype=int) if cat_feats else pd.DataFrame()

    X = pd.concat([X_num, X_cat], axis=1)

    # ── Sanity-check: no NaNs in X ────────────────────────────────────────────
    nan_count = X.isnull().sum().sum()
    if nan_count:
        safe_log(f"Found {nan_count} NaN values in feature matrix — filling with column medians.", "WARN")
        X = X.fillna(X.median())

    feature_names = X.columns.tolist()
    safe_log(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features.")
    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
def get_feature_display_names(feature_names: list) -> dict:
    """
    Return a human-readable label for each feature column.
    """
    labels = {
        "Hour":                     "Hour of Day",
        "Minute":                   "Minute",
        "Is_Weekend":               "Is Weekend",
        "Peak_Hour":                "Peak Hour",
        "Month":                    "Month",
        "Temperature_C":            "Temperature (°C)",
        "Rainfall_mm":              "Rainfall (mm)",
        "Humidity_pct":             "Humidity (%)",
        "Traffic_Index":            "Traffic Index",
        "Distance_km_Proxy":        "Route Distance (km)",
        "Passenger_Count_Simulated":"Passenger Count (Simulated)",
    }
    return {f: labels.get(f, f.replace("_", " ")) for f in feature_names}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_processing import load_cleaned_data
    df = load_cleaned_data()
    X, y, feats = build_features(df)
    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Features: {feats[:10]} ...")
