"""
prediction.py — Load the trained model, predict delays, and explain WHY a
delay is predicted (top contributing factors) for the dashboard.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib

from src.utils import BEST_MODEL, FEATURE_COLS, classify_delay, safe_log
from src.train_model import load_best_model, get_feature_importance, train_and_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Shared input → model-row builder (used by both prediction and cause analysis)
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_MAP = {
    "Hour":                      "Hour",
    "Minute":                    "Minute",
    "Is_Weekend":                "Is_Weekend",
    "Peak_Hour":                 "Peak_Hour",
    "Month":                     "Month",
    "Temperature_C":             "Temperature_C",
    "Rainfall_mm":               "Rainfall_mm",
    "Humidity_pct":              "Humidity_pct",
    "Traffic_Index":             "Traffic_Index",
    "Distance_km_Proxy":         "Distance_km_Proxy",
    "Passenger_Count_Simulated": "Passenger_Count_Simulated",
}

OHE_PREFIXES = {
    "Weather":       "Weather_",
    "Season":        "Season_",
    "Traffic_Level": "Traffic_Level_",
    "Weekday":       "Weekday_",
    "Time_Period":   "Time_Period_",
}


def build_feature_row(input_dict: dict, feature_names: list) -> dict:
    """Build a single model-ready row (dict of {feature: value}) from raw UI input."""
    row = {col: 0 for col in feature_names}

    for feat_col, inp_key in NUMERIC_MAP.items():
        if feat_col in row and inp_key in input_dict:
            try:
                row[feat_col] = float(input_dict[inp_key])
            except (ValueError, TypeError):
                pass

    for inp_key, prefix in OHE_PREFIXES.items():
        val = input_dict.get(inp_key, None)
        if val:
            col_name = f"{prefix}{val}"
            if col_name in row:
                row[col_name] = 1

    return row


def get_model_and_features():
    """Return (model, feature_names), training if necessary."""
    return load_best_model()


def predict_delay(input_dict: dict, model=None, feature_names: list = None) -> dict:
    """
    Predict delay for a single input.

    Returns
    -------
    dict with keys: predicted_delay, delay_category, input_summary
    """
    if model is None or feature_names is None:
        model, feature_names = get_model_and_features()

    row = build_feature_row(input_dict, feature_names)
    X_input = pd.DataFrame([row], columns=feature_names).astype(float)
    predicted = float(model.predict(X_input)[0])
    predicted = max(0.0, round(predicted, 1))  # delay cannot be negative

    return {
        "predicted_delay":  predicted,
        "delay_category":   classify_delay(predicted),
        "input_summary":    input_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Delay-cause explanation
# ─────────────────────────────────────────────────────────────────────────────

# Numeric drivers: (display label, reason template using value + typical/reference)
# Deliberately limited to factors that actually vary trip-to-trip and are meaningful
# delay drivers (traffic, rain) — structural facts like route distance live in the
# journey summary instead, not in "causes", since they don't explain THIS delay.
_NUMERIC_CAUSES = {
    "Traffic_Index": ("Heavy Traffic",
        lambda v, typ: f"Traffic index is {v:.0f} vs a typical {typ:.0f} on this route — congestion is pushing the delay up."),
    "Rainfall_mm": ("Rain",
        lambda v, typ: f"Rainfall is around {v:.1f} mm — wet conditions slow buses and increase delay."),
}

# One-hot / flag drivers: {model column name: (display label, reason)}
_FLAG_CAUSES = {
    "Weather_Heavy Rain":        ("Heavy Rain", "Heavy rain is expected — this is one of the strongest delay drivers in the data."),
    "Weather_Rainy":             ("Rain", "Rainy weather conditions are contributing to the delay."),
    "Traffic_Level_Very High":   ("Very Heavy Traffic", "Traffic congestion on this route is categorised as Very High."),
    "Traffic_Level_High":        ("Heavy Traffic", "Traffic congestion on this route is categorised as High."),
    "Peak_Hour":                 ("Peak Hour Congestion", "This trip falls in a peak commuting window, when roads are busiest."),
    "Season_Southwest_Monsoon":  ("Monsoon Season", "Southwest monsoon season typically brings more rain-related delay."),
    "Season_Northeast_Monsoon":  ("Monsoon Season", "Northeast monsoon (Chennai's main rainy season) typically brings more delay."),
    "Time_Period_Morning Peak":  ("Morning Peak Period", "Morning peak period sees the densest traffic of the day."),
    "Time_Period_Evening":       ("Evening Peak Period", "Evening period traffic tends to be heavier than average."),
}


def get_delay_causes(input_dict: dict, model=None, feature_names: list = None,
                      df_reference: pd.DataFrame = None, top_k: int = 3) -> list:
    """
    Rank the top contributing factors behind a predicted delay.

    Combines each factor's global model importance with how far its value is
    from the "typical" trip, so the causes shown are both influential AND
    unusual for this particular trip.

    Returns a list of up to `top_k` dicts: {"label": str, "reason": str}.
    Falls back to a "Normal Conditions" message if nothing stands out.
    """
    if model is None or feature_names is None:
        model, feature_names = get_model_and_features()

    row = build_feature_row(input_dict, feature_names)
    fi = get_feature_importance(model, feature_names)
    fi_map = dict(zip(fi["Feature"], fi["Importance"])) if fi is not None else {}

    scored = []

    # Numeric factors — score = importance x how far above typical the value is
    for col, (label, template) in _NUMERIC_CAUSES.items():
        if col not in row or col not in fi_map:
            continue
        val = row[col]
        if df_reference is not None and col in df_reference.columns and len(df_reference):
            typical = float(df_reference[col].median())
            std = float(df_reference[col].std()) or 1.0
        else:
            typical, std = val, 1.0
        z = (val - typical) / std
        if z > 0.15:
            scored.append((fi_map[col] * z, label, template(val, typical)))

    # Flag / categorical factors — active one-hot columns
    for col, (label, reason) in _FLAG_CAUSES.items():
        if row.get(col, 0) == 1 and col in fi_map:
            scored.append((fi_map[col], label, reason))

    scored.sort(key=lambda x: -x[0])

    seen, causes = set(), []
    for score, label, reason in scored:
        if label in seen:
            continue
        seen.add(label)
        causes.append({"label": label, "reason": reason, "score": max(score, 0.0)})
        if len(causes) >= top_k:
            break

    if not causes:
        causes = [{"label": "Normal Conditions", "score": 1.0,
                   "reason": "No unusual weather, traffic, or timing factors detected — this trip looks close to typical."}]

    total = sum(c["score"] for c in causes) or 1.0
    for c in causes:
        c["pct"] = round(100 * c["score"] / total, 1)
    return causes


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function used by the dashboard for actual-vs-predicted plots
# ─────────────────────────────────────────────────────────────────────────────

def get_test_predictions(df, model=None, feature_names=None):
    """
    Re-run the train/test split and return (y_test, y_pred).
    Used for rendering model-performance-style diagnostics if needed.
    """
    from src.feature_engineering import build_features
    from sklearn.model_selection  import train_test_split
    import numpy as np

    if model is None or feature_names is None:
        model, feature_names = get_model_and_features()

    X, y, _ = build_features(df)
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_pred = model.predict(X_test)
    return np.array(y_test), np.array(y_pred)
