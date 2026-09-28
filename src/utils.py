"""
utils.py — Shared utility functions for the Chennai Public Transport Delay Analytics project.
"""

import os
import pathlib

# ─── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUT_DIR / "plots"
REPORTS_DIR  = OUTPUT_DIR / "reports"

RAW_CSV      = DATA_DIR / "chennai_public_transport_delay_integrated.csv"
CLEANED_CSV  = DATA_DIR / "cleaned_transport_data.csv"
BEST_MODEL   = MODEL_DIR / "best_delay_model.pkl"
FEATURE_COLS = MODEL_DIR / "feature_columns.pkl"
MODEL_CMP    = MODEL_DIR / "model_comparison.csv"

# ─── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL = "Delay_Minutes"

DELAY_THRESHOLDS = {
    "On Time":  (None, 5),
    "Moderate": (5, 15),
    "High":     (15, 30),
    "Severe":   (30, None),
}

# Columns that come directly from the dataset and are metadata / leakage risks
EXCLUDE_FROM_ML = [
    "Trip_ID",
    "Route_ID",            # high-cardinality ID; Route_Name used instead
    "Route_Description",
    "Representative_Stop",
    "Scheduled_Arrival",
    "Actual_Arrival_Simulated",  # leakage — derived from delay
    "Delay_Minutes",             # target
    "Delay_Severity",            # target-derived
    "Transport_Data_Source",
    "Transport_Data_Status",
    "Weather_Data_Status",
    "Traffic_Data_Status",
    "Delay_Data_Status",
    "Passenger_Data_Status",
    "Representative_Coordinate_Note",
    "Date",                      # extracted into numeric features
    "Latitude",
    "Longitude",
]

SIMULATED_COLUMNS = [
    "Actual_Arrival_Simulated",
    "Passenger_Count_Simulated",
    "Passenger_Disruption_Score",
    "Delay_Minutes",
    "Delay_Severity",
]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all required project directories if they do not exist."""
    for d in [DATA_DIR, MODEL_DIR, PLOTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def classify_delay(minutes: float) -> str:
    """Classify a delay value into a severity category."""
    if minutes < 5:
        return "On Time"
    elif minutes < 15:
        return "Moderate"
    elif minutes < 30:
        return "High"
    else:
        return "Severe"


def safe_log(msg: str, level: str = "INFO"):
    """Simple console logger (ASCII-safe for Windows cp1252 terminals)."""
    import sys
    safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
    try:
        print(f"[{level}] {safe_msg}")
    except Exception:
        sys.stdout.buffer.write(f"[{level}] {safe_msg}\n".encode("utf-8", errors="replace"))
