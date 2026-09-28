"""
data_processing.py — Data loading, cleaning, and preprocessing pipeline.

Steps:
  1. Load raw CSV
  2. Display dimensions
  3. Check duplicates / missing values
  4. Impute missing values (median for numeric, mode/"Unknown" for categorical)
  5. Parse date, extract temporal features
  6. Validate numerical ranges
  7. Save cleaned CSV

Run standalone:  python src/data_processing.py
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.utils import (RAW_CSV, CLEANED_CSV, SIMULATED_COLUMNS, safe_log, ensure_dirs)


# ─────────────────────────────────────────────────────────────────────────────
def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw CSV.  Returns an empty DataFrame on failure."""
    csv_path = pathlib.Path(path) if path else RAW_CSV
    if not csv_path.exists():
        safe_log(f"CSV not found: {csv_path}", "ERROR")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        safe_log(f"Loaded {len(df):,} rows x {len(df.columns)} cols from {csv_path.name}")
        return df
    except Exception as exc:
        safe_log(f"Failed to read CSV: {exc}", "ERROR")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a summary profile of the raw dataset."""
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()

    profile = {
        "total_rows":        len(df),
        "total_cols":        len(df.columns),
        "duplicate_rows":    int(df.duplicated().sum()),
        "missing_total":     int(df.isnull().sum().sum()),
        "missing_by_col":    df.isnull().sum().to_dict(),
        "numeric_cols":      num_cols,
        "categorical_cols":  cat_cols + bool_cols,
        "columns":           df.columns.tolist(),
        "dtypes":            df.dtypes.astype(str).to_dict(),
        "simulated_cols":    [c for c in SIMULATED_COLUMNS if c in df.columns],
    }

    # Date range
    if "Date" in df.columns:
        try:
            dates = pd.to_datetime(df["Date"], errors="coerce")
            profile["date_min"] = str(dates.min().date())
            profile["date_max"] = str(dates.max().date())
        except Exception:
            profile["date_min"] = profile["date_min"] = "N/A"

    return profile


# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline.  Returns cleaned DataFrame."""
    if df.empty:
        return df

    df = df.copy()

    # ── Step 1: Remove exact duplicates ──────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed:
        safe_log(f"Removed {removed:,} duplicate rows.")

    # ── Step 2: Parse Date ───────────────────────────────────────────────────
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"]    = df["Date"].dt.year
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
        df["Day"]     = df["Date"].dt.day
        df["Quarter"] = df["Date"].dt.quarter

    # ── Step 3: Ensure Hour / Minute already exist (they do in this dataset) ─
    for col in ["Hour", "Minute"]:
        if col not in df.columns:
            df[col] = 0

    # ── Step 4: Validate numeric ranges ──────────────────────────────────────
    validations = {
        "Rainfall_mm":    (0, 500),
        "Humidity_pct":   (0, 100),
        "Temperature_C":  (15, 55),
        "Traffic_Index":  (0, 150),
        "Delay_Minutes":  (0, 300),
        "Distance_km_Proxy": (0, 200),
    }
    for col, (lo, hi) in validations.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad:
                safe_log(f"  Clamping {bad} out-of-range values in '{col}' to [{lo},{hi}].", "WARN")
                df[col] = df[col].clip(lo, hi)

    # ── Step 5: Impute missing numeric values (median) ───────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        miss = df[col].isnull().sum()
        if miss:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            safe_log(f"  Filled {miss} missing numeric values in '{col}' with median={med:.2f}.")

    # ── Step 6: Impute missing categorical values ────────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        miss = df[col].isnull().sum()
        if miss:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            safe_log(f"  Filled {miss} missing categorical values in '{col}' with '{fill_val}'.")

    # ── Step 7: Standardise Weekday (ensure string form exists) ─────────────
    if "Weekday" not in df.columns and "Date" in df.columns:
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        df["Weekday"] = df["Date"].dt.dayofweek.map(lambda x: day_names[x])

    # ── Step 8: Ensure Is_Weekend is boolean-int ─────────────────────────────
    if "Is_Weekend" in df.columns:
        df["Is_Weekend"] = df["Is_Weekend"].astype(int)

    # ── Step 9: Create time-period feature ───────────────────────────────────
    if "Hour" in df.columns and "Time_Period" not in df.columns:
        bins   = [-1, 5, 9, 12, 17, 20, 23]
        labels = ["Night","Morning Peak","Mid-Morning","Afternoon Peak","Evening","Late Night"]
        df["Time_Period"] = pd.cut(df["Hour"], bins=bins, labels=labels)
        df["Time_Period"] = df["Time_Period"].astype(str)

    safe_log(f"Cleaning complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def save_cleaned(df: pd.DataFrame, path=None):
    """Save cleaned DataFrame to CSV."""
    out = pathlib.Path(path) if path else CLEANED_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    safe_log(f"Cleaned dataset saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
def load_cleaned_data(path=None) -> pd.DataFrame:
    """Load pre-cleaned CSV if available; otherwise run full pipeline."""
    cleaned = pathlib.Path(path) if path else CLEANED_CSV
    if cleaned.exists():
        df = pd.read_csv(cleaned, low_memory=False)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        safe_log(f"Loaded cleaned data: {len(df):,} rows.")
        return df
    safe_log("Cleaned data not found — running pipeline from raw CSV.", "WARN")
    df = load_raw_data()
    df = clean_data(df)
    save_cleaned(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_dirs()
    raw  = load_raw_data()
    profile = profile_dataset(raw)
    print("\n=== Dataset Profile ===")
    for k, v in profile.items():
        if k not in ("missing_by_col", "dtypes", "numeric_cols", "categorical_cols", "columns"):
            print(f"  {k}: {v}")
    cleaned = clean_data(raw)
    save_cleaned(cleaned)
    print("\nDone.")
