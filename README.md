# Chennai Public Transport Delay Analytics

## Project Title
Data-Driven Public Transport Delay Analysis Using Exploratory Data Analysis (EDA) and Predictive Analytics

---

## What's in this version
The dashboard was redesigned around one job: **pick a route and stop, and immediately see
the expected delay, why it's happening, and the corrected arrival time.**

Everything that didn't directly serve that — the KPI tile wall (Total Records, Avg Delay,
Routes, Stops, etc.), the 7-page sidebar navigation (Dashboard Overview / Delay Analysis /
Spatial Analysis / Delay Prediction / Model Performance / Passenger Impact / Dataset
Information), and the always-visible charts — has been removed from the main view.
City-wide charts and the data dictionary are still available, tucked into two collapsed
"more info" panels at the bottom, so they don't compete with the main task.

### Main dashboard flow
1. Choose a **Route** and **Stop** — the app looks up the real scheduled arrival time and
   typical historical weather/traffic conditions for that stop.
2. Choose the **day of travel** and optionally tweak conditions (weather, traffic, rainfall,
   season) in an "Adjust conditions" panel — defaults are pre-filled from history.
3. Click **Check Delay & ETA** to see:
   - **Corrected Arrival Time** — scheduled time + predicted delay
   - **Status** (On Time / Moderate / High / Severe) with the predicted delay in minutes
   - **Top 2-3 likely causes** of the delay in plain English (e.g. "Heavy Traffic — traffic
     index is 95 vs a typical 50 on this route")

### Still there, just out of the way
- **📊 City-wide delay insights** (collapsed) — hourly delay pattern, delay by weather, most
  delayed routes
- **ℹ️ About this data** (collapsed) — the data transparency statement

---

## Problem Statement
Public transport delays in Chennai significantly affect commuter experience and productivity.
Existing systems primarily provide live tracking without sufficiently analysing historical delay
patterns or offering predictive insights. This project addresses that gap.

---

## Dataset
**File:** `chennai_public_transport_delay_integrated.csv`
**Source:** ChennaiGTFS / UngalSoththu MTC GTFS (public route/stop data)
**Records:** ~30,000 trip records
**Features:** 37 columns including route, stop, time, weather (simulated), traffic (simulated),
and delay (simulated)

### Data Transparency
Weather, traffic, delay, and passenger fields are **simulated/experimental features**
included for capstone research purposes. Transport route and stop information is from public
MTC GTFS data. Coordinates are representative GTFS stop points, not live GPS positions.
Predictions are indicative only, not suitable for live operational deployment.

---

## Technologies
| Technology | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Machine learning |
| Plotly | Interactive visualisations |
| Streamlit | Dashboard framework |
| Joblib | Model persistence |
| Statsmodels | OLS trendlines (insights panel) |

## Machine Learning Models
Linear Regression (baseline), Random Forest Regressor, Gradient Boosting Regressor — the best
model by R² is auto-selected and saved to `models/best_delay_model.pkl`.

---

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 (optional): Re-run preprocessing / retrain
The repo already ships with a cleaned model and comparison file, so this is optional unless
you're using new/updated data.
```bash
python src/data_processing.py
python src/train_model.py
```

### Step 3: Launch the dashboard
```bash
streamlit run app.py
```

---

## Project Structure
```
public_transport_delay_project/
|-- data/
|   +-- chennai_public_transport_delay_integrated.csv   (original - never modified)
|-- models/
|   |-- best_delay_model.pkl
|   |-- feature_columns.pkl
|   +-- model_comparison.csv
|-- src/
|   |-- __init__.py
|   |-- utils.py
|   |-- data_processing.py
|   |-- eda.py
|   |-- feature_engineering.py
|   |-- train_model.py
|   +-- prediction.py     (now also explains WHY a delay is predicted)
|-- app.py                (redesigned: single-focus Route & ETA lookup)
|-- requirements.txt
+-- README.md
```

---

## Limitations
- Weather, traffic, delay, and passenger data are simulated for capstone purposes
- Coordinates are representative GTFS stop points, not live GPS
- Predictions are indicative only; not suitable for live operational deployment
- Model performance reflects simulated data relationships

## Future Enhancements
- Real-time GPS integration from MTC vehicles
- Live traffic API (Google Maps / HERE) and live weather API (OpenWeatherMap)
- Actual MTC operational data feeds
- Advanced deep learning (LSTM for temporal patterns)
- Mobile application for passengers
