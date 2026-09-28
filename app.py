"""
app.py — Chennai Public Transport Delay & ETA Dashboard (Streamlit)
Run:  streamlit run app.py

Focus: pick a route + stop, see the predicted delay, why it's happening,
and the corrected arrival time — with enough visual depth to be genuinely
useful, without turning back into a wall of stats.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Chennai Transport ETA",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.utils import (RAW_CSV, CLEANED_CSV, BEST_MODEL, FEATURE_COLS,
                        classify_delay, ensure_dirs)
from src.data_processing import load_raw_data, clean_data, save_cleaned
import src.eda as eda

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1220px;}

.hero {
    background: linear-gradient(120deg, #123b5e 0%, #1a5b8f 55%, #1abc9c 130%);
    border-radius: 18px;
    padding: 1.9rem 2.2rem;
    margin-bottom: 1.6rem;
    color: white;
}
.hero h1 {font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;}
.hero p  {font-size: .98rem; opacity: .88; margin: .35rem 0 0 0;}

.card {
    background: #ffffff;
    border: 1px solid #e7ebef;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 1px 3px rgba(20,30,50,0.04);
}
.card h3 {margin-top: 0;}

.result-placeholder {
    display:flex; align-items:center; justify-content:center; text-align:center;
    height: 100%; min-height: 320px; color:#8a97a3; font-size:.95rem;
    border: 1.5px dashed #d7dde3; border-radius: 16px; padding: 2rem;
}

.eta-block {
    background: #0f2a44;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    color: white;
    height: 100%;
}
.eta-label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7; margin:0;}
.eta-time {font-size: 2.4rem; font-weight: 800; margin: .1rem 0 0 0; line-height: 1;}
.eta-sub {font-size: .85rem; opacity: .8; margin-top: .35rem;}

.status-chip {
    display:inline-block; padding: .3rem .85rem; border-radius: 999px;
    font-weight: 700; font-size: .85rem; color: white; margin: .8rem 0 .6rem 0;
}

.cause-chip {
    background: #f8f9fb; border: 1px solid #e7ebef; border-left: 4px solid #2980b9;
    border-radius: 10px; padding: .65rem .9rem; margin-bottom: .5rem;
}
.cause-chip b {color:#123b5e;}
.cause-chip span {font-size: .85rem; color: #55606b;}
.cause-pct {float:right; color:#2980b9; font-weight:700;}

.schedule-strip {
    display:flex; justify-content:space-between; align-items:center; flex-wrap: wrap; gap:.4rem;
    background:#f8f9fb; border-radius: 12px; padding: .8rem 1.1rem; margin-bottom: 1rem;
    font-size: .92rem; color:#3c4a58;
}

.summary-card {
    background: #f8f9fb;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.summary-card h4 {margin: 0 0 .7rem 0; font-size: .95rem; color:#123b5e;}
.pill-row {display:flex; flex-wrap: wrap; gap: .4rem;}
.pill {
    display:inline-block; padding:.28rem .7rem; border-radius: 999px;
    font-size: .8rem; font-weight:600; background:#eef2f6; color:#3c4a58;
}
.big-stat {font-size: 1.5rem; font-weight: 800; color:#123b5e; margin:0;}
.big-stat-label {font-size:.78rem; color:#8a97a3; margin:0 0 .1rem 0; text-transform:uppercase; letter-spacing:.04em;}

.reliability-card {
    background: #f8f9fb;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.reliability-card h4 {margin: 0 0 .6rem 0; font-size: .95rem; color:#123b5e;}
.reliability-row {
    display:flex; justify-content:space-between; align-items:baseline;
    padding: .35rem 0; border-bottom: 1px solid #e7ebef; font-size: .87rem; color:#3c4a58;
}
.reliability-row:last-of-type {border-bottom:none;}
.reliability-row b {color:#123b5e; font-size: .95rem;}
.compare-up   {color:#e74c3c; font-weight:700;}
.compare-down {color:#27ae60; font-weight:700;}
.tip-line {
    margin-top: .6rem; padding-top: .6rem; border-top: 1px dashed #d7dde3;
    font-size: .82rem; color:#3c4a58;
}

.rank-badge {
    display:inline-block; width:1.6rem; text-align:center; margin-right:.4rem; font-size:1rem;
}
.od-strip {
    font-size: .88rem; color:#5a6773; margin-top:.25rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached data / model loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset ...")
def get_data() -> pd.DataFrame:
    ensure_dirs()
    if CLEANED_CSV.exists():
        df = pd.read_csv(CLEANED_CSV, low_memory=False)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    raw = load_raw_data()
    if raw.empty:
        return raw
    cleaned = clean_data(raw)
    save_cleaned(cleaned)
    return cleaned


@st.cache_resource(show_spinner="Loading ML model ...")
def get_model():
    import joblib
    if BEST_MODEL.exists() and FEATURE_COLS.exists():
        return joblib.load(BEST_MODEL), joblib.load(FEATURE_COLS)
    from src.train_model import train_and_evaluate
    df = get_data()
    if df.empty:
        return None, []
    best, _, feats = train_and_evaluate(df)
    return best, feats


@st.cache_data(show_spinner=False)
def get_route_options(df: pd.DataFrame) -> list:
    return sorted(df["Route_Name"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_stops_for_route(df: pd.DataFrame, route: str) -> list:
    return sorted(df.loc[df["Route_Name"] == route, "Representative_Stop"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_typical_trip(df: pd.DataFrame, route: str, stop: str) -> dict:
    """Typical schedule + conditions for a given route/stop, learned from history."""
    sub = df[(df["Route_Name"] == route) & (df["Representative_Stop"] == stop)]
    if sub.empty:
        sub = df[df["Route_Name"] == route]
    if sub.empty:
        sub = df

    def mode_or(col, default):
        if col in sub.columns and not sub[col].dropna().empty:
            m = sub[col].mode()
            if not m.empty:
                return m.iloc[0]
        return default

    def median_or(col, default):
        if col in sub.columns and not sub[col].dropna().empty:
            return float(sub[col].median())
        return default

    return {
        "Hour":               int(mode_or("Hour", 8)),
        "Minute":             int(mode_or("Minute", 0)),
        "Scheduled_Arrival":  mode_or("Scheduled_Arrival", "08:00"),
        "Origin":             mode_or("Origin", "—"),
        "Destination":        mode_or("Destination", "—"),
        "Weather":            mode_or("Weather", "Clear"),
        "Traffic_Level":      mode_or("Traffic_Level", "Moderate"),
        "Season":             mode_or("Season", "Summer"),
        "Traffic_Index":      median_or("Traffic_Index", 50.0),
        "Rainfall_mm":        median_or("Rainfall_mm", 0.0),
        "Temperature_C":      median_or("Temperature_C", 30.0),
        "Humidity_pct":       median_or("Humidity_pct", 70.0),
        "Distance_km_Proxy":  median_or("Distance_km_Proxy", 15.0),
    }


TIME_PERIOD_BINS = [(0, 5, "Night"), (5, 9, "Morning Peak"), (9, 12, "Mid-Morning"),
                     (12, 17, "Afternoon Peak"), (17, 20, "Evening"), (20, 24, "Late Night")]

STATUS_COLOURS = {"On Time": "#27ae60", "Moderate": "#f39c12", "High": "#e67e22", "Severe": "#e74c3c"}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

WEATHER_EMOJI = {"Clear": "☀️", "Cloudy": "☁️", "Rainy": "🌦️", "Heavy Rain": "⛈️"}

TRAFFIC_COLOURS = {"Low": "#1abc9c", "Moderate": "#2980b9", "High": "#e67e22", "Very High": "#c0392b"}


def time_period_for(hour: int) -> str:
    for lo, hi, label in TIME_PERIOD_BINS:
        if lo <= hour < hi:
            return label
    return "Night"


def add_minutes_to_clock(hour: int, minute: int, delay_minutes: float):
    total = (hour * 60 + minute + delay_minutes) % (24 * 60)
    nh, nm = divmod(int(round(total)), 60)
    return nh, nm


# ─────────────────────────────────────────────────────────────────────────────
# Load data & model
# ─────────────────────────────────────────────────────────────────────────────
df_full = get_data()
if df_full.empty:
    st.error(f"Dataset not found. Place the CSV at: {RAW_CSV}")
    st.stop()

model, feature_names = get_model()

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚌 Chennai Transport — Delay & ETA Predictor</h1>
  <p>Pick a route and stop to see the expected delay, why it's happening, and the corrected arrival time.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main: input card (left) + result card (right)
# ─────────────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1.05, 1.4], gap="large")

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Plan your trip")

    routes = get_route_options(df_full)
    route = st.selectbox("Route", routes, key="sel_route",
                          help="The MTC route number/name to check.")

    stops = get_stops_for_route(df_full, route)
    stop = st.selectbox("Stop", stops, key="sel_stop",
                         help="Representative stop on this route. Coordinates are GTFS reference points, not live GPS.")

    typical = get_typical_trip(df_full, route, stop)

    weekday = st.selectbox("Day of travel", WEEKDAYS, key="sel_weekday")
    is_weekend = 1 if weekday in ("Saturday", "Sunday") else 0

    custom_time = st.toggle("Simulate a different departure time", value=False,
                             help="By default the trip uses this stop's real scheduled time. "
                                  "Turn this on to test 'what if I leave at a different hour?'")
    if custom_time:
        c_h, c_m = st.columns(2)
        hour = c_h.slider("Hour", 0, 23, typical["Hour"])
        minute = c_m.select_slider("Minute", options=[0, 15, 30, 45], value=0)
        st.caption(f"Simulated departure: **{hour:02d}:{minute:02d}** (not this stop's real schedule)")
    else:
        hour, minute = typical["Hour"], typical["Minute"]
        st.caption(f"Scheduled arrival at this stop: **{typical['Scheduled_Arrival']}**")

    with st.expander("Adjust conditions (optional — defaults to typical conditions for this stop)"):
        weather = st.selectbox(
            "Weather", ["Clear", "Cloudy", "Rainy", "Heavy Rain"],
            index=["Clear", "Cloudy", "Rainy", "Heavy Rain"].index(typical["Weather"])
            if typical["Weather"] in ["Clear", "Cloudy", "Rainy", "Heavy Rain"] else 0,
            help="Simulated weather condition for the trip — one of the strongest delay drivers in this dataset.")
        traffic_lvl = st.selectbox(
            "Traffic Level", ["Low", "Moderate", "High", "Very High"],
            index=["Low", "Moderate", "High", "Very High"].index(typical["Traffic_Level"])
            if typical["Traffic_Level"] in ["Low", "Moderate", "High", "Very High"] else 1,
            help="Simplified traffic congestion category — a rounded-off version of Traffic Index.")
        traffic_idx = st.slider(
            "Traffic Index", 0.0, 120.0, float(typical["Traffic_Index"]), 1.0,
            help="A 0–120 congestion score combining vehicle density and road speed on this route. "
                 "0 = empty roads, 120 = gridlock. Typical city value is around 50.")
        rainfall = st.slider(
            "Rainfall (mm)", 0.0, 100.0, float(typical["Rainfall_mm"]), 0.5,
            help="Simulated rainfall in millimetres for the trip. More rainfall generally means slower "
                 "traffic and a higher predicted delay.")
        season = st.selectbox(
            "Season", ["Summer", "Southwest_Monsoon", "Northeast_Monsoon", "Winter"],
            index=["Summer", "Southwest_Monsoon", "Northeast_Monsoon", "Winter"].index(typical["Season"])
            if typical["Season"] in ["Summer", "Southwest_Monsoon", "Northeast_Monsoon", "Winter"] else 0,
            help="Chennai's Northeast Monsoon (Oct–Dec) is its main rainy season and typically the most delay-prone.")

    check = st.button("Check Delay & ETA", type="primary", width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    if not check:
        st.markdown(
            '<div class="result-placeholder">Select a route and stop on the left, '
            'then click <b>&nbsp;Check Delay &amp; ETA&nbsp;</b> to see the prediction here — '
            'delay estimate, likely causes, corrected ETA, and a stop map.</div>',
            unsafe_allow_html=True)
    elif model is None:
        st.error("Model not available. Run: python src/train_model.py")
    else:
        peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
        tp = time_period_for(hour)
        month_map = {"Summer": 5, "Southwest_Monsoon": 8, "Northeast_Monsoon": 11, "Winter": 1}

        inp = dict(
            Hour=hour, Minute=minute, Weekday=weekday, Is_Weekend=is_weekend,
            Peak_Hour=peak_hour, Month=month_map.get(season, 6),
            Weather=weather, Temperature_C=typical["Temperature_C"],
            Rainfall_mm=rainfall, Humidity_pct=typical["Humidity_pct"],
            Traffic_Level=traffic_lvl, Traffic_Index=traffic_idx,
            Distance_km_Proxy=typical["Distance_km_Proxy"], Season=season, Time_Period=tp,
            Passenger_Count_Simulated=40,
        )

        from src.prediction import predict_delay, get_delay_causes
        result = predict_delay(inp, model, feature_names)
        pred, cat = result["predicted_delay"], result["delay_category"]
        eta_h, eta_m = add_minutes_to_clock(hour, minute, pred)
        colour = STATUS_COLOURS.get(cat, "#2980b9")

        # ── Row 1: ETA block + Route Reliability snapshot ───────────────────
        rel = eda.route_reliability_stats(df_full, route, weekday=weekday, time_period=tp)

        r1a, r1b = st.columns([1.1, 1], gap="medium")
        with r1a:
            sched_note = (f"Simulated departure {hour:02d}:{minute:02d}" if custom_time
                          else f"Scheduled for {typical['Scheduled_Arrival']}")
            st.markdown(f"""
            <div class="eta-block">
              <p class="eta-label">Corrected Arrival Time</p>
              <p class="eta-time">{eta_h:02d}:{eta_m:02d}</p>
              <p class="eta-sub">{sched_note} · +{pred:.0f} min expected delay</p>
              <span class="status-chip" style="background:{colour};">{cat} · {pred:.1f} min</span>
            </div>
            """, unsafe_allow_html=True)

        with r1b:
            if rel:
                route_avg = rel["route_avg"]
                diff_pct = ((pred - route_avg) / route_avg * 100) if route_avg > 0 else 0
                if diff_pct > 5:
                    compare_html = f'<span class="compare-up">{diff_pct:.0f}% worse than usual ▲</span>'
                elif diff_pct < -5:
                    compare_html = f'<span class="compare-down">{abs(diff_pct):.0f}% better than usual ▼</span>'
                else:
                    compare_html = '<span style="color:#8a97a3;">about typical for this route</span>'

                similar_row = ""
                if rel.get("similar_avg") is not None:
                    similar_row = (f'<div class="reliability-row"><span>Similar trips ({weekday}, {tp})</span>'
                                    f'<b>{rel["similar_avg"]:.1f} min</b></div>')

                tip_html = ""
                if "best_hour" in rel:
                    tip_html = (f'<div class="tip-line">💡 This route tends to run least delayed around '
                                f'<b>{rel["best_hour"]:02d}:00</b> (avg {rel["best_hour_delay"]:.1f} min).</div>')

                st.markdown(f"""
                <div class="reliability-card">
                  <h4>Route Reliability Snapshot</h4>
                  <div class="reliability-row"><span>This prediction</span> {compare_html}</div>
                  <div class="reliability-row"><span>Route's historical average</span><b>{route_avg:.1f} min</b></div>
                  {similar_row}
                  <div class="reliability-row"><span>On-time rate (route history)</span><b>{rel['on_time_pct']:.0f}%</b></div>
                  {tip_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="reliability-card"><h4>Route Reliability Snapshot</h4>'
                            '<p style="color:#8a97a3; font-size:.85rem;">Not enough historical data for this route.</p></div>',
                            unsafe_allow_html=True)

        st.markdown(f"""
        <div class="schedule-strip">
          <div>
            <span>📍 <b>Route {route}</b> · {stop}</span>
            <div class="od-strip">🛫 {typical['Origin']} &nbsp;→&nbsp; 🏁 {typical['Destination']}</div>
          </div>
          <span>🗓️ {weekday}, {hour:02d}:{minute:02d} departure</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Row 2: Journey summary + Likely causes ──────────────────────────
        r2a, r2b = st.columns([1, 1.1], gap="medium")
        with r2a:
            emoji = WEATHER_EMOJI.get(weather, "🌤️")
            tcolour = TRAFFIC_COLOURS.get(traffic_lvl, "#2980b9")
            st.markdown(f"""
            <div class="summary-card">
              <h4>Journey Summary</h4>
              <p class="big-stat-label">Route Distance</p>
              <p class="big-stat">{typical['Distance_km_Proxy']:.0f} km</p>
              <div class="pill-row" style="margin-top:.6rem;">
                <span class="pill">{emoji} {weather}</span>
                <span class="pill">🌡️ {typical['Temperature_C']:.0f}°C</span>
                <span class="pill">💧 {typical['Humidity_pct']:.0f}% humidity</span>
                <span class="pill" style="background:{tcolour}22; color:{tcolour};">🚦 {traffic_lvl} traffic</span>
                <span class="pill">🕒 {tp}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with r2b:
            causes = get_delay_causes(inp, model, feature_names, df_reference=df_full, top_k=3)
            rank_labels = ["Primary Factor", "Secondary Factor", "Additional Factor"]
            rank_icons = ["🥇", "🥈", "🥉"]
            chips = ""
            for i, c in enumerate(causes):
                chips += (f'<div class="cause-chip"><span class="rank-badge">{rank_icons[i] if i < 3 else "•"}</span>'
                          f'<b>{c["label"]}</b>'
                          f'<div style="font-size:.75rem; color:#8a97a3; margin-left:2rem;">'
                          f'{rank_labels[i] if i < 3 else ""}</div>'
                          f'<span style="margin-left:2rem;">{c["reason"]}</span></div>')
            st.markdown(f'<div class="summary-card"><h4>Likely Causes of This Delay</h4>{chips}</div>',
                        unsafe_allow_html=True)

        # ── Row 3: Mini map of stops on this route ──────────────────────────
        st.plotly_chart(eda.fig_route_stops_map(df_full, route, stop), width="stretch")

        st.caption("Delay is a model estimate based on historical patterns (weather, traffic and timing "
                   "fields are simulated/experimental for this project) — not a live operational feed.")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# City-wide insights — tucked away, richer than a plain expander
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📊 City-wide Delay Insights", expanded=False):
    kpi = eda.compute_kpis(df_full)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg delay across the city", f"{kpi['avg_delay']:.1f} min")
    k2.metric("Trips with High/Severe delay", f"{kpi['high_delay_pct']:.1f}%")
    k3.metric("Routes tracked", f"{kpi['num_routes']:,}")
    k4.metric("Stops tracked", f"{kpi['num_stops']:,}")

    tab_time, tab_routes, tab_weather = st.tabs(["⏰ Time Patterns", "🛣️ Routes", "🌦️ Weather & Traffic"])

    with tab_time:
        st.plotly_chart(eda.fig_hour_weekday_heatmap(df_full), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(eda.fig_hourly_delay(df_full), width="stretch")
        with c2:
            st.plotly_chart(eda.fig_peak_comparison(df_full), width="stretch")

    with tab_routes:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(eda.fig_route_delay(df_full, top_n=8, worst=True), width="stretch")
        with c2:
            st.plotly_chart(eda.fig_route_delay(df_full, top_n=8, worst=False), width="stretch")
        insight = eda.route_insight(df_full)
        if insight:
            st.info(insight)

    with tab_weather:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(eda.fig_weather_box(df_full), width="stretch")
        with c2:
            st.plotly_chart(eda.fig_traffic_level_bar(df_full), width="stretch")
        st.caption("Correlation between weather/traffic and delay does not imply causation — "
                   "these fields are simulated for this project.")
