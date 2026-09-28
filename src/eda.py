"""
eda.py — Exploratory Data Analysis helpers used by the Streamlit dashboard.

All functions accept a DataFrame and return Plotly figures or summary dicts
so the dashboard stays thin and this module can be tested independently.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import TARGET_COL, safe_log

# Check whether statsmodels is available (needed for OLS trendlines in Plotly)
try:
    import statsmodels  # noqa: F401
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False



# ── Colour palette ────────────────────────────────────────────────────────────
SEVERITY_COLOURS = {
    "On Time":  "#27ae60",
    "Moderate": "#f39c12",
    "High":     "#e67e22",
    "Severe":   "#e74c3c",
}
TEAL   = "#1abc9c"
BLUE   = "#2980b9"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"


# ─────────────────────────────────────────────────────────────────────────────
# KPI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Return a dictionary of key performance indicators."""
    delay = df[TARGET_COL] if TARGET_COL in df.columns else pd.Series(dtype=float)
    kpis = {
        "total_records":    len(df),
        "avg_delay":        round(delay.mean(), 2) if not delay.empty else 0,
        "max_delay":        round(delay.max(),  2) if not delay.empty else 0,
        "min_delay":        round(delay.min(),  2) if not delay.empty else 0,
        "median_delay":     round(delay.median(),2) if not delay.empty else 0,
        "num_routes":       df["Route_Name"].nunique() if "Route_Name" in df.columns else 0,
        "num_stops":        df["Representative_Stop"].nunique() if "Representative_Stop" in df.columns else 0,
        "high_delay_pct":   round(100 * (delay > 15).sum() / max(len(delay), 1), 1),
        "avg_rainfall":     round(df["Rainfall_mm"].mean(), 2) if "Rainfall_mm" in df.columns else 0,
        "avg_traffic_idx":  round(df["Traffic_Index"].mean(), 2) if "Traffic_Index" in df.columns else 0,
    }
    return kpis


def route_reliability_stats(df: pd.DataFrame, route: str, weekday: str = None, time_period: str = None) -> dict:
    """
    Historical context for one route: overall average delay, on-time rate,
    the average for trips under similar conditions, and the best hour to travel.
    """
    if TARGET_COL not in df.columns or "Route_Name" not in df.columns:
        return {}
    route_df = df[df["Route_Name"] == route]
    if route_df.empty:
        return {}

    stats = {
        "route_avg":   float(route_df[TARGET_COL].mean()),
        "route_n":     int(len(route_df)),
        "on_time_pct": float((route_df[TARGET_COL] < 5).mean() * 100),
    }

    similar_df = route_df
    if weekday and "Weekday" in route_df.columns:
        similar_df = similar_df[similar_df["Weekday"] == weekday]
    if time_period and "Time_Period" in route_df.columns:
        similar_df = similar_df[similar_df["Time_Period"] == time_period]
    if len(similar_df) >= 5:
        stats["similar_avg"] = float(similar_df[TARGET_COL].mean())
        stats["similar_n"] = int(len(similar_df))
    else:
        stats["similar_avg"] = None
        stats["similar_n"] = 0

    if "Hour" in route_df.columns:
        by_hour = route_df.groupby("Hour")[TARGET_COL].mean()
        if not by_hour.empty:
            stats["best_hour"] = int(by_hour.idxmin())
            stats["best_hour_delay"] = float(by_hour.min())

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# DELAY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def fig_delay_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x=TARGET_COL, nbins=60,
        title="Distribution of Delay Minutes",
        labels={TARGET_COL: "Delay (minutes)"},
        color_discrete_sequence=[BLUE],
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05)
    return fig


def fig_delay_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, y=TARGET_COL,
        title="Delay Distribution & Outliers",
        labels={TARGET_COL: "Delay (minutes)"},
        color_discrete_sequence=[TEAL],
        template="plotly_white",
    )
    return fig


def fig_severity_bar(df: pd.DataFrame) -> go.Figure:
    if "Delay_Severity" not in df.columns:
        return go.Figure()
    counts = df["Delay_Severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]
    order = ["On Time", "Moderate", "High", "Severe"]
    counts["Severity"] = pd.Categorical(counts["Severity"], categories=order, ordered=True)
    counts = counts.sort_values("Severity")
    colours = [SEVERITY_COLOURS.get(s, BLUE) for s in counts["Severity"]]
    fig = px.bar(
        counts, x="Severity", y="Count",
        title="Delay Severity Distribution",
        color="Severity",
        color_discrete_map=SEVERITY_COLOURS,
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fig_route_delay(df: pd.DataFrame, top_n: int = 10, worst: bool = True) -> go.Figure:
    if "Route_Name" not in df.columns:
        return go.Figure()
    grp = df.groupby("Route_Name")[TARGET_COL].mean().reset_index()
    grp.columns = ["Route", "Avg_Delay"]
    grp = grp.sort_values("Avg_Delay", ascending=not worst).head(top_n)
    title = f"Top {top_n} {'Most' if worst else 'Least'} Delayed Routes"
    fig = px.bar(
        grp, x="Avg_Delay", y="Route",
        orientation="h",
        title=title,
        labels={"Avg_Delay": "Avg Delay (min)", "Route": ""},
        color="Avg_Delay",
        color_continuous_scale="RdYlGn_r" if worst else "RdYlGn",
        template="plotly_white",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def route_insight(df: pd.DataFrame) -> str:
    if "Route_Name" not in df.columns:
        return ""
    grp = df.groupby("Route_Name")[TARGET_COL].mean()
    worst = grp.idxmax(); best = grp.idxmin()
    return (
        f"**{worst}** shows the highest average observed delay "
        f"({grp[worst]:.1f} min) in the filtered records. "
        f"**{best}** shows the lowest ({grp[best]:.1f} min)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fig_hourly_delay(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Hour")[TARGET_COL].mean().reset_index()
    grp.columns = ["Hour", "Avg_Delay"]
    fig = px.line(
        grp, x="Hour", y="Avg_Delay",
        title="Average Delay by Hour of Day",
        labels={"Hour": "Hour (24h)", "Avg_Delay": "Avg Delay (min)"},
        markers=True,
        color_discrete_sequence=[BLUE],
        template="plotly_white",
    )
    return fig


def fig_weekday_delay(df: pd.DataFrame) -> go.Figure:
    if "Weekday" not in df.columns:
        return go.Figure()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grp = df.groupby("Weekday")[TARGET_COL].mean().reset_index()
    grp["Weekday"] = pd.Categorical(grp["Weekday"], categories=order, ordered=True)
    grp = grp.sort_values("Weekday")
    fig = px.bar(
        grp, x="Weekday", y=TARGET_COL,
        title="Average Delay by Weekday",
        labels={TARGET_COL: "Avg Delay (min)"},
        color="Weekday",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    return fig


def fig_peak_comparison(df: pd.DataFrame) -> go.Figure:
    if "Peak_Hour" not in df.columns:
        return go.Figure()
    grp = df.groupby("Peak_Hour")[TARGET_COL].mean().reset_index()
    grp["Period"] = grp["Peak_Hour"].map({0: "Non-Peak", 1: "Peak"})
    fig = px.bar(
        grp, x="Period", y=TARGET_COL,
        title="Peak Hour vs Non-Peak Average Delay",
        labels={TARGET_COL: "Avg Delay (min)"},
        color="Period",
        color_discrete_map={"Peak": ORANGE, "Non-Peak": TEAL},
        template="plotly_white",
    )
    return fig


def fig_time_period_delay(df: pd.DataFrame) -> go.Figure:
    if "Time_Period" not in df.columns:
        return go.Figure()
    order = ["Night","Morning Peak","Mid-Morning","Afternoon Peak","Evening","Late Night"]
    grp = df.groupby("Time_Period")[TARGET_COL].mean().reset_index()
    grp["Time_Period"] = pd.Categorical(grp["Time_Period"], categories=order, ordered=True)
    grp = grp.sort_values("Time_Period")
    fig = px.bar(
        grp, x="Time_Period", y=TARGET_COL,
        title="Average Delay by Time Period",
        labels={"Time_Period": "Time Period", TARGET_COL: "Avg Delay (min)"},
        color=TARGET_COL,
        color_continuous_scale="Oranges",
        template="plotly_white",
    )
    return fig


def fig_hour_weekday_heatmap(df: pd.DataFrame) -> go.Figure:
    """Average delay heatmap: hour of day (x) vs weekday (y)."""
    if "Weekday" not in df.columns or "Hour" not in df.columns:
        return go.Figure()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df.pivot_table(index="Weekday", columns="Hour", values=TARGET_COL, aggfunc="mean")
    pivot = pivot.reindex(order)
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="", color="Avg Delay (min)"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        template="plotly_white",
        title="Average Delay — Hour of Day vs Weekday",
    )
    fig.update_xaxes(dtick=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fig_rainfall_scatter(df: pd.DataFrame) -> go.Figure:
    if "Rainfall_mm" not in df.columns:
        return go.Figure()
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Rainfall_mm", y=TARGET_COL,
        title="Rainfall vs Delay (sample 3,000 records)",
        labels={"Rainfall_mm": "Rainfall (mm)", TARGET_COL: "Delay (min)"},
        opacity=0.5,
        trendline="ols" if STATSMODELS_AVAILABLE else None,
        color_discrete_sequence=[BLUE],
        template="plotly_white",
    )
    return fig


def fig_weather_box(df: pd.DataFrame) -> go.Figure:
    if "Weather" not in df.columns:
        return go.Figure()
    fig = px.box(
        df, x="Weather", y=TARGET_COL,
        title="Delay Distribution by Weather Condition",
        labels={"Weather": "Weather", TARGET_COL: "Delay (min)"},
        color="Weather",
        color_discrete_sequence=px.colors.qualitative.Safe,
        template="plotly_white",
    )
    return fig


def fig_temperature_scatter(df: pd.DataFrame) -> go.Figure:
    if "Temperature_C" not in df.columns:
        return go.Figure()
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Temperature_C", y=TARGET_COL,
        title="Temperature vs Delay",
        labels={"Temperature_C": "Temperature (°C)", TARGET_COL: "Delay (min)"},
        opacity=0.5,
        trendline="ols" if STATSMODELS_AVAILABLE else None,
        color_discrete_sequence=[ORANGE],
        template="plotly_white",
    )
    return fig


def fig_humidity_scatter(df: pd.DataFrame) -> go.Figure:
    if "Humidity_pct" not in df.columns:
        return go.Figure()
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Humidity_pct", y=TARGET_COL,
        title="Humidity vs Delay",
        labels={"Humidity_pct": "Humidity (%)", TARGET_COL: "Delay (min)"},
        opacity=0.5,
        trendline="ols" if STATSMODELS_AVAILABLE else None,
        color_discrete_sequence=[PURPLE],
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fig_traffic_level_bar(df: pd.DataFrame) -> go.Figure:
    if "Traffic_Level" not in df.columns:
        return go.Figure()
    order = ["Low","Moderate","High","Very High"]
    grp = df.groupby("Traffic_Level")[TARGET_COL].mean().reset_index()
    grp["Traffic_Level"] = pd.Categorical(grp["Traffic_Level"], categories=order, ordered=True)
    grp = grp.sort_values("Traffic_Level")
    fig = px.bar(
        grp, x="Traffic_Level", y=TARGET_COL,
        title="Average Delay by Traffic Level",
        labels={"Traffic_Level": "Traffic Level", TARGET_COL: "Avg Delay (min)"},
        color="Traffic_Level",
        color_discrete_map={"Low":TEAL,"Moderate":BLUE,"High":ORANGE,"Very High":"#c0392b"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    return fig


def fig_traffic_index_scatter(df: pd.DataFrame) -> go.Figure:
    if "Traffic_Index" not in df.columns:
        return go.Figure()
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Traffic_Index", y=TARGET_COL,
        title="Traffic Index vs Delay",
        labels={"Traffic_Index": "Traffic Index", TARGET_COL: "Delay (min)"},
        opacity=0.5,
        trendline="ols" if STATSMODELS_AVAILABLE else None,
        color_discrete_sequence=[ORANGE],
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL / MAP
# ─────────────────────────────────────────────────────────────────────────────

def fig_spatial_map(df: pd.DataFrame) -> go.Figure:
    needed = ["Latitude","Longitude",TARGET_COL]
    if not all(c in df.columns for c in needed):
        return go.Figure()
    sample = df.dropna(subset=needed).sample(min(5000, len(df)), random_state=42)

    hover_cols = [c for c in ["Route_Name","Representative_Stop","Delay_Severity"] if c in sample.columns]

    fig = px.scatter_mapbox(
        sample,
        lat="Latitude", lon="Longitude",
        color=TARGET_COL,
        size=TARGET_COL,
        size_max=14,
        color_continuous_scale="RdYlGn_r",
        hover_data=hover_cols,
        zoom=10,
        center={"lat": 13.05, "lon": 80.22},
        mapbox_style="carto-positron",
        title="Representative Stop Location — Delay Heatmap",
        template="plotly_white",
        labels={TARGET_COL: "Delay (min)"},
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def fig_route_stops_map(df: pd.DataFrame, route: str, selected_stop: str = None) -> go.Figure:
    """Show the stops recorded for one route, highlighting the selected stop."""
    needed = ["Latitude", "Longitude", "Route_Name", "Representative_Stop"]
    if not all(c in df.columns for c in needed):
        return go.Figure()
    sub = df[df["Route_Name"] == route][needed + ([TARGET_COL] if TARGET_COL in df.columns else [])]
    sub = sub.dropna(subset=["Latitude", "Longitude"]).drop_duplicates(subset=["Representative_Stop"])
    if sub.empty:
        return go.Figure()

    sub["is_selected"] = sub["Representative_Stop"] == selected_stop
    fig = go.Figure()

    others = sub[~sub["is_selected"]]
    fig.add_trace(go.Scattermapbox(
        lat=others["Latitude"], lon=others["Longitude"],
        mode="markers",
        marker=dict(size=9, color=BLUE, opacity=0.65),
        text=others["Representative_Stop"],
        hovertemplate="%{text}<extra></extra>",
        name="Other stops",
    ))

    sel = sub[sub["is_selected"]]
    if not sel.empty:
        fig.add_trace(go.Scattermapbox(
            lat=sel["Latitude"], lon=sel["Longitude"],
            mode="markers",
            marker=dict(size=20, color="#e74c3c"),
            text=sel["Representative_Stop"],
            hovertemplate="📍 %{text} (selected)<extra></extra>",
            name="Selected stop",
        ))
        center = {"lat": float(sel["Latitude"].iloc[0]), "lon": float(sel["Longitude"].iloc[0])}
    else:
        center = {"lat": float(sub["Latitude"].mean()), "lon": float(sub["Longitude"].mean())}

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=center, zoom=10),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title=f"Stops recorded for Route {route}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        height=360,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def fig_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    desired = [TARGET_COL, "Rainfall_mm", "Temperature_C", "Humidity_pct",
               "Traffic_Index", "Distance_km_Proxy", "Hour", "Month",
               "Passenger_Count_Simulated", "Passenger_Disruption_Score"]
    cols = [c for c in desired if c in df.columns]
    if len(cols) < 2:
        return go.Figure()
    corr = df[cols].corr().round(3)
    fig = px.imshow(
        corr, text_auto=True,
        title="Correlation Heatmap (Numerical Features)",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template="plotly_white",
        aspect="auto",
    )
    fig.update_layout(margin={"t":60})
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PASSENGER IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def fig_passenger_disruption_by_route(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    col = "Passenger_Disruption_Score"
    if col not in df.columns or "Route_Name" not in df.columns:
        return go.Figure()
    grp = df.groupby("Route_Name")[col].mean().nlargest(top_n).reset_index()
    grp.columns = ["Route", "Avg_Disruption"]
    fig = px.bar(
        grp, x="Avg_Disruption", y="Route", orientation="h",
        title=f"Top {top_n} Routes by Avg Passenger Disruption Score (Simulated)",
        labels={"Avg_Disruption": "Avg Disruption Score", "Route": ""},
        color="Avg_Disruption",
        color_continuous_scale="Reds",
        template="plotly_white",
    )
    return fig


def fig_passenger_disruption_by_severity(df: pd.DataFrame) -> go.Figure:
    col = "Passenger_Disruption_Score"
    if col not in df.columns or "Delay_Severity" not in df.columns:
        return go.Figure()
    grp = df.groupby("Delay_Severity")[col].mean().reset_index()
    fig = px.bar(
        grp, x="Delay_Severity", y=col,
        title="Avg Passenger Disruption Score by Delay Severity (Simulated)",
        color="Delay_Severity",
        color_discrete_map=SEVERITY_COLOURS,
        template="plotly_white",
    )
    return fig


def fig_delay_vs_passenger_count(df: pd.DataFrame) -> go.Figure:
    col = "Passenger_Count_Simulated"
    if col not in df.columns:
        return go.Figure()
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x=col, y=TARGET_COL,
        title="Delay vs Passenger Count (Simulated)",
        labels={col: "Passenger Count (Simulated)", TARGET_COL: "Delay (min)"},
        opacity=0.5,
        trendline="ols" if STATSMODELS_AVAILABLE else None,
        color_discrete_sequence=[PURPLE],
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig_actual_vs_predicted(y_test, y_pred, model_name: str = "") -> go.Figure:
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    lo = min(y_test.min(), y_pred.min()) - 2
    hi = max(y_test.max(), y_pred.max()) + 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(color=BLUE, opacity=0.4, size=4),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="red", dash="dash"),
        name="Perfect Fit",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Delay — {model_name}",
        xaxis_title="Actual Delay (min)",
        yaxis_title="Predicted Delay (min)",
        template="plotly_white",
    )
    return fig


def fig_residuals(y_test, y_pred, model_name: str = "") -> go.Figure:
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    residuals = y_test - y_pred
    fig = px.histogram(
        x=residuals, nbins=60,
        title=f"Residuals Distribution — {model_name}",
        labels={"x": "Residual (Actual − Predicted)"},
        color_discrete_sequence=[ORANGE],
        template="plotly_white",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    return fig


def fig_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    if results_df.empty:
        return go.Figure()
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["MAE (lower=better)",
                                        "RMSE (lower=better)",
                                        "R² (higher=better)"])
    colours = [TEAL, BLUE, ORANGE, PURPLE][:len(results_df)]

    for i, metric in enumerate(["MAE", "RMSE", "R2"], start=1):
        fig.add_trace(
            go.Bar(
                x=results_df["Model"],
                y=results_df[metric],
                marker_color=colours,
                showlegend=False,
                text=results_df[metric].round(3),
                textposition="outside",
            ),
            row=1, col=i,
        )
    fig.update_layout(
        title_text="Model Performance Comparison",
        template="plotly_white",
        height=420,
    )
    return fig


def fig_feature_importance(fi_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if fi_df is None or fi_df.empty:
        return go.Figure()
    top = fi_df.head(top_n).copy()
    fig = px.bar(
        top, x="Importance", y="Feature", orientation="h",
        title=f"Top {top_n} Predictive Factors (Feature Importance)",
        color="Importance",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig
