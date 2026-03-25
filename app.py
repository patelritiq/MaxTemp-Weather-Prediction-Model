"""
MaxTemp Weather Prediction Dashboard
Streamlit app for interactive temperature prediction
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add src to path for config imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import (
    NULL_THRESHOLD, ROLLING_HORIZONS, FEATURE_COLUMNS,
    EXCLUDE_COLUMNS, LAG_DAYS, MODELS_DIR, BEST_MODEL_FILENAME
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MaxTemp Weather Prediction",
    page_icon="🌡️",
    layout="wide"
)

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(MODELS_DIR, BEST_MODEL_FILENAME)
DATA_PATH  = os.path.join("data", "weather.csv")
MAE        = 4.77   # Best model MAE from training

MODEL_METRICS = {
    "Model":     ["Ridge", "Random Forest", "XGBoost", "LightGBM"],
    "MAE (°C)":  [4.77,    5.02,            4.84,       6.89],
    "RMSE (°C)": [6.10,    6.44,            6.21,       8.43],
    "R²":        [0.8759,  0.8620,          0.8717,     0.7632],
    "MAPE (%)":  [8.88,    9.32,            9.03,       13.18],
}

SEASONAL_METRICS = {
    "Season":    ["Winter", "Spring", "Summer", "Fall"],
    "MAE (°C)":  [5.44,     5.46,     3.82,     4.36],
    "RMSE (°C)": [6.78,     6.95,     4.99,     5.47],
    "R²":        [0.4846,   0.6535,   0.4301,   0.7698],
}

# ─── Data & Model Loading ─────────────────────────────────────────────────────

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, index_col="DATE", parse_dates=True)

    # Clean
    null_pct = df.isnull().sum() / len(df)
    valid_cols = df.columns[null_pct < NULL_THRESHOLD]
    weather = df[valid_cols].copy()
    weather.columns = weather.columns.str.lower()
    weather = weather.ffill()

    # Rolling features
    for horizon in ROLLING_HORIZONS:
        for col in FEATURE_COLUMNS:
            if col in weather.columns:
                label = f"rolling_{horizon}_{col}"
                weather[label] = weather[col].rolling(horizon).mean()
                weather[f"{label}_pct"] = (weather[col] - weather[label]) / weather[label]

    # Temporal aggregates
    for col in FEATURE_COLUMNS:
        if col in weather.columns:
            weather[f"month_avg_{col}"] = (
                weather[col].groupby(weather.index.month, group_keys=False).apply(
                    lambda x: x.expanding(1).mean()
                )
            )
            weather[f"day_avg_{col}"] = (
                weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(
                    lambda x: x.expanding(1).mean()
                )
            )

    # Year feature
    weather["year"] = weather.index.year

    # Lag features
    for lag in LAG_DAYS:
        weather[f"tmax_lag_{lag}"] = weather["tmax"].shift(lag)

    weather = weather.iloc[14:].fillna(0)
    return weather


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    
    # Model not found — train Ridge on the fly for Streamlit Cloud
    st.info("No saved model found. Training Ridge Regression model... (this takes ~30 seconds)")
    from sklearn.linear_model import Ridge
    
    weather = load_and_prepare_data()
    predictors = weather.columns[~weather.columns.isin(EXCLUDE_COLUMNS + ["target"])]
    weather["target"] = weather["tmax"].shift(-1)
    weather = weather.ffill()
    
    model = Ridge(alpha=100.0)
    model.fit(weather[predictors], weather["target"])
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def get_predictors(weather):
    return weather.columns[~weather.columns.isin(EXCLUDE_COLUMNS + ["target"])]


def get_season(month):
    if month in [12, 1, 2]:  return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else:                    return "Fall"


# ─── App ──────────────────────────────────────────────────────────────────────

st.title("🌡️ MaxTemp Weather Prediction Dashboard")
st.markdown("Interactive dashboard for exploring historical weather predictions using Ridge Regression.")
st.markdown("---")

# Load data and model
weather = load_and_prepare_data()
model   = load_model()

predictors   = get_predictors(weather)
min_date     = weather.index.min().date()
max_date     = weather.index.max().date()
default_date = pd.Timestamp("2020-06-15").date()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    selected_date = st.date_input(
        "Select a date",
        value=default_date,
        min_value=min_date,
        max_value=max_date
    )
    st.caption(f"Available range: {min_date} to {max_date}")
    st.markdown("---")
    st.markdown("**Model:** Ridge Regression")
    st.markdown("**Best Alpha:** 100.0 (GridSearchCV)")
    st.markdown("**Training data:** 1970–2022")
    st.markdown("**Records:** 19,287")

# ─── Prediction Section ───────────────────────────────────────────────────────

selected_ts = pd.Timestamp(selected_date)

if selected_ts not in weather.index:
    st.warning(f"No data available for {selected_date}. Please select a date within the dataset range.")
    st.stop()

row        = weather.loc[selected_ts]
features   = row[predictors].values.reshape(1, -1)
prediction = model.predict(features)[0]
season     = get_season(selected_ts.month)

# Get seasonal MAE for uncertainty
seasonal_mae_map = dict(zip(SEASONAL_METRICS["Season"], SEASONAL_METRICS["MAE (°C)"]))
uncertainty      = seasonal_mae_map[season]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted Max Temp", f"{prediction:.1f} °C")
with col2:
    st.metric("Uncertainty (±MAE)", f"± {uncertainty:.2f} °C")
with col3:
    actual = weather.loc[selected_ts, "tmax"] if "tmax" in weather.columns else None
    if actual is not None:
        error = abs(prediction - actual)
        st.metric("Actual TMAX", f"{actual:.1f} °C", delta=f"{prediction - actual:+.1f} °C")
    else:
        st.metric("Actual TMAX", "N/A")
with col4:
    st.metric("Season", season)

st.markdown(f"**Prediction range:** {prediction - uncertainty:.1f}°C — {prediction + uncertainty:.1f}°C (based on seasonal MAE)")

st.markdown("---")

# ─── 14-Day Trend Chart ───────────────────────────────────────────────────────

st.subheader("14-Day Temperature Trend")

trend_data = weather.loc[:selected_ts].tail(14)[["tmax", "tmin"]].copy()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(trend_data.index, trend_data["tmax"], color="#e74c3c", marker="o",
        linewidth=2, markersize=5, label="TMAX (actual)")
ax.plot(trend_data.index, trend_data["tmin"], color="#3498db", marker="o",
        linewidth=2, markersize=5, label="TMIN (actual)")
ax.fill_between(trend_data.index, trend_data["tmin"], trend_data["tmax"],
                alpha=0.1, color="gray")
ax.axvline(x=selected_ts, color="black", linestyle="--", linewidth=1.5, label="Selected date")
ax.scatter([selected_ts], [prediction], color="#e74c3c", s=120, zorder=5,
           marker="*", label=f"Prediction: {prediction:.1f}°C")
ax.fill_between([selected_ts], [prediction - uncertainty], [prediction + uncertainty],
                alpha=0.2, color="#e74c3c", label=f"± {uncertainty:.2f}°C range")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"14-Day Temperature Trend ending {selected_date}")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ─── Model Metrics & Feature Importance ──────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Model Comparison")
    metrics_df = pd.DataFrame(MODEL_METRICS).set_index("Model")
    st.dataframe(
        metrics_df.style
            .highlight_min(subset=["MAE (°C)", "RMSE (°C)", "MAPE (%)"], color="#c8f7c5")
            .highlight_max(subset=["R²"], color="#c8f7c5")
            .format({"MAE (°C)": "{:.2f}", "RMSE (°C)": "{:.2f}",
                     "R²": "{:.4f}", "MAPE (%)": "{:.2f}"}),
        use_container_width=True
    )

with col_right:
    st.subheader("Seasonal Performance")
    seasonal_df = pd.DataFrame(SEASONAL_METRICS).set_index("Season")
    st.dataframe(
        seasonal_df.style
            .highlight_min(subset=["MAE (°C)", "RMSE (°C)"], color="#c8f7c5")
            .highlight_max(subset=["R²"], color="#c8f7c5")
            .format({"MAE (°C)": "{:.2f}", "RMSE (°C)": "{:.2f}", "R²": "{:.4f}"}),
        use_container_width=True
    )

st.markdown("---")

# ─── Seasonal Performance Chart ───────────────────────────────────────────────

st.subheader("Seasonal Performance Breakdown")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
season_colors = ["#3498db", "#2ecc71", "#e74c3c", "#e67e22"]
seasons       = SEASONAL_METRICS["Season"]

# Highlight current season
bar_colors = [
    "#f39c12" if s == season else c
    for s, c in zip(seasons, season_colors)
]

axes[0].bar(seasons, SEASONAL_METRICS["MAE (°C)"], color=bar_colors, edgecolor="white")
axes[0].set_title("MAE by Season")
axes[0].set_ylabel("MAE (°C)")
for i, val in enumerate(SEASONAL_METRICS["MAE (°C)"]):
    axes[0].text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=10)

axes[1].bar(seasons, SEASONAL_METRICS["R²"], color=bar_colors, edgecolor="white")
axes[1].set_title("R² by Season")
axes[1].set_ylabel("R²")
for i, val in enumerate(SEASONAL_METRICS["R²"]):
    axes[1].text(i, val + 0.01, f"{val:.4f}", ha="center", fontsize=10)

fig.suptitle(f"Current season highlighted: {season}", fontsize=11, style="italic")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ─── Feature Importance Chart ─────────────────────────────────────────────────

st.subheader("Feature Importance (Ridge Coefficients — Top 20)")

if hasattr(model, "coef_"):
    coef_series = pd.Series(model.coef_, index=predictors)
    top_features = coef_series.abs().nlargest(20).index
    plot_data    = coef_series[top_features].sort_values()
    colors       = ["#d73027" if v < 0 else "#1a9850" for v in plot_data]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_data.index, plot_data.values, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Importance — Ridge Regression (Top 20)\nGreen = positive influence, Red = negative influence")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Feature importance is only available for Ridge Regression.")

st.markdown("---")
st.caption("MaxTemp Weather Prediction Model | Ritik Pratap Singh Patel | Data spans 1970–2022")
