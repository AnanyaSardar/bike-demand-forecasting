# app.py
# Interactive Streamlit app showing demand forecasts with uncertainty bands
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

st.set_page_config(page_title="Bike Demand Forecaster", layout="wide")

st.title("🚲 Bike Sharing Demand Forecaster")
st.markdown("""
This app predicts hourly bike-sharing demand along with an **80% confidence band**,
showing not just a single number but a realistic range of outcomes.
""")

# --- Load and prep data (cached so it only runs once) ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["lag_1"] = df["count"].shift(1)
    df["lag_24"] = df["count"].shift(24)
    df["rolling_mean_3"] = df["count"].shift(1).rolling(window=3).mean()
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

feature_cols = [
    "hour", "day_of_week", "month", "year",
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "lag_1", "lag_24", "rolling_mean_3"
]

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].reset_index(drop=True)

X_train = train_df[feature_cols]
y_train = train_df["count"]
X_test = test_df[feature_cols]
y_test = test_df["count"]

# --- Train models (cached so it only trains once, not on every interaction) ---
@st.cache_resource
def train_models(X_train, y_train):
    def train_quantile_model(alpha):
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha,
            n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42
        )
        model.fit(X_train, y_train)
        return model

    model_low = train_quantile_model(0.1)
    model_mid = train_quantile_model(0.5)
    model_high = train_quantile_model(0.9)
    return model_low, model_mid, model_high

with st.spinner("Training models... (only happens once)"):
    model_low, model_mid, model_high = train_models(X_train, y_train)

pred_low = np.clip(model_low.predict(X_test), 0, None)
pred_mid = np.clip(model_mid.predict(X_test), 0, None)
pred_high = np.clip(model_high.predict(X_test), 0, None)
pred_low = np.minimum(pred_low, pred_high)

# --- Sidebar controls ---
st.sidebar.header("Controls")
n_hours = st.sidebar.slider("Number of hours to display", 24, len(test_df), 200, step=24)
start_idx = st.sidebar.slider("Start position in test set", 0, len(test_df) - n_hours, 0)

end_idx = start_idx + n_hours

# --- Main chart ---
fig, ax = plt.subplots(figsize=(14, 6))
dates_show = test_df["datetime"].iloc[start_idx:end_idx]

ax.fill_between(
    dates_show, pred_low[start_idx:end_idx], pred_high[start_idx:end_idx],
    color="steelblue", alpha=0.25, label="80% Confidence Band"
)
ax.plot(dates_show, y_test.values[start_idx:end_idx],
        label="Actual", color="black", linewidth=1.5)
ax.plot(dates_show, pred_mid[start_idx:end_idx],
        label="Median Prediction", color="steelblue", linewidth=1.5)

ax.set_title("Demand Forecast with Uncertainty Band")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# --- Metrics ---
col1, col2, col3 = st.columns(3)

def pinball_loss(actual, predicted, quantile):
    diff = actual - predicted
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))

with col1:
    rmse = np.sqrt(np.mean((y_test.values - pred_mid) ** 2))
    st.metric("RMSE (Median Model)", f"{rmse:.2f}")

with col2:
    inside_band = ((y_test.values >= pred_low) & (y_test.values <= pred_high))
    coverage = inside_band.mean() * 100
    st.metric("Coverage (80% target)", f"{coverage:.1f}%")

with col3:
    avg_band_width = (pred_high - pred_low).mean()
    st.metric("Avg Band Width", f"{avg_band_width:.1f}")

st.markdown("---")
st.markdown("""
**How to read this:** The shaded blue region shows where we expect actual demand
to fall 80% of the time. Wider bands = more uncertainty (e.g., during unpredictable
weather), narrower bands = more confidence.
""")