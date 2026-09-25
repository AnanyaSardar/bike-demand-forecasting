# day5_quantile.py
# Goal: Train 3 LightGBM models (low, median, high estimates) to show a confidence band
# instead of just a single point prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

# --- Load and prep data (same as Day 4) ---
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

feature_cols = [
    "hour", "day_of_week", "month", "year",
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "lag_1", "lag_24", "rolling_mean_3"
]

X = df[feature_cols]
y = df["count"]

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_test = df["datetime"].iloc[split_index:]

print(f"Train size: {len(X_train)} rows")
print(f"Test size: {len(X_test)} rows")

# --- STEP 1: Define a helper to train a model at a specific quantile ---
def train_quantile_model(alpha):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,          # this is the quantile level (0.1, 0.5, or 0.9)
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# --- STEP 2: Train 3 models - low (10th), median (50th), high (90th) ---
print("\nTraining low estimate model (10th percentile)...")
model_low = train_quantile_model(0.1)

print("Training median model (50th percentile)...")
model_mid = train_quantile_model(0.5)

print("Training high estimate model (90th percentile)...")
model_high = train_quantile_model(0.9)

# --- STEP 3: Predict with all 3 models ---
pred_low = np.clip(model_low.predict(X_test), 0, None)
pred_mid = np.clip(model_mid.predict(X_test), 0, None)
pred_high = np.clip(model_high.predict(X_test), 0, None)

# Safety check: low should never be above high (can happen rarely with quantile models)
pred_low = np.minimum(pred_low, pred_high)

# --- STEP 4: Evaluate using Pinball Loss (the correct metric for quantile predictions) ---
def pinball_loss(actual, predicted, quantile):
    diff = actual - predicted
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))

loss_low = pinball_loss(y_test.values, pred_low, 0.1)
loss_mid = pinball_loss(y_test.values, pred_mid, 0.5)
loss_high = pinball_loss(y_test.values, pred_high, 0.9)

print("\n--- Pinball Loss (lower is better) ---")
print(f"10th percentile model: {loss_low:.2f}")
print(f"50th percentile model: {loss_mid:.2f}")
print(f"90th percentile model: {loss_high:.2f}")

# --- STEP 5: Check coverage - what % of actual values fell inside our band? ---
inside_band = ((y_test.values >= pred_low) & (y_test.values <= pred_high))
coverage = inside_band.mean() * 100
print(f"\nCoverage: {coverage:.1f}% of actual values fell inside the 10-90 band")
print("(Ideally this should be close to 80%, since we used 10th-90th percentile)")

# --- STEP 6: Visualize the confidence band ---
plt.figure(figsize=(14, 6))
n_show = 200

plt.fill_between(
    dates_test.values[:n_show],
    pred_low[:n_show],
    pred_high[:n_show],
    color="steelblue", alpha=0.25, label="80% Confidence Band"
)
plt.plot(dates_test.values[:n_show], y_test.values[:n_show],
         label="Actual", color="black", linewidth=1.5)
plt.plot(dates_test.values[:n_show], pred_mid[:n_show],
         label="Median Prediction", color="steelblue", linewidth=1.5)

plt.title("Demand Forecast with Uncertainty Band (first 200 test hours)")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("day5_uncertainty_band.png")
print("\nSaved: day5_uncertainty_band.png")
print("Done! This is your 'wow factor' chart - a real confidence band, not just a point forecast.")