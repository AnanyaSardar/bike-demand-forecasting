# day4_lightgbm.py
# Goal: Build a LightGBM model using engineered features (hour, weather, lags)
# and compare against naive baseline + Prophet from Day 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

# --- Load data ---
df = pd.read_csv("data/train.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# --- STEP 1: Feature engineering ---
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

# Lag features: what was demand 1 hour ago, and 24 hours ago (same time yesterday)?
df["lag_1"] = df["count"].shift(1)
df["lag_24"] = df["count"].shift(24)

# Rolling average: average demand over the last 3 hours (smooths out noise)
df["rolling_mean_3"] = df["count"].shift(1).rolling(window=3).mean()

# Drop the first 24 rows since they don't have lag_24 available yet
df = df.dropna().reset_index(drop=True)

# --- STEP 2: Define features (X) and target (y) ---
feature_cols = [
    "hour", "day_of_week", "month", "year",
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "lag_1", "lag_24", "rolling_mean_3"
]

X = df[feature_cols]
y = df["count"]

# --- STEP 3: Time-based train/test split (same 80/20 as Day 3) ---
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_test = df["datetime"].iloc[split_index:]

print(f"Train size: {len(X_train)} rows")
print(f"Test size: {len(X_test)} rows")

# --- STEP 4: Train LightGBM model ---
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

print("\nTraining LightGBM model...")
model.fit(X_train, y_train)

# --- STEP 5: Predict and evaluate ---
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, None)  # demand can't be negative

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

lgb_rmse = rmse(y_test.values, predictions)
print(f"\nLightGBM RMSE: {lgb_rmse:.2f}")

# --- STEP 6: Compare with Day 3 results (typed in manually from your Day 3 output) ---
naive_rmse = 132.61   # from your Day 3 run
prophet_rmse = 152.99  # from your Day 3 run

print("\n--- FULL COMPARISON ---")
print(f"Naive Baseline RMSE: {naive_rmse:.2f}")
print(f"Prophet RMSE:        {prophet_rmse:.2f}")
print(f"LightGBM RMSE:       {lgb_rmse:.2f}")

best = min(naive_rmse, prophet_rmse, lgb_rmse)
if best == lgb_rmse:
    print("\nLightGBM is the best model so far!")
elif best == naive_rmse:
    print("\nNaive baseline is still winning - surprising but informative!")
else:
    print("\nProphet is still winning.")

# --- STEP 7: Feature importance - which features mattered most? ---
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Feature Importance ---")
print(importance)

plt.figure(figsize=(8, 6))
importance.plot(kind="barh", color="seagreen")
plt.title("LightGBM Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("day4_feature_importance.png")
print("\nSaved: day4_feature_importance.png")

# --- STEP 8: Visualize predictions vs actual (first 200 test points) ---
plt.figure(figsize=(14, 6))
n_show = 200
plt.plot(dates_test.values[:n_show], y_test.values[:n_show],
         label="Actual", color="black", linewidth=1.5)
plt.plot(dates_test.values[:n_show], predictions[:n_show],
         label="LightGBM", color="seagreen", alpha=0.8)
plt.title("Actual vs LightGBM Predictions (first 200 test hours)")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("day4_lightgbm_comparison.png")
print("Saved: day4_lightgbm_comparison.png")
print("\nDone! Compare this graph to day3_comparison.png - much closer to actual?")