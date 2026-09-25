# day3_baseline.py
# Goal: Build a simple "naive" baseline, then a Prophet model, and compare them

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# --- Load and prep data (same as before) ---
df = pd.read_csv("data/train.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# --- STEP 1: Split into train and test (TIME-BASED, not random!) ---
# We use the last 20% of the data as "test" - pretending we don't know it yet
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print(f"Train size: {len(train_df)} rows")
print(f"Test size: {len(test_df)} rows")

# --- STEP 2: NAIVE BASELINE ---
# Prediction = the value from exactly 1 step (1 hour) before
naive_predictions = test_df["count"].shift(1)
naive_predictions.iloc[0] = train_df["count"].iloc[-1]  # first prediction uses last train value

# --- STEP 3: Evaluate naive baseline using RMSE ---
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

naive_rmse = rmse(test_df["count"].values, naive_predictions.values)
print(f"\nNaive Baseline RMSE: {naive_rmse:.2f}")

# --- STEP 4: PROPHET MODEL ---
# Prophet needs columns named exactly "ds" (date) and "y" (value)
prophet_train = train_df[["datetime", "count"]].rename(columns={"datetime": "ds", "count": "y"})

model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
print("\nTraining Prophet model... (this may take a minute)")
model.fit(prophet_train)

# --- STEP 5: Predict on the test period ---
future = test_df[["datetime"]].rename(columns={"datetime": "ds"})
forecast = model.predict(future)

prophet_predictions = forecast["yhat"].values
prophet_predictions = np.clip(prophet_predictions, 0, None)  # demand can't be negative

prophet_rmse = rmse(test_df["count"].values, prophet_predictions)
print(f"Prophet RMSE: {prophet_rmse:.2f}")

# --- STEP 6: Compare and plot ---
print("\n--- COMPARISON ---")
print(f"Naive Baseline RMSE: {naive_rmse:.2f}")
print(f"Prophet RMSE:        {prophet_rmse:.2f}")

if prophet_rmse < naive_rmse:
    improvement = ((naive_rmse - prophet_rmse) / naive_rmse) * 100
    print(f"Prophet is {improvement:.1f}% better than naive baseline!")
else:
    print("Naive baseline actually did better - Prophet may need tuning.")

# --- STEP 7: Visualize both predictions vs actual (first 200 test points for clarity) ---
plt.figure(figsize=(14, 6))
n_show = 200
plt.plot(test_df["datetime"].values[:n_show], test_df["count"].values[:n_show],
         label="Actual", color="black", linewidth=1.5)
plt.plot(test_df["datetime"].values[:n_show], naive_predictions.values[:n_show],
         label="Naive Baseline", color="orange", alpha=0.7, linestyle="--")
plt.plot(test_df["datetime"].values[:n_show], prophet_predictions[:n_show],
         label="Prophet", color="steelblue", alpha=0.8)
plt.title("Actual vs Predicted Demand (first 200 test hours)")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("day3_comparison.png")
print("\nSaved: day3_comparison.png")
print("Done! Look at the graph - is Prophet tracking the pattern better than naive?")