# day2_eda.py
# Goal: Dig deeper into WHY demand changes - weather, hour of day, weekday vs weekend

import pandas as pd
import matplotlib.pyplot as plt

# --- Load data (same as Day 1) ---
df = pd.read_csv("data/train.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# --- STEP 1: Extract useful time features from the datetime column ---
# This breaks one column into several, so we can group by them
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()   # Monday, Tuesday, etc.
df["month"] = df["datetime"].dt.month

# --- STEP 2: Average demand by hour of day ---
# This shows if there are "rush hour" spikes
hourly_avg = df.groupby("hour")["count"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_avg.index, hourly_avg.values, marker="o")
plt.title("Average Demand by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Average Count")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day2_hourly_pattern.png")
print("Saved: day2_hourly_pattern.png")

# --- STEP 3: Average demand by day of week ---
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_avg = df.groupby("day_of_week")["count"].mean().reindex(day_order)

plt.figure(figsize=(10, 5))
plt.bar(daily_avg.index, daily_avg.values, color="steelblue")
plt.title("Average Demand by Day of Week")
plt.ylabel("Average Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("day2_weekly_pattern.png")
print("Saved: day2_weekly_pattern.png")

# --- STEP 4: Demand by weather condition ---
# weather: 1=Clear, 2=Mist/Cloudy, 3=Light Rain/Snow, 4=Heavy Rain/Snow
weather_avg = df.groupby("weather")["count"].mean()

plt.figure(figsize=(8, 5))
plt.bar(weather_avg.index.astype(str), weather_avg.values, color="coral")
plt.title("Average Demand by Weather Condition\n(1=Clear, 2=Cloudy, 3=Light Rain, 4=Heavy Rain)")
plt.xlabel("Weather Code")
plt.ylabel("Average Count")
plt.tight_layout()
plt.savefig("day2_weather_pattern.png")
print("Saved: day2_weather_pattern.png")

# --- STEP 5: Correlation check - which numeric columns relate to demand? ---
numeric_cols = ["temp", "atemp", "humidity", "windspeed", "count"]
correlations = df[numeric_cols].corr()["count"].sort_values(ascending=False)

print("\n--- Correlation with demand (count) ---")
print(correlations)

print("\nDone! Check the 3 PNG files saved in your folder.")
print("Look at them and think: which pattern is strongest?")