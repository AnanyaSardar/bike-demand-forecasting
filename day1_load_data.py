# day1_load_data.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/train.csv")

print("First 5 rows of data:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

plt.figure(figsize=(12, 5))
plt.plot(df["datetime"], df["count"])
plt.title("Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("day1_plot.png")
plt.show()

print("\nDone! Check day1_plot.png to see your time series plot.")