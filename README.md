# Bike Sharing Demand Forecasting with Uncertainty Quantification

This project forecasts hourly bike-sharing demand and, instead of stopping at a single
predicted number, produces an 80% confidence interval using quantile regression. It also
includes a small Streamlit dashboard to explore the forecasts interactively.

## What this project does

- Compares three forecasting approaches (naive baseline, Facebook Prophet, and LightGBM)
  to see which actually performs best on hourly demand data
- Improves RMSE by 61% over the naive baseline (132.61 → 51.97) by adding lag features and
  calendar features to LightGBM
- Trains three separate quantile models (10th, 50th, 90th percentile) to produce a
  confidence band around each forecast, rather than a single number
- Wraps the final model in an interactive Streamlit app

## Results

| Model | RMSE | Notes |
|---|---|---|
| Naive Baseline | 132.61 | "Last hour repeats" - held up surprisingly well |
| Facebook Prophet | 152.99 | Smoothed over the sharp rush-hour spikes |
| LightGBM (final) | 51.97 | Best model, using lag + calendar features |

The uncertainty model achieved 72.8% empirical coverage against an 80% target
(evaluated with pinball loss).

## Key insight

Prophet is built for time series, but it still underperformed a plain naive baseline on
this hourly dataset. It captured the daily/weekly seasonality fine, but smoothed over the
sharp rush-hour spikes that actually drive most of the variance in this data. Switching to
LightGBM with explicit lag features (demand 1 hour ago, demand 24 hours ago) and calendar
features (hour, day of week) picked up both the "recent momentum" signal that made the
naive baseline work, and the multi-factor relationships Prophet was going for - giving a
66% RMSE improvement over Prophet.

## Tech stack

- Data processing: Pandas, NumPy
- Modeling: LightGBM (quantile regression), Facebook Prophet
- Visualization: Matplotlib
- App: Streamlit
- Evaluation: RMSE, pinball loss, empirical coverage

## Project structure

```
forecast_project/
├── data/
│   └── train.csv                       # Kaggle Bike Sharing Demand dataset
├── day1_load_data.py                   # Initial data loading & exploration
├── day2_eda.py                         # Exploratory analysis (hourly/weekly/weather patterns)
├── day3_baseline.py                    # Naive baseline vs Prophet comparison
├── day4_lightgbm.py                    # LightGBM with feature engineering
├── day5_quantile.py                    # Quantile regression for uncertainty bands
├── app.py                              # Interactive Streamlit dashboard
└── README.md
```

## Running this project

1. Clone the repo and install dependencies:
   ```
   pip install pandas numpy matplotlib prophet lightgbm streamlit scikit-learn
   ```

2. Download the [Bike Sharing Demand dataset](https://www.kaggle.com/c/bike-sharing-demand)
   from Kaggle and place `train.csv` in a `data/` folder.

3. Run the analysis scripts in order (day1 through day5), or jump straight to the
   interactive app:
   ```
   streamlit run app.py
   ```

## Methodology

1. **Exploratory analysis**: Identified rush-hour demand spikes (8am, 5-6pm), weather
   sensitivity, and temperature correlation
2. **Time-based train/test split** (80/20) — never random splits for time series
3. **Baseline comparison**: Naive vs Prophet, evaluated with RMSE
4. **Feature engineering**: Added lag features (1hr, 24hr), rolling averages, and calendar
   features
5. **Quantile regression**: Trained separate LightGBM models at the 10th, 50th, and 90th
   percentiles to construct confidence intervals
6. **Evaluation**: Used pinball loss (correct metric for quantile predictions) and coverage
   analysis

## Future improvements

- Hierarchical forecasting across multiple bike-station locations
- Incorporating external data (local events, holidays calendar)
- Experimenting with Temporal Fusion Transformer for longer-horizon forecasts
- Automated retraining pipeline with drift detection

---

Built as a self-directed project to learn time series forecasting end to end.