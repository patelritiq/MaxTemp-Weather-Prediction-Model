# FILE: maxtempweatherpredict.py
# AUTHOR: Ritik Pratap Singh Patel
# COMPLETION DATE: 07 May 2024
# DESCRIPTION: A weather prediction model to forecast maximum temperatures based on historical weather data using dataset i.e. weather.csv
# GUIDANCE: Zidio Development
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all messages

# File handler: logs everything (DEBUG and above)
file_handler = logging.FileHandler('model_training.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler: only INFO and above (keeps terminal clean)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add both handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ─── Helper Functions ────────────────────────────────────────────────────────

def pct_diff(old, new):
    return (new - old) / old


def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather


def expand_mean(df):
    return df.expanding(1).mean()


def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    # --- Load Data ---
    logger.info("Starting weather prediction pipeline...")
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "data", "weather.csv")
    weather = pd.read_csv(file_path, index_col="DATE")
    logger.info(f"Data loaded successfully. Shape: {weather.shape}")
    logger.debug(f"First few rows:\n{weather.head()}")

    # --- Clean Data ---
    logger.info("Cleaning data: removing columns with >5% null values...")
    null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
    logger.debug(f"Null percentage per column:\n{null_pct}")

    valid_columns = weather.columns[null_pct < 0.05]
    logger.info(f"Valid columns retained: {list(valid_columns)}")

    weather = weather[valid_columns].copy()
    weather.columns = weather.columns.str.lower()
    weather = weather.ffill()
    logger.info(f"Data cleaned. Remaining nulls: {weather.apply(pd.isnull).sum().sum()}")
    logger.debug(f"Invalid values (9999): {weather.apply(lambda x: (x == 9999).sum()).sum()}")
    logger.debug(f"Data types:\n{weather.dtypes}")

    # --- Explore & Visualize ---
    logger.info("Exploring temporal distribution...")
    weather.index = pd.to_datetime(weather.index)
    year_counts = weather.index.year.value_counts().sort_index()
    logger.info(f"Data spans {year_counts.index.min()} to {year_counts.index.max()}")
    weather["snwd"].plot()
    plt.xlabel('Year')
    plt.ylabel('Snow Depth')
    plt.title('Snow Depth Over Years')
    plt.show()

    # --- Define Target ---
    logger.info("Defining target variable (next day's max temperature)...")
    weather["target"] = weather.shift(-1)["tmax"]
    weather = weather.ffill()
    logger.info(f"Target variable created. Shape: {weather.shape}")

    # --- Initial Model & Backtest ---
    logger.info("Training initial Ridge Regression model (alpha=0.1)...")
    rr = Ridge(alpha=0.1)
    predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]

    predictions = backtest(weather, rr, predictors)
    mae_initial = mean_absolute_error(predictions["actual"], predictions["prediction"])
    logger.info(f"Initial model MAE: {mae_initial:.2f} °C")
    logger.debug(f"Top 5 feature coefficients:\n{pd.Series(rr.coef_, index=predictors).nlargest(5)}")

    # --- Feature Engineering: Rolling Windows ---
    logger.info("Engineering rolling window features (3-day and 14-day horizons)...")
    rolling_horizons = [3, 14]
    for horizon in rolling_horizons:
        for col in ["tmax", "tmin", "prcp"]:
            weather = compute_rolling(weather, horizon, col)
    logger.info(f"Rolling features added. New shape: {weather.shape}")

    # --- Feature Engineering: Temporal Aggregates ---
    logger.info("Engineering temporal aggregate features (monthly and daily averages)...")
    for col in ["tmax", "tmin", "prcp"]:
        weather[f"month_avg_{col}"] = (
            weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
        )
        weather[f"day_avg_{col}"] = (
            weather[col]
            .groupby(weather.index.day_of_year, group_keys=False)
            .apply(expand_mean)
        )
    logger.info(f"Temporal features added. Final shape: {weather.shape}")

    # --- Final Model & Evaluation ---
    logger.info("Training final model with engineered features...")
    weather = weather.iloc[14:, :]
    weather = weather.fillna(0)
    predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
    predictions = backtest(weather, rr, predictors)
    mae = mean_absolute_error(predictions["actual"], predictions["prediction"])
    mse = mean_squared_error(predictions["actual"], predictions["prediction"])

    logger.info(f"Final model trained successfully")
    logger.info(f"Mean Absolute Error: {mae:.2f} °C")
    logger.info(f"Mean Squared Error: {mse:.2f} °C²")
    logger.debug(f"Worst predictions:\n{predictions.sort_values('diff', ascending=False).head()}")

    # --- Plot: Error Distribution ---
    logger.info("Generating error distribution plot...")
    plt.plot(predictions["diff"].round().value_counts().sort_index() / predictions.shape[0])
    plt.xlabel("Rounded Prediction Error (°C)")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(int(predictions["diff"].min()), int(predictions["diff"].max()) + 1))
    plt.locator_params(axis='x', nbins=15)
    plt.tight_layout()
    plt.show()

    logger.info("Pipeline completed successfully!")
    logger.info(f"Final Results - MAE: {mae:.2f} °C | MSE: {mse:.2f} °C²")


if __name__ == "__main__":
    main()
