# FILE: maxtempweatherpredict.py
# AUTHOR: Ritik Pratap Singh Patel
# DESCRIPTION: A weather prediction model to forecast maximum temperatures based on historical weather data using dataset i.e. weather.csv
# GUIDANCE: Zidio Development

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from config import (
    ALPHA, BACKTEST_START, BACKTEST_STEP, NULL_THRESHOLD,
    ROLLING_HORIZONS, FEATURE_COLUMNS, EXCLUDE_COLUMNS, ROLLING_WINDOW_OFFSET,
    RANDOM_FOREST_N_ESTIMATORS, XGBOOST_N_ESTIMATORS, LIGHTGBM_N_ESTIMATORS,
    RANDOM_STATE, MODELS_DIR, BEST_MODEL_FILENAME
)

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


def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(actual, predicted))


def backtest(weather, model, predictors, start=BACKTEST_START, step=BACKTEST_STEP):
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
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "..", "data", "weather.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found at {file_path}")
            return
        
        weather = pd.read_csv(file_path, index_col="DATE")
        logger.info(f"Data loaded successfully. Shape: {weather.shape}")
        logger.debug(f"First few rows:\n{weather.head()}")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        return
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return

    # --- Clean Data ---
    logger.info("Cleaning data: removing columns with >5% null values...")
    try:
        null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
        logger.debug(f"Null percentage per column:\n{null_pct}")

        valid_columns = weather.columns[null_pct < NULL_THRESHOLD]
        logger.info(f"Valid columns retained: {list(valid_columns)}")

        weather = weather[valid_columns].copy()
        weather.columns = weather.columns.str.lower()
        weather = weather.ffill()
        logger.info(f"Data cleaned. Remaining nulls: {weather.apply(pd.isnull).sum().sum()}")
        logger.debug(f"Invalid values (9999): {weather.apply(lambda x: (x == 9999).sum()).sum()}")
        logger.debug(f"Data types:\n{weather.dtypes}")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        return

    # --- Explore & Visualize ---
    logger.info("Exploring temporal distribution...")
    try:
        weather.index = pd.to_datetime(weather.index)
        year_counts = weather.index.year.value_counts().sort_index()
        logger.info(f"Data spans {year_counts.index.min()} to {year_counts.index.max()}")
        weather["snwd"].plot()
        plt.xlabel('Year')
        plt.ylabel('Snow Depth')
        plt.title('Snow Depth Over Years')
        plt.show()
    except Exception as e:
        logger.warning(f"Error during visualization: {e}")
        # Continue execution even if visualization fails

    # --- Define Target ---
    logger.info("Defining target variable (next day's max temperature)...")
    try:
        if "tmax" not in weather.columns:
            logger.error("Required column 'tmax' not found in data")
            return
        
        weather["target"] = weather.shift(-1)["tmax"]
        weather = weather.ffill()
        logger.info(f"Target variable created. Shape: {weather.shape}")
    except Exception as e:
        logger.error(f"Error defining target variable: {e}")
        return

    # --- Prepare Predictors ---
    logger.info("Preparing predictors for model training...")
    try:
        predictors = weather.columns[~weather.columns.isin(EXCLUDE_COLUMNS)]
        
        if len(predictors) == 0:
            logger.error("No valid predictors found after filtering")
            return
    except Exception as e:
        logger.error(f"Error preparing predictors: {e}")
        return

    # --- Feature Engineering: Rolling Windows ---
    logger.info("Engineering rolling window features (3-day and 14-day horizons)...")
    try:
        for horizon in ROLLING_HORIZONS:
            for col in FEATURE_COLUMNS:
                if col not in weather.columns:
                    logger.warning(f"Column '{col}' not found, skipping rolling feature for {col}")
                    continue
                weather = compute_rolling(weather, horizon, col)
        logger.info(f"Rolling features added. New shape: {weather.shape}")
    except Exception as e:
        logger.error(f"Error during rolling feature engineering: {e}")
        return

    # --- Feature Engineering: Temporal Aggregates ---
    logger.info("Engineering temporal aggregate features (monthly and daily averages)...")
    try:
        for col in FEATURE_COLUMNS:
            if col not in weather.columns:
                logger.warning(f"Column '{col}' not found, skipping temporal features for {col}")
                continue
            weather[f"month_avg_{col}"] = (
                weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
            )
            weather[f"day_avg_{col}"] = (
                weather[col]
                .groupby(weather.index.day_of_year, group_keys=False)
                .apply(expand_mean)
            )
        logger.info(f"Temporal features added. Final shape: {weather.shape}")
    except Exception as e:
        logger.error(f"Error during temporal feature engineering: {e}")
        return

    # --- Final Model & Evaluation with Model Comparison ---
    logger.info("Training and comparing multiple models with engineered features...")
    try:
        weather = weather.iloc[ROLLING_WINDOW_OFFSET:, :]
        weather = weather.fillna(0)
        predictors = weather.columns[~weather.columns.isin(EXCLUDE_COLUMNS)]
        
        if len(predictors) == 0:
            logger.error("No valid predictors found for model comparison")
            return

        # Define models to compare
        models = {
            'Ridge': Ridge(alpha=ALPHA),
            'Random Forest': RandomForestRegressor(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=XGBOOST_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                verbosity=0
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=LIGHTGBM_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        }
        
        logger.info(f"Training {len(models)} models for comparison...")
        model_results = {}
        
        for model_name, model in tqdm(models.items(), desc="Training models", unit="model"):
            logger.info(f"Training {model_name}...")
            try:
                predictions = backtest(weather, model, predictors)
                
                if predictions.empty:
                    logger.warning(f"No predictions from {model_name}, skipping")
                    continue
                
                mae = mean_absolute_error(predictions["actual"], predictions["prediction"])
                mse = mean_squared_error(predictions["actual"], predictions["prediction"])
                rmse = calculate_rmse(predictions["actual"], predictions["prediction"])
                
                model_results[model_name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'model': model,
                    'predictions': predictions
                }
                
                logger.info(f"{model_name} - MAE: {mae:.2f} °C, RMSE: {rmse:.2f} °C")
            except Exception as e:
                logger.warning(f"Error training {model_name}: {e}")
                continue
        
        if not model_results:
            logger.error("No models trained successfully")
            return
        
        # Create comparison table
        comparison_data = {
            model_name: {
                'MAE (°C)': f"{results['MAE']:.2f}",
                'MSE (°C²)': f"{results['MSE']:.2f}",
                'RMSE (°C)': f"{results['RMSE']:.2f}"
            }
            for model_name, results in model_results.items()
        }
        
        comparison_df = pd.DataFrame(comparison_data).T
        logger.info(f"\n{'='*70}")
        logger.info("MODEL COMPARISON RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"\n{comparison_df.to_string()}\n")
        logger.info(f"{'='*70}")
        
        # Find best model
        best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['MAE'])
        best_model_data = model_results[best_model_name]
        best_mae = best_model_data['MAE']
        best_rmse = best_model_data['RMSE']
        best_mse = best_model_data['MSE']
        
        logger.info(f"\n[BEST MODEL] {best_model_name}")
        logger.info(f"   MAE: {best_mae:.2f} °C")
        logger.info(f"   MSE: {best_mse:.2f} °C²")
        logger.info(f"   RMSE: {best_rmse:.2f} °C\n")
        
        # Save best model
        try:
            if not os.path.exists(MODELS_DIR):
                os.makedirs(MODELS_DIR)
                logger.info(f"Created models directory: {MODELS_DIR}")
            
            model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILENAME)
            joblib.dump(best_model_data['model'], model_path)
            logger.info(f"[SAVED] Best model saved to: {model_path}")
        except Exception as e:
            logger.warning(f"Error saving best model: {e}")
        
        # Use best model for final results
        predictions = best_model_data['predictions']
        mae = best_model_data['MAE']
        mse = best_model_data['MSE']
        rmse = best_model_data['RMSE']
        
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        return

    # --- Plot: Error Distribution ---
    logger.info("Generating error distribution plot...")
    try:
        plt.plot(predictions["diff"].round().value_counts().sort_index() / predictions.shape[0])
        plt.xlabel("Rounded Prediction Error (°C)")
        plt.ylabel("Relative Frequency")
        plt.title("Distribution of Prediction Errors")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(int(predictions["diff"].min()), int(predictions["diff"].max()) + 1))
        plt.locator_params(axis='x', nbins=15)
        plt.tight_layout()
        plt.show()
        logger.info("Error distribution plot generated successfully")
    except Exception as e:
        logger.warning(f"Error generating plot: {e}")
        # Continue execution even if plotting fails

    logger.info("Pipeline completed successfully!")
    logger.info(f"Final Results - MAE: {mae:.2f} °C | MSE: {mse:.2f} °C² | RMSE: {rmse:.2f} °C")


if __name__ == "__main__":
    main()
