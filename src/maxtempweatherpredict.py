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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from config import (
    ALPHA, RIDGE_ALPHA_GRID, BACKTEST_START, BACKTEST_STEP, NULL_THRESHOLD,
    ROLLING_HORIZONS, FEATURE_COLUMNS, EXCLUDE_COLUMNS, ROLLING_WINDOW_OFFSET,
    RANDOM_FOREST_N_ESTIMATORS, XGBOOST_N_ESTIMATORS, LIGHTGBM_N_ESTIMATORS,
    RANDOM_STATE, MODELS_DIR, BEST_MODEL_FILENAME, RETRAIN_MODEL
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


def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error.
    Handles division by zero by excluding rows where actual temp is 0.
    """
    mask = actual != 0
    if mask.sum() == 0:
        return float('nan')
    return (((actual[mask] - predicted[mask]).abs() / actual[mask].abs()) * 100).mean()


def tune_ridge_alpha(X_train, y_train, alpha_grid):
    """Use GridSearchCV with time-series split to find best Ridge alpha"""
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        Ridge(),
        param_grid={'alpha': alpha_grid},
        scoring='neg_mean_absolute_error',
        cv=tscv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Build alpha comparison table
    results_df = pd.DataFrame({
        'Alpha': grid_search.cv_results_['param_alpha'],
        'MAE (°C)': (-grid_search.cv_results_['mean_test_score']).round(4),
        'Std (°C)': grid_search.cv_results_['std_test_score'].round(4)
    }).sort_values('Alpha').reset_index(drop=True)

    return grid_search.best_params_['alpha'], results_df


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

    # --- Feature Engineering: Year ---
    logger.info("Adding year as a numeric feature to capture long-term climate trend...")
    try:
        weather['year'] = weather.index.year
        logger.info(f"Year feature added. Range: {weather['year'].min()} to {weather['year'].max()}")
    except Exception as e:
        logger.error(f"Error adding year feature: {e}")
        return

    # --- Final Model & Evaluation with Model Comparison ---
    model_path = os.path.join(MODELS_DIR, BEST_MODEL_FILENAME)

    if not RETRAIN_MODEL and os.path.exists(model_path):
        # --- Load saved model ---
        logger.info(f"Loading saved model from {model_path} (RETRAIN_MODEL=False)...")
        try:
            weather = weather.iloc[ROLLING_WINDOW_OFFSET:, :]
            weather = weather.fillna(0)
            predictors = weather.columns[~weather.columns.isin(EXCLUDE_COLUMNS)]

            best_model = joblib.load(model_path)
            preds = best_model.predict(weather[predictors])
            preds = pd.Series(preds, index=weather.index)
            predictions = pd.concat([weather["target"], preds], axis=1)
            predictions.columns = ["actual", "prediction"]
            predictions["diff"] = (predictions["prediction"] - predictions["actual"]).abs()

            mae  = mean_absolute_error(predictions["actual"], predictions["prediction"])
            mse  = mean_squared_error(predictions["actual"], predictions["prediction"])
            rmse = calculate_rmse(predictions["actual"], predictions["prediction"])
            r2   = r2_score(predictions["actual"], predictions["prediction"])
            mape = calculate_mape(predictions["actual"], predictions["prediction"])

            logger.info(f"Loaded model results - MAE: {mae:.2f} °C | RMSE: {rmse:.2f} °C | R²: {r2:.4f}")
        except Exception as e:
            logger.error(f"Error loading saved model: {e}")
            return
    else:
        logger.info("Training and comparing multiple models with engineered features...")
        try:
            weather = weather.iloc[ROLLING_WINDOW_OFFSET:, :]
            weather = weather.fillna(0)
            predictors = weather.columns[~weather.columns.isin(EXCLUDE_COLUMNS)]
            
            if len(predictors) == 0:
                logger.error("No valid predictors found for model comparison")
                return

            # Tune Ridge alpha using GridSearchCV
            logger.info("Tuning Ridge alpha with GridSearchCV...")
            try:
                train_data = weather.iloc[:BACKTEST_START]
                best_alpha, alpha_results = tune_ridge_alpha(
                    train_data[predictors],
                    train_data["target"],
                    RIDGE_ALPHA_GRID
                )
                logger.info(
                    f"\n{'='*50}\n"
                    f"RIDGE ALPHA TUNING RESULTS\n"
                    f"{'='*50}\n"
                    f"{alpha_results.to_string(index=False)}\n"
                    f"{'='*50}"
                )
                logger.info(f"Best Ridge alpha: {best_alpha}")
            except Exception as e:
                logger.warning(f"GridSearchCV failed, using default alpha={ALPHA}: {e}")
                best_alpha = ALPHA

            # Define models to compare
            models = {
                'Ridge': Ridge(alpha=best_alpha),
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
                    preds = backtest(weather, model, predictors)

                    if preds.empty:
                        logger.warning(f"No predictions from {model_name}, skipping")
                        continue

                    mae  = mean_absolute_error(preds["actual"], preds["prediction"])
                    mse  = mean_squared_error(preds["actual"], preds["prediction"])
                    rmse = calculate_rmse(preds["actual"], preds["prediction"])
                    r2   = r2_score(preds["actual"], preds["prediction"])
                    mape = calculate_mape(preds["actual"], preds["prediction"])

                    model_results[model_name] = {
                        'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                        'R2': r2, 'MAPE': mape,
                        'model': model, 'predictions': preds
                    }

                    logger.info(f"{model_name} - MAE: {mae:.2f} °C, RMSE: {rmse:.2f} °C, R²: {r2:.4f}, MAPE: {mape:.2f}%")
                except Exception as e:
                    logger.warning(f"Error training {model_name}: {e}")
                    continue

            if not model_results:
                logger.error("No models trained successfully")
                return

            # Comparison table
            comparison_data = {
                name: {
                    'MAE (°C)': f"{r['MAE']:.2f}", 'MSE (°C²)': f"{r['MSE']:.2f}",
                    'RMSE (°C)': f"{r['RMSE']:.2f}", 'R²': f"{r['R2']:.4f}",
                    'MAPE (%)': f"{r['MAPE']:.2f}"
                }
                for name, r in model_results.items()
            }
            comparison_df = pd.DataFrame(comparison_data).T
            logger.info(
                f"\n{'='*70}\n"
                f"MODEL COMPARISON RESULTS\n"
                f"{'='*70}\n"
                f"{comparison_df.to_string()}\n"
                f"{'='*70}"
            )

            # Find best model
            best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['MAE'])
            best_model_data = model_results[best_model_name]

            logger.info(
                f"\n{'='*70}\n"
                f"[BEST MODEL] {best_model_name}\n"
                f"   MAE:  {best_model_data['MAE']:.2f} °C\n"
                f"   MSE:  {best_model_data['MSE']:.2f} °C²\n"
                f"   RMSE: {best_model_data['RMSE']:.2f} °C\n"
                f"   R²:   {best_model_data['R2']:.4f}\n"
                f"   MAPE: {best_model_data['MAPE']:.2f}%\n"
                f"{'='*70}"
            )

            # Per-year breakdown
            try:
                bp = best_model_data['predictions'].copy()
                bp['year'] = bp.index.year
                yearly = bp.groupby('year').apply(
                    lambda g: pd.Series({
                        'MAE (°C)': round(mean_absolute_error(g['actual'], g['prediction']), 2),
                        'RMSE (°C)': round(calculate_rmse(g['actual'], g['prediction']), 2),
                        'R²': round(r2_score(g['actual'], g['prediction']), 4)
                    })
                )
                logger.info(
                    f"\n{'='*50}\n"
                    f"PERFORMANCE BY YEAR\n"
                    f"{'='*50}\n"
                    f"{yearly.to_string()}\n"
                    f"{'='*50}"
                )
            except Exception as e:
                logger.warning(f"Error generating yearly breakdown: {e}")

            # Per-season breakdown
            try:
                def get_season(month):
                    if month in [12, 1, 2]:  return 'Winter'
                    elif month in [3, 4, 5]: return 'Spring'
                    elif month in [6, 7, 8]: return 'Summer'
                    else:                    return 'Fall'

                bp['season'] = bp.index.month.map(get_season)
                seasonal = bp.groupby('season').apply(
                    lambda g: pd.Series({
                        'MAE (°C)': round(mean_absolute_error(g['actual'], g['prediction']), 2),
                        'RMSE (°C)': round(calculate_rmse(g['actual'], g['prediction']), 2),
                        'R²': round(r2_score(g['actual'], g['prediction']), 4)
                    })
                ).reindex(['Winter', 'Spring', 'Summer', 'Fall'])
                logger.info(
                    f"\n{'='*50}\n"
                    f"PERFORMANCE BY SEASON\n"
                    f"{'='*50}\n"
                    f"{seasonal.to_string()}\n"
                    f"{'='*50}"
                )
            except Exception as e:
                logger.warning(f"Error generating seasonal breakdown: {e}")

            # Save best model
            try:
                if not os.path.exists(MODELS_DIR):
                    os.makedirs(MODELS_DIR)
                    logger.info(f"Created models directory: {MODELS_DIR}")
                joblib.dump(best_model_data['model'], model_path)
                logger.info(f"[SAVED] Best model saved to: {model_path}")
            except Exception as e:
                logger.warning(f"Error saving best model: {e}")

            # Set final result variables
            predictions = best_model_data['predictions']
            mae  = best_model_data['MAE']
            mse  = best_model_data['MSE']
            rmse = best_model_data['RMSE']
            r2   = best_model_data['R2']
            mape = best_model_data['MAPE']

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

    # --- Plot: Feature Importance (Ridge only) ---
    logger.info("Generating feature importance plot...")
    try:
        best_model_obj = joblib.load(os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))
        if hasattr(best_model_obj, 'coef_'):
            coef_series = pd.Series(best_model_obj.coef_, index=predictors)

            # Top 20 by absolute value, sorted for readability
            top_features = coef_series.abs().nlargest(20).index
            plot_data = coef_series[top_features].sort_values()

            colors = ['#d73027' if v < 0 else '#1a9850' for v in plot_data]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(plot_data.index, plot_data.values, color=colors)
            ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel("Coefficient Value")
            ax.set_title("Feature Importance - Ridge Regression (Top 20)")
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            logger.info("Feature importance plot generated successfully")
        else:
            logger.info(f"Best model is not Ridge — skipping feature importance plot")
    except Exception as e:
        logger.warning(f"Error generating feature importance plot: {e}")

    logger.info("Pipeline completed successfully!")
    logger.info(f"Final Results - MAE: {mae:.2f} °C | MSE: {mse:.2f} °C² | RMSE: {rmse:.2f} °C | R²: {r2:.4f} | MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
