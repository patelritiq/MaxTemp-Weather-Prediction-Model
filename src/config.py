"""
Configuration file for weather prediction model.
Adjust these values to experiment with different model parameters.
"""

# Model Parameters
ALPHA = 0.1  # Ridge regression regularization strength (default/fallback)
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # GridSearchCV candidates

# Backtest Parameters
BACKTEST_START = 3650  # Starting index for backtesting
BACKTEST_STEP = 90     # Step size for sliding window backtest

# Data Cleaning Parameters
NULL_THRESHOLD = 0.05  # Drop columns with >5% null values

# Feature Engineering Parameters
ROLLING_HORIZONS = [3, 14]  # Days for rolling window features
FEATURE_COLUMNS = ["tmax", "tmin", "prcp"]  # Columns to engineer features from
EXCLUDE_COLUMNS = ["target", "name", "station"]  # Columns to exclude from predictors

# Data Processing
ROLLING_WINDOW_OFFSET = 14  # Rows to skip after rolling features (matches max rolling horizon)

# Model Comparison Parameters
RANDOM_FOREST_N_ESTIMATORS = 10
XGBOOST_N_ESTIMATORS = 10
LIGHTGBM_N_ESTIMATORS = 10
RANDOM_STATE = 42  # For reproducibility

# Model Saving
MODELS_DIR = "models"  # Directory to save trained models
BEST_MODEL_FILENAME = "best_model.pkl"  # Filename for best model
RETRAIN_MODEL = True  # Set to False to load saved model instead of retraining
