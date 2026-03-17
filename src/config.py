"""
Configuration file for weather prediction model.
Adjust these values to experiment with different model parameters.
"""

# Model Parameters
ALPHA = 0.1  # Ridge regression regularization strength

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
