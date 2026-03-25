# MaxTemp Weather Prediction Model

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **time-series weather prediction model** trained on **19,288 historical weather records** to forecast maximum daily temperatures. Implements a complete analytics pipeline with data cleaning, feature engineering, multi-model comparison, and backtesting validation.

---

## Project Overview

### What is MaxTemp Weather PM?

This project develops a machine learning model to predict maximum daily temperatures using historical weather data. Through systematic feature engineering with rolling averages and temporal aggregates, the model compares multiple ML algorithms and selects the best performer for temperature forecasting.

### Key Statistics
- **19,287 Historical Records**: Weather dataset spanning 1970 to 2022 (52 years)
- **Multi-Model Comparison**: Ridge Regression, Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit for Ridge alpha optimization
- **Feature Engineering**: Rolling averages (3-day, 7-day, 14-day) and temporal aggregates (monthly, daily)
- **Backtesting Validation**: 10-year training window with 90-day evaluation steps
- **Best Model Performance**: MAE of 4.79°C, MSE of 37.62°C², RMSE of 6.13°C, R² of 0.8747, MAPE of 8.90%
- **Structured Logging**: Full execution logs saved to `model_training.log`
- **Model Persistence**: Best model saved and reusable without retraining

---

## Project Impact & Applications

### Real-World Use Cases

**Weather Forecasting Services**
- Provide temperature predictions for daily weather reports
- Support meteorological analysis and climate studies

**Agricultural Planning**
- Help farmers plan planting and harvesting schedules
- Predict frost risks and heat waves

**Event Planning & Management**
- Assist in scheduling outdoor events
- Support logistics planning for weather-sensitive operations

**Energy Management**
- Forecast heating and cooling demand
- Optimize energy distribution and pricing

---

## Technical Overview

### Model Architecture
- **Algorithms Compared**: Ridge Regression, Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit (5 folds) for Ridge alpha — searches `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- **Best Model**: Automatically selected based on lowest MAE, saved to disk
- **Training Data**: 19,287 historical weather records (1970–2022)
- **Validation Strategy**: Backtesting with 10-year initial training, 90-day evaluation steps
- **Target Variable**: Maximum daily temperature (TMAX)
- **Model Persistence**: Best model saved to `models/best_model.pkl`, reloadable via `RETRAIN_MODEL = False` in config

### Feature Engineering
- **Rolling Averages**: 3-day, 7-day, and 14-day windows for TMAX, TMIN, PRCP
  - 3-day: captures short-term momentum
  - 7-day: captures weekly weather cycle patterns
  - 14-day: captures medium-term temperature trends
- **Percentage Differences**: Relative change between today's value and each rolling average
- **Temporal Aggregates**: Expanding monthly and daily averages
- **Data Cleaning**: Null value handling (<5% threshold), forward-fill imputation

### Evaluation Metrics
All models are evaluated on five metrics:
- **MAE** — Average error in °C
- **MSE** — Penalizes large errors more heavily
- **RMSE** — MAE in °C scale, sensitive to outliers
- **R²** — How much temperature variation the model explains (1.0 = perfect)
- **MAPE** — Error as a percentage (handles scale differences across seasons)

### Model Comparison Results
| Model         | MAE (°C) | MSE (°C²) | RMSE (°C) | R²     | MAPE (%) |
|---------------|----------|-----------|-----------|--------|----------|
| Ridge         | 4.79     | 37.62     | 6.13      | 0.8747 | 8.90     |
| Random Forest | 5.00     | 41.39     | 6.43      | 0.8621 | 9.25     |
| XGBoost       | 4.81     | 38.08     | 6.17      | 0.8732 | 8.94     |
| LightGBM      | 6.89     | 71.10     | 8.43      | 0.7631 | 13.19    |

### Performance Breakdown
Model performance is reported per-year and per-season to identify where the model is strong or weak:
- **Per-year**: Tracks MAE, RMSE, R² for each year in the dataset
- **Per-season**: Winter / Spring / Summer / Fall breakdown

### Reusable Analytics Pipeline
- `backtest()`: Time-series cross-validation function
- `compute_rolling()`: Rolling window feature generation
- `expand_mean()`: Temporal aggregate computation
- `pct_diff()`: Percentage difference calculation
- `calculate_rmse()`: RMSE metric computation
- `calculate_mape()`: MAPE metric with zero-division handling
- `tune_ridge_alpha()`: GridSearchCV with TimeSeriesSplit for alpha tuning

---

## Getting Started

### Prerequisites
- Python 3.x
- Required packages: see `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/patelritiq/MaxTemp-Weather-Prediction-Model.git
   cd MaxTemp-Weather-Prediction-Model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Navigate to source directory:**
   ```bash
   cd src
   ```

2. **Run the prediction model:**
   ```bash
   python maxtempweatherpredict.py
   ```

3. **Output:**
   - Ridge alpha tuning table (GridSearchCV results across all alpha values)
   - Model comparison table (MAE, MSE, RMSE, R², MAPE for all models)
   - Per-year and per-season performance breakdown
   - Best model logged and saved to `models/best_model.pkl`
   - Execution log saved to `model_training.log`
   - Visualizations: Snow depth trends, prediction error distribution, feature importance chart

---

## Project Structure

```
MaxTemp-Weather-Prediction-Model/
├── src/
│   ├── maxtempweatherpredict.py  # Main prediction script
│   └── config.py                 # Hyperparameters and settings
├── data/
│   └── weather.csv               # Historical weather data (19,288 records)
├── models/                       # Saved trained models (git-ignored)
│   └── best_model.pkl
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## Model Configuration

All hyperparameters are centralized in `src/config.py`:

```python
# Model Parameters
ALPHA = 0.1                          # Ridge fallback alpha
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # GridSearchCV candidates
RANDOM_FOREST_N_ESTIMATORS = 10
XGBOOST_N_ESTIMATORS = 10
LIGHTGBM_N_ESTIMATORS = 10
RANDOM_STATE = 42                    # For reproducibility

# Backtest Configuration
BACKTEST_START = 3650                # 10-year initial training period
BACKTEST_STEP = 90                   # 90-day evaluation windows

# Feature Engineering
ROLLING_HORIZONS = [3, 7, 14]       # Short, weekly, and medium-term patterns
NULL_THRESHOLD = 0.05                # Drop columns with >5% null values

# Model Persistence
RETRAIN_MODEL = True                 # Set False to load saved model instead of retraining
```

---

## Future Enhancements

- Add LSTM / Transformer model for deep learning comparison
- Season-specific feature engineering to improve winter performance
- Real-time weather data integration via Open-Meteo API
- Streamlit dashboard for interactive predictions
- Multi-day forecasting (3-day, 7-day predictions)
- Confidence intervals and uncertainty quantification
- Optuna hyperparameter tuning for Random Forest and XGBoost

---

## Contributing

Contributions welcome! Areas for improvement:
- Model optimization and hyperparameter tuning
- Additional feature engineering techniques
- Enhanced visualization and reporting
- Documentation improvements

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Author & Context

**Ritik Pratap Singh Patel**
Data Science & Machine Learning

**Project Context:**
- Developed as a learning internship project with guidance from Zidio Development
- Purpose: Demonstrates time-series prediction, feature engineering, and multi-model ML pipeline

**Connect:**
- Email: patelritiq@gmail.com
- LinkedIn: [linkedin.com/in/patelritiq](https://www.linkedin.com/in/patelritiq)
- GitHub: [github.com/patelritiq](https://github.com/patelritiq)

---

## Acknowledgments

- **Zidio Development**: For project guidance and dataset provision
- **scikit-learn, XGBoost, LightGBM**: For robust machine learning tools
- **pandas & matplotlib**: For data manipulation and visualization

---

## Project Statistics

![GitHub stars](https://img.shields.io/github/stars/patelritiq/MaxTemp-Weather-Prediction-Model?style=social)
![GitHub forks](https://img.shields.io/github/forks/patelritiq/MaxTemp-Weather-Prediction-Model?style=social)
![GitHub issues](https://img.shields.io/github/issues/patelritiq/MaxTemp-Weather-Prediction-Model)

---

*Predicting tomorrow's weather with today's data.🌡️☀️* 
