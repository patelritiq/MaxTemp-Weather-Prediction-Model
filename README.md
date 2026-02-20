# MaxTemp Weather Prediction Model 🌡️

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **time-series weather prediction model** trained on **19,288 historical weather records** to forecast maximum daily temperatures. Implements a complete analytics pipeline with EDA, data cleaning, feature engineering, Ridge Regression modeling, and backtesting validation.

---

## Project Overview 📊 

### What is MaxTemp Weather PM?

This project develops a machine learning model to predict maximum daily temperatures using historical weather data. Through systematic feature engineering with rolling averages and temporal aggregates, the model achieves reliable temperature forecasts suitable for weather prediction, agricultural planning, and event scheduling.

### Key Statistics
- **19,288 Historical Records**: Multi-decade weather dataset spanning 50+ years
- **Ridge Regression Model**: Optimized with alpha=0.1 for temperature prediction
- **Feature Engineering**: Rolling averages (3-day, 14-day) and temporal aggregates (monthly, daily)
- **Backtesting Validation**: 10-year training window with 90-day evaluation steps
- **Model Performance**: MAE of 4.79°C, MSE of 37.62°C²
- **Reusable Pipeline**: Modular functions for preprocessing, training, and evaluation

---

## Project Impact & Applications 🎯 

### Real-World Use Cases

**Weather Forecasting Services**
- Provide temperature predictions for daily weather reports
- Support meteorological analysis and climate studies
- Enable short-term and medium-term temperature forecasting

**Agricultural Planning**
- Help farmers plan planting and harvesting schedules
- Predict frost risks and heat waves
- Optimize irrigation and crop protection strategies

**Event Planning & Management**
- Assist in scheduling outdoor events
- Support logistics planning for weather-sensitive operations
- Enable proactive decision-making for event organizers

**Energy Management**
- Forecast heating and cooling demand
- Optimize energy distribution and pricing
- Support renewable energy planning (solar, wind)

**Internal Training & Reference**
- Serves as a reference implementation for time-series prediction
- Demonstrates end-to-end ML pipeline development
- Provides reusable analytics framework for similar projects

---

## Technical Overview 💡 

### Model Architecture
- **Algorithm**: Ridge Regression (L2 regularization, alpha=0.1)
- **Training Data**: 19,288 historical weather records
- **Validation Strategy**: Backtesting with 10-year initial training, 90-day evaluation steps
- **Target Variable**: Maximum daily temperature (TMAX)

### Feature Engineering
- **Rolling Averages**: 3-day and 14-day windows for TMAX, TMIN, PRCP
- **Percentage Differences**: Relative changes from rolling averages
- **Temporal Aggregates**: Expanding monthly and daily averages
- **Data Cleaning**: Null value handling (<5% threshold), forward-fill imputation

### Model Performance
- **Mean Absolute Error (MAE)**: 4.79°C
- **Mean Squared Error (MSE)**: 37.62°C²
- **Evaluation**: 15,623 predictions across backtesting windows

### Reusable Analytics Pipeline
- `backtest()`: Time-series cross-validation function
- `compute_rolling()`: Rolling window feature generation
- `expand_mean()`: Temporal aggregate computation
- `pct_diff()`: Percentage difference calculation

---

## Getting Started 🚀 

### Prerequisites
- Python 3.x
- Required packages: pandas, matplotlib, scikit-learn

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
   - Model training and evaluation metrics (MAE, MSE)
   - Visualizations: Snow depth trends, prediction error distribution
   - Backtesting results with actual vs. predicted temperatures

---

## Project Structure 📁 

```
MaxTemp-Weather-PM/
├── src/                         # Source code
│   └── maxtempweatherpredict.py # Main prediction script
├── data/                        # Dataset
│   └── weather.csv              # Historical weather data (19,288 records)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## Model Configuration 🔧 

### Ridge Regression Parameters
```python
Ridge(alpha=0.1)  # L2 regularization strength
```

### Backtesting Configuration
```python
start = 3650      # 10-year initial training period
step = 90         # 90-day evaluation windows
```

### Rolling Window Sizes
```python
rolling_horizons = [3, 14]  # Short-term and medium-term patterns
```

---

## Future Enhancements 🚧 

- Incorporate additional weather features (humidity, pressure, wind patterns)
- Implement advanced models (LSTM, XGBoost, Prophet)
- Add real-time weather data integration via APIs
- Develop web-based interface for interactive predictions
- Extend to multi-day forecasting (3-day, 7-day predictions)
- Include confidence intervals and uncertainty quantification

---

## Contributing 🤝 

Contributions welcome! Areas for improvement:
- Model optimization and hyperparameter tuning
- Additional feature engineering techniques
- Alternative ML algorithms comparison
- Enhanced visualization and reporting
- Documentation improvements

---

## License 📄 

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Author & Context 👨‍💻 

**Ritik Pratap Singh Patel**  
Data Science & Machine Learning

**Project Context:**
- Developed as a learning internship project with guidance from Zidio Development
- Completion Date: May 7, 2024
- Purpose: Demonstrates time-series prediction, feature engineering, and ML pipeline development

**Connect:**
- Email: patelritiq@gmail.com
- LinkedIn: [linkedin.com/in/patelritiq](https://www.linkedin.com/in/patelritiq)
- GitHub: [github.com/patelritiq](https://github.com/patelritiq)

---

## Acknowledgments 🙏 

- **Zidio Development**: For project guidance and dataset provision
- **scikit-learn**: For robust machine learning tools
- **pandas & matplotlib**: For data manipulation and visualization

---

## Project Statistics 📈 

![GitHub stars](https://img.shields.io/github/stars/patelritiq/MaxTemp-Weather-Prediction-Model?style=social)
![GitHub forks](https://img.shields.io/github/forks/patelritiq/MaxTemp-Weather-Prediction-Model?style=social)
![GitHub issues](https://img.shields.io/github/issues/patelritiq/MaxTemp-Weather-Prediction-Model)

---

*Predicting tomorrow's weather with today's data.* 🌡️☀️
