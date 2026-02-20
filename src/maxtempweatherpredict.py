# FILE: maxtempweatherpredict.py
# AUTHOR: Ritik Pratap Singh Patel
# COMPLETION DATE: 07 May 2024
# DESCRIPTION: A weather prediction model to forecast maximum temperatures based on historical weather data using dataset i.e. weather.csv
# GUIDANCE: Zidio Development
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


weather = pd.read_csv(r"../data/weather.csv", index_col="DATE")
print(weather)

null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
print(null_pct)

valid_columns = weather.columns[null_pct < 0.05]
print(valid_columns)

weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()
print(weather.apply(pd.isnull).sum())  # weather.isnull().sum()
print(weather.apply(lambda x: (x == 9999).sum()))
print(weather.dtypes)

weather.index = pd.to_datetime(weather.index)
print(weather.index.year.value_counts().sort_index())
weather["snwd"].plot()
plt.xlabel('Year')  
plt.ylabel('Snow Depth')  
plt.title('Snow Depth Over Years')  
plt.show()

weather["target"] = weather.shift(-1)["tmax"]
print(weather)
weather = weather.ffill()

rr = Ridge(alpha=0.1)
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]


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

predictions = backtest(weather, rr, predictors)
print(mean_absolute_error(predictions["actual"], predictions["prediction"]))
predictions.sort_values("diff", ascending=False)
print(pd.Series(rr.coef_, index=predictors))


def pct_diff(old, new):
    return (new - old) / old


def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather


rolling_horizons = [3, 14]
for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)


def expand_mean(df):
    return df.expanding(1).mean()


for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_col{col}"] = (
        weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    )
    weather[f"day_avg_{col}"] = (
        weather[col]
        .groupby(weather.index.day_of_year, group_keys=False)
        .apply(expand_mean)
    )

weather = weather.iloc[14:, :]
weather = weather.fillna(0)
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictions = backtest(weather, rr, predictors)
mae=(mean_absolute_error(predictions["actual"], predictions["prediction"]))
mse=(mean_squared_error(predictions["actual"], predictions["prediction"]))
print(predictions.sort_values("diff", ascending=False))
print(weather.loc["1990-03-07":"1990-03-17"])

plt.plot(predictions["diff"].round().value_counts().sort_index() / predictions.shape[0])
plt.xlabel("Rounded Prediction Error (°C)")  
plt.ylabel("Relative Frequency") 
plt.title("Distribution of Prediction Errors") 
plt.grid(True, linestyle='--', alpha=0.7)  
plt.xticks(range(int(predictions["diff"].min()), int(predictions["diff"].max()) + 1)) 
plt.locator_params(axis='x', nbins=15)
plt.tight_layout() 
plt.show()
print(predictions)
print(f"Mean Absolute Error: {mae:.2f} °C")  
print(f"Mean Squared Error: {mse:.2f} °C²")
