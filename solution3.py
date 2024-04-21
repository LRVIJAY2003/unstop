import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def real_time_market_insights():

    time_series_data = {
        'date': pd.date_range(start='2022-01-01', end='2022-12-31'),
        'job_demand': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
        'salary_trend': [50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000, 61000]
    }

    df = pd.DataFrame(time_series_data)
    df.set_index('date', inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    arima_model = ARIMA(train['job_demand'], order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_predictions = arima_model_fit.forecast(steps=len(test))
    arima_rmse = mean_squared_error(test['job_demand'], arima_predictions, squared=False)
    print("ARIMA RMSE:", arima_rmse)

    X_train, y_train = train.index.factorize(), train['salary_trend']
    X_test, y_test = test.index.factorize(), test['salary_trend']
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train.reshape(-1, 1), y_train)
    rf_predictions = rf_regressor.predict(X_test.reshape(-1, 1))
    rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
    print("Random Forest RMSE:", rf_rmse)

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test['job_demand'], label='Actual')
    plt.plot(test.index, arima_predictions, label='ARIMA Predictions')
    plt.xlabel('Date')
    plt.ylabel('Job Demand')
    plt.title('Actual vs ARIMA Predictions for Job Demand')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test['salary_trend'], label='Actual')
    plt.plot(test.index, rf_predictions, label='Random Forest Predictions')
    plt.xlabel('Date')
    plt.ylabel('Salary Trend')
    plt.title('Actual vs Random Forest Predictions for Salary Trend')
    plt.legend()
    plt.show()

    pass
