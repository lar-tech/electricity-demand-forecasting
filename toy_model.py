import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def plot_predictions(df_power, predictions, model_name, mae):
    val_week = df_power.loc[predictions.index.min():predictions.index.max()]
    fig, ax = plt.subplots()
    val_week.plot(ax=ax, label='Actual Power', color='tab:blue')
    predictions['pred'].plot(ax=ax, label='Forecast', color='tab:red')
    plt.ylabel("Power [MWh]")
    plt.xlabel("Date")
    plt.title(f"{model_name} with MAE: {mae:.2f} MWh")
    plt.grid()
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()

def backtesting(df_power, forecaster, cv):
    metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = df_power.loc['2015-01-01':'2024-03-02']['Power'].asfreq('h'),
                            cv         = cv,
                            metric     = 'mean_absolute_error'
                        )
    
    return metric, predictions 

def baseline(data_train):
    forecaster = ForecasterEquivalentDate(offset = pd.DateOffset(days=1), n_offsets = 1)
    forecaster.fit(y=data_train['Power'])

    return forecaster

def forecaster_recursive(estimator, lags, window_features, data_train):
    forecaster = ForecasterRecursive(estimator = estimator, lags = lags, window_features = window_features)
    forecaster.fit(y=data_train['Power'])

    return forecaster

if __name__ == "__main__":
    # load dataset
    df = pd.read_csv('data/dataset.csv', delimiter=';')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()

    # create train and validation sets
    df_power = df[['Power']]
    data_train = df_power.loc['2015-01-01':'2024-02-29'].asfreq('h')
    data_val = df_power.loc['2024-03-01':'2024-03-02'].asfreq('h')

    # define cross-validation
    cv = TimeSeriesFold(steps = 24,
                        initial_train_size = len(data_train['Power']),
                        refit = True)

    # baseline
    model_baseline = baseline(data_train)
    metric_baseline, predictions_baseline = backtesting(df_power, model_baseline, cv)
    plot_predictions(df_power, predictions_baseline, "Seasonal Naive Forecast", metric_baseline["mean_absolute_error"].values[0])

    # recursive autoregressive model
    window_features = RollingFeatures(  stats = ['mean', 'std', 'min', 'max'],
                                        window_sizes = [24*3, 24*7, 24*7, 24*7])
    estimator = LGBMRegressor(random_state=123, verbose=-1)
    model_recursive = forecaster_recursive(estimator, lags=24*7, window_features=window_features, data_train=data_train)
    metric_recursive, predictions_recursive = backtesting(df_power, model_recursive, cv)
    plot_predictions(df_power, predictions_recursive, "Recursive LGBM Model", metric_recursive["mean_absolute_error"].values[0])