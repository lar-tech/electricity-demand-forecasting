import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
from skforecast.exceptions import MissingValuesWarning
warnings.simplefilter('ignore', category=MissingValuesWarning)

def plot_predictions(df, predictions, model_name, eval_metric, metric):
    """
    Plots the actual vs predicted values for a given model.
    
    :param df: dataframe containing the actual values
    :param predictions: dataframe containing the predicted values
    :param model_name: name of the forecasting model
    :param eval_metric: evaluation metric used
    :param metric: value of the evaluation metric
    """
    val_week = df['grid_load'].loc[predictions.index.min():predictions.index.max()]
    fig, ax = plt.subplots(figsize=(10,6))
    val_week.plot(ax=ax, label='Actual Load', color='tab:blue')
    predictions['pred'].plot(ax=ax, label='Forecast', color='tab:red')
    plt.xlim(pd.Timestamp('2024-03-04'), pd.Timestamp('2024-03-11'))
    plt.ylabel("Grid Load in MWh")
    plt.xlabel("")
    if eval_metric == 'mean_absolute_percentage_error':
        plt.title(f"{model_name} with MAPE {metric*100:.2f}% ")
    elif eval_metric == 'mean_squared_error':
        plt.title(f"{model_name} with MSE {metric:.2f} (MWh)^2 ")
    elif eval_metric == 'mean_absolute_error':
        plt.title(f"{model_name} with MAE {metric:.2f} MWh ")
    plt.grid()
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()

def backtesting(model, df, cv, test_end, exog=False):
    """
    Backtesting function to evaluate forecasting models.
    
    :param model: forecaster model
    :param df: dataframe with the time series data and exogenous variables
    :param cv: cross-validation scheme
    :param test_end: end date for the test set
    :param exog: boolean indicating whether to use exogenous variables
    """
    metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    metric, predictions = backtesting_forecaster(
                                forecaster = model,
                                y = df.loc['2015-01-01':test_end]['grid_load'].asfreq('h'),
                                exog = df.loc['2015-01-01':test_end].drop(columns=['grid_load']) if exog else None,
                                cv = cv,
                                metric = metrics)
    return metric, predictions

if __name__ == "__main__":
    # params
    start = '2015-01-01'
    train_end = '2023-12-31'
    test_end = '2024-12-31'
    eval_metric = 'mean_absolute_percentage_error'
    window_features = RollingFeatures(stats=['mean', 'std', 'min', 'max'], window_sizes=[24*3, 24*7, 24*7, 24*7])

    # load dataset
    df = pd.read_csv('dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    data = df.copy()
    data_train = data.loc[start:train_end].asfreq('h') # explicity annotate frequency is hourly

    # define cross-validation scheme
    cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)

    # create LGBMRegressor estimator
    lags = 24
    estimator = LGBMRegressor(learning_rate=0.04802270167510142, n_estimators=2900, num_leaves=166,
                              max_depth=6, min_child_samples=110, subsample=0.9404067086517863,
                              subsample_freq=1, colsample_bytree=0.6772559947743917, reg_alpha=4.588474569608268e-05,
                              reg_lambda=0.0004650965158027969,
                              random_state=15926, verbose=-1)
    
    # create backtesting forecasters
    forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)
    metric, predictions = backtesting(forecaster, data, cv, test_end, exog=True)
    plot_predictions(data, predictions, "Tuned LGBMRegressor with exogenous Features", eval_metric, metric[eval_metric].values[0])

    # create XGBRegressor estimator
    lags = list(range(1,25)) + [167, 168, 169]
    estimator = XGBRegressor(learning_rate=0.02176000708646436, n_estimators=3900, max_depth=7,
                              min_child_weight=11.730521341347089, subsample=0.7375957356031976,
                              colsample_bytree=0.6752518587096829, gamma=7.429905815763753,
                              reg_alpha=0.009558252940803038, reg_lambda=0.0007996684456278945,
                              random_state=15926, verbosity=0)
    
    # create backtesting forecasters
    forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)
    metric, predictions = backtesting(forecaster, data, cv, test_end, exog=True)
    plot_predictions(data, predictions, "Tuned XGBRegressor with exogenous Features", eval_metric, metric[eval_metric].values[0])
