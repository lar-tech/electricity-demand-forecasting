import matplotlib.pyplot as plt
import ast
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

    # get tuning results
    tuning_results_lgbm = pd.read_csv('results/tuning/LGBMRegressor_tuning.csv')
    tuning_results_xgb = pd.read_csv('results/tuning/XGBRegressor_tuning.csv')
    tuning_results_lgbm = tuning_results_lgbm.iloc[::-1].reset_index(drop=True)
    tuning_results_xgb = tuning_results_xgb.iloc[::-1].reset_index(drop=True)
    best_params_lgbm = tuning_results_lgbm.iloc[-1]
    best_params_xgb = tuning_results_xgb.iloc[-1]
    params_lgbm = ast.literal_eval(best_params_lgbm['params'])
    params_xgb = ast.literal_eval(best_params_xgb['params'])
    best_lags_lgbm = list(map(int, best_params_lgbm['lags'].strip("[]").split()))
    best_lags_xgb = list(map(int, best_params_xgb['lags'].strip("[]").split()))

    # create estimators
    estimators = [LGBMRegressor(**params_lgbm, verbose=-1), XGBRegressor(**params_xgb, verbosity=0)]

    # fit and backtest each estimator
    for estimator in estimators:
        estimator_name = estimator.__class__.__name__
        print(f"Fitting and backtesting: {estimator_name}")
        if estimator_name == "LGBMRegressor":
            lags = best_lags_lgbm
        else:
            lags = best_lags_xgb
        forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)
        metric, predictions = backtesting(forecaster, data, cv, test_end, exog=True)
        plot_predictions(data, predictions, f"Tuned {estimator_name} with exogenous Features", eval_metric, metric[eval_metric].values[0])
