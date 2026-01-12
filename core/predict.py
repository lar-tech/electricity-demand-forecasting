import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from lightgbm import LGBMRegressor

def plot_predictions(df, predictions, model_name, mae):
    val_week = df['grid_load'].loc[predictions.index.min():predictions.index.max()]
    fig, ax = plt.subplots(figsize=(10,6))
    val_week.plot(ax=ax, label='Actual Load', color='tab:blue')
    predictions['pred'].plot(ax=ax, label='Forecast', color='tab:red')
    plt.ylabel("Electricity Demand [MWh]")
    plt.xlabel("Date")
    plt.title(f"{model_name} with MAPE: {mae*100:.2f} %")
    plt.grid()
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()

def backtesting(model, df, cv, exog=False):
    metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    metric, predictions = backtesting_forecaster(
                                forecaster = model,
                                y = df.loc['2015-01-01':'2024-03-31']['grid_load'].asfreq('h'),
                                exog = df.loc['2015-01-01':'2024-03-31'].drop(columns=['grid_load']) if exog else None,
                                cv = cv,
                                metric = metrics,
                                n_jobs=-1)
    return metric, predictions

if __name__ == "__main__":
    # load dataset
    df = pd.read_csv('data/dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    # create train and validation sets
    data = df.copy()
    data_train = data.loc['2015-01-01':'2024-03-02'].asfreq('h')
    data_val = data.loc['2024-03-01':'2024-03-07'].asfreq('h')

    # define cross-validation
    cv = TimeSeriesFold(steps = 24,
                        initial_train_size = len(data_train),
                        refit = False)
    # # baseline model
    # model_baseline = ForecasterEquivalentDate(offset=pd.DateOffset(days=1), n_offsets=1)
    # metric, predictions = backtesting(model_baseline, data, cv)
    # plot_predictions(data, predictions, "Seasonal Naive Forecast", metric["mean_absolute_percentage_error"].values[0])

    # autoregressive model with LightGBM
    estimator = LGBMRegressor(random_state=123, verbose=-1)
    window_features = RollingFeatures(stats = ['mean', 'std', 'min', 'max'], window_sizes = [24*3, 24*7, 24*7, 24*7])
    lags = 168*2
    forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)
    metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    metric, predictions = backtesting(forecaster, data, cv, exog=True)
    plot_predictions(df, predictions, "Recursive LGBM Model", metric["mean_absolute_percentage_error"].values[0])

# %%
