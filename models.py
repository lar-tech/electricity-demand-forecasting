import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def plot_predictions(df_load, predictions, model_name, mae):
    val_week = df_load.loc[predictions.index.min():predictions.index.max()]
    fig, ax = plt.subplots(figsize=(10,6))
    val_week.plot(ax=ax, label='Actual Load', color='tab:blue')
    predictions['pred'].plot(ax=ax, label='Forecast', color='tab:red')
    plt.ylabel("Electricity Demand [MWh]")
    plt.xlabel("Date")
    plt.title(f"{model_name} with MAE: {mae:.2f} MWh")
    plt.grid()
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()

def backtesting(model, df, cv, exog=None):
    metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    metric, predictions = backtesting_forecaster(
                                forecaster = model,
                                y = df.loc['2015-01-01':'2024-03-07']['grid_load'].asfreq('h'),
                                exog = exog,
                                cv = cv,
                                metric = metrics,
                                n_jobs=-1
                                )
    return metric, predictions

if __name__ == "__main__":
    # load dataset
    df = pd.read_csv('data/dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    # create train and validation sets
    df = df[['grid_load']]
    data_train = df.loc['2015-01-01':'2024-03-02'].asfreq('h')
    data_val = df.loc['2024-03-01':'2024-03-07'].asfreq('h')

    # define cross-validation
    cv = TimeSeriesFold(steps = 24,
                        initial_train_size = len(data_train),
                        refit = False)

    # baseline model
    model_baseline = ForecasterEquivalentDate(offset=pd.DateOffset(days=1), n_offsets=1)
    metric, predictions = backtesting(model_baseline, df, cv)
    plot_predictions(df, predictions, "Seasonal Naive Forecast", metric["mean_absolute_error"].values[0])

    # define different configs
    window_features = RollingFeatures(stats = ['mean', 'std', 'min', 'max'],
                                        window_sizes = [24*3, 24*7, 24*7, 24*7])
    configs = [
        ("Ridge_24lags", Ridge(alpha=1.0), 24, None),
        # ("Ridge_72lags", Ridge(alpha=1.0), 24*3, None),
        # ("Ridge_168lags", Ridge(alpha=1.0), 168, None),
        # ("Ridge_336lags", Ridge(alpha=1.0), 168*2, None),
        # ("Ridge_168lags_window", Ridge(alpha=1.0), 168, window_features),
        
        # ("LGBM_24lags", LGBMRegressor(random_state=123, verbose=-1), 24, None),
        # ("LGBM_72lags", LGBMRegressor(random_state=123, verbose=-1), 24*3, None),
        # ("LGBM_168lags", LGBMRegressor(random_state=123, verbose=-1), 168, None),
        # ("LGBM_336lags", LGBMRegressor(random_state=123, verbose=-1), 168*2, None),
        # ("LGBM_test", LGBMRegressor(random_state=123,
        #                             n_estimators=350,
        #                             max_depth=9,
        #                             learning_rate=0.1,
        #                             # reg_alpha=0.7636828414433382,
        #                             # reg_lambda=0.243666374536874,
        #                             verbose=-1),
        #                             168, window_features)

        # ("XGB_72lags", XGBRegressor(random_state=123, verbosity=0), 24*3, None),
        # ("XGB_168lags", XGBRegressor(random_state=123, verbosity=0), 168, None),
        # ("XGB_336lags", XGBRegressor(random_state=123, verbosity=0), 168*2, None),
        # ("XGB_168lags_window", XGBRegressor(random_state=123, verbosity=0), 168, window_features),
        # ("XGB_test", XGBRegressor(random_state=123,
        #                           n_estimators=600,
        #                           max_depth=8,
        #                           learning_rate=0.05,
        #                           gamma=5,
        #                           min_child_weight=3,
        #                           colsample_bytree=0.7,
        #                           subsample=0.7,
        #                           verbosity=0),
        #                           168, window_features),       
        ]
    
    results = []
    for name, estimator, lags, window_feature in configs:
        # create forecaster
        forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)

        # backtesting
        metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
        metric, predictions = backtesting_forecaster(
                                forecaster = forecaster,
                                y = df.loc['2015-01-01':'2024-03-07']['grid_load'].asfreq('h'),
                                cv = cv,
                                metric = metrics
                            )
        
        plot_predictions(df, predictions, f"Recursive {name} Model", metric["mean_absolute_error"].values[0])
        results.append({'model': name,
                        'MAE': metric['mean_absolute_error'].values[0],
                        'MSE': metric['mean_squared_error'].values[0],
                        'MAPE': metric['mean_absolute_percentage_error'].values[0]})

    results_df = pd.DataFrame(results).sort_values('MAE')
    # results_df.to_csv("data/model_comparison.csv", sep=";", index=False)