import time
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from lightgbm import LGBMRegressor
import warnings
from skforecast.exceptions import MissingValuesWarning
warnings.simplefilter('ignore', category=MissingValuesWarning)

if __name__ == "__main__":
    # load dataset
    exog = pd.read_csv('data/results/selected_exog_features.csv', delimiter=';', skiprows=1).squeeze("columns").tolist()

    df = pd.read_csv('data/dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    metrics = ['mean_absolute_percentage_error', 'mean_absolute_error', 'root_mean_squared_scaled_error']

    for i in range(2):
        if i == 1:
            df = df[['grid_load'] + exog]
        data = df.copy()
        data_train = data.loc['2015-01-01':'2024-03-02'].asfreq('h')
        data_val = data.loc['2024-03-03':'2024-03-07'].asfreq('h')

        # search space
        lags_grid = [
            tuple(range(1, 25)),                                    # intraday
            tuple(list(range(1, 25)) + [48, 72, 96]),               # intraday + harmonics
            tuple(list(range(1, 25)) + [168]),                      # intraday + weekly
            tuple(list(range(1, 25)) + [167, 168, 169]),            # weekly +/- 1
            tuple(list(range(1, 25)) + [47, 48, 49, 167, 168, 169]) # daily +/-1 + weekly +/-1
        ]
        def search_space(trial):
            return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "lags": trial.suggest_categorical("lags", lags_grid)}
        
        # forecaster
        cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)
        estimator = LGBMRegressor(random_state=123, verbose=-1)
        window_features = RollingFeatures(stats = ['mean', 'std', 'min', 'max'], window_sizes = [24*3, 24*7, 24*7, 24*7])
        lags = 24
        forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)

        for metric in metrics:
        # baysian search
            start_time = time.time()
            results_search, frozen_trial = bayesian_search_forecaster(
                                            forecaster   = forecaster,
                                            y            = data.loc['2015-01-01':'2024-03-31']['grid_load'].asfreq('h'),
                                            exog         = data.loc['2015-01-01':'2024-03-31'].drop(columns=['grid_load']),
                                            cv           = cv,
                                            metric       = metric,
                                            search_space = search_space,
                                            n_trials     = 50,
                                            return_best  = True)
            end_time = time.time()

            # save best params
            best_params = results_search.at[0, 'params']
            best_params.update({'random_state': 15926, 'verbose': -1})
            best_lags = results_search.at[0, 'lags']
            best_params = pd.DataFrame([best_params])
            best_lags = pd.DataFrame({'best_lags': [best_lags]})
            best_params.to_csv(f'data/results/tuning/tuning_{i}_{metric}.csv')
            best_lags.to_csv(f'data/results/tuning/best_lags_{i}_{metric}.csv')
            with open(f'data/results/tuning/tuning_time_{i}_{metric}.txt', 'w') as f:
                f.write(f'Tuning time (seconds): {end_time - start_time}')