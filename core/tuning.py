import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import time
import json
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
from skforecast.exceptions import MissingValuesWarning
warnings.simplefilter('ignore', category=MissingValuesWarning)

# define lags and search space
LAGS_GRID = [
    tuple(range(1, 25)),                                    # intraday
    tuple(range(1, 73)),                                    # three days
    tuple(range(1, 169)),                                   # weekly
    tuple(list(range(1, 25)) + [48, 72, 96]),               # intraday+harmonics
    tuple(list(range(1, 25)) + [168]),                      # intraday+weekly
    tuple(list(range(1, 25)) + [167, 168, 169]),            # weekly+-1
    tuple(list(range(1, 25)) + [47, 48, 49, 167, 168, 169]) # daily+-1+weekly+-1
]

def search_space_lgbm(trial):
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
        "lags": trial.suggest_categorical("lags", LAGS_GRID)}

def search_space_xgb(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 5000, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "lags": trial.suggest_categorical("lags", LAGS_GRID),}

if __name__ == "__main__":
    os.makedirs('results/tuning/', exist_ok=True)
    SEARCH_SPACE_BY_CLASS = {   LGBMRegressor: search_space_lgbm,
                                XGBRegressor: search_space_xgb,}

    FIXED_PARAMS_BY_CLASS = {   LGBMRegressor: {"random_state": 123, "verbose": -1, "n_jobs": 1},
                                XGBRegressor: {"random_state": 123, "verbosity": 0, "n_jobs": 1},}

    # load dataset
    df = pd.read_csv('dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    estimators = [LGBMRegressor(random_state=123, verbose=-1, n_jobs=1),
                  XGBRegressor(random_state=123, verbosity=0, n_jobs=1)]
    
    for estimator in estimators:
        data = df.copy()
        data_train = data.loc['2015-01-01':'2024-03-02'].asfreq('h')
        data_val = data.loc['2024-03-03':'2024-03-07'].asfreq('h')

        # get estimator and search space
        est_cls = estimator.__class__
        search_space = SEARCH_SPACE_BY_CLASS[est_cls]

        # forecaster
        cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)
        window_features = RollingFeatures(stats = ['mean', 'std', 'min', 'max'], window_sizes = [24*3, 24*7, 24*7, 24*7])
        lags = 24
        forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)

        start_time = time.time()
        # baysian search
        results_search, frozen_trial = bayesian_search_forecaster(
                                        forecaster   = forecaster,
                                        y            = data.loc['2015-01-01':'2024-03-31']['grid_load'].asfreq('h'),
                                        exog         = data.loc['2015-01-01':'2024-03-31'].drop(columns=['grid_load']),
                                        cv           = cv,
                                        metric       = 'root_mean_squared_scaled_error',
                                        search_space = search_space,
                                        n_trials     = 1,
                                        n_jobs = 16,
                                        return_best  = True)
        end_time = time.time()

        # save best params
        best_params = results_search.at[0, 'params']
        best_params.update(FIXED_PARAMS_BY_CLASS.get(est_cls, {}))
        best_lags = results_search.at[0, 'lags']
        output = {
        "best_params": best_params,
        "best_lags": best_lags.tolist(),
        "tuning_time_seconds": end_time - start_time}
        with open(f'results/tuning/{estimator.__class__.__name__}_tuning_results.json', 'w') as f:
            json.dump(output, f, indent=4)