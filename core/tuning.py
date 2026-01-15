import pandas as pd
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor

# load dataset
df = pd.read_csv('../data/dataset.csv', delimiter=';')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.set_index('datetime').sort_index()
data = df.copy()
data_train = data.loc['2015-01-01':'2024-03-02'].asfreq('h')
data_val = data.loc['2024-03-03':'2024-03-07'].asfreq('h')

# search space
lags_grid = [24, (1, 2, 3, 23, 24, 25, 47, 48, 49)]
def search_space(trial):
    search_space  = {
        'n_estimators' : trial.suggest_int('n_estimators', 300, 1000, step=100),
        'max_depth'    : trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'reg_alpha'    : trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda'   : trial.suggest_float('reg_lambda', 0, 1),
        'lags'         : trial.suggest_categorical('lags', lags_grid)
    } 
    return search_space

# forecaster
cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)
estimator = LGBMRegressor(random_state=123, verbose=-1)
window_features = RollingFeatures(stats = ['mean', 'std', 'min', 'max'], window_sizes = [24*3, 24*7, 24*7, 24*7])
lags = 168*2
forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)

# baysian search
results_search, frozen_trial = bayesian_search_forecaster(
                                   forecaster   = forecaster,
                                   y            = df.loc['2015-01-01':'2024-03-31']['grid_load'].asfreq('h'),
                                   exog         = df.loc['2015-01-01':'2024-03-31'].drop(columns=['grid_load']),
                                   cv           = cv,
                                   metric       = 'mean_absolute_percentage_error',
                                   search_space = search_space,
                                   n_trials     = 10,
                                   return_best  = True)

# save best params
best_params = results_search.at[0, 'params']
best_params = best_params | {'random_state': 15926, 'verbose': -1}
best_lags = results_search.at[0, 'lags']
best_params.to_csv('../results/tuning.csv')
best_lags.to_csv('../results/best_lags.csv')