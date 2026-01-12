import pandas as pd
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# load dataset
df = pd.read_csv('data/dataset.csv', delimiter=';')
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df = df.set_index('Datetime').sort_index()

# create train and validation sets
df_load = df[['Grid Load']]
data = df_load.loc['2022-01-01 00:00:00':'2024-12-31 23:00:00', :].asfreq('h').copy()
end_train = '2023-12-31 23:59:00'
end_validation = '2024-11-30 23:59:00'

# define cross-validation
cv = TimeSeriesFold(steps=24,
                    initial_train_size=len(data[:end_train]),
                    refit=False)

estimator = XGBRegressor(random_state=123, verbosity=0)
forecaster = ForecasterRecursive(estimator=estimator, lags=72)

# Grid Search
lags_grid = [(1, 2, 3, 23, 24, 25, 47, 48, 49, 71, 72, 73, 167, 168, 169)]
param_grid = {
                'n_estimators':  [50, 100, 150, 200, 250, 300, 350, 400],
                'max_depth':     [6, 7, 8, 9, 10],
                'learning_rate': [0.01, 0.03, 0.1, 0.3],
            }

results = grid_search_forecaster(
                                    forecaster       = forecaster,
                                    y                = data.loc[:end_validation, 'Grid Load'],
                                    param_grid       = param_grid,
                                    lags_grid        = lags_grid,
                                    cv               = cv,
                                    metric           = 'mean_absolute_error',
                                    return_best      = True,
                                    n_jobs           = -1,
                                    verbose          = False)

results.to_csv("data/xgb_grid_search.csv", sep=";", index=False)