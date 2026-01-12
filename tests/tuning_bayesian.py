import pandas as pd
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def search_space(trial):
    search_space = {
        'n_estimators' : trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth'    : trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'reg_alpha'    : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda'   : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'lags'         : trial.suggest_categorical('lags', lags_grid)
    }
    return search_space

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

lags_grid = [(1, 2, 3, 23, 24, 25, 47, 48, 49, 71, 72, 73, 167, 168, 169)]

# forecaster
estimator = LGBMRegressor(random_state=123, verbose=-1, n_jobs=-1)
forecaster = ForecasterRecursive(estimator=estimator, lags=72) 

results_df, best_trial = bayesian_search_forecaster(
                                                forecaster=forecaster,
                                                y=data.loc[:end_validation, 'Grid Load'],
                                                search_space=search_space,
                                                cv=cv,
                                                metric='mean_absolute_error',
                                                n_trials=10,
                                                return_best=True,
                                                n_jobs=8
                                                )

results_df.to_csv("data/bayesian_optimization_results.csv", sep=";", index=False)