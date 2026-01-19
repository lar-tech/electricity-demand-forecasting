import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.feature_selection import select_features
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from lightgbm import LGBMRegressor
import warnings
from skforecast.exceptions import MissingValuesWarning
warnings.simplefilter('ignore', category=MissingValuesWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

if __name__ == "__main__":
    # load dataset
    df = pd.read_csv('data/dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    important_features = df.drop(columns=['grid_load']).columns.tolist()
    important_features = [feat for feat in important_features if not feat[0].isdigit()]
    # print(len(important_features), important_features)

    # create train and validation sets
    data = df.copy()
    data_train = data.loc['2015-01-01':'2023-12-31'].asfreq('h')

    # define cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)
    window_features = RollingFeatures(stats=['mean', 'std', 'min', 'max'], window_sizes=[24*3, 24*7, 24*7, 24*7])
    lags = 24
    # estimator = LGBMRegressor(random_state=123, verbose=-1)
    estimator = LGBMRegressor(random_state=123, verbose=-1, learning_rate=0.04802270167510142, n_estimators=500,
                                num_leaves=166, max_depth=6, min_child_samples=110, subsample=0.9404067086517863,
                                subsample_freq=1, colsample_bytree=0.6772559947743917, reg_alpha=4.588474569608268e-05,
                                reg_lambda=0.0004650965158027969, n_jobs=1)
    forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)

    # Recursive Feature Elimination
    selector = RFECV(estimator=estimator, step=1, cv=tscv, scoring='neg_mean_absolute_error', verbose=1, n_jobs=16)
    _, _, exog_select = select_features(forecaster      = forecaster,
                                        selector        = selector,
                                        y               = df.loc['2015-01-01':'2024-12-31']['grid_load'].asfreq('h'),
                                        exog            = df.loc['2015-01-01':'2024-12-31'].drop(columns=['grid_load']),
                                        select_only     = None,
                                        force_inclusion = None,
                                        subsample       = 0.25,
                                        random_state    = 123,
                                        verbose=True)

    # save selected features
    pd.DataFrame({'selected_exog_features': exog_select}).to_csv('data/selected_exog_features.csv', index=False)