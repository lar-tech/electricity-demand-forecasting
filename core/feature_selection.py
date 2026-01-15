import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.feature_selection import select_features
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

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
    cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)

    window_features = RollingFeatures(stats=['mean', 'std', 'min', 'max'], window_sizes=[24*3, 24*7, 24*7, 24*7])
    lags = 24
    estimator = LGBMRegressor(random_state=123, verbose=-1, n_estimators=100, max_depth=4, learning_rate=0.20213758391513376, reg_alpha=0.3431780161508694, reg_lambda=0.7290497073840416)
    forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)

    # Recursive Feature Elimination
    selector = RFECV(estimator=estimator, step=1, cv=3)
    # selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select="auto", direction="forward", scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)
    _, _, exog_select = select_features(forecaster      = forecaster,
                                        selector        = selector,
                                        y               = df.loc['2015-01-01':'2024-03-31']['grid_load'].asfreq('h'),
                                        exog            = df.loc['2015-01-01':'2024-03-31'].drop(columns=['grid_load']),
                                        select_only     = None,
                                        force_inclusion = None,
                                        subsample       = 0.5,
                                        random_state    = 123)

    # save selected features
    pd.DataFrame({'selected_exog_features': exog_select}).to_csv('data/results/selected_exog_features.csv', index=False)