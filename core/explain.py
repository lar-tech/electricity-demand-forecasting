import os
import ast
import numpy as np
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
from skforecast.exceptions import MissingValuesWarning, LongTrainingWarning
warnings.simplefilter('ignore', category=MissingValuesWarning)
warnings.simplefilter('ignore', category=LongTrainingWarning)

def sequential_feature_importance(importances_sorted, estimator, data, cv, test_end, metrics):
    features_use = []
    lags = []
    roll_pairs = []
    seen_pairs = set()
    results = []
    for i, feature in enumerate(importances_sorted):
        if feature.startswith('lag_'):
            lags.append(int(feature.split('_')[1]))
        elif feature.startswith('roll_'):
            _, stat, win = feature.split('_', 2)
            pair = (stat, int(win))
            if pair not in seen_pairs:
                roll_pairs.append(pair)
                seen_pairs.add(pair)
        else:
            if feature in data.columns:
                features_use.append(feature)
        wf = None
        if roll_pairs:
            wf = RollingFeatures(stats=[p[0] for p in roll_pairs], window_sizes=[p[1] for p in roll_pairs]) 

        forecaster_i = ForecasterRecursive(estimator=estimator, lags=sorted(set(lags)) if lags else None, window_features=wf)
        metric, _ = backtesting_forecaster(
                                    forecaster=forecaster_i,
                                    y = df.loc['2015-01-01':test_end]['grid_load'].asfreq('h'),
                                    exog=df.loc['2015-01-01':test_end][features_use] if features_use else None,
                                    cv=cv,
                                    metric=metrics,
                                    verbose=False)
        results.append(metric["mean_absolute_percentage_error"].iloc[0])
        if i % 5 == 0:
            print(f"Processed {i} features")
    return results

def feature_importance(reg, estimator_name):
    # XGBoost
    if estimator_name == "XGBRegressor":
        print("Extracting feature importance for XGBRegressor")
        booster = reg.get_booster()

        score_gain = booster.get_score(importance_type="gain")
        score_split = booster.get_score(importance_type="weight")  # split count

        imp_gain = (pd.DataFrame({"feature": list(score_gain.keys()),
                                  "importance": list(score_gain.values())})
                    .sort_values("importance", ascending=False))
        imp_split = (pd.DataFrame({"feature": list(score_split.keys()),
                                   "importance": list(score_split.values())})
                     .sort_values("importance", ascending=False))
    # LightGBM
    elif estimator_name == "LGBMRegressor":
        booster = reg.booster_
        feature_names = booster.feature_name()

        imp_gain = (pd.DataFrame({"feature": feature_names,
                                  "importance": booster.feature_importance(importance_type="gain")})
                    .sort_values("importance", ascending=False))
        imp_split = (pd.DataFrame({"feature": feature_names,
                                   "importance": booster.feature_importance(importance_type="split")})
                     .sort_values("importance", ascending=False))
        
    return imp_gain, imp_split

if __name__ == "__main__":
    # params
    start = '2015-01-01'
    train_end = '2023-12-31'
    test_end = '2024-12-31'
    eval_metric = 'mean_absolute_percentage_error'
    lags = 24
    window_features = RollingFeatures(stats=['mean', 'std', 'min', 'max'], window_sizes=[24*3, 24*7, 24*7, 24*7])
    os.makedirs('results/explain_model', exist_ok=True)

    # load dataset
    df = pd.read_csv('dataset.csv', delimiter=';')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    data = df.copy()
    data_train = data.loc[start:train_end].asfreq('h')

    # define cross-validation
    cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)

    # define estimators
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
        
    estimators = [LGBMRegressor(**params_lgbm, verbose=-1), XGBRegressor(**params_xgb, verbosity=0)]

    for estimator in estimators:
        estimator_name = estimator.__class__.__name__
        print(f"Processing estimator: {estimator_name}")
        if estimator_name == "LGBMRegressor":
            lags = best_lags_lgbm
        else:
            lags = best_lags_xgb
        forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)
        forecaster.fit(y=data_train['grid_load'], exog=data_train.drop(columns=['grid_load']))
        imp_gain, imp_split = feature_importance(forecaster.estimator, estimator_name)
        
        # Sequentially add features
        results_gain = sequential_feature_importance(imp_gain["feature"].to_list(), estimator, data, cv, test_end, eval_metric)
        results_split = sequential_feature_importance(imp_split["feature"].to_list(), estimator, data, cv, test_end, eval_metric)

        # save results
        results_gain_df = pd.DataFrame({"feature": imp_gain["feature"].to_numpy(),
                                        "importance_gain": imp_gain["importance"].to_numpy(),
                                        "sequential_mape_gain": np.asarray(results_gain)})
        results_gain_df["k"] = np.arange(1, len(results_gain_df) + 1)

        results_split_df = pd.DataFrame({"feature": imp_split["feature"].to_numpy(),
                                         "importance_split": imp_split["importance"].to_numpy(),
                                         "sequential_mape_split": np.asarray(results_split)})
        results_split_df["k"] = np.arange(1, len(results_split_df) + 1)

        results_gain_df.to_csv(f"results/explain_model/feature_importance_gain_{estimator_name}.csv", index=False)
        results_split_df.to_csv(f"results/explain_model/feature_importance_split_{estimator_name}.csv", index=False)
