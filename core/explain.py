import matplotlib.pyplot as plt
import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor

def sequential_feature_importance(importances_sorted, estimator, data, cv, metrics):
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
        metric, predictions = backtesting_forecaster(
                                    forecaster=forecaster_i,
                                    y=data['grid_load'].asfreq('h'),
                                    exog=data[features_use] if features_use else None,
                                    cv=cv,
                                    metric=metrics,
                                    verbose=False)
        results.append(metric["mean_absolute_percentage_error"].iloc[0])
        if i == 30:
            break
        if i % 5 == 0:
            print(f"Processed {i} features")

    return results

# load dataset
df = pd.read_csv('data/dataset.csv', delimiter=';')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.set_index('datetime').sort_index()

# create train and validation sets
data = df.copy()
data_train = data.loc['2015-01-01':'2024-03-02'].asfreq('h')
data_val = data.loc['2024-03-03':'2024-03-09'].asfreq('h')

# define cross-validation
cv = TimeSeriesFold(steps=24, initial_train_size=len(data_train), refit=False)

# autoregressive model with LightGBM
estimator = LGBMRegressor(random_state=123, verbose=-1, n_estimators=800, max_depth=6, learning_rate=0.20213758391513376, reg_alpha=0.3431780161508694, reg_lambda=0.7290497073840416)
window_features = RollingFeatures(stats=['mean', 'std', 'min', 'max'], window_sizes=[24*3, 24*7, 24*7, 24*7])
lags = 24
forecaster = ForecasterRecursive(estimator=estimator, lags=lags, window_features=window_features)
forecaster.fit(y=data_train['grid_load'], exog=data_train.drop(columns=['grid_load']))

# Feature importance
reg = forecaster.estimator
booster = reg.booster_
feature_names = booster.feature_name()
imp_gain = pd.DataFrame({"feature": feature_names,
                        "importance": booster.feature_importance(importance_type="gain")
                        }).sort_values("importance", ascending=True)
imp_split = pd.DataFrame({"feature": feature_names,
                          "importance": booster.feature_importance(importance_type="split")
                          }).sort_values("importance", ascending=True)

metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
imp_gain_sorted = imp_gain.sort_values('importance', ascending=False)['feature'].tolist()
imp_split_sorted = imp_split.sort_values('importance', ascending=False)['feature'].tolist()

# Sequentially add features
results_gain = sequential_feature_importance(imp_gain_sorted, estimator, data, cv, metrics)
results_split = sequential_feature_importance(imp_split_sorted, estimator, data, cv, metrics)

# plot
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].barh(imp_gain["feature"].tail(10), imp_gain["importance"].tail(10))
ax[0].set_xscale('log')
ax[0].set_xlabel("Total gain")
ax[1].barh(imp_split["feature"].tail(10), imp_split["importance"].tail(10))
ax[1].set_xlabel("Split count")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].plot(range(1, len(results_gain)+1), results_gain, marker='o')
ax[0].set_xlabel('Number of Features Used')
ax[0].set_ylabel('MAPE')
ax[0].set_title('Feature importance: Gain')
ax[0].grid()

ax[1].plot(range(1, len(results_split)+1), results_split, marker='o')
ax[1].set_xlabel('Number of Features Used')
ax[1].set_ylabel('MAPE')
ax[1].set_title('Feature importance: Split count')
ax[1].grid()
plt.tight_layout()
plt.show()