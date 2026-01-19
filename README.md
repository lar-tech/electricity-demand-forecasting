# electricity-demand-forecasting
### Roadmap
1. Create a dataset 
    - Power generation of one power plant ([Heizkraftwerk Berlin-Mitte, Region 50 Hertz](https://www.smard.de/home/ueberblick#!?mapAttributes=%7B%22state%22:%22plant%22,%22plantState%22:%22split%22,%22date%22:1761897600000,%22resolution%22:%22hour%22%7D&filterAttributes=%7B%22company%22:%22%22,%22region%22:%22%22,%22resource%22:%22%22,%22searchText%22:%22%22,%22state%22:%22%22,%22network%22:%22%22,%22commissioning%22:%5B1900,2025%5D,%22power%22:%5B0,3000%5D,%22radius%22:100,%22placeId%22:null,%22zoom%22:11,%22center%22:%5B52.47519539082483,13.448862944133161%5D,%22plant%22:%22KW-Name.Heizkraftwerk%20Berlin%20Mitte%22%7D))
    - Holiday [API](https://digidates.de/api/v1/germanpublicholidays?year={year}&region=de-be)
    - Weather data from nearby Weather station with [meteostat library](https://dev.meteostat.net/python/)
    - Electricity consumption
    - Energy generation (from all sources)
    - Market data

2. Graphical exploration of time series to identify trends, patterns, and seasonal variations. This should help the selection of the most appropriate forecasting model and corresponding features.
    - Plot time series
    - Seasonality plots
    - Autocorrelation plots

3. Comparing and choosing the best model e.g. Forecast Recursive

4. Training and backtesting
    - Hyperparameter tuning

5. Feature selection
    - Model explanaibility and interpretability
    - Feature importance

6. Iterate through 3-5 until satified with the results




    <!-- # lags_grid = [24, 168, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 47, 48, 49, 167, 168, 169)]
    # def search_space(trial):
    #     search_space  = {
    #         'n_estimators' : trial.suggest_int('n_estimators', 300, 1000, step=100),
    #         'max_depth'    : trial.suggest_int('max_depth', 3, 10),
    #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
    #         'reg_alpha'    : trial.suggest_float('reg_alpha', 0, 1),
    #         'reg_lambda'   : trial.suggest_float('reg_lambda', 0, 1),
    #         'lags'         : trial.suggest_categorical('lags', lags_grid)
    #     } 
    #     return search_space


    # 30:57min ohne feature selection -->

    <!-- # match presentation template
color = (39/255, 54/255, 58/255)
plt.rcParams.update({
    "font.family": "Fira Sans",
    "font.size": 18,
    "text.color": color,
    "axes.labelcolor": color,
    "axes.edgecolor": color,
    "xtick.color": color,
    "ytick.color": color,
}) -->


    <!-- # # save metric and predictions
    # refitted_predictions = pd.read_csv('data/results/predictions.csv', delimiter=',')
    # refitted_predictions = refitted_predictions.rename(columns={'Unnamed: 0': 'datetime'})
    # refitted_predictions['datetime'] = pd.to_datetime(refitted_predictions['datetime'], utc=True)
    # refitted_predictions = refitted_predictions.set_index('datetime').sort_index()

    # # metric_2, predictions_2 = backtesting(forecaster, data, cv, exog=None)
    # # plot_predictions(data, predictions, "Recursive LGBM Model", metric["mean_absolute_percentage_error"].values[0])
    # # print(metric_2["mean_absolute_percentage_error"].values[0])

    # plt.figure(figsize=(10,6))
    # val_week = data['grid_load'].loc['2024-03-04':'2024-03-11']
    # val_week.plot(label='Actual Load', color='tab:blue')
    # predictions['pred'].plot(label='Forecast without Refitting', color='tab:red')
    # plt.plot(refitted_predictions['pred'], color='tab:green', label='Forecast with Refitting')
    # plt.xlim(pd.Timestamp('2024-03-04'), pd.Timestamp('2024-03-11'))
    # plt.ylabel("Grid Load in MWh")
    # plt.xlabel("")
    # plt.grid()
    # plt.legend(loc = 'upper right', fontsize=14)
    # plt.tight_layout()
    # plt.savefig("data/plots/final_model.pdf", bbox_inches="tight")
    # plt.show() -->