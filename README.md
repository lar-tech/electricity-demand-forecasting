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


### To-Do
- Wetterdaten aus der 50Hertz Regio fetchen
    - Mittelwert bilden aus alle Wetterstationen?

Ferien:
Winterferien: 1
Osterferien: 2
Pfingstferien: 3
Sommerferien: 4
Herbstferien: 5
Weihnachtsferien: 6

It is very important to use the closed='left' and label='right' arguments correctly to avoid introducing future information into the training, leakage). Suppose that values are available for 10:10, 10:30, 10:45, 11:00, 11:12, and 11:30. To obtain the hourly average, the value assigned to 11:00 must be calculated using the values for 10:10, 10:30, and 10:45; and the value assigned to 12:00 must be calculated using the value for 11:00, 11:12 and 11:30.