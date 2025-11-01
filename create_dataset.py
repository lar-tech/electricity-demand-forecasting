from datetime import timedelta
from meteostat import Hourly
import pandas as pd
import requests

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=';')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:00')
    df['Datetime'] = (df['Datetime']
                      .dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
                      .dt.tz_convert('UTC'))
    return df

def load_market_data(path: str = 'data/day_ahead_prices.csv') -> pd.DataFrame:
    df_market = pd.read_csv(path, delimiter=';')
    df_market['Datetime'] = pd.to_datetime(df_market['Datetime'], format='%Y-%m-%d %H:%M:00')
    df_market['Datetime'] = (df_market['Datetime']
                                .dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
                                .dt.tz_convert('UTC'))
    return df_market

def fetch_holiday_data(years: list[int], region: str = 'de-be') -> pd.DataFrame:
    holiday_dates = []
    for year in years:
        url = f"https://digidates.de/api/v1/germanpublicholidays?year={year}&region={region}"
        response = requests.get(url)
        holidays = response.json()
        [holiday_dates.append(holiday) for holiday in holidays.keys()]

    df_holidays = pd.DataFrame(data={"Holiday": holiday_dates})
    return df_holidays

def fetch_weather_data(start: pd.Timestamp, end: pd.Timestamp, station_id: str) -> pd.DataFrame:
    df_weather = Hourly(station_id, start, end).fetch()
    df_weather.index = df_weather.index.tz_localize('UTC')
    df_weather = df_weather.reset_index()

    df_weather = df_weather.rename(columns={
        'time': 'Datetime',
        'temp': 'Temperature',
        'dwpt': 'Dew Point',
        'rhum': 'Relative Humidity',
        'prcp': 'Precipitation',
        'snow': 'Snow Depth',
        'wdir': 'Wind Direction',
        'wspd': 'Average Wind Speed',
        'wpgt': 'Peak Wind Speed',
        'pres': 'Average Sea-Level Air Pressure',
        'tsun': 'Sunshine Duration',
        'coco': 'Weather Condition Code'
    })

    df_weather = (df_weather
                    .set_index('Datetime')
                    .sort_index()
                    .resample('15min')
                    .interpolate(method='linear')
                    .reset_index())
    
    return df_weather

# load power plant data
df = load_dataset(path='data/power_plant.csv')

# load power consumption data
df_power_consumption = load_dataset(path='data/power_consumption.csv')

# load power generation data
df_power_generation = load_dataset(path='data/power_generation.csv')

# load market data
df_market = load_dataset(path='data/day_ahead_prices.csv')

# fetch holiday data
years = df['Datetime'].dt.strftime("%Y").unique()
df_holidays = fetch_holiday_data(years=years, region='de-be')

# fetch weather data
start = df['Datetime'].min().tz_localize(None)
end = df['Datetime'].max().tz_localize(None) + timedelta(hours=1)
df_weather = fetch_weather_data(start=start, end=end, station_id='10582')
df_weather.to_csv('data/weather.csv', index=False)

# time-based features
df['Holiday'] = df['Datetime'].dt.date.isin(df_holidays['Holiday'])
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

# merge dataframes
df = pd.merge(df, df_weather, on=['Datetime'], how='left')
df = pd.merge(df, df_power_consumption, on=['Datetime'], how='left')
df = pd.merge(df, df_power_generation, on=['Datetime'], how='left')
df = pd.merge(df, df_market, on=['Datetime'], how='left')
df.to_csv('data/dataset.csv', sep=';', index=False)
print(df.head())
print(df.tail())