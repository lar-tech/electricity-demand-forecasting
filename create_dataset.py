import pandas as pd
import matplotlib.pyplot as plt
import requests
from meteostat import Hourly

def load_power_plant_data(path: str = 'data/power_plant.csv') -> pd.DataFrame:
    df_power_plant = pd.read_csv(path, delimiter=';')
    df_power_plant = df_power_plant[df_power_plant['Power'] != '-']
    df_power_plant['Power'] = df_power_plant['Power'].str.replace(',', '.').astype(float)
    df_power_plant['Datetime'] = pd.to_datetime(df_power_plant['Datetime'], format='%d.%m.%Y %H:%M')
    df_power_plant = df_power_plant.reindex(columns=['Datetime', 'Power'])
    return df_power_plant

def load_market_data(path: str = 'data/day_ahead_prices.csv') -> pd.DataFrame:
    df_market = pd.read_csv(path, delimiter=';')
    df_market['Datetime'] = pd.to_datetime(df_market['Datetime'], format='%Y-%m-%d %H:%M:00')
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
    weather = Hourly(station_id, start, end)
    df_weather = weather.fetch()
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

    df_weather['Datetime'] = pd.to_datetime(df_weather['Datetime'])
    df_weather = df_weather.set_index('Datetime').sort_index()
    df_weather = df_weather.resample('15min').interpolate(method='linear')
    df_weather = df_weather.reset_index()
    return df_weather

# load power plant data
df = load_power_plant_data(path='data/power_plant.csv')

# load market data
df_market = load_market_data(path='data/day_ahead_prices.csv')

# fetch holiday data
years = df['Datetime'].dt.strftime("%Y").unique()
df_holidays = fetch_holiday_data(years=years, region='de-be')

# fetch weather data
start = df['Datetime'].min()
end = df['Datetime'].max()
df_weather = fetch_weather_data(start=start, end=end, station_id='10582')

# time-based features
df['Holiday'] = df['Datetime'].dt.date.isin(df_holidays['Holiday'])
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

# merge dataframes
df = pd.merge(df, df_weather, on=['Datetime'], how='left')