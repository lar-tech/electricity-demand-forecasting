from datetime import timedelta, datetime
import re
from meteostat import Hourly
import pandas as pd 
import requests
from tqdm import tqdm

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=';')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:00')
    df['Datetime'] = (df['Datetime']
                      .dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
                      .dt.tz_convert('UTC'))
    # resample to hourly resolution
    df = (df.set_index('Datetime')
          .resample('1h', closed="left", label="right")
          .mean()
          .multiply(4.0)
          .reset_index())

    return df

def fetch_smard_data(start_date: datetime, end_date: datetime, filters: dict, region: str, resolution: str) -> pd.DataFrame:
    base_url = "https://www.smard.de/app/chart_data"
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    def get_timestamps(filter_id: int) -> list[int]:
        url = f"{base_url}/{filter_id}/{region}/index_{resolution}.json"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()["timestamps"]

    def get_series(filter_id: int, timestamp: int) -> list:
        url = f"{base_url}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()["series"]

    all_data = {}
    for filter_id, column_name in tqdm(filters.items(), desc="Downloading"):
        timestamps = get_timestamps(filter_id)
        relevant_ts = [ts for ts in timestamps if ts <= end_ts and ts + 7*24*3600*1000 >= start_ts]
        
        series_data = {}
        for ts in relevant_ts:
            for timestamp_ms, value in get_series(filter_id, ts):
                if timestamp_ms and start_ts <= timestamp_ms <= end_ts:
                    series_data[timestamp_ms] = value
        
        all_data[column_name] = series_data

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index, unit='ms', utc=True).tz_convert('Europe/Berlin')
    df = (df.sort_index()
            .resample('1h', closed="left", label="right")
            .interpolate(method='linear')
            .reset_index(names='Datetime'))
    
    return df

def fetch_holiday_data(years: list[int], region: str = 'de-be') -> pd.DataFrame:
    holiday_dates = []
    for year in years:
        url = f"https://digidates.de/api/v1/germanpublicholidays?year={year}&region={region}"
        response = requests.get(url)
        holidays = response.json()
        [holiday_dates.append(pd.to_datetime(holiday)) for holiday in holidays.keys()]

    df_holidays = pd.DataFrame(data={"Holiday": holiday_dates})
    return df_holidays

def scrap_school_holidays_data(year_start=2015, year_end=2025):
    def convert(d_str, year):
        d_str = d_str.strip(".")
        day, month = map(int, d_str.split("."))
        return datetime(year, month, day)
                
    df_schoolholidays = pd.DataFrame({"date": pd.to_datetime([]), "holiday": []})
    for year in range(year_start, year_end + 1):
        url = f"https://www.schulferien.org/deutschland/ferien/{year}/"
        df = pd.read_html(url)[0]
        df_berlin = df[(df.iloc[:, 0] == "Berlin") | (df.iloc[:, 0] == "*  Berlin")].copy()
        df_berlin.dropna()
        for i in range(1,7,1):
            header = df_berlin.columns[i]
            holiday_name = header[1].strip()
            # convert times
            holiday_dates = df_berlin.iloc[:, i].values[0].split(", ")[0]
            holiday_dates = holiday_dates.replace("*", "")
            parts = holiday_dates.split("+")

            range_part = None
            extra_parts = []
            for p in parts:
                if "-" in p:
                    range_part = p
                else:
                    extra_parts.append(p)

            if range_part:
                # Extract start and end date
                start_str, end_str = re.split(r"\s*-\s*", range_part.strip().replace(" ", ""))
                start = convert(start_str, year)
                end   = convert(end_str, year)

                # for christmas holidays
                if end < start:
                    end = end.replace(year=year + 1)

                # generate all dates in the given range
                days = list(pd.date_range(start, end, freq="D"))
                for extra in extra_parts:
                    extra_date = convert(extra.strip(), year)
                    days.append(extra_date)
            else:
                days = []
                for extra in extra_parts:
                    extra_date = convert(extra.strip(), year)
                    days.append(extra_date)

            holiday_mapping = {"Winterferien": "1",
                               "Osterferien": "2",
                               "Pfingstferien": "3",
                               "Sommerferien": "4",
                               "Herbstferien": "5",
                               "Weihnachtsferien": "6"}
            holiday_name = holiday_mapping.get(holiday_name, holiday_name)
            new_rows = pd.DataFrame({"date": days, "holiday": holiday_name})
            df_schoolholidays = pd.concat([df_schoolholidays, new_rows], ignore_index=True).sort_values(by="date")

    # correct bridge holidays
    bridge_holidays = {(datetime(2015, 5, 15), "7"),
                       (datetime(2016, 5, 6), "7"),
                       (datetime(2017, 5, 24), "7"),
                       (datetime(2017, 5, 26), "7"),
                       (datetime(2017, 10, 2), "7"),
                       (datetime(2018, 4, 30), "7"),
                       (datetime(2018, 5, 11), "7"),
                       (datetime(2019, 5, 31), "7"),
                       (datetime(2019, 10, 4), "7"), 
                       (datetime(2020, 5, 8), "7"),
                       (datetime(2020, 5, 22), "7"),
                       (datetime(2021, 5, 14), "7"),
                       (datetime(2021, 10, 4), "7"),
                       (datetime(2021, 12, 23), "7"),
                       (datetime(2022, 3, 7), "7"),
                       (datetime(2022, 5, 27), "7"),
                       (datetime(2023, 5, 19), "7"),
                       (datetime(2023, 10, 2), "7"),
                       (datetime(2024, 5, 10), "7"),
                       (datetime(2024, 10, 4), "7"),
                       (datetime(2025, 5, 2), "7"),
                       (datetime(2025, 5, 30), "7")}
    for date, holiday_code in bridge_holidays:
        df_schoolholidays.loc[(df_schoolholidays['date'] == date), 'holiday'] = holiday_code

    return df_schoolholidays

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

    return df_weather

# fetch power consumption data
consumption = {
    410: "Grid Load",
    4359: "Residual Load",
    4387: "Pumped Storage Load"
}
df = fetch_smard_data(start_date=datetime(2015, 1, 1), end_date=datetime(2015, 2, 1), filters=consumption, region="50Hertz", resolution="hour")

# fetch power generation data
generation = {
    1223: "Lignite",
    4071: "Natural Gas",
    4069: "Hard Coal",
    1227: "Other Conventional",
    1225: "Wind Offshore",
    4067: "Wind Onshore",
    4068: "Solar",
    1226: "Hydro",
    4066: "Biomass",
    4070: "Pumped Storage",
    1228: "Other Renewable",
}
df_generation = fetch_smard_data(start_date=datetime(2015, 1, 1), end_date=datetime(2015, 2, 1), filters=generation, region="50Hertz", resolution="hour")

# fetch forcasted generation data
forcasted_generation = {
    3791: "Forecast Wind Offshore",
    123: "Forecast Wind Onshore",
    125: "Forecast Solar",
    715: "Forecast Other"
}
df_forcasted_generation = fetch_smard_data(start_date=datetime(2015, 1, 1), end_date=datetime(2015, 2, 1), filters=forcasted_generation, region="50Hertz", resolution="hour")

# # fetch market data
# df_market = load_dataset(path='data/day_ahead_prices.csv')

# fetch holiday data
years = df['Datetime'].dt.strftime("%Y").unique()
df_holidays = fetch_holiday_data(years=years, region='de-be')

# scrap school holidays data
df_school_holidays = scrap_school_holidays_data()

# fetch weather data
start = df['Datetime'].min().tz_localize(None)
end = df['Datetime'].max().tz_localize(None) + timedelta(hours=1)
df_weather = fetch_weather_data(start=start, end=end, station_id='10582')
df_weather.to_csv('data/weather.csv', index=False)

# time-based features
df['Holiday'] = df['Datetime'].dt.date.isin(df_holidays['Holiday'].dt.date)
mapping = df_school_holidays.set_index('date')['holiday']
df['SchoolHoliday'] = df['Datetime'].dt.date.map(mapping).fillna(0)
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

# merge dataframes
df = pd.merge(df, df_weather, on=['Datetime'], how='left')
df = pd.merge(df, df_generation, on=['Datetime'], how='left')
# df = pd.merge(df, df_market, on=['Datetime'], how='left')
df.to_csv('data/dataset.csv', sep=';', index=False)
print(df.head())
print(df.tail())