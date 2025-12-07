from datetime import timedelta, datetime
import re
from meteostat import Hourly
import pandas as pd 
import requests
from tqdm import tqdm

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
            .reset_index(names='datetime'))
    
    return df

def fetch_market_data(start: str, end: str, country: str) -> pd.DataFrame:
    url = f"https://api.energy-charts.info/price?bzn={country}&start={start}&end={end}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['unix_seconds'], unit='s', utc=True).dt.tz_convert('Europe/Berlin')
    df.rename(columns={'price': 'day_ahead_price'}, inplace=True)
    df = df[['datetime', 'day_ahead_price']]

    df = (df.set_index('datetime')
          .resample('1h', closed="left", label="right")
          .mean()
          .reset_index())
    
    return df

def fetch_holiday_data(years: list[int]) -> pd.DataFrame:
    regions = ['de-be', 'de-bb', 'de-mv', 'de-sn', 'de-st', 'de-th', 'de-hh']
    holiday_counts = {}
    
    for year in years:
        for region in regions:
            url = f"https://digidates.de/api/v1/germanpublicholidays?year={year}&region={region}"
            response = requests.get(url)
            holidays = response.json()
            
            for date_str in holidays.keys():
                date = pd.to_datetime(date_str).date()
                holiday_counts[date] = holiday_counts.get(date, 0) + 1
    
    df_holidays = pd.DataFrame({
        'datetime': pd.to_datetime(list(holiday_counts.keys())).tz_localize('Europe/Berlin'),
        'holiday_count': list(holiday_counts.values())
    })
    return df_holidays

def scrape_school_holidays_data(year_start=2015, year_end=2025):
    def convert(d_str, year):
        d_str = d_str.strip(".")
        day, month = map(int, d_str.split("."))
        return datetime(year, month, day)
    
    states=["Berlin", "Brandenburg", "Sachsen", "Sachsen-Anhalt", "Thüringen", "Mecklenburg-Vorpommern", "Hamburg"]

    df_schoolholidays = pd.DataFrame({"datetime": pd.to_datetime([]), **{f"holiday_{state}": [] for state in states}})
    for year in range(year_start, year_end + 1):
        url = f"https://www.schulferien.org/deutschland/ferien/{year}/"
        df = pd.read_html(url)[0]
        for state in states:
            df_state = df[(df.iloc[:, 0] == state) | (df.iloc[:, 0] == f"*  {state}")].copy()
            df_state.dropna()
            for i in range(1,7,1):
                # convert times
                holiday_dates = df_state.iloc[:, i].values[0].split(", ")[0]
                holiday_dates = holiday_dates.replace("*", "")
                if not any(char.isdigit() for char in holiday_dates):
                    continue
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

                new_rows = pd.DataFrame({"datetime": days, f"holiday_{state}": True})
                df_schoolholidays = pd.concat([df_schoolholidays, new_rows], ignore_index=True).sort_values(by="datetime")


    agg_dict = {col: "any" for col in df_schoolholidays.columns if col != "datetime"}
    df_schoolholidays = (df_schoolholidays.groupby("datetime", as_index=False).agg(agg_dict))
    holiday_cols = [col for col in df_schoolholidays.columns if col.startswith("holiday_")]
    df_schoolholidays["school_holiday"] = df_schoolholidays[holiday_cols].sum(axis=1)
    df_schoolholidays = df_schoolholidays[["datetime", "school_holiday"]]
        
    return df_schoolholidays

def fetch_weather_data(start, end):
    station_ids = [10384, 10091, 10131, 10488, 10554]
    df_weather_all = pd.DataFrame()
    for station_id in station_ids:
        df_weather = Hourly(station_id, start, end).fetch()
        df_weather.index = df_weather.index.tz_localize('UTC')
        df_weather = df_weather.reset_index()

        df_weather = df_weather.rename(columns={'time': 'datetime'})
        df_weather = df_weather.rename(columns={
            'temp': f'{station_id}_temperature',
            'dwpt': f'{station_id}_dew_point',
            'rhum': f'{station_id}_relative_humidity',
            'prcp': f'{station_id}_precipitation',
            'snow': f'{station_id}_snow_depth',
            'wdir': f'{station_id}_wind_direction',
            'wspd': f'{station_id}_average_wind_speed',
            'wpgt': f'{station_id}_peak_wind_speed',
            'pres': f'{station_id}_average_sea_level_air_pressure',
            'tsun': f'{station_id}_sunshine_duration',
            'coco': f'{station_id}_weather_condition_code'
        })

        if df_weather_all.empty:
            df_weather_all = df_weather
        else:
            df_weather_all = pd.merge(df_weather_all, df_weather, on=['datetime'], how='left')
    return df_weather_all

# fetch power consumption data
consumption = {
    410: "grid_load",
    4359: "residual_load",
    4387: "pumped_storage_load"
}
df = fetch_smard_data(start_date=datetime(2015, 1, 1), end_date=datetime(2025, 1, 1), filters=consumption, region="50Hertz", resolution="hour")

# fetch power generation data
generation = {
    1223: "lignite",
    4071: "natural_gas",
    4069: "hard_coal",
    1227: "other_conventional",
    1225: "wind_offshore",
    4067: "wind_onshore",
    4068: "solar",
    1226: "hydro",
    4066: "biomass",
    4070: "pumped_storage",
    1228: "other_renewable",
}
df_generation = fetch_smard_data(start_date=datetime(2015, 1, 1), end_date=datetime(2025, 1, 1), filters=generation, region="50Hertz", resolution="hour")

# fetch forcasted generation data
forcasted_generation = {
    3791: "forecast_wind_offshore",
    123: "forecast_wind_onshore",
    125: "forecast_solar",
    715: "forecast_other"
}
df_forcasted_generation = fetch_smard_data(start_date=datetime(2015, 1, 1), end_date=datetime(2025, 1, 1), filters=forcasted_generation, region="50Hertz", resolution="hour")

# fetch market data
df_market = fetch_market_data("2015-01-01", "2018-09-30", "DE-AT-LU")
df_market_2 = fetch_market_data("2018-10-01", "2025-01-01", "DE-LU")
df_market = pd.concat([df_market, df_market_2], ignore_index=True)

# fetch holiday data
years = df['datetime'].dt.strftime("%Y").unique()
df_holidays = fetch_holiday_data(years=years)
df['date'] = df['datetime'].dt.date
df_holidays['date'] = df_holidays['datetime'].dt.date
df = df.merge(df_holidays[['date', 'holiday_count']], on='date', how='left').drop(columns=['date'])
df['holiday_count'] = df['holiday_count'].fillna(0)

# scrape school holidays data
df_school_holidays = scrape_school_holidays_data()
df['date'] = df['datetime'].dt.date
df_school_holidays['date'] = df_school_holidays['datetime'].dt.date
df = df.merge(df_school_holidays[['date', 'school_holiday']], on='date', how='left').drop(columns=['date'])
df['school_holiday'] = df['school_holiday'].fillna(0)

# fetch weather data
start = df['datetime'].min().tz_localize(None)
end = df['datetime'].max().tz_localize(None) + timedelta(hours=1)
df_weather = fetch_weather_data(start=start, end=end)

# time-based features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['week'] = df['datetime'].dt.isocalendar().week
df['month'] = df['datetime'].dt.month

# merge dataframes
df = pd.merge(df, df_generation, on=['datetime'], how='left')
df = pd.merge(df, df_market, on=['datetime'], how='left')
df = pd.merge(df, df_weather, on=['datetime'], how='left')
df = df.astype({col: 'float64' for col in df.select_dtypes(include=['Float64', 'UInt32']).columns})
df.to_csv('data/dataset.csv', sep=';', index=False)
print(df.head())
print(df.tail())