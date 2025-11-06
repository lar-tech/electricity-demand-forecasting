import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import calendar

def plot_power_time(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Datetime'], df['Power'])
    plt.title("Power over Time")
    plt.ylabel("Power [MWh]")
    plt.grid()
    plt.tight_layout()
    # plt.show()

def plot_power_distribution_by_year(df):
    df['Year'] = df['Datetime'].dt.year
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Year', y='Power')
    plt.title("Distribution of Power by Year")
    plt.xlabel("")
    plt.ylabel("Power [MWh]")
    plt.grid()
    plt.tight_layout()
    # plt.show()

def plot_average_annual_course(df):
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    yearly_mean = df.groupby('DayOfYear')['Power'].mean()

    month_days = [pd.Timestamp(2021, m, 1).day_of_year for m in range(1, 13)]
    month_days.append(366)
    month_names = list(calendar.month_abbr[1:])

    plt.figure(figsize=(10,6))
    plt.plot(yearly_mean, color='tab:blue', label='Mean Power')
    for i in range(12):
        color = 'lightgray' if i % 2 == 0 else 'white'
        plt.axvspan(month_days[i], month_days[i+1], color=color, alpha=0.2)
    plt.xticks([(month_days[i] + month_days[i+1]) / 2 for i in range(12)], month_names)
    plt.title("Average Annual Course of Power")
    plt.xlabel("Month")
    plt.ylabel("Average Power [MWh]")
    plt.grid()
    plt.tight_layout()
    # plt.show()

def plot_power_distribution_by_day_and_season(df):
    month = df['Datetime'].dt.month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['Season'] = month.map(season_map)
    order = ['Winter', 'Spring', 'Summer', 'Autumn']

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, season in zip(axes.flat, order):
        temp = df[df['Season'] == season]
        data = [temp[temp['DayOfWeek'] == d]['Power'] for d in range(7)]
        ax.violinplot(data, showmeans=True)
        ax.set_title(season)
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Average Power [MWh]")
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.grid()
    fig.suptitle("Average Power by Day – per Season", y=0.98)
    plt.tight_layout()
    # plt.show()

def plot_average_power_by_day_and_season(df):
    month = df['Datetime'].dt.month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['Season'] = month.map(season_map)
    order = ['Winter', 'Spring', 'Summer', 'Autumn']

    grp = df.groupby(['Season','DayOfWeek'])['Power'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, season in zip(axes.flat, order):
        temp = grp[grp['Season'] == season]
        ax.plot(temp['DayOfWeek'], temp['mean'])
        ax.fill_between(temp['DayOfWeek'], temp['mean'] - temp['std'], temp['mean'] + temp['std'], alpha=0.2)
        ax.set_title(season)
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Average Power [MWh]")
        ax.set_xticks(range(0, 6))
        ax.grid()

    fig.suptitle("Average Power by Day – per Season", y=0.98)
    plt.tight_layout()
    # plt.show()

def plot_average_power_by_hour_and_season(df):
    df['Hour'] = df['Datetime'].dt.hour
    month = df['Datetime'].dt.month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['Season'] = month.map(season_map)
    order = ['Winter', 'Spring', 'Summer', 'Autumn']

    grp = df.groupby(['Season','Hour'])['Power'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, season in zip(axes.flat, order):
        temp = grp[grp['Season'] == season]
        ax.plot(temp['Hour'], temp['mean'])
        ax.fill_between(temp['Hour'], temp['mean'] - temp['std'], temp['mean'] + temp['std'], alpha=0.2)
        ax.set_title(season)
        ax.set_xlabel("Hour of the Day")
        ax.set_ylabel("Average Power [MWh]")
        ax.set_xticks(range(0, 24, 3))
        ax.grid()

    fig.suptitle("Average Power by Hour – per Season", y=0.98)
    plt.tight_layout()
    # plt.show()

def plot_power_generation_by_source(df):
    plt.figure(figsize=(10,6))
    plt.stackplot(df['Datetime'],
                df['Lignite Generation'],
                df['Wind Onshore Generation'],
                df['Biomass Generation'],
                df['Hydro Generation'],
                df['Wind Offshore Generation'],
                df['Solar Generation'],
                df['Hard Coal Generation'],
                df['Natural Gas Generation'],
                df['Pumped Storage Generation'],
                df['Other Conventional Generation'])
    plt.legend([
                'Lignite',
                'Wind Onshore',
                'Biomass',
                'Hydro',
                'Wind Offshore',
                'Solar',
                'Hard Coal',
                'Natural Gas',
                'Pumped Storage',
                'Other Conventional'], loc='upper right')
    plt.xlim(pd.Timestamp('2023-07-10'), pd.Timestamp('2023-07-20'))
    plt.xticks(rotation=45)
    plt.grid()
    plt.title("Power Generation by Source")
    plt.tight_layout()
    # plt.show()

def plot_grid_and_residual_load_over_time(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['Datetime'], df['Grid Load'], color='tab:blue', label='Grid Load')
    plt.plot(df['Datetime'], df['Residual Load'], color='tab:orange', alpha=0.7, label='Residual Load')
    plt.title("Grid and Residual Load Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power [MW]")
    plt.grid()
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.show()

def plot_power_and_temperature_over_time(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Datetime'], df['Power'], label='Power')
    plt.plot(df['Datetime'], df['Temperature'], color='tab:orange', alpha=0.7, label='Temperature')
    plt.title("Power and Temperature over Time")
    plt.ylabel("Power [MWh]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.show()

def plot_correlation_heatmap(df):
    num_cols = ['Power','Solar Generation','Wind Onshore Generation', 'Lignite Generation' ,'Grid Load', 'Residual Load',
                'Temperature','Average Wind Speed','Sunshine Duration']

    corr = df[num_cols].corr()
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", center=0, square=True,annot_kws={"size": 8})
    plt.title("Correlation between Power, Generation and Weather")
    plt.tight_layout()
    plt.show()

# load dataset
df = pd.read_csv('data/dataset.csv', delimiter=';')
df['Datetime'] = pd.to_datetime(df['Datetime'])

# plotting
plot_power_time(df)
plot_power_distribution_by_year(df)
plot_average_annual_course(df)
plot_power_distribution_by_day_and_season(df)
plot_average_power_by_day_and_season(df)
plot_average_power_by_hour_and_season(df)
plot_power_generation_by_source(df)
plot_grid_and_residual_load_over_time(df)
plot_power_and_temperature_over_time(df)
plot_correlation_heatmap(df)
