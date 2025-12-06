import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import calendar

# plt.rcParams['font.family'] = 'Fira Sans'
# plt.rcParams['font.size'] = 20
EXPORT = True 
PLOT = False

def plot_consumption(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['Datetime'], df['Grid Load'], color='tab:blue')
    if EXPORT:
        plt.title("Power Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Grid Load [MWh]")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/power_consumption_over_time.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_load_distribution_by_year(df):
    df = df[df['Datetime'].dt.year != 2014].copy().copy()
    df['Year'] = df['Datetime'].dt.year
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Year', y='Grid Load')
    plt.title("Distribution of Consumption by Year")
    plt.xlabel("")
    plt.ylabel("Grid Load [MWh]")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/load_distribution_by_year.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_average_annual_course(df):
    df = df[df['Datetime'].dt.year != 2014].copy()
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    yearly_mean = df.groupby('DayOfYear')['Grid Load'].mean()

    month_days = [pd.Timestamp(2021, m, 1).day_of_year for m in range(1, 13)]
    month_days.append(366)
    month_names = list(calendar.month_abbr[1:])

    plt.figure(figsize=(10,6))
    plt.plot(yearly_mean, color='tab:blue', label='Mean Grid Load')
    for i in range(12):
        color = 'lightgray' if i % 2 == 0 else 'white'
        plt.axvspan(month_days[i], month_days[i+1], color=color, alpha=0.2)
    plt.xticks([(month_days[i] + month_days[i+1]) / 2 for i in range(12)], month_names)
    plt.title("Average Annual Course of Grid Load")
    plt.xlabel("Month")
    plt.ylabel("Average Grid Load [MWh]")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/average_annual_course.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_average_load_by_day_and_season(df):
    df = df[df['Datetime'].dt.year != 2014].copy()
    month = df['Datetime'].dt.month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['Season'] = month.map(season_map)
    order = ['Winter', 'Spring', 'Summer', 'Autumn']

    grp = df.groupby(['Season','DayOfWeek'])['Grid Load'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True, sharex=True)

    for ax, season in zip(axes.flat, order):
        temp = grp[grp['Season'] == season]
        ax.plot(temp['DayOfWeek'], temp['mean'])
        ax.fill_between(temp['DayOfWeek'], temp['mean'] - temp['std'], temp['mean'] + temp['std'], alpha=0.2)
        ax.set_title(season)
        # ax.set_xlabel("Day of the Week")
        # ax.set_ylabel("Avg. Power [MWh]")
        ax.set_xticks(range(0, 6))
        ax.grid()

    axes[1,0].set_xlabel("Day of the Week")
    axes[1,1].set_xlabel("Day of the Week")
    axes[0,0].set_ylabel("Avg. Load [MWh]")
    axes[1,0].set_ylabel("Avg. Load [MWh]")

    fig.suptitle("Average Load by Day – per Season", y=0.98)
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/average_load_by_day_and_season.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_average_load_by_hour_and_season(df):
    df = df[df['Datetime'].dt.year != 2014].copy()
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

    grp = df.groupby(['Season','Hour'])['Grid Load'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True, sharex=True)
    for ax, season in zip(axes.flat, order):
        temp = grp[grp['Season'] == season]
        ax.plot(temp['Hour'], temp['mean'])
        ax.fill_between(temp['Hour'], temp['mean'] - temp['std'], temp['mean'] + temp['std'], alpha=0.2)
        ax.set_title(season)
        # ax.set_xlabel("Hour of the Day")
        # ax.set_ylabel("Avg. Power [MWh]")
        ax.set_xticks(range(0, 24, 3))
        ax.grid()
    axes[1,0].set_xlabel("Hour of the Day")
    axes[1,1].set_xlabel("Hour of the Day")
    axes[0,0].set_ylabel("Avg. Load [MWh]")
    axes[1,0].set_ylabel("Avg. Load [MWh]")

    fig.suptitle("Average Load by Hour – per Season", y=0.98)
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/average_load_by_hour_and_season.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_power_generation_by_source(df):
    plt.figure(figsize=(10,6))
    plt.stackplot(df['Datetime'],
                df['Lignite'],
                df['Wind Onshore'],
                df['Biomass'],
                df['Hydro'],
                df['Wind Offshore'],
                df['Solar'],
                df['Hard Coal'],
                df['Natural Gas'],
                df['Pumped Storage'],
                df['Other Conventional'])
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
                'Other Conventional'], loc='upper right', ncols=3, fontsize=15)
    plt.xlim(pd.Timestamp('2023-07-10'), pd.Timestamp('2023-07-20'))
    plt.xticks(rotation=45)
    plt.grid()
    plt.title("Power Generation by Source")
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/power_generation_by_source.png', dpi=500, bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_grid_and_residual_load_over_time(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['Datetime'], df['Grid Load'], color='tab:blue', label='Grid Load')
    plt.plot(df['Datetime'], df['Residual Load'], color='tab:orange', alpha=0.7, label='Residual Load')
    plt.title("Grid and Residual Load Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power [MW]")
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/grid_and_residual_load_over_time.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_load_and_temperature_over_time(df):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df['Datetime'], df['Grid Load'], label='Grid Load', color='tab:blue')
    ax1.set_ylabel("Grid Load [MWh]", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(df['Datetime'], df['10384 Temperature'], color='tab:orange', alpha=0.7, label='Temperature')
    ax2.set_ylabel("Temperature [°C]", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    fig.suptitle("Load and Temperature over Time")
    
    ax1.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/load_and_temperature_over_time.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_correlation_heatmap(df):
    # plt.rcParams['font.size'] = 12
    plt.figure()
    num_cols = ['Grid Load', 'Residual Load', 'Solar', 'Wind Onshore', 'Wind Offshore',
                '10384 Temperature', '10384 Average Wind Speed', '10384 Sunshine Duration']

    corr = df[num_cols].corr()
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", center=0, square=True, annot_kws={"size": 8})
    plt.title("Correlation between Power, Generation and Weather")
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/correlation_heatmap.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

def plot_autocorrelation(df):
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_acf(df['Grid Load'], ax=ax)
    plt.grid()
    plt.title("Autocorrelation of Grid Load")
    plt.tight_layout()
    if EXPORT:
        plt.savefig('data/plots/autocorrelation_grid_load.svg', bbox_inches='tight')
    if PLOT:
        plt.show()

# load dataset
df = pd.read_csv('data/dataset.csv', delimiter=';')
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

# plotting
plot_consumption(df)
plot_load_distribution_by_year(df)
plot_average_annual_course(df)
plot_average_load_by_day_and_season(df)
plot_average_load_by_hour_and_season(df)
plot_power_generation_by_source(df)
plot_grid_and_residual_load_over_time(df)
plot_load_and_temperature_over_time(df)
plot_correlation_heatmap(df)
plot_autocorrelation(df)
