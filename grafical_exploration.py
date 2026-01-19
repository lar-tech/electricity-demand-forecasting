import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import calendar

# match presentation template
color = (39/255, 54/255, 58/255)
plt.rcParams.update({
    "font.family": "Fira Sans",
    "font.size": 18,
    "text.color": color,
    "axes.labelcolor": color,
    "axes.edgecolor": color,
    "xtick.color": color,
    "ytick.color": color,
})

def plot_grid_load(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['datetime'], df['grid_load'], color='tab:blue')
    plt.title("Grid Load Over Time")
    plt.xlabel("Time")
    plt.ylabel("Grid Load in MWh")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/grid_load_over_time.pdf', bbox_inches='tight')
    plt.show()

def plot_average_year(df):
    df = df[df['datetime'].dt.year != 2014].copy()
    yearly_mean = df.groupby('day_of_year')['grid_load'].mean()

    month_days = [pd.Timestamp(2021, m, 1).day_of_year for m in range(1, 13)]
    month_days.append(366)
    month_names = list(calendar.month_abbr[1:])

    plt.figure(figsize=(10,6))
    plt.plot(yearly_mean, color='tab:blue', label='Mean Grid Load')
    plt.annotate('New Year', xy=(1, yearly_mean[1]), xytext=(20, 10000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    plt.annotate('1st May', xy=(121, yearly_mean[121]), xytext=(140, 10000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    plt.annotate('3rd Oct', xy=(276, yearly_mean[276]), xytext=(246, 10000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    plt.annotate('Reformation Day', xy=(304, yearly_mean[304]), xytext=(290, 10500),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    plt.annotate('Christmas Holidays', xy=(359, yearly_mean[359]), xytext=(290, 9500),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    for i in range(12):
        color = 'lightgray' if i % 2 == 0 else 'white'
        plt.axvspan(month_days[i], month_days[i+1], color=color, alpha=0.2)
    plt.xticks([(month_days[i] + month_days[i+1]) / 2 for i in range(12)], month_names)
    plt.xlabel("Month")
    plt.ylabel("Grid Load in MWh")
    plt.title("Average Grid Load Over the Year")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/average_grid_load.pdf', bbox_inches='tight')
    plt.show()

def plot_average_day_hour(df):
    df = df[df['datetime'].dt.year != 2014].copy()
    grp = df.groupby(['day_of_week'])['grid_load'].agg(['mean', 'std']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, sharex=False)

    axes[0].plot(grp['day_of_week'], grp['mean'])
    axes[0].set_ylim(9000, 14000)
    axes[0].set_ylabel("Grid Load in MWh")
    labels = 'Mon. Tue. Wed. Thu. Fri. Sat. Sun.'.split()
    positions = range(len(labels))
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].grid()

    grp = df.groupby(['hour'])['grid_load'].agg(['mean', 'std']).reset_index()
    axes[1].plot(grp['hour'], grp['mean'])
    axes[1].set_xlabel("Hour of the Day")
    axes[1].set_xticks(range(0, 24, 4))
    axes[1].set_xlim(0, 23)
    axes[1].grid()

    fig.suptitle("Average Grid Load by Day and Hour", y=0.95)
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/average_day_hour.pdf', bbox_inches='tight')
    plt.show()

def plot_average_load_by_day_and_season(df):
    df = df[df['datetime'].dt.year != 2014].copy()
    month = df['datetime'].dt.month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['Season'] = month.map(season_map)
    order = ['Winter', 'Spring', 'Summer', 'Autumn']

    grp = df.groupby(['Season','day_of_week'])['grid_load'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True, sharex=True)

    for ax, season in zip(axes.flat, order):
        temp = grp[grp['Season'] == season]
        ax.plot(temp['day_of_week'], temp['mean'])
        ax.fill_between(temp['day_of_week'], temp['mean'] - temp['std'], temp['mean'] + temp['std'], alpha=0.2)
        ax.set_title(season)
        ax.set_xticks(range(0, 6))
        ax.grid()

    axes[1,0].set_xlabel("Day of the Week")
    axes[1,1].set_xlabel("Day of the Week")
    axes[0,0].set_ylabel("Grid Load in MWh")
    axes[1,0].set_ylabel("Grid Load in MWh")

    fig.suptitle("Average Grid Load by Day – per Season", y=0.95)
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/average_grid_load_by_day_and_season.pdf', bbox_inches='tight')
    plt.show()

def plot_average_load_by_hour_and_season(df):
    df = df[df['datetime'].dt.year != 2014].copy()
    df['Hour'] = df['datetime'].dt.hour
    month = df['datetime'].dt.month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['Season'] = month.map(season_map)
    order = ['Winter', 'Spring', 'Summer', 'Autumn']

    grp = df.groupby(['Season','Hour'])['grid_load'].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True, sharex=True)
    for ax, season in zip(axes.flat, order):
        temp = grp[grp['Season'] == season]
        ax.plot(temp['Hour'], temp['mean'])
        ax.fill_between(temp['Hour'], temp['mean'] - temp['std'], temp['mean'] + temp['std'], alpha=0.2)
        ax.set_title(season)
        ax.set_xticks(range(0, 24, 3))
        ax.grid()
    axes[1,0].set_xlabel("Hour of the Day")
    axes[1,1].set_xlabel("Hour of the Day")
    axes[1,0].set_ylabel("Grid Load in MWh")
    axes[0,0].set_ylabel("Grid Load in MWh")

    fig.suptitle("Average Load by Hour – per Season", y=0.95)
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/average_load_by_hour_and_season.pdf', bbox_inches='tight')
    plt.show()

def plot_autocorrelation(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(df['grid_load'], ax=ax, title="", lags=60, alpha=None)
    plt.xlabel("Lags in hours")
    plt.ylabel("Normalized Correlation")
    plt.title("Autocorrelation of Grid Load")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/autocorrelation.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(df['grid_load'], ax=ax, lags=60, title="")
    plt.xlabel("Lags in hours")
    plt.ylabel("Normalized Correlation")
    plt.title("Partial Autocorrelation of Grid Load")
    plt.grid()
    plt.tight_layout()
    if EXPORT:
        plt.savefig('results/grafical_exploration/partial_autocorrelation.pdf', bbox_inches='tight', transparent=True)
    plt.show()

# params
EXPORT = True
os.makedirs('results/grafical_exploration', exist_ok=True)

# load dataset
df = pd.read_csv('dataset.csv', delimiter=';')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# plotting functions
plot_grid_load(df)
plot_average_year(df)
plot_average_day_hour(df)
plot_average_load_by_day_and_season(df)
plot_average_load_by_hour_and_season(df)
plot_autocorrelation(df)
