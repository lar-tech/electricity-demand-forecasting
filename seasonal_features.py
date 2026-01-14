import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/dataset.csv', delimiter=';')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.set_index('datetime').sort_index()

base_dates = [
    pd.Timestamp("2024-01-08"),
    pd.Timestamp("2024-06-15"),
    pd.Timestamp("2024-12-10"),
]

equivalent_dates = []
seasonal_features = []

for date in base_dates:
    eq_dates = []
    seas = []
    for i in range(1, 9):
        year_ago = date - pd.DateOffset(years=i)
        weekday_diff = date.weekday() - year_ago.weekday()
        if weekday_diff < 0:
            weekday_diff += 7
        equivalent_date = year_ago + pd.Timedelta(days=weekday_diff)

        day_start = pd.to_datetime(equivalent_date, utc=True).normalize()
        day_end = day_start + pd.Timedelta(days=1)

        print(day_start)
        print(day_end)

        eq_dates.append(equivalent_date)
        seas.append(df.loc[day_start:day_end]['grid_load'].sum())

    equivalent_dates.append(eq_dates)
    seasonal_features.append(seas)

fig, ax = plt.subplots(figsize=(12, 6))
labels = ["2024-01-08", "2024-06-15", "2024-12-10"]

for i in range(len(equivalent_dates)):
    ax.plot(equivalent_dates[i], seasonal_features[i], marker="o", label=labels[i])

ax.set_ylabel("Electricity Demand [MWh]")
ax.set_xlabel("Year (Equivalent Dates)")
ax.grid(True)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("seasonal_features.pdf", bbox_inches='tight', transparent=True)
plt.show()