import pandas as pd

df = pd.read_csv("data/pattern.csv")
df = df.rename(columns={df.columns[0]: 'year', df.columns[1]: 'month'})
df = df.iloc[2:, :]
df.reset_index(inplace=True, drop=True)
month_count = df.year.value_counts().reset_index(name='sum of months per year', drop=True)
print(month_count.head(10))

day_count = df.month.value_counts().reset_index(name='sum of days per month', drop=True)
print(day_count.head(10))