import pandas as pd
import plotly.express as px
import core_functions as cf
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv('data/train.csv')

# Rename columns, even if column heading is missing
df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'sales'})

# Clean time series
df = cf.clean_timeseries(df)

print("Grouping by date, just in case, if one date has more than one row of sales")
df = df.groupby('date').sum()
# Resetting the index
df = df.reset_index()

# Identify timeseries pattern
timely_df = cf.identify_ts_pattern(df)
cf.sales_duration(timely_df)

print(timely_df.head())

# Interactive plots

'''fig = px.line(timely_df.reset_index(), x='date', y='sales', title='Sales')
fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(buttons=list([
                     dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(count=2, label="3y", step="year", stepmode="backward"),
                     dict(count=3, label="5y", step="year", stepmode="backward"),
                     dict(step="all")
                 ]))
                 )

fig.show()'''
