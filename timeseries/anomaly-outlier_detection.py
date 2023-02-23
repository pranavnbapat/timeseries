import pandas as pd
import plotly.express as px
import core_functions as cf
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv('data/train.csv')

# Rename columns, even if column heading is missing
df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'sales'})
print(df.shape)
# Clean time series
df = cf.clean_timeseries(df)
print(df.shape)
df = cf.remove_outliers_zscore(df)
print(df.shape)
df = cf.remove_outliers_iqr(df)
print(df.shape)
