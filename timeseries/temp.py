import sys

import pandas as pd
import core_functions as cf
import seaborn as sns
import datetime
import matplotlib
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings('ignore')


# Declare variables
trend_flag = 0      # 0 for no trend, 1 for trend
seasonality_flag = 0    # 0 for no seasonality, 1 for seasonality
regressor = 'ct'


# Load data
df = pd.read_csv('data/train.csv')


# Rename columns, even if column heading is missing
df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'sales'})


# Clean time series
df = cf.clean_timeseries(df)


# Remove outliers
# df = cf.remove_outliers_iqr(df)


# To get info
# cf.get_info(df)


print("Grouping by date, just in case, if one date has more than one row of sales")
df = df.groupby('date').sum()
# Resetting the index
df = df.reset_index()


# Identify timeseries pattern
timely_df = cf.identify_ts_pattern(df)
cf.sales_duration(timely_df)
# From now, we'll use timely_df


# Get average sales
cf.avg_timely_sales(timely_df)


# Get trend and seasonality plots
# cf.trend_seasonality_plot(timely_df)


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


# Get time plot
# cf.time_plot(timely_df, 'date', 'sales', 'Ssales with mean')


# Get line plot
# cf.get_line_plot(timely_df)


# Checking for trend
'''
trend: This tells the trend. Possible output includes increasing, decreasing, or no trend.
h: True if trend is present. False if no trend is present.
p: The p-value of the test.
z: The normalize test statistic.
Tau: Kendall Tau.
s: Mann-Kendal’s score
var_s: Variance S
slope: Theil-Sen estimator/slope
intercept: Intercept of Kendall-Theil Robust Line
'''

trend_check_for_kpss = cf.checking_for_trend(timely_df['sales'])
if trend_check_for_kpss[1]:
    regressor = 'c'
    trend_flag = 1


# Checking for stationarity
adf_p = cf.adfuller_test(timely_df['sales'])
kpss_p = cf.kpss_test(timely_df['sales'], regressor=regressor)
cf.draw_line()
'''
"Case 1: Both tests conclude that the series is not stationary - The series is not stationary \n"
"Case 2: Both tests conclude that the series is stationary - The series is stationary \n"
"Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. "
"Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity. \n"
"Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. "
"Differencing is to be used to make series stationary. The differenced series is checked for stationarity."
'''

if adf_p <= 0.05 and kpss_p >= 0.05:
    print("Timeseries is stationary")
elif adf_p >= 0.05 and kpss_p <= 0.05:
    print("Timeseries is not stationary")
elif adf_p <= 0.05 and kpss_p <= 0.05:
    print("Timeseries is difference stationary")
elif adf_p >= 0.05 and kpss_p >= 0.05:
    print("Timeseries is trend stationary")


# Making time series stationary
stationary_df = cf.get_diff(timely_df, 1)
cf.time_plot(stationary_df, 'date', 'sales_diff', 'Monthly sales with mean')
# timely_df = timely_df.reset_index(drop=True)
timely_df = timely_df.drop(columns=['sales_diff'])


# Seasonal decomposition
timely_df = timely_df.set_index('date')
# decomposition = cf.get_seasonal_decomposition(timely_df)
timely_df = timely_df.reset_index(drop=True)


# Identifying seasonality
acf = plot_acf(stationary_df['sales_diff'])
plt.show()


