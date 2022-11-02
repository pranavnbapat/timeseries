import sys

import pandas as pd
import core_functions as cf

import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/train.csv')


# Rename columns, even if column heading is missing
df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'sales'})


# Clean time series
df = cf.clean_timeseries(df)


# Remove outliers
df = cf.remove_outliers_iqr(df)


# To get info
# cf.get_info(df)


print("Grouping by date, just in case, if one date has more than one row of sales")
df = df.groupby('date').sum()
# Resetting the index
df = df.reset_index()


# Identify timeseries pattern
monthly_df = cf.identify_ts_pattern(df)
# From now, we'll use monthly_df

# Needs a lot of improvement. Temporary provision
if monthly_df.shape[0] > 48:
    print("Data is monthly.")
else:
    print("Data is weekly.")

# For weekly
'''
Weekly forecasting is inadequate due to the deterministic effect of holidays and other events. 
Weekly data can be severely skewed by when the holiday occurs and activity before and after the holiday.
df = df.set_index('date')
weekly_df = df['sales'].resample('W').sum()
df = df.reset_index()
weekly_df = weekly_df.reset_index()
print(weekly_df.head())
print(weekly_df.shape)
sys.exit()
'''


cf.sales_duration(monthly_df)


# cf.trend_seasonality_plot(monthly_df)
monthly_df = monthly_df.set_index('date')
cf.time_plot(monthly_df, 'date', 'sales', 'Monthly sales with mean')
monthly_df = monthly_df.reset_index()

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
regressor = 'ct'
trend_check_for_kpss = cf.checking_for_trend(monthly_df['sales'])
if trend_check_for_kpss[1]:
    regressor = 'c'


# Checking for stationarity
adf_p = cf.adfuller_test(monthly_df['sales'])
kpss_p = cf.kpss_test(monthly_df['sales'], regressor=regressor)
'''
"Case 1: Both tests conclude that the series is not stationary - The series is not stationary \n"
"Case 2: Both tests conclude that the series is stationary - The series is stationary \n"
"Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. "
"Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity. \n"
"Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. "
"Differencing is to be used to make series stationary. The differenced series is checked for stationarity."
'''
print("\n")
if adf_p <= 0.05 and kpss_p >= 0.05:
    print("Timeseries is stationary")
elif adf_p >= 0.05 and kpss_p <= 0.05:
    print("Timeseries is not stationary")
elif adf_p <= 0.05 and kpss_p <= 0.05:
    print("Timeseries is difference stationary")
elif adf_p >= 0.05 and kpss_p >= 0.05:
    print("Timeseries is trend stationary")


cf.make_stationary(monthly_df)


# Reset the index
# drop=True prevents reset_index to create a new 'index' column
# df = df.reset_index(drop=True)

