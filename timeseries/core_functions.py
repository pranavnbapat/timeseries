import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib
from matplotlib import pyplot as plt
import pymannkendall as mk
matplotlib.use('TkAgg')     # Only for pycharm
from statsmodels.tsa.stattools import adfuller, kpss


def get_info(df):
    print("Getting info...")
    if isinstance(df, pd.DataFrame):
        print(df.describe())
        print(df.info())


def clean_timeseries(df):
    # Remove white noise
    # coerce: invalid parsing will be set as NaN
    print("Removing white noise")
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').dropna()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dropna()

    # Sort the values by date
    print("Sorting by date")
    df = df.sort_values(by=['date'])

    return df


# The IQR is calculated as the difference between the data's 75th and 25th percentiles and defines the box in a box
# and whisker plot.
def remove_outliers_iqr(data):
    print("Removing outliers (interquartile range)")
    df = data
    df.set_index('date', inplace=True)

    q25, q75 = np.percentile(df['sales'], 25), np.percentile(df['sales'], 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    # identify outliers
    outliers = [x for x in df['sales'] if x < lower or x > upper]

    print('Identified outliers: %d' % len(outliers))

    # remove outliers
    outliers_removed = [x for x in df['sales'] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))

    df = df[~df['sales'].isin(outliers)]
    df = df.reset_index()

    return df


def remove_outliers_quantile(data):
    df = data
    df.set_index('date', inplace=True)
    y = df['sales']
    removed_outliers = y.between(y.quantile(.05), y.quantile(.95))
    index_names = df[~removed_outliers].index  # INVERT removed_outliers!!
    df.drop(index_names, inplace=True)
    df = df.reset_index()

    return df


# If you know the distribution of values in the sample is Gaussian or Gaussian-like, you can use the
# standard deviation of the sample as a cut-off for identifying outliers.
def remove_outliers_std(data):
    print("Removing outliers (standard deviation)")
    df = data
    data_mean, data_std = np.mean(df['sales']), np.std(df['sales'])

    # identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off

    # identify outliers
    outliers = [x for x in df['sales'] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))

    # remove outliers
    outliers_removed = [x for x in df['sales'] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))

    df = df[~df['sales'].isin(outliers)]
    df = df.reset_index()

    return df


def mod_zscore(col, thresh):
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.6745 * ((col - med_col) / med_abs_dev)
    mod_z = mod_z[np.abs(mod_z) > thresh]
    return np.abs(mod_z)


def identify_ts_pattern(df):
    # Resampling generates a unique sampling distribution on the basis of the actual data.
    # W : weekly frequency
    # M : month end frequency
    # SM : semi-month end frequency (15th and end of month)
    # Q : quarter end frequency
    # MS : month start frequency
    print("Counting years/months/weeks/days to identify time pattern")
    df = df.set_index('date')
    monthly_df = df['sales'].resample('MS').sum()
    df = df.reset_index()
    monthly_df = monthly_df.reset_index()
    # print(monthly_df.head())
    # print(monthly_df.shape)
    return monthly_df


# Duration of dataset
def sales_duration(data):
    data.date = pd.to_datetime(data.date)
    number_of_days = data.date.max() - data.date.min()
    number_of_months = number_of_days.days / 30
    number_of_years = number_of_days.days / 365
    print(number_of_days.days, 'days')
    print(round(number_of_months, 2), 'months')
    print(round(number_of_years, 2), 'years')


def trend_seasonality_plot(df):
    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
    sns.boxplot(x=df['date'].dt.year, y='sales', data=df, ax=axes[0])
    sns.boxplot(x=df['date'].dt.month, y='sales', data=df.loc[~df.date.isin([df['date'].dt.year.iloc[0], df['date'].dt.year.iloc[-1]]), :])

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    plt.show()


def adfuller_test(sales):
    print("\nResults of Dickey-Fuller Test:")
    dftest = adfuller(sales, autolag="AIC")
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(dftest, labels):
        print(label + ' : ' + str(value))

    if dftest[1] <= 0.05:
        print("Strong evidence against the null hypothesis. Therefore, reject the null hypothesis and accept "
              "alternate hypothesis. Time series has no unit root and is stationary.")
    else:
        print("Weak evidence against null hypothesis. Therefore, accept null hypothesis and reject alternate "
              "hypothesis. Time series has a unit root, indicating it is non-stationary.")

    return dftest[1]


def kpss_test(sales, regressor='c'):
    # “c” : The data is stationary around a constant (default).
    # “ct” : The data is stationary around a trend.
    print("\nResults of KPSS Test:")
    kpsstest = kpss(sales, regression=regressor, nlags="auto")
    labels = ['KPSS Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(kpsstest, labels):
        print(label + ' : ' + str(value))

    if kpsstest[1] <= 0.05:
        print("Weak evidence against null hypothesis. Therefore, accept null hypothesis and reject alternate "
              "hypothesis. Time series has a unit root, indicating it is non-stationary.")
    else:
        print("Strong evidence against the null hypothesis. Therefore, reject the null hypothesis and accept "
              "alternate hypothesis. Data has no unit root and is stationary.")

    return kpsstest[1]


def checking_for_trend(sales):
    # https://www.statology.org/mann-kendall-test-python/
    return mk.original_test(sales)


def get_diff(data, time_diff):
    # time is in months for monthly data
    data['sales_diff'] = data.sales.diff(time_diff)
    data = data.dropna()
    return data


def time_plot(data, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x_col, y_col, data=data, ax=ax, color='mediumblue', label='Total Sales')
    second = data.groupby(data.date.dt.year)[y_col].mean().reset_index()
    second.date = pd.to_datetime(second.date, format='%Y')
    sns.lineplot((second.date + datetime.timedelta(6*365/12)), y_col, data=second, ax=ax, color='red',
                 label='Mean Sales')
    ax.set(xlabel="Date", ylabel="Sales", title=title)
    sns.despine()


def make_stationary(df):
    df = get_diff(df, 1)
    time_plot(df, 'date', 'sales_diff', 'Monthly sales with differencing')
