from statsmodels.tsa.stattools import adfuller, kpss


def adfuller_test(sales):
    print("Results of Dickey-Fuller Test:")
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

