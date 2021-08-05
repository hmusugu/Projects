# Time series is different from more traditional classification and regression predictive modeling problems.
#
# The temporal structure adds an order to the observations. This imposed order means that
# important assumptions about the consistency of those observations needs to be handled specifically.
#
# For example, when modeling, there are assumptions that the summary statistics of observations are consistent.
# In time series terminology, we refer to this expectation as the time series being stationary.
#
# These assumptions can be easily violated in time series by the addition of a trend, seasonality,
# and other time-dependent structures.

# Types of Stationary Time Series
# The notion of stationarity comes from the theoretical study of time series and it is a
# useful abstraction when forecasting.
#
# There are some finer-grained notions of stationarity that you may come across if you
# dive deeper into this topic. They are:
#
# Stationary Process: A process that generates a stationary series of observations.
# Stationary Model: A model that describes a stationary series of observations.
# Trend Stationary: A time series that does not exhibit a trend.
# Seasonal Stationary: A time series that does not exhibit seasonality.
# Strictly Stationary: A mathematical definition of a stationary process, specifically that the
# joint distribution of observations is invariant to time shift.

# Augmented Dickey-Fuller test
# Statistical tests make strong assumptions about your data. They can only be used to inform the degree to
# which a null hypothesis can be rejected or fail to be reject. The result must be interpreted
# for a given problem to be meaningful.
#
# Nevertheless, they can provide a quick check and confirmatory evidence that
# your time series is stationary or non-stationary.
#
# The Augmented Dickey-Fuller test is a type of statistical test called a unit root test.
#
# The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.

# It uses an autoregressive model and optimizes an information criterion across multiple different lag values.

# The null hypothesis of the test is that the time series can be represented by a unit root, that it is
# not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis)
# is that the time series is stationary.
#
# Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root,
# meaning it is non-stationary. It has some time dependent structure.
#
# Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root,
# meaning it is stationary. It does not have time-dependent structure.

# We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%)
# suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests
# we fail to reject the null hypothesis (non-stationary).
#
# p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
# p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

# The KPSS test is based on linear regression. It breaks up a series into three parts: a deterministic trend (βt),
# a random walk (rt), and a stationary error (εt), with the regression equation:  xt = rt + βt + ε1.

# Interpreting the Results
# The KPSS test authors derived one-sided LM statistics for the test. If the LM statistic is greater than the
# critical value (given in the table below for alpha levels of 10%, 5% and 1%), then the null hypothesis is rejected;
# the series is non-stationary.
#
# You can also look at the p-value returned by the test and compare it to your chosen alpha level.
# For example, a p-value of 0.02 (2%) would cause the null hypothesis to be rejected at an alpha level of 0.05 (5%).

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd


class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults=True):

        try:
            # Dickey-Fuller test:
            adfTest = adfuller(timeseries, autolag='AIC')

            self.pValue = adfTest[1]

            if (self.pValue < self.SignificanceLevel):
                self.isStationary = True
            else:
                self.isStationary = False

            if printResults:
                dfResults = pd.Series(adfTest[0:4],
                                      index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])

                # Add Critical Values
                for key, value in adfTest[4].items():
                    dfResults['Critical Value (%s)' % key] = value

                print('Augmented Dickey-Fuller Test Results:')
                print(dfResults)
        except Exception as exc:
            self.isStationary = False
            print("ADF could not analyze time-series due to missing data.")
            print(exc.args)

    def KPSS_Test(self, timeseries, printResults=True):

        try:
            # KPSS test:
            kpsstest = kpss(timeseries, regression='c', lags="auto")

            self.pValue = kpsstest[1]

            if (self.pValue < self.SignificanceLevel):
                self.isStationary = False
            else:
                self.isStationary = True

            if printResults:
                kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])

                # Add Critical Values
                for key, value in kpsstest[3].items():
                    kpss_output['Critical Value (%s)' % key] = value

                print('KPSS Test Results:')
                print(kpss_output)
        except Exception as exc:
            self.isStationary = False
            print("KPSS could not analyze time-series due to missing data.")
            print(exc.args)


def runAllTestsOnSeries(sTest, series):
    sTest.ADF_Stationarity_Test(series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))
    resultsDict = ({series.name: [['ADF', sTest.isStationary]]})
    adf_stationary = sTest.isStationary
    sTest.KPSS_Test(series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))
    resultsDict[series.name].append(['KPSS', sTest.isStationary])
    kpss_stationary = sTest.isStationary
    isStationary = adf_stationary & kpss_stationary
    resultsDict[series.name].append(['isStationary', isStationary])
    return resultsDict


def RunTests(sTest, df):
    resultsDict = dict()
    for column in df:
        resultsDict[column] = runAllTestsOnSeries(sTest, df[column])

    return resultsDict
