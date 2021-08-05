import NonStationaryDataGenerator as nsdg
import StationarityTests
import StationarizeSeries
import CalculateMapeAndBias
import pandas as pd

# '''Constants'''
startDate = '8-1-2016'
endDate = '8-1-2019'
mu = 10000
sigma = mu * 0.05

# from stationarizer import simple_auto_stationarize

# Currently only the following simple flow - dealing with unit roots - is implemented:
#
# Data validation is performed: all columns are checked to be numeric, and the time dimension is
# assumed to be larger than the number of series (although this is not mandatory, and so only a warning
# is thrown in case of violation).
#
# Both the Augmented Dickey-Fuller unit root test and the KPSS test are performed for each of the series.

# The p-values of all tests are corrected to control the false discovery rate (FDR) at some given level,
# using the Benjamini–Yekutieli procedure.
#
# The joint ADF-KPSS results are interpreted for each test.
#
# For each time series for which the presence of a unit root cannot be rejected, the series is diffentiated.
#
# For each time series for which the presence of a trend cannot be rejected, the series is de-trended.
#
# If any series was differentiated, then any un-differentiated time series (if any) are trimmed by
# one step to match the resulting series length.


# EMD for nonstationary/nonlinear data

# What physical "process" does the data represent?  What are the underlying mechanisms that drive change?
# The trend of the data should be an intrinsic property of the data driven by same mechanisms that create the data.
# The (intrinsic) method used will then be adaptive, trend extracted will be derived from and based on the data.
# Trend should exist within the data span and be a property associated with the corresponding local time scales.

# Definition: The trend is an intrinsically fitted monotonic function or a function in which there can be at
# most one extremum within a given data span.

# Definition: Detrending is the operation of removing the trend.  The variability is the residue of the data
# after the removal of the trend withing a given data span.

# Process
# The EMD will break down a signal into its component IMFs.
# An IMF is a function that:
# 1. has only one extreme between zero crossings, and
# 2. has a mean value of zero.
# In order to describe the process, we borrow from our poster the following section:
# The Sifting Process
# The sifting process is what EMD uses to decomposes the signal into IMFs.
# The sifting process is as follows:
# * For a signal X(t), let m1 be the mean of its upper and lower envelopes as determined from a
# cubic-spline interpolation of local maxima and minima. The locality is determined by an arbitrary parameter;
# the calculation time and the effectiveness of the EMD depends greatly on such a parameter.
# * The first component h1 is computed:
#       h1=X(t)-m1
# * In the second sifting process, h1 is treated as the data, and m11 is the mean of h1's upper and lower envelopes:
#       h11=h1-m11
# * This sifting procedure is repeated k times, until h1k is an IMF, that is:
#       h1(k-1)-m1k=h1k
# * Then it is designated as c1=h1k, the first IMF component from the data, which contains the
# shortest period component of the signal. We separate it from the rest of the data: X(t)-c1 = r1
# The procedure is repeated on rj: r1-c2 = r2,....,rn-1 - cn = rn
# The result is a set of functions; the number of functions in the set depends on the original signal.


# region initialize StationarityTests, date list and dataframe
sTest = StationarityTests.StationarityTests()
dateList = nsdg.getDayDateList(startDate, endDate)
shippingDf = nsdg.generateTimeSeriesDataFrame(dateList)
demandDf = nsdg.generateTimeSeriesDataFrame(dateList)
# endregion

# region fill dataframe with distribution types
nsdg.fillNormalDistribution(shippingDf, mu, sigma, "Shipment")
nsdg.fillNormalDistribution(demandDf, mu, sigma, "Demand")
#nsdg.randomWalk_wDrift_and_DeterministicPositiveTrend(shippingDf, mu, sigma, "Shipment")
#nsdg.randomWalk_wDrift_and_DeterministicPositiveTrend(demandDf, mu, sigma, "Demand")

hist_demandDf = CalculateMapeAndBias.getValuesFromDemandHistory(demandDf, mu, sigma, "Demand")
hist_shippingDf = CalculateMapeAndBias.getValuesFromDemandHistory(shippingDf, mu, sigma, "Shipping")

'''Need to join not concat these two dfs'''
finalDf = pd.merge(hist_demandDf, hist_shippingDf, left_index=True, right_index=True)

# Export Demand and Shipping history
pd.DataFrame(finalDf).to_csv("C:/Users/jamesharris29/Desktop/DemandandShippingHistoryReport.csv")

# nsdg.pureRandomWalk(df, mu, sigma)
# nsdg.randomWalk_wPositiveDrift(df, mu, sigma)
# nsdg.randomWalk_wNegativeDrift(df, mu, sigma)
# nsdg.deterministicPositiveTrend(df, mu, sigma)
# nsdg.deterministicNegativeTrend(df, mu, sigma)
# nsdg.randomWalk_wDrift_and_DeterministicPositiveTrend(df, mu, sigma)
# nsdg.randomWalk_wDrift_and_DeterministicNegativeTrend(df, mu, sigma)
# nsdg.fillNrmlNonStnryMuConstantSigma(df, mu, sigma)
# nsdg.fillNrmlNonStnrySigmaConstantMean(df, mu, sigma)
# nsdg.fillNrmlNonStnryCovarianceConstantMean(df, mu, sigma)
# endregion

#Join Mape and Bias to shipment data


# region run stationarity tests
results = StationarityTests.RunTests(sTest, finalDf)
for k in results:
    isStationary = False
    isStationaryADF = False
    isStationaryKPSS = False
    for j in results[k]:
        type = j[0]
        if type == 'ADF':
            isStationaryADF = j[1]
        elif type == 'KPSS':
            isStationaryKPSS = j[1]
        elif type == 'isStationary':
            isStationary == j[1]

    if not isStationary:
        StationarizeSeries.differenceSeries(k, finalDf, sTest)

# endregion





