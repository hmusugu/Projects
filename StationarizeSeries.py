import pandas as pd


def differenceSeries(key, df, sTest):
    diffdf = pd.DataFrame()
    for column in df:
        if key == column:
            differencedSeries = df[column].diff()
            diffdf[key] = differencedSeries[1:]
            sTest.ADF_Stationarity_Test(differencedSeries, printResults=True)
            print("Is the differenced time series stationary? {0}".format(sTest.isStationary))
            pd.DataFrame(differencedSeries).to_csv("C:/Users/hmusugu/Desktop/Molex/CogForecasting/MVA/Stationarized Data" + key + " StationarizedData.csv")
    return diffdf[key]


# Trend and Difference Stationary
# A random walk with or without a drift can be transformed to a stationary process
# by differencing (subtracting Yt-1 from Yt, taking the difference Yt - Yt-1)
# correspondingly to Yt - Yt-1 = εt or Yt - Yt-1 = α + εt and then the process becomes difference-stationary.
# The disadvantage of differencing is that the process loses one observation each time the difference is taken.

# A non-stationary process with a deterministic trend becomes stationary after removing the trend, or detrending.
# For example, Yt = α + βt + εt is transformed into a stationary process by subtracting the trend βt: Yt - βt = α + εt.
# No observation is lost when detrending is used to transform a non-stationary process to a stationary one.

# In the case of a random walk with a drift and deterministic trend, detrending can remove
# the deterministic trend and the drift, but the variance will continue to go to infinity.
# As a result, differencing must also be applied to remove the stochastic trend.

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col) + '_2d'] = (df_train[col].iloc[-1] - df_train[col].iloc[-2]) + df_fc[
                str(col) + '_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col) + '_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
    return df_fc
