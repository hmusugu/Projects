 # -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:02:30 2019

@author: hmusugu
"""
from datetime import datetime
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

import StationarityTests
import StationarizeSeries

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn import metrics
from datetime import datetime
from sklearn.metrics import confusion_matrix
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Make dynamic file name 

#
def import_files():
    train = pd.read_csv('C:/Users/hmusugu/Desktop/Initiatives/Supplier Agility/10/Train.csv',parse_dates=['date'],index_col='date' )#('../input/train.csv' ,parse_dates=['date'],index_col='date')
    train = train.dropna()
    test = pd.read_csv('C:/Users/hmusugu/Desktop/Initiatives/Supplier Agility/10/Test.csv',parse_dates=['date'],index_col='date')#('../input/test.csv', parse_dates=['date'],index_col='date')
    test = test.dropna()
    future = pd.read_csv('C:/Users/hmusugu/Desktop/Initiatives/Supplier Agility/10/Forecast.csv',parse_dates=['date'],index_col='date')#('../input/test.csv', parse_dates=['date'],index_col='date')
    future = future.dropna()
    df = pd.concat([train,test,future],sort=True)
    #df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.groupby(df.index).mean()
    return df,train, test,future
#df =df.replace([np.inf, -np.inf], np.nan)
#df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Shipping"], how="all")
df, train, test, future = import_files()


#def intial_plots():
#Plotting initial Plots:
plt.plot(df['Shipped'])



#df['Shipping'] = df['Shipping'].dropna()


#def Stationarizer():
sTest = StationarityTests.StationarityTests()
results2 = StationarityTests.RunTests(sTest, df)

nonStationaryData = False
df_Stnry = pd.DataFrame()
for k in results2.items():
    isStationary = False
    isStationaryADF = False
    isStationaryKPSS = False
    columns = k[1].items()
    for j in columns:
        testStats = j[1]
        for t in testStats:
            testType = t[0]
            if testType == 'ADF':
                isStationaryADF = t[1]
            elif testType == 'KPSS':
                isStationaryKPSS = t[1]
            elif testType == 'isStationary':
                isStationary = t[1]
        if not isStationary:
            df_Stnry[k[0]] = StationarizeSeries.differenceSeries(k[0], df, sTest)
            nonStationaryData = True
        else:
            df_Stnry[k[0]] = df[k[0]]
plt.plot(df_Stnry['Shipped'])





#Include correlation co-efficient function (Nisarg)***



#Setting train and test  start and end dates
#train dates
x1=[]
for i1 in range(len(train.index)):
    x1.append(str(train.index[i1]))
y1=[]
z1=[]
for i2 in range(len(train.index)):
    y1.append(str(x1[i2]).split(' '))
    z1.append(y1[i2][0])
    
#Test Dates
x2=[]
for i3 in range(len(test.index)):
    x2.append(str(test.index[i3]))
y2=[]
z2=[]
for i4 in range(len(test.index)):
    y2.append(str(x2[i4]).split(' '))
    z2.append(y2[i4][0])
    

tr_start,tr_end = z1[0],z1[len(z1)-1]  # Make it dynamic by including locations of these dates
te_start,te_end = z2[0],z2[len(z2)-1]

# Dropping rows with NULL values
tra = df['Shipped'][tr_start:tr_end].dropna()
tes = df['Shipped'][te_start:te_end].dropna()
#fut = df_Stnry['Shipped'][fu_start:fu_end].dropna()

#Model Building
#def model_muild():
exog_train = df[['Demand','Mape','Bias']][tr_start:tr_end].dropna()
exog_test  = df[['Demand','Mape','Bias']][te_start:te_end].dropna()
#exog_future= df_Stnry[['Demand','Mape','Bias']][fu_start:fu_end].dropna()

tra_original = df_Stnry['Shipped'][tr_start:tr_end].dropna()
#arimax = sm.tsa.statespace.SARIMAX(tra,order=(0,0,1),seasonal_order=(0,0,0,0),exog = exog_train,enforce_stationarity=False, enforce_invertibility=False,).fit()
#arimax.summary()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


AIC = []
PDQ=[]
S_PDQ = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(tra_original,
                                            order=param,freq = 'D',
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,exog = exog_train)
            results = mod.fit()
            AIC.append(results.aic)
            PDQ.append(param)
            S_PDQ.append(param_seasonal)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
#AIC = AIC
min_AIC = min(AIC)
for a1 in range(len(AIC)):
    if (AIC[a1]==min_AIC):
        min_index = a1

opt_PDQ = PDQ[min_index]
opt_S_PDQ = S_PDQ[min_index]
p,d,q=opt_PDQ[0],opt_PDQ[1],opt_PDQ[2]
P,D,Q=opt_S_PDQ[0],opt_S_PDQ[1],opt_S_PDQ[2]

#def run_model(): '''Make sure to include time stamps so that we know the run-time'''
mod = sm.tsa.statespace.SARIMAX(tra,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 12),freq = 'D', exog = exog_train,enforce_stationarity=False,enforce_invertibility=False)
res = mod.fit()
print(res.summary().tables[1])

#results.plot_diagnostics(figsize=(16, 8))
#plt.show()

pred = res.get_prediction(te_start,te_end, exog = exog_test)

#pred_fu = res.get_prediction('2020-04-07','2020-05-25',exog = exog_future)
demand_predicted = test.iloc[:,1:2]
shipped_predicted = round(pred.predicted_mean)
plt.plot(shipped_predicted)
#shipped_predicted_future = round(pred_fu.predicted_mean)
#pred = np.array(pred)

pred_ci = pred.conf_int()
'''
ax = tes.plot(label='Actuals')
pred.predicted_mean.plot(ax=ax, label='Shipped_Forecasts', alpha=.7, figsize=(14, 7))
demand_predicted.plot(ax=ax, label='Demand', alpha=2, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Ship Date')
ax.set_ylabel('Part1 Shipped Units')
plt.legend()
plt.show()
'''
#plt.plot(exog_test['Demand'])
#plt.plot(shipped_predicted)


#Inversion
tra2 = pd.DataFrame(tra_original, columns=["Shipped"])
shippedPrediction = pd.DataFrame()
shippedPrediction["Shipped"] = round(pred.predicted_mean)
if nonStationaryData:
    invertedData = StationarizeSeries.invert_transformation(tra2, shippedPrediction)
    #pd.DataFrame(invertedData).to_csv("C:/Users/hmusugu/Desktop/Supplier Agility/invertedData.csv")
else:
    pd.DataFrame(shippedPrediction).to_csv("C:/Users/hmusugu/Desktop/Supplier Agility/invertedData.csv")

shipped_pred_inv = invertedData.iloc[:,1:2]



plt.plot(test['Demand'])
plt.plot(shipped_predicted)
# Choosing the best model with the least AIC score value:
#def choose_model():
'''Code for choosing model based on least AIC values is missing '''# (Harshit***)

shipped_predicted = round(pred.predicted_mean)

def flag_column(sup,dem):
    flag = []
    sup = np.array(sup)
    dem = np.array(dem)
    for i1 in range (len(sup)):
        if (sup[i1]<dem[i1]):
            flag.append("Yes")
        else:
            flag.append("No")
    return flag

shipped_pred_inv.to_csv("predictions.csv")
#On the chosen model we now want to see how the metrics look like:           
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

invertedData.index=invertedData.index.strftime("%m/%d/%Y")
future.index = future.index.strftime("%m/%d/%Y")
tes.index = tes.index.strftime("%m/%d/%Y")
#Future Dates
x3=[]
for i5 in range(len(future.index)):
    x3.append(str(future.index[i5]))
y3=[]
z3=[]
for i6 in range(len(future.index)):
    y3.append(str(x3[i6]).split(' '))
    z3.append(y3[i6][0])
invertedData = invertedData.drop(z3)
tes = tes.drop(z3)

pred_test = invertedData.iloc[:,1:2].values
pred_tes=[]
for i11 in range (len(pred_test)):
    pred_tes.append(pred_test[i11][0])
pred_tes = np.array(shipped_predicted)
actuals = np.array(tes)

forecast_accuracy(pred_tes, actuals)

plt.plot(pred_tes)
plt.plot(actuals, color = 'red')
plt.show()

#confusion_matrix
f1_actuals = flag_column(actuals, demand_predicted)
f1_forecast = flag_column(pred_tes, demand_predicted)
cm = confusion_matrix(f1_actuals,f1_forecast)



'''
plt.plot(train['Shipped'])
plt.plot(test['Shipped'])
plt.plot(shipped_pred_inv, linewidth = 2, alpha = 0.9)
plt.show()



def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)
cointegration_test(df)'''