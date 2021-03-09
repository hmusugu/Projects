# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:07:55 2020

@author: hmusugu
"""

#SARIMA Modelling Univariate


from datetime import datetime
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
#from datetime import datetime

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

def import_files():
    train = pd.read_csv('C:/Users/hmusugu/Desktop/Initiatives/Market Intelligence/DJT.csv',parse_dates=['Date'],index_col='Date' )#('../input/train.csv' ,parse_dates=['date'],index_col='date')
    train = train.dropna()
    test = pd.read_csv('C:/Users/hmusugu/Desktop/Initiatives/Market Intelligence/DJT_test.csv',parse_dates=['Date'],index_col='Date')#('../input/test.csv', parse_dates=['date'],index_col='date')
    test = test.dropna()
    #future = pd.read_csv('C:/Users/hmusugu/Desktop/Supplier Agility/10/Forecast.csv',parse_dates=['date'],index_col='date')#('../input/test.csv', parse_dates=['date'],index_col='date')
    #future = future.dropna()
    df = pd.concat([train,test],sort=True)
    #df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.asfreq(freq = 'B')
    df = df.groupby(df.index).mean()
    return df,train,test
#df =df.replace([np.inf, -np.inf], np.nan)
#df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Shipping"], how="all")



def train_test(index1,index2):
    tra = df.iloc[:int(0.8 * len(df)),index1:index2]
    tes = df.iloc[int(0.8 * len(df)):,index1:index2]
    ful = df.iloc[:,index1:index2]
    return tra,tes,ful


def model(data):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    AIC = []
    PDQ=[]
    S_PDQ = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(endog = data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
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
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, 12),enforce_stationarity=False,enforce_invertibility=False)
    res = mod.fit()
    print(res.summary().tables[1])
    return res

#Main

##Individual column  Univariate Analysis

df, train, test= import_files()
tra, tes, ful = train_test(2,3)

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

tr_start,tr_end = z1[0],z1[len(z1)-1]  
te_start,te_end = z2[0],z2[len(z2)-1]
fu_start,fu_end = z1[0],z2[len(z2)-1]

tra = df['High'][tr_start:tr_end].dropna()
tes = df['High'][te_start:te_end].dropna()
fu =  df['High'][fu_start:fu_end].dropna()

res_high = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_high.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_high.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting high
forecast_high = res_high.forecast(14)
plt.plot(forecast_high)

#Forecasting Low
tra, tes, ful = train_test(3,4)

tra = df['Low'][tr_start:tr_end].dropna()
tes = df['Low'][te_start:te_end].dropna()
fu =  df['Low'][fu_start:fu_end].dropna()

res_high = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_high.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_high.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting low
forecast_low = res_high.forecast(14)
plt.plot(forecast_low)

#Open
tra, tes, ful = train_test(4,5)
tra = df['Open'][tr_start:tr_end].dropna()
tes = df['Open'][te_start:te_end].dropna()
fu =  df['Open'][fu_start:fu_end].dropna()

res_open = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_high.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_open.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting
forecast_open = res_open.forecast(14)
plt.plot(forecast_open)


#Volume
tra, tes, ful = train_test(5,6)

tra = df['Volume'][tr_start:tr_end].dropna()
tes = df['Volume'][te_start:te_end].dropna()
fu =  df['Volume'][fu_start:fu_end].dropna()

res_vol = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_vol.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_vol.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting
forecast_vol = res_vol.forecast(14)
plt.plot(forecast_vol)


#Multi-Variate Analysis

tra, tes, ful = train_test(1,2)
tra = df['Close'][tr_start:tr_end].dropna()
tes = df['Close'][te_start:te_end].dropna()
fu =  df['Close'][fu_start:fu_end].dropna()

exog_train = df[['High','Low','Open','Volume']][tr_start:tr_end].dropna()
exog_test  = df[['High','Low','Open','Volume']][te_start:te_end].dropna()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


AIC = []
PDQ=[]
S_PDQ = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(endog = tra,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=True,
                                            enforce_invertibility=True, exog = exog_train)
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
                                seasonal_order=(P, D, Q, 12),enforce_stationarity=True,enforce_invertibility=True,exog = exog_train)
res = mod.fit()
print(res.summary().tables[1])



exog_fu = pd.concat([forecast_high,forecast_low,forecast_open,forecast_vol], axis = 1,column_name = ['High','Low','Open','Volume'])
index=pd.date_range(
        start=fu_end,
        periods= 15,
        freq='B')
index = index[1:]
exog_fu.columns = ['High','Low','Open','Volume']
exog_fu.index = index

pred = res.predict(start=res.nobs, end=res.nobs + (len(exog_fu)-1), exog = exog_fu)

plt.plot(pred)

pred.index = index


pred.to_csv("predictions_1.csv")









