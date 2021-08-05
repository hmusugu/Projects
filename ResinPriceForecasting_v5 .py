# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:07:55 2020

@author: hmusugu
"""

#SARIMA Modelling Univariate


from datetime import datetime
from openpyxl import load_workbook
from dateutil.relativedelta import relativedelta
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
import sqlalchemy as db
from sqlalchemy import create_engine
import os
from sqlalchemy import create_engine, MetaData
import StationarityTests
import StationarizeSeries
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kstest, shapiro
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#Importing input files
def import_files():
    df = pd.read_excel("Styrene_PS_Input_Data_20200914 v2.xlsx", sheet_name="endog", index_col = "date")
    exog_var = pd.read_excel("Styrene_PS_Input_Data_20200914 v2.xlsx",sheet_name="exog",index_col = "date")
    #full_data1 = exog_var.iloc[:52, 5:8]
    #df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #df = df.asfreq(freq = 'B')
    #df = df.groupby(df.index).mean()
    return df, exog_var

'''
#Needs to be built-in
def train_test(index1,index2):
    #tra = df.iloc[0:31,index1:index2]
    #tes = df.iloc[30:36,index1:index2]
    ful = df.iloc[0:40,index1:index2]
    return  ful
'''
#Stationarizer func to stationarize non-stationary exog variables
def Stationarizer(data1):
    data1 = data1.dropna()
    sTest = StationarityTests.StationarityTests()
    results2 = StationarityTests.RunTests(sTest, data1)
    
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
                df_Stnry[k[0]] = StationarizeSeries.differenceSeries(k[0], data1, sTest)
                nonStationaryData = True
            else:
                df_Stnry[k[0]] = data1[k[0]]
    return df_Stnry, nonStationaryData

#Func for creating lag columns
def create_lagged_columns(df_Stnry,exog_name):
    # load dataset
    series = df_Stnry[exog_name]
    dataframe = DataFrame()
    for i in range(12,0,-1):
    	dataframe['t-'+str(i)] = series.shift(i)
    dataframe['t'] = series.values
    dataframe = dataframe[12:]
    return dataframe

#Correlation Tests (Pearson and Spearman)
def corr_tests(lag_columns,endog_var1):
    #Pearson
    p_coef = []
    for i1 in lag_columns.columns:
        corr, _ = pearsonr(endog_var1, lag_columns[i1])
        p_coef.append(corr)
    max_p_coef = max(p_coef)
    for i2 in range(len(p_coef)):
        if p_coef[i2] == max_p_coef:
            corr_column1 = lag_columns.columns[i2]
            
    #Spearman
    s_coef = []
    for i3 in lag_columns.columns:
        corr, _ = spearmanr(endog_var1, lag_columns[i3])
        s_coef.append(corr)
    max_s_coef = max(s_coef)
    for i4 in range(len(s_coef)):
        if s_coef[i4] == max_s_coef:
            corr_column2 = lag_columns.columns[i4]
    #Which column to take
    if corr_column1 == corr_column2:
        corr_column = corr_column1
        return corr_column
    else:
        if shapiro(lag_columns[corr_column1])[1]>0.05:
            return (corr_column1)
        else: return (corr_column2)



#Feature Importance/ Feature Selection
def Random_Forest_regressor(df_ex_en):
    array = df_ex_en.values
    X = array[:,1:]
    y = array[:,0]
    
    model = RandomForestRegressor(n_estimators=500, random_state=1)
    model.fit(X, y)
    # show importance scores to choose components that explain 80% of variance 
    # plot importance scores
    names = df_ex_en.columns.values[1:]
    ticks = [i for i in range(len(names))]
    pyplot.bar(ticks, model.feature_importances_)
    pyplot.xticks(ticks, names)
    pyplot.show()
    
    imp_col_dict = dict()
    for k3 in range(len(names)):
        imp_col_dict[names[k3]] = model.feature_importances_[k3]
    sort_orders = sorted(imp_col_dict.items(), key=lambda x: x[1], reverse=True)
    imp_col_name = []
    var_val = []
    for i in sort_orders:
        imp_col_name.append(i[0])
        var_val.append(i[1])
    
    imp_col_final = []
    c=0
    for k4 in range(len(var_val)):
        c = c + var_val[k4]
        imp_col_final.append(imp_col_name[k4])
        if c >=0.85:
            break
        else:
            continue
    
    return imp_col_final, var_val, imp_col_name

        
    
#Func for uni-var training
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


#Func for multivariate training:
def mult_var_model(endog_data, exog_data):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    AIC = []
    PDQ=[]
    S_PDQ = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(endog = endog_data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True, exog = exog_data)
                results = mod.fit()
                AIC.append(results.aic)
                PDQ.append(param)
                S_PDQ.append(param_seasonal)
                print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    #Choosing model with min AIC
    min_AIC = min(AIC)
    for a1 in range(len(AIC)):
        if (AIC[a1]==min_AIC):
            min_index = a1
    
    opt_PDQ = PDQ[min_index]
    opt_S_PDQ = S_PDQ[min_index]
    p,d,q=opt_PDQ[0],opt_PDQ[1],opt_PDQ[2]
    P,D,Q=opt_S_PDQ[0],opt_S_PDQ[1],opt_S_PDQ[2]
    
    mod = sm.tsa.statespace.SARIMAX(endog_data,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 12),enforce_stationarity=True,enforce_invertibility=True,exog = exog_data)
    res = mod.fit()
    print(res.summary().tables[1])
    return res

#Function to invert first order differencing
def inverter(diff_val,orig_data):
    inv = []
    c = orig_data.iloc[-len(diff_val)-1]
    for k6 in range(len(diff_val)):
        c = c + diff_val[k6]
        inv.append(c)
    return inv

#Performance metrics
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


def normalize(series):
    series = pd.DataFrame(series)
    values = series.values
    #values = values.reshape((len(values),np.shape(series)[1]))
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    inversed = scaler.inverse_transform(normalized)
    return inversed, normalized, scaler
        

    


#Main
    
#Import data
df_endog, df_exog= import_files()


#Need to Make this generic for all material types***********************************
df_type_endog = dict(tuple(df_endog.groupby('material')))
df_type_exog = dict(tuple(df_exog.groupby('material')))
typ_list =tuple(df_endog.groupby('material'))


endog_type_dict = {}
for number in range(len(df_type_endog)):
    endog_type_dict["endog_typ%s" %number] = df_type_endog[typ_list[number][0]]

exog_type_dict = {}
for number in range(len(df_type_exog)):
    exog_type_dict["exog_typ%s" %number] =  df_type_exog[typ_list[number][0]]


out_df_list = []
out_df_list1 = []
perf_list = []
for q1 in range(len(endog_type_dict)):
    for q2 in range(len(exog_type_dict)):
        if (list(endog_type_dict.keys())[0][-1] == list(exog_type_dict.keys())[0][-1]):
            endog_typ = endog_type_dict[list(endog_type_dict.keys())[0]]
            exog_typ = exog_type_dict[list(exog_type_dict.keys())[0]]
            exog_typ= exog_typ.dropna(axis=1, how='all')
            exog_typ = exog_typ.drop(['material', 'year','month'], axis = 1)
            exog_typ = exog_typ.dropna()
            exog_backup = exog_typ
            
            endog_var = endog_typ['cost_per_unit'].dropna()
            # Normalize Endog 
            endog_orig, endog_norm, sca_endog = normalize(endog_var)
            
            endog_var = pd.DataFrame(endog_var)
            endog_var['cost_per_unit'] = endog_norm
            endog_var = endog_var.T.iloc[0]
            
            # Normalize Exog 
            exog_orig, exog_norm, sca_exog = normalize(exog_typ)
            
            #exog_var = pd.DataFrame(exog_var)
            
            exog_typ1 = pd.DataFrame(exog_norm)
            exog_typ1.index = exog_typ.index
            exog_typ1.columns = exog_typ.columns
            exog_typ = exog_typ1
            
            #Stationarizing Endog
            endog_var_df = endog_var.to_frame()
            endog_stationarized,nonStationaryData = Stationarizer(endog_var_df)
            endog_final = endog_stationarized['cost_per_unit']
            
            #Stationarize exog
            df_stationarized, nonStationaryData = Stationarizer(exog_typ)
            df_stationarized = df_stationarized.dropna()

            #Taking optimal lag columns using Pearson and Spearmen Co-eff. 
            selected_lag_col = []
            exog_final1 = pd.DataFrame()
            for k1 in df_stationarized.columns:
                print (k1)
                df_lagged_columns = create_lagged_columns(df_stationarized, k1)
                df_lagged_columns = df_lagged_columns[:'2019-04-01']
                #selected_lag_col.append(k1+'_'+corr_tests(df_lagged_columns,endog_final))
                lag_name = corr_tests(df_lagged_columns,endog_final)
                exog_final1[k1+'_'+corr_tests(df_lagged_columns,endog_final)] = df_lagged_columns[lag_name]
                
            df_ex_en1 = pd.DataFrame()
            df_ex_en1['cost_per_unit'] = endog_final
            for k2 in exog_final1.columns:
                df_ex_en1[k2] = exog_final1[k2]   
            #df_typ2 = df_type['Styr[]ene']
                
            #Taking columns selected by Feature Selection 
            feature_selected_col, var_val, col_name = Random_Forest_regressor(df_ex_en1)
            exog_final2 = pd.DataFrame()
            for k5 in feature_selected_col:
                exog_final2[k5] = df_ex_en1[k5]
                
            #Storing color for selected columns:
            clean_col_name = []
            for m2 in range(len(col_name)):
                clean_col_name.append(col_name[m2][:-4])   
            color = []
            for m1 in col_name:
                if m1 in feature_selected_col:
                    color.append('green')
                else:color.append('red')
            
            #Dataframe for feature Selection Criterion
            fs = pd.DataFrame()
            fs['Index Type'] = clean_col_name
            fs['Variance'] = var_val
            fs['color'] = color
                    
            
            #Forecasting  exogs with optimal lags using univariate analysis
            res_dict = dict()
            for k7 in exog_final2.columns:
                res_dict['res_'+k7] = model(exog_final2[k7])
            
            pred_uni_exog =pd.DataFrame()
            for k8 in range(len(res_dict)):
                pred_uni_exog[list(res_dict.keys())[k8][4:]] = list(res_dict.values())[k8].forecast(14)
            
            #Creating exog_old to normalize it so that we can de-normalize the array with selected columns
            exog_old = pd.DataFrame()
            #original data with selected columns
            for s1 in range(len(exog_final2.columns)):
                exog_old[exog_final2.columns[s1][:len(exog_final2.columns[s1])-4]] = exog_backup[exog_final2.columns[s1][:len(exog_final2.columns[s1])-4]]
            
            #normalizing exog_old
            exog_orig_old, exog_norm_old, sca_exog_old = normalize(exog_old)
            
            #Adding column names and index to exog_old
            exog_old1 = pd.DataFrame(exog_norm_old)
            exog_old1.index = exog_old.index
            exog_old1.columns = exog_old.columns
            exog_old = exog_old1
            
            
            #Forecasting orginal exogs using univariate analysis (Needs to refined)************************************
            res_dict2 = dict()
            for k9 in exog_old.columns:
                res_dict2['res_'+k9] = model(exog_old[k9])
                #exog forecasts
            pred_uni_exog2 =pd.DataFrame()
            for k10 in range(len(res_dict)):
                pred_uni_exog2[list(res_dict2.keys())[k10][4:]] = list(res_dict2.values())[k10].forecast(14)
            
            
            #de-normalizing
            pred_uni_exog_norm = sca_exog_old.inverse_transform(pred_uni_exog2.values)
            pred_uni_exog_norm= pd.DataFrame(pred_uni_exog_norm)
            pred_uni_exog_norm.index = pred_uni_exog2.index
            pred_uni_exog_norm.columns = pred_uni_exog2.columns
            pred_uni_exog2 = pred_uni_exog_norm
                
            #x = sca_exog.inverse_transform(pred_uni_exog2.values)
            
            
            #exog in-samp predictions
            pred_uni_exog_insamp =pd.DataFrame()
            for k10 in range(len(res_dict)):
                pred_uni_exog_insamp[list(res_dict2.keys())[k10][4:]] = list(res_dict2.values())[k10].predict(start=pd.to_datetime('2015-01-01'), dynamic=False) #Starting date of exog_tup
                # Filling 1st values with orig values
            for k11 in exog_old.columns:
                pred_uni_exog_insamp[k11][0] = exog_old[k11][0]
                
            #de-normalizing
            pred_uni_exog_insamp_norm = sca_exog_old.inverse_transform(pred_uni_exog_insamp.values)
            pred_uni_exog_insamp_norm= pd.DataFrame(pred_uni_exog_insamp_norm)
            pred_uni_exog_insamp_norm.index = pred_uni_exog_insamp.index
            pred_uni_exog_insamp_norm.columns = pred_uni_exog_insamp.columns
            pred_uni_exog_insamp = pred_uni_exog_insamp_norm
            
    
            exog_typ2 = pd.DataFrame(exog_orig_old)
            exog_typ2.index = exog_old.index
            exog_typ2.columns = exog_old.columns
            exog_typ = exog_typ2
            
            
            out_df_list2 = []
            for k12 in exog_typ.columns:
                output_df2 = pd.DataFrame()
                output_df2['Index_price'] = exog_typ[k12]
                output_df2['Adjusted_Index_price'] = pred_uni_exog_insamp[k12]
    
                forecast_df2 = pd.DataFrame()
                forecast_df2['Forecasted_Index_price'] = pred_uni_exog2[k12]
                
                out2 = pd.concat([output_df2,forecast_df2])
                out2['Forecasted_Index_price'][output_df2.index[-1]] = output_df2['Adjusted_Index_price'][output_df2.index[-1]]
                out2['Material Type'] = endog_typ['material'][0]
                out2['Index Type'] = k12
                
                out_df_list2.append(out2)
            
            exog_price_forecasts = pd.DataFrame()
            exog_price_forecasts = pd.concat(out_df_list2)
            
            out_df_list1.append(exog_price_forecasts)
            
            #Split data train, test and future. Needs to be made generic******************************
            en_train = endog_final[:'2018-09-01']
            ex_train = exog_final2[:'2018-09-01']
            
            en_test =  endog_final['2018-10-01':]
            ex_test =  exog_final2['2018-10-01':]
            
            ex_fut =  pred_uni_exog
            
            #Train model on train data
            res_tra = mult_var_model(en_train, ex_train)
            
            #Out-of-sample predictions and Model Performance Evaluation (using test data)
            pred_test = res_tra.predict(start=res_tra.nobs, end=res_tra.nobs + (len(ex_test)-1), exog = ex_test)  
            

            
            pred_test_inv = inverter(pred_test, endog_var)
            en_test_inv = inverter(en_test, endog_var)
            
            pred_test_inv = np.array(pred_test_inv)
            en_test_inv = np.array(en_test_inv)
            
            
            #de-normalizing
            pred_test_inv = sca_endog.inverse_transform(pred_test_inv)
            en_test_inv = sca_endog.inverse_transform(en_test_inv)
                        
            
            perf_metrics=forecast_accuracy(pred_test_inv, en_test_inv)
            perf = pd.DataFrame(perf_metrics, index = range(0,1))
            perf['Material Type'] = endog_typ['material'][0]
            
            perf_list.append(perf)
            
            
            plt.plot(pred_test_inv)
            plt.plot(en_test_inv)
            
            #Train model on Full data
            res_ful = mult_var_model(endog_final, exog_final2)    
            
            #In-sample predictions:
            pred1 = res_ful.get_prediction(start=pd.to_datetime('2016-02-01'), dynamic=False)
            pred_in_samp = pred1.predicted_mean
            
            
            pred_in_samp_inv= []
            var_c1 = endog_var.iloc[-1]
            for k10 in range(len(pred_in_samp)):
                var_c1 = var_c1 + pred_in_samp[k10]
                pred_in_samp_inv.append(var_c1)
            
            
            pred_in_samp_inv = [endog_var[0]] + pred_in_samp_inv
            pred_in_samp_inv = sca_endog.inverse_transform(pred_in_samp_inv)
            #Out-of-sample predictions
            pred_fut = res_ful.predict(start=res_ful.nobs, end=res_ful.nobs + (len(ex_fut)-1), exog = ex_fut) 
            
            pred_fut_inv= []
            var_c = endog_var.iloc[-1]
            for k9 in range(len(pred_fut)):
                var_c = var_c + pred_fut[k9]
                pred_fut_inv.append(var_c)
                
            #denormalizing
            pred_fut_inv = sca_endog.inverse_transform(pred_fut_inv)
            
            plt.plot(pred_fut_inv)
            
            output_df = pd.DataFrame()
            output_df['Material Type']= endog_typ['material']
            #output_df.index = list(endog_typ.index) + fut_index[1:]
            #output_df['Commodity Adjusted Price'] = forecast_df['preds']
            output_df['Commodity Price'] = endog_typ['cost_per_unit']
            output_df['Adjusted Commodity Price'] = pd.Series(pred_in_samp_inv, index = output_df.index)
     
            #output_df['Commodity Adjusted Price']['2019-04-01':] = pred_fut_inv
            
            fut_index=[]
            for q3 in range(15):
                fut_index.append(endog_typ.index[-1]+ relativedelta(months=+q3))
            
            forecast_df = pd.DataFrame()
            forecast_df['Forecasted Commodity Price'] = pred_fut_inv
            forecast_df['Material Type'] = endog_typ['material'][0]
            forecast_df.index= fut_index[1:]
            
            out1 = pd.concat([output_df, forecast_df])
            out1['Forecasted Commodity Price'][endog_typ.index[-1]] = endog_typ['cost_per_unit'][endog_typ.index[-1]]
            out_df_list.append(out1)

path = r"C:\Users\hmusugu\Desktop\Molex\CogForecasting\MVA\Output_test.xlsx"
comm_price_forecasts = pd.DataFrame()
comm_price_forecasts = pd.concat(out_df_list)

index_price_forecasts = pd.DataFrame()
index_price_forecasts = pd.concat(out_df_list1)

perf_df = pd.DataFrame()
perf_df = pd.concat(perf_list)

#Are we using this?
#pred_uni_exog2.to_excel("ADAPT_Data_Source_20200817 v9.xlsx",sheet_name = endog_typ['material'][0])


book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book

comm_price_forecasts.to_excel(writer, sheet_name = 'commodity_forecast_data')
index_price_forecasts.to_excel(writer, sheet_name = 'index_forecast_data')
perf_df.to_excel(writer, sheet_name = 'perf_metrics')
fs.to_excel(writer, sheet_name = 'Feature Selection')
writer.save()
writer.close()
        

#Output Data ****************

'''

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


#tra, tes, ful = train_test(3,4)

tra = train['cost_per_unit'][tr_start:tr_end].dropna()
tes = test['cost_per_unit'][te_start:te_end].dropna()

#fu =  df['cost_per_unit'][fu_start:fu_end].dropna()

exog_train = train[['Crude Oil WTI Futures_Orig','Natural Gas Futures_Orig']][tr_start:tr_end].dropna()
exog_test  = test[['Crude Oil WTI Futures_Orig','Natural Gas Futures_Orig']][te_start:te_end].dropna()
exog_ful = full_data[['Crude Oil WTI Futures_Orig','Natural Gas Futures_Orig']][fu_start:fu_end].dropna()

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    AIC = []
    PDQ=[]
    S_PDQ = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(endog = full,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True, exog = exog_ful)
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

#def run_model(): Make sure to include start/end time stamps so that we know the run-time
mod = sm.tsa.statespace.SARIMAX(full,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 12),enforce_stationarity=True,enforce_invertibility=True,exog = exog_ful)
res = mod.fit()
print(res.summary().tables[1])
#pred = res.predict(start=res.nobs, end=res.nobs + (len(exog_test)-1), exog = exog_test)

pred1 = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
pred_in_samp = pred1.predicted_mean

full = ful['cost_per_unit'][fu_start:fu_end].dropna()

ful = train_test(3,4)

pred_out_samp = res.predict(start=res.nobs, end=res.nobs + (len(exog_future)-1), exog = exog_future)

#Visualizations

tra, tes, ful = train_test(3,4)

plt.plot(pred, color = "red",label = "Forecast")
plt.plot(tes, color ="blue", label = "Actuals")
plt.legend(loc='best')
plt.show()


plt.plot(pred_in_samp, color = "red",label = "Forecast")
plt.plot(tra, color ="blue", label = "Actuals")
plt.legend(loc='best')
plt.show()

plt.plot(ful, figsize=(3,4))

ax = train['Crude Oil WTI Futures_Orig'].plot(figsize=(14, 7))
#ful.plot(ax=ax, label='Resin Price Time Series', figsize=(14, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Price in $')
plt.show()


test_acc_metrics = forecast_accuracy(pred,tes)

exog_fu = pd.concat([forecast_high,forecast_low,forecast_open,forecast_vol], axis = 1,column_name = ['High','Low','Open','Volume'])
index=pd.date_range(
        start=fu_end,
        periods= 15,
        freq='B')
index = index[1:]
exog_fu.columns = ['High','Low','Open','Volume']
exog_fu.index = index

pred = res.predict(start=res.nobs, end=res.nobs + (len(exog_fu)-1), exog = exog_fu)

plt.plot(pred, color = "red")
plt.plot(tes, color ="blue")

pred.index = index


pred.to_csv("predictions_1.csv")


#Importing from Postgres db

#df =df.replace([np.inf, -np.inf], np.nan)
#df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Shipping"], how="all")
def import_db():
    engine = create_engine(r'postgresql://postgres:Bk10987654321@10.5.9.93/LOGISTICS_DB')
    df = pd.read_sql_table(table_name = 'DOW_JONES_TRANSPORT', schema = 'public', con=engine,parse_dates=['ENTRY_DATE'],index_col='ENTRY_DATE' )    
    df = df.sort_index(ascending = True, axis = 0)
    df = df.asfreq(freq = 'B')
    df=df.iloc[10000:,:]
    df = df.groupby(df.index).mean()
    df=df.dropna()
    train = df.iloc[:int(0.99728682 * len(df)),:]
    test = df.iloc[int(0.99728682 * len(df)):,:]
    return df, train, test


##Individual column  Univariate Analysis

df, train, test= import_db()
tra, tes, ful = train_test(1,2)

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

tra = df['HIGH'][tr_start:tr_end].dropna()
tes = df['HIGH'][te_start:te_end].dropna()
fu =  df['HIGH'][fu_start:fu_end].dropna()

res_high = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_high.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_high.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting high
forecast_high = res_high.forecast(15)
plt.plot(forecast_high)

#Forecasting Low
tra, tes, ful = train_test(2,3)

tra = df['LOW'][tr_start:tr_end].dropna()
tes = df['LOW'][te_start:te_end].dropna()
fu =  df['LOW'][fu_start:fu_end].dropna()

res_high = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_high.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_high.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting low
forecast_low = res_high.forecast(15)
plt.plot(forecast_low)

#Open
tra, tes, ful = train_test(0,1)
tra = df['OPEN'][tr_start:tr_end].dropna()
tes = df['OPEN'][te_start:te_end].dropna()
fu =  df['OPEN'][fu_start:fu_end].dropna()

res_open = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_high.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_open.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting
forecast_open = res_open.forecast(15)
plt.plot(forecast_open)


#Volume
tra, tes, ful = train_test(3,4)

tra = df['VOLUME'][tr_start:tr_end].dropna()
tes = df['VOLUME'][te_start:te_end].dropna()
fu =  df['VOLUME'][fu_start:fu_end].dropna()

res_vol = model(fu)

plt.plot(fu, label = 'Actuals',linewidth = 1.4)
plt.plot(res_vol.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

res_vol.plot_diagnostics(figsize=(15,8))
plt.show()
#Forecasting
forecast_vol = res_vol.forecast(15)
plt.plot(forecast_vol)
'''



'''
#Inverttra2 = pd.DataFrame(endog_var, columns=["cost_per_unit"])
price_pred = pd.DataFrame()
price_pred["cost_per_unit"] = en_test.to_list()
pred_test_inv = []
if nonStationaryData:
    invertedData = StationarizeSeries.invert_transformation(tra2, price_pred)
    pred_test_inv = invertedData['cost_per_unit_forecast'].to_list()
else:
    pd.DataFrame(shippedPrediction).to_csv("C:/Users/hmusugu/Desktop/Supplier Agility/invertedData.csv")

shipped_pred_inv = invertedData.iloc[:,1:2]

'''




# Normalize time series data
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
series = read_csv('daily-minimum-temperatures-in-me.csv', header=0, index_col=0)
print(series.head())
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print the first 5 rows
normalized = scaler.transform(values)
for i in range(5):
	print(normalized[i])
# inverse transform and print the first 5 rows
inversed = scaler.inverse_transform(normalized)
for i in range(5):
	print(inversed[i])
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
# Normalize time series data
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
series = exog_var['Natural_Gas_Orig']
print(series.head())
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print the first 5 rows
normalized = scaler.transform(values)

# inverse transform and print the first 5 rows
inversed = scaler.inverse_transform(normalized)
for i in range(5):
	print(inversed[i])


