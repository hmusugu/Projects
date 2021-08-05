# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:20:12 2020

@author: rvermani
"""

import random
random.seed(12345)

import os
import pandas as pd
import numpy as np
#import sklearn
import itertools
import warnings
   
import matplotlib.pyplot as plt
import statsmodels.api as sm

import math
import re
from datetime import datetime

warnings.filterwarnings("ignore")

os.getcwd()
os.chdir('C:\\Users\\rvermani\\Documents\\RV\\Projects\\Molex\\Project\\Cognitive Forecasting\\Pactiv')

##Read the input data 
input_data = pd.read_csv("Styrene_PS_Input_Data_20200818 v1.csv")
input_data.columns = input_data .columns.str.strip().str.replace('#', '').str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
Enter_material='Polystyrene HHC'
Enter_indep_var = ['date','crude_oil_price','natural_gas_futures']
output_file='Polystyrene_output.csv'

##Enter dates
tr_start,tr_end = '2016-01-01','2018-06-01'
te_start,te_end = '2018-07-01','2018-12-01'
fo_start,fo_end = '2019-01-01','2019-12-01'
input_data=input_data[(input_data['material']==Enter_material)].reset_index()

input_data2=input_data[["material","year","month","cost_per_unit"]]
input_data2=input_data2.assign(date=pd.to_datetime(input_data2[['year','month']].assign(day=1)))

##Assuming the missing product cost unit as 0 
input_data2['cost_per_unit'] = input_data2['cost_per_unit'].fillna(0)

##Making the forecasted dataset for endogenous variables
#Making a dataset to merge inorder to have all years and months available for all the below files
from  itertools import product
year=[2019]
month = list(range(5, 13))
ym = pd.DataFrame(list(product(year,month)))
ym.columns = ['year','month']
ym=ym.assign(date=pd.to_datetime(ym[['year','month']].assign(day=1)))
ym['material']=Enter_material
ym['cost_per_unit']=0.0
fc_data=ym[['material','date','cost_per_unit']]

plt.plot(input_data['date'],input_data['cost_per_unit'])


#Merging the input data with current target
data_hist=input_data2[["material","date","cost_per_unit"]]
data_hist["date"] = pd.to_datetime(data_hist["date"])
data_hist = pd.concat([data_hist,fc_data],axis=0)

##Make 12 Month lag of the indices
## Change here : subset the indices relevant for your model here
###Read the indices 
ext_data = pd.read_csv("Styrene_PS_Indices_Data_20200818 v1.csv")
ext_data.columns = ext_data.columns.str.strip().str.replace(',', '').str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')
#print(ext_data.columns)
ext_data=ext_data[(ext_data.material==Enter_material)].reset_index() 
ext_data=ext_data[Enter_indep_var]
for col in ext_data.iloc[:,1:].columns:
    ext_data.loc[:,col+"_"+str(1)] = ext_data[col].shift(1)
    ext_data.loc[:,col+"_"+str(2)] = ext_data[col].shift(2)
    ext_data.loc[:,col+"_"+str(3)] = ext_data[col].shift(3)
    ext_data.loc[:,col+"_"+str(4)] = ext_data[col].shift(4)
    ext_data.loc[:,col+"_"+str(5)] = ext_data[col].shift(5)
    ext_data.loc[:,col+"_"+str(6)] = ext_data[col].shift(6)
    ext_data.loc[:,col+"_"+str(7)] = ext_data[col].shift(7)
    ext_data.loc[:,col+"_"+str(8)] = ext_data[col].shift(8)
    ext_data.loc[:,col+"_"+str(9)] = ext_data[col].shift(9)
    ext_data.loc[:,col+"_"+str(10)] = ext_data[col].shift(10)
    ext_data.loc[:,col+"_"+str(11)] = ext_data[col].shift(11)
    ext_data.loc[:,col+"_"+str(12)] = ext_data[col].shift(12)


##Final Exogenous data
##Change the date here for your exog data
ext_data['date'] = pd.to_datetime(ext_data['date'])
data_exog=ext_data[ext_data.date>='2016/01/01']
data_exog=data_exog.set_index(data_exog['date'])
exog_fut = data_exog[fo_start:fo_end]
data_exog = data_exog[tr_start:te_end]

##Final Endogenous data
##change your date here for your endogenous data 
data_endog=data_hist[data_hist.date>='2016/01/01']
data_endog=data_endog.set_index(data_endog['date'])
DATA=data_endog

##change the test and train time period here
data_exog=sm.add_constant(data_exog)
exog_fut=sm.add_constant(exog_fut)   # not adding constant in the df
tra = DATA[tr_start:tr_end].dropna()
tes = DATA[te_start:te_end].dropna()
tra_tes = DATA[tr_start:te_end].dropna()
fut = DATA[fo_start:fo_end].dropna()
exog_train = data_exog[tr_start:tr_end].dropna()
exog_test = data_exog[te_start:te_end].dropna()
exog_train_test = data_exog[tr_start:te_end].dropna()
print(exog_train.shape)
print(exog_test.shape)
print(exog_fut.shape)
print(tra.shape)
print(tes.shape)
print(fut.shape)

##running the selected model on traintest
varlist=['crude_oil_price_1','natural_gas_futures_1']
mod_ch_tr_tes = sm.tsa.statespace.SARIMAX(tra_tes['cost_per_unit'], exog_train_test[varlist],
                                    order=(1,0,0),
                                    seasonal_order=(0,0,0,12))
result_fin_tr_tes = mod_ch_tr_tes.fit(disp=False)
res_tr_tes = mod_ch_tr_tes.filter(result_fin_tr_tes.params)

# ##Get prediction on train and test###
pred_tr_tes = res_tr_tes.get_prediction(start=tr_start,dynamic=False)
pred_tr_tes_res= pd.DataFrame(pred_tr_tes.predicted_mean)
tr_tes_ci=pred_tr_tes.conf_int()
fin_train_test=pd.DataFrame(pd.concat([pd.DataFrame(pred_tr_tes_res),pd.DataFrame(tr_tes_ci)],axis=1))
fin_train_test.columns=['pred','lower_pred','upper_pred']
fin_train_test=pd.concat([tra_tes,fin_train_test,exog_train_test[varlist]],axis=1)
# ##Getting the final MAPE###
MAPE_ALL=np.mean(abs((np.array(fin_train_test['cost_per_unit'][1:])-np.array(fin_train_test['pred'][1:])))/np.array(fin_train_test['cost_per_unit'][1:]))*100
print(MAPE_ALL)


# ##Get prediction on future
pred_fut = result_fin_tr_tes.get_prediction(start=fo_start,end=fo_end,exog=exog_fut[varlist], dynamic=True)
pred_fut1=pred_fut.predicted_mean
fin_fut=pd.DataFrame(pred_fut1)
fin_fut.columns=['pred_orig']






 
