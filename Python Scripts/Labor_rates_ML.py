## -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:57:17 2020

@author: hmusugu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  sklearn
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.linear_model import LinearRegression as lin
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score as acc
from sklearn.ensemble import RandomForestRegressor as RR
from sklearn.linear_model import ElasticNet as EN
from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 



df = pd.read_excel("sample.xlsx")

df_f = df


#DATA MANIPULATION
df_f = df[['ST Base hourly wage rate $','Raw Labor Classification','State','city','State Index','City Index']]

df_f.isnull().sum()
df_f = df_f.dropna()

df_f.describe()

#df_f = df_f.sample(frac = 1 )

#df_f['Latitude'] = df_f['Latitude'].dropna()

df_f['ST Base hourly wage rate $'] = df_f['ST Base hourly wage rate $'].fillna(df_f['ST Base hourly wage rate $'].mean())
df_f = df_f.dropna()

#df_f.to_csv("Labor_Rates_v2.csv")


#df_f = pd.read_csv("Labor_Rates_v2.csv")

df_f['City Index']= df_f['City Index'].fillna(df_f['City Index'].mean())
#df_f['Zip']= df_f['Zip'].fillna(df_f['Zip'].mean())
#df_f['ST Billable Hourly Rate']= df_f['ST Billable Hourly Rate'].fillna(df_f['ST Billable Hourly Rate'].mean())
df_f['State Index']= df_f['State Index'].fillna(df_f['State Index'].mean())
#df_f['Zip_new'] = df['Zip'].str[0:5]

df_final = df_f

#df['Raw Labor Classification'].head(10)

#Label and one hot encoding the Roles



le = LabelEncoder()
one = OneHotEncoder(handle_unknown = 'ignore')

df_f['Raw Labor Classification'] = le.fit_transform(df_f['Raw Labor Classification'])
one_df = pd.DataFrame(one.fit_transform(df_f[['Raw Labor Classification']]).toarray())
df_f = df_f.join(one_df)


df_f['city'] = le.fit_transform(df_f['city'])
two_df = pd.DataFrame(one.fit_transform(df_f[['city']]).toarray())
df_f = df_f.join(two_df,rsuffix = 'city_')

df_f['State'] = le.fit_transform(df_f['State'])
three_df = pd.DataFrame(one.fit_transform(df_f[['State']]).toarray())
df_f = df_f.join(three_df,rsuffix = '_state')

df_f = df_f.drop(['Raw Labor Classification','State','city'], axis = 1)

df_f = df_f.dropna()

y = df_f.iloc[:,0:1].values
x= df_f.iloc[:,1:].values    
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Linear Regression ( Very low Accuracy )
regressor = lin()
regressor.fit(x_train, y_train)
regressor.score(x_test,y_test)
cv_score = cross_val_score(regressor,x_train,y_train,cv = 10)
cv_score.mean()
y_pred = regressor.predict(x_test)
forecast_accuracy(y_pred,y_test)  

#Random Forest Regressor (good cv, good mape)
reg =RR(n_estimators = 100)
reg.fit(x_train, y_train)
cv_score = cross_val_score(reg,x_train,y_train,cv = 10)
cv_score.mean()
y_pred = reg.predict(x_test)
reg.score(x_test,y_test)
forecast_accuracy(y_pred,y_test) 

#Elastic Net regressor (very bad cv, very good mape)
regr = EN(random_state=0)
regr.fit(x_train, y_train)
cv_score = cross_val_score(regr,x_train,y_train,cv = 10)
cv_score.mean()
y_pred = regr.predict(x_test)
regr.score(x_test,y_test)
forecast_accuracy(y_pred,y_test) 

#K-neighbors Regressor ( bad cv, good mape)
regr2 = knr(10)
regr2.fit(x_train, y_train)
cv_score = cross_val_score(regr2,x_train,y_train,cv = 10)
cv_score.mean()
y_pred = regr2.predict(x_test)
regr2.score(x_test,y_test)
forecast_accuracy(y_pred,y_test) 

#SVR (bad cv, very good mape)
regr3 = SVR()
regr3.fit(x_train, y_train)
cv_score = cross_val_score(regr3,x_train,y_train,cv = 10)
print(cv_score.mean())
y_pred = regr3.predict(x_test)
regr3.score(x_test,y_test)
forecast_accuracy(y_pred,y_test) 



def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
                # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse})


forecast_accuracy(y_pred,y_test)   

accuracy = regressor.score(x_test,y_test)


plt.plot(df_f['ST Base hourly wage rate $'], label = 'Actuals',linewidth = 1.4)
plt.plot(regressor.fittedvalues, color='red',linewidth = 0.6, label = 'predictions')
plt.legend()
plt.show()

plt.plot(y_test,linewidth = 0.2, color = 'blue')
plt.plot(y_pred,linewidth = 0.5, color = 'red')



out_df = df.iloc[119601:,]
out_df['Predicted ST Billable Rate'] = y_pred.astype
out_df["ACtual ST Billable Rates"] = y_test.astype(float)

out_df.to_csv("Final Predictions.csv")

new = pd.DataFrame(x_test)

new.to_csv("See_v2.csv")



plt.plot(y_test)
plt.plot(y_pred,color = 'red')





