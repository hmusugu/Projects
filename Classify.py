# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:56:29 2021

@author: guna_
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, f1_score as f1   

data= pd.read_excel("Data.xlsx",sheet_name = 'Sheet1')

imp_col = data.drop(['Application: Application ID','Full App Submitted Date','Fraud Score V1',
                     'Giact BA Account Code',	'Giact BA Account Description',	'Giact BA Account Added Date',	
                     'Giact BA Account Last Updated Date',	'Giact BA Customer Code',
                     'Giact BA Customer Description','utm_source','Physical State',	'Physical City',
                     'Physical Zip Code','Bank Name'],axis = 1)


df = imp_col


#Imputing required values into fields with missing values


df.isnull().sum()
df['NS Phone Score']= df['NS Phone Score'].fillna(int(df['NS Phone Score'].mean()))
df['NS Email Found in Repository']= df['NS Email Found in Repository'].fillna(0)
df['NS Email Score']= df['NS Email Score'].fillna(int(df['NS Email Score'].mean()))
df['Co-App FICO']= df['Co-App FICO'].fillna(int(df['Co-App FICO'].median()))
df['Inquires Last 6 Months']= df['Inquires Last 6 Months'].fillna(int(df['Inquires Last 6 Months'].mean()))
df['Total Credit Card Cash Adv']= df['Total Credit Card Cash Adv'].fillna(int(df['Total Credit Card Cash Adv'].mode()))
df['Retirement Balances (401k/non-pension)']= df['Retirement Balances (401k/non-pension)'].fillna(int(df['Retirement Balances (401k/non-pension)'].mean()))
df['Highest level of education']= df['Highest level of education'].fillna('NA')
df['Total Inquiries']= df['Total Inquiries'].fillna(int(df['Total Inquiries'].mode()))


#Label and One-hot Encoding for categorical variable
le = LabelEncoder()
one = OneHotEncoder(handle_unknown = 'ignore')
#Encoding labels in 'CR SSN Match (Any)' field to numbers 
df['CR SSN Match (Any)'] = le.fit_transform(df['CR SSN Match (Any)'])
one_df = pd.DataFrame(one.fit_transform(df[['CR SSN Match (Any)']]).toarray())
one_df.columns = ['CR SSN Match (Any)1','CR SSN Match (Any)2','CR SSN Match (Any)3']
one_df = one_df.drop(['CR SSN Match (Any)1'],axis = 1)
df = df.join(one_df)

#Encoding labels in 'CR DOB Match (Any)' field to numbers 
df['CR DOB Match (Any)'] = le.fit_transform(df['CR DOB Match (Any)'])
one_df = pd.DataFrame(one.fit_transform(df[['CR DOB Match (Any)']]).toarray())
one_df.columns = ['CR DOB Match (Any)1','CR DOB Match (Any)2','CR DOB Match (Any)3']
one_df = one_df.drop(['CR DOB Match (Any)1'],axis = 1)
df = df.join(one_df)

#Encoding labels in 'Highest level of education' field to numbers 
df['Highest level of education'] = le.fit_transform(df['Highest level of education'])
one_df = pd.DataFrame(one.fit_transform(df[['Highest level of education']]).toarray())
one_df.columns = ['Highest level of education1','Highest level of education2',
                  'Highest level of education3','Highest level of education4',
                  'Highest level of education5','Highest level of education6']
one_df = one_df.drop(['Highest level of education1'],axis = 1)
df = df.join(one_df)

#Dropping the original fields that were one hot encoded
df = df.drop(['CR SSN Match (Any)','CR DOB Match (Any)','Highest level of education'], axis = 1)

Y = df.iloc[:,0:1].values
Y=Y.reshape(23802,)
X = df.iloc[:,1:].values

#Feature selection using ExtraTreeClassifier
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, Y)
print(model.feature_importances_*100)
feature_select_df = pd.DataFrame()
feature_select_df['Field'] = df.columns[1:]
feature_select_df['Feature Importance %'] = model.feature_importances_*100
#Expt1 : Selecting all fields which have importance more than 1% = There exist 29 such fields
final_df= df.drop(['Co-App FICO','# Linked Applications in Past 30 Days',
                   '# Linked Applications in Past 90 Days','Total Credit Card Cash Adv'
                   ,'ALL2700 (# Derog)','CR SSN Match (Any)2','CR SSN Match (Any)3',
                   'CR DOB Match (Any)2','CR DOB Match (Any)3',
                   'Highest level of education3','Highest level of education4','Highest level of education5',
                   'Highest level of education6'],axis = 1)
                   
#Train-test Split:
x=final_df.iloc[:,1:]  # Features
y=final_df.iloc[:,0:1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)    


#Classification algorithms:
#1. RandomForest
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)   

#Model Evaluation 
cm = confusion_matrix(y_test,y_pred)
f1_score = f1(y_test,y_pred,average = None)

#2. XgBoost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
f1_score = f1(y_test,y_pred,average = None)






