# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:18:50 2019

@author: hmusugu
"""

#Code to generate the 0's and 1's that make up the output


import pandas as pd
import numpy as np

#df1 consists all unique codes of the automobile configuratios
df1 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\New data\\Uniques_codes_IDR_18-19.xlsx",sheet_name = 'Uniques_codes_IDR_18-19')

#Run the corresponding lines based on fleet/retail and proposed/current

#For Fleet and current use:
df2 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\All_Config_String_US_Fleet_current.xlsx", sheet_name = "Sheet1")

df3 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\Config_Fleet_Current.xlsx", sheet_name = "Sheet1")

#For Fleet and Proposed use:
df2 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\All_Config_String_US_Fleet.xlsx", sheet_name = "Sheet1")

df3 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\Config_Fleet_Current.xlsx", sheet_name = "Sheet1")

#For Fleet and current use:
df2 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\All_Config_String_US_Fleet.xlsx", sheet_name = "Sheet1")

df3 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\Config_Fleet_Current.xlsx", sheet_name = "Sheet1")

#For Fleet and current use:
df2 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\All_Config_String_US_Fleet.xlsx", sheet_name = "Sheet1")

df3 = pd.read_excel("C:\\Users\\hmusugu\\Desktop\\FCA\\Input Data Latest\\Fleet\\Config_Fleet_Current.xlsx", sheet_name = "Sheet1")


body_str =df3.iloc[:,5:6].values.tolist()
body_str1=[]
[body_str1.append(j) for sub in body_str for j in sub] 

cpos_str =df3.iloc[:,6:7].values.tolist()
cpos_str1=[]
[cpos_str1.append(j) for sub in cpos_str for j in sub] 

sub_str =df3.iloc[:,1:5].values.tolist()

#x--->y

x = df2.iloc[:,0:1].values
x = x.tolist()

x1=[]
[x1.append(j) for sub in x for j in sub]

y = df1.iloc[:,0:1].values
y = y.tolist()

y1=[]
[y1.append(j) for sub in y for j in sub]


final=[]
for i2 in range(len(x1)):
    tes = x1[i2]
    spli= tes.split(',')
    fin = []
    for i in y1:
        if (i in spli):
            fin.append(1)
        else:
            fin.append(0)
    final.append(fin)
    

final = np.array(final)


engine = ['2.0L I4 DOHC DI TURBO ENGINE W/ ESS','2.4L I4 ZERO EVAP M-AIR ENGINE W/ESS','3.2L V6 24V VVT ENGINE W/ESS']
engine_code = ['EC1','EDE','EHK']


trans = ['9-SPD 948TE FWD/AWD AUTO TRANS' , '9-SPD 948TE 4WD AUTO TRANS']
trans_code = ['DFH','DFJ']

bdy_cpos_features = ['LATITUDE',
'LATITUDE LUX',
'LATITUDE PLUS',
'LIMITED',
'OVERLAND',
'TRAILHAWK',
'FWD',
'AWD',
'4WD',
'EC1',
'EDE',
'EHK',
'DFH',
'DFJ']

sub_final = []
#a = ['KLJE74,2BD,*P7,-X9,']
for i3 in range(len(x1)):
    x_split= x1[i3].split(',')
    for i in range(len(body_str)):
        if (x_split[0]==body_str1[i] and x_split[1]==cpos_str1[i]):
            sub_final.append(sub_str[i]) 
    for i1 in range(len(engine)):
        if sub_final[i3][2] == '2.0L I4 DOHC DI TURBO ENGINE W/ ESS':
            sub_final[i3][2]='EC1'
        elif sub_final[i3][2] == '2.4L I4 ZERO EVAP M-AIR ENGINE W/ESS' :
            sub_final[i3][2]='EDE'
        elif sub_final[i3][2] == '3.2L V6 24V VVT ENGINE W/ESS' :
            sub_final[i3][2]='EHK'
    for i2 in range(len(trans)):
        if sub_final[i3][3] == trans[i2]:
            sub_final[i3][3]=trans_code[i2]
            print("y")



final_bdy_cpos=[]
for i2 in range(len(sub_final)):
    bdy_cpos = sub_final[i2]
    fin2 = []
    for i in bdy_cpos_features:
        if (i in bdy_cpos):
            fin2.append(1)
        else:
            fin2.append(0)
    final_bdy_cpos.append(fin2)

final_bdy_cpos = np.array(final_bdy_cpos)

output_matrix = np.concatenate((final_bdy_cpos, final), axis = 1) 





'''
a=['KLTE74,2BD,*P7,-X9,JPM,NHL,NHS,SCC,TBC,XBM']
sub_final = []
ind = []
a_split= a[0].split(',')
for i in range(len(body_str)):
    if (a_split[0]==body_str1[i] and a_split[1]==cpos_str1[i]):
        sub_final.append(sub_str[i]) 
        ind.append(i)
        
for i1 in range(len(engine)):
    if sub_final[0][2] == engine[i1]:
        sub_final[0][2]=engine_code[i1]
        print('y')
for i2 in range(len(trans)):
    if sub_final[0][3] == trans[i2]:
        sub_final[0][3]=trans_code[i2]
        print("y")
'''


















