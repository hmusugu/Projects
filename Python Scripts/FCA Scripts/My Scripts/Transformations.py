# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:14:43 2019

@author: hmusugu
"""

import pandas as pd
import numpy as np
from functools import reduce
from itertools import chain

df = pd.read_csv("IDR Data 18 19MY KL Only CLEANED Extract (local copy).csv")
tes = df.Configuration.unique().tolist()

z=[]
for i in df.columns:
    z.append(i)
    
    
configs = df.iloc[:,46:47].values.tolist()

confi = np.array(configs)

confi = np.unique(confi)

configs = confi.tolist()

tot_list = reduce(lambda x,y:x+y,configs)

for i in configs:
    configs[i] = configs[i].replace("'","")

tot_configs = []
[tot_configs.append(j) for sub in configs for j in sub]

fin_configs =[]
[fin_configs.append(j1) for subs in tot_configs for j1 in subs]

x = list(chain.from_iterable(configs))

x = ['asd,asd,sdf']
c = 0
for i in range(int((len(x[0])+1)/4)):
    print(x[0][c:c+3])
    c = c+4
    
tot_config=[]
for i in range(len(configs)):
    for j in range(len(configs[i])):
        tot_config.append(configs[i][j])

c = 0
con = []
for j in range(len(configs)):
    for i in range(int((len(configs[j][0]))/4)):
        con.append(configs[j][0][c:c+3])
        c = c+4        

tot = []
x = ''
for i in range(len(tes)):
    x= x+tes[i]

x[0:3]
x[4:7]

con = []
c=0
for i in range(1176982):
    con.append(x[c:c+3])
    c = c+4

df1 = pd.DataFrame(con,columns = ['Config']) 

df1.to_csv("configs.csv",index = False)


df1 = df1.dropna()

unique_configs = df1.Config.unique().tolist()


for i in con:
    if i == ' ':
        print("T")
        
        
        
df3 = pd.read_csv("Old_Sales_codes.csv")      

old_sc = df3.iloc[:,0:1].values.tolist()
osc1=[]
[osc1.append(j) for sub in old_sc for j in sub]
old_sc_1 = df3.iloc[:,1:2].values.tolist()
osc2=[]
[osc2.append(k) for subs in old_sc_1 for k in subs]


df3


old_sc_2=[]
old_sc_3=[]
for i in range(len(old_sc)):
        old_sc_2.append(osc1[i].strip())
        old_sc_3.append(osc2[i].strip())
        
    

final = []
for p in range(len(unique_configs)):
    for q in range(len(old_sc_3)):
        if (unique_configs[p]==old_sc_3[q]):
            final.append(old_sc_2[q]+'('+old_sc_3[q]+')')
        

df1 = pd.read_csv("2019_Sales_Codes.csv")
df2 = pd.read_csv("2021_Sales_Codes.csv")


x = df1.iloc[:,1:2].values.tolist()
x1=[]
[x1.append(j) for sub in x for j in sub]

y=df2.iloc[:,1:2].values.tolist()
x2=[]
[x2.append(j1) for subs in y for j1 in subs]

d_19 = df1.iloc[:,0:1].values.tolist()
a1=[]
[a1.append(j) for sub in d_19 for j in sub]


d_21 = df2.iloc[:,0:1].values.tolist()
a2=[]
[a2.append(j) for sub in d_21 for j in sub]

final = []
com =[]
for p in range(len(x1)):
    for q in range(len(x2)):
        if (x1[p]==x2[q]):
            final.append(a2[q]+'('+x2[q]+')')
            com.append(x2[q])
            
uncom_19=[]
for i in range(len(com)):
    for j in range(len(x2)):
        if (x1[i]==com[i]):
            continue
        else:
            print('i')
            uncom_19.append(a1[i]+ '(' + x1[i] +')')
            
def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

z=Diff(x2,com)

df5 = pd.read_csv("IDR_FormattedConfig_Aman.csv")
            


        


































