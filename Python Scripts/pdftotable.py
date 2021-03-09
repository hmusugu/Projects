# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:17:36 2020

@author: hmusugu
"""
import tabula
import pandas as pd

df1 = tabula.read_pdf('ethylene-prices.pdf', pages = 1, lattice = True)
df2 = tabula.read_pdf('ethylene-prices.pdf', pages = 2, lattice = True)
df3 = tabula.read_pdf('ethylene-prices.pdf', pages = 3, lattice = True)
df4 = tabula.read_pdf('ethylene-prices.pdf', pages = 4, lattice = True)
df5 = tabula.read_pdf('ethylene-prices.pdf', pages = 5, lattice = True)
df6 = tabula.read_pdf('ethylene-prices.pdf', pages = 6, lattice = True)

list1 = (df1[0],df2[0],df3[0],df4[0],df5[0],df6[0])

final = pd.DataFrame()
final = pd.concat(list1)