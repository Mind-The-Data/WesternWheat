#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:40:25 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm



datapath = '../Processed/'
df23Summer = pd.read_csv(datapath + 'df23Summer.csv')
df23Summer.index=df23Summer['system:indexviejuno']


############
# Trimming the FAT
#############
df23Summer.drop(['B1','B2', 'B7', 'B5', 'cropland_mode', 'system:index'], axis=1)
df23Summer.drop(['countyears', 'eto', 'etr', 'system:indexviejuno.1'], axis=1)


#######
##Creating Time Lag Variables
#######

## VPD
######
## creating individual arrays for each month
zVPDjune = df23Summer.loc[df23Summer['month'] == 6, 'zVPD']
zVPDmay = df23Summer.loc[df23Summer['month'] == 5, 'zVPD']
#zVPDapril = df23Summer.loc[df23Summer['month'] == 4, 'zVPD']
#zVPDmarch = df23Summer.loc[df23Summer['month'] == 3, 'zVPD']
# adding together various months zVPD's
zVPDmj = zVPDjune + zVPDmay  
# How can I align zVPDmj to pixel (system:indexviejuno) and to a certain month in df23Summer?
# Motivation: no hacks, simple, have df with all info necessary for statistical analysis, graphing etc. 
# df23Summer['zVPDmj'] =


#zVPD_J_M = df23Summer.groupby(['system:indexviejuno', 'month'=])
# = df23Summer[df23Summer['month'] == 6]
#df23Summer['zVPD_JuneMay'] = 


"""
# EXAMPLE CODE

df23['month'] = ((df23['fractionofyear'] - df23['fractionofyear'].astype(int))*12).round().astype(int)

# Monthly dfs
#df23July = df23[df23['month']==7]
df23June = df23[df23['month'] == 6]
#df23May = df23[df23['month'] == 5]

meanP = df23.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df23.groupby(['system:indexviejuno', 'month'])['P'].std() 
df23 = df23.join(meanP, on=['system:indexviejuno', 'month'], rsuffix='overAllyearsmean')
df23 = df23.join(stdP, on=['system:indexviejuno', 'month'], rsuffix='overAllyearsstd')
fractionofyearmeanP = df23.groupby(['system:indexviejuno', 'fractionofyear'])['P'].mean()
df23 = df23.join(fractionofyearmeanP, on=['system:indexviejuno', 'fractionofyear'], rsuffix='fractionofyearmean')
df23['Panomaly']=(df23['Pfractionofyearmean']-df23['PoverAllyearsmean'])/df23['PoverAllyearsstd']
print df23.info()
"""


id_pixel = pd.unique(df23Summer.index)
#p = df23Summer[df23Summer.index==id_pixel[0]]
#p500 = df23Summer[df23Summer.index==id_pixel[500]]
#p1000 = df23Summer[df23Summer.index==id_pixel[1000]]
#p1500 = df23Summer[df23Summer.index==id_pixel[1500]]

modeled_pixel_count = 0 
R_squared = []
Rsquared_adj = []

for pixel in range(id_pixel.size):
    model = sm.ols(formula='zNDVI ~ zVPD + zP', data = df23Summer[df23Summer.index==id_pixel[pixel]])
    results = model.fit()
    modeled_pixel_count += 1
    R_squared = np.append([results.rsquared],[R_squared])
    Rsquared_adj = np.append([results.rsquared_adj], [Rsquared_adj])
    #results.rsquared.append(Rsquared)
    #print results.params
    #print results.summary()
    print modeled_pixel_count


zVPDcumsum = df23Summer.groupby('month').zVPD.cumsum()
