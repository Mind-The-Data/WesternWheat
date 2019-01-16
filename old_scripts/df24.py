#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:23:04 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

#Loading dataframes
data_path = '../rawData/AgMet/'

dfndvi24 = pd.DataFrame()
for f in sorted(glob.glob(data_path + 'monthlyLSclass24*')):
    dfndvi24 = pd.concat((dfndvi24, pd.read_csv(f)))
dfndvi24.index = dfndvi24['system:indexviejuno']

datapath = '..'
dfMet24 = pd.read_csv(datapath + '/rawData/AgMet/monthlymeteo24.csv')
dfMet24.index= dfMet24['system:indexviejuno']
#concatenation
df24 = pd.concat([dfMet24, dfndvi24])
#Index
df24.index = df24['system:indexviejuno']

#Renaming/Adding columns
df24 = df24.rename(columns={'vpd':'VPD', 'pr':'P'})
df24['month'] = ((df24['fractionofyear'] - df24['fractionofyear'].astype(int))*12).round().astype(int)

"""
#VPD
meanVPD = df24.groupby(['system:indexviejuno', 'month'])['VPD'].mean()
stdVPD = df24.groupby(['system:indexviejuno', 'month'])['VPD'].std()
#join to df24
df24 = df24.join(meanVPD, on=['system:indexviejuno', 'month'], rsuffix='mean')
df24 = df24.join(stdVPD, on=['system:indexviejuno', 'month'], rsuffix='std')
df24['VPDanomaly'] = (df24['VPD'] - df24['VPDmean'])/df24['VPDstd']
"""

#VPD: a more straightforward method?
meanVPD = df24.groupby(['system:indexviejuno', 'month'])['VPD'].mean()
stdVPD = df24.groupby(['system:indexviejuno', 'month'])['VPD'].std()
monthlymeanVPD = df24.groupby(['system:indexviejuno', 'fractionofyear'])['VPD'].mean()
df24 = df24.join(meanVPD, on=['system:indexviejuno', 'month'], rsuffix='mean')
df24 = df24.join(stdVPD, on=['system:indexviejuno', 'month'], rsuffix='std')
df24 = df24.join(monthlymeanVPD, on=['system:indexviejuno', 'month'], rsuffix='monthlymean')
df24['VPDanomaly'] = (df24['monthlymeanVPD'] - df24['meanVPD'])/df24['stdVPD']
print df24.head()

"""
#Precipitation
meanP = df24.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df24.groupby(['system:indexviejuno', 'month'])['P'].std() 
df24 = df24.join(meanP, on=['system:indexviejuno', 'month'], rsuffix='mean')
df24 = df24.join(stdP, on=['system:indexviejuno', 'month'], rsuffix='std')
df24['Panomaly'] = (df24['P'] - df24['Pmean'])/df24['Pstd']
#mean over the month at each pixel, 
meananomalyP = df24.groupby(['system:indexviejuno', 'fractionofyear'])['Panomaly'].mean()
df24 = df24.join(meananomalyP, on=['system:indexviejuno', 'fractionofyear'], rsuffix='Mean')
"""
#Precipitation: a more straight forward method
meanP = df24.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df24.groupby(['system:indexviejuno', 'month'])['P'].std()
monthlymeanP = df24.groupby(['system:indexviejuno', 'fractionofyear'])['P'].mean()
df24['Panomaly'] = (monthlymeanP - meanP)/stdP

                            

#NDVI
df24['NDVI24'] = (df24['B4'] - df24['B3'])/(df24['B4'] + df24['B3'])
meanNDVI24 = df24.groupby(['system:indexviejuno', 'month'])['NDVI24'].mean()
stdNDVI24 = df24.groupby(['system:indexviejuno', 'month'])['NDVI24'].std()

df24 = df24.join(meanNDVI24, on=['system:indexviejuno', 'month'], rsuffix='mean')
df24 = df24.join(stdNDVI24, on=['system:indexviejuno', 'month'], rsuffix='std')
df24['NDVI24anomaly'] = (df24['NDVI24'] - df24['NDVI24mean']) / df24['NDVI24std']

#print df24




id_pixel = pd.unique(df24.index)

p = df24[df24.index==id_pixel[0]]
p50 = df24[df24.index==id_pixel[50]]
p500 = df24[df24.index==id_pixel[500]]
#plt.plot(p['fractionofyear'], p['NDVI24anomaly'])
plt.plot(p['fractionofyear'], p['NDVI24mean'])
#plt.plot(p['fractionofyear'], p['NDVI24std'])
plt.plot(p50['fractionofyear'], p50['NDVI24mean'])
plt.plot(p500['fractionofyear'], p500['NDVI24mean'])

plt.show()


#print df24['fractionofyear']
