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
dfMaster24 = None

data_path = '../rawData/AgMet/'

dfMaster24 = pd.DataFrame()
for f in sorted(glob.glob(data_path + 'monthlyLSclass24*')):
    dfMaster24 = pd.concat((dfMaster24, pd.read_csv(f)))


dfMaster24.index = dfMaster24['system:indexviejuno']
dfMaster24['NDVI24'] = (dfMaster24['B4'] - dfMaster24['B3'])/(dfMaster24['B4'] + dfMaster24['B3'])
dfMaster24['month'] = (((dfMaster24['fractionofyear'] - dfMaster24['fractionofyear'].astype(int))*12).round().astype(int)+1)
dfMaster24['year'] = (dfMaster24['fractionofyear'].astype(int)).round()
###############################
#######################investigating ndvi debacle
dfMaster24['system:indexviejuno'].unique
dfMaster24.index.unique

dfMaster24.info
dfMaster24 = dfMaster24.loc[(dfMaster24.cropland_mode == 24)]

dfMasterJan = dfMaster24.loc[(dfMaster24['month'] ==1)]
dfMasterFeb = dfMaster24.loc[(dfMaster24['month'] ==2)]
dfMasterMarch = dfMaster24.loc[(dfMaster24['month'] ==3)]
dfMasterApril = dfMaster24.loc[(dfMaster24['month'] ==4)]
dfMasterMay = dfMaster24.loc[(dfMaster24['month'] ==5)]
dfMasterJune = dfMaster24.loc[(dfMaster24['month'] ==6)]
dfMasterJuly = dfMaster24.loc[(dfMaster24['month'] ==7)]
dfMasterAug= dfMaster24.loc[(dfMaster24['month'] ==8)]
dfMasterSep = dfMaster24.loc[(dfMaster24['month'] ==9)]
dfMasterOct = dfMaster24.loc[(dfMaster24['month'] ==10)]
dfMasterNov = dfMaster24.loc[(dfMaster24['month'] ==11)]
dfMasterDec = dfMaster24.loc[(dfMaster24['month'] ==12)]


sns.kdeplot(dfMasterFeb.NDVI24)
sns.kdeplot(dfMasterMarch.NDVI24)
sns.kdeplot(dfMasterApril.NDVI24)
sns.kdeplot(dfMasterMay.NDVI24)
sns.kdeplot(dfMasterJune.NDVI24)
sns.kdeplot(dfMasterJuly.NDVI24)
sns.kdeplot(dfMasterAug.NDVI24)
sns.kdeplot(dfMasterSep.NDVI24)
sns.kdeplot(dfMasterOct.NDVI24)
sns.kdeplot(dfMasterNov.NDVI24)
sns.kdeplot(dfMasterDec.NDVI24)

plt.scatter(dfMaster24.fractionofyear, dfMaster24.NDVI24, s=.001)
plt.scatter(dfMaster24.loc[(dfMaster24.year == 2012)].month, dfMaster24.loc[(dfMaster24.year == 2012)].NDVI24, s =.01)


#pd.DataFrame.to_csv(path="../rawData/AgMet", sep=',')

meanNDVI24 = dfMaster24.groupby(['system:indexviejuno', 'month'])['NDVI24'].mean()
stdNDVI24 = dfMaster24.groupby(['system:indexviejuno', 'month'])['NDVI24'].std()

dfMaster24 = dfMaster24.join(meanNDVI24, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMaster24 = dfMaster24.join(stdNDVI24, on=['system:indexviejuno', 'month'], rsuffix='std')
dfMaster24['NDVI24anomaly'] = (dfMaster24['NDVI24'] - dfMaster24['NDVI24mean']) / dfMaster24['NDVI24std']

#print dfMaster24




id_pixel = pd.unique(dfMaster24.index)

p = dfMaster24[dfMaster24.index==id_pixel[0]]
p50 = dfMaster24[dfMaster24.index==id_pixel[50]]
p500 = dfMaster24[dfMaster24.index==id_pixel[500]]
p1000 = dfMaster24[dfMaster24.index==id_pixel[1000]]
p1500 = dfMaster24[dfMaster24.index==id_pixel[1500]]
p1900 = dfMaster24[dfMaster24.index==id_pixel[1900]]


#plt.plot(p['fractionofyear'], p['NDVI24anomaly'])
plt.plot(p['fractionofyear'], p['NDVI24mean'])
#plt.plot(p['fractionofyear'], p['NDVI24std'])
plt.plot(p50['fractionofyear'], p50['NDVI24mean'])
plt.plot(p500['fractionofyear'], p500['NDVI24mean'])

plt.scatter(p['month'], p['NDVI24mean'])
plt.scatter(p50['month'], p50['NDVI24mean'])
plt.scatter(p500['month'], p500['NDVI24mean'])
plt.scatter(p1000['month'], p1000['NDVI24mean'])
plt.scatter(p1500['month'], p1500['NDVI24mean'])
plt.scatter(p1900['month'], p1900['NDVI24mean'])


plt.show()


#print dfMaster24['fractionofyear']
