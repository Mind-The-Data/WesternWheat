#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:38:32 2018

@author: brian
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as st
#sns.set(style="ticks")






met = pd.read_csv(data_path + 'MeteorMonthly_experiment_' + crop_id + ".csv")
ndvi = pd.read_csv(data_path + 'VegInd3_' + crop_id + ".csv")    
del ndvi['system:indexviejuno.1']
met.date = pd.to_datetime(met.date).astype(str)
dfMaster = ndvi.merge(met, on = ['system:indexviejuno', 'date'], how= 'left')


###############################################

met = pd.read_csv(data_path + 'MeteorMonthly_' + crop_id + ".csv", index_col='system:indexviejuno')
ndvi = pd.read_csv(data_path + 'VegInd' + crop_id + ".csv", index_col='system:indexviejuno')
met = met.reset_index()
ndvi = ndvi.reset_index()
met.index = met['date']
ndvi.index = ndvi['date']

# merge datasets to perform correlations
# merge combines by index and by columns that have the same name and merges values that line up
dfMaster1 = ndvi.merge(met)
#############################################
# adding one to the month column...
dfMaster1.month = dfMaster1.month.apply(lambda x: x+1)
# reset index to.... 
dfMaster1.index = dfMaster1['system:indexviejuno']
## Geo split.
dfMaster1['.geo'] = dfMaster1['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
dfMaster1['.geo'] = dfMaster1['.geo'].map(lambda x: str(x)[:-2])
dfMaster1['Longitude'], dfMaster1['Latitude']= dfMaster1['.geo'].str.split(',', 1).str


#rename column
#dfMaster.rename(columns= {'zPrecip':'zP'}, inplace=True)
dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfMasterSummer.dropna(inplace=True)

dfMaster1Jan = dfMaster1.loc[(dfMaster1['month'] ==1)]
dfMaster1Feb = dfMaster1.loc[(dfMaster1['month'] ==2)]
dfMaster1March = dfMaster1.loc[(dfMaster1['month'] ==3)]
dfMaster1April = dfMaster1.loc[(dfMaster1['month'] ==4)]
dfMaster1May = dfMaster1.loc[(dfMaster1['month'] ==5)]
dfMaster1June = dfMaster1.loc[(dfMaster1['month'] ==6)]
dfMaster1July = dfMaster1.loc[(dfMaster1['month'] ==7)]
dfMaster1Aug = dfMaster1.loc[(dfMaster1['month'] ==8)]
dfMaster1Sep = dfMaster1.loc[(dfMaster1['month'] ==9)]
dfMaster1Oct = dfMaster1.loc[(dfMaster1['month'] ==10)]
dfMaster1Nov = dfMaster1.loc[(dfMaster1['month'] ==11)]
dfMaster1Dec = dfMaster1.loc[(dfMaster1['month'] ==12)]

dfMasterJan = dfMaster.loc[(dfMaster['month'] ==1)]
dfMasterFeb = dfMaster.loc[(dfMaster['month'] ==2)]
dfMasterMarch = dfMaster.loc[(dfMaster['month'] ==3)]
dfMasterApril = dfMaster.loc[(dfMaster['month'] ==4)]
dfMasterMay = dfMaster.loc[(dfMaster['month'] ==5)]
dfMasterJune = dfMaster.loc[(dfMaster['month'] ==6)]
dfMasterJuly = dfMaster.loc[(dfMaster['month'] ==7)]
dfMasterAug = dfMaster.loc[(dfMaster['month'] ==8)]
dfMasterSep = dfMaster.loc[(dfMaster['month'] ==9)]
dfMasterOct = dfMaster.loc[(dfMaster['month'] ==10)]
dfMasterNov = dfMaster.loc[(dfMaster['month'] ==11)]
dfMasterDec = dfMaster.loc[(dfMaster['month'] ==12)]

###

# KDE plots

sns.kdeplot(dfMaster1Jan.NDVI)
sns.kdeplot(dfMaster1Feb.NDVI)
sns.kdeplot(dfMaster1March.NDVI)
sns.kdeplot(dfMaster1April.NDVI)
sns.kdeplot(dfMaster1May.NDVI)
sns.kdeplot(dfMaster1June.NDVI)
sns.kdeplot(dfMaster1July.NDVI)
sns.kdeplot(dfMaster1Aug.NDVI)
sns.kdeplot(dfMaster1Sep.NDVI)
sns.kdeplot(dfMaster1Oct.NDVI)
sns.kdeplot(dfMaster1Nov.NDVI)
sns.kdeplot(dfMaster1Dec.NDVI)


sns.kdeplot(dfMasterJan.NDVI)
sns.kdeplot(dfMasterFeb.NDVI)
sns.kdeplot(dfMasterMarch.NDVI)
sns.kdeplot(dfMasterApril.NDVI)
sns.kdeplot(dfMasterMay.NDVI)
sns.kdeplot(dfMasterJune.NDVI)
sns.kdeplot(dfMasterJuly.NDVI)
sns.kdeplot(dfMasterAug.NDVI)
sns.kdeplot(dfMasterSep.NDVI)
sns.kdeplot(dfMasterOct.NDVI)
sns.kdeplot(dfMasterNov.NDVI)
sns.kdeplot(dfMasterDec.NDVI)


sns.kdeplot(dfMaster1Feb.EVI)
sns.kdeplot(dfMaster1March.EVI)
sns.kdeplot(dfMaster1April.EVI)
sns.kdeplot(dfMaster1May.EVI)
sns.kdeplot(dfMaster1June.EVI)
sns.kdeplot(dfMaster1July.EVI)
sns.kdeplot(dfMaster1Aug.EVI)
sns.kdeplot(dfMaster1Sep.EVI)



dfMasterApril.loc[(dfMasterApril.Latitude<36) & (dfMasterApril.Latitude>35)].NDVI
sns.kdeplot(dfMasterMarch.NDVI.loc[(dfMasterMarch.Latitude<32) & (dfMasterMarch.Latitude>30)])
sns.kdeplot(dfMasterApril.loc[(dfMasterApril.Latitude<36) & (dfMasterApril.Latitude>35)].NDVI)

                                  
#dflowlat = dfMaster.loc[(dfMaster.Latitude<37.5)]

#dfMaster is current, large and low ndvi dataset
#dfMaster1 is old, small and high ndvi value dataset

sns.relplot(x='NDVI', y='EVI', hue='year', data=dfMasterApril, legend='full')
sns.relplot(x='NDVI', y='EVI', hue='year', data=dfMaster1April, legend='full')

dfMaster.index.unique()
dfMaster1.index.unique()

dfMaster['cropland_mode'].value_counts()
dfMaster1['CDL'].value_counts()

dfMasterApril['cropland_mode'].value_counts()
dfMaster1April['CDL'].value_counts()

dfMaster.year.value_counts()
dfMaster1.year.value_counts()

dfMasterApril.loc(dfMasterApril.cropland_mode == 24)

df.info()

df.index
p = None
p = df['system:indexviejuno']==69981.0



## Uncomment to diplay timeseries of pixel number pix
pix = 1600
p = dfMaster[dfMaster['system:indexviejuno']==id_pixel[pix]]
pclim = dfMasterMetMonthly[dfMasterMetMonthly['system:indexviejuno']==id_pixel[pix]]
plt.plot(p.index.to_timestamp(), p['anomalyNDVI'], 'x-', label='Veget Anomaly')
plt.plot(pclim.index.to_timestamp(), pclim['zPrecip'], 'o-', label='Clim Anomaly')
plt.legend()
plt.show()

p = 
plt.plot(dfMaster.loc[(dfMaster['system:indexviejuno'] ==p)].fractionofyear, dfMaster.loc[(dfMaster['system:indexviejuno'] ==p)].NDVI)
