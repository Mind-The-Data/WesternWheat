#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:34:19 2018

@author: brian
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import glob
import scipy.stats as st
import seaborn as sns


crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'


ndvi = pd.read_csv(data_path + 'VegInd3_' + crop_id + ".csv")    
met = pd.read_csv(data_path + 'MeteorMonthly_experiment_' + crop_id + ".csv")   
del ndvi['system:indexviejuno.1']
met.date = pd.to_datetime(met.date).astype(str)
dfMaster=None
dfMaster = ndvi.merge(met, on = ['system:indexviejuno', 'date'])
##### DatetimeIndex set with frequency M
dfMaster.index = pd.DatetimeIndex(dfMaster.date)
dfMaster = dfMaster.to_period('M')

df = dfMaster


'''
for month in range(pd.unique(dfMaster.month).size):
    g = sns.kdeplot(dfMaster.loc[(dfMaster.month==month)].NDVI)
''' 
########## NDVIi
########### w ndvi df not dfmaster   
def Add_NDVI(df):
    iNDVI = df.loc[(df.month>1) & (df.month<8)].groupby(['system:indexviejuno','year'])['NDVI'].sum()
    df = df.join(iNDVI, on=['system:indexviejuno', 'year'], rsuffix='i')
    return df

iNDVI = df.loc[(df.month>1) & (df.month<8)].groupby(['system:indexviejuno','year'])['NDVI'].sum()
df = df.join(iNDVI, on=['system:indexviejuno', 'year'], rsuffix='i')

df = df.apply(Add_NDVI)

#########################
#####Plotting to investigate
###### TIME SERIES
#########################

df.NDVIi.plot(kind='hist', bins=30)
plt.scatter(df.NDVI, df.NDVIi, s=.01)
df.NDVIi.sort_values()

sns.relplot(x='NDVIi', y='NDVI', data=df, hue='month', legend='full', s=6)

df.index
df['system:indexviejuno']
df.loc[df['system:indexviejuno']==194579.0].year.value_counts()

pixels = [194579.0, 151137.0,188626.0,67733.0,96691.0,
29190.0,109519.0,192428.0, 122094.0, 47684.0,
119859.0,193811.0,154059.0,143324.0,215141.0,
164352.0,199867.0,153157.0,72012.0,130948.0,
30678.0,40447.0,14212.0,84091.0,
68739.0,131657.0,54702.0,201793.0,169412.0,
 77862.0,137094.0,165439.0,142039.0,108240.0,42510.0]
pixels=[194579.0, 151137.0,137094.0,42510.0]
for i in pixels:
    #sns.barplot(y='NDVIi', x='year', hue=None, data = df.loc[df['system:indexviejuno']==i])
    #sns.relplot(y='NDVI', x='month', hue='year', data = df.loc[df['system:indexviejuno']==i])
    #sns.relplot(y='NDVI', x='fractionofyear', hue='month', data = df.loc[df['system:indexviejuno']==i], kind='line')
    sns.relplot(y='NDVI', x='fractionofyear', hue=None, data = df.loc[df['system:indexviejuno']==i], kind='line')


##############################
##############################
############################
########ROLLING WINDOW CALC
###########################
##########################
##########################
# simple first
one = df.loc[(df['system:indexviejuno']==42510.0)]

NDVImean3=one.groupby('year').rolling(3).NDVI.mean()
NDVImean3.index
one['NDVImean3'] = NDVImean3.reset_index(level=0, drop=True)

NDVIsum3 = one.groupby('year').rolling(3).NDVI.sum()
one['NDVIsum3'] = NDVIsum3.reset_index(level=0, drop=True)

plt.figure()

one.plot(x='fractionofyear', y='NDVIsum3')
one.plot(x='fractionofyear', y='NDVImean3')
one.plot(x='fractionofyear', y='NDVI')

###################




NDVIsum3 = df.groupby(['system:indexviejuno','year']).rolling(3).NDVI.sum()

NDVIsum3 = NDVIsum3.reset_index(level=[0,1,2], drop=True).to






###########################
#####################
### GROWING DEGREE DAYS
######################
###########################


dailydf = dfMasterMet


def Degree_Days(df):
    GDD = df.loc[(df.tmmx>5.5)].groupby(['system:indexviejuno','year'])['avgtemp'].sum()
    df = df.join(GDD, on=['system:indexviejuno','year'], rsuffix='GDD')
    df.rename(columns={'avgtempGDD':'GDD'}, inplace=True)
    return df

Degree_Days(dailydf)


df.loc[(df.month>1) & (df.month<8)]
df= dfMaster.join(iNDVI, on=['system:indexviejuno', 'year'], rsuffix='i')



########################
########### VISUALIZE
######################
#sns.relplot(y='Latitude', x='Longitude', palette="seismic", s=9, legend = 'full', data=df)

map =Basemap(projection='stere', lon_0=-105, lat_0=90.,\
            llcrnrlat=29,urcrnrlat=49,\
            llcrnrlon=-117,urcrnrlon=-87.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)

x, y = map(df.loc[(df.month>1) & (df.month<8)].Longitude, df.loc[(df.month>1) & (df.month<8)].Latitude)
x = np.array(x)
y = np.array(y)
map.drawcoastlines(linewidth=1)
map.drawstates()
map.drawcountries(linewidth=1.1)
map.drawmeridians(range(-140, -80, 5), linewidth=.3)
map.drawparallels(range(20, 60, 5),linewidth=.3)
#map.drawrivers(linewidth=.1)
#map.scatter(x[pvalues.zVPD.values<0.1], y[pvalues.zVPD<.1], c=params.zVPD.values[pvalues.zVPD<0.1], marker='^', cmap='seismic', alpha=1., s=5.8) #.
#map.scatter(x[pvalues.zVPD.values>0.1], y[pvalues.zVPD>0.1], c=params.zVPD.values[pvalues.zVPD>0.1], marker='v', cmap='seismic', alpha=1., s=5.8) #.
map.scatter(x, y, c=iNDVI, marker='^', cmap='seismic', alpha=1., s=5.8) #.
#map.scatter(x, y, c=pvalues, cmap='gray', alpha=.08, s=5.8) #.
plt.colorbar()
plt.clim(-1,1)