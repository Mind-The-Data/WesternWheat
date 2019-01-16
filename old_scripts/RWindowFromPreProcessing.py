#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 17:10:12 2018

@author: brian
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import glob
import scipy.stats as st


def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date

crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

print "Reading CDL dataframe from harddrive..."
dfCDL = pd.read_csv(data_path + "CDL" + crop_id + ".csv", usecols=['system:index','countyears','cropland_mode','first','system:indexviejuno'])
dfCDL['year'] = dfCDL['system:index'].apply(lambda x: x.split("_")[0]).astype(int)

dfMaster = None
dfMaster = pd.DataFrame()
print "Loading EM spectral dataframes..."
for f in glob.glob(data_path + 'monthlyLSclass' + crop_id + '*'):
    dfMaster = pd.concat((dfMaster,  pd.read_csv(f)))

print 'Performing Marcos date magic....'
# for 23 I had to add one to month as fractionofyear started at 2008.00 and ended at 2008.916 ... 
# Then make sure date calculation is correct. 
dfMaster['month'] = (((dfMaster['fractionofyear'] - dfMaster['fractionofyear'].astype(int))*12).round().astype(int)+1)
dfMaster['year'] = (np.modf(dfMaster['fractionofyear'])[1]).astype(int)
dfMaster['date'] = pd.to_datetime(dfMaster.year * 100 + (dfMaster.month), format='%Y%m')
dfMaster.index = dfMaster['date']
dfMaster = dfMaster.to_period('M')
# these two lines below not necessary... as dfMaster already has cropland_mode
lkupTable = dfCDL.pivot('year', 'system:indexviejuno', 'first')
dfMaster['CDL'] = lkupTable.lookup(dfMaster['year'], dfMaster['system:indexviejuno'])
print "Crop/fallow pixel filtering..."
dfMaster = dfMaster.loc[dfMaster['CDL'] == float(crop_id)]
####
# Investigating remote sensing data structure
#dfMaster.groupby(['system:indexviejuno', 'month'])['B1'].count()
###
print "Calculating Vegetation Indices..."
dfMaster.index = dfMaster['system:indexviejuno']
dfMaster.dropna(inplace=True)
dfMaster['NDVI'] = (dfMaster['B4'] - dfMaster['B3'])/(dfMaster['B4'] + dfMaster['B3'])
dfMaster['EVI'] = 2.5*(dfMaster['B4'] - dfMaster['B3'])/(dfMaster['B4'] + 6*dfMaster['B3']-7.5*dfMaster['B1']+1)
dfMaster['NDWI1'] = (dfMaster['B4'] - dfMaster['B5'])/(dfMaster['B4'] + dfMaster['B5'])
dfMaster['NDWI2'] = (dfMaster['B4'] - dfMaster['B7'])/(dfMaster['B4'] + dfMaster['B7'])



#tell me how many points we throw out
# initially I did below -1, however I had one point between 0 and -1 NDVI I want to throw out. 
NDVI_datapointsthrownout = dfMaster.loc[(dfMaster.NDVI<0) | (dfMaster.NDVI>1)].count()
EVI_datapointsthrownout = dfMaster.loc[(dfMaster.EVI<-2) | (dfMaster.EVI>2)].count()
print "NDVI data points thrown out: " +str(NDVI_datapointsthrownout.NDVI)
print "EVI data points thrown out: " +str(EVI_datapointsthrownout.EVI)
#Investigating
#df = dfMaster.loc[(dfMaster.NDVI<-1) | (dfMaster.NDVI>1)]
pd.set_option('display.max_columns', 25)
neg =dfMaster.loc[(dfMaster.B3<0)]
#neg.info()
#df
#dfMaster.B3.sort_values().head(5)
#dfMaster.B4.sort_values().head(5)

#Throwing out points...
dfMaster=dfMaster.loc[(dfMaster.NDVI > 0.) & (dfMaster.NDVI <1.)]
dfMaster=dfMaster.loc[(dfMaster.EVI > -2.) & (dfMaster.EVI <2.)]


meanNDVI = dfMaster.groupby(['system:indexviejuno', 'month'])['NDVI'].mean()
stdNDVI = dfMaster.groupby(['system:indexviejuno', 'month'])['NDVI'].std()
#maxNDVI = dfMaster.groupby(['system:indexviejuno', 'month', 'year'])['NDVI'].max()
#NDVI = dfMaster.groupby(['system:indexviejuno', 'month' , 'year'])['NDVI'].mean()

meanEVI = dfMaster.groupby(['system:indexviejuno', 'month'])['EVI'].mean()
stdEVI = dfMaster.groupby(['system:indexviejuno', 'month'])['EVI'].std()
#maxEVI = dfMaster.groupby(['system:indexviejuno', 'month', 'year'])['EVI'].max()
#EVI = dfMaster.groupby(['system:indexviejuno', 'month','year'])['EVI'].mean()

meanNDWI1 = dfMaster.groupby(['system:indexviejuno', 'month'])['NDWI1'].mean()
stdNDWI1 = dfMaster.groupby(['system:indexviejuno', 'month'])['NDWI1'].std()
#NDWI1 = dfMaster.groupby(['system:indexviejuno', 'month' ,'year'])['NDWI1'].mean()

meanNDWI2 = dfMaster.groupby(['system:indexviejuno', 'month'])['NDWI2'].mean()
stdNDWI2 = dfMaster.groupby(['system:indexviejuno', 'month'])['NDWI2'].std()
#NDWI2 = dfMaster.groupby(['system:indexviejuno', 'month' ,'year'])['NDWI2'].mean()

print "Joining vegetation indices arrays into dfMaster..."
dfMaster = dfMaster.join(meanNDVI, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVI, on=['system:indexviejuno', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(maxNDVI, on=['system:indexviejuno', 'month' ,'year'], rsuffix='max')
#dfMaster = dfMaster.join(NDVI, on=['system:indexviejuno', 'month', 'year'], rsuffix='monthly')

dfMaster = dfMaster.join(meanEVI, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdEVI, on=['system:indexviejuno', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(EVI, on=['system:indexviejuno', 'month', 'year'], rsuffix='monthly')
#dfMaster = dfMaster.join(maxEVI, on=['system:indexviejuno', 'month' ,'year'], rsuffix='max')

dfMaster = dfMaster.join(meanNDWI1, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDWI1, on=['system:indexviejuno', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(NDWI1, on=['system:indexviejuno', 'month', 'year'], rsuffix='monthly')

dfMaster = dfMaster.join(meanNDWI2, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDWI2, on=['system:indexviejuno', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(NDWI2, on=['system:indexviejuno', 'month', 'year'], rsuffix='monthly')

print 'Calculating Anomalies...'
dfMaster['zNDVI'] = (dfMaster['NDVI'] - dfMaster['NDVImean']) / dfMaster['NDVIstd']
dfMaster['zEVI'] = (dfMaster['EVI'] - dfMaster['EVImean']) / dfMaster['EVIstd']
dfMaster['zNDWI1'] = (dfMaster['NDWI1'] - dfMaster['NDWI1mean']) / dfMaster['NDWI1std']
dfMaster['zNDWI2'] = (dfMaster['NDWI2'] - dfMaster['NDWI2mean']) / dfMaster['NDWI2std']


###### GEO split
dfMaster['.geo'] = dfMaster['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
dfMaster['.geo'] = dfMaster['.geo'].map(lambda x: str(x)[:-2])
dfMaster['Longitude'], dfMaster['Latitude']= dfMaster['.geo'].str.split(',', 1).str

##########################
##########################
##### ROLLING WINDOW NDVI
##########################
##########################
'''
one=None
one = dfMaster.loc[(dfMaster['system:indexviejuno']==69981.0)]
plt.plot(one.fractionofyear, one.NDVI)

NDVIsum3=None
NDVIsum3 = one.groupby('year').rolling(3).NDVI.sum()
NDVIsum3 = NDVIsum3.reset_index(level=0)
one = one.join(NDVIsum3, rsuffix='sum3')
one.plot(x='fractionofyear',y='NDVI'), one.plot(x='fractionofyear',y='NDVIsum3')
one.plot(x='month',y='NDVI'), one.plot(x='month',y='NDVIsum2')
'''
#########################
#########################
#MulitPixel##############
#########################

dfMaster.index = dfMaster.date # has to be a date index so when recombined it has date and pixel which makes unique

NDVIsum3=None
NDVIsum3 = dfMaster.groupby(['system:indexviejuno','year']).rolling(3).NDVI.sum()

NDVIsum3 = NDVIsum3.reset_index(level=[0,1]) # from multi index to one index
NDVIsum3= NDVIsum3.reset_index(level=0) # change index to default range/period so date can be used by merge as a column
#NDVIsum3.index = NDVIsum3.level_2
#dfMaster = dfMaster.merge(NDVIsum3, left_index=True, right_index=True)
NDVIsum3.rename(index=str, columns={'NDVI':'NDVIsum3'}, inplace=True)
dfMaster = dfMaster.merge(NDVIsum3, on=['system:indexviejuno', 'date'])
dfMaster.rename(index=str, columns={'year_x':'year'}, inplace=True)
dfMaster.drop(columns='year_y',inplace=True)
#
#
#Means, stds, zs

meanNDVIsum3 = dfMaster.groupby(['system:indexviejuno', 'month'])['NDVIsum3'].mean()
stdNDVIsum3 = dfMaster.groupby(['system:indexviejuno', 'month'])['NDVIsum3'].std()
dfMaster = dfMaster.join(meanNDVIsum3, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVIsum3, on=['system:indexviejuno', 'month'], rsuffix='std')
dfMaster['zNDVIsum3'] = (dfMaster['NDVIsum3'] - dfMaster['NDVIsum3mean']) / dfMaster['NDVIsum3std']

plt.scatter(dfMaster.NDVI,dfMaster.zNDVIsum3, s=.01)

dfMaster.zNDVIsum3.value_counts()


###################
#INVESTIGATING
##################

## Investigating anomalies
### Lots of values near that abs 0.707107 value... can't just delete... right?
############ Most are in months we wont be using... but still curious on why they are showing up.
# It cannot be a figment of my data because it shows up with different algorithms, zNDVI and zNDVIsum3

dfMaster.zNDVIsum3.sort_values().head()
dfMaster.zNDVIsum3.sort_values().tail()
dfMaster.zNDVIsum3.describe()
dfMaster.zNDVIsum3.value_counts().sort_values()
dfMaster.groupby(['system:indexviejuno','year']).zNDVIsum3.max().value_counts()
#Mean sum 3
dfMaster.groupby(['system:indexviejuno','year']).NDVIsum3.max()
dfMaster.groupby(['system:indexviejuno','year']).NDVIsum3.min()
dfMaster.groupby(['system:indexviejuno','year']).NDVIsum3.describe()
#dfMaster.zEVI.value_counts().sort_values()
#dfMaster.zNDWI1.value_counts().sort_values()
#dfMaster.zNDWI2.value_counts().sort_values()

df = dfMaster.loc[np.isclose(dfMaster['zNDVIsum3'].abs(), .7071)]
df['system:indexviejuno'].value_counts()
df.B5.value_counts()
df.B4.value_counts()
df.NDVIsum3.value_counts()

df.columns
df['system:indexviejuno'].value_counts()
df.NDVIsum3mean.value_counts()
plt.scatter(df.NDVIsum3mean, df['Latitude'], s=.1)


df.NDVIsum3std.value_counts()
df.zNDVIsum3.value_counts()
df.NDVIsum3
df.Latitude.value_counts()
df.Longitude.value_counts()
df.month.value_counts()
df.CDL.value_counts()




#compared to dfmaster... how likely is this?
dfMaster.Longitude.value_counts()
dfMaster.Latitude.value_counts()
dfMaster.NDVIsum3.value_counts()
dfMaster.NDVIsum3mean.value_counts()
dfMaster.NDVIsum3std.value_counts()

dfSpring = dfMaster.loc[(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfSpring.Longitude.value_counts()
dfSpring.Latitude.value_counts()
dfSpring.NDVIsum3.value_counts()
dfSpring.NDVIsum3mean.value_counts()
dfSpring.NDVIsum3std.value_counts()
dfSpring.zNDVIsum3.value_counts()

dfspring = dfMaster.loc[np.isclose(dfMaster['zNDVIsum3'].abs(), .7071) & (dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfspring.Longitude.value_counts()
dfspring.Latitude.value_counts()
dfspring.NDVIsum3.value_counts()
dfspring.NDVIsum3mean.value_counts()
dfspring.NDVIsum3std.value_counts()
dfspring.zNDVIsum3.value_counts()











#NDVIsum3 = NDVIsum3.reset_index()
#NDVIsum3.index = pd.DatetimeIndex(dfMaster.date)
#NDVIsum3 = NDVIsum3.to_period('M')
#dates dont line up... what happened???

#dfMaster=dfMaster.join(NDVIsum3, rsuffix='sum3')
#plt.plot(dfMaster.year_y, dfMaster.year_x) # just making sure both copies of year are the same! whoof!



'''