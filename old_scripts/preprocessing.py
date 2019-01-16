#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:07:45 2018

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

dfCDL.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)

dfMaster = None
dfMaster = pd.DataFrame()
print "Loading EM spectral dataframes..."
for f in glob.glob(data_path + 'monthlyLSclass' + crop_id + '*'):
    dfMaster = pd.concat((dfMaster,  pd.read_csv(f)))

dfMaster.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)


print 'Performing Marcos date magic....'
# for 23 I had to add one to month as fractionofyear started at 2008.00 and ended at 2008.916 ... 
# Then make sure date calculation is correct. 
dfMaster['month'] = (((dfMaster['fractionofyear'] - dfMaster['fractionofyear'].astype(int))*12).round().astype(int)+1)
dfMaster['year'] = (np.modf(dfMaster['fractionofyear'])[1]).astype(int)
dfMaster['date'] = pd.to_datetime(dfMaster.year * 100 + (dfMaster.month), format='%Y%m')
dfMaster.index = dfMaster['date']
dfMaster = dfMaster.to_period('M')
# these two lines below not necessary... as dfMaster already has cropland_mode
lkupTable = dfCDL.pivot('year', 'pixel', 'first')
dfMaster['CDL'] = lkupTable.lookup(dfMaster['year'], dfMaster['pixel'])
print "Crop/fallow pixel filtering..."
dfMaster = dfMaster.loc[dfMaster['CDL'] == float(crop_id)]
####
# Investigating remote sensing data structure
#dfMaster.groupby(['pixel', 'month'])['B1'].count()
###
print "Calculating Vegetation Indices..."
dfMaster.index = dfMaster['pixel']
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
dfMaster=dfMaster.loc[(dfMaster.NDVI > -1.) & (dfMaster.NDVI <1.)]
dfMaster=dfMaster.loc[(dfMaster.EVI > -2.) & (dfMaster.EVI <2.)]


meanNDVI = dfMaster.groupby(['pixel', 'month'])['NDVI'].mean()
stdNDVI = dfMaster.groupby(['pixel', 'month'])['NDVI'].std()
#maxNDVI = dfMaster.groupby(['pixel', 'month', 'year'])['NDVI'].max()
#NDVI = dfMaster.groupby(['pixel', 'month' , 'year'])['NDVI'].mean()

meanEVI = dfMaster.groupby(['pixel', 'month'])['EVI'].mean()
stdEVI = dfMaster.groupby(['pixel', 'month'])['EVI'].std()
#maxEVI = dfMaster.groupby(['pixel', 'month', 'year'])['EVI'].max()
#EVI = dfMaster.groupby(['pixel', 'month','year'])['EVI'].mean()

meanNDWI1 = dfMaster.groupby(['pixel', 'month'])['NDWI1'].mean()
stdNDWI1 = dfMaster.groupby(['pixel', 'month'])['NDWI1'].std()
#NDWI1 = dfMaster.groupby(['pixel', 'month' ,'year'])['NDWI1'].mean()

meanNDWI2 = dfMaster.groupby(['pixel', 'month'])['NDWI2'].mean()
stdNDWI2 = dfMaster.groupby(['pixel', 'month'])['NDWI2'].std()
#NDWI2 = dfMaster.groupby(['pixel', 'month' ,'year'])['NDWI2'].mean()

print "Joining vegetation indices arrays into dfMaster..."
dfMaster = dfMaster.join(meanNDVI, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVI, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(maxNDVI, on=['pixel', 'month' ,'year'], rsuffix='max')
#dfMaster = dfMaster.join(NDVI, on=['pixel', 'month', 'year'], rsuffix='monthly')

dfMaster = dfMaster.join(meanEVI, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdEVI, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(EVI, on=['pixel', 'month', 'year'], rsuffix='monthly')
#dfMaster = dfMaster.join(maxEVI, on=['pixel', 'month' ,'year'], rsuffix='max')

dfMaster = dfMaster.join(meanNDWI1, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDWI1, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(NDWI1, on=['pixel', 'month', 'year'], rsuffix='monthly')

dfMaster = dfMaster.join(meanNDWI2, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDWI2, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(NDWI2, on=['pixel', 'month', 'year'], rsuffix='monthly')

print 'Calculating Anomalies...'
dfMaster['zNDVI'] = (dfMaster['NDVI'] - dfMaster['NDVImean']) / dfMaster['NDVIstd']
dfMaster['zEVI'] = (dfMaster['EVI'] - dfMaster['EVImean']) / dfMaster['EVIstd']
dfMaster['zNDWI1'] = (dfMaster['NDWI1'] - dfMaster['NDWI1mean']) / dfMaster['NDWI1std']
dfMaster['zNDWI2'] = (dfMaster['NDWI2'] - dfMaster['NDWI2mean']) / dfMaster['NDWI2std']




#0.707107 comes up A LOT
#Mostly in winter months... snow? clouds? 
dfMaster.loc[np.isclose(dfMaster['zNDVI'].abs(), 0.707107), 'zNDVI'] = np.nan

## Investigating anomalies
### Lots of values near that abs 0.707107 value... can't just delete... right?
'''
dfMaster.zNDVI.sort_values().head()
dfMaster.zNDVI.sort_values().tail()
dfMaster.zNDVI.describe()
dfMaster.zNDVI.value_counts().sort_values()
dfMaster.zEVI.value_counts().sort_values()
dfMaster.zNDWI1.value_counts().sort_values()
dfMaster.zNDWI2.value_counts().sort_values()

df = dfMaster.loc[np.isclose(dfMaster['zNDVI'].abs(), .7071)]
df['pixel'].value_counts()
df.B5.value_counts()
df.NDVI.value_counts()
df.NDVI.value_counts().sort_values()
df.columns
df.NDVImean.value_counts()
df.NDVIstd.value_counts()
df.zNDVI.value_counts()
df.NDVI
df.Latitude.value_counts()
df.Longitude.value_counts()
df.month.value_counts()
#compared to dfmaster... how likely is this?
dfMaster.Longitude.value_counts()
dfMaster.Latitude.value_counts()
dfMaster.NDVI.value_counts()
dfMaster.NDVImean.value_counts()
dfMaster.NDVIstd.value_counts()

dfSpring = dfMaster.loc[(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfSpring.Longitude.value_counts()
dfSpring.Latitude.value_counts()
dfSpring.NDVI.value_counts()
dfSpring.NDVImean.value_counts()
dfSpring.NDVIstd.value_counts()
dfSpring.zNDVI.value_counts()

dfspring = dfMaster.loc[np.isclose(dfMaster['zNDVI'].abs(), .7071) & (dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfspring.Longitude.value_counts()
dfspring.Latitude.value_counts()
dfspring.NDVI.value_counts()
dfspring.NDVImean.value_counts()
dfspring.NDVIstd.value_counts()
dfspring.zNDVI.value_counts()
'''
###### GEO split
dfMaster['.geo'] = dfMaster['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
dfMaster['.geo'] = dfMaster['.geo'].map(lambda x: str(x)[:-2])
dfMaster['Longitude'], dfMaster['Latitude']= dfMaster['.geo'].str.split(',', 1).str
##################
#Investigating .geo split
##################
'''
lat = dfMaster['Latitude'].sort_values()
L = dfMaster['Longitude'].sort_values()
L.value_counts().sort_values()
l = dfMaster['longitude'].sort_values()
l.value_counts().sort_values()
plt.scatter( L, l)
l.count()
L.count()
'''
######################
######################
######################
# RollingWindowNDVI
######################
######################
######################

dfMaster.index = dfMaster.date # has to be a date index so when recombined it has date and pixel which makes unique

NDVIsum3=None
NDVIsum3 = dfMaster.groupby(['pixel','year']).rolling(3).NDVI.sum()
NDVIsum3 = NDVIsum3.reset_index(level=[0,1]) # from multi index to one index
NDVIsum3= NDVIsum3.reset_index(level=0) # change index to default range/period so date can be used by merge as a column
NDVIsum3.rename(index=str, columns={'NDVI':'NDVIsum3'}, inplace=True)
dfMaster = dfMaster.merge(NDVIsum3, on=['pixel', 'date'])
dfMaster.rename(index=str, columns={'year_x':'year'}, inplace=True)
dfMaster.drop(columns='year_y',inplace=True)






#dfMaster.index
#id_pixel = pd.unique(dfMaster.index)


#dfMaster.drop(columns=['','','','','','','','','',''])
print "Saving Vegetation Indices (" + crop_id + ") dataframe to hard drive..."
dfMaster[[u'fractionofyear',
       u'pixel', u'.geo', u'Latitude', u'Longitude', u'month', u'year', u'date',
       u'CDL',u'countyears',
       u'NDVI', u'EVI', u'NDWI1', u'NDWI2', u'NDVImean', u'NDVIstd', u'EVImean', u'EVIstd',
       u'NDWI1mean',u'NDWI2mean', u'NDWI1std', u'NDWI2std', u'zNDVI',
       u'zEVI', u'zNDWI1', u'zNDWI2']].to_csv(data_path + 'VegInd3_' + crop_id + '.csv')
#free memory
#dfMaster = None

##################################
############## MET ###############
##################################

# Pre processes meteorological data
print "Loading Meteorological data..." + crop_id
dfMasterMet = pd.read_csv(data_path + 'dailymeteo' + crop_id + '.csv')
print 'Met data loaded'

#index to date time function
def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date

dfMasterMet.index = dfMasterMet['pixel']
print 'applying index_to_datetime function...'
dfMasterMet['date'] = dfMasterMet['system:index'].apply(index_to_datetime)
dfMasterMet.index = dfMasterMet['date']
dfMasterMet['month'] = pd.DatetimeIndex(dfMasterMet['date']).month
dfMasterMet['year'] = pd.DatetimeIndex(dfMasterMet['date']).year

# Low pr days in a row function
# change to tenth of an inch
def add_days_in_a_row (df):
    df_sorted = df.sort_values(by='date', ascending=True)
    condition = (df_sorted['pr']<2.5)
    df_sorted['cum_daysinarow_lowpr'] = condition.cumsum()-condition.cumsum().mask(condition).ffill().fillna(0)
    return df_sorted
# calling function which inserts new column in dfMasterMet
print '...Calculating days in a row with low precipitation...this may take a few minutes and gigabytes of RAM...'
dfMasterMet = (dfMasterMet.groupby(['pixel', 'year'])).apply(add_days_in_a_row)

#converting kelvin to celsius
print 'coverting kelvin to celsius...'
dfMasterMet.tmmn = dfMasterMet.tmmn.transform(lambda x: x - 273.15)
dfMasterMet.tmmx = dfMasterMet.tmmx.transform(lambda x: x- 273.15)
print 'selecting days above threshold temp...'
dfMasterMet['daysabove28']=dfMasterMet.tmmx>28
dfMasterMet['daysabove30']=dfMasterMet.tmmx>30
dfMasterMet['daysabove35']=dfMasterMet.tmmx>35
dfMasterMet['daysbelowneg5']=dfMasterMet.tmmn<-5
dfMasterMet['daysabove_avg5.5']=((dfMasterMet.tmmn + dfMasterMet.tmmx)/2)>5.5
dfMasterMet['avgtemp']= (dfMasterMet.tmmn + dfMasterMet.tmmx)/2

dfMasterMet.index = dfMasterMet.date
dfMasterMet.columns
#cropland_mode not cropland type
#dfMasterMet = dfMasterMet.loc[dfMasterMet.cropland_mode == float(crop_id)]

####################################
####################################
print "writing daily met to  harddrive..."
dfMasterMet.to_csv(data_path + "Meteordaily_" + crop_id + ".csv")
##################################
##################################
#dfMasterMet=pd.read_csv(data_path + "Meteordaily_" + crop_id + ".csv")
print 'Daily  --->   Monthly....'
print 'aggregating daily into monthly values...this may take awhile'
# having trouble now carrying over columns that aren't aggregated from daily to monthly
# How to fix this???? You cant. you have to add them to the .agg
# u'system:index':'median', u'countyears':'median', u'cropland_mode',
#       u'fractionofyear', u'longitude' u'pixel', u'.geo', u'date', u'month'
dfMasterMetMonthly = None
dfMasterMetMonthly = dfMasterMet.groupby('pixel').resample('M')\
.agg({u'pr':'sum','eto':'mean','etr':'mean',\
'vpd': 'mean', 'srad': 'mean', 'tmmn':'mean', 'daysbelowneg5':'sum',\
'tmmx':'mean', 'daysabove28':'sum', 'daysabove30':'sum', 'daysabove35':'sum',\
 'cum_daysinarow_lowpr':'max', 'avgtemp' : 'sum' })
#.agg({'cum_daysinarow_lowpr':['sum', 'max']})
#Have to sum (or mean) days above temp because max will either be 1 or 0
    
#converting index to date, then type datetimeindex, then dropping the day
#code created by Brian, but now doesn't work, and marco's works... weird... 
#dfMasterMetMonthly.date = dfMasterMetMonthly.index
#dfMasterMetMonthly = dfMasterMetMonthly.to_timestamp()
#dfMasterMetMonthly = dfMasterMetMonthly.to_period('M')

##Marco's code that wasn't working. Somehow my index  was periodindex instead of datetimeindex
dfMasterMetMonthly = dfMasterMetMonthly.reset_index(level=0)
dfMasterMetMonthly = dfMasterMetMonthly.to_period('M')

#Why do we do this? to speed up zscore calculation i assume
###is not working currently....
#print 'values converted to np.nan:' + dfMasterMetMonthly[dfMasterMetMonthly == 0].count()
#dfMasterMetMonthly[dfMasterMetMonthly == 0] = np.nan



print "Calculating monthly meterological anomalies...." + crop_id
"""
dfMasterMetMonthly['zVPD'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['vpd']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['zP'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['pr']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['zETO'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['eto']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['zSRAD'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['srad']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['zETR'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['etr']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['ztmmn'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['tmmn']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['ztmmx'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['tmmx']\
    .transform(st.mstats.zscore)
print "classics have been calculated...now to the new z's....."
dfMasterMetMonthly['zcum_daysinarow_lowpr'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['cum_daysinarow_lowpr']\
    .transform(st.mstats.zscore)
dfMasterMetMonthly['zavgtemp'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['avgtemp']\
    .transform(st.mstats.zscore)
#dfMasterMetMonthly['zdaysabove28'] = dfMasterMetMonthly\
#    .groupby(['pixel', dfMasterMetMonthly.index.month])['daysabove28']\
#    .transform(st.mstats.zscore)
dfMasterMetMonthly['zdaysabove30'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['daysabove30']\
    .transform(st.mstats.zscore)
#dfMasterMetMonthly['zdaysabove35'] = dfMasterMetMonthly\
#    .groupby(['pixel', dfMasterMetMonthly.index.month])['daysabove35']\
#    .transform(st.mstats.zscore)
dfMasterMetMonthly['zdaysbelow-5'] = dfMasterMetMonthly\
    .groupby(['pixel', dfMasterMetMonthly.index.month])['daysbelowneg5']\
    .transform(st.mstats.zscore)
"""
'''
#Will need to manipulate because month is not a column yet   
print "Calculating VPD statistics...."
meanVPD = dfMasterMetMonthly.groupby(['pixel', 'month'])['vpd'].mean()
stdVPD = dfMasterMetMonthly.groupby(['pixel', 'month'])['vpd'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanVPD, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdVPD, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zVPD'] = (dfMasterMetMonthly['vpd'] - dfMasterMetMonthly['vpdmean']) / dfMasterMetMonthly['vpdstd']
#
print "Calculating Precip statistics...."
meanP = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].mean()
stdP = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanP, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdP, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zP'] = (dfMasterMetMonthly['pr'] - dfMasterMetMonthly['prmean']) / dfMasterMetMonthly['prstd'] 

print "Calculating Cumulative days with low pr statistics...."
mean_cum_daysinarow_lowpr = dfMasterMetMonthly.groupby(['pixel', 'month'])['cum_daysinarow_lowpr'].mean()
std_cum_daysinarow_lowpr = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(mean_cum_daysinarow_lowpr, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(std_cum_daysinarow_lowpr, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zCumDaysLowPr'] = (dfMasterMetMonthly['cum_daysinarow_lowpr'] - dfMasterMetMonthly['cum_daysinarow_lowprmean']) / dfMasterMetMonthly['cum_daysinarow_lowprstd'] 

print "Calculating avgtemp statistics...."
meanavgtemp = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].mean()
stdavgtemp = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanavgtemp, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdavgtemp, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zavgtemp'] = (dfMasterMetMonthly['avgtemp'] - dfMasterMetMonthly['avgtempmean']) / dfMasterMetMonthly['avgtempstd'] 
'''
print "writing meteorological anomalies to drive..."
#dfMasterMet[['date', 'month', 'anomalyVPD', 'anomalyPrecip']].to_csv(data_path + "Meteor_" + crop_id + ".csv")
dfMasterMetMonthly.to_csv(data_path + "MeteorMonthly_experiment_" + crop_id + ".csv")
#to reduce RAM usage
dfMasterMet = None
dfMasterMetMonthly = None

## Uncomment to diplay timeseries of pixel number pix
#pix = 1600
#p = dfMaster[dfMaster['pixel']==id_pixel[pix]]
#pclim = dfMasterMetMonthly[dfMasterMetMonthly['pixel']==id_pixel[pix]]
#plt.plot(p.index.to_timestamp(), p['anomalyNDVI'], 'x-', label='Veget Anomaly')
#plt.plot(pclim.index.to_timestamp(), pclim['zPrecip'], 'o-', label='Clim Anomaly')
#plt.legend()
#plt.show()
#                                                          160,1         Bot


# Met Anomalies

