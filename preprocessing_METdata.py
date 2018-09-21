#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:54:04 2018

@author: brian
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import glob
import scipy.stats as st

##################################
############## MET ###############
##################################
crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'


# Pre processes meteorological data
print "Loading Meteorological data..." + crop_id
dfMasterMet = pd.read_csv(data_path + 'dailymeteo' + crop_id + '.csv')
print 'Met data loaded'

#index to date time function
def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date

dfMasterMet.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)
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
dfMasterMet['avgtempabovefiveandhalf']= dfMasterMet.loc[(dfMasterMet.avgtemp>5.5)].avgtemp

dfMasterMet.index = dfMasterMet.date
dfMasterMet.columns
#cropland_mode is not cropland type it is the most common freqency of CDL
# GDD not necessary yet, will accumulate when transfer to monthly time steps....

'''
def Degree_Days(df):
    GDD = df.loc[(df.tmmx>5.5)].groupby(['pixel','year','month'])['avgtemp'].sum()
    GDD = GDD.reset_index(level=[0,1])
    GDD = GDD.reset_index(level=0)
    GDD.rename(index=str, columns={'avgtemp':'GDD'}, inplace=True)
    df = df.merge(GDD, on=['pixel','year','month'])
    return df
Degree_Days(dfMasterMet)
'''

#The simple way did not work....
#dfMasterMet['GDD'] = dfMasterMet.loc[(dfMasterMet.tmmx>5.5)].groupby(['pixel','year','month'])['avgtemp'].sum()
'''
GDD = dfMasterMet.loc[(dfMasterMet.tmmx>5.5)].groupby(['pixel','year','month'])['avgtemp'].sum()
GDD = GDD.reset_index(level=[0,1])
GDD = GDD.reset_index(level=0)
GDD.rename(index=str, columns={'avgtemp':'GDD'}, inplace=True)
dfMasterMet=dfMasterMet.merge(GDD, on=['pixel','year','month'])

plt.scatter(dfMasterMet.GDD, dfMasterMet.avgtemp, s=.01)
'''


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
.agg({u'year':'mean', u'month':'mean' ,u'pr':'sum','eto':'mean','etr':'mean',\
'vpd': 'mean', 'srad': 'mean', 'tmmn':'mean', 'daysbelowneg5':'sum',\
'tmmx':'mean', 'daysabove28':'sum', 'daysabove30':'sum', 'daysabove35':'sum',\
 'cum_daysinarow_lowpr':'max', 'avgtemp' : 'sum','avgtempabovefiveandhalf':'sum'})
#.agg({'cum_daysinarow_lowpr':['sum', 'max']})
#Have to sum (or mean) days above temp because max will either be 1 or 0
# TO INVESTIGATE
#df = dfMasterMet.groupby('pixel').resample('M')\
#.agg({u'year':'mean',u'month':'mean'})
    
#converting index to date, then type datetimeindex, then dropping the day
#code created by Brian, but now doesn't work, and marco's works... weird... 
#dfMasterMetMonthly.date = dfMasterMetMonthly.index
#dfMasterMetMonthly = dfMasterMetMonthly.to_timestamp()
#dfMasterMetMonthly = dfMasterMetMonthly.to_period('M')
#####################
# reset multiindex   
####################
w=3  #window period

print "Calculating rolling Window statistics for: " + str(w) +"months"
dfMasterMetMonthly = dfMasterMetMonthly.reset_index(level=0) #pops out of last as muliindex need date index
dfMasterMetMonthly = dfMasterMetMonthly.to_period('M')  #part of the date index reseting want date index to be just year and month

dfMasterMetMonthly.index = dfMasterMetMonthly.date # toggle on or off depending on dfMMM index type

dfMasterMetMonthly.rename(index=str, columns={'avgtempabovefiveandhalf':'GDDm','cum_daysinarow_lowpr':'dayslowpr'}, inplace=True)
GDD=None
GDD = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).avgtempabovefiveandhalf.mean()
GDD=GDD.reset_index(level=[0,1,2])
GDD.rename(index=str,columns={'GDDm':'GDDmean3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(GDD, on=['pixel','date','year'])
#dfMasterMetMonthly.rename(index=str, columns={'year_x':'year'}, inplace=True)
#dfMasterMetMonthly.drop(columns='year_y',inplace=True)


dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
VPD=None
VPD = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).vpd.mean()
VPD=VPD.reset_index(level=[0,1,2])
VPD.rename(index=str,columns={'vpd':'VPDmean3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(VPD, on=['pixel','date','year'])

dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
PR=None
PR = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).pr.sum()
PR= PR.reset_index(level=[0,1,2])
PR.rename(indexcrop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'=str,columns={'pr':'prsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(PR, on=['pixel','date','year'])


dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
dayslowpr=None
dayslowpr = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).dayslowpr.sum()
dayslowpr= dayslowpr.reset_index(level=[0,1,2])
dayslowpr.rename(index=str,columns={'dayslowpr':'dayslowprsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(dayslowpr, on=['pixel','date','year'])




###################################
###################################
######Statistics on Rolling Windows
###################################
###################################


'''
meanGDDsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDsum3'].mean()
stdGDDsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDsum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanGDDsum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdGDDsum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zGDDsum3'] = (dfMasterMetMonthly['GDDsum3'] - dfMasterMetMonthly['GDDsum3mean']) / dfMasterMetMonthly['GDDsum3std']
'''

meanGDDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDmean3'].mean()
stdGDDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDmean3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanGDDmean3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdGDDmean3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zGDDmean3'] = (dfMasterMetMonthly['GDDmean3'] - dfMasterMetMonthly['GDDmean3mean']) / dfMasterMetMonthly['GDDmean3std']

meanVPDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['VPDmean3'].mean()
stdVPDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['VPDmean3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanVPDmean3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdVPDmean3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zVPDmean3'] = (dfMasterMetMonthly['VPDmean3'] - dfMasterMetMonthly['VPDmean3mean']) / dfMasterMetMonthly['VPDmean3std']


meandayslowprsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['dayslowprsum3'].mean()
stddayslowprsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['dayslowprsum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meandayslowprsum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stddayslowprsum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zdayslowprsum3'] = (dfMasterMetMonthly['dayslowprsum3'] - dfMasterMetMonthly['dayslowprsum3mean']) / dfMasterMetMonthly['dayslowprsum3std']

meanprsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['prsum3'].mean()
stdprsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['prsum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanprsum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdprsum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zprsum3'] = (dfMasterMetMonthly['prsum3'] - dfMasterMetMonthly['prsum3mean']) / dfMasterMetMonthly['prsum3std']






##########################
##########################
#TESTING ALGORITHMS
##########################
##########################


plt.scatter(dfMasterMetMonthly.zGDDsum3,dfMasterMetMonthly.GDDsum3_y, s=.1)
plt.scatter(dfMasterMetMonthly.zGDDsum3,dfMasterMetMonthly.GDD, s=.1)
plt.scatter(dfMasterMetMonthly.GDD,dfMasterMetMonthly.GDDsum3_y, s=.1)

plt.scatter(dfMasterMetMonthly.GDDsum3,dfMasterMetMonthly.avgtempabovefiveandhalf, s=.1) #FUCK
plt.scatter(dfMasterMetMonthly.prsum3,dfMasterMetMonthly.pr, s=.1) 
plt.scatter(dfMasterMetMonthly.VPDsum3,dfMasterMetMonthly.vpd, s=.1) #Should I average over all vpds for sum3??

plt.scatter(dfMasterMetMonthly.VPDsum3,dfMasterMetMonthly.VPDmean3, s=.1)
plt.scatter(dfMasterMetMonthly.zGDDsum3,dfMasterMetMonthly.zGDDmean3, s=.1)

one=None
one = dfMasterMetMonthly.loc[(dfMasterMetMonthly['pixel']==69981.0) & (dfMasterMetMonthly.year==2015) ]
plt.plot(one.month, one.avgtempabovefiveandhalf)
plt.plot(one.month, one.GDDsum3)

plt.scatter(one.avgtemp, one.avgtempabovefiveandhalf, s=.1)



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

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import glob
import scipy.stats as st
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

#Will need to manipulate because month is not a column yet   
#It is now succcckkkkaaaa!!!

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
mean_cum_daysinarow_lowpr = dfMasterMetMonthly.groupby(['pixel', 'month'])['dayslowpr'].mean()
std_cum_daysinarow_lowpr = dfMasterMetMonthly.groupby(['pixel', 'month'])['dayslowpr'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(mean_cum_daysinarow_lowpr, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(std_cum_daysinarow_lowpr, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zCumDaysLowPr'] = (dfMasterMetMonthly['dayslowpr'] - dfMasterMetMonthly['dayslowprmean']) / dfMasterMetMonthly['dayslowprstd'] 

print "Calculating GDD statistics...."
meanavgtemp = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDm'].mean()
stdavgtemp = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDm'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanavgtemp, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdavgtemp, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zGDDm'] = (dfMasterMetMonthly['GDDm'] - dfMasterMetMonthly['GDDmmean']) / dfMasterMetMonthly['GDDmstd'] 



print "writing meteorological anomalies to drive..."
#dfMasterMet[['date', 'month', 'anomalyVPD', 'anomalyPrecip']].to_csv(data_path + "Meteor_" + crop_id + ".csv")
dfMasterMetMonthly.to_csv(data_path + "MeteorMonthly_roll_" + crop_id + ".csv")
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

