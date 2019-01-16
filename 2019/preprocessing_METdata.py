#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:54:04 2018

@author: brian
"""
import matplotlib.pyplot as plt 
#import numpy as np
#import glob
#import scipy.stats as st
import pandas as pd
from functions import index_to_datetime
from functions import add_days_in_a_row, add_days_in_a_row_vpd

##################################
############## MET ###############
##################################
crop_id = 23
crop_id = str(crop_id)
data_path = '../../Data/AgMet/'


##
# Pre processesing original meteorological data
##
print "Loading Meteorological data..." + crop_id + '\n'
dfMasterMet = pd.read_csv(data_path + 'dailymeteo' + crop_id + 'allyear.csv')
print 'Met data loaded\n'
dfMasterMet.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)
dfMasterMet.index = dfMasterMet['pixel']
###
# more points
##
print "Loading extra points met data..." + crop_id + '\n'
data_path = '../../Data/AgMet2/'
dfMasterMet2 = pd.read_csv(data_path + 'dailymeteo' + crop_id + 'allyearmorepoints.csv')
print 'Met data loaded \n'
dfMasterMet2.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)
dfMasterMet2.index = dfMasterMet2['pixel']
####
# extended met data
###
print "Loading extended met data..." + crop_id + '\n'
data_path = '../../Data/AgMet/'
dfextended = pd.read_csv(data_path + 'dailymeteo' + crop_id + 'allyear_1997_2006.csv')
print 'Met data loaded \n'
dfextended.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)
dfextended.index = dfextended['pixel']
#####
#### extended for more points
####
print "Loading extended met data for more points...\n" + crop_id + '\n'
data_path = '../../Data/AgMet2/'
dfextended2 = pd.read_csv(data_path + 'dailymeteo' + crop_id + 'allyearmorepoints_1997_2006.csv')
print 'Met data loaded \n'
dfextended2.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)
dfextended2.index = dfextended2['pixel']

#############
## optional strip for faster debugging
### TURN OFF!!!!!!!!
#############

print 'selecting subset for faster debugging\n'
size = 10000.
dfMasterMet = dfMasterMet[(dfMasterMet.pixel<size)]
dfMasterMet2 = dfMasterMet2[(dfMasterMet2.pixel<size)]
dfextended = dfextended[(dfextended.pixel<size)]
dfextended2 = dfextended2[(dfextended2.pixel<size)]

##############################
###############################
########## inserting datetime
##############################
##############################

print 'applying index_to_datetime function...\n'
dfMasterMet['date'] = dfMasterMet['system:index'].apply(index_to_datetime)
dfMasterMet.index = dfMasterMet['date']
dfMasterMet['month'] = pd.DatetimeIndex(dfMasterMet['date']).month
dfMasterMet['year'] = pd.DatetimeIndex(dfMasterMet['date']).year
# more points
print 'index to datetime on more points...\n'
dfMasterMet2['date'] = dfMasterMet2['system:index'].apply(index_to_datetime)
dfMasterMet2.index = dfMasterMet2['date']
dfMasterMet2['month'] = pd.DatetimeIndex(dfMasterMet2['date']).month
dfMasterMet2['year'] = pd.DatetimeIndex(dfMasterMet2['date']).year
# extended
print 'index to datetime on extended meterology...\n'
dfextended['date'] = dfextended['system:index'].apply(index_to_datetime)
dfextended.index = dfextended['date']
dfextended['month'] = pd.DatetimeIndex(dfextended['date']).month
dfextended['year'] = pd.DatetimeIndex(dfextended['date']).year
# more points extended
print 'index to datetime on more points extended met...\n'
dfextended2['date'] = dfextended2['system:index'].apply(index_to_datetime)
dfextended2.index = dfextended2['date']
dfextended2['month'] = pd.DatetimeIndex(dfextended2['date']).month
dfextended2['year'] = pd.DatetimeIndex(dfextended2['date']).year

####################
####################
###### combine
####################
###################
print 'combining...\n'

dfMasterMet = pd.concat([dfMasterMet, dfMasterMet2, dfextended,\
                         dfextended2], axis=0, join='outer')
# Clear
dfMasterMet2, dfextended, dfextended2 = None, None, None



# Low pr days in a row function
# changed threshold to tenth of an inch
# calling function which inserts new column in dfMasterMet
print '...Calculating days in a row with low precipitation...\
this may take a few minutes and gigabytes of RAM...\n'
dfMasterMet = (dfMasterMet.groupby(['pixel', 'year'])).apply(add_days_in_a_row)
#converting temperature from kelvin to celsius simply a subtraction 
print 'coverting kelvin to celsius...'
dfMasterMet.tmmn = dfMasterMet.tmmn.transform(lambda x: x - 273.15)
dfMasterMet.tmmx = dfMasterMet.tmmx.transform(lambda x: x- 273.15)
print 'selecting days above threshold temp...'
#dfMasterMet['daysabove28']=dfMasterMet.tmmx>28
#dfMasterMet['daysabove30']=dfMasterMet.tmmx>30
#dfMasterMet['daysabove35']=dfMasterMet.tmmx>35
dfMasterMet['daysbelowneg5']=dfMasterMet.tmmn<-5
dfMasterMet['tmmnbelowneg5']=dfMasterMet[dfMasterMet.tmmn<-5].tmmn
#dfMasterMet['daysabove_avg5.5']=((dfMasterMet.tmmn + dfMasterMet.tmmx)/2)>5.5
dfMasterMet['avgtemp']= (dfMasterMet.tmmn + dfMasterMet.tmmx)/2
#dfMasterMet['avgtempabovefiveandhalf']= dfMasterMet.loc[(dfMasterMet.avgtemp>5.5)].avgtemp
dfMasterMet['GDD']= dfMasterMet.loc[(dfMasterMet.avgtemp>5.5) & (dfMasterMet.tmmx< 30)].avgtemp
dfMasterMet['EDD']=dfMasterMet[dfMasterMet.tmmx>34].tmmx
dfMasterMet['tmmxplus']=dfMasterMet[dfMasterMet.tmmx>30].tmmx
dfMasterMet['vpdplus']=dfMasterMet[dfMasterMet.vpd>1].vpd
dfMasterMet['prplus']=dfMasterMet[dfMasterMet.pr>15].pr


####################################
####################################
#print "writing daily met to  harddrive..."
#dfMasterMet.to_csv(data_path + "Meteordaily_" + crop_id + ".csv")
##################################
##################################
#LOAD DAILY THEN CONVERT TO MONTHLY SAVES TIME
##########################################

#dfMasterMet = pd.read_csv(data_path + "Meteordaily_" + crop_id + ".csv")
#dfMasterMet.index = dfMasterMet.date
#dfMasterMet.index = pd.to_datetime(dfMasterMet.index)

print 'Daily  --->   Monthly....\n'
print 'aggregating daily into monthly values...this may take awhile\n'
# having trouble now carrying over columns that aren't aggregated from daily to monthly
# How to fix this???? You cant. you have to add them to the .agg
# u'system:index':'median', u'countyears':'median', u'cropland_mode',
#       u'fractionofyear', u'longitude' u'pixel', u'.geo', u'date', u'month'
dfMasterMetMonthly = None
dfMasterMet.index = dfMasterMet['date']

    
dfMasterMetMonthly = dfMasterMet.groupby(['pixel','year','month'])\
.agg({u'pr':'sum','eto':'sum','etr':'sum',\
'vpd': 'mean', 'srad': 'mean', 'tmmn':'mean', 'daysbelowneg5':'sum',\
'tmmx':'sum','drydays':'mean', 'avgtemp' : 'sum',\
 'GDD':'sum','EDD':'sum','tmmxplus':'sum', 'vpdplus':'sum', 'prplus':'sum',\
 'tmmnbelowneg5':'sum'})

 
#.agg({'cum_daysinarow_lowpr':['sum', 'max']})index
#Have to sum (or mean) days above temp because max will either be 1 or 0
# TO INVESTIGATE
#df = dfMasterMet.groupby('pixel').resample('M')\
#.agg({u'year':'mean',u'month':'mean'})

# tuned cumdayslowpr to mean in lieu of max, to reduce range of values, was 0-300 for sum3...     
#converting index to date, then type datetimeindex, then dropping the day
#code created by Brian, but now doesn't work, and marco's idea works... weird... 
#dfMasterMetMonthly.date = dfMasterMetMonthly.index
#dfMasterMetMonthly = dfMasterMetMonthly.to_timestamp()
#dfMasterMetMonthly = dfMasterMetMonthly.to_period('M')




#####################
# reset multiindex   
####################

dfMasterMetMonthly = dfMasterMetMonthly.reset_index() #pops out of last as muliindex need date index
#dfMasterMetMonthly.index = pd.DatetimeIndex(dfMasterMetMonthly.index)  
# This might need to be toggled, need to reset from H:M:S object 
dfMasterMetMonthly.year = dfMasterMetMonthly.year.astype(int)
dfMasterMetMonthly.month = dfMasterMetMonthly.month.astype(int)
dfMasterMetMonthly['date'] = pd.to_datetime((dfMasterMetMonthly.year * 100) +\
                  (dfMasterMetMonthly.month), format='%Y%m')
 # to a datetime index at Monthly frequency
dfMasterMetMonthly.index = dfMasterMetMonthly.date
dfMasterMetMonthly = dfMasterMetMonthly.to_period('M')
dfMasterMetMonthly.date = dfMasterMetMonthly.index # Important!!! turns back into object date at M freq
# this helps merge with GDD date column, so they are the same dtype
#dfMasterMetMonthly.date = dfMasterMetMonthly.date.astype(str)  
#part of the date index reseting want date index to be just year and month
#dfMasterMetMonthly.index = dfMasterMetMonthly.date 
# toggle on or off depending on df index type mostly off if coming from .agg daily to monthly


from functions import rolling_sum
from functions import statistics
from functions import rolling_mean

print 'rolling sum..\n'

columns = [ u'pr',u'eto', u'etr', u'tmmx',u'avgtemp', u'GDD'\
           ,u'EDD', u'tmmxplus', u'vpdplus','prplus',u'daysbelowneg5', 'tmmnbelowneg5']

columns_mean = ['vpd','srad','drydays','tmmn']
df = dfMasterMetMonthly

for w in range(2,7):
    df = rolling_sum(df, columns, w=w )
    df = rolling_mean(df, columns_mean, w=w)
    
    
    
#df = rolling_sum(dfMasterMetMonthly, columns, w=3 )
#df = rolling_sum(df, columns_mean, w=3)

print 'statistics...\n'
#df = dfMasterMetMonthly
columns = ['pr',               u'etr',              u'tmmx',\
           u'daysbelowneg5',            u'prplus',              u'srad',\
                 u'avgtemp',     u'tmmnbelowneg5',              u'tmmn',\
                     u'vpd',          u'tmmxplus',               u'GDD',\
                 u'drydays',           u'vpdplus',               u'EDD',\
                     u'eto',                                 u'prsum2',\
                 u'etosum2',           u'etrsum2',          u'tmmxsum2',\
             u'avgtempsum2',           u'GDDsum2',           u'EDDsum2',\
            u'tmmxplussum2',       u'vpdplussum2',        u'prplussum2',\
       u'daysbelowneg5sum2', u'tmmnbelowneg5sum2',          u'vpdmean2',\
               u'sradmean2',      u'drydaysmean2',         u'tmmnmean2',\
                  u'prsum3',           u'etosum3',           u'etrsum3',\
                u'tmmxsum3',       u'avgtempsum3',           u'GDDsum3',\
                 u'EDDsum3',      u'tmmxplussum3',       u'vpdplussum3',\
              u'prplussum3', u'daysbelowneg5sum3', u'tmmnbelowneg5sum3',\
                u'vpdmean3',         u'sradmean3',      u'drydaysmean3',\
               u'tmmnmean3',            u'prsum4',           u'etosum4',\
                 u'etrsum4',          u'tmmxsum4',       u'avgtempsum4',\
                 u'GDDsum4',           u'EDDsum4',      u'tmmxplussum4',\
             u'vpdplussum4',        u'prplussum4', u'daysbelowneg5sum4',\
       u'tmmnbelowneg5sum4',          u'vpdmean4',         u'sradmean4',\
            u'drydaysmean4',         u'tmmnmean4',            u'prsum5',\
                 u'etosum5',           u'etrsum5',          u'tmmxsum5',\
             u'avgtempsum5',           u'GDDsum5',           u'EDDsum5',\
            u'tmmxplussum5',       u'vpdplussum5',        u'prplussum5',\
       u'daysbelowneg5sum5', u'tmmnbelowneg5sum5',          u'vpdmean5',\
               u'sradmean5',      u'drydaysmean5',         u'tmmnmean5',\
                  u'prsum6',           u'etosum6',           u'etrsum6',\
                u'tmmxsum6',       u'avgtempsum6',           u'GDDsum6',\
                 u'EDDsum6',      u'tmmxplussum6',       u'vpdplussum6',\
              u'prplussum6', u'daysbelowneg5sum6', u'tmmnbelowneg5sum6',\
                u'vpdmean6',         u'sradmean6',      u'drydaysmean6',\
               u'tmmnmean6']
    
df = statistics(df, columns)


print "writing meteorological anomalies to drive..."
data_path = '../../Data/Processed/'
#dfMasterMet[['date', 'month', 'anomalyVPD', 'anomalyPrecip']].to_csv(data_path + "Meteor_" + crop_id + ".csv")
#df.to_csv(data_path + "MeteorMonthly_" + crop_id + ".csv")
#to reduce RAM usage
#dfMasterMet = None
#dfMasterMetMonthly = None










##### OLD SCRIPT ##############




###########################
### Rolling Statistics ###
#########################
##### the old way
######################


'''
w=3  #window period
print "Calculating rolling Window statistics for: " + str(w) +" months"


GDD=None
GDD = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).GDD.mean()
GDD=GDD.reset_index(level=[0,1,2])
GDD.rename(index=str,columns={'GDD':'GDDmean3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(GDD, on=['pixel','year', 'date'])
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
PR.rename(index=str,columns={'pr':'prsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(PR, on=['pixel','date','year'])


dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
dayslowpr=None
dayslowpr = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).drydays.sum()
dayslowpr= dayslowpr.reset_index(level=[0,1,2])
dayslowpr.rename(index=str,columns={'drydays':'drydayssum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(dayslowpr, on=['pixel','date','year'])

#############
##### SRAD
########
dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
column=None
column = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).srad.sum()
column= column.reset_index(level=[0,1,2])
column.rename(index=str,columns={'srad':'sradsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(column, on=['pixel','date','year'])

#############
##### tmmx
########
dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
column=None
column = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).tmmx.sum()
column= column.reset_index(level=[0,1,2])
column.rename(index=str,columns={'tmmx':'tmmxsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(column, on=['pixel','date','year'])

#############
##### SRAD
########
dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
column=None
column = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).srad.sum()
column= column.reset_index(level=[0,1,2])
column.rename(index=str,columns={'srad':'sradsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(column, on=['pixel','date','year'])

#############
##### EDD
########
dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
column=None
column = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).srad.sum()
column= column.reset_index(level=[0,1,2])
column.rename(index=str,columns={'srad':'sradsum3'},inplace=True)
dfMasterMetMonthly = dfMasterMetMonthly.merge(column, on=['pixel','date','year'])
'''

########
#
#######
#dfMasterMetMonthly.index = dfMasterMetMonthly.date #pops out of last series as a range index, need date index
#column=None
#column = dfMasterMetMonthly.groupby(['pixel','year']).rolling(w).srad.sum()
#column= column.reset_index(level=[0,1,2])
#column.rename(index=str,columns={'srad':'sradsum3'},inplace=True)
#dfMasterMetMonthly = dfMasterMetMonthly.merge(column, on=['pixel','date','year'])
'''
meansradsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['sradsum3'].mean()
stdsradsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['sradsum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meansradsum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdsradsum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zsradsum3'] =\
 (dfMasterMetMonthly['sradsum3'] - dfMasterMetMonthly['sradsum3mean']) / dfMasterMetMonthly['sradsum3std']
'''




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
dfMasterMetMonthly['zGDDsum3'] = \
(dfMasterMetMonthly['GDDsum3'] - dfMasterMetMonthly['GDDsum3mean']) / dfMasterMetMonthly['GDDsum3std']



meansradsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['sradsum3'].mean()
stdsradsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['sradsum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meansradsum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdsradsum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zsradsum3'] = \
(dfMasterMetMonthly['sradsum3'] - dfMasterMetMonthly['sradsum3mean']) / dfMasterMetMonthly['sradsum3std']


meanGDDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDmean3'].mean()
stdGDDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDDmean3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanGDDmean3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdGDDmean3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zGDDmean3'] = \
(dfMasterMetMonthly['GDDmean3'] - dfMasterMetMonthly['GDDmean3mean']) / dfMasterMetMonthly['GDDmean3std']


meandrydayssum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['drydayssum3'].mean()
stddrydayssum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['drydayssum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meandrydayssum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stddrydayssum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zdrydayssum3'] = \
(dfMasterMetMonthly['drydayssum3'] - dfMasterMetMonthly['drydayssum3mean']) / dfMasterMetMonthly['drydayssum3std']

meanprsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['prsum3'].mean()
stdprsum3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['prsum3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanprsum3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdprsum3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zPrsum3'] = \
(dfMasterMetMonthly['prsum3'] - dfMasterMetMonthly['prsum3mean']) / dfMasterMetMonthly['prsum3std']

meanVPDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['VPDmean3'].mean()
stdVPDmean3 = dfMasterMetMonthly.groupby(['pixel', 'month'])['VPDmean3'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanVPDmean3, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdVPDmean3, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zVPDmean3'] = \
(dfMasterMetMonthly['VPDmean3'] - dfMasterMetMonthly['VPDmean3mean']) / dfMasterMetMonthly['VPDmean3std']
'''




##########################
##########################
#TESTING ALGORITHMS
##########################
##########################
'''

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

'''

#Why do we do this? to speed up zscore calculation i assume
###is not working currently....
#print 'values converted to np.nan:' + dfMasterMetMonthly[dfMasterMetMonthly == 0].count()
#dfMasterMetMonthly[dfMasterMetMonthly == 0] = np.nan


"""
print "Calculating monthly meterological anomalies...." + crop_id

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

#Will need to manipulate because month is not a column yet   
#It is now succcckkkkaaaa!!!
'''
print "Calculating VPD statistics...."
meanVPD = dfMasterMetMonthly.groupby(['pixel', 'month'])['vpd'].mean()
stdVPD = dfMasterMetMonthly.groupby(['pixel', 'month'])['vpd'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanVPD, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdVPD, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zVPD'] = \
(dfMasterMetMonthly['vpd'] - dfMasterMetMonthly['vpdmean']) / dfMasterMetMonthly['vpdstd']
#
print "Calculating Precip statistics...."
meanP = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].mean()
stdP = dfMasterMetMonthly.groupby(['pixel', 'month'])['pr'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanP, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdP, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zP'] = \
(dfMasterMetMonthly['pr'] - dfMasterMetMonthly['prmean']) / dfMasterMetMonthly['prstd'] 

print "Calculating Cumulative days with low pr statistics...."
mean_cum_daysinarow_lowpr = dfMasterMetMonthly.groupby(['pixel', 'month'])['dayslowpr'].mean()
std_cum_daysinarow_lowpr = dfMasterMetMonthly.groupby(['pixel', 'month'])['dayslowpr'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(mean_cum_daysinarow_lowpr, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(std_cum_daysinarow_lowpr, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zcumdayslowr'] = \
(dfMasterMetMonthly['dayslowpr'] - dfMasterMetMonthly['dayslowprmean']) / dfMasterMetMonthly['dayslowprstd'] 

print "Calculating GDD statistics...."
meanavgtemp = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDD'].mean()
stdavgtemp = dfMasterMetMonthly.groupby(['pixel', 'month'])['GDD'].std()
dfMasterMetMonthly = dfMasterMetMonthly.join(meanavgtemp, on=['pixel', 'month'], rsuffix='mean')
dfMasterMetMonthly = dfMasterMetMonthly.join(stdavgtemp, on=['pixel', 'month'], rsuffix='std')
dfMasterMetMonthly['zGDD'] = \
(dfMasterMetMonthly['GDD'] - dfMasterMetMonthly['GDDmean']) / dfMasterMetMonthly['GDDstd'] 
'''



 
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

