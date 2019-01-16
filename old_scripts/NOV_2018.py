#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:52:24 2018

@author: brian
"""

import pandas as pd
#import glob
#import scipy.stats as st


##################################
############## MET ###############
##################################
crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/newdata_NOV_2018/'


# Pre processes meteorological data
print "Loading Meteorological data..." + crop_id
dfMasterMet = pd.read_csv(data_path + 'dailyMETEOmeteo' + crop_id + '_typyearmorepoints.csv')
print 'Met data loaded'

dfMasterMet.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)

newdf = pd.read_csv(data_path + 'dataset_class' + crop_id + 'locationsmorepints.csv')
# This one doesn't help
newdf['system:index'].head()
pd.unique(newdf['system:index']).size

newLSdf = pd.read_csv(data_path + 'monthlyLSclass' + crop_id + '2007morepoints.csv')

newLSdf.columns
pd.unique(newLSdf['system:index']).size
newLSdf['system:index'].head()
newLSdf.info()

monthlymet = pd.read_csv(data_path + 'monthlymeteo' + crop_id + '_2007_2012morepoints.csv')
monthlymet.info()
monthlymet['system:index']
pd.unique(monthlymet['system:index']).size




#### this used to be the key met column for time
dfMasterMet['system:index']


#### This was the function for the previous df, which shows we were extracting date from system:index
def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date

### does not work on new system:index because it only has day value, not year and month and day. 
#print 'applying index_to_datetime function...'
#dfMasterMet['date'] = dfMasterMet['system:index'].apply(index_to_datetime)
    
dfMasterMet['system:index']

def index_to_systemindex(arg):
    systemindex = arg.split('_')[1]
    return systemindex

dfMasterMet['systeminfo'] = dfMasterMet['system:index'].apply(index_to_systemindex)


pd.unique(dfMasterMet.systeminfo).size
dfMasterMet.day.describe()

######## So we have unique spatial information. 
# and we have day of year
# but we still need year to define space and time uniquely
###########################################################
