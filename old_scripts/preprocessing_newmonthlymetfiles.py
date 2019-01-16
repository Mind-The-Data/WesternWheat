#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:52:38 2018

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

crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

dfMasterMet = pd.DataFrame()
print "loading monthly met data...." + crop_id
for f in glob.glob(data_path + 'monthlymeteo' + crop_id + '*'):
    dfMasterMet = pd.concat((dfMasterMet,  pd.read_csv(f)))
print 'Met data loaded'

print 'Throwing out non wheat (' + crop_id + ') pixels...a total of: ' + \
    str(dfMasterMet.loc[dfMasterMet['cropland_mode'] != float(crop_id)].cropland_mode.count())
    
dfMasterMet = dfMasterMet.loc[dfMasterMet['cropland_mode'] == float(crop_id)]
#######
#PROBLEM: SYSTEM:INDEX  does not contain date anymore....
dfMasterMet['system:index']
# SOLUTION == Marco's Date Magic
# Only in dailymet files did we use the system:index to extract date... 


#Marco's Date Magic
dfMasterMet['month'] = ((dfMasterMet['fractionofyear'] - dfMasterMet['fractionofyear'].astype(int))*12).round().astype(int)
dfMasterMet['year'] = (np.modf(dfMasterMet['fractionofyear'])[1]).astype(int)
dfMasterMet['date'] = pd.to_datetime(dfMasterMet.year * 10000 + (dfMasterMet.month+1)*100+1, format='%Y%m%d')
dfMasterMet.index = dfMasterMet['date']
dfMasterMet = dfMasterMet.to_period('M')
#lkupTable = dfCDL.pivot('year', 'system:indexviejuno', 'first')
#dfMasterMet['CDL'] = lkupTable.lookup(dfMasterMet['year'], dfMasterMet['system:indexviejuno'])

'''
dfMasterMet.index = dfMasterMet['system:indexviejuno']
print 'applying index_to_datetime function...'
dfMasterMet['date'] = dfMasterMet['system:index'].apply(index_to_datetime)
dfMasterMet.index = dfMasterMet['date']
dfMasterMet['month'] = pd.DatetimeIndex(dfMasterMet['date']).month
'''