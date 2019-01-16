#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:21:44 2018

@author: brian
"""

#Merging Month data for SOM input and data analysis

import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="whitegrid")

crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

## Loading Data
##
#def load_data():
    met = pd.read_csv(data_path + 'MeteorMonthly_experiment_' + crop_id + ".csv")
    ndvi = pd.read_csv(data_path + 'VegInd3_' + crop_id + ".csv")    
    del ndvi['system:indexviejuno.1']
    met.date = pd.to_datetime(met.date).astype(str)
    #return
    dfMaster = ndvi.merge(met, on = ['system:indexviejuno', 'date'])



dfApril = dfMaster.loc[(dfMaster['month']==4)]
dfMay = dfMaster.loc[(dfMaster['month']==5)]
dfJune = dfMaster.loc[(dfMaster['month']==6)]
dfJuly = dfMaster.loc[(dfMaster['month']==7)]

dfJune.info()

dfJune.rename(index=str, columns={'pr':'june_pr', 'vpd': 'june_vpd',  'avgtemp':'june_avgtemp'}, inplace=True)
dfMay.rename(index=str, columns={'pr':'may_pr', 'vpd': 'may_vpd', 'avgtemp':'may_avgtemp'}, inplace=True)


dfsubjune = dfJune[['system:indexviejuno', 'june_pr', 'june_vpd', 'june_avgtemp','year']]
dfsubmay = dfMay[['system:indexviejuno', 'may_pr', 'may_vpd', 'may_avgtemp','year']]


dfsubjune.info()
dfsubmay.info()

dfJulywithjune = dfsubjune.merge(dfJuly, on=['system:indexviejuno', 'year'])
dfJulywithjunemay = dfsubmay.merge(dfJulywithjune, on=['system:indexviejuno', 'year'])

dfJulywithjune.info()
dfJulywithjunemay.info()

dfJulywithjunemay['GDDmayjuly'] = dfJulywithjunemay.avgtemp + dfJulywithjunemay.june_avgtemp + dfJulywithjunemay.may_avgtemp
#key that defines unique rows
key = dfJulywithjunemay[['system:indexviejuno', 'year']]

dfJulywithjunemayindex = dfJulywithjunemay[['system:indexviejuno', 'year']]
index = dfJulywithjunemay[['system:indexviejuno', 'year']]
dfJulywithjunemay.index



df = dfJulywithjunemay[['pr','june_pr','may_pr','vpd','june_vpd','may_vpd','cum_daysinarow_lowpr','GDDmayjuly','avgtemp', 'NDVI']]