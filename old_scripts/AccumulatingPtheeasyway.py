#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:53:44 2018

@author: brian
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns



crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

met = pd.read_csv(data_path + 'MeteorMonthly_' + crop_id + ".csv", index_col='system:indexviejuno')
ndvi = pd.read_csv(data_path + 'VegInd' + crop_id + ".csv", index_col='system:indexviejuno')

#resetting index because default index is system:indexviejuno and we would rather have dates
met = met.reset_index()
ndvi = ndvi.reset_index()
met.index = met['date']
ndvi.index = ndvi['date']

# merge datasets to perform correlations
# merge combines by index and by columns that have the same name and merges values that line up
dfMaster = ndvi.merge(met)
# reset index to.... 
#dfMaster= dfMaster.resest_index()
dfMaster.index = dfMaster('system:indexviejuno', 'fractionofyear')['zP']
#Insert Interaction column

#rename column
dfMaster.rename(columns= {'zPrecip':'zP'}, inplace=True)

#Accumulate
dfMaster['zP6543']=dfMaster.groupby['system:indexviejuno','year']['zP'].apply(lambda x: .loc[(dfMaster.month==6 | dfMaster.month==5 |dfMaster.month==4 | dfMaster.month==3)])

#SummerDF
dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 2) & (dfMaster['month']< 9)]

