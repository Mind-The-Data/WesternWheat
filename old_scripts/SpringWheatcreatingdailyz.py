#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:29:03 2018

@author: brian
"""
import numpy as np
import pandas as pd
import glob
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

data_path = '../Data/AgMet/'
dfMet23 = pd.read_csv(data_path + '/monthlymeteo23.csv')
# adding month column
dfMet23['month'] = ((dfMet23['fractionofyear'] - dfMet23['fractionofyear'].astype(int))*12).round().astype(int)
dfMet23['day'] = dfMet23['systemindex']
#selecting the 23 subset of the df
dfMet23 = dfMet23.loc[(dfMet23['cropland_mode'] == 23)]
# Rename column
dfMet23 = dfMet23.rename(columns={'system:indexviejuno':'pixel', 'system:index':'systemindex'})
# Change index to pixel
dfMet23.index = dfMet23['pixel']

###############VPD
###############
vpd_pdy = dfMet23.groupby('pixel', 'month) 

