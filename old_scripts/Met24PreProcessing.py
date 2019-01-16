#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:04:19 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datapath = '..'
dfMet24 = pd.read_csv(datapath + '/rawData/AgMet/monthlymeteo24.csv')

dfMet24 = dfMet24.rename(columns={'vpd':'VPD'})
dfMet24['month'] = ((dfMet24['fractionofyear'] - dfMet24['fractionofyear'].astype(int))*12).round().astype(int)


dfMet24.index = dfMet24['system:indexviejuno']
meanVPD = dfMet24.groupby(['system:indexviejuno', 'month'])['VPD'].mean()
stdVPD = dfMet24.groupby(['system:indexviejuno', 'month'])['VPD'].std()

dfMet24 = dfMet24.join(meanVPD, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMet24 = dfMet24.join(stdVPD, on=['system:indexviejuno', 'month'], rsuffix='std')
dfMet24['VPDanomaly'] = (dfMet24['VPD'] - dfMet24['VPDmean'])/dfMet24['VPDstd']

#print dfMet24['vpdstd']

##calculating anomaly with vpd instaneous - monthlymean is not quite what I want. 
#errors could accumulate due to night/daytime differences of instaneous for a paticular month


id_pixel = pd.unique(dfMet24.index)
p = dfMet24[dfMet24.index==id_pixel[0]]
p50 = dfMet24[dfMet24.index==id_pixel[50]]
p500 = dfMet24[dfMet24.index==id_pixel[500]]
p1000 = dfMet24[dfMet24.index==id_pixel[1000]]
p1500 = dfMet24[dfMet24.index==id_pixel[1500]]

plt.figure(figsize=(20,10))
plt.plot(p['fractionofyear'],p['VPDanomaly'])
plt.plot(p50['fractionofyear'],p50['VPDanomaly'])
plt.plot(p500['fractionofyear'],p500['VPDanomaly'])
plt.plot(p1000['fractionofyear'],p1000['VPDanomaly'])
plt.plot(p1500['fractionofyear'],p1500['VPDanomaly'])
plt.title('*24* VPD monthly anomalies')