#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datapath = '..'
dfMet23 = pd.read_csv(datapath + '/rawData/AgMet/monthlymeteo23.csv')

dfMet23 = dfMet23.rename(columns={'vpd':'VPD'})
dfMet23['month'] = ((dfMet23['fractionofyear'] - dfMet23['fractionofyear'].astype(int))*12).round().astype(int)



dfMet23.index = dfMet23['system:indexviejuno']

meanVPD = dfMet23.groupby(['system:indexviejuno', 'month'])['VPD'].mean()
stdVPD = dfMet23.groupby(['system:indexviejuno', 'month'])['VPD'].std()
#meanP = dfMet23.groupby(['system:indexviejuno', 'month'])['P'].mean()
#stdP = dfMet23.groupby(['system:indexviejuno', 'month'])['P'].std() 



#adding mean, std, anomaly VPD to dfMet23
dfMet23 = dfMet23.join(meanVPD, on=['system:indexviejuno', 'month'], rsuffix='mean')
dfMet23 = dfMet23.join(stdVPD, on=['system:indexviejuno', 'month'], rsuffix='std')
dfMet23['VPDanomaly'] = (dfMet23['VPD'] - dfMet23['VPDmean'])/dfMet23['VPDstd']
##calculating anomaly with vpd instaneous - monthlymean is not quite what I want.What do i want?


#adding mean, std, anomaly SM to dfMet23
#dfMet23 = dfMet23.join(meanpr, on=['system:indexviejuno', 'month'], rsuffix='mean')
#dfMet23 = dfMet23.join(stdpr, on=['system:indexviejuno', 'month'], rsuffix='std')
#dfMet23['pranomaly'] = (dfMet23['pr'] - dfMet23['prmean'])/dfMet23['prstd']



 


id_pixel = pd.unique(dfMet23.index)
p = dfMet23[dfMet23.index==id_pixel[0]]
p50 = dfMet23[dfMet23.index==id_pixel[50]]
p500 = dfMet23[dfMet23.index==id_pixel[500]]
p1000 = dfMet23[dfMet23.index==id_pixel[1000]]
p1500 = dfMet23[dfMet23.index==id_pixel[1500]]

plt.figure(figsize=(20,10))
#plt.plot(p['fractionofyear'], p['VPD'])

#plt.plot(p['fractionofyear'],p['VPDanomaly'])
#plt.plot(p50['fractionofyear'],p50['VPDanomaly'])
#plt.plot(p500['fractionofyear'],p500['VPDanomaly'])
#plt.scatter(p1000['fractionofyear'],p1000['VPDanomaly'])
#plt.scatter(p1500['fractionofyear'],p1500['VPDanomaly'])

dfMet23
