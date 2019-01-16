#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:38:55 2018

@author: brian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.formula.api as sm

datapath = '../Data/Processed/'
df23June = pd.read_csv(datapath + 'df23June.csv')
df23June.index=df23June['system:indexviejuno']

###STATS#####

model = sm.ols(formula='NDVImean ~ VPDfractionofyearmean + Pfractionofyearmean', data = df23June).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()



###########################
## PLOTTING
###########################
FractionofYear = df23June['fractionofyear']
MeanNDVI = df23June['NDVImean']


####################
#specific pixel plots
#####################
###################
######indentification for specific spatial locations
##########
id_pixel = pd.unique(df23June.index)
p = df23June[df23June.index==id_pixel[0]]
p500 = df23June[df23June.index==id_pixel[500]]
p1000 = df23June[df23June.index==id_pixel[1000]]
p1500 = df23June[df23June.index==id_pixel[1500]]
p47883=df23June[df23June.index==47883.0]
#################
################
## plotting 
plt.figure(1, figsize=(20,10))
#plt.scatter(p['fractionofyear'], p['VPD'])
plt.plot(p['fractionofyear'], p['NDVIanomaly'], 'green')
plt.scatter(p['fractionofyear'],p['VPDanomaly']*-1, marker='x', color='r')
plt.scatter(p['fractionofyear'], p['Panomaly'], color='b')
#plt.plot(p['fractionofyear'], p['NDVImean'])
#plt.plot(p['fractionofyear'], p['NDVIstd'])

plt.figure(2, figsize=(20,10))
plt.plot(p500['fractionofyear'], p500['NDVIanomaly'], 'green')
plt.scatter(p500['fractionofyear'],p500['VPDanomaly']*-1, marker='x', color='r')
plt.scatter(p500['fractionofyear'], p500['Panomaly'], color='b')


plt.figure(3, figsize=(20,10))
plt.plot(p1000['fractionofyear'], p1000['NDVIanomaly'], 'green')
plt.scatter(p1000['fractionofyear'],p1000['VPDanomaly']*-1, marker='x', color='r')
plt.scatter(p1000['fractionofyear'], p1000['Panomaly'], color='b')

plt.figure(4, figsize=(20,10))
plt.plot(p1500['fractionofyear'], p1500['NDVIanomaly'], 'green')
plt.scatter(p1500['fractionofyear'],p1500['VPDanomaly']*-1, marker='x', color='r')
plt.scatter(p1500['fractionofyear'], p1500['Panomaly'], color='b')

plt.figure(5, figsize=(20,10))
plt.scatter(p47883['fractionofyear'], p47883['NDVIanomaly'], color='g', s=20)
plt.scatter(p47883['fractionofyear'],p47883['VPDanomaly'], marker='x', color='r')
plt.scatter(p47883['fractionofyear'], p47883['Panomaly'], color='b')
#plt.scatter(p47883['fractionofyear'], p47883['Pfractionofyearmean'], color='b', s=.1)
#plt.scatter(p47883['fractionofyear'], p47883['P'], color='b', s=.01)
plt.scatter(p47883['fractionofyear'], p47883['VPD'], marker='x', color='r', s=.01)
plt.scatter(p47883['fractionofyear'], p47883['VPDfractionofyearmean'], marker='o', color='r', s=.1)

model = sm.ols(formula=p['NDVImean'] ~ p['VPDfractionofyearmean'] + p['Pfractionofyearmean'], data = df23June).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()






######################
### Mean Plots
########################
plt.figure(1, figsize=(20,10))
plt.scatter(df23June['PoverAllyearsmean'], df23June['VPDoverAllyearsmean'], s=.001, color='b')
plt.xlabel('Mean over all years Precip')
plt.ylabel('Mean over all years VPD')
plt.title('Meterologic Mean Correlation June Spring Wheat each pixel avg over all years')
#plt.savefig('../Images/Graphs/JuneMeanVPDandP.png')

plt.figure(2, figsize=(20,10))
plt.scatter(df23June['VPDoverAllyearsmean'],df23June['NDVImean'], s=.001, marker='x', color='r')
plt.xlabel('VPD Mean over all years')
plt.ylabel('NDVI mean over all years')
plt.title('Mean June Spring Wheat each pixel avg over all years')
#plt.savefig('../Images/Graphs/JuneMeanVPDandNDVI.png')

plt.figure(3, figsize=(20,10))
plt.scatter(df23June['NDVImean'],df23June['PoverAllyearsmean'], s=.001, color='b')
plt.xlabel('Precip Mean over all years')
plt.ylabel('NDVI mean over all years')
plt.title('Mean June Spring Wheat each pixels averaged over all years')
#plt.savefig('../Images/Graphs/JuneMeanPandNDVI.png')

plt.figure(4, figsize=(10,5), dpi=300)
plt.scatter(df23June['VPDfractionofyearmean'], df23June['Pfractionofyearmean'], s=.0001, color='b')
plt.ylabel('Precipitation fraction of year mean')
plt.xlabel('VPD fraction of year mean')
plt.title('Spring Wheat June Meterological fraction of year mean')
#plt.savefig('../Images/JuneAcrossAllpixels/')

plt.figure(4, figsize=(10,5), dpi=300)
plt.scatter(df23June['Pfractionofyearmean'], df23June['NDVI'], s=.01, color='b')
plt.ylabel('NDVI fraction of year mean')
plt.xlabel('P fraction of year mean (mm/day)')
plt.title('Spring Wheat June NDVI to P fraction of year means')
#plt.savefig('../Images/JuneAcrossAllpixels/NDVIandPfoymean.png')

plt.figure(4, figsize=(10,5), dpi=300)
plt.scatter(df23June['VPDfractionofyearmean'], df23June['NDVI'], s=.01, color='b')
plt.ylabel('NDVI fraction of year mean')
plt.xlabel('VPD fraction of year mean (kPa)')
plt.title('Spring Wheat -June- fraction of year means')
#plt.savefig('../Images/JuneAcrossAllpixels/NDVIandVPDfoymean.png')


###################################
#anomalous plots
###################################

P = df23June['Panomaly']
V = df23June['VPDanomaly']
N = df23June['NDVIanomaly']

fig, ax=plt.subplots(figsize=(20,10))

ax.scatter(V,P, s=.02)
ax.set(xlabel='VPD anomaly', ylabel='Precipitation anomaly', title='June Meterological Correlation across Spring Wheat(23) pixels averaged over all years')
#fig.savefig("../Images/Graphs/PandVPDcorrJuneSpringWheat.png")

fig, ax=plt.subplots(figsize=(20,10))
ax.scatter(V,N, s=.02)
ax.set(xlabel='VPDanomaly', ylabel='NDVIanomaly', title='June across all Spring Wheat pixels, 2008-2017')
#fig.savefig("../Images/Graphs/JuneVPDandNDVIspringWheat23.png")


plt.figure(figsize=(20, 10))
plt.scatter(P,N, s=.025)
plt.ylabel('NDVIanomaly')
plt.xlabel('Panomaly')
plt.title('June Spring Wheat (23), all pixels 2008-2017')
#plt.savefig('../Images/Graphs/JuneSpringWheatNDVIandP.png')