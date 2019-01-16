#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:09:43 2018

@author: brian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import scipy


datapath = '../Data/Processed/'
df23June23 = pd.read_csv(datapath + 'df23June23.csv')
df23June23.index=df23June23['system:indexviejuno']


###STATS#####

model = sm.ols(formula='zNDVI ~ zVPD + zP', data = df23June23).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()



#convenience
savepath = '../Images/JuneOnly23/'
fractionofYear = df23June23['fractionofyear']
meanNDVI = df23June23['NDVImean']
meanfoyNDVI = df23June23['NDVI']
meanVPD = df23June23['VPDoverAllyearsmean']
meanfoyVPD = df23June23['VPDfractionofyearmean']
meanP = df23June23['PoverAllyearsmean']
meanfoyP = df23June23['Pfractionofyearmean']
vPD = df23June23['VPD']
p = df23June23['P']
nDVIz = df23June23['zNDVI']
pz= df23June23['zP']
vPDz = df23June23['zVPD']
latitude = df23June23['Latitude']
longitude = df23June23['Longitude']

#correlations
(vPDz).corr(pz)


#Ploting Means

plt.figure(dpi=300)
plt.scatter(meanNDVI, meanP, s=.0005)
plt.xlabel('NDVI mean over all years')
plt.ylabel('Mean Precipitation over all years (mm/day)')
plt.title('Spring Wheat Precipitation and NDVI June averages')
#plt.savefig(savepath + '/JuneMeanPandNDVIonly23.png')

plt.figure(dpi=300)
plt.scatter(meanNDVI, meanVPD, s=.0005, color='r')
plt.xlabel('Mean NDVI over all years')
plt.ylabel('Mean VPD over all years')
plt.title('Spring Wheat VPD and NDVI June averages')
#plt.savefig(savepath + '/JuneMeanVPDandNDVIonly23.png')


plt.figure(dpi=300)
sns.kdeplot(meanVPD)
plt.title('VPD oay mean')
plt.ylabel('frequency')
plt.xlabel('Vapor Pressure Difference Over All years, (kPa)')
#plt.savefig(savepath + '/VPDoverallyearsrmean_only23.png')

plt.figure(dpi=300)
sns.kdeplot(meanfoyP)
plt.ylabel('frequency')
plt.xlabel('Precipitation Fraction of year Mean (mm/day)')
plt.title('Precipitation mean fraction of year')
#plt.savefig(savepath + '/PMeanFoY_only23.png')


plt.figure(dpi=300)
sns.kdeplot(meanNDVI, color='g')
plt.ylabel('frequency')
plt.xlabel('NDVI oay mean')
plt.title('NDVI mean (greeness)')
#plt.savefig(savepath + '/NDVI_oay_mean_only23.png')

#plotting anomalies

sns.kdeplot(nDVIz, color="g")
plt.ylabel('frequency')
#plt.savefig(savepath + 'NDVIz_only23', dpi=300, figsize=[5,5])
sns.kdeplot(pz, color="b")
#plt.savefig(savepath + '/Pz_only23', dpi=300, figsize=[5,5])
sns.kdeplot(vPDz, color="r")
#plt.savefig(savepath + '/N_P_VPDz_only23', dpi=300, figsize=[5,5])


#df23ndvi= df23ndvi[df23ndvi.index ==47883.0]   #selecting single pixel 47883