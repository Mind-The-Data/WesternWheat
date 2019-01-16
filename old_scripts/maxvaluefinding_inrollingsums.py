#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:53:39 2018

@author: brian
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'


met = None
ndvi = None
dfMaster =None        
met = pd.read_csv(data_path + 'MeteorMonthly_roll_' + crop_id + ".csv")
ndvi = pd.read_csv(data_path + 'VegInd4_' + crop_id + ".csv")
met.date = pd.to_datetime(met.date).astype(str)
dfMaster = ndvi.merge(met, on = ['pixel', 'date', 'year','month'])
dfMaster.drop(columns=['Unnamed: 0_x'],inplace=True)
# changing met.date so to an object in 12-06-1 format so fits with ndvi.date for a successful merge
# dfMaster is now 75,000 rows instead of 120,000 simply because ndvi is date for all months where weather only 4-9

df=dfMaster
plt.scatter(df.zNDVIsum3,df.zVPDmean3, s=.01)

#df.groupby(['pixel', 'year']).zNDVIsum3 #argmax() throws error suggests using .apply method

maxNDVIsum3s = df.groupby(['pixel', 'year']).NDVIsum3.apply(np.max) #Outputs a series with multi index
maxNDVIsum3s = maxNDVIsum3s.reset_index(level=[0,1]) # now df with range index and pixel, year, and value to merge with
maxNDVIsum3s.rename(columns={'NDVIsum3':'maxNDVIsum3'}, inplace=True)

df = df.merge(maxNDVIsum3s,how='outer')  # This gives a repeat of maxNDVIsum3 values over all instances that are pixel and year..
#NDVI = df[['NDVIsum3','maxNDVIsum3']]  

df['maxNDVIbool']= np.where(df['NDVIsum3']==df['maxNDVIsum3'], True, False)

#############################################
#alter NDVIsum3 column in lieu of below method of selecting based on bool
#############################################
# different selection criterian
for pixel in df:
    if groupby pixel, 
df.groupby(['pixel']).maxNDVIbool


# select the subset where maxNDVIbool is True
maxdf = df.loc[(df.maxNDVIbool==True)]

maxdf.month.value_counts()

maxdf.month.plot(kind='hist')
maxdf.NDVI.plot(kind='hist',bins=30)
maxdf.NDVIsum3.plot(kind='hist',bins=30)

#############################################################
#############################################################
# Cleaning df from second-crop / weeds  / really green late in the season
# USE USDA Crop harvest dates as cut offs for certain latitudes...
#############################################################
#############################################################




#############################################################
#############################################################
#################### Spatial temporal ndvi z-score column addition.
#################### across both space and time
#############################################################
#############################################################


maxdf['zNDVIsum3spacetime'] = (maxdf.NDVIsum3 - maxdf.NDVIsum3.mean())/maxdf.NDVIsum3.std()


plt.scatter(maxdf.prsum3, maxdf.zNDVIsum3spacetime,s=.1)
plt.xlabel('maxdf.prsum3')
plt.ylabel('maxdf.zNDVIsum3spacetime')

plt.scatter(maxdf.VPDmean3, maxdf.zNDVIsum3spacetime,s=.1)
plt.xlabel('maxdf.VPDmean3')
plt.ylabel('maxdf.zNDVIsum3spacetime')

plt.scatter(maxdf.zVPDmean3, maxdf.zNDVIsum3spacetime,s=.1)
plt.xlabel('maxdf.zVPDmean3')
plt.ylabel('maxdf.zNDVIsum3spacetime')


########################################################
########3 VISUALIZE MAXDF SELECTING GROWING SEASON
######################################################


years = np.arange(2008,2017)

for year in years:
    for i in range(1,7):
        plt.subplot(2,3,i)
        plt.plot(df[(df.pixel==216500.0) & (df.year == year)].month, df[(df.pixel==216500.0) & (df.year == year)].NDVI)
        #plt.show()
 for year in years:

        plt.plot(df[(df.pixel==216500.0) & (df.year == year)].month, df[(df.pixel==216500.0) & (df.year == year)].vpd, label=year)
        plt.ylabel('vpd')  
        plt.xlabel('month')
        plt.legend()
        
        
plt.scatter(df[(df.pixel==216500.0) & (df.maxNDVIbool == True)].VPDmean3mean, df[(df.pixel==216500.0) & (df.maxNDVIbool == True)].NDVIsum3mean)