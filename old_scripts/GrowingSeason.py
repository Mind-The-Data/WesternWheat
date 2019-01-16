#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:22:51 2018

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
#plt.scatter(df.zNDVIsum3,df.zVPDmean3, s=.01)

#df.groupby(['pixel', 'year']).zNDVIsum3 #argmax() throws error suggests using .apply method

maxNDVIsum3s = df.groupby(['pixel', 'year']).NDVIsum3.apply(np.max) #Outputs a series with multi index
maxNDVIsum3s = maxNDVIsum3s.reset_index(level=[0,1]) # now df with range index and pixel, year, and value to merge with
maxNDVIsum3s.rename(columns={'NDVIsum3':'maxNDVIsum3'}, inplace=True)
#max_avgNDVIsum3s = maxNDVIsum3s.groupby(['pixel']).maxNDVIsum3.apply(np.mean) # average maximum maxNDVIsum3... not quite what I want.

df = df.merge(maxNDVIsum3s,how='outer')  # This gives a repeat of maxNDVIsum3 values over all instances that are pixel and year..
#NDVI = df[['NDVIsum3','maxNDVIsum3']]  

df['maxNDVIbool']= np.where(df['NDVIsum3']==df['maxNDVIsum3'], True, False)

#df['median_maxNDVI_month'] = df[df.maxNDVIbool==True].groupby('pixel').month.apply(np.median) # does not work, only size 508 instead of 2000

median_maxNDVI_month = df[df.maxNDVIbool==True].groupby('pixel').month.apply(np.median) # outputs pixel and median month column
median_maxNDVI_month = median_maxNDVI_month.apply(np.round) # round so that month matches median motnh and don't throw away pixels that
# aren't lined up perfectly. Could do somthing better here in the future potentially. like average other columns for 7.5 median month
# grouped by pixel we find the median month across all years for that pixel that has a true maxndvibool
#median_maxNDVI_month = median_maxNDVI_month.to_frame() # convert to df for easy outer merge below
#median_maxNDVI_month.rename(columns={'month':'median_month'}, inplace = True) # rename month column so it is differentiable
#median_maxNDVI_month =  median_maxNDVI_month.set_index()
#df = df.merge(median_maxNDVI_month, how='outer') # This gives union of both dfs. so that there is a repeat of month column in median months
df=df.join(median_maxNDVI_month,how='outer',on='pixel',rsuffix='_median') # combines series/df.columns with df, union, on pixel index level

#df_stripped = df[df.month_median == df.month]
df_stripped = df[np.isclose(df.month_median,df.month)]

# after AGU maybe I play with this so that I am averaging the months slightly better.... 5.5 - 8.5 instead of simply 6-9 say
#pd.unique(df_stripped.pixel).size

##########################################
##########################################
# Investigating fidelity of above selection
##########################################
##########################################

for year in range(2007,2017):
    print df_stripped[(df.year==year) & (df.pixel==317)].month

######### histograms 

plt.hist(df_stripped.month)

plt.hist(df[df.maxNDVIbool==True].month, bins=30)
plt.xlabel('month')

plt.hist(df.month_median)


######################################
###### BASEMAP #######
######################################
######################################
from mpl_toolkits.basemap import Basemap
    

    
map = Basemap(projection='stere', lon_0=-105, lat_0=90.,\
            llcrnrlat=29,urcrnrlat=49,\
            llcrnrlon=-117,urcrnrlon=-87.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)   
    
def mappingfunction():


                
                x, y = map(df_stripped.Longitude.values, df_stripped.Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates(linewidth=.8)
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)

                map.scatter(x,y, c = df_stripped.month, s=.5)
                plt.colorbar()
                
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018//'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "colorbar_.png", dpi=500)
                plt.show()

mappingfunction()
