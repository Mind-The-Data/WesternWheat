#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:02:14 2018

@author: brian
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap


# import maxNDVI df
df = df #maxNDVI df

######### histograms 

plt.hist(df.month)

plt.hist(df[df.maxNDVIbool==True].month)
plt.xlabel('month')
        
plt.hist(df[df.maxNDVIbool==True].NDVI)
plt.xlabel('NDVI')

X = NDVIsum3mean
plt.hist(df[df.maxNDVIbool==True].X, bins=30)
plt.xlabel(str(X))


plt.hist(df[df.maxNDVIbool==True].NDVIsum3mean, bins=30)
plt.xlabel('NDVIsum3mean')



############## HISTOGRAM of months by latitude bins ############       
         
for i in range(35, 55):
    print 'Latitude < ',i, '& > ', i-1
    south = maxdf.loc[(df.Latitude<float(i)) & (maxdf.Latitude>float((i - 1)))]
    plt.hist(south.month)
    plt.show()


    
######################################
###### BASEMAP #######
######################################
######################################
    
map = Basemap(projection='stere', lon_0=-105, lat_0=90.,\
            llcrnrlat=29,urcrnrlat=49,\
            llcrnrlon=-117,urcrnrlon=-87.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)   
    
def mappingfunction():


                
                x, y = map(maxdf.Longitude.values, maxdf.Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates(linewidth=.8)
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)

                map.scatter(x,y, c = maxdf.month, s=.5)
                plt.colorbar()
                
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018//'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "colorbar_.png", dpi=500)
                plt.show()

mappingfunction()




def mappingfunction():

    for c in range(pd.unique(df.cluster).size):   
                print 'cluster: ', c
                dfc = df[df.cluster==c]
                
                x, y = map(dfc.Longitude.values, dfc.Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates(linewidth=.8)
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)

                map.scatter(x,y, c = dfc.month, s=.5)
                plt.colorbar()
                
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018//'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "colorbar_.png", dpi=500)
                plt.show()

mappingfunction()

#################################################
#################################################
################# TIME SERIES ###################
#################################################
#################################################
# Import non max df
#####################
####### Locate the maxNDVIbool months that are 9 often
south = df.loc[(df.Latitude<35) & (df.Longitude<-102)]
south = maxdf



counter = 0
for p in pd.unique(south.pixel.values):
    counter +=1
    if counter < 20:
        #plt.plot(south[south.pixel==p].fractionofyear, south[south.pixel==p].NDVI, label='NDVI')
        plt.scatter(south[south.pixel==p].month, south[south.pixel==p].NDVI, label='NDVI')
        #plt.plot(south[south.pixel==p].fractionofyear, south[south.pixel==p].NDVIsum3, label='sum3')
        plt.show()


# pulled in df from maxvaluefinding_rollingsums.py, now i have maxndvisum3 column appended
south = df.loc[(df.Latitude<35) & (df.Longitude<-102)]       
counter = 0
for p in pd.unique(south.pixel.values):
    counter +=1
    if counter < 3:
        for j in range(2008,2017):
        
            plt.plot(south[south.pixel==p].fractionofyear, south[south.pixel==p].NDVI, label='NDVI')
            #plt.scatter(south[south.pixel==p].month, south[south.pixel==p].NDVI, label='NDVI')
            plt.plot(south.loc[(south.pixel==p) & (south.year == j)].month, south.loc[(south.pixel==p) & (south.year == j)].NDVI, label='NDVI')
            #plt.plot(south[south.pixel==p].fractionofyear, south[south.pixel==p].NDVIsum3, label='sum3')
            plt.show()
            
counter =0
for p in pd.unique(maxdf.pixel.values):
    counter +=1
    if counter < 30:
        print p
        #plt.plot(south[south.pixel==p].fractionofyear, south[south.pixel==p].NDVI, label='NDVI')
        plt.scatter(maxdf[maxdf.pixel==p].month, maxdf[maxdf.pixel==p].NDVI, label='NDVI')
        #plt.plot(south[south.pixel==p].fractionofyear, south[south.pixel==p].NDVIsum3, label='sum3')
        plt.show()