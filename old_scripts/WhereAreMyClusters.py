#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:54:53 2018

@author: brian
"""
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import seaborn as sns


#crop_id = 24
# import df from SOM-2-statistics-data-visualization.py

df = maxdf_labeled

#How many points are where

print pd.unique(df.pixel).size
for c in range(pd.unique(df.cluster).size):
    print pd.unique(df[df.cluster==c].pixel).size

###### atempts at interesting cluster info
##############
#df.groupby('cluster').Latitude.describe()
#pixels = df.groupby('pixel').cluster.apply(scipy.stats.mode)
#pixels = pd.DataFrame(pixels)
#pixels.cluster.values

years = np.arange(2008,2018, step=1)
for i in years:    
    sns.relplot(y='Latitude', x='Longitude', hue='cluster', palette="seismic", s=20, legend = 'full', data=df[df.year==i])
    
for i in range(pd.unique(df.cluster).size):    
    dfi = df[df.cluster==i]
    for i in years:
        plt.scatter(dfi[dfi.year==i].Longitude,dfi[dfi.year==i].Latitude, s=5)
        plt.axis([-122,-95,41,50])
        plt.show()
        
####################
##################        
#BASEMAP
################

#map =Basemap(projection='stere', lon_0=-110, lat_0=90.,\
#            llcrnrlat=40,urcrnrlat=49,\
#            llcrnrlon=-122,urcrnrlon=-90,\
#            rsphere=6371200., resolution='l', area_thresh=10000)
        

map = Basemap(projection='stere', lon_0=-105, lat_0=90.,\
            llcrnrlat=29,urcrnrlat=49,\
            llcrnrlon=-117,urcrnrlon=-87.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)

map = Basemap(projection='stere', lon_0=-106, lat_0=90.,\
            llcrnrlat=40,urcrnrlat=49.5,\
            llcrnrlon=-119,urcrnrlon=-93.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)



def mappingfunction():
    x, y = map(df.Longitude.values,df.Latitude.values)
    x = np.array(x)
    y = np.array(y)
    map.drawcoastlines(linewidth=1)
    map.drawstates()
    map.drawcountries(linewidth=1.1)
    map.drawmeridians(range(-140, -80, 5), linewidth=.3)
    map.drawparallels(range(20, 60, 5),linewidth=.3)
    #map.drawrivers(linewidth=.1)
    
    #map.scatter(x[pvalues.zETO.values<0.05], y[pvalues.zETO<0.05], c=params.zETO.values[pvalues.zETO<.05], cmap='seismic', alpha=.8, s=.5)#.
    #map.scatter(x,y,c=df.prsum3mean, s=.5)
    #map.scatter(x,y,c=df.VPDmean3mean, s=.5)
    #map.scatter(x,y,c=df.NDVIsum3mean, s=.5)
    map.scatter(x,y,c=df.cluster, cmap=plt.cm.get_cmap('viridis',pd.unique(df.cluster).size),s=.3)
    plt.colorbar(ticks=pd.unique(df.cluster), label='cluster', fraction=.03)
    #plt.clim(-1,1)

mappingfunction()

def mappingfunction():
    '''Just NDVIsum3mean over all pixels and clusters and years'''
    x, y = map(df.Longitude.values,df.Latitude.values)
    x = np.array(x)
    y = np.array(y)
    map.drawcoastlines(linewidth=1)
    map.drawstates()
    map.drawcountries(linewidth=1.1)
    map.drawmeridians(range(-140, -80, 5), linewidth=.3)
    map.drawparallels(range(20, 60, 5),linewidth=.3)
    map.scatter(x,y,c=df.NDVIsum3mean, s=.3)
    plt.clim(.7,2)
    #plt.title(str(crop_id))
    #column = dayslowprsum3mean #VPDmean3mean, prsum3mean, GDDmean3mean
    #map.scatter(x,y,c=df.prsum3mean,s=.3)
    #map.scatter(x,y,c=df.VPDmean3mean,s=.3)
    #map.scatter(x,y,c=df.drydayssum3mean,s=.3)
    #plt.clim(10,30)
    #map.scatter(x,y,c=df.sradsum3mean, s =.3)
    #map.scatter(x,y,c=df.GDDmean3mean, s=.3)
    #plt.clim(10,75)
    plt.colorbar()
    #plt.colorbar(label='drydays3mean',fraction=0.03,ticks=[10,15,20,25,30,35,40])#, pad=0.04)
    
    #plt.colorbar(fraction=.03)
    

mappingfunction()


############ 
# BASEMAP ALTERED
#################
#emulate
"""
for i in range(pd.unique(df.cluster).size):    
    dfi = df[df.cluster==i]
    for i in years:
        plt.scatter(dfi[dfi.year==i].Longitude,dfi[dfi.year==i].Latitude, s=5)
"""

def mapping_function():
    years = np.arange(2008,2018, step=1)
    for c in range(pd.unique(df.cluster).size):   
        print 'cluster: ', c
        dfc = df[df.cluster==c]
        for i in years:
                print 'year: ', i
                x, y = map(dfc[dfc.year==i].Longitude.values, dfc[dfc.year==i].Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates()
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)
                
                map.scatter(x,y, c = dfc[dfc.year==i].NDVIsum3, s=.5)
                plt.clim(.55,2)
                #map.scatter(x,y, c = dfc[dfc.year==i].VPDmean3, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].VPDmean3mean, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].prsum3mean, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].prsum3, s=.5)
                #plt.clim(25,350)
                plt.colorbar()
                #plt.show()
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018/4featureSOM/Clusters_Locations/'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "_Year:" + str(i) + ".png", dpi=500)
                plt.show()

mapping_function()


def mapping_function():
    years = np.arange(2008,2018, step=1)
    for i in years:
                print 'year: ', i
                x, y = map(df[df.year==i].Longitude.values, df[df.year==i].Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates()
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)
                
                map.scatter(x,y, c = df[df.year==i].NDVIsum3, cmap='YlGn', s=.5)
                plt.title(str(i))
                plt.clim(.55,2.1)
                #map.scatter(x,y,c=df[df.year==i].NDWI1, cmap='YlGn', s=.5)
                #map.scatter(x,y, c = df[df.year==i].VPDmean3, s=.5)
                #map.scatter(x,y, c = df[df.year==i].VPDmean3mean, s=.5)
                #map.scatter(x,y, c = df[df.year==i].prsum3mean, s=.5)
                #map.scatter(x,y, c = df[df.year==i].prsum3, s=.5)
                #plt.clim(25,350)
                plt.colorbar(label='NDVIsum3')
                #plt.show()
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018/4featureSOM/Clusters_Locations/'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "_Year:" + str(i) + ".png", dpi=500)
                plt.show()

mapping_function()




###########################
################# all years
###########################

def mappingfunction():

    for c in range(pd.unique(df.cluster).size):   
                print 'cluster: ', c
                dfc = df[df.cluster==c]
                
                x, y = map(dfc.Longitude.values, dfc.Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates()
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)
                
                #map.scatter(x,y, c = dfc.NDVIsum3std, s=.5)
                #plt.clim(0,.6)
                
                map.scatter(x,y, c = dfc.NDVIsum3mean, s=.5)
                plt.clim(.55,2)
                #map.scatter(x,y, c = dfc.VPDmean3mean, s=.5)
                #plt.clim(.7,1.7)
                #plt.clim(.8,2.2)
                #map.scatter(x,y, c = dfc.VPDmean3std, s=.5)
                #plt.clim(.1,.35)
                
                #map.scatter(x,y, c = dfc.prsum3mean, s=.5)
                #map.scatter(x,y, c = dfc.prsum3, s=.05)
                #plt.clim(25,350)
                #map.scatter(x,y, c = dfc.prsum3std, s=.5)
                #plt.clim(20,100) #prsum3std
                
                
                
                plt.colorbar()
                
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018/4featureSOM/Clusters_Locations/AllYears/NDVIsum3std/'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "colorbar_.png", dpi=500)
                plt.show()

mappingfunction()

def mapping_function():
    ''' mapping each year and resulting cluster for pixel.'''
    years = np.arange(2008,2018, step=1)
    #for c in range(pd.unique(df.cluster).size):   
        #print 'cluster: ', c
        #dfc = df[df.cluster==c]
    dfc = df
    for i in years:
                print 'year: ', i
                x, y = map(dfc[dfc.year==i].Longitude.values, dfc[dfc.year==i].Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates()
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)
                
                #map.scatter(x,y, c = dfc[dfc.year==i].NDVIsum3, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].VPDmean3, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].VPDmean3mean, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].prsum3mean, s=.5)
                #map.scatter(x,y, c = dfc[dfc.year==i].prsum3, s=.5)
                map.scatter(x,y,c = dfc[dfc.year==i].cluster, s = .5)
                
                plt.colorbar()
                #plt.show()
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018/4featureSOM/Clusters_Locations/'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "_Year:" + str(i) + ".png", dpi=500)
                plt.show()
                print (i)

mapping_function()

#####################
##### kde by cluster
####################

##### PR

for c in range(pd.unique(df.cluster).size):   
    print 'cluster: ', c
    dfc = df[df.cluster==c]
    sns.kdeplot(dfc.NDVIsum3, dfc.prsum3)
    plt.show()

##### VPD
    
for c in range(pd.unique(df.cluster).size):   
    print 'cluster: ', c
    dfc = df[df.cluster==c]
    sns.kdeplot(dfc.NDVIsum3, dfc.VPDmean3)
    plt.show()    
    
    
    
    