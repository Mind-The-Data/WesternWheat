#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:31:27 2018

@author: brian
"""
import numpy as np
import statsmodels.api as sm

lowess = sm.nonparametric.lowess
###### y first, x second as input########
####### returns (x,y), wtf!!!!!!! 

for i in (pd.unique(df.cluster)):
    
    x = np.array(df[df.cluster==i].zVPDmean3)
    y = np.array(df[df.cluster==i].zNDVIsum3)
    w = lowess(y,x, frac=1./2)
    plt.scatter(w[:,0],w[:,1], s=.5)
    plt.scatter(x,y,s=.1)
    
    plt.show()

w = lowess(y, x, frac=5./6)
plt.scatter(w[:,0],w[:,1])
plt.scatter(x,y,s=.01)

w = lowess( df.maxNDVIsum3,df.VPDmean3mean, frac=1./3)
plt.plot(w[:,0],w[:,1])
plt.scatter(df.VPDmean3,df.maxNDVIsum3, s=.01)

w = lowess( df.NDVIsum3mean, df.prsum3mean, frac=1./3)
plt.plot(w[:,0],w[:,1], 'g')
plt.scatter(df.prsum3mean, df.NDVIsum3mean,s=.01)
#plt.legend(loc='upper right')
plt.xlabel('prsum3mean')
plt.ylabel('NDVIsum3mean')

######################################
######### ANOMALIES WITH MAX ONLY
#####################################

w = lowess( maxdf.zNDVIsum3, maxdf.zPrsum3, frac=1./2)
plt.plot(w[:,0],w[:,1], 'g')
plt.scatter(maxdf.zPrsum3, maxdf.zNDVIsum3,s=.1)
plt.xlabel('maxdf.zPrsum3')
plt.ylabel('maxdf.zNDVIsum3')



w = lowess(maxdf.zNDVIsum3,maxdf.zVPDmean3, frac=1./6)
plt.plot(w[:,0],w[:,1], 'g')
plt.scatter(maxdf.zVPDmean3, maxdf.zNDVIsum3,s=.1)
plt.xlabel('maxdf.zVPDmean3')
plt.ylabel('maxdf.zNDVIsum3')



w = lowess( maxdf.zNDVIsum3, maxdf.zdrydayssum3,frac=1./2)
plt.plot(w[:,0],w[:,1], 'g')
plt.scatter(maxdf.zdrydayssum3, maxdf.zNDVIsum3,s=.1)
plt.xlabel('maxdf.zdrydayssum3')
plt.ylabel('maxdf.zNDVIsum3')


w = lowess(maxdf.zNDVIsum3,maxdf.zGDDmean3,  frac=1./2)
plt.plot(w[:,0],w[:,1], 'g')
plt.scatter(maxdf.zGDDmean3, maxdf.zNDVIsum3,s=.1)
plt.xlabel('maxdf.zGDDmean3')
plt.ylabel('maxdf.zNDVIsum3')








########### Trying to locate visible cluster in prsum3mean and NDVIsum3mean space

plt.scatter(df.loc[(df.prsum3mean<100)].Longitude, df.loc[(df.prsum3mean<100)].Latitude)

def mappingfunction():
                
                x, y = map(df.loc[(df.prsum3mean<100)].Longitude.values, df.loc[(df.prsum3mean<100)].Latitude.values)
                x, y = np.array(x), np.array(y)
                map.drawcoastlines(linewidth=1)
                map.drawstates()
                map.drawcountries(linewidth=1.1)
                map.drawmeridians(range(-140, -80, 5), linewidth=.3)
                map.drawparallels(range(20, 60, 5),linewidth=.3)
                
                #map.scatter(x,y, c = dfc.NDVIsum3std, s=.5)
                #plt.clim(0,.6)
                
                map.scatter(x,y, c = df.loc[(df.prsum3mean<100)].maxNDVIsum3, s=.5)
                #plt.clim(.55,2)
                
                #map.scatter(x,y, c = df.loc[(df.prsum3mean<100)].VPDmean3mean, s=.5)
                #plt.clim(.7,1.7)
                #plt.clim(.8,2.2)
                #map.scatter(x,y, c = dfc.VPDmean3std, s=.5)
                #plt.clim(.1,.35)
                
               # map.scatter(x,y, c = df.loc[(df.prsum3mean<100)].prsum3mean, s=.5)
                #map.scatter(x,y, c = dfc.prsum3, s=.05)
                #plt.clim(25,350)
                #map.scatter(x,y, c = dfc.prsum3std, s=.5)
                #plt.clim(20,100) #prsum3std
                
                #map.scatter(x,y, c = df.loc[(df.prsum3mean<100)].month, s=.5)
                plt.colorbar()
                
                #datapath = '/home/brian/Documents/westernWheat/Images/Fall2018/4featureSOM/Clusters_Locations/AllYears/NDVIsum3std/'
                #plt.savefig(datapath + str(crop_id) + "_Cluster:" + str(c) + "colorbar_.png", dpi=500)
                plt.show()

mappingfunction()





