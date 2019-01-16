#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:11:01 2018

@author: brian
"""

#AGU GRAPHS
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as st

df = maxdf_labeled

plt.scatter(topo,quant, c=array)
plt.title('SOM grid selection')
plt.xlabel('Topographic Error')
plt.ylabel('Quantization Error')
plt.colorbar(label='grid size (nxn)')
#plt.xlim(-.01,0.2)
#plt.ylim(0,.4)






def mappingfunction(pr=False,drydays=False,vpd=False,ndvi=False,gdd=False,cluster=False,month=False, map_provided=False):
    '''mapping average climate, etc variables across western us. need map predefined.'''
    if not map_provided:
        map = Basemap(projection='stere', lon_0=-106, lat_0=90.,\
                llcrnrlat=40,urcrnrlat=49.5,\
                llcrnrlon=-119,urcrnrlon=-93.5,\
                rsphere=6371200., resolution='l', area_thresh=10000)
    x, y = map(df.Longitude.values,df.Latitude.values)
    x, y = np.array(x), np.array(y)
    map.drawcoastlines(linewidth=1),map.drawstates(),map.drawcountries(linewidth=1.1)
    map.drawmeridians(range(-140, -80, 5), linewidth=.3),map.drawparallels(range(20, 60, 5),linewidth=.3)
    
    if drydays:
        map.scatter(x,y,c=df.drydayssum3mean,s=.9)
        plt.clim(10,30)
        plt.colorbar(label='dry days',fraction=0.025,ticks=[10,15,20,25,30,35,40])#, pad=0.04)
        plt.figure(figsize=(80,40), dpi=1000)
        #plt.colorbar(fraction=.03)
    
    if pr:
        map.scatter(x,y,c=df.prsum3mean, s=.9)
        plt.clim(50,275)
        plt.colorbar(label='precipitation',fraction=.025,shrink=.8, ticks=[50,100,150,200,250])
        plt.figure(figsize=(80,40), dpi=1000)
        
    if vpd:
        map.scatter(x,y,c=df.VPDmean3mean, s=.9)
        plt.clim()
        plt.colorbar(label='VPD',fraction=.025,shrink=.8)
        plt.figure(figsize=(80,40), dpi=1000)
        
    if ndvi:
        map.scatter(x,y,c=df.NDVIsum3mean, s=.9)
        plt.clim(.7,2)
        plt.colorbar(label = 'NDVI', ticks=[.8,1.2,1.6,2], fraction=.025,shrink=.8)
        plt.figure(figsize=(10,5), dpi=300)
        
    if gdd:
        map.scatter(x,y,c=df.GDDmean3mean, s=.9)
        plt.clim(450,650)
        plt.colorbar(label='GDD',ticks=[450,500,550,600,650], fraction=.025,shrink=.8)
        plt.figure(figsize=(80,40), dpi=1000)

        
    if cluster:
        map.scatter(x,y,c=df.cluster, cmap=plt.cm.get_cmap('viridis',pd.unique(df.cluster).size),s=.9)
        plt.colorbar(ticks=pd.unique(df.cluster), label='cluster', fraction=.025, shrink=.8)
        plt.figure(figsize=(10,5), dpi=1000)
    if month:
        map.scatter(x,y,c=df.month_median,cmap=plt.cm.get_cmap('viridis',pd.unique(df.month_median).size),s=.9)
        plt.colorbar(label='month',fraction=.025,shrink=.8,ticks=pd.unique(df.month_median))
        plt.figure(figsize=(10,5), dpi=300)
    else:
        print 'please provide what variable you would like plotted'
    plt.show()
    
    
    
    
mappingfunction(ndvi=True)

####################################
######### MAP SLOPES
####################################

def mappingfunction_slope(slope=False, map_provided=False):#,pr=False,drydays=False,vpd=False,ndvi=False,gdd=False,cluster=False,month=False, map_provided=False):
    '''mapping average climate, etc variables across western us. need map predefined.'''
    if not map_provided:
        map = Basemap(projection='stere', lon_0=-106, lat_0=90.,\
                llcrnrlat=40,urcrnrlat=49.5,\
                llcrnrlon=-119,urcrnrlon=-93.5,\
                rsphere=6371200., resolution='l', area_thresh=10000)

    map.drawcoastlines(linewidth=1),map.drawstates(),map.drawcountries(linewidth=1.1)
    map.drawmeridians(range(-140, -80, 5), linewidth=.3),map.drawparallels(range(20, 60, 5),linewidth=.3)
    from scipy import stats
    if slope:
         for i in range(pd.unique(df.cluster).size):
             dfc = df[df['cluster']==i]
             x, y = map(dfc.Longitude.values,dfc.Latitude.values)
             x, y = np.array(x), np.array(y)
           
             mask = ~np.isnan(dfc['zVPDmean3']) & ~np.isnan(dfc['zNDVIsum3'])  #mask out nan values
             stats_object = stats.linregress(dfc['zVPDmean3'].values[mask], dfc.zNDVIsum3.values[mask])
             print stats_object[0]
             c = stats_object[0]*mask
             print c.size
             #map.scatter(x*mask,y*mask,c=c)#, cmap=plt.cm.get_cmap('viridis',pd.unique(df.cluster).size),s=.9)
             plt.scatter(x*mask,y*mask,c=c)
             
             #plt.colorbar(ticks=pd.unique(df.cluster), label='cluster', fraction=.025, shrink=.8)
          
             #plt.figure(figsize=(80,40), dpi=1000)
    plt.colorbar(fraction=.025, shrink=.8)
    plt.figure()

mappingfunction_slope(slope=True)





var = ['cluster','month','ndvi','pr','vpd','gdd','drydays']

for i in var:
    print i
    mappingfunction(i)
    
    
    
    
    
    
    
    
    
    
############# 
# CORRELATION r VALUES

    #VPD
for cluster in range(pd.unique(df.cluster).size):    
    print 'zVPD r values for cluster ', cluster
    print np.round(df.loc[(df.cluster==cluster)].zVPDmean3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    #print 'VPD r values for cluster ', cluster
    #print np.round(df.loc[(df.cluster==cluster)].VPDmean3.corr(df.loc[(df.cluster==cluster)].NDVIsum3),2)
    
    #PRECIP
for cluster in range(pd.unique(df.cluster).size):    
    print 'zP r values for cluster ', cluster
    print np.round(df.loc[(df.cluster==cluster)].zPrsum3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    #CUMALATIVE
for cluster in range(pd.unique(df.cluster).size):  
    print np.round(df.loc[(df.cluster==cluster)].zdrydayssum3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    #AVGTEMP
for cluster in range(pd.unique(df.cluster).size):   
    print np.round(df.loc[(df.cluster==cluster)].zGDDmean3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    
    
    
    
    
##############################
###########################
# CORRELATION GRAPHS
############################
###############################
    #### COLORS!!!!!!!
##################################
color_array = ['red','gray','orange','brown','green']
color_array = ['xkcd:purple','xkcd:blue','xkcd:teal','xkcd:yellowish green','xkcd:bright yellow']
color_array = [(67, 24, 76),(50, 59, 130),(45, 147, 141),(95, 211, 101),(236, 239, 23)] # too light... 

#color_array = map(lambda x: x/256., color_array)
colors = []
for i in color_array:
    color = [x/256. for x in i]
    colors.append(tuple(color))
    
color_array = colors

for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVPDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.5})
    g = g.plot_joint(plt.scatter, color=color_array[cluster], s=15, edgecolor='black')
    g = g.plot_marginals(sns.distplot, kde=False, color=color_array[cluster])
    g = g.annotate(st.pearsonr, fontsize='medium')
    datapath = '../AGU/Images/'
    g.savefig(datapath + 'zVPD_zNDVI_' + str(cluster) + '.svg', transparent=False, dpi=300, bbox_inches='tight')

for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrsum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.5})
    g = g.plot_joint(plt.scatter, color=color_array[cluster], s=15, edgecolor=None)
    g = g.plot_marginals(sns.distplot, kde=False, color=color_array[cluster])
    g = g.annotate(st.pearsonr, fontsize='small')
    datapath = '../AGU/Images/'
    g.savefig(datapath + 'zP_zNDVI_' + str(cluster) + '.svg', transparent=False, dpi=1000, bbox_inches='tight')

for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zdrydayssum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.5})
    
    g = g.plot_joint(plt.scatter, color=color_array[cluster], s=15, edgecolor=None)
    g = g.plot_marginals(sns.distplot, kde=False, color=color_array[cluster])
    g = g.annotate(st.pearsonr, fontsize='small')
    datapath = '../AGU/Images/'
    g.savefig(datapath + 'zdrydays_zNDVI_' + str(cluster) + '.svg', transparent=False, dpi=300, bbox_inches='tight')
    
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zGDDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.5})
    g = g.plot_joint(plt.scatter, color=color_array[cluster], s=15, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=False, color=color_array[cluster])
    g = g.annotate(st.pearsonr, fontsize='xx-small')
    datapath = '../AGU/Images/'
    g.savefig(datapath + 'zGDD_zNDVI_' + str(cluster) + '.svg', transparent=False, dpi=300, bbox_inches='tight')
    

    
    
#######
# slopes
########
from scipy import stats
mask = ~np.isnan(df.zVPDmean3.values) & ~np.isnan(df.zNDVIsum3.values)
slope, intercept, r_value, p_value, std_err = stats.linregress(df.zVPDmean3.values[mask], df.zNDVIsum3.values[mask])
#################
# slopes
##############
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]
    #if zVPDmean3:
    mask = ~np.isnan(dfc.zVPDmean3.values) & ~np.isnan(dfc.zNDVIsum3.values)
    slope = stats.linregress(dfc.zVPDmean3.values[mask], dfc.zNDVIsum3.values[mask])[0]
    print np.round(slope,2)
    
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]
    #if zVPDmean3:
    mask = ~np.isnan(dfc.zPrsum3.values) & ~np.isnan(dfc.zNDVIsum3.values)
    slope = stats.linregress(dfc.zPrsum3.values[mask], dfc.zNDVIsum3.values[mask])[0]
    print np.round(slope,2)       
    
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]
    #if zVPDmean3:
    mask = ~np.isnan(dfc.zdrydayssum3.values) & ~np.isnan(dfc.zNDVIsum3.values)
    slope = stats.linregress(dfc.zdrydayssum3.values[mask], dfc.zNDVIsum3.values[mask])[0]
    print np.round(slope,2)    
    
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]
    #if zVPDmean3:
    mask = ~np.isnan(dfc.zGDDmean3.values) & ~np.isnan(dfc.zNDVIsum3.values)
    slope = stats.linregress(dfc.zGDDmean3.values[mask], dfc.zNDVIsum3.values[mask])[0]
    print np.round(slope,2)  

    
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]
    #if zVPDmean3:
    mask = ~np.isnan(dfc.zsradsum3.values) & ~np.isnan(dfc.zNDVIsum3.values)
    slope = stats.linregress(dfc.zsradsum3.values[mask], dfc.zNDVIsum3.values[mask])[0]
    print np.round(slope,2)

)

#########################################
#########################################
#########################################
#########################################
#BINNED
#########################################    
#########################################
#########################################
# Goal to make plots NDVI by vpd etc
# binned by other variables, values?
import statsmodels.api as sm

lowess = sm.nonparametric.lowess
###### y first, x second as input########
####### returns (x,y) !!!!!!! 


for lower in range(0,450,50):
    upper = lower + 50
    binned = df[(df.prsum3>lower) & (df.prsum3<upper)]
    if binned.prsum3.size < 50 & lower > 300:
        upper = lower + 150
        binned = df[(df.prsum3>lower) & (df.prsum3<upper)]
        
    
    #plt.scatter(binned.zVPDmean3, binned.zNDVIsum3, s=.9)
    #plt.show()
    
    w = lowess(binned.NDVIsum3,binned.zVPDmean3, frac=2./3)
    plt.plot(w[:,0],w[:,1], 'g')
    plt.scatter(binned.zVPDmean3, binned.NDVIsum3,s=.1)
    plt.title(str(lower) + ' < P > ' + str(upper))
    plt.xlabel('vpd')
    plt.ylabel('NDVI')
    plt.show()
    
    #g = sns.jointplot(binned.zVPDmean3, binned.zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    #g = g.plot_joint(plt.scatter, color='b', s=15, edgecolor='white')
    #g = g.plot_marginals(sns.distplot, kde=False, color='b')
    #g = g.annotate(st.pearsonr, fontsize='xx-small')
    
    
array = np.arange(.6,2.2,.2)
for lower in array:
    upper = lower + .2
    
    binned = df[(df.VPDmean3>lower) & (df.VPDmean3<upper)]
    
    g = sns.jointplot(binned.zPrsum3, binned.zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    g = g.plot_joint(plt.scatter, color='b', s=15, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=False, color='b')
    g = g.annotate(st.pearsonr, fontsize='xx-small')
    
    
    
    
    
    
#####################################################
    ############################################
    ####################COUNTOUR###################
    ###########################################
###################################################
    
    
plt.scatter(df.zVPDmean3, df.zPrsum3, c=df.zNDVIsum3, s=1.8)
plt.colorbar(label='zNDVIsum3')
plt.xlabel('zVPDmean3')
plt.ylabel('zPrsum3')

plt.scatter(df.zGDDmean3, df.zPrsum3, c=df.zNDVIsum3, s=.3)
plt.colorbar(label='zNDVIsum3')
plt.xlabel('zGDDmean3')
plt.ylabel('zPrsum3')

plt.scatter(df.zdrydayssum3, df.zPrsum3, c=df.zNDVIsum3, s=.5)
plt.colorbar(label='zNDVIsum3')
plt.xlabel('zdrydayssum3')
plt.ylabel('zPrsum3')


plt.scatter(df.VPDmean3mean, df.prsum3mean, c=df.cluster, s=.9)
plt.colorbar(label='zNDVIsum3')
plt.xlabel('VPDmean3')
plt.ylabel('Prsum3')

columns = [str(VPDmean3mean),str(prsum3mean)]#,GDDmean3mean,drydayssum3]

for i in columns:
    plt.scatter(df.columns[i], df.columns[i+1], c=df.cluster, s=.9)
    plt.colorbar(label='zNDVIsum3')
    plt.xlabel(columns[i])
    plt.ylabel(columns[i+1])
    

df.plot.hexbin(x='VPDmean3',y='prsum3', C='NDVIsum3', gridsize=100)
#############################################
##############################################
############## lowess #######################
##########################################
############################################

import statsmodels.api as sm
lowess = sm.nonparametric.lowess
###### y first, x second as input########
####### returns (x,y) !!!!!!! 

w = lowess(df.zNDVIsum3,df.zVPDmean3, frac=2./3)
plt.plot(w[:,0],w[:,1], 'g')
plt.scatter(df.zVPDmean3, df.zNDVIsum3,s=.1)
plt.xlabel('vpd')
plt.ylabel('NDVI')
plt.show()



     #################
     ########## LOWESS seaborn
    ###
# non lowess    
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster == cluster]
    g = sns.jointplot(dfc.zVPDmean3, dfc.zNDVIsum3, kind='reg', scatter_kws={'s':.5})
    g = g.plot_joint(plt.scatter, color=color_array[cluster], s=15, edgecolor='black')
    g = g.plot_marginals(sns.distplot, kde=False, color=color_array[cluster])
    g = g.annotate(st.pearsonr, fontsize='medium')

#lowess
##### looping through Lowess
##########################


### multiple plots
fig, ax = plt.subplots(ncols=1, nrows=pd.unique(df.cluster).size,
                   figsize=(4.5,20), sharex=True,gridspec_kw={'wspace':.05,'hspace':.05},
                   dpi=200)
 
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]        
    w = lowess(dfc.zNDVIsum3,dfc.zVPDmean3, frac=2./3)
    ax[cluster].plot(w[:,0],w[:,1], 'g', linewidth=5)
    ax[cluster].scatter(dfc.zVPDmean3, dfc.zNDVIsum3,s=1, color = 'b')
    #ax[cluster].tick_params(which='major', width=1.0)
    #ax[cluster].tick_params(which='major', length=1.0)
    #ax[cluster].tick_params(which='minor', width=1.0, labelsize=10)
    #ax[cluster].tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

    #plt.title('cluster: ' + str(cluster))
    #plt.xlabel('vpd')
    #plt.ylabel('NDVI')
    fig.tight_layout()
    fig.show()
    
fig, ax = plt.subplots(ncols=1, nrows=pd.unique(df.cluster).size,
                   figsize=(4.5,20), sharex=True,gridspec_kw={'wspace':.05,'hspace':.05},
                   dpi=200)
 
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]        
    w = lowess(dfc.zNDVIsum3,dfc.zPrsum3, frac=2./3)
    ax[cluster].plot(w[:,0],w[:,1], 'g', linewidth=5)
    ax[cluster].scatter(dfc.zPrsum3, dfc.zNDVIsum3,s=1, color = 'b')
    fig.tight_layout()
    fig.show()    
    
 
fig, ax = plt.subplots(ncols=1, nrows=pd.unique(df.cluster).size,
                   figsize=(4.5,20), sharex=True,gridspec_kw={'wspace':.05,'hspace':.05},
                   dpi=200)
 
for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]        
    w = lowess(dfc.zNDVIsum3,dfc.zdrydayssum3, frac=2./3)
    ax[cluster].plot(w[:,0],w[:,1], 'g', linewidth=5)
    ax[cluster].scatter(dfc.zdrydayssum3, dfc.zNDVIsum3,s=1, color = 'b')
    fig.tight_layout()
    fig.show()        

    
########## SINGLE

for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]        
    w = lowess(dfc.zNDVIsum3,dfc.zVPDmean3, frac=2./3)
    plt.plot(w[:,0],w[:,1], 'g', linewidth=7)
    plt.scatter(dfc.zVPDmean3, dfc.zNDVIsum3,s=2, color = 'b')
    #plt.title('cluster: ' + str(cluster))
    #plt.xlabel('zprec')
    #plt.ylabel('NDVI')
    plt.figure(figsize=(10,5),dpi=800)
    plt.show()




for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]        
    w = lowess(dfc.zNDVIsum3,dfc.zPrsum3, frac=2./3)
    plt.plot(w[:,0],w[:,1], 'g', linewidth=7)
    plt.scatter(dfc.zPrsum3, dfc.zNDVIsum3,s=2, color = 'b')
    #plt.title('cluster: ' + str(cluster))
    #plt.xlabel('zprec')
    #plt.ylabel('NDVI')
    plt.figure(figsize=(20,10),dpi=1000)
    plt.show()


for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster==cluster]        
    w = lowess(dfc.zNDVIsum3,dfc.zdrydayssum3, frac=2./3)
    plt.plot(w[:,0],w[:,1], 'green', linewidth=7)
    plt.scatter(dfc.zdrydayssum3, dfc.zNDVIsum3,s=2, color = 'b')
    #plt.title('cluster: ' + str(cluster))
    #plt.xlabel('zprec')
    #plt.ylabel('NDVI')
    plt.show()









for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster == cluster]
    sns.lmplot(x="zVPDmean3", y="zNDVIsum3", data=dfc,
           lowess=True, scatter_kws={'s':.2, 'linewidths':10,'c':green})    
    
sns.lmplot(x="zVPDmean3", y="zNDVIsum3", data=df, col ='cluster',
           lowess=True, scatter_kws={'s':.2, 'linewidths':10,})



for cluster in range(pd.unique(df.cluster).size):
    dfc = df[df.cluster == cluster]
    
    ax = sns.regplot(x='zVPDmean3', y='zNDVIsum3', data = dfc, 
                lowess=True, scatter_kws={'s':.2})
    ax = ax.
    plt.show()
    


g = sns.FacetGrid(df, col="cluster", sharex=True)
#g.map(plt.scatter, "zVPDmean3", "zNDVIsum3",alpha=.2)
g.map(sns.regplot(x='zVPDmean3', y='zNDVIsum3', data = dfc, 
                lowess=True, scatter_kws={'s':.2}))