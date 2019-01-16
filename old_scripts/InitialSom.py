#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:50:00 2018

@author: brian
"""
import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="whitegrid")

crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

## Loading Data
##
#def load_data():
    #met = pd.read_csv(data_path + 'MeteorMonthly_experiment_' + crop_id + ".csv")
    #ndvi = pd.read_csv(data_path + 'VegInd3_' + crop_id + ".csv")    
    #del ndvi['system:indexviejuno.1']
    #met.date = pd.to_datetime(met.date).astype(str)
    #return
    #dfMaster = ndvi.merge(met, on = ['system:indexviejuno', 'date'])
# changing met.date so to an object in 12-06-1 format so fits with ndvi.date for a successful merge
#ndvi['system:indexviejuno']=ndvi['system:indexviejuno.1']


#resetting index because default index is system:indexviejuno and we would rather have dates
#met = met.reset_index()
#ndvi = ndvi.reset_index()
#met.index = met['date']
#met.index = pd.to_datetime(met.index)
#met.index = met.index.astype(str)
#ndvi.index = ndvi['date']

#dfMaster = ndvi.merge(met)

dfMaster = pd.read_csv(data_path + 'dfMaster' + crop_id + '.csv')
###
#refining, dfMaster, selecting variables, making smaller. 
###

#df = dfMaster[['zP','zVPD','pr','vpd','Latitude','Longitude','cum_daysinarow_lowpr','tmmx','ztmmn','daysabove28','NDVI','zNDVI']]
#names = ['zP','zVPD','pr','vpd','Latitude','Longitude','cum_daysinarow_lowpr','tmmx','ztmmn','daysabove28','NDVI','zNDVI']
#df = df.loc[(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
df = dfMaster[['pr','vpd','cum_daysinarow_lowpr','avgtemp', 'NDVI']]
names = ['pr','vpd','cum_daysinarow_lowpr','avgtemp', 'NDVI']

###
#Selecting Months
###
#df = df.loc[(df.month > 4) & (df.month<8)]
#df = df.loc[(dfMaster['month']==4)]
df = df[(dfMaster['month']==4)]
#df = df.apply(pd.to_numeric,  errors='coerce')
#df = df.dropna(how='any')

#df = df.loc[(dfMaster['month']==4) | (dfMaster['month']==5)]
df = df.apply(pd.to_numeric,  errors='coerce')
df = df.dropna(how='any')

# Resetting index so it goes in order, since I only selected april, a standard index is all messed up april 2008 to april 2009 skips 
#a ton of index values, and this makes it impossible to combine cluster output index with df 
df.index
df = df.reset_index(level=0)
del df['index']
#names = ['zP', 'zVPD', 'pr' , 'vpd', 'Latitude', 'Longitude', 'cum_daysinarow_lowpr','tmmx','tmmn','daysabove28','NDVI','zNDVI']
#names = ['pr','vpd','cum_daysinarow_lowpr','tmmx','NDVI', 'Latitude', 'Longitude']



##Investigating
df.info()

som = None
# create the SOM network and train it. You can experiment with different normalizations and initializations
som = SOMFactory().build(df.values, mapsize=[50,50], normalization = 'var', initialization='pca', component_names=names)
som.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
topographic_error = som.calculate_topographic_error()
quantization_error = np.mean(som._bmu[1])
print "Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error)

from sompy.visualization.mapview import View2D
view2D  = View2D(4,4,"rand data",text_size=16)
view2D.show(som, col_sz=2, which_dim="all", desnormalize=True)

# U-matrix plot
from sompy.visualization.umatrix import UMatrixView

umat  = UMatrixView(width=10,height=10,title='U-matrix')
umat.show(som)


from sompy.visualization.hitmap import HitMapView
K=10
Kluster = som.cluster(K)
hits  = HitMapView(20,20,"K-Means Clustering",text_size=16)
a=hits.show(som)


#
som.cluster(n_clusters=K) 
#som.cluster() returns the k-means cluster labels for each neuron of the map, 
#but it is straightforward to retrieve the cluster labels for the whole training set, 
#by assigning them the label of the BMUs (best-matching units). You can can do for example:
#Make sure indices line up.... 
map_labels = som.cluster(n_clusters=K)
# som._bmu[0]
data_labels = np.array([map_labels[int(k)] for k in som._bmu[0]])
clusters = pd.Series(data_labels)
clusters = clusters.rename('cluster').to_frame()
#concat cluster column with small original som input df
df_labeled = None
df_labeled = pd.concat([df,clusters], axis=1)




# merging df_label with original large df

dfmonth = None
dfmonth= dfMaster.loc[(dfMaster['month']==4)]
#del dfmonth['level_0']
dfmonth = dfmonth.reset_index()

#df = dfMaster.loc[(dfMaster.month > 3) & (dfMaster.month<9)]
#df = df.reset_index()
#del df['level_0']

#df_labeled_master = pd.concat([df_labeled, dfApril], axis=1)
df_labled_master=None
df_labeled_master = pd.merge(df_labeled, dfmonth)

#df_labeled.info()
#df_labeled_master.info()