#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:56:05 2018

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

crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

met = pd.read_csv(data_path + 'MeteorMonthly_roll_' + crop_id + ".csv")
ndvi = pd.read_csv(data_path + 'VegInd4_' + crop_id + ".csv")
    
# mess with data types in individual ararys, for a clean merge
met.date = pd.to_datetime(met.date).astype(str)
met.month = met.month.astype(int)
met.year = met.year.astype(int)
# Merge
dfMaster = None
dfMaster = met.merge(ndvi, on = ['pixel', 'date','year','month'])
##########################################################
# Selecting subset of dataset that has maxNDVIsum3bool == True
##########################################################
df=dfMaster

'''
#df.groupby(['pixel', 'year_y']).zNDVIsum3 #argmax() throws error suggests using .apply method

maxNDVIsum3s = df.groupby(['pixel', 'year']).NDVIsum3.apply(np.max) #Outputs a series with multi index
maxNDVIsum3s = maxNDVIsum3s.reset_index(level=[0,1]) # now df with range index and pixel, year, and value to merge with
maxNDVIsum3s.rename(columns={'NDVIsum3':'maxNDVIsum3'}, inplace=True)
df = df.merge(maxNDVIsum3s,how='outer')  # This gives a repeat of maxNDVIsum3 values over all instances that are pixel and year..
#NDVI = df[['NDVIsum3','maxNDVIsum3']]
df['maxNDVIbool']= np.where(df['NDVIsum3']==df['maxNDVIsum3'], True, False)
maxdf = df.loc[(df.maxNDVIbool==True)]

'''
from functions import GrowingSeasonSelection
maxdf = GrowingSeasonSelection(dfMaster)





###
#refining, dfMaster, selecting variables, making smaller. 
###
#I PRESUME I DO NOT WANT TO BRING IN ZSCORES.... BECAUSE THE SOM TAKES THE SPATIAL/TEMPORAL ZSCORE AND I WANT TO DIFFERENTIATE BETWEEN AREAS

#df=None
#df = maxdf[['zPrsum3','zVPDmean3','zCumdlowprsum3','zGDDmean3', 'zNDVIsum3']]
#names = ['zPrsum3','zVPDmean3','zCumdlowprsum3','zGDDmean3', 'zNDVIsum3']
#names = ['zP', 'zVPD', 'pr' , 'vpd', 'Latitude', 'Longitude', 'cum_daysinarow_lowpr','tmmx','tmmn','daysabove28','NDVI','zNDVI']
#names = ['pr','vpd','cum_daysinarow_lowpr','tmmx','NDVI', 'Latitude', 'Longitude']


names = ['prsum3mean','VPDmean3mean','GDDmean3mean','drydayssum3mean']
df = maxdf[names]#,'NDVIsum3']]#'GDDmean3mean','dayslowprsum3mean']]  #'NDVIsum3

df = df.apply(pd.to_numeric,  errors='coerce')
df = df.dropna(how='any')
# Resetting index so it goes in order, since I only selected april, a standard index is all messed up april 2008 to april 2009 skips 
#a ton of index values, and this makes it impossible to combine cluster output index with df 
df.index
df = df.reset_index(level=0)
del df['index']


##Investigating
df.info()

som = None
# create the SOM network and train it. You can experiment with different normalizations and initializations
som = SOMFactory().build(df.values, mapsize=[25,25], normalization = 'var', initialization='pca', component_names=names)
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
K = np.arange(2,6)
for i in K:
    Kluster = som.cluster(i)
    hits  = HitMapView(20,20,"K-Means Clustering",text_size=16)
    a=hits.show(som)


#
som.cluster(n_clusters=K) 
#som.cluster() returns the k-means cluster labels for each neuron of the map, 
#but it is straightforward to retrieve the cluster labels for the whole training set, 
#by assigning them the label of the BMUs (best-matching units). You can can do for example:
#Make sure indices line up.... 
map_labels = som.cluster(n_clusters=3)
# som._bmu[0]
data_labels = np.array([map_labels[int(k)] for k in som._bmu[0]]) # mapping labels from 2500 to total size of df
clusters = pd.Series(data_labels)
clusters = clusters.rename('cluster').to_frame()
#concat cluster column with small original som input df
df_labeled = None
df_labeled = pd.concat([df,clusters], axis=1)
##### merging df_label with original large df #####
maxdf_labeled = pd.merge(maxdf, df_labeled)


# stripping maxdf so same number of rows/indices
#maxdf_stripped = maxdf.apply(pd.to_numeric, errors='coerce')
#maxdf_stripped = maxdf_stripped.dropna(axis='columns',how='any')
#maxdf_stripped = maxdf_stripped.dropna(axis='rows',how='any')
#maxdf_labeled = pd.merge(maxdf_stripped,df_labeled)
################# LOL ########################
#########################################
#  maxdf_labeled = pd.merge(maxdf, df_labeled)
##########################################
############################################


















####### MONTH ###########
## with maxNDVIsum3 logic setting month, must use different method###
#########################
###########dfmonth = None
##########dfmonth= dfMaster.loc[(dfMaster['month']==7)]
#del dfmonth['level_0']
##########dfmonth = dfmonth.reset_index()
#df = dfMaster.loc[(dfMaster.month > 3) & (dfMaster.month<9)]
#df = df.reset_index()
#del df['level_0']
#df_labeled_master = pd.concat([df_labeled, dfApril], axis=1)
############df_labled_master=None
############df_labeled_master = pd.merge(df_labeled, dfmonth)

#df_labeled.info()
#df_labeled_master.info()

'''
maxdf_stripped['fractionofyear', u'pixel', u'Latitude', u'Longitude',
       u'month', u'year', u'date', u'CDL', u'countyears', u'NDVI', u'EVI',
       u'NDWI1', u'NDWI2', u'NDVImean', u'NDVIstd', u'EVImean', u'EVIstd',
       u'NDWI1mean', u'NDWI2mean', u'NDWI1std', u'NDWI2std', u'zNDVI', u'zEVI',
       u'zNDWI1', u'zNDWI2', u'NDVIsum3', u'NDVIsum3mean', u'NDVIsum3std',
       u'zNDVIsum3', u'pr', u'etr', u'daysabove30', u'tmmx',
       u'daysbelowneg5', u'daysabove35', u'GDD', u'srad', u'avgtemp', u'tmmn',
       u'vpd', u'dayslowpr', u'eto', u'daysabove28', u'GDDmean3', u'VPDmean3',
       u'prsum3', u'dayslowprsum3', u'GDDmean3mean', u'GDDmean3std',
       u'zGDDmean3', u'VPDmean3mean', u'VPDmean3std', u'zVPDmean3',
       u'dayslowprsum3mean', u'dayslowprsum3std', u'zCumdlowprsum3',
       u'prsum3mean', u'prsum3std', u'zPrsum3', u'vpdmean', u'vpdstd', u'zVPD',
       u'prmean', u'prstd', u'zP', u'dayslowprmean', u'dayslowprstd',
       u'zcumdayslowr', u'GDDmean', u'GDDstd', u'zGDD', u'maxNDVIsum3',]
'''

#Investigating the Nan values that reduce size of df. 
#it occurs in all the 
########pd.options.display.max_columns=75
#######NaN = maxdf.isna()
#########NaN.describe()
