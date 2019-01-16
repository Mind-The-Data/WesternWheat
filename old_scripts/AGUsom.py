#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:49:37 2018

@author: brian
"""

import numpy as np
from sompy.sompy import SOMFactory

import pandas as pd

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


from functions import GrowingSeasonSelection
maxdf = GrowingSeasonSelection(dfMaster)


names = ['prsum3mean','VPDmean3mean','GDDmean3mean','drydayssum3mean']
df = maxdf[names]#,'NDVIsum3']]#'GDDmean3mean','dayslowprsum3mean']]  #'NDVIsum3
names = ['Precipitation','Vapor Pressure Deficit','Growing Degree Days','Cumulative Dry Days']

df = df.apply(pd.to_numeric,  errors='coerce')
df = df.dropna(how='any')
# Resetting index so it goes in order, since I only selected april, a standard index is all messed up april 2008 to april 2009 skips 
#a ton of index values, and this makes it impossible to combine cluster output index with df 
df.index
df = df.reset_index(level=0)
del df['index']
unique = df

##Investigating
# 5*sqrt(row*column)



som = SOMFactory().build(df.values, mapsize=[20,20], normalization = 'var', initialization='pca', component_names=names,\
                    neighborhood = 'gaussian', lattice='rect')
som.train(n_job=1, verbose='info', train_rough_len=30, train_finetune_len=30)




# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
topographic_error = som.calculate_topographic_error()
quantization_error = np.mean(som._bmu[1])
print "Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error)

from sompy.visualization.mapview import View2D
view2D  = View2D(4,4,"rand data",text_size=16)
view2D.show(som, col_sz=2, which_dim="all", denormalize=True)

# U-matrix plot
from sompy.visualization.umatrix import UMatrixView

umat  = UMatrixView(width=10,height=10,title='U-matrix')
umat.show(som)



from sompy.visualization.hitmap import HitMapView
from sompy.visualization.bmuhits import BmuHitsView
bmuhitsview = BmuHitsView(12,12,'Data per node', text_size=24)
bmuhitsview.show(som, anotate=False, onlyzeros=False, labelsize=7, logaritmic=False)


Kluster = som.cluster(5)
hits  = HitMapView(20,20,"K-Means Clustering",text_size=16)
a=hits.show(som)


from ThirdSOM import bootstrap, HowManyK

SSE_Matrix = bootstrap(runs=10,k=20)
##################### average columns in 
SSE_Matrix = np.mean(SSE_Matrix, axis=0)

# SSE of K-means
plt.plot(np.arange(2,SSE_Matrix.size+2), SSE_Matrix)
plt.title('K-Means Optimal k')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Sum Square Error')




#som.cluster() returns the k-means cluster labels for each neuron of the map, 
#but it is straightforward to retrieve the cluster labels for the whole training set, 
#by assigning them the label of the BMUs (best-matching units). You can can do for example:
#Make sure indices line up.... 
map_labels = som.cluster(n_clusters=7)
# som._bmu[0]
data_labels = np.array([map_labels[int(k)] for k in som._bmu[0]]) # mapping labels from size of grid to total size of df
clusters = pd.Series(data_labels)
clusters = clusters.rename('cluster').to_frame()
#concat cluster column with small original som input df
df_labeled = None
df_labeled = pd.concat([unique,clusters], axis=1)


##### merging df_label with original large df #####
maxdf = maxdf.reset_index()

# merge sudenly stopped working on 12/3/18 and now I am concatenating... I am not sure why? I don't remember updating pandas...
#maxdf_labeled = pd.merge(maxdf, df_labeled)
#maxdf_labeled = pd.merge(maxdf, df_labeled, how='inner', on=[u'prsum3mean', u'VPDmean3mean', u'GDDmean3mean', u'drydayssum3mean'])
maxdf_labeled = pd.concat([maxdf,clusters], axis=1)




y = KMeans(n_clusters=5).fit_predict(codebooknorm)

map_labels = y  # from simple.py
# som._bmu[0]
data_labels = np.array([map_labels[int(k)] for k in som._bmu[0]]) # mapping labels from size of grid to total size of df
clusters = pd.Series(data_labels)
clusters = clusters.rename('cluster').to_frame()
#concat cluster column with small original som input df
#df_labeled = None
#df_labeled = pd.concat([unique,clusters], axis=1)
##### merging df_label with original large df #####
maxdf = maxdf.reset_index()
maxdf_labeled = pd.concat([maxdf,clusters], axis=1)

maxdf_labeled.shape