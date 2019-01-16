#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:41:47 2018

@author: brian
"""

import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import GrowingSeasonSelection
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

simple=dfMaster
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
##########################################
# selecting median maxNDVIsum3bool.month
#########################################




###
#refining, dfMaster, selecting variables, making smaller. 
###
#I PRESUME I DO NOT WANT TO BRING IN ZSCORES.... BECAUSE THE SOM TAKES THE SPATIAL/TEMPORAL ZSCORE AND I WANT TO DIFFERENTIATE BETWEEN AREAS
maxdf = GrowingSeasonSelection(simple)

simple = maxdf[['prsum3mean','VPDmean3mean']]#'GDDmean3mean','dayslowprsum3mean']]  #'NDVIsum3
names = ['prsum3mean','VPDmean3mean']#,'GDDmean3mean','dayslowprsum3mean']   #'NDVIsum3'
simple = simple.apply(pd.to_numeric,  errors='coerce')
simple = simple.dropna(how='any') # drops only thirty rows out of 12,000


#######################################
######################################
#K_means
######################################

###### distance metric, therefore susceptible to scale #
#### normalize
####### take pixel mean - all pixel mean
pr_mean = simple.prsum3mean.mean()
vpd_mean = simple.VPDmean3mean.mean()
#ndvi_mean = simple.NDVIsum3mean.mean()

pr_std = simple.prsum3mean.std()
vpd_std = simple.VPDmean3mean.std()
#ndvi_std = simple.NDVIsum3mean.std()

zpr = (simple.prsum3mean - pr_mean)/pr_std
zvpd = (simple.VPDmean3mean - vpd_mean)/vpd_std
#zndvi = (simple.NDVIsum3mean - ndvi_mean)/ndvi_std

normalized_df = pd.DataFrame()
normalized_df['zpr'],normalized_df['zvpd'], = zpr, zvpd#,normalized_df['zndvi']  zndvi

#####################
#####################
#####################


#kmeans = sklearn.cluster.KMeans(n_clusters=2).fit(df)
y_predict  = KMeans(n_clusters=3).fit_predict(normalized_df)

def plot():
    
    for j in range(5):
        score = []
        for i in range(2,10):
            y = KMeans(n_clusters=i).fit(normalized_df)
            score.append(y.score(normalized_df))
        #print score
        plt.scatter(range(2,10), [i * -1 for i in score], s=20)
        plt.ylabel('score')
        plt.xlabel('number of clusters')
            
            
plot()


df['cluster']= y_predict
#maxdf = maxdf.dropna(how='any') they are the same row size in this instance... 
maxdf['cluster'] = y_predict

######## Do it on the som.codebook.matrix
########################################
#########################################
non_norm_som_codebook = som._normalizer.denormalize_by(som.data_raw,som.codebook.matrix) # this is not what we should be using.......
codebook = som.codebook.matrix
codebooknorm = som._normalizer.normalize(som.codebook.matrix)
norm_data = som._normalizer.normalize(som.data_raw)

def plot(k=20, runs=10):

    SSE_matrix = np.array(0)
    for j in range(runs):
        score = []
        for i in range(2,k):
            y = KMeans(n_clusters=i).fit(codebooknorm) # Are these really normalized ? mean of columns=0 but std of columns not rows
            score.append(y.score(norm_data))
            
        #print score
        plt.scatter(range(2,k), [i*-1e-4 for i in score], s=40, )
        plt.ylabel('score')
        plt.xlabel('number of clusters')
        plt.title('K-means elbow plot')
        SSE_matrix = np.append(SSE_matrix,np.array(score)) 
    SSE_matrix_avg = np.mean(SSE_matrix, axis=1)
    plt.plot(range(2,k),SSE_matrix_avg)
    
plot(runs=14, k=20)




y = KMeans(n_clusters=6).fit_predict(codebooknorm)
#y.score(norm_data)




#############################
### lower case z's are not spatial-temporal z-scores useful for distance metrics (SOM) or .. 
###################################
################### combining with maxdf frame #######3

#maxdf_labeled = maxdf.merge(df)#, how='right')  # not working right... look at shape
'''
= maxdf



########### plotting




sns.relplot(x='prsum3mean', y='NDVIsum3mean', palette="Paired",hue='cluster', s=9, data=df)
sns.relplot(x='VPDmean3mean', y='NDVIsum3mean', palette="Paired",hue='cluster', s=9, data=df)
sns.relplot(x='VPDmean3mean', y='prsum3mean', palette="Paired",hue='cluster', s=9, data=df)

# append to  maxdf to have access to other columns

#maxdf = maxdf.apply(pd.to_numeric,  errors='coerce')
#maxdf = maxdf.dropna(how='any')
##### Is this robust???
#maxdf['cluster']=y_predict

############## plotting

sns.relplot(x='zPrsum3', y='zNDVIsum3', palette="Paired",hue='cluster', s=9, data=df)
sns.relplot(x='zVPDmean3', y='zNDVIsum3', palette="Paired",hue='cluster', s=9, data=df)
sns.relplot(x='zVPDmean3', y='zPrsum3', palette="Paired",hue='cluster', s=9, data=df)
sns.relplot(x='zPrsum3', y='zNDVIsum3', palette="Paired",hue='cluster', s=9, data=df)


'''