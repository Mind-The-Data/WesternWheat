#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:03:30 2018

@author: brian
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#crop_id = 23
#crop_id = str(crop_id)
#data_path = '../rawData/AgMet/'

def GrowingSeasonSelection(df):
    ''' takes in a df with specific column names e.g. pixel, year, month, NDVIsum3, etc.
    and defines growing season based on the median 
    months of maximum NDVI sum 3 rolling window such 
    that each pixel has a consistently defined growing season 
    throughout years. If only maxNDVIsum3 is used this will likely 
    oscillate due to remote sensining error and shifting weather, so that some years the 
    growing season is 5,6,7 and others is defined as 6,7,8. This creates problems with 
    average and anomalous growing season conditions,
    as a dry spring may be considered anomalously wet compared to the normal summer.'''

    maxNDVIsum3s = df.groupby(['pixel', 'year']).NDVIsum3.apply(np.max) #Outputs a series with multi index
    maxNDVIsum3s = maxNDVIsum3s.reset_index(level=[0,1]) # now df with range index and pixel, year, and value to merge with
    maxNDVIsum3s.rename(columns={'NDVIsum3':'maxNDVIsum3'}, inplace=True)
    print 'defining maxNDVIsum3s...'
    df = df.merge(maxNDVIsum3s,how='outer')  # This gives a repeat of maxNDVIsum3 values over all instances that are pixel and year..
    
    df['maxNDVIbool']= np.where(df['NDVIsum3']==df['maxNDVIsum3'], True, False)
    
    median_maxNDVI_month = df[df.maxNDVIbool==True].groupby('pixel').month.apply(np.median) # outputs pixel and median month column
    median_maxNDVI_month = median_maxNDVI_month.apply(np.round) # round, in future maybe a better way to handle the below join
    # to take into account the midpoint between months.... 7.5 could be 6,7, and 5+8/2 say... for all relevant variables
    df=df.join(median_maxNDVI_month,how='outer',on='pixel',rsuffix='_median') # combines series/df.columns with df, union, on pixel index level
    print 'joining....'
    df_stripped = df[np.isclose(df.month_median,df.month)]
    df = df_stripped
    print 'stripped df to growing season'
    return df_stripped



#index to date time function
def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date
