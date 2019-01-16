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
    print 'defining maxNDVIsum3s...\n'
    df = df.merge(maxNDVIsum3s,how='outer')  # This gives a repeat of maxNDVIsum3 values over all instances that are pixel and year..
    
    df['maxNDVIbool']= np.where(df['NDVIsum3']==df['maxNDVIsum3'], True, False)
    
    median_maxNDVI_month = df[df.maxNDVIbool==True].groupby('pixel').month.apply(np.median) # outputs pixel and median month column
    median_maxNDVI_month = median_maxNDVI_month.apply(np.round) # round, in future maybe a better way to handle the below join
    # to take into account the midpoint between months.... 7.5 could be 6,7, and 5+8/2 say... for all relevant variables
    df=df.join(median_maxNDVI_month,how='outer',on='pixel',rsuffix='_median') # combines series/df.columns with df, union, on pixel index level
    print 'joining....\n'
    df_stripped = df[np.isclose(df.month_median, df.month)]
    print 'stripped df to growing season\n'
    return df_stripped


def interpolate(df, columns, limit=2):
    for column in columns:
        df[column] = df[column].groupby('pixel').apply(lambda x: \
          x.interpolate(method='linear',limit=limit))
    print "Interpolated!"
    return df

#index to date time function
def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date
############
#daily
##############
    
def add_days_in_a_row (df):
    '''precip only, 2.5 mm threshold'''
    df_sorted = df.sort_values(by='date', ascending=True)
    condition = (df_sorted['pr']<2.5) 
    df_sorted['drydays'] = condition.cumsum()-condition.cumsum().mask(condition).ffill().fillna(0)
    return df_sorted

def add_days_in_a_row_vpd (df):
    '''precip & vpd, 2.5 mm threshold and 1 kPa'''
    df_sorted = df.sort_values(by='date', ascending=True)
    condition = ((df_sorted['pr']<2.5) &  (df_sorted['vpd']>1))
    df_sorted['prvpd'] = condition.cumsum()-condition.cumsum().mask(condition).ffill().fillna(0)
    return df_sorted

def add_vpd (df):
    '''precip 2.5 mm threshold, add vpd after that. not finished'''
    df_sorted = df.sort_values(by='date', ascending=True)
    condition = ((df_sorted['pr']<2.5))
    df_sorted['drydays'] = condition.cumsum()-condition.cumsum().mask(condition).ffill().fillna(0)
    return df_sorted

# building add_vpd
# vpd is close to zero when raining, so... this will be the same as accumulated vpd....
#df_sorted = dfMasterMet.sort_values(by='date', ascending=True)
#condition = ((df_sorted['pr']<2.5))
#series = df_sorted.vpd.mask(~condition)

#############
##### Monthly
#############


def rolling_sum(df, columns, w=3, year=False):
    '''insert rolling sum for each column in list of columns for window (w) in months. '''
    if not year:
        for column in columns:
            print column + 'sum'+ str(3)
            df.index = df.date
            series = df.groupby(['pixel']).rolling(w)[column].sum() # no year. so rolls over dec-jan
            series = series.reset_index(level=[0,1])
            series.rename(index=str, columns={column: column + 'sum' + str(w)}, inplace=True)
            #df = df.merge(series, on=['pixel','date'])
            df = df.merge(series, on=['pixel','date'])
            print 'successful merge'
        return df
    if year:
        print 'groupying by year...discontinuos data?...CDL filtered?'
        for column in columns:
            print column + 'sum'+ str(3)
            df.index = df.date
            series = df.groupby(['pixel', 'year']).rolling(w)[column].sum()
            series = series.reset_index(level=[0,1,2])
            series.rename(index=str, columns={column: column + 'sum' + str(w)}, inplace=True)
            #df = df.merge(series, on=['pixel','date'])
            df = df.merge(series, on=['pixel','date','year'], how='inner')
            print 'merged'
        return df




def rolling_mean(df, columns, w=3):
    '''insert rolling mean for each column in list of columns for window (w) in months.'''
    for column in columns:
        print column + 'mean' + str(w)
        df.index = df.date
        series = df.groupby(['pixel']).rolling(w)[column].mean() # no year. so rolls over dec-jan
        series = series.reset_index(level=[0,1])
        series.rename(index=str, columns={column: column + 'mean' + str(w)}, inplace=True)
        df = df.merge(series, on=['pixel','date']) 
    return df

def statistics(df, columns):
    ''' need to check how many instances statistics are calculated over.'''
    for column in columns:
        mean = df.groupby(['pixel','month'])[column].mean()
        sd = df.groupby(['pixel','month'])[column].std() 
        df = df.join(mean, on=['pixel','month'], rsuffix='mean')
        df = df.join(sd, on=['pixel','month'], rsuffix='sd')
        df['z' + column] = (df[column] - df[column + 'mean']) / df[column + 'sd']
        #thrown_out = df.loc[np.isclose(df['z' + column].abs(), 0.707107), 'z' + column].count()
        #print str(thrown_out) + ' z' + column + ' data points thrown out due to insufficient instances'
        # this value occurs due to machine precision issues when only one or two values are used
        # to calculate z score. 
        # eventually redo this so you are only deleting df.groupby(['pixel','month']).NDVI.size()
        # with small size then no need to delete this silly .707107
        #df.loc[np.isclose(df['z' + column].abs(), 0.707107), 'z' + column] = np.nan


    return df