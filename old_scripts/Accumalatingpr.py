#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:33:19 2018

@author: brian
"""

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from datetime import datetime

crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

met = pd.read_csv(data_path + 'dailymeteo' + crop_id + ".csv", index_col='system:indexviejuno')
ndvi = pd.read_csv(data_path + 'VegInd' + crop_id + ".csv", index_col='system:indexviejuno')
#resetting index because default index is system:indexviejuno and we would rather have dates
met = met.reset_index()
ndvi = ndvi.reset_index()
#met.index = met['date']
#ndvi.index = ndvi['date']

# merge datasets
# merge combines by index and by columns that have the same name and merges values that line up
dfMaster = ndvi.merge(met)

#creating date function
def systemindex_to_datetime(series):
    date = pd.to_datetime(series.split('_')[0])
    return date
#create date column by calling prior function
dfMaster['date'] = dfMaster['system:index'].apply(systemindex_to_datetime)
dfMaster.set_index('date', inplace=True)


#Another attempt from StackOverflow

def sum_days_in_row_with_condition(df):
    sorted_df = df.sort_values(by='date', ascending=True)
    condition = sorted_df['pr'] < 1
    sorted_df['days_in_a_row'] = condition.cumsum() - condition.cumsum().where(~condition).fillna()
    return sorted_df

dfMaster = (dfMaster.groupby('system:indexviejuno').apply(sum_days_in_row_with_condition).reset_index(drop=True))


# rewriting function in my own words
def add_days_in_a_row (df):
    dfMaster_sorted = dfMaster.sort_values(by='date', ascending=True)
    condition = (dfMaster_sorted['pr']<1)
    dfMaster_sorted['cumdayslowpr'] = condition.cumsum()-condition.cumsum().mask(condition).ffill().fillna(0)

dfMaster = (dfMaster.groupby('system:indexviejuno')).apply(add_days_in_a_row)

#investigating
dfJune = dfMaster.loc[('month'==6)]
maximum = dfJune.groupby(['system:indexviejuno']).days_in_a_row.max()




### code check

df = pd.DataFrame(
    [
     [datetime(2016, 1, 9), 1000, 1.9],
     [datetime(2016, 1, 8), 1001, 1.9],
     [datetime(2016, 1, 7), 1001, 2],
     [datetime(2016, 1, 7), 1000, 5],
     [datetime(2016, 1, 8), 1000, 2],
     [datetime(2016, 1, 1), 1000, 5], 
     [datetime(2016, 1, 1), 1001, .1], 
     [datetime(2016, 1, 2), 1000, 2],
     [datetime(2016, 1, 5), 1000, .2],
     [datetime(2016, 1, 6), 1000, .1],
     [datetime(2016, 1, 6), 1001, .2],
     [datetime(2016, 1, 2), 1001, .5], 
     [datetime(2016, 1, 3), 1000, 0], 
     [datetime(2016, 1, 3), 1001, 5], 
     [datetime(2016, 1, 4), 1000, 0], 
     [datetime(2016, 1, 4), 1001, 0],
     [datetime(2017, 1, 1), 1001, 0],
     [datetime(2017, 1, 2), 1001, 0],
     [datetime(2017, 1, 3), 1001, 0],
     [datetime(2017, 1, 4), 1001, 2],
     [datetime(2017, 1, 5), 1001, 0],
     [datetime(2017, 1, 1), 1000, 0],
     [datetime(2017, 1, 2), 1000, 0],
     [datetime(2017, 1, 3), 1000, 3],
     [datetime(2017, 1, 4), 1000, 0],
    ], 
    columns=['date', 'spatial_pixel', 'column_A'])


##############
#WORKED
###############

def sum_days_in_row_with_condition(df):
    df = df.sort_values(by='date', ascending=True)
    df['condition'] = df['column_A'] < 2
    df['condition.cumsum'] = df.condition.cumsum()
    df['condition.cumsum.where(~condition)'] = df.condition.cumsum().mask(df.condition)
    df['df.condition.cumsum().mask(df.condition).ffill()'] = df.condition.cumsum().mask(df.condition).ffill()
    df['days_inarow_lowpr'] = df.condition.cumsum() - df.condition.cumsum().mask(df.condition).ffill().fillna(0)
    return df

# added grouby year to restart counter every year to make up for the lack up winter df data...
# ... easily removed if winter df becomes available. 
df_new = (df.groupby(['spatial_pixel', df['date'].dt.year])
   .apply(sum_days_in_row_with_condition)
   .reset_index(drop=True))
########
#with array instead of column
#######

def sum_days_in_row_with_condition(df):
    df = df.sort_values(by='date', ascending=True)
    condition = df['column_A'] < 2
    df['days_inarow_lowpr'] = condition.cumsum() - condition.cumsum().mask(condition).ffill().fillna(0)
    return df

df_new = (df.groupby('spatial_pixel')
   .apply(sum_days_in_row_with_condition)
   .reset_index(drop=True))







c=dfnew.condition

d=c.groupby(c.diff().cumsum()).cumcount()


c.

##########Failed Atempts################


df = df.sort_values(by='date', ascending=True)
condition = df['column_A']<2
df['daysinarow'] = df.groupby(['spatial_pixel']).apply(condition.cumsum()-condition.cumsum().where(~condition).ffill().astype(int))



'''


def rollingday():
    for pixel in range(dfMaster['system:indexviejuno']:
        if dfMaster['pr']<1:
            dfMaster['rollingday'] = dfMaster.groupby[['system:indexviejuno','year']].rolling_apply()
        else:
            
#locating subsets of dataframe
lowpr = dfMaster.loc[(dfMaster.pr<1)]
lowpr = lowpr.loc[(lowpr['cropland_mode']==int(crop_id))]

#setting index
lowpr.index=lowpr['system:indexviejuno']

#creating date function
def systemindex_to_datetime(series):
    date = pd.to_datetime(series.split('_')[0])
    return date
#create date column by calling prior function
lowpr['date'] = lowpr['system:index'].apply(systemindex_to_datetime)

#lowpr['stdpr']= lowpr.groupby(['system:indexviejuno','month'].pr.std()

# day function
def date_to_day(series):
    day = pd.series.split('-')[2]
    return day

# creating day colummn
lowpr['day'] = lowpr['system:index'].map(lambda x: x.split('_')[0])
lowpr.day = lowpr.day.map(lambda x: str(x)[6:])
lowpr.day = lowpr.day.astype(int)



#one attempt
daysinarow = []
def lowpr_daysinarow(series):
    
    for x in series:
        if x  == x +1 | x-1:
            daysinarow +=1
    return daysinarow    

#test

lowpr_daysinarow(lowpr.day)

daysinarow = lowpr.groupby(['system:indexviejuno', 'year', 'month'])['day'].apply(lowpr_daysinarow)

#another attempt

def rolling_count(val):
    if val == rolling_count.previous + 1 :
        rolling_count.count +=1
    else:
        rolling_count.previous = val
        rolling_count.count = 1
    return rolling_count.count
rolling_count.count = 0 #static variable
rolling_count.previous = None #static variable

lowpr['count'] == lowpr.groupby(['system:indexviejuno','date'])['day'].apply(rolling_count)

# attempt 3
lowpr['block'] = (lowpr.groupby(['system:indexviejuno','year'])['day'] == lowpr['day'].shift(1)).astype(int).cumsum() 
lowpr['count'] = lowpr.groupby(['system:indexviejuno', 'year', 'day'])


#lowpr.groupby(['system:indexviejuno','month']).cumsum 
'''