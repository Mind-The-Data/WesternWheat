#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:50:39 2019

@author: brian
"""

# filling in missing dates 
# work in progress


p = dfMaster[(dfMaster.pixel == 10750) | (dfMaster.pixel == 13154.0)]

p.sort_values(by='date', inplace=True)


###
p.drop_duplicates(subset=['pixel','date'], keep = 'first', inplace=True)

time_range = pd.date_range(p.date.min().to_timestamp(),\
                           p.date.max().to_timestamp(), freq='M')\
                           .to_period('M')
                           

# adds Nan rows where there were missing dates/data
#p = p.reindex(time_range)
p = p.groupby('pixel').apply(lambda x: x.reindex(time_range))



p['NDVI'] = p.NDVI.groupby('pixel').apply(lambda x: x.interpolate())




p.NDVI.interpolate()

def interpolate(df, columns):
    for column in columns:
        df[column] = df[column].groupby('pixel').apply(lambda x: \
          x.interpolate(method='linear',limit=2))
        
    return df

df = interpolate(df = p, columns = ['NDVI','EVI'])



#p['date']=pd.to_datetime(p.date.astype('string'))


# bring in from preprocessing

df = dfMaster

df = df.groupby(['pixel']).pixel.fillna(method='ffill')

df['pixel'] = df.groupby('pixel')['pixel'].transform(lambda x: x.fillna(method='ffill'))









