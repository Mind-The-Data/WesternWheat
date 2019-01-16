#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:51:46 2018

@author: brian
"""
import pandas as pd

crop_id = 23
crop_id = str(crop_id)

print "Reading CDL dataframe from harddrive..."
data_path = '../../rawData/AgMet/'
dfCDL = pd.read_csv(data_path + "CDL" + crop_id + ".csv", 
            usecols=['system:index','countyears','cropland_mode','first','system:indexviejuno'])
dfCDL['year'] = dfCDL['system:index'].apply(lambda x: x.split("_")[0]).astype(int)




dfCDL
def function(df,columns):
    df = df[columns]
    for i in range(len(columns)):
        df[columns[i]] = 5
        print df[columns[i]]
        
function(dfCDL, columns = ['year', 'countyears'])