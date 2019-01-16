#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:48:13 2018

@author: brian
"""
import seaborn as sns
########Correlation Matrix

#Seaborn's heatmap version:
df3 = dfMaster


corr = df1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')

pd.set_option('display.max_columns', 25)

df1= dfMaster[[ u'Latitude', u'Longitude', u'month',
       u'year', u'NDVI', u'EVI',
       u'NDWI1',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35', u'eto', u'zVPD', u'zP', u'zETO',
       u'zSRAD', u'zETR', u'ztmmn', u'ztmmx', u'zcum_daysinarow_lowpr',
       u'zdaysabove30']]
# u'NDWI2', u'NDVImean', u'NDVIstd', u'EVImean', u'EVIstd',
#       u'NDWI1mean', u'NDWI2mean', u'NDWI1std', u'NDWI2std', u'zNDVI', u'zEVI',

corr = df1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')


df2= dfMaster[[ u'Latitude', u'Longitude',
        u'NDVI',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35', u'eto', u'zVPD', u'zP', u'zETO',
       u'zSRAD', u'zETR', u'ztmmn', u'ztmmx', u'zcum_daysinarow_lowpr',
       u'zdaysabove30']]

corr = df2.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')


dfapril = dfMaster.loc[(dfMaster.month==4)]


dfapril= dfapril[[ u'Latitude', u'Longitude',
        u'NDVI', 'EVI',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35']]

dfapril = dfapril.loc[(dfapril.NDVI >.4)]

corr = dfapril.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')


dfmay  = dfMaster.loc[(dfMaster.month==5)]
dfmay= dfmay[[ u'Latitude', u'Longitude',
        u'NDVI', 'EVI',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35']]

corr = dfmay.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')

dfmay  = dfMaster.loc[(dfMaster.NDVI>.4)]
dfmay= dfmay[[ u'Latitude', u'Longitude',
        u'NDVI', 'EVI', 'zNDVI',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35']]

corr = dfmay.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')


dfjulywinter= df[[
        u'NDVI', 'EVI', 'zNDVI',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'pr','avgtemp',
       u'daysabove28']]
corr = dfjulywinter.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')
