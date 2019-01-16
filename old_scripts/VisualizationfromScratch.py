#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:56:46 2018

@author: brian
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as st
sns.set(style="ticks")




crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

met = pd.read_csv(data_path + 'MeteorMonthly_experiment_' + crop_id + ".csv", index_col='system:indexviejuno')
ndvi = pd.read_csv(data_path + 'VegInd3_' + crop_id + ".csv", index_col='system:indexviejuno')

#resetting index because default index is system:indexviejuno and we would rather have dates. Why? I'm not sure why would we not want to merge on system:indexviejuno?
#del ndvi['system:indexviejuno.1']
#met = met.reset_index()
#ndvi = ndvi.reset_index()
#met.index = met.date
#ndvi.index = ndvi.date

#met.index = pd.to_datetime(met.date)
met.date = met.date.astype('str')
met.date = pd.to_datetime(met.date)
met.date = met.date.astype('str')
met['system:indexviejuno']= met.index

#ndvi.index = pd.to_datetime(ndvi.date)
#ndvi.date = ndvi.date.astype('str')
#ndvi.date = pd.to_datetime(ndvi.date)
ndvi['system:indexviejuno']= ndvi['system:indexviejuno.1']
del ndvi['system:indexviejuno.1']

# merge datasets to perform correlations
# merge combines by index and by columns that have the same name and merges values that line up
dfMaster = ndvi.merge(met)
# reset index to.... 
dfMaster.index = dfMaster['system:indexviejuno']
#rename column
#dfMaster.rename(columns= {'zPrecip':'zP'}, inplace=True)
dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfMasterSummer.dropna(inplace=True)
dfMasterFeb = dfMaster.loc[(dfMaster['month'] ==2)]
dfMasterMarch = dfMaster.loc[(dfMaster['month'] ==3)]
dfMasterApril = dfMaster.loc[(dfMaster['month'] ==4)]
dfMasterMay = dfMaster.loc[(dfMaster['month'] ==5)]
dfMasterJune = dfMaster.loc[(dfMaster['month'] ==6)]
dfMasterJuly = dfMaster.loc[(dfMaster['month'] ==7)]
dfMasterAug = dfMaster.loc[(dfMaster['month'] ==8)]
dfMasterSep = dfMaster.loc[(dfMaster['month'] ==9)]

##################### PLOTTING

datapath = '/home/brian/Documents/westernWheat/Images/April-May-June/'
sns.kdeplot(dfMasterFeb.EVI)
sns.kdeplot(dfMasterMarch.EVI)
sns.kdeplot(dfMasterApril.EVI)
sns.kdeplot(dfMasterMay.EVI)
sns.kdeplot(dfMasterJune.EVI)


g = sns.PairGrid(df, x_vars=['pr', 'vpd', 'cum_daysinarow_lowpr', 'tmmx', 'NDVI', 'Latitude',
       'Longitude'], y_vars=['pr', 'vpd', 'cum_daysinarow_lowpr', 'tmmx', 'NDVI', 'Latitude',
       'Longitude'])
g = g.map_upper(plt.scatter,s=.01)
g = g.map_diag(sns.kdeplot)
g = g.map_lower(sns.kdeplot, cmap='Blues_d')




g=sns.PairGrid(dfMasterSummer, x_vars=['zVPD','zP','ztmmx','ztmmn'], y_vars=[ 'anomalyNDVI','anomalyEVI'])
g =g.map(plt.scatter,s=.01)

g = sns.PairGrid(dfMasterSummer, x_vars=['zVPD','zP','ztmmx','ztmmn'], y_vars=[ 'anomalyNDVI','anomalyEVI'])
g = g.map(plt.scatter, s=.01) 
g = g.map(sns.regplot)

g = sns.PairGrid(met, x_vars=['vpd','pr','tmmx','tmmn', 'srad', 'eto', 'daysabove28', 'daysabove35'], y_vars=['vpd','pr','tmmx','tmmn', 'srad', 'eto', 'daysabove28', 'daysabove35'])
g = g.map_offdiag(plt.scatter, s=.01)
g = g.map_diag(plt.hist, bins=20)
g = g.map_lower(sns.kdeplot)

plt.savefig(datapath + str(crop_id) + "MeterologicalVariables_kde_05-06.png", dpi=500)

g = sns.PairGrid(dfMasterSummer, x_vars=['zVPD','zP','ztmmx','ztmmn', 'zSRAD', 'zETO'], y_vars=['zVPD','zP','ztmmx','ztmmn', 'zSRAD', 'zETO'])
g = g.map_offdiag(plt.scatter,s=.01)
g = g.map_diag(plt.hist, bins=20)
g = g.map_lower(sns.kdeplot)
g = sns.jointplot(df.groupby([df.cluster]).pr, df.groupby([df.cluster]).NDVI, kind='reg', scatter_kws={'s':.1})
g = sns.PairGrid(dfMasterSummer, x_vars=['anomalyNDVI','anomalyEVI','anomalyNDWI1','anomalyNDWI2'], y_vars=['anomalyNDVI','anomalyEVI','anomalyNDWI1','anomalyNDWI2'])
g = g.map_offdiag(plt.scatter,s=.01)
g = g.map_diag(sns.kdeplot)
g = g.map_lower(sns.kdeplot)

g = sns.PairGrid(dfMasterSummer, x_vars=['zVPD','zP','ztmmx','ztmmn', 'zSRAD', 'zETO'], y_vars=['zVPD','zP','ztmmx','ztmmn', 'zSRAD', 'zETO'])
g = g.map_upper(plt.scatter,s=.01)
g = g.map_diag(plt.hist)
g = g.map_lower(sns.kdeplot)

g = sns.PairGrid(dfMasterSummer, x_vars=['NDVI','EVI','NDWI1','NDWI2'], y_vars=['NDVI','EVI','NDWI1','NDWI2'])
g = g.map_offdiag(plt.scatter,s=.01)
g = g.map_diag(plt.hist, bins=20)
g = g.map_lower(sns.kdeplot)


g = sns.PairGrid(dfMasterSummer, x_vars=['NDVI','EVI','NDWI1','NDWI2'], y_vars=['NDVI','EVI','NDWI1','NDWI2'])
g = g.map_offdiag(sns.regplot)
g = g.map_diag(plt.hist, bins=20)
g = g.map_lower(sns.kdeplot)



g = sns.jointplot(dfMasterSummer.zEVI, dfMasterSummer.ztmmx, kind='reg', scatter_kws={'s':.01})
g = sns.jointplot(dfMasterSummer.zEVI, dfMasterSummer.zVPD, kind='reg', scatter_kws={'s':.01})
g = sns.jointplot(dfMasterSummer.anomalyEVI, dfMasterSummer.zP, kind='reg', scatter_kws={'s':.01})
g = sns.jointplot(dfMasterSummer.anomalyEVI, dfMasterSummer.ztmmn, kind='reg', scatter_kws={'s':.01})
g = sns.jointplot(dfMasterSummer.anomalyEVI, dfMasterSummer.zSRAD, kind='reg', scatter_kws={'s':.01})
g = sns.jointplot(dfMasterSummer.anomalyEVI, dfMasterSummer.zETO, kind='reg', scatter_kws={'s':.01})

df = dfMasterSummer
g = sns.jointplot(df.anomalyEVI, df.ztmmx, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.anomalyEVI, df.zVPD, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.anomalyEVI, df.zP, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.anomalyEVI, df.ztmmn, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.anomalyEVI, df.zSRAD, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.anomalyEVI, df.zETO, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.EVI, df.eto, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.vpd, df.EVI, kind='reg', scatter_kws={'s':.1}, robust=True)
g = sns.jointplot(df.EVI, df.daysabove28, kind='reg', scatter_kws={'s':.1})
g = sns.jointplot(df.EVI, df.daysabove3Marco.maneta5, kind='reg', scatter_kws={'s':.1}, robust=True)



g = sns.jointplot(dfMasterSummer.anomalyEVI, dfMasterSummer.anomalyNDVI, kind='reg', scatter_kws={'s':.1})

#REsidual plot
ax = sns.regplot(x='tmmx', y='EVI', data = dfMasterSummer, scatter_kws={'s':.1}, order=2)
sns.residplot(x='anomalyEVI', y='anomalyNDVI', data = dfMasterSummer, scatter_kws={'s':.1})


######################
# statistics by cluster
#######################


df = df_labeled

for cluster in range(pd.unique(df.cluster)):
    
    g = sns.jointplot(df.tmmx, df.NDVI, kind='reg', scatter_kws={'s':.1})
    g = sns.jointplot(df.vpd, df.NDVI, kind='reg', scatter_kws={'s':.1})
    g = sns.jointplot(df.pr, df.NDVI, kind='reg', scatter_kws={'s':.1})


g = sns.jointplot(df.tmmx, df.NDVI, kind='reg', scatter_kws={'s':.1})

g = sns.jointplot(df.loc[(df.cluster==4)].tmmx, df.loc[(df.cluster==4)].NDVI, kind='reg', scatter_kws={'s':.1})

sns.relplot(x='tmmx', y='NDVI', hue='cluster', data=df)


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
        u'NDVI',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35', u'eto', u'zVPD', u'zP', u'zETO',
       u'zSRAD', u'zETR', u'ztmmn', u'ztmmx', u'zcum_daysinarow_lowpr',
       u'zdaysabove30']]

corr = dfapril.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')


