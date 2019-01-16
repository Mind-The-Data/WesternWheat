#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:07:34 2018

@author: brian
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as st
sns.set(style="darkgrid")



######################
# statistics by cluster
#######################


#df = df_labeled
df = None
df=df_labeled_master



## CLUSTER LOCATION
sns.relplot(y='Latitude', x='Longitude', hue='cluster', palette="seismic", s=9, legend = 'full', data=df)

# CORRELATION GRAPHS
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].cum_daysinarow_lowpr, df.loc[(df.cluster==cluster)].NDVI, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].avgtemp, df.loc[(df.cluster==cluster)].NDVI, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].vpd, df.loc[(df.cluster==cluster)].NDVI, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].pr, df.loc[(df.cluster==cluster)].NDVI, kind='reg', scatter_kws={'s':.1})

for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrecipitation, df.loc[(df.cluster==cluster)].NDVI, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVaporPressureDeficit, df.loc[(df.cluster==cluster)].NDVI, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrecipitation, df.loc[(df.cluster==cluster)].zNDVI, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVaporPressureDeficit, df.loc[(df.cluster==cluster)].zNDVI, kind='reg', scatter_kws={'s':.1})



#
for cluster in range(pd.unique(df.cluster).size):
    g = sns.kdeplot(df.loc[(df.cluster==cluster)].NDVI)

# met correlation graphs



# CORRELATION r VALUES
    #CUMALATIVE
for cluster in range(pd.unique(df.cluster).size):  
    print df.loc[(df.cluster==cluster)].cum_daysinarow_lowpr.corr(df.loc[(df.cluster==cluster)].NDVI)
    #AVGTEMP
for cluster in range(pd.unique(df.cluster).size):   
    print df.loc[(df.cluster==cluster)].avgtemp.corr(df.loc[(df.cluster==cluster)].NDVI)
    #VPD
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].NDVI)
    #PRECIP
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].pr.corr(df.loc[(df.cluster==cluster)].NDVI)


### Other variables
    
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].zVPD.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].zP.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].daysabove28.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].ztmmx.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].ztmmn.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].EVI)

#met correlation graphs
    
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].pr, df.loc[(df.cluster==cluster)].vpd, kind='reg', scatter_kws={'s':.1})

# Met correlate heat graphs
df1 = None
df1 = df[['vpd','pr','avgtemp','cum_daysinarow_lowpr']]
for cluster in range(pd.unique(df.cluster).size):
   g = sns.heatmap(df1.loc[(df.cluster==cluster)].corr(), 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')
 corr = df1.loc[(df.cluster==cluster)].corr()


#met correlates    
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].pr)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].avgtemp)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].cum_daysinarow_lowpr)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].cum_daysinarow_lowpr.corr(df.loc[(df.cluster==cluster)].pr)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].avgtemp.corr(df.loc[(df.cluster==cluster)].pr)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].avgtemp.corr(df.loc[(df.cluster==cluster)].cum_daysinarow_lowpr)



sns.relplot(x='tmmx', y='NDVI', hue='cluster', s=9,palette="Paired", data=df)
sns.relplot(x='vpd', y='NDVI', hue='cluster', s=9, palette="Paired",data=df)
sns.relplot(x='ztmmn', y='NDVI', hue='cluster', s=9,palette="Paired", data=df)
sns.relplot(x='pr', y='NDVI', hue='cluster', s=9, palette="Paired",data=df)
sns.relplot(x='cum_daysinarow_lowpr', y='NDVI', palette="Paired", hue='cluster', s=9, data=df, legend='full')
#sns.relplot(x='Longitude', y='NDVI', hue='cluster',palette="Paired", s=9, data=df)
#sns.relplot(y='Latitude', x='NDVI', hue='cluster', palette="Paired",s=9, data=df)
#sns.relplot(x='daysabove28', y='NDVI', hue='cluster',palette="Paired", s=9, data=df)#

sns.relplot(x='ztmmn', y='zNDVI', hue='cluster',palette="Paired", s=9, data=df)
sns.relplot(x='tmmx', y='zNDVI', hue='cluster',palette="Paired", s=9, data=df)
sns.relplot(x='cum_daysinarow_lowpr', y='zNDVI', palette="Paired",hue='cluster', s=9, data=df)
sns.relplot(x='pr', y='zNDVI', hue='cluster', s=9, palette="Paired",data=df)
sns.relplot(x='zVPD', y='zNDVI', hue='cluster', s=9,palette="Paired", data=df)
sns.relplot(x='zP', y='zNDVI', hue='cluster', s=9,palette="Paired", data=df)
#sns.relplot(x='Longitude', y='zNDVI', hue='cluster', palette="Paired",s=9, data=df)
#sns.relplot(y='Latitude', x='zNDVI', hue='cluster', palette="Paired", s=9, data=df)
#sns.relplot(x='daysabove28', y='zNDVI', hue='cluster',palette="Paired", s=9, data=df)


#####################################
#################
#################
## INVESTIGATING ANOMALY CORRELATION 
#################
#################
####################################


sns.relplot(y='zNDVI', x='NDVI', hue='cluster', s=5, data=df, legend='full')
sns.relplot(y='zVPD', x='vpd', hue='cluster', s=6, data=df, legend='full')
sns.relplot(y='zVaporPressureDeficit', x='vpd', hue='cluster', s=6, data=df, legend='full')
sns.relplot(y='zP', x='pr', hue='cluster', s=6, data=df, legend='full')
sns.relplot(y='zPrecipitation', x='pr', hue='cluster', s=6, data=df, legend='full')
sns.relplot(y='zcum_daysinarow_lowpr', x='cum_daysinarow_lowpr', hue='cluster', s=4, data=df, legend='full')
#With means and stds
sns.relplot(y='NDVImean', x='NDVI', hue='cluster', s=4, data=df, legend='full')
sns.relplot(y='NDVIstd', x='NDVI', hue='cluster', s=4, data=df, legend='full')
sns.relplot(y='VPDmean', x='vpd', hue='cluster', s=7, data=df, legend='full')
sns.relplot(y='vpdstd', x='vpd', hue='cluster', s=7, data=df, legend='full')
sns.relplot(y='Pmean', x='pr', hue='cluster', s=7, data=df, legend='full')
sns.relplot(y='prstd', x='pr', hue='cluster', s=7, data=df, legend='full')
sns.relplot(y='prstd', x='Pmean', hue='cluster', s=9, data=df, legend='full')

#############
#Investigating P.mean == 0
###########
sns.relplot(y='Latitude', x='Longitude', hue='cluster', palette="seismic", s=15, legend = 'full', data=df[df.Pmean==0])
sns.relplot(y='Pmean', x='pr', hue='cluster', s=15, data=df[df.Pmean==0], legend='full')
sns.relplot(y='prstd', x='pr', hue='cluster', s=15, data=df[df.Pmean==0], legend='full')
sns.relplot(y='zP', x='pr', hue='cluster', s=15, data=df[df.Pmean==0], legend='full')


sns.relplot(y='avgtemp', x='tmmx', hue='cluster', s=4, data=df, legend='full')
sns.relplot(y='ztmmx', x='tmmx', hue='cluster', s=7, data=df, legend='full')
sns.relplot(y='avgtemp', x='vpd', hue='cluster', s=5, data=df, legend='full')

#With NDVI
sns.relplot(x='avgtemp', y='NDVI', hue='cluster', s=7, data=df, legend='full')
sns.relplot(x='vpd', y='NDVI', hue='cluster', s=7, data=df, legend='full')
sns.relplot(x='pr', y='NDVI', hue='cluster', s=7, data=df, legend='full', palette = 'seismic')
sns.relplot(x='cum_daysinarow_lowpr', y='NDVI', hue='cluster', s=7, data=df, legend='full')
sns.relplot(x='zVPD', y='NDVI', hue='cluster', s=7, data=df, legend='full')




#####################################################
#####################################################
#creating mean and std for met variables
#####################################################
#####################################################
dfMasterMetMonthly = df
dfMasterMetMonthly['VPDmean'] = dfMasterMetMonthly\
    .groupby(['system:indexviejuno', 'month'])['vpd']\
    .transform(np.mean)
dfMasterMetMonthly['Pmean'] = dfMasterMetMonthly\
    .groupby(['system:indexviejuno', 'month'])['pr']\
    .transform(np.mean)

VPDstd = dfMaster.groupby(['system:indexviejuno', 'month'])['vpd'].std()
dfMasterMetMonthly =dfMasterMetMonthly.join(VPDstd, on=['system:indexviejuno', 'month'], rsuffix='std')

prstd = dfMaster.groupby(['system:indexviejuno', 'month'])['pr'].std()
dfMasterMetMonthly =dfMasterMetMonthly.join(prstd, on=['system:indexviejuno', 'month'], rsuffix='std')
# adding z columns calculated the easy and intuitive way
dfMasterMetMonthly['zPrecipitation'] = (dfMasterMetMonthly.pr - dfMasterMetMonthly.Pmean)/dfMasterMetMonthly.prstd
dfMasterMetMonthly['zVaporPressureDeficit'] = (dfMasterMetMonthly.vpd - dfMasterMetMonthly.VPDmean)/dfMasterMetMonthly.vpdstd


#cumstd = dfMaster.groupby(['system:indexviejuno', 'month'])['cum_daysinarow_lowpr'].std()
#dfMasterMetMonthly =dfMasterMetMonthly.join(cumstd, on=['system:indexviejuno', 'month'], rsuffix='std')



df= dfMasterMetMonthly

#############################
#############################
#############################



##########################
############# violin plots
#########################

for i in range(8):
    sns.violinplot(x = df.loc[(df.cluster==i)].month, y = df.loc[(df.cluster==i)].NDVI)
    plt.show()

### Correlation Matrix
# creating subset df for easier visualizations
dfnew = df[[u'cluster', u'Latitude', u'Longitude', u'month',
       u'year', u'NDVI', u'EVI',
       u'NDWI1',
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5', u'pr','avgtemp', u'daysabove30',
       u'daysabove28', u'daysabove35', u'eto', u'zVPD', u'zP', u'zETO',
       u'zSRAD', u'zETR', u'ztmmn', u'ztmmx', u'zcum_daysinarow_lowpr',
       u'zdaysabove30']]
dfnew = df[[u'cluster', u'Latitude', u'Longitude',
       u'year', u'NDVI', 
        u'srad', u'etr', u'tmmx', u'tmmn', u'vpd',
       u'cum_daysinarow_lowpr', u'daysbelowneg5','daysabove30', u'pr','avgtemp', u'eto', u'zVPD', u'zP']]
#new df's for easier plotting grouping by cluster
df0 = dfnew.loc[(dfnew.cluster==0)]
df1= dfnew.loc[(dfnew.cluster==1)]
df2= dfnew.loc[(dfnew.cluster==2)]
df3= dfnew.loc[(dfnew.cluster==3)]



corr = df0.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')


pd.set_option('display.max_columns', 25)

df1= dfMaster[]



data_path = "../rawData/AgMet/"
df.to_csv("~/Documents/westernWheat/rawData/AgMet/df_24_April_3.3.18.csv")