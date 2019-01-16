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
#sns.set(style="darkgrid")



######################
# statistics by cluster
#######################


#df = df_labeled
df = None
df=maxdf_labeled
df = maxdf

#####################
## CLUSTER LOCATIONS
####################
sns.relplot(y='Latitude', x='Longitude', hue='cluster', palette="Paired", s=8, legend = 'full', data=df)
### plot by year
years = np.arange(2008,2018, step=1)
for i in years:    
    sns.regplot(y='Latitude', x='Longitude', hue='NDVIsum3', palette="Paired", s=20, legend = 'full', data=df[df.year==i])
# How many unique pixels in each cluster
for i in range(pd.unique(df.cluster).size):
    print pd.unique(df[df.cluster==i].pixel).size
    

###########################
# CORRELATION GRAPHS
############################

for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zdrydayssum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    g = g.plot_joint(plt.scatter, color='g', s=15, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=False, color='g')
    g = g.annotate(st.pearsonr, fontsize='xx-small')
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zGDDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    g = g.plot_joint(plt.scatter, color='g', s=15, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=False, color='g')
    g = g.annotate(st.pearsonr, fontsize='xx-small')
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVPDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    g = g.plot_joint(plt.scatter, color='g', s=15, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=False, color='g')
    g = g.annotate(st.pearsonr, fontsize='xx-small')
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrsum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    g = g.plot_joint(plt.scatter, color='g', s=15, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=False, color='g')
    g = g.annotate(st.pearsonr, fontsize='xx-small')

'''
###### Not by cluster
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zGDDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVPDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrsum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
######################
'''    
    
    
    
    
    

######### JOINTGRID method
    
g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
g = g.plot_joint(plt.scatter, color='g', s=5, edgecolor='white')
g = g.plot_marginals(sns.distplot, kde=False, color='g')
g = g.annotate(st.pearsonr, fontsize=('xx-small'))RAM...\n'

############## regplot method

for cluster in range(pd.unique(df.cluster).size):
    g = sns.regplot(x='zdrydayssum3', y ='zNDVIsum3', data = df.loc[(df.cluster==cluster)], ci=99,  scatter_kws={'s':.1})
    #g = sns.jointplot(df.loc[(df.cluster==cluster)].zCumdlowprsum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
    #g = sns.JointGrid(x='zPrsum3', y = 'zNDVIsum3', data=df)
    #g = g.plot_joint(plt.scatter, color='g', s=5, edgecolor='white')
    #g = g.plot_marginals(sns.distplot, kde=False, color='g')
    #g = g.annotate(st.pearsonr)

for cluster in range(pd.unique(df.cluster).size):
    g = sns.regplot(x='zPrsum3', y ='zNDVIsum3', data = df.loc[(df.cluster==cluster)], ci=99,  scatter_kws={'s':.1}, label=cluster)
    print (cluster)
for cluster in range(pd.unique(df.cluster).size):
    g = sns.regplot(x='zVPDmean3', y ='zNDVIsum3', data = df.loc[(df.cluster==cluster)], ci=99,  scatter_kws={'s':.1})
    plt.show()
for cluster in range(pd.unique(df.cluster).size):
    g = sns.regplot(x='zGDDmean3', y ='zNDVIsum3', data = df.loc[(df.cluster==cluster)], ci=99,  scatter_kws={'s':.1})


#########

    
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrsum3, df.loc[(df.cluster==cluster)].NDVIsum3, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVPDmean3, df.loc[(df.cluster==cluster)].NDVIsum3, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zCumdlowprsum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.1})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zGDDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.1})

## LOWESS
w = lowess(maxdf.prsum3mean, maxdf.NDVIsum3mean, frac=2./3)
plt.plot(w[:,1],w[:,0], 'g')
plt.scatter(maxdf.prsum3mean, maxdf.NDVIsum3,s=.1)


#
for cluster in range(pd.unique(df.cluster).size):
    g = sns.kdeplot(df.loc[(df.cluster==cluster)].NDVI)

# met correlation graphs



# CORRELATION r VALUES

    #VPD
for cluster in range(pd.unique(df.cluster).size):    
    print 'zVPD r values for cluster ', cluster
    print np.round(df.loc[(df.cluster==cluster)].zVPDmean3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    #PRECIP
for cluster in range(pd.unique(df.cluster).size):    
    print 'zP r values for cluster ', cluster
    print np.round(df.loc[(df.cluster==cluster)].zPrsum3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    #CUMALATIVE
for cluster in range(pd.unique(df.cluster).size):  
    print np.round(df.loc[(df.cluster==cluster)].zdrydayssum3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
    #AVGTEMP
for cluster in range(pd.unique(df.cluster).size):   
    print np.round(df.loc[(df.cluster==cluster)].zGDDmean3.corr(df.loc[(df.cluster==cluster)].zNDVIsum3),2)
##########################
######## YEARLY GRAPHS
######################

for year in range(pd.unique(df.year).size):
     g = sns.jointplot(df.loc[(df.year==year)].zdrydayssum3, df.loc[(df.year==year)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
     for cluster in range(pd.unique(df.cluster).size):
        g = sns.jointplot(df.loc[(df.cluster==cluster)].zdrydayssum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zGDDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zVPDmean3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].zPrsum3, df.loc[(df.cluster==cluster)].zNDVIsum3, kind='reg', scatter_kws={'s':.2})

### Other variables
    
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].zVPD.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].zP.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].daysabove28.corr(df.loc[(df.cluster==cluster)].zNDVI)
#for cluster in range(pd.unique(df.cluster).size):    
#    print df.loc[(df.cluster==cluster)].ztmmx.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].ztmmn.corr(df.loc[(df.cluster==cluster)].zNDVI)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].EVI)

#met correlation graphs
    
for cluster in range(pd.unique(df.cluster).size):
    g = sns.jointplot(df.loc[(df.cluster==cluster)].pr, df.loc[(df.cluster==cluster)].vpd, kind='reg', scatter_kws={'s':.1})

# Met correlate heat graphs
df1 = None
df1 = df[['prsum3','GDDmean3','VPDmean3','dayslowprsum3', 'NDVIsum3']]
g = sns.heatmap(df1)#, col='cluster')
for cluster in range(pd.unique(df.cluster).size):
    corr = df1.loc[(df.cluster==cluster)].corr()
    g = sns.heatmap(df1.loc[(df.cluster==cluster)].corr(), 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')
    g.plot()
   



#met correlates    
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].pr)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].avgtemp)
for cluster in range(pd.unique(df.cluster).size):    
    print df.loc[(df.cluster==cluster)].vpd.corr(df.loc[(df.cluster==cluster)].dayslowprsum3)
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






#cumstd = dfMaster.groupby(['system:indexviejuno', 'month'])['cum_daysinarow_lowpr'].std()
#dfMasterMetMonthly =dfMasterMetMonthly.join(cumstd, on=['system:indexviejuno', 'month'], rsuffix='std')



df= dfMasterMetMonthly

#############################
#############################
#############################



##########################
############# violin plots
#########################

for i in range(4):
    sns.violinplot(x = df.loc[(df.cluster==i)].month_y, y = df.loc[(df.cluster==i)].NDVI)
    plt.show()
    
for i in range(pd.unique(df.cluster).size):
    sns.violinplot(x = df.loc[(df.cluster==i)].month_y, y = df.loc[(df.cluster==i)].maxNDVIsum3)
    plt.show()

for i in range(pd.unique(df.cluster).size):
    sns.violinplot(x = df.loc[(df.cluster==i)].month_y, y = df.loc[(df.cluster==i)].VPD)
    plt.show()    
for i in range(pd.unique(df.cluster).size):
    sns.violinplot(y=df.cluster, x = df.loc[(df.cluster==i)].vpd)
    plt.show()
    
sns.violinplot(x='cluster', y='pr', data=df)
sns.violinplot(x='cluster', y='GDD', data=df)
sns.violinplot(x='cluster', y='vpd', data=df)
sns.violinplot(x='cluster', y='NDVI', data=df)
sns.violinplot(x='cluster', y='dayslowpr', data=df)
sns.violinplot(x='cluster', y='eto', data=df)

sns.violinplot(x='cluster', y='prsum3', data=df)
sns.violinplot(x='cluster', y='GDDmean3', data=df)
sns.violinplot(x='cluster', y='VPDmean3', data=df)
sns.violinplot(x='cluster', y='NDVIsum3', data=df)
sns.violinplot(x='cluster', y='dayslowprsum3', data=df)


#### These anomaly plots show that some pixels must be switching from cluster to cluster...
sns.violinplot(x='cluster', y='zPrsum3', data=df)
sns.violinplot(x='cluster', y='zVPDmean3', data=df)
sns.violinplot(x='cluster', y='zGDDmean3', data=df)
sns.violinplot(x='cluster', y='zNDVIsum3', data=df)
sns.violinplot(x='cluster', y='zCumdlowprsum3', data=df)




########### other variables

sns.violinplot(x='cluster', y='srad', data=df)
sns.violinplot(x='cluster', y='month', data=df)
sns.violinplot(x='cluster', y='avgtemp', data=df)
sns.violinplot(x='cluster', y='daysabove28', data=df)
sns.violinplot(x='cluster', y='daysabove30', data=df)
sns.violinplot(x='cluster', y='daysabove35', data=df)
sns.violinplot(x='cluster', y='tmmn', data=df)
sns.violinplot(x='cluster', y='tmmx', data=df)



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

#########################################
################# Correlation Plots
########################################


for c in range(pd.unique(df.cluster).size):   
    print 'cluster: ', c
    dfc = df[df.cluster==c]
    sns.kdeplot(dfc.NDVIsum3mean, dfc.prsum3mean)
    plt.show()


# maxdf has twice as many points as for ndvi as met variables, aren't months to sum over. 
sns.kdeplot(maxdf.NDVIsum3mean, maxdf.prsum3mean, n_levels=20)
sns.kdeplot(maxdf.NDVIsum3mean, maxdf.VPDmean3mean, n_levels=20)
sns.kdeplot(maxdf.NDVIsum3, maxdf.VPDmean3)
sns.kdeplot(maxdf.NDVIsum3, maxdf.prsum3)
sns.kdeplot(maxdf.NDVIsum3, maxdf.prsum3)
sns.kdeplot(maxdf.zVPD, maxdf.zP)
sns.kdeplot(maxdf.vpdmean, maxdf.prmean, n_levels=30)
sns.kdeplot(maxdf.vpd, maxdf.pr, n_levels=20)
sns.kdeplot(maxdf.vpdmean, maxdf.NDVImean, n_levels=20)
sns.kdeplot(maxdf.prmean, maxdf.NDVImean, n_levels=20)

sns.kdeplot(maxdf.zPrsum3, maxdf.zNDVIsum3, n_levels=20)
sns.kdeplot(maxdf.zVPDmean3, maxdf.zNDVIsum3, n_levels=20)
# non max df
maxdf = maxdf.dropna(how='any')


sns.kdeplot(df.NDVIsum3mean, df.VPDmean3mean)
sns.kdeplot(df.NDVIsum3, df.VPDmean3)
sns.kdeplot(df.NDVIsum3, df.prsum3)
sns.kdeplot(df.NDVIsum3, df.prsum3)
sns.kdeplot(df.zVPD, df.zP)
sns.kdeplot(df.vpdmean, df.prmean, n_levels=30)
sns.kdeplot(df.vpd, df.pr, n_levels=20)
sns.kdeplot(df.vpdmean, df.NDVImean, n_levels=20)
sns.kdeplot(df.prmean, df.NDVImean, n_levels=20)



        

data_path = "../rawData/AgMet/"
df.to_csv("~/Documents/westernWheat/rawData/AgMet/df_24_April_3.3.18.csv")