#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:53:06 2018

@author: brian
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as st


crop_id = 23
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'

met = pd.read_csv(data_path + 'MeteorMonthly_' + crop_id + ".csv", index_col='system:indexviejuno')
ndvi = pd.read_csv(data_path + 'VegInd' + crop_id + ".csv", index_col='system:indexviejuno')

#resetting index because default index is system:indexviejuno and we would rather have dates
met = met.reset_index()
ndvi = ndvi.reset_index()
met.index = met['date']
ndvi.index = ndvi['date']

# merge datasets to perform correlations
# merge combines by index and by columns that have the same name and merges values that line up
dfMaster = ndvi.merge(met)
# reset index to.... 
#dfMaster= dfMaster.resest_index()
dfMaster.index = dfMaster['system:indexviejuno']
#Insert Interaction column
#dfMaster['zP_zVPD']= (dfMaster.zPrecip)*(dfMaster.zVPD)
#rename column
#dfMaster.rename(columns= {'zPrecip':'zP'}, inplace=True)
dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 3) & (dfMaster['month']< 6)]


###########################
#locate summer months.
'''
def selectmonths():
    if crop_id == 24:
        dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 0) & (dfMaster['month']< 9)]
    elif crop_id == 23:
        #dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 2) & (dfMaster['month']< 10)]
    else:
        print "What crop?"

    return dfMasterSummer
selectmonths()
'''


dfMasterJanuary =dfMasterSummer.loc[(dfMasterSummer.month==1)]
dfMasterFebruary = dfMasterSummer.loc[(dfMasterSummer.month==2)]
dfMasterMarch = dfMasterSummer.loc[(dfMasterSummer['month']==3)]
dfMasterApril = dfMasterSummer.loc[(dfMasterSummer['month']==4)]
dfMasterMay = dfMasterSummer.loc[(dfMasterSummer['month']==5)]
dfMasterJune = dfMasterSummer.loc[(dfMasterSummer['month']==6)]
dfMasterJuly = dfMasterSummer.loc[(dfMasterSummer['month']==7)]
dfMasterAugust = dfMasterSummer.loc[(dfMasterSummer['month']==8)]
dfMasterSeptember = dfMasterSummer.loc[(dfMasterSummer['month']==9)]

#dfMasterSummer.rename(columns={'system:indexviejuno':'pixel'})
#dfMasterSummer= dfMaster.loc[(dfMaster['month']==4) | (dfMaster['month'] == 5) | (dfMaster['month'] == 6) | (dfMaster['month'] == 7)]


#univariate
#t = dfMaster.groupby('system:indexviejuno')[['anomalyNDVI', 'zVPD']].corr(min_periods=10)
#corr_vpd = t.loc[(t.index.get_level_values(0), 'anomalyNDVI'), 'zVPD']
#t = dfMaster.groupby('system:indexviejuno')[['anomalyNDVI', 'zPrecip']].corr(min_periods=10)
#corr_precip = t.loc[(t.index.get_level_values(0), 'anomalyNDVI'), 'zPrecip']

# bivariate linear parameters
NDVIparams = dfMasterSummer.groupby('system:indexviejuno')[['anomalyNDVI', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDVI'], x[['zPrecip', 'zVPD']], missing='drop').fit().params)
EVIresult = dfMasterSummer.groupby(['system:indexviejuno'])[['anomalyEVI', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyEVI'], x[['zPrecip', 'zVPD']], missing='drop').fit().rsquared_adj)
NDWI1params = dfMasterSummer.groupby('system:indexviejuno')[['anomalyNDWI1', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDWI1'], x[['zPrecip', 'zVPD']], missing='drop').fit().params)
# NDWI2 uses a larger wavelength of SWIR 2.08-2.35 in lieu of 1.55-1.75 nm used in NDWI1
NDWI2params = dfMasterSummer.groupby('system:indexviejuno')[['anomalyNDWI2', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDWI2'], x[['zPrecip', 'zVPD']], missing='drop').fit().params)




# GEOGRAPHIC FOR Select DATE and therefore fewer pixels
date = '2011-04'
selected_date_ndvi_df = ndvi.loc[(ndvi['date']==date)]
shot = selected_date_ndvi_df.groupby('system:indexviejuno')['.geo'].last()
geo = shot.apply(lambda x: x.replace("true", "True")).apply(eval).apply(pd.Series)
lons, lats = zip(*geo['coordinates'])
###############################
# GEOGRAPHIC FOR all 2000 pixels (when all years are plotted)
#################################
shot = ndvi.groupby('system:indexviejuno')['.geo'].last()
geo = shot.apply(lambda x: x.replace("true", "True")).apply(eval).apply(pd.Series)
lons, lats = zip(*geo['coordinates'])

#BASEMAP
map =Basemap(projection='stere', lon_0=-105, lat_0=90.,\
            llcrnrlat=29,urcrnrlat=49,\
            llcrnrlon=-117,urcrnrlon=-87.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)

############
#STAT MODEL
############

pvalues = dfMasterSummer.groupby(['system:indexviejuno'])[['anomalyEVI', 'zETO']].apply(lambda x: sm.OLS(x['anomalyEVI'], x['zETO']).fit().pvalues)
params = dfMasterSummer.groupby(['system:indexviejuno'])[['anomalyEVI', 'zETO']].apply(lambda x: sm.OLS(x['anomalyEVI'], x['zETO']).fit().params)
rsquared_adj = dfMasterSummer.groupby(['system:indexviejuno'])[['anomalyEVI', 'zETO']].apply(lambda x: sm.OLS(x['anomalyEVI'], x['zETO']).fit().rsquared_adj)
rsquared = dfMasterSummer.groupby(['system:indexviejuno'])[['anomalyEVI', 'zETO']].apply(lambda x: sm.OLS(x['anomalyEVI'], x['zETO']).fit().rsquared)


## zNDVI zEVI
#zNDVI = dfMasterSummer.anomalyNDVI.loc[(dfMasterSummer['date']==date)]
#zEVI = dfMasterSummer.anomalyEVI.loc[(dfMasterSummer['date']==date)]
#zVPD = dfMasterSummer.zVPD.loc[(dfMasterSummer['date']==date)]


def mappingfunction():
    x, y = map(lons,lats)
    x = np.array(x)
    y = np.array(y)
    map.drawcoastlines(linewidth=1)
    map.drawstates()
    map.drawcountries(linewidth=1.1)
    map.drawmeridians(range(-140, -80, 5), linewidth=.3)
    map.drawparallels(range(20, 60, 5),linewidth=.3)
    #map.drawrivers(linewidth=.1)
    #map.scatter(x[pvalues.zVPD<0.05], y[pvalues.zVPD<.05], c=params.zVPD[pvalues.zVPD<0.05], cmap='seismic', alpha=.8, s=.5)
    map.scatter(x[pvalues.zETO.values<0.05], y[pvalues.zETO<0.05], c=params.zETO.values[pvalues.zETO<.05], cmap='seismic', alpha=.8, s=.5)#.
    #map.scatter(x, y, c=params.zVPD, cmap='seismic', s=.1)
    #map.scatter(x, y, c=rsquared_adj.values, marker='^', cmap='RdYlGn', alpha=1., s=5.8) #.
    #map.scatter(x, y, c=pvalues.zVPD, cmap='bone', alpha=.8, s=1) 
    #plt.clim(0,.1)
    #map.scatter(x,y, c=zNDVI, cmap='BrBG', s=1)
    #map.scatter(x,y, c=zVPD, cmap='seismic', s=1)
    #map.scatter(x,y, c=zEVI, cmap='BrBG', s=1)
    plt.colorbar()
    plt.clim(-1,1)

mappingfunction()


datapath = '/home/brian/Documents/westernWheat/Images/InitialMaps/OnlyzVPD/zEVI/'
plt.savefig(datapath + str(crop_id) + "Bzpvalues<.05_04-05_zEVI_zETO.png", dpi=500)



#sns.distplot(dfMasterJanuary.NDVI.dropna())
#plt.savefig(datapath + str(crop_id) + "NDVI_1january.png")
#sns.distplot(ParameterResults.zP_zVPD.dropna(), hist_kws={'histtype':'step', 'linewidth':2}, kde_kws={'label':'zP_zVPD_rsq'})
#sns.distplot(pvalues.zETO.dropna(), hist_kws={'histtype':'step', 'linewidth': 2}, kde_kws={'label':'pvalues_zETO'})
sns.distplot(pvalues.zETO.dropna(), hist_kws={'histtype':'step', 'linewidth': 2}, kde_kws={'label':'pvalues_zETO_zEVI'})
sns.distplot(rsquared_adj.dropna(), hist_kws={'histtype':'step', 'linewidth': 2}, kde_kws={'label':'rsq_adj_zETO_zEVI'})
sns.distplot(rsquared.dropna(), hist_kws={'histtype':'step', 'linewidth': 2}, kde_kws={'label':'rsq_zETO_zEVI'})

plt.savefig(datapath + str(crop_id) + "pvalues&rsqadj&rsq_04-05_zEVI-zETO.png", dpi=300)

point = dfMasterMay[dfMasterMay.index==150991.0]
plt.scatter(dfMasterSummer.meanvpd, dfMasterSummer.NDVI, s=.1)



dfMaster['meanvpd']=dfMaster.groupby(['system:indexviejuno', 'month', 'year'])['vpd'].transform(np.mean)