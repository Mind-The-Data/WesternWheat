#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:26:07 2018

@author: brian
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns



crop_id = 24
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
dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 0) & (dfMaster['month']< 9)]


###########################
#locate summer months.
def selectmonths():
    if crop_id == 24:
        dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 0) & (dfMaster['month']< 9)]
    elif crop_id == 23:
        #dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 2) & (dfMaster['month']< 10)]
    else:
        print "What crop?"
    

selectmonths()

dfMasterJanuary =dfMasterSummer.loc[(dfMasterSummer.month==1)]
dfMasterFebruary = dfMasterSummer.loc[(dfMasterSummer.month==2)]
dfMasterMarch = dfMasterSummer.loc[(dfMasterSummer['month']==3)]
dfMasterApril = dfMasterSummer.loc[(dfMasterSummer['month']==4)]
dfMasterMay = dfMasterSummer.loc[(dfMasterSummer['month']==5)]
dfMasterJune = dfMasterSummer.loc[(dfMasterSummer['month']==6)]
dfMasterJuly = dfMasterSummer.loc[(dfMasterSummer['month']==7)]
dfMasterAugust = dfMasterSummer.loc[(dfMasterSummer['month']==8)]
dfMasterSeptember = dfMasterSummer.loc[(dfMasterSummer['month']==9)]

dfMasterSummer.rename(columns={'system:indexviejuno':'pixel'})
#dfMasterSummer= dfMaster.loc[(dfMaster['month']==4) | (dfMaster['month'] == 5) | (dfMaster['month'] == 6) | (dfMaster['month'] == 7)]


#univariate
t = dfMaster.groupby('system:indexviejuno')[['anomalyNDVI', 'zVPD']].corr(min_periods=10)
corr_vpd = t.loc[(t.index.get_level_values(0), 'anomalyNDVI'), 'zVPD']
t = dfMaster.groupby('system:indexviejuno')[['anomalyNDVI', 'zPrecip']].corr(min_periods=10)
corr_precip = t.loc[(t.index.get_level_values(0), 'anomalyNDVI'), 'zPrecip']

# bivariate linear parameters
NDVIparams = dfMasterSummer.groupby('system:indexviejuno')[['anomalyNDVI', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDVI'], x[['zPrecip', 'zVPD']], missing='drop').fit().params)
EVIresult = dfMasterSummer.groupby(['system:indexviejuno'])[['anomalyEVI', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyEVI'], x[['zPrecip', 'zVPD']], missing='drop').fit().rsquared_adj)
NDWI1params = dfMasterSummer.groupby('system:indexviejuno')[['anomalyNDWI1', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDWI1'], x[['zPrecip', 'zVPD']], missing='drop').fit().params)
# NDWI2 uses a larger wavelength of SWIR 2.08-2.35 in lieu of 1.55-1.75 nm used in NDWI1
NDWI2params = dfMasterSummer.groupby('system:indexviejuno')[['anomalyNDWI2', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDWI2'], x[['zPrecip', 'zVPD']], missing='drop').fit().params)

print EVIresult
print NDVIparams

#df23ndvi= df23ndvi[df23ndvi.index ==47883.0] 

# Rsquares for bivariate

#ndvi_model_results = dfMaster.groupby('system:indexviejuno')[['anomalyNDVI', 'zPrecip', 'zVPD']].apply(lambda x: sm.OLS(x['anomalyNDVI'], x[['zPrecip', 'zVPD']], missing='drop').fit())
#print ndvi_model_results[ndvi_model_results.index==28].values

"""
R_squared = []
Rsquared_adj = []

for pixel in range(id_pixel.size):
    model = sm.ols(formula='zNDVI ~ zVPD + zP', data = df23June[df23June.index==id_pixel[pixel]])
    results = model.fit()
    modeled_pixel_count += 1
    R_squared = np.append([results.rsquared],[R_squared])
    Rsquared_adj = np.append([results.rsquared_adj], [Rsquared_adj])
    #results.rsquared.append(Rsquared)
    #print results.params
    #print results.summary()
    print modeled_pixel_count
"""


#######################
#dataframes whose values will plot as color on basemap
#Marco's fiddling
######################
#param = (np.abs(corr_precip) - np.abs(corr_vpd))/(np.abs(corr_precip) + np.abs(corr_vpd))
#param = corr_precip
#param = corr_vpd
#param = (temp['zPrecip'].abs() - temp['zVPD'].abs())/(temp['zPrecip'].abs() + temp['zVPD'].abs())

#paramNDVI_VPD = NDVIparams['zVPD']
#paramNDWI1_VPD = NDWI1params['zVPD']
#paramNDWI2_VPD = NDWI2params['zVPD']
#paramEVI_VPD = EVIparams['zVPD']
#paramNDVI_P = NDVIparams['zPrecip']
#paramNDWI1_P = NDWI1params['zPrecip']
#paramNDWI2_P = NDWI2params['zPrecip']
#paramEVI_P = EVIparams['zPrecip']

# GEOGRAPHIC
shot = ndvi.groupby('system:indexviejuno')['.geo'].last()
geo = shot.apply(lambda x: x.replace("true", "True")).apply(eval).apply(pd.Series)
lons, lats = zip(*geo['coordinates'])dfMasterSummer = dfMaster.loc[(dfMaster['month'] > 0) & (dfMaster['month']< 9)]

#BASEMAP
map =Basemap(projection='stere', lon_0=-105, lat_0=90.,\
            llcrnrlat=29,urcrnrlat=49,\
            llcrnrlon=-117,urcrnrlon=-87.5,\
            rsphere=6371200., resolution='l', area_thresh=10000)
#MODEL

pvalues = dfMasterJune.groupby(['system:indexviejuno'])[['anomalyNDVI', 'zVPD', ]].apply(lambda x: sm.OLS(x['anomalyNDVI'], x[['zVPD']]).fit().pvalues)
params = dfMasterJune.groupby(['system:indexviejuno'])[['anomalyNDVI', 'zVPD', ]].apply(lambda x: sm.OLS(x['anomalyNDVI'], x[['zVPD']]).fit().params)
rsquared_adj = dfMasterJune.groupby(['system:indexviejuno'])[['anomalyNDVI', 'zVPD', ]].apply(lambda x: sm.OLS(x['anomalyNDVI'], x[['zVPD']]).fit().rsquared_adj)

#paramEVI_P = ParameterResults['zP_zVPD']
#paramEVI_VPD = ParameterResults['zVPD']

x, y = map(lons,lats)
x = np.array(x)
y = np.array(y)
map.drawcoastlines(linewidth=1)
map.drawstates()
map.drawcountries(linewidth=1.1)
map.drawmeridians(range(-140, -80, 5), linewidth=.3)
map.drawparallels(range(20, 60, 5),linewidth=.3)
#map.drawrivers(linewidth=.1)
#map.scatter(x[pvalues.zVPD.values<0.1], y[pvalues.zVPD<.1], c=params.zVPD.values[pvalues.zVPD<0.1], marker='^', cmap='seismic', alpha=1., s=5.8) #.
#map.scatter(x[pvalues.zVPD.values>0.1], y[pvalues.zVPD>0.1], c=params.zVPD.values[pvalues.zVPD>0.1], marker='v', cmap='seismic', alpha=1., s=5.8) #.
map.scatter(x, y, c=rsquared_adj.values, marker='^', cmap='seismic', alpha=1., s=5.8) #.
#map.scatter(x, y, c=pvalues, cmap='gray', alpha=.08, s=5.8) #.
plt.colorbar()
plt.clim(-1,1)

datapath = '../Images/OnlyzVPD/'
plt.savefig(datapath + str(crop_id) + "BzVPD_06June_onlypvalues<.1.png", dpi=500)


#sns.distplot(dfMasterJanuary.NDVI.dropna())
#plt.savefig(datapath + str(crop_id) + "NDVI_1january.png")


#sns.distplot(ParameterResults.zP_zVPD.dropna(), hist_kws={'histtype':'step', 'linewidth':2}, kde_kws={'label':'zP_zVPD_rsq'})
sns.distplot(pvalues.dropna(), hist_kws={'histtype':'step', 'linewidth': 1}, kde_kws={'label':'pvalues_zVPDonly'})
sns.distplot(rsquared_adj.dropna(), hist_kws={'histtype':'step', 'linewidth': 1}, kde_kws={'label':'rsq_adj_zVPDonly'})
plt.savefig(datapath + str(crop_id) + "rsquared&pvalues_dist_06June_onlyzVPD.png", dpi=300)