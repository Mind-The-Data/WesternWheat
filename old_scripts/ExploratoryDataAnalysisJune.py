# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import scipy



datapath = '../Data/Processed/'
df23June = pd.read_csv(datapath + 'df23June.csv')
df23June = df23June.rename(columns={'system:indexviejuno':'pixel'})
df23June.index=df23June['pixel']


###STATS#####

model = sm.ols(formula='NDVIanomaly ~ VPDanomaly + Panomaly', data = df23June).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()

#MeanNDVI.to_csv('../Data/Processed/meanJuneNDVI23.csv')
###########################
## PLOTTING
###########################
#Renaming variables
savepath = '../Images/JuneAcrossAllpixels'
FractionofYear = df23June['fractionofyear']
MeanNDVI = df23June['NDVImean']
MeanfoyNDVI = df23June['NDVI']
MeanVPD = df23June['VPDoverAllyearsmean']
MeanfoyVPD = df23June['VPDfractionofyearmean']
MeanP = df23June['PoverAllyearsmean']
MeanfoyP = df23June['Pfractionofyearmean']
VPD = df23June['VPD']
P = df23June['P']
NDVIz = df23June['NDVIanomaly']
Pz= df23June['Panomaly']
VPDz = df23June['VPDanomaly']
Latitude = df23June['Latitude']
Longitude = df23June['Longitude']

plt.figure(dpi=300)
plt.scatter(MeanNDVI, MeanP, s=.0005)
plt.xlabel('NDVI mean over all years')
plt.ylabel('Mean Precipitation over all years (mm/day)')
plt.title('Spring Wheat Precipitation and NDVI June averages')
#plt.savefig(savepath + '/JuneMeanPandNDVI.png')

plt.figure(dpi=300)
plt.scatter(MeanNDVI, MeanVPD, s=.0005, color='r')
plt.xlabel('Mean NDVI over all years')
plt.ylabel('Mean VPD over all years')
plt.title('Spring Wheat VPD and NDVI June averages')
plt.savefig(savepath + '/JuneMeanVPDandNDVI.png')

#plt.title('June NDVI (notice 115 mile swaths)')
#plt.savefig(savepath + '/MeanNDVIdistribution.png')

plt.figure(dpi=300)
sns.kdeplot(MeanVPD)
plt.title('VPD oay mean')
plt.ylabel('frequency')
plt.xlabel('Vapor Pressure Difference Over All years, (kPa)')
#plt.savefig(savepath + '/VPDoverallyearsrmean.png')

plt.figure(dpi=300)
sns.kdeplot(MeanfoyP)
plt.ylabel('frequency')
plt.xlabel('Precipitation Fraction of year Mean (mm/day)')
plt.title('Precipitation mean fraction of year')
#plt.savefig(savepath + '/PMeanFoY.png')


plt.figure(dpi=300)
sns.kdeplot(MeanNDVI, color='g')
plt.ylabel('frequency')
plt.xlabel('NDVI mean')
plt.title('NDVI mean (greeness)')
#plt.savefig(savepath + '/NDVImean.png')

sns.kdeplot(P, MeanNDVI)
sns.distplot(df23June['VPDoverAllyearsmean'])


#plt.savefig(savepath + '/VPDdistribution', dpi=300)
sns.violinplot([P,VPD])
sns.distplot(MeanNDVI)
#plt.savefig(savepath, dpi=300, figsize=[5,5])
sns.distplot(df23June['PoverAllyearsmean'])
#plt.savefig(savepath + '/Pyearmeanhisto', dpi=300, figsize=[5,5])


sns.kdeplot(NDVIz, color="g")
plt.ylabel('frequency')
#plt.savefig(savepath + 'NDVIz', dpi=300, figsize=[5,5])
sns.kdeplot(Pz, color="b")
#plt.savefig(savepath + '/Pz', dpi=300, figsize=[5,5])
sns.kdeplot(VPDz, color="r")
#plt.savefig(savepath + '/VPDz', dpi=300, figsize=[5,5])
#plt.savefig(savepath + '/NDVIpVPDz', dpi=300, figsize=[5,5])


#sns.kdeplot(Pz, VPDz)

#####################
######### lat, long
#####################
plt.figure(dpi=300)
plt.scatter(Longitude, Latitude, s=.0001)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spring Wheat Fields under study')
#plt.savefig(savepath + '/Locations.png')

###### constraints
# WESTERN or other specific locations
#https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/
######## subsets


dflocations = df23June.loc[(df23June['PoverAllyearsmean']<=1) & (df23June['VPDoverAllyearsmean']>= 1.5) & (df23June['NDVI']>= .75)]
#  & (df23June['PoverAllyearsmean']<=2)
#& (df23June['VPDoverAllyearsmean']>= 1.5)
# (df23June["Longitude"]<= -100) & 
# (df23June['VPDoverAllyearsmean']>= 1.5) &
#  & (df23June['cropland_mode']==23)
# (df23June['PoverAllyearsmean']>=1.3) & (df23June['VPDoverAllyearsmean']>= 1.5)
plt.figure(dpi=300)
sns.kdeplot(dflocations['NDVIanomaly'], color='g')
#plt.savefig(savepath + '/NDVIzWestOf100', dpi=300)
sns.distplot(dflocations['Panomaly'], color='b')
#plt.savefig(savepath + '/PzEastOf100histo', dpi=300)
sns.distplot(dflocations['VPDanomaly'], color='r')
plt.savefig(savepath + '/NDVIzPzVPDzlowPhighVPDhighNDVI', dpi=300)
#sns.distplot(dflocations['srad'])
model = sm.ols(formula='NDVIanomaly ~ VPDanomaly + Panomaly', data = dflocations).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()
#### correlation
dflocations['VPD'].corr(dflocations['Panomaly'])
#df23June['VPDanomaly'].corr(df23June['PanomalyMean'])

"""
# picking out high VPDz's

dflocationsandVPDs = df23June.loc[(df23June["Longitude"]<= -100) & (df23June['VPDoverAllyearsmean']>= 1.5)]
dflocationsandVPDs.info()

sns.kdeplot(dflocationsandVPDs['NDVIanomaly'])
#plt.savefig(savepath + '/NDVIzWestOf100VPDmeanOveronenhalf', dpi=300)

sns.kdeplot(dflocationsandVPDs['VPDanomaly'])
#plt.savefig(savepath + '/VPDzWestOf100VPDmeanOveronenhalf', dpi=300)

sns.kdeplot(dflocationsandVPDs['Panomaly'])
#plt.savefig(savepath + '/PzWestOf100VPDmeanOveronenhalf', dpi=300)

model = sm.ols(formula='NDVIanomaly ~ VPDanomaly + Panomaly', data = dflocationsandVPDs).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()
##########################3
"""
df = df23June.loc[(df23June["Longitude"]<= -100) & (df23June['VPDoverAllyearsmean']>= 1.5) & (df23June['PoverAllyearsmean']<=2) & (df23June['cropland_mode']==23)]
df.info()

model = sm.ols(formula='NDVIanomaly ~ VPDanomaly + Panomaly', data = dflocations).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()


"""

sns.kdeplot(df['Panomaly'])
#plt.savefig(savepath + '/PzWestOf100VPDmeanOveronenhalf', dpi=300)

sns.kdeplot(df['VPDanomaly'])
#plt.savefig(savepath + '/VPDzWestOf100VPDmeanOveronenhalf', dpi=300)

sns.kdeplot(df['NDVIanomaly'])
plt.savefig(savepath + '/NDVIzVPDzPzonly23', dpi=300)



model = sm.ols(formula='NDVIanomaly ~ VPDanomaly + Panomaly', data = df).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()



dfVPDsandPs = df23June.loc[(df23June["Longitude"]>= -100) & (df23June['VPDoverAllyearsmean']>= 1.5) & (df23June['PoverAllyearsmean']<=2)]

sns.kdeplot(dfVPDsandPs['NDVIanomaly'])
#plt.savefig(savepath + '/NDVIzWestOf100VPDmeanOveronenhalf', dpi=300)

sns.kdeplot(dfVPDsandPs['VPDanomaly'])
#plt.savefig(savepath + '/VPDzWestOf100VPDmeanOveronenhalf', dpi=300)

sns.kdeplot(dfVPDsandPs['Panomaly'])
#plt.savefig(savepath + '/PzWestOf100VPDmeanOveronenhalf', dpi=300)
#plt.savefig(savepath + '/PzVPDzNzEastof100V>onenhalfP<two', dpi=300)

model = sm.ols(formula='NDVIanomaly ~ VPDanomaly + Panomaly', data = dflocationsandVPDs).fit()
#res_fit = sm.OLS(ols_resid[1:], ols_resid[;-1]).fit
print model.params
print model.summary()
"""