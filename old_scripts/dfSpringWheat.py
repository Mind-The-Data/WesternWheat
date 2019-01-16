

import numpy as np
import pandas as pd
import glob
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

data_path = '../Data/AgMet/'

df23ndvi = pd.DataFrame()
for f in sorted(glob.glob(data_path + 'monthlyLSclass23*')):
    df23ndvi = pd.concat((df23ndvi, pd.read_csv(f)))


df23ndvi.index = df23ndvi['system:indexviejuno']
#df23ndvi= df23ndvi[df23ndvi.index ==47883.0]   #selecting single pixel 47883

datapathMet = '..'
dfMet23 = pd.read_csv(datapathMet + '/Data/AgMet/monthlymeteo23.csv')

dfMet23.index = dfMet23['system:indexviejuno']
#dfMet23 = dfMet23[dfMet23.index == 47883.0] #selecting single pixel 47883

frames = [df23ndvi, dfMet23]
df23 = pd.concat(frames)

#df23.to_csv('..Data/df23concat.csv')




#renaming columns
df23 = df23.rename(columns={'vpd':'VPD'})
df23 = df23.rename(columns={'pr':'P'})

#adding a new column: 'month'
df23['month'] = ((df23['fractionofyear'] - df23['fractionofyear'].astype(int))*12).round().astype(int)

##################
#VPD #############
##################
overAllyearsmeanVPD = df23.groupby(['system:indexviejuno', 'month'])['VPD'].mean()
overAllyearsstdVPD = df23.groupby(['system:indexviejuno', 'month'])['VPD'].std()
fractionofyearVPD = df23.groupby(['system:indexviejuno', 'fractionofyear'])['VPD'].mean()
df23 = df23.join(overAllyearsmeanVPD, on=['system:indexviejuno', 'month'], rsuffix='overAllyearsmean')
df23 = df23.join(overAllyearsstdVPD, on=['system:indexviejuno', 'month'], rsuffix='overAllyearsstd')
df23 = df23.join(fractionofyearVPD, on=['system:indexviejuno', 'fractionofyear'], rsuffix='fractionofyearmean')
df23['VPDanomaly'] = (df23['VPDfractionofyearmean'] - df23['VPDoverAllyearsmean'])/df23['VPDoverAllyearsstd']



#################
### Precipitation
################
meanP = df23.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df23.groupby(['system:indexviejuno', 'month'])['P'].std() 
df23 = df23.join(meanP, on=['system:indexviejuno', 'month'], rsuffix='overAllyearsmean')
df23 = df23.join(stdP, on=['system:indexviejuno', 'month'], rsuffix='overAllyearsstd')
fractionofyearmeanP = df23.groupby(['system:indexviejuno', 'fractionofyear'])['P'].mean()
df23 = df23.join(fractionofyearmeanP, on=['system:indexviejuno', 'fractionofyear'], rsuffix='fractionofyearmean')
df23['Panomaly']=(df23['Pfractionofyearmean']-df23['PoverAllyearsmean'])/df23['PoverAllyearsstd']
print df23.info()

##############
#NDVI
###########
df23['NDVI'] = (df23['B4'] - df23['B3'])/(df23['B4'] + df23['B3'])
df23['month'] = ((df23['fractionofyear'] - df23['fractionofyear'].astype(int))*12).round().astype(int)
meanNDVI = df23.groupby(['system:indexviejuno', 'month'])['NDVI'].mean()
stdNDVI = df23.groupby(['system:indexviejuno', 'month'])['NDVI'].std()
df23 = df23.join(meanNDVI, on=['system:indexviejuno', 'month'], rsuffix='mean')
df23 = df23.join(stdNDVI, on=['system:indexviejuno', 'month'], rsuffix='std')
df23['NDVIanomaly'] = (df23['NDVI'] - df23['NDVImean']) / df23['NDVIstd']

# Monthly dfs
#df23July = df23[df23['month']==7]
df23June = df23[df23['month'] == 6]
#df23May = df23[df23['month'] == 5]

#df23June.to_csv('df23June1.csv')
"""
#summerframes would have to have renamed columns or
ls #use a boolean mask to sort which months are which.. this is the preferred method
#summerframes = [df23July, df23June, df23May]
#df23Summer = pd.concat(summerframes)

#TRYING TO ADD MAYPRECIP TO JUNE DF SO I CAN DO AN OLS WITH JUNE NDVI, VPD AND MAYP
## having trouble concatenating, but this worked.
#df23May = df23May.rename(columns={'PanomalyMean':'PanomalyMeanMay'})
#df23MayJustPanomalyMean = df23May['PanomalyMeanMay']
#df23JunewithMayP= df23June.join(df23MayJustPanomalyMean)

#print df23JunewithMayP.info()
"""
#######################
######################
#Statistics
####################
###################

#### correlation
df23June['VPDanomaly'].corr(df23June['NDVIanomaly'])
df23June['NDVIanomaly'].corr(df23June['Panomaly'])

## Ordinary Least Squares
model = sm.ols(formula="NDVIanomaly ~ Panomaly + VPDanomaly + Panomaly*VPDanomaly +tmmx", data = df23June).fit()
model2 = sm.ols(formula='NDVIanomaly ~Panomaly', data = df23June).fit()
print model.params
print model.summary()
#print model2.params
#print model2.summary()

"""


###########################
## PLOTTING
###########################

P = df23June['Panomaly']
V = df23June['VPDanomaly']
N = df23June['NDVIanomaly']
F = df23['fractionofyear']

fig, ax=plt.subplots(figsize=(20,10))

ax.scatter(V,N, s=1)



fig, ax=plt.subplots(figsize=(20,10))

ax.scatter(V,P, s=.02)
ax.set(xlabel='VPD anomaly', ylabel='Precipitation anomaly', title='June Meterological Correlation across Spring Wheat(23) pixels averaged over all years')
fig.savefig("../Images/Graphs/PandVPDcorrJuneSpringWheat.png")

fig, ax=plt.subplots(figsize=(20,10))
ax.scatter(V,N, s=.02)
ax.set(xlabel='VPDanomaly', ylabel='NDVIanomaly', title='June across all Spring Wheat pixels, 2008-2017')
fig.savefig("../Images/Graphs/JuneVPDandNDVIspringWheat23.png")


plt.figure(figsize=(20, 10))
plt.scatter(P,N, s=.025)
plt.ylabel('NDVIanomaly')
plt.xlabel('Panomaly')
plt.title('June Spring Wheat (23), all pixels 2008-2017')
plt.savefig('../Images/Graphs/JuneSpringWheatNDVIandP.png')

ax.scatter(V,P, s=.01)
ax.set(xlabel='VPD anomaly', ylabel='Precipitation anomaly', title='June Meterological Correlation across Spring Wheat(23) pixels averaged over all years')
#fig.savefig("../Images/Graphs/PandVPDcorrJuneSpringWheat.png")

fig, ax=plt.subplots(figsize=(20,10))
ax.scatter(V,N, s=.02)
ax.set(xlabel='VPDanomaly', ylabel='NDVIanomaly', title='June across all Spring Wheat pixels, 2008-2017')
#fig.savefig("../Images/Graphs/JuneVPDandNDVIspringWheat23.png")


plt.figure(figsize=(20, 10))
plt.scatter(P,N, s=.025)
plt.ylabel('NDVIanomaly')
plt.xlabel('Panomaly')
#plt.title('June Spring Wheat (23), all pixels 2008-2017')
#plt.savefig('../Images/Graphs/JuneSpringWheatNDVIandP.png')






###### pixel indentification

id_pixel = pd.unique(df23.index)
p = df23[df23.index==id_pixel[0]]
p500 = df23[df23.index==id_pixel[500]]
p1000 = df23[df23.index==id_pixel[1000]]
p1500 = df23[df23.index==id_pixel[1500]]
## plotting 
plt.figure(1, figsize=(20,10))
#plt.scatter(p['fractionofyear'], p['VPD'])
plt.plot(p['fractionofyear'], p['NDVIanomaly'], 'green')
plt.scatter(p['fractionofyear'],p['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p['fractionofyear'], p['PanomalyMean'], color='b')
#plt.plot(p['fractionofyear'], p['NDVImean'])
#plt.plot(p['fractionofyear'], p['NDVIstd'])

plt.figure(2, figsize=(20,10))
plt.plot(p500['fractionofyear'], p500['NDVIanomaly'], 'green')
plt.scatter(p500['fractionofyear'],p500['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p500['fractionofyear'], p500['PanomalyMean'], color='b')


plt.figure(3, figsize=(20,10))
plt.plot(p1000['fractionofyear'], p1000['NDVIanomaly'], 'green')
plt.scatter(p1000['fractionofyear'],p1000['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p1000['fractionofyear'], p1000['PanomalyMean'], color='b')

plt.figure(4, figsize=(20,10))
plt.scatter(p1500['fractionofyear'], p1500['NDVIanomaly'], marker='^', color='g')
plt.scatter(p1500['fractionofyear'],p1500['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p1500['fractionofyear'], p1500['PanomalyMean'], color='b')


#Just VPD and P
plt.figure(5, figsize=(20,10))
plt.scatter(p['fractionofyear'],p['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p['fractionofyear'], p['PanomalyMean'], color='b')

#############################################################
#############################################################
#JUNE
#####################
####################

id_pixel = pd.unique(df23June.index)
p = df23June[df23June.index==id_pixel[0]]
p500 = df23June[df23June.index==id_pixel[500]]
p1000 = df23June[df23June.index==id_pixel[1000]]
p1500 = df23June[df23June.index==id_pixel[1500]]

plt.figure(1, figsize=(20,10))
#plt.scatter(p['fractionofyear'], p['VPD'])
plt.scatter(p['fractionofyear'], p['NDVIanomaly'], marker='^', color='g')
plt.scatter(p['fractionofyear'],p['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p['fractionofyear'], p['PanomalyMean'], color='b')
#plt.plot(p['fractionofyear'], p['NDVImean'])
#plt.plot(p['fractionofyear'], p['NDVIstd'])

plt.figure(2, figsize=(20,10))
plt.scatter(p500['fractionofyear'], p500['NDVIanomaly'],  marker='^', color='g' )
plt.scatter(p500['fractionofyear'],p500['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p500['fractionofyear'], p500['PanomalyMean'], color='b')


plt.figure(3, figsize=(20,10))
plt.scatter(p1000['fractionofyear'], p1000['NDVIanomaly'], marker='^', color='g')
plt.scatter(p1000['fractionofyear'],p1000['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p1000['fractionofyear'], p1000['PanomalyMean'], color='b')

plt.figure(4, figsize=(20,10))
plt.scatter(p1500['fractionofyear'], p1500['NDVIanomaly'], marker='^', color='g')
plt.scatter(p1500['fractionofyear'],p1500['VPDanomalyMean']*-1, marker='x', color='r')
plt.scatter(p1500['fractionofyear'], p1500['PanomalyMean'], color='b')
"""
