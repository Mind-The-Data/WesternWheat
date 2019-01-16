
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import statsmodels.formula.api as sm

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






#renaming columns
df23 = df23.rename(columns={'vpd':'VPD'})
df23 = df23.rename(columns={'pr':'P'})

#adding a new column: 'month'
df23['month'] = ((df23['fractionofyear'] - df23['fractionofyear'].astype(int))*12).round().astype(int)

#Calculating arrays? of means and stds across all years, grouped by month
meanVPD = df23.groupby(['system:indexviejuno', 'month'])['VPD'].mean()
stdVPD = df23.groupby(['system:indexviejuno', 'month'])['VPD'].std()
meanP = df23.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df23.groupby(['system:indexviejuno', 'month'])['P'].std() 


"""
#Vapor Pressure Deficit to dataframe
df23 = df23.join(meanVPD, on=['system:indexviejuno', 'month'], rsuffix='mean')
df23 = df23.join(stdVPD, on=['system:indexviejuno', 'month'], rsuffix='std')
df23['VPDanomaly'] = (df23['VPD'] - df23['VPDmean'])/df23['VPDstd'] 
meananomalyVPD = df23.groupby(['system:indexviejuno', 'fractionofyear'])['VPDanomaly'].mean()
df23 = df23.join(meananomalyVPD, on=['system:indexviejuno', 'fractionofyear'], rsuffix='Mean')
##calculating anomaly with vpd instaneous - monthlymean is not quite what I want.What do i want? vpd monthly average- vpd yearly average bined by month
"""
#VPD: a more straightforward way to calculate anomalies
monthlymeanVPD = df23.groupby(['system:indexviejuno', 'fractionofyear'])['VPD'].mean()
df23 = df23.join(meanVPD, on=['system:indexviejuno', 'month'], rsuffix='mean')
df23 = df23.join(stdVPD, on=['system:indexviejuno', 'month'], rsuffix='std')
df23 = df23.join(monthlymeanVPD, on=['system:indexviejuno', 'month'], rsuffix='monthlymean')
df23['VPDanomaly'] = (df23['monthlymeanVPD'] - df23['meanVPD'])/df23['stdVPD']




#Precipitation
meanP = df23.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df23.groupby(['system:indexviejuno', 'month'])['P'].std() 
########Previous method
df23 = df23.join(meanP, on=['system:indexviejuno', 'month'], rsuffix='mean')
df23 = df23.join(stdP, on=['system:indexviejuno', 'month'], rsuffix='std')
df23['Panomaly'] = (df23['P'] - df23['Pmean'])/df23['Pstd']
#average of the daily anomalies (take the (VPD-yearlymean_binned_month/std) and average over all VPD's in given fractionofyear
meananomalyP = df23.groupby(['system:indexviejuno', 'fractionofyear'])['Panomaly'].mean()
df23 = df23.join(meananomalyP, on=['system:indexviejuno', 'fractionofyear'], rsuffix='Mean')
########## better way to do it######################################
meanP = df23.groupby(['system:indexviejuno', 'month'])['P'].mean()
stdP = df23.groupby(['system:indexviejuno', 'month'])['P'].std() 
df23 = df23.join(meanP, on=['system:indexviejuno', 'month'], rsuffix='yearlymean')
df23 = df23.join(stdP, on=['system:indexviejuno', 'month'], rsuffix='yearlystd')
fractionofyearmeanP = df23.groupby(['system:indexviejuno', 'fractionofyear'])['P'].mean()
df23 = df23.join(fractionofyearmeanP, on=['system:indexviejuno', 'month'], rsuffix='fractionofyearmean')
df23['Panomaly']=(df23['Pfractionofyearmean']-df23['Pyearlymean'])/df23['Pyearlystd']




#NDVI
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
#edf23May = df23[df23['month'] == 5]

#summerframes would have to have renamed columns
#summerframes = [df23July, df23June, df23May]
#df23Summer = pd.concat(summerframes)

#TRYING TO ADD MAYPRECIP TO JUNE DF SO I CAN DO AN OLS WITH JUNE NDVI, VPD AND MAYP
## having trouble concatenating, but this worked.
#df23May = df23May.rename(columns={'PanomalyMean':'PanomalyMeanMay'})
#df23MayJustPanomalyMean = df23May['PanomalyMeanMay']
#df23JunewithMayP= df23June.join(df23MayJustPanomalyMean)

#print df23JunewithMayP.info()

#######################
######################
#Statistics
####################
###################

#### correlation
#df23June['negVPDanomaly'].corr(df23June['Panomaly'])
#df23June['negVPDanomalyMean'].corr(df23June['PanomalyMean'])

## Ordinary Least Squares
#model = sm.ols(formula="NDVIanomaly ~ PanomalyMeanMay + VPDanomalyMean", data = df23JunewithMayP).fit()
model = sm.ols(formula='NDVIanomaly ~ PanomalyMean + VPDanomalyMean', data = df23June).fit()
print model.params
print model.summary()

"""


###########################
## PLOTTING
###########################

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
