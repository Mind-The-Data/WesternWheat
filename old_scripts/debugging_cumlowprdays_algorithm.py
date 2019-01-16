#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:12:31 2018

@author: brian
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#######
#Function does not know there is a 'whole' in the daily met data
#from sept to april, i could fix this. or wait until we get continuous data...
# Not worth the time at 7-30-18 to fix, come back later to fix. 
#######


#rename previously loaded dfMasterMet (daily) from preprocessing.py
df = dfMasterMet[['pr','cum_daysinarow_lowpr','system:indexviejuno']]
#loading previuosly loaded df_labeled_Master from InitialSom.py (April 24)
df = dfMasterMet
df = dfMaster
df = dfMasterMetMonthly
df.index = df.date
df['month'] = df.index.month


df.info()
df.describe()
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 150)
df[100600:100750]
df.describe()
df.cum_daysinarow_lowpr.plot(kind='hist', bins=200)


df20 = df.loc[(df.cum_daysinarow_lowpr>20)]
df20.cum_daysinarow_lowpr.plot(kind='hist')

df40 = df.loc[(df.cum_daysinarow_lowpr>40)]
df40.cum_daysinarow_lowpr.plot(kind='hist')

df75 = df.loc[(df.cum_daysinarow_lowpr>75)]
df75.cum_daysinarow_lowpr.plot(kind='hist', bins=200)

df100 = df.loc[(df.cum_daysinarow_lowpr>100)]
df100.cum_daysinarow_lowpr.plot(kind='hist')

df75.info()



######################
#Now check Monthly df

crop_id = 24
crop_id = str(crop_id)
data_path = '../rawData/AgMet/'
met = pd.read_csv(data_path + 'MeteorMonthly_experiment_' + crop_id + ".csv", index_col='system:indexviejuno')
ndvi = pd.read_csv(data_path + 'VegInd3_' + crop_id + ".csv", index_col='system:indexviejuno')
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
df = ndvi.merge(met)









df.describe()
df.index = df.date

df75 = df.loc[(df.cum_daysinarow_lowpr>75)]
df75.cum_daysinarow_lowpr.plot(kind='hist')

#Only spring months
df75spring = df75.loc[(df75.month == 3) | (df75.month == 4)]
df75spring.cum_daysinarow_lowpr.plot(kind='hist')

dfapril = df75spring.loc[(df75spring.month == 4)]
dfapril.cum_daysinarow_lowpr.plot(kind='hist')

dfmarch = df75spring.loc[(df75spring.month == 3)]
dfmarch.cum_daysinarow_lowpr.plot(kind='hist')

sns.relplot(x='cum_daysinarow_lowpr', y='pr', hue='month', s=9,palette="Paired", data=df)
sns.relplot(x='cum_daysinarow_lowpr', y='pr', hue='month', s=9,palette="Paired", data=df75spring)
sns.relplot(x='cum_daysinarow_lowpr', y='pr', hue='year', s=9,palette="Paired", data=dfapril)
sns.relplot(x='vpd', y='pr', hue='year', s=20,palette="Paired", data=dfapril)
sns.relplot(x='tmmx', y='pr', hue='year', s=20,palette="Paired", data=dfapril)
dfapril.info()


sns.relplot(y='Latitude', x='Longitude', hue='year', s=20,palette="Paired", data=dfapril)

# correlation matrix for april with high cum days
dfapril = dfapril[['pr','vpd','cum_daysinarow_lowpr','zNDVI','tmmx','daysabove30','eto','etr']]
corr = dfapril.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0, cmap='seismic')




#DAILY MET
daily = pd.read_csv(data_path + "Meteordaily_" + crop_id + ".csv")


daily75 = daily.loc[(daily.cum_daysinarow_lowpr>75)]

daily75.info()

#index to date time function
def index_to_datetime(arg):
    date = pd.to_datetime(arg.split('_')[0])
    return date
# adding month
daily.index = daily['system:indexviejuno']
print 'applying index_to_datetime function...'
daily['date'] = daily['system:index'].apply(index_to_datetime)
daily.index = daily['date']
daily['month'] = pd.DatetimeIndex(daily['date']).month
daily['year'] = pd.DatetimeIndex(daily['date']).year

daily['.geo'] = daily['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
daily['.geo'] = daily['.geo'].map(lambda x: str(x)[:-2])
daily['Longitude'], daily['Latitude']= daily['.geo'].str.split(',', 1).str




sns.relplot(y='pr', x='cum_daysinarow_lowpr', hue='year', s=20,palette="Paired", data=daily75)

daily75.year.value_counts()

daily100 =daily75.loc[(daily75.cum_daysinarow_lowpr>100)] 
daily100.info()
daily100.year.value_counts()

#daily did not have .geo split so no lat long
###### GEO split
daily100['.geo'] = daily100['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
daily100['.geo'] = daily100['.geo'].map(lambda x: str(x)[:-2])
daily100['Longitude'], daily100['Latitude']= daily100['.geo'].str.split(',', 1).str

sns.relplot(y='pr', x='cum_daysinarow_lowpr', hue='year', s=20,palette="Paired", data=daily100)
sns.relplot(y='Latitude', x='Longitude', hue='year', s=20,palette="Paired", data=daily100)

daily100.info()
daily100['system:indexviejuno']



###########selecting indivudal pixels

sixtyfour = daily100.loc[(daily100['system:indexviejuno']==64)]
sns.relplot(y='pr', x='cum_daysinarow_lowpr', hue='year', s=20,palette="Paired", data=sixtyfour)
sns.relplot(y='Latitude', x='Longitude', hue='year', s=20,palette="Paired", data=sixtyfour)

# not restricted to cum days above 100....
pixel64 = daily.loc[(daily['system:indexviejuno']==64)]
sns.relplot(y='pr', x='cum_daysinarow_lowpr', hue='year', s=10,palette="Paired", data=pixel64)

pixel64dash2015 = pixel64.loc[(pixel64.year==2015)]
sns.relplot(y='pr', x='cum_daysinarow_lowpr', hue='month', s=10,palette="Paired", data=pixel64dash2015)

daily100.month.value_counts()



########## 
d = daily.groupby(daily['system:indexviejuno']).loc[(daily.cum_daysinarow_lowpr>75)]