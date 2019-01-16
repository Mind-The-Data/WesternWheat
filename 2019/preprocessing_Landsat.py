#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:55:17 2018

@author: brian
"""

##########################################
# Landsat spectral data transformations
##########################################



import numpy as np
import pandas as pd
import glob
from functions import statistics, rolling_sum, interpolate




crop_id = 24
crop_id = str(crop_id)



dfMaster = None
dfMaster = pd.DataFrame()
data_path = '../../Data/AgMet/Monthly/'
print "Loading EM spectral dataframes..."
for f in glob.glob(data_path + 'monthlyLSclass' + crop_id + '*'):
    dfMaster = pd.concat((dfMaster,  pd.read_csv(f)))

dfMaster.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)


print 'Performing date magic....'
# for 23 I had to add one to month as fractionofyear started at 2008.00 and ended at 2008.916 ... 
# Then make sure date calculation is correct. 
dfMaster['month'] = (((dfMaster['fractionofyear'] -\
        dfMaster['fractionofyear'].astype(int))*12).round().astype(int)+1)
dfMaster['year'] = (np.modf(dfMaster['fractionofyear'])[1]).astype(int)
dfMaster['date'] = pd.to_datetime(dfMaster.year * 100 + (dfMaster.month), format='%Y%m')
dfMaster.index = dfMaster['date']
dfMaster = dfMaster.to_period('M')






######################################################
### VEGETATION INDICES
#####################################################

# Note on efficiency: CDL could mask first to speed up by ~50%, however, 
# makes the interpolation step more involved since you'd be interpolating from one
# CDL !=24 year to one CDL==24 year, you'd need a for loop or some other function
# that would likely slow down operation. 
# No need to speed up yet as this df is relatively easy to handle on the server 
#(300rows X 10 columns) ,  35 MB. 

dfMaster.date = dfMaster.index
dfMaster.drop_duplicates(subset=['pixel','date'], keep = 'first', inplace=True)


print "Calculating Vegetation Indices..."
dfMaster.index = dfMaster['pixel']
#dfMaster.dropna(inplace=True)
dfMaster['NDVI'] = (dfMaster['B4'] - dfMaster['B3'])/(dfMaster['B4'] + dfMaster['B3'])
dfMaster['EVI'] = 2.5*(dfMaster['B4'] - dfMaster['B3'])/(dfMaster['B4'] + 6*dfMaster['B3']\
        -7.5*dfMaster['B1']+1)
dfMaster['NDWI1'] = (dfMaster['B4'] - dfMaster['B5'])/(dfMaster['B4'] + dfMaster['B5'])
dfMaster['NDWI2'] = (dfMaster['B4'] - dfMaster['B7'])/(dfMaster['B4'] + dfMaster['B7'])

#tell me how many points we throw out
# initially I did below -1 and 1, however I had one point between 0 and -1 NDVI I want to throw out. 
NDVI_datapointsthrownout = dfMaster.loc[(dfMaster.NDVI<-1) | (dfMaster.NDVI>1)].count()
EVI_datapointsthrownout = dfMaster.loc[(dfMaster.EVI<-2) | (dfMaster.EVI>2)].count()
print "NDVI data points that I will throw out: " +str(NDVI_datapointsthrownout.NDVI)
print "anomalous EVI points not thrown out yet: " +str(EVI_datapointsthrownout.EVI)
#Investigating
#df = dfMaster.loc[(dfMaster.NDVI<-1) | (dfMaster.NDVI>1)]
pd.set_option('display.max_columns', 25)

#Throwing out points...
dfMaster=dfMaster.loc[(dfMaster.NDVI > -1.) & (dfMaster.NDVI <1.)]
#dfMaster=dfMaster.loc[(dfMaster.EVI > -2.) & (dfMaster.EVI <2.)]

### Dropping unnecessary columns
dfMaster.drop(axis = 1, columns = ['system:index', u'B1', u'B2', u'B3',\
                                   u'B4', u'B5', u'B7'], inplace = True)
####################
###### INTERPOLATION
######################
# fill missing rows with Nan Values
# to be interpolated over later
dfMaster.drop_duplicates(subset=['pixel','date'], keep = 'first', inplace=True)

time_range = pd.date_range(dfMaster.date.min().to_timestamp(),\
                           dfMaster.date.max().to_timestamp(), freq='M')\
                           .to_period('M')
dfMaster.index = dfMaster.date
                         
dfMaster = dfMaster.groupby('pixel').apply(lambda x: x.reindex(time_range))

dfMaster = interpolate(df = dfMaster, columns = ['NDVI','EVI', 'NDWI1', 'NDWI2'])

# fill other nan columns with appropriate stuff







#####################################
###################################
####### MORE DATA #######
###################################
#######################################



dfMaster2 = None
dfMaster2 = pd.DataFrame()
data_path = '../../Data/AgMet2/Monthly/'
print "Loading EM spectral dataframes..."
for f in glob.glob(data_path + 'monthlyLSclass' + crop_id + '*'):
    dfMaster2 = pd.concat((dfMaster2,  pd.read_csv(f)))

dfMaster2.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)


print 'Performing date magic.... adding one to month...!!!!'
# for 23 I had to add one to month as fractionofyear started at 2008.00 and ended at 2008.916 ... 
# make sure to check 24!!!!!!!!!!
dfMaster2['month'] = (((dfMaster2['fractionofyear'] - \
         dfMaster2['fractionofyear'].astype(int))*12).round().astype(int)+1)
dfMaster2['year'] = (np.modf(dfMaster2['fractionofyear'])[1]).astype(int)
dfMaster2['date'] = pd.to_datetime(dfMaster2.year * 100 + (dfMaster2.month), format='%Y%m')
dfMaster2.index = dfMaster2['date']
dfMaster2 = dfMaster2.to_period('M')


dfMaster2.date = dfMaster2.index
dfMaster2.drop_duplicates(subset=['pixel','date'], keep = 'first', inplace=True)

print "Calculating Vegetation Indices on new data..."
dfMaster2.index = dfMaster2['pixel']
#dfMaster2.dropna(inplace=True)
dfMaster2['NDVI'] = (dfMaster2['B4'] - dfMaster2['B3'])/(dfMaster2['B4'] + dfMaster2['B3'])
dfMaster2['EVI'] = 2.5*(dfMaster2['B4'] - dfMaster2['B3'])/(dfMaster2['B4'] + 6*dfMaster2['B3']\
        -7.5*dfMaster2['B1']+1)
dfMaster2['NDWI1'] = (dfMaster2['B4'] - dfMaster2['B5'])/(dfMaster2['B4'] + dfMaster2['B5'])
dfMaster2['NDWI2'] = (dfMaster2['B4'] - dfMaster2['B7'])/(dfMaster2['B4'] + dfMaster2['B7'])

#tell me how many points we throw out
# initially I did below -1 and 1, however I had one point between 0 and -1 NDVI I want to throw out. 
NDVI_datapointsthrownout = dfMaster2.loc[(dfMaster2.NDVI<-1) | (dfMaster2.NDVI>1)].count()
EVI_datapointsthrownout = dfMaster2.loc[(dfMaster2.EVI<-2) | (dfMaster2.EVI>2)].count()
print "NDVI data points that I will throw out: " +str(NDVI_datapointsthrownout.NDVI)
print "anomalous EVI points not thrown out yet: " +str(EVI_datapointsthrownout.EVI)
#Investigating
#df = dfMaster2.loc[(dfMaster2.NDVI<-1) | (dfMaster2.NDVI>1)]
pd.set_option('display.max_columns', 25)

#Throwing out points...
dfMaster2=dfMaster2.loc[(dfMaster2.NDVI > -1.) & (dfMaster2.NDVI <1.)]
#dfMaster2=dfMaster2.loc[(dfMaster2.EVI > -2.) & (dfMaster2.EVI <2.)]

### Dropping unnecessary columns
dfMaster2.drop(axis = 1, columns = ['system:index', u'B1', u'B2', u'B3',\
                                   u'B4', u'B5', u'B7'], inplace = True)

####################################
#### INTERPOLATION STEP
###################################
   

dfMaster2.drop_duplicates(subset=['pixel','date'], keep = 'first', inplace=True)

time_range = pd.date_range(dfMaster2.date.min().to_timestamp(),\
                           dfMaster2.date.max().to_timestamp(), freq='M')\
                           .to_period('M')
                           
dfMaster2.index = dfMaster2.date # convert from pixel to date index for next step
                         
dfMaster2 = dfMaster2.groupby('pixel').apply(lambda x: x.reindex(time_range))
# fill missing rows with Nan Values
# to be interpolated over next

dfMaster2 = interpolate(df = dfMaster2, columns = ['NDVI','EVI', 'NDWI1', 'NDWI2'])



'''make function...'''





###########################
###########################
# CDL
# filtering
##########################
##########################


print "Reading CDL dataframe from harddrive..."
data_path = '../../Data/AgMet/'
dfCDL = pd.read_csv(data_path + "CDL" + crop_id + ".csv", 
            usecols=['system:index','countyears','cropland_mode','first','system:indexviejuno'])
dfCDL['year'] = dfCDL['system:index'].apply(lambda x: x.split("_")[0]).astype(int)
dfCDL.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)


lkupTable = dfCDL.pivot('year', 'pixel', 'first')
dfMaster['CDL'] = lkupTable.lookup(dfMaster['year'], dfMaster['pixel'])
print "Crop/fallow pixel filtering..."
dfMaster = dfMaster.loc[dfMaster['CDL'] == float(crop_id)]

##############
### NEW DATA
# same algorithm
#################

print "Reading new CDL dataframe from harddrive..."
data_path = '../../Data/AgMet2/'
dfCDL2 = pd.read_csv(data_path + "CDL" + crop_id + "morepoints.csv", 
            usecols=['system:index','countyears','cropland_mode','first','system:indexviejuno'])
dfCDL2['year'] = dfCDL2['system:index'].apply(lambda x: x.split("_")[0]).astype(int)

dfCDL2.rename(index=str, columns={'system:indexviejuno':'pixel'},inplace=True)

lkupTable2 = dfCDL2.pivot('year', 'pixel', 'first')
dfMaster2 = dfMaster2[dfMaster2.year != 2007] # no CDL 2007 data
dfMaster2['CDL'] = lkupTable2.lookup(dfMaster2['year'], dfMaster2['pixel']) # row labels, col labels
print "Crop/fallow pixel filtering..."
dfMaster2 = dfMaster2.loc[dfMaster2['CDL'] == int(crop_id)]


##########################################
##########################
##################### COMBINE
##########################
#########################################

# way too slow of way to test if there are copies, I have gotten atleast 2 copies
# but rather impatient waiting for this thing. 
# odd that I have some repeat values.... must be how Google Earth Engine
# psuedo-randomly samples...
''''
for i in range(2000):
    for j in range(2000):
        if pd.unique(dfMaster.pixel)[i] == pd.unique(dfMaster2.pixel)[j]:
            print 'copy'
'''

print 'concatenating...\n' 
dfMaster = pd.concat([dfMaster, dfMaster2], axis=0, join='outer')





#########################
#### Rolling SUM
#########################


    
    
#messing with date and index
dfMaster.index = dfMaster.date
dfMaster = dfMaster.to_period('M')
dfMaster.date = dfMaster.index 
# Important!!! turns back into object date at M freq





d = dfMaster[dfMaster.pixel < 19260]


columns = ['NDVI']#, 'EVI']
w=3
for column in columns:
    print column
    d.index = d.date
    series = d.groupby(['pixel','year']).rolling(w)[column].sum() # no year. so rolls over dec-jan
    # need year for landsat data, already filtered by cdl... discontinuous
    series = series.reset_index(level=[0,1,2])
    series.rename(index=str, columns={column: column + 'sum' + str(w)}, inplace=True)
    d = d.merge(series, on=['pixel','date','year'], how='inner') 
    print 'merged'


#################
### Rolling Sums
################
from functions import statistics, rolling_sum

columns = ['NDVI','EVI','NDWI1']#,'NDWI2']
d = rolling_sum(d, columns, w=3)

for w in range(2,7):
    dfMaster = rolling_sum(dfMaster, columns, w=w)


column = 'NDVI'
w=3

dfMaster.index = dfMaster.date
series = dfMaster.groupby(['pixel']).rolling(w)[column].sum() # no year. so rolls over dec-jan
series = series.reset_index(level=[0,1])
series.rename(index=str, columns={column: column + 'sum' + str(w)}, inplace=True)
dfMaster = dfMaster.merge(series, on=['pixel','date']) 

###############
## statistics
##############

columns = ['NDVI']#,'EVI','NDWI1','NDWI2']
df = statistics(dfMaster, columns)


###### GEO split
dfMaster['.geo'] = dfMaster['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
dfMaster['.geo'] = dfMaster['.geo'].map(lambda x: str(x)[:-2])
dfMaster['Longitude'], dfMaster['Latitude']= dfMaster['.geo'].str.split(',', 1).str
dfMaster.drop(columns = ['.geo'], inplace = True)

#dfMaster.drop(columns=['','','','','','','','','',''])
data_path = '../../Processed'
print "Saving Vegetation Indices (" + crop_id + ") dataframe to hard drive..."
dfMaster[[u'fractionofyear',
       u'pixel', u'.geo', u'Latitude', u'Longitude', u'month', u'year', u'date',
       u'CDL',u'countyears',
       u'NDVI', u'EVI', u'NDWI1', u'NDWI2', u'NDVImean', u'NDVIstd', u'EVImean', u'EVIstd',
       u'NDWI1mean',u'NDWI2mean', u'zNDVI',  # u'NDWI1std', u'NDWI2std
       u'zEVI', u'zNDWI1', u'zNDWI2','NDVIsum3',
       u'NDVIsum3mean', u'NDVIsum3std', u'zNDVIsum3',
       u'NDVIsum2', u'NDVIsum2mean', u'NDVIsum2std', u'zNDVIsum2',
       u'NDVIsum4',u'NDVIsum4mean', u'NDVIsum4std', u'zNDVIsum4'
       ]].to_csv(data_path + 'VegInd5_' + crop_id + '.csv')
#free memory
dfMaster = None
















'''
meanNDVI = dfMaster.groupby(['pixel', 'month'])['NDVI'].mean()
stdNDVI = dfMaster.groupby(['pixel', 'month'])['NDVI'].std()
#maxNDVI = dfMaster.groupby(['pixel', 'month', 'year'])['NDVI'].max()
#NDVI = dfMaster.groupby(['pixel', 'month' , 'year'])['NDVI'].mean()

meanEVI = dfMaster.groupby(['pixel', 'month'])['EVI'].mean()
stdEVI = dfMaster.groupby(['pixel', 'month'])['EVI'].std()
#maxEVI = dfMaster.groupby(['pixel', 'month', 'year'])['EVI'].max()
#EVI = dfMaster.groupby(['pixel', 'month','year'])['EVI'].mean()

meanNDWI1 = dfMaster.groupby(['pixel', 'month'])['NDWI1'].mean()
stdNDWI1 = dfMaster.groupby(['pixel', 'month'])['NDWI1'].std()
#NDWI1 = dfMaster.groupby(['pixel', 'month' ,'year'])['NDWI1'].mean()

meanNDWI2 = dfMaster.groupby(['pixel', 'month'])['NDWI2'].mean()
stdNDWI2 = dfMaster.groupby(['pixel', 'month'])['NDWI2'].std()
#NDWI2 = dfMaster.groupby(['pixel', 'month' ,'year'])['NDWI2'].mean()

print "Joining vegetation indices arrays into dfMaster..."
dfMaster = dfMaster.join(meanNDVI, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVI, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(maxNDVI, on=['pixel', 'month' ,'year'], rsuffix='max')
#dfMaster = dfMaster.join(NDVI, on=['pixel', 'month', 'year'], rsuffix='monthly')

dfMaster = dfMaster.join(meanEVI, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdEVI, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(EVI, on=['pixel', 'month', 'year'], rsuffix='monthly')
#dfMaster = dfMaster.join(maxEVI, on=['pixel', 'month' ,'year'], rsuffix='max')

dfMaster = dfMaster.join(meanNDWI1, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDWI1, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(NDWI1, on=['pixel', 'month', 'year'], rsuffix='monthly')

dfMaster = dfMaster.join(meanNDWI2, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDWI2, on=['pixel', 'month'], rsuffix='std')
#dfMaster = dfMaster.join(NDWI2, on=['pixel', 'month', 'year'], rsuffix='monthly')

print 'Calculating Anomalies...'
dfMaster['zNDVI'] = (dfMaster['NDVI'] - dfMaster['NDVImean']) / dfMaster['NDVIstd']
dfMaster['zEVI'] = (dfMaster['EVI'] - dfMaster['EVImean']) / dfMaster['EVIstd']
dfMaster['zNDWI1'] = (dfMaster['NDWI1'] - dfMaster['NDWI1mean']) / dfMaster['NDWI1std']
dfMaster['zNDWI2'] = (dfMaster['NDWI2'] - dfMaster['NDWI2mean']) / dfMaster['NDWI2std']



#0.707107 comes up A LOT
#Mostly in winter months... Explaination:
# In months where there are very few data values (one or two), 
#during the zscore operation, one value- one value equals some machine precision
# value, and it so happens these values produce a repeating .707107
### SOLVED!!!!!!!!!!
dfMaster.loc[np.isclose(dfMaster['zNDVI'].abs(), 0.707107), 'zNDVI'] = np.nan

## Investigating anomalies
### Lots of values near that abs 0.707107 value... can't just delete... right?
'''
'''
dfMaster.zNDVI.sort_values().head()
dfMaster.zNDVI.sort_values().tail()
dfMaster.zNDVI.describe()
dfMaster.zNDVI.value_counts().sort_values()
dfMaster.zEVI.value_counts().sort_values()
dfMaster.zNDWI1.value_counts().sort_values()
dfMaster.zNDWI2.value_counts().sort_values()

df = dfMaster.loc[np.isclose(dfMaster['zNDVI'].abs(), .7071)]
df['pixel'].value_counts()
df.B5.value_counts()
df.NDVI.value_counts()
df.NDVI.value_counts().sort_values()
df.columns
df.NDVImean.value_counts()
df.NDVIstd.value_counts()
df.zNDVI.value_counts()
df.NDVI
df.Latitude.value_counts()
df.Longitude.value_counts()
df.month.value_counts()
#compared to dfmaster... how likely is this?
dfMaster.Longitude.value_counts()
dfMaster.Latitude.value_counts()
dfMaster.NDVI.value_counts()
dfMaster.NDVImean.value_counts()
dfMaster.NDVIstd.value_counts()

dfSpring = dfMaster.loc[(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfSpring.Longitude.value_counts()
dfSpring.Latitude.value_counts()
dfSpring.NDVI.value_counts()
dfSpring.NDVImean.value_counts()
dfSpring.NDVIstd.value_counts()
dfSpring.zNDVI.value_counts()

dfspring = dfMaster.loc[np.isclose(dfMaster['zNDVI'].abs(), .7071) & \
(dfMaster['month'] > 1) & (dfMaster['month']< 5)]
dfspring.Longitude.value_counts()
dfspring.Latitude.value_counts()
dfspring.NDVI.value_counts()
dfspring.NDVImean.value_counts()
dfspring.NDVIstd.value_counts()
dfspring.zNDVI.value_counts()
'''

##################
#Investigating .geo split
##################
'''
lat = dfMaster['Latitude'].sort_values()
L = dfMaster['Longitude'].sort_values()
L.value_counts().sort_values()
l = dfMaster['longitude'].sort_values()
l.value_counts().sort_values()
plt.scatter( L, l)
l.count()
L.count()
'''
######################
######################
######################
# RollingWindowNDVI
######################
######################
######################
'''
dfMaster.index = dfMaster.date 
# has to be a date index so when recombined it has date and pixel which makes unique

NDVIsum3=None
NDVIsum3 = dfMaster.groupby(['pixel','year']).rolling(3).NDVI.sum()
NDVIsum3 = NDVIsum3.reset_index(level=[0,1]) # from multi index to one index
NDVIsum3= NDVIsum3.reset_index(level=0) 
# change index to default range/period so date can be used by merge as a column
NDVIsum3.rename(index=str, columns={'NDVI':'NDVIsum3'}, inplace=True)
dfMaster = dfMaster.merge(NDVIsum3, on=['pixel', 'date'])
dfMaster.rename(index=str, columns={'year_x':'year'}, inplace=True)
dfMaster.drop(columns='year_y',inplace=True)

meanNDVIsum3 = dfMaster.groupby(['pixel', 'month'])['NDVIsum3'].mean()
stdNDVIsum3 = dfMaster.groupby(['pixel', 'month'])['NDVIsum3'].std()
dfMaster = dfMaster.join(meanNDVIsum3, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVIsum3, on=['pixel', 'month'], rsuffix='std')
dfMaster['zNDVIsum3'] = (dfMaster['NDVIsum3'] - dfMaster['NDVIsum3mean']) / dfMaster['NDVIsum3std']

dfMaster.loc[np.isclose(dfMaster['zNDVIsum3'].abs(), 0.707107), 'zNDVIsum3'] = np.nan 

###############
#### 2 month 
##############
dfMaster.index = dfMaster.date 
NDVIsum2=None
NDVIsum2 = dfMaster.groupby(['pixel']).rolling(2).NDVI.sum()
NDVIsum2 = NDVIsum2.reset_index(level=[0,1]) # from multi index to one index
NDVIsum2= NDVIsum2.reset_index(level=0) 
# change index to default range/period so date can be used by merge as a column
NDVIsum2.rename(index=str, columns={'NDVI':'NDVIsum2'}, inplace=True)
dfMaster = dfMaster.merge(NDVIsum2, on=['pixel', 'date'])
#dfMaster.rename(index=str, columns={'year_x':'year'}, inplace=True)
#dfMaster.drop(columns='year_y',inplace=True)

meanNDVIsum2 = dfMaster.groupby(['pixel', 'month'])['NDVIsum2'].mean()
stdNDVIsum2 = dfMaster.groupby(['pixel', 'month'])['NDVIsum2'].std()
dfMaster = dfMaster.join(meanNDVIsum2, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVIsum2, on=['pixel', 'month'], rsuffix='std')
dfMaster['zNDVIsum2'] = (dfMaster['NDVIsum2'] - dfMaster['NDVIsum2mean']) / dfMaster['NDVIsum2std']

dfMaster.loc[np.isclose(dfMaster['zNDVIsum2'].abs(), 0.707107), 'zNDVIsum2'] = np.nan

###############
#### 4 month 
##############
dfMaster.index = dfMaster.date 
NDVIsum4=None
NDVIsum4 = dfMaster.groupby(['pixel','year']).rolling(4).NDVI.sum()
NDVIsum4 = NDVIsum4.reset_index(level=[0,1]) # from multi index to one index
NDVIsum4= NDVIsum4.reset_index(level=0)
 # change index to default range/period so date can be used by merge as a column
NDVIsum4.rename(index=str, columns={'NDVI':'NDVIsum4'}, inplace=True)
dfMaster = dfMaster.merge(NDVIsum4, on=['pixel', 'date'])
dfMaster.rename(index=str, columns={'year_x':'year'}, inplace=True)
dfMaster.drop(columns='year_y',inplace=True)

meanNDVIsum4 = dfMaster.groupby(['pixel', 'month'])['NDVIsum4'].mean()
stdNDVIsum4 = dfMaster.groupby(['pixel', 'month'])['NDVIsum4'].std()
dfMaster = dfMaster.join(meanNDVIsum4, on=['pixel', 'month'], rsuffix='mean')
dfMaster = dfMaster.join(stdNDVIsum4, on=['pixel', 'month'], rsuffix='std')
dfMaster['zNDVIsum4'] = (dfMaster['NDVIsum4'] - dfMaster['NDVIsum4mean']) / dfMaster['NDVIsum4std']

dfMaster.loc[np.isclose(dfMaster['zNDVIsum4'].abs(), 0.707107), 'zNDVIsum4'] = np.nan


###############
#### 5 month 
##############






#dfMaster.index
#id_pixel = pd.unique(dfMaster.index)


#dfMaster.drop(columns=['','','','','','','','','',''])
data_path = '../../Processed'
print "Saving Vegetation Indices (" + crop_id + ") dataframe to hard drive..."
dfMaster[[u'fractionofyear',
       u'pixel', u'.geo', u'Latitude', u'Longitude', u'month', u'year', u'date',
       u'CDL',u'countyears',
       u'NDVI', u'EVI', u'NDWI1', u'NDWI2', u'NDVImean', u'NDVIstd', u'EVImean', u'EVIstd',
       u'NDWI1mean',u'NDWI2mean', u'zNDVI',  # u'NDWI1std', u'NDWI2std
       u'zEVI', u'zNDWI1', u'zNDWI2','NDVIsum3',
       u'NDVIsum3mean', u'NDVIsum3std', u'zNDVIsum3',
       u'NDVIsum2', u'NDVIsum2mean', u'NDVIsum2std', u'zNDVIsum2',
       u'NDVIsum4',u'NDVIsum4mean', u'NDVIsum4std', u'zNDVIsum4'
       ]].to_csv(data_path + 'VegInd5_' + crop_id + '.csv')
#free memory
dfMaster = None
'''