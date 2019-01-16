#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:23:10 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns


datapath = '../Data/Processed/'
df23June = pd.read_csv(datapath + 'df23June.csv')
df23June.index=df23June['system:indexviejuno']




id_pixel = pd.unique(df23June.index)
#p = df23June[df23June.index==id_pixel[0]]
#p500 = df23June[df23June.index==id_pixel[500]]
#p1000 = df23June[df23June.index==id_pixel[1000]]
#p1500 = df23June[df23June.index==id_pixel[1500]]

modeled_pixel_count = 0 
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
    

# dir(results)
print dir(results)

plt.figure(dpi=300)
sns.kdeplot(Rsquared_adj)
plt.title("Frequency Plot of adjusted $R^2$")
plt.xlabel("$R^2$")

plt.figure(dpi=300)
sns.kdeplot(R_squared)
plt.title("Frequency Plot of $R^2$")
plt.xlabel("$R^2$")


plt.figure(dpi=300)
plt.scatter(id_pixel, R_squared, s=.1)
plt.xlabel('pixel id')
plt.ylabel('$R^2$')
plt.title('Variance Explained with zNDVI ~ zP + zVPD')
plt.savefig('../Images/JuneOnly23/RsquaredVSpixelsJune.png')

plt.figure(dpi=300)
plt.scatter(id_pixel, Rsquared_adj, s=.1)
plt.xlabel('pixel id')
plt.ylabel('$R^2 adjusted$')
plt.title('Variance Explained with $zNDVI$ ~ $zP$ + $zVPD$ + $E$')
plt.savefig('../Images/JuneOnly23/Rsquared_adjusted_VSpixelsJune.png')