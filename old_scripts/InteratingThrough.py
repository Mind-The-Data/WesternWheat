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

for pixel in range(id_pixel.size):
    model = sm.ols(formula='zNDVI ~ zVPD + zP', data = df23June[df23June.index==id_pixel[pixel]])
    results = model.fit()
    modeled_pixel_count += 1
    R_squared = np.append([results.rsquared],[R_squared])
    #results.rsquared.append(Rsquared)
    #print results.params
    #print results.summary()
    print modeled_pixel_count
    print results.rsquared


plt.scatter(id_pixel, R_squared, s=.01)