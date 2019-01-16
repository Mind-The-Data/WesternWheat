#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:56:13 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm



datapath = '../Data/Processed/'
df23 = pd.read_csv(datapath + 'df23.csv')
df23.index=df23['system:indexviejuno']

df23.drop('B2', 'B3', 'B4', 'B5', 'B6', 'B7', axis=1)


id_pixel = pd.unique(df23.index)
#p = df23[df23.index==id_pixel[0]]
#p500 = df23[df23.index==id_pixel[500]]
#p1000 = df23[df23.index==id_pixel[1000]]
#p1500 = df23[df23.index==id_pixel[1500]]

modeled_pixel_count = 0 
R_squared = []

for pixel in range(id_pixel.size):
    model = sm.ols(formula='zNDVI ~ zVPD + zP', data = df23[df23.index==id_pixel[pixel]])
    results = model.fit()
    modeled_pixel_count += 1
    R_squared = np.append([results.rsquared],[R_squared])
    #results.rsquared.append(Rsquared)
    #print results.params
    #print results.summary()
    print modeled_pixel_count
    print results.rsquared
