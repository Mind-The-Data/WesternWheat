#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:58:41 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import scipy
import pymc3 as pm


datapath = '../Data/Processed/'
df23June = pd.read_csv(datapath + 'df23June.csv')
df23June.index=df23June['system:indexviejuno']
