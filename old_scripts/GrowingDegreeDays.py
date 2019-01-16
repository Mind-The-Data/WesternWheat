#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:12:56 2018

@author: brian
"""
import pandas as pd

dailydf = dfMasterMet


def Degree_Days(df):
    GDD = df.loc[(df.tmmx>5.5)].groupby(['system:indexviejuno','year'])['avgtemp'].sum()
    