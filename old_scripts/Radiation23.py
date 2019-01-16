#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:30:01 2018

@author: Brian
"""
import pandas as pd
import numpy as np
import matplotlib as mpl  #F9 loads; use console


Radiation23_2013_2017 = pd.read_csv("../Data/AgMet/monthlyLSclass23_2013_2017.csv")
Radiation23_2008 = pd.read_csv("../Data/AgMet/monthlyLSclass232008.csv")
Radiation23_2009 = pd.read_csv("../Data/AgMet/monthlyLSclass232009.csv")
Radiation23_2010 =  pd.read_csv("../Data/AgMet/monthlyLSclass232010.csv")
Radiation23_2011 = pd.read_csv("../Data/AgMet/monthlyLSclass232011.csv") 
Radiation23_2012 = pd.read_csv("../Data/AgMet/monthlyLSclass232012.csv")

Radiation23_frames = [Radiation23_2008,Radiation23_2009,Radiation23_2010,Radiation23_2011,Radiation23_2012,Radiation23_2013_2017]
Radiation23 = pd.concat(Radiation23_frames)
