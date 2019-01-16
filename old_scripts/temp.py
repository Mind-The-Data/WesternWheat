# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

CDL23 = pd.read_csv("~/Documents/Research/Data/AgMet/CDL23.csv")

CDL23['.geo'] = CDL23['.geo'].map(lambda x: str(x)[47:])    #deleting strings in column .geo
CDL23['.geo'] = CDL23['.geo'].map(lambda x: str(x)[:-2])


#CDL23['.geo'] = pd.DataFrame(CDL23['.geo'].str.split(",", expand=True))
CDL23['.geo'] = pd.DataFrame(CDL23['.geo'].str.split().values.tolist())
CDL23['.geo'] = CDL23['.geo'].apply(lambda x: pd.Series(x.split(',')))


print CDL23.head()
print CDL23.info()
