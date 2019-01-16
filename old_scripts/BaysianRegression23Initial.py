#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:44:52 2018

@author: brian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import scipy
import pymc3 as pm
import scipy.stats as stats

palette = 'muted'
sns.set_palette(palette); sns.set_color_codes(palette)
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)


datapath = '../Data/Processed/'
dfJune23 = pd.read_csv(datapath + 'df23June23.csv')
dfJune23.index=dfJune23['system:indexviejuno']


singlepixel = dfJune23[dfJune23.index ==41857.0]

x = singlepixel['zVPD']
y = singlepixel['zNDVI']

sns.kdeplot(y)
sns.kdeplot(x)

plt.scatter(x, y)

#GOAL: 
####### pixel indentification
####### iterate through pixels
####### for each pixel come up with a linear model
"""
id_pixel = pd.unique(df23.index)
p = df23[df23.index==id_pixel[0]]
p500 = df23[df23.index==id_pixel[500]]
p1000 = df23[df23.index==id_pixel[1000]]
p1500 = df23[df23.index==id_pixel[1500]]
"""

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=1)
    epsilon = pm.HalfCauchy('epsilon', 5)

    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
    
    rb = pm.Deterministic('rb', (beta * x.std() / y.std()) ** 2)

    y_mean = y.mean()
    ss_reg = pm.math.sum((mu - y_mean) ** 2)
    ss_tot = pm.math.sum((y - y_mean) ** 2)
    rss = pm.Deterministic('rss', ss_reg/ss_tot)
    
    start = pm.find_MAP() 
    step = pm.NUTS() 
    tracez = pm.sample(1000, step, start, chains=1)
    
pm.traceplot(tracez)

varnames = ['alpha', 'beta', 'epsilon']
pm.autocorrplot(tracez, varnames)

sns.kdeplot(tracez['alpha'], tracez['beta'])
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\beta$', fontsize=16, rotation=0)
plt.title('parameter correlation')

##############################################
#Posterior Predictive Checks
# Does the model replicate observed data?
# useful to see if model is in the ball park of data
# does not validate model! need external data to do that of course. 
#################################################################


ppc = pm.sample_ppc(tracez, samples=231, model=model)

# predicted data
for y_tilde in ppc['y_pred']:
    sns.kdeplot(y_tilde, alpha=0.1, c='gray')
# actual data
sns.kdeplot(y, linewidth=3, color='k')
plt.xlabel('$y$', fontsize=16);

######################
#interpreting the posterior
#####################

plt.plot(x, y, 'b.');
alpha_m = tracez['alpha'].mean()
beta_m = tracez['beta'].mean()
plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$zVPD$', fontsize=12)
plt.ylabel('$zNDVI$', fontsize=12)
plt.title('Single Pixel Simple Linear Regression, Using NUTS sampler')
plt.legend(loc=4, fontsize=8)


# with uncertainty

plt.plot(x, y, 'b.');

idx = range(0, len(tracez['alpha']), 10)
plt.plot(x, tracez['alpha'][idx] + tracez['beta'][idx] *  x[:,np.newaxis], c='gray', alpha=0.05);

plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))

plt.title('Single Pixel Simple Linear Regression')
plt.xlabel('$zVPD$', fontsize=16)
plt.ylabel('$zNDVI$', fontsize=16)
plt.legend(loc=4, fontsize=10)



#######
#Bivariate Regression
#######


