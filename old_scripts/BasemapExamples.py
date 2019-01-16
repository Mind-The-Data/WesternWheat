#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:02:40 2018

@author: brian
"""
"""
map =Basemap(projection='stere', lon_0=-100, lat_0=90.,\
            llcrnrlat=min(lats),urcrnrlat=max(lats),\
            llcrnrlon=min(lons),urcrnrlon=max(lons),\
            rsphere=6371200., resolution='l', area_thresh=10000)
"""

from mpl_toolkits.basemap import Basemap
import matplotlib as pyplot

map = Basemap(projection='cyl', lat_0=0, lon_0=0)

#fill globe with color
map.drawmapboundary(fill_color='aqua')
#fill the continents with the land color
map.fillcontinents(color='coral', lake_color='aqua')

map.drawcoastlines()

plt.show()
