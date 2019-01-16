#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:10:40 2018

@author: brian
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
import os
import matplotlib.cm as cm
import pandas as pd
import cartopy.feature as cfeature


plt.rcParams.update({'font.size': 10})

def Plot_point(x,y,value,ax,proj):
	pcount = 0 
	plt.plot(x,y,'.',color = 'red',label='Ponderosa')
	pcount +=1

def scale_bar(ax, length, location=(0.1, 0.22), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    """
    #Projection in metres, need to change this to suit your own figure
    proj = ccrs.AlbersEqualArea(central_longitude=-105,
                central_latitude=45.5,standard_parallels=(29.5,45.5))
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(proj)
    #Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * 500, sbcx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sbcy, sbcy], transform=proj, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbcx, sbcy, str(length) + ' km', transform=proj,
            horizontalalignment='center', verticalalignment='bottom')


def main(forest):

#        dem = pcr.readmap('../Data/DEM.map')
#        DEM = pcr.pcr2numpy(dem, 9999)
#        DEM[DEM==9999]=np.nan

        stamen_terrain = cimgt.StamenTerrain()
        proj = ccrs.AlbersEqualArea(central_longitude=-110,
                    central_latitude=45.5,standard_parallels=(29.5,45.5))
        ax = plt.axes(projection=proj)
        #ax.outline_patch.set_visible(False)


#        minx = -1446763
#        maxy = 2732249
#        maxx = -1340013
#        miny = 2615749
#
#        x = np.linspace(minx,maxx,DEM.shape[1])
#        y = np.linspace(miny,maxy,DEM.shape[0])
#        xx, yy = np.meshgrid(x,y)
#        data = np.flipud(DEM)
#        colormesh = ax.pcolor(xx, yy, data,
#                        transform=proj,
#                        cmap='spring_r',linewidth=0,alpha = .3,vmin=10000,vmax=32000)
	
		
        ax.add_image(stamen_terrain,4)
        datapath = '../Data/Processed/'
        df23June = pd.read_csv(datapath + 'df23June.csv')	
        print list(df23June)
        #df23June = df23June[df23June.Stock_Type == '1-0']
        #df23June = df23June[df23June.Stakes >= 30]
        #df23June = df23June[df23June.Species != 'ES']
        #df23June = df23June[df23June.Stock_Type != 'LP']
        pcount = 0
        
        for index,column in df23June.iterrows():
            plt.plot(column['Longitude'],column['Latitude'],'d',markersize=4,color = 'red',label='Wheat')
            pcount +=1
	#plt.plot(-583100,3062000,'.',color='white',label='Correct')	
	#plt.plot(-583100,3062000,'.',color='black',label='Incorrect')
	#x = np.linspace(-1550000,-1550001,2)
        #y = np.linspace(29000000,29000001,2)
        #xx, yy = np.meshgrid(x,y)
        #data = xx/xx*.5
        #colormesh = ax.pcolor(xx, yy,data ,
        #               transform=proj,
        #               cmap='spectral',linewidth=0,vmin=0,vmax=20)
	
	#plt.plot(-583100,3262000,'.',color='white',label='_nolegend_')	
	#plt.plot(-2500000,1062000,'.',color='white',label='_nolegend_')

	#ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(states_provinces, edgecolor='black')
        plt.legend(loc=3)	
        scale_bar(ax,50)
        #plt.savefig('../Figures/Seedling_defense.png')
        plt.show()

main(1)
#for i in np.arange(1,18):
#	main(i)