# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
#import cartopy
#import cartopy.crs as ccrs



data = pd.read_csv("..\\data\\time-series-19-covid-combined.csv")

recent_date = data['Date'].max()

recent_data_bool = data['Date'] == recent_date

latest_data = data[recent_data_bool]

# number of total cases as of latest date in data
total_cases = latest_data['Confirmed'].sum()

total_cases2 = data[data['Date'] == data['Date'].max()]['Confirmed'].sum()

#print(total_cases)
#print(total_cases2)

# plot confirmed, recovered, and deaths through time
def plot_all(data):
    ax = plt.gca()
    data.plot(y='Confirmed', ax=ax)
    data.plot(y='Recovered', ax=ax)
    data.plot(y='Deaths', ax=ax)
    plt.show()



## Plot global virus trends
#ax = plt.gca()
#global_data = data.groupby(['Date']).sum()
#global_data.plot(y='Confirmed', ax=ax)
#global_data.plot(y='Recovered', ax=ax)
#global_data.plot(y='Deaths', ax=ax)
#plt.show()



#anhui_data = data[data['Province/State'] == 'Anhui']
#
#ax = plt.gca()
#
#anhui_data.plot(x='Date', y='Confirmed', ax=ax)
#anhui_data.plot(x='Date', y='Recovered', ax=ax)
#anhui_data.plot(x='Date', y='Deaths', ax=ax)
#plt.show()


#anhui_data = data[data['Country/Region'] == 'Mainland China']
#
## get current axis
#ax = plt.gca()
#
#anhui_data.plot(x='Date', y='Confirmed', ax=ax)
#anhui_data.plot(x='Date', y='Recovered', ax=ax)
#anhui_data.plot(x='Date', y='Deaths', ax=ax)
#plt.show()


#def retrieve_country(data, country):
#    country_data = data[data['Country/Region'] == country].groupby(['Date']).sum()      
#    return country_data
#
#plot_all(retrieve_country(data, 'Mainland China'))
#
#confirmed_data = retrieve_country(data, 'Mainland China')['Confirmed'].to_numpy()

## Fit logistic curve
#def logistic(x, L ,x0, k):
#    y = L / (1 + np.exp(-k*(x-x0)))
#    return y
#
## initial guess parameters
#p0 = [50000, 20, 10]
#
#popt, pcov = curve_fit(logistic, range(len(confirmed_data)), confirmed_data, p0)#, method='dogbox')
#
## logistic function with optimised parameters
#def logistic_optimised(x):
#    L, x0, k = popt
#    y = L / (1 + np.exp(-k*(x-x0)))
##    return y
#
#
## Fit double logistic curve
#def double_logistic(x, L1, x01, k1, L2, x02, k2):
#    y = L1 / (1 + np.exp(-k1*(x - x01))) + L2 / (1 + np.exp(-k2*(x - x02)))
#    return y
#
## initial guess parameters
#p0 = [10000, 15, 1, 60000, 25, 1]
#bounds = ([0, 5, 0, 40000, 15, 0], [25000, 15, 1000000, 70000, 50, 1000000])
#
#popt, pcov = curve_fit(double_logistic, range(len(confirmed_data)), confirmed_data, p0)#, bounds=bounds)
#
## logistic function with optimised parameters
#def double_logistic_optimised(x):
#    L1, x01, k1, L2, x02, k2 = popt
#    y = L1 / (1 + np.exp(-k1*(x - x01))) + L2 / (1 + np.exp(-k2*(x - x02)))
#    return y
#
#x = np.linspace(0, 50)
#
#plt.plot(x, double_logistic_optimised(x))

#
#
#
#style.use('default')
#
## Number of cases per country
#fig = plt.figure()
#ax = plt.gca()
#country_data = data.groupby(['Country/Region']).sum().sort_values(by='Confirmed')
#country_data.plot.bar(y='Confirmed', log=True, ax=ax, alpha = 0.5, edgecolor='k')
#country_data[['Deaths', 'Recovered']].plot.bar(stacked=True, log=True, ax=ax,
#            color=['r', 'limegreen'], alpha = 0.5, edgecolor='k')
#plt.xlabel('Country')
#plt.ylabel('Number of Cases')
#plt.show()







ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

plt.show()