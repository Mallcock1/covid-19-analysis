# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#import cartopy
#import cartopy.crs as ccrs


# read covid-19 dataset
data = pd.read_csv("..\\covid-19-data\\time-series-19-covid-combined.csv")


# Read original covid-19 dataset
BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_19-covid-Confirmed.csv'
DEATH = 'time_series_19-covid-Deaths.csv'
RECOVERED = 'time_series_19-covid-Recovered.csv'

con_df_ungrouped = pd.read_csv(BASE_URL + CONFIRMED)
rec_df_ungrouped = pd.read_csv(BASE_URL + RECOVERED)
dea_df_ungrouped = pd.read_csv(BASE_URL + DEATH)

# list of headings
headings = list(con_df_ungrouped)

# Combine provinces
con_df = con_df_ungrouped.groupby(['Country/Region'], as_index=False).sum(by=headings[3:])
rec_df = rec_df_ungrouped.groupby(['Country/Region'], as_index=False).sum(by=headings[3:])
dea_df = dea_df_ungrouped.groupby(['Country/Region'], as_index=False).sum(by=headings[3:])

## Plot time-series for a given country
#country = 'United Kingdom'
#plt.figure()
#ax1 = plt.gca()
#con_df.loc[con_df['Country/Region'] == country].iloc[0][4:].plot(ax=ax1, label='Confirmed', color='b')
#rec_df.loc[rec_df['Country/Region'] == country].iloc[0][4:].plot(ax=ax1, label='Recovered', color='limegreen')
#dea_df.loc[dea_df['Country/Region'] == country].iloc[0][4:].plot(ax=ax1, label='Deaths', color='r')
#plt.legend()
#plt.xlabel('Date')
#plt.ylabel('Number of cases')
#plt.title('Covid-19 cases in ' + country)
#plt.show()


## Plot total time series
#plt.figure()
#ax2 = plt.gca()
#con_df[headings[4:]].sum().plot(ax=ax2,
#      label='Confirmed', color='b')
#rec_df[headings[4:]].sum().plot(ax=ax2,
#      label='Recovered', color='limegreen')
#dea_df[headings[4:]].sum().plot(ax=ax2,
#      label='Deaths', color='r')
#plt.legend()
#plt.xlabel('Date')
#plt.ylabel('Number of cases')
#plt.title('Global Covid-19 cases')
#plt.show()

#
## Fit logistic curve
#def logistic(x, L ,x0, k):
#    y = L / (1 + np.exp(-k*(x-x0)))
#    return y
#
## initial guess parameters
##p0 = [50000, 20, 10]
#p0 = [1000, 50, 1]
#
## global time-series
#data = con_df[headings[4:]].sum()
#
### Country time_series
##data = np.array(con_df.loc[con_df['Country/Region'] == country].iloc[0][4:]).astype(float)
#
### Fit curve
##popt, pcov = curve_fit(logistic, np.arange(len(data)), data, p0)#, method='dogbox')
##
### logistic function with optimised parameters
##def logistic_optimised(x):
##    L, x0, k = popt
##    y = L / (1 + np.exp(-k*(x-x0)))
##    return y
##
##x = np.linspace(0, len(headings)*2, 100)
##
##ax2.plot(x, double_logistic_optimised(x))
##plt.xlim([0, 100])
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
#popt, pcov = curve_fit(double_logistic, range(len(data)), data, p0)#, bounds=bounds)
#
## logistic function with optimised parameters
#def double_logistic_optimised(x):
#    L1, x01, k1, L2, x02, k2 = popt
#    y = L1 / (1 + np.exp(-k1*(x - x01))) + L2 / (1 + np.exp(-k2*(x - x02)))
#    return y
#
#
#x = np.linspace(0, len(headings)*2, 100)
#
#ax2.plot(x, double_logistic_optimised(x))
#plt.xlim([0, 100])






#
### Number of cases per country
#
##latest data
#con_df_latest = con_df.drop(headings[4:-1], axis=1)
#dea_df_latest = dea_df.drop(headings[4:-1], axis=1)
#rec_df_latest = rec_df.drop(headings[4:-1], axis=1)
#
## combine latest cases into one df
#comb_df = con_df_latest.join(dea_df_latest[headings[-1]], rsuffix='_D')
#comb_df = comb_df.join(rec_df_latest[headings[-1]], rsuffix='_R')
#
#comb_df = comb_df.rename(columns={headings[-1]: "Confirmed", headings[-1] + '_D': "Deaths", headings[-1] + '_R': "Recovered"})
#
## sort by confirmed
#comb_df = comb_df.sort_values(by='Confirmed')
#fig = plt.figure()
#ax = plt.gca()
#
## log?
#log = False
#
#comb_df.plot.bar(y='Confirmed', log=log, ax=ax, alpha = 0.5, edgecolor='k')
#comb_df[['Deaths', 'Recovered']].plot.bar(stacked=True, log=log, ax=ax, alpha = 0.5, edgecolor='k',
#        color=['r', 'limegreen'])
#plt.xlabel('Country')
#plt.ylabel('Number of Cases')
#ax.set_xticklabels(comb_df['Country/Region'])
#plt.tight_layout()
#if log == True:
#    plt.ylim(bottom=0.5)
#plt.show()


import cartopy.crs as ccrs
import cartopy as ct


plt.figure(figsize=[19,10])
#ax = plt.axes(projection=ccrs.PlateCarree())
ax = plt.axes(projection=ccrs.Robinson())


ax.coastlines()
ax.add_feature(ct.feature.OCEAN)
ax.add_feature(ct.feature.BORDERS)
#ax.add_feature(ct.feature.LAND, edgecolor='black')
#ax.add_feature(ct.feature.LAKES, edgecolor='black')
#ax.add_feature(ct.feature.RIVERS)

# size of points
sizes = 10000 * con_df_ungrouped[headings[-1]]/con_df_ungrouped[headings[-1]].max()
plt.scatter(con_df_ungrouped['Long'], con_df_ungrouped['Lat'],
         color='r', s=sizes, marker='o', alpha=0.5,
         transform=ccrs.Geodetic(),
         )
scale_size = 10000*10000/con_df_ungrouped[headings[-1]].max()
plt.scatter([-150],[-30], color='r', s=scale_size, transform=ccrs.Geodetic())
#plt.annotate('Here', [-15000000,-30], transform=ccrs.Geodetic())
plt.text(-150,-30, '10,000 people', transform=ccrs.Geodetic(), ha='center')
plt.title('Global Covid-19 cases')
ax.set_global()
plt.show()


# animation
def update_sizes(i, sizes, scatter):
    scatter.set_sizes(array[i])
    return scatter,

fig = plt.figure()
scat = plt.scatter(x, y, c=c, s=100)

ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                              fargs=(color_data, scat))
plt.show()


















#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.coastlines()
#
#plt.show()
#
#
## set max number of rows displayed
#pd.set_option('display.max_rows', 10)
#    
#
## read population dataset
#pop = pd.read_csv("..\\population-data\\population-figures-by-country.csv")
#
## Latest year's population
#pop_16 = pop[['Country', 'Year_2016']]
##print(data.groupby(['Country/Region']).sum())
#
#
#
#
##population_covid_countries = population_data_2016
#cov_countries_drop_dup = data['Country/Region'].drop_duplicates()
#countries = list(cov_countries_drop_dup[cov_countries_drop_dup.isin(pop_16['Country'])])
#
## population data for only countries with cov data
#pop_cov_countries = pop_16[pop_16['Country'].isin(countries)]
#
## cov data for only countries with population data
#cov_pop_countries = country_data[country_data['Country/Region'].isin(countries)]
#
##rel_cov = cov_pop_df['Confirmed'] / cov_pop_df['Population']
#
## Sort and reindex countries
#cov_pop_countries =  cov_pop_countries.sort_values('Country/Region')
#cov_pop_countries = cov_pop_countries.reset_index()
#del cov_pop_countries['index']
#
#pop_cov_countries =  pop_cov_countries.sort_values('Country')
#pop_cov_countries = pop_cov_countries.reset_index()
#del pop_cov_countries['index']
#
## Add population column to data
#cov_pop_df = cov_pop_countries.join(pd.DataFrame({'Population': np.array(pop_cov_countries['Year_2016'])}))
#
## Sort cov_pop_df by Confirmed
#cov_pop_df = cov_pop_df.sort_values(by='Confirmed')
#
#cov_pop_df['Confirmed Relative'] = cov_pop_df['Confirmed'] / cov_pop_df['Population']
#cov_pop_df['Recovered Relative'] = cov_pop_df['Recovered'] / cov_pop_df['Population']
#cov_pop_df['Deaths Relative'] = cov_pop_df['Deaths'] / cov_pop_df['Population']
#
#
## plot cov rates per country
#fig = plt.figure()
#ax = plt.gca()
#cov_pop_df.plot.bar(y='Confirmed', log=True, ax=ax, alpha = 0.5, edgecolor='k')
#cov_pop_df[['Deaths', 'Recovered']].plot.bar(stacked=True, log=True, ax=ax,
#            color=['r', 'limegreen'], alpha = 0.5, edgecolor='k')
#plt.xlabel('Country')
#plt.ylabel('Number of cases')
#ax.set_xticklabels(cov_pop_df['Country/Region'])
#
#
## plot relative cov rates per country ordered by number of confirmed cases
#fig = plt.figure()
#ax = plt.gca()
#cov_pop_df.plot.bar(y='Confirmed Relative', log=True, ax=ax, alpha = 0.5, edgecolor='k')
#cov_pop_df[['Deaths Relative', 'Recovered Relative']].plot.bar(stacked=True, log=True, ax=ax,
#            color=['r', 'limegreen'], alpha = 0.5, edgecolor='k')
#plt.xlabel('Country')
#plt.ylabel('Proportion of population')
#ax.set_xticklabels(cov_pop_df['Country/Region'])
#
#
## plot relative cov rates per country
#fig = plt.figure()
#ax = plt.gca()
#cov_pop_df = cov_pop_df.sort_values(by='Confirmed Relative')
#cov_pop_df.plot.bar(y='Confirmed Relative', log=True, ax=ax, alpha = 0.5, edgecolor='k')
#cov_pop_df[['Deaths Relative', 'Recovered Relative']].plot.bar(stacked=True, log=True, ax=ax,
#            color=['r', 'limegreen'], alpha = 0.5, edgecolor='k')
#plt.xlabel('Country')
#plt.ylabel('Proportion of population')
#ax.set_xticklabels(cov_pop_df['Country/Region'])
#
#plt.show()
