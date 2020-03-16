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


# read covid-19 dataset
data = pd.read_csv("..\\covid-19-data\\time-series-19-covid-combined.csv")

# find most recent date
recent_date = data['Date'].max()
recent_data_bool = data['Date'] == recent_date

# data only from most recent date
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
fig = plt.figure()
ax = plt.gca()
country_data = data.groupby(['Country/Region']).sum().sort_values(by='Confirmed').reset_index()
country_data.plot.bar(y='Confirmed', log=True, ax=ax, alpha = 0.5, edgecolor='k')
country_data[['Deaths', 'Recovered']].plot.bar(stacked=True, log=True, ax=ax,
            color=['r', 'limegreen'], alpha = 0.5, edgecolor='k')
plt.xlabel('Country')
plt.ylabel('Number of Cases')
ax.set_xticklabels(country_data['Country/Region'])
plt.show()







#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.coastlines()
#
#plt.show()


# set max number of rows displayed
pd.set_option('display.max_rows', 10)
    

# read population dataset
pop = pd.read_csv("..\\population-data\\population-figures-by-country.csv")

# Latest year's population
pop_16 = pop[['Country', 'Year_2016']]
#print(data.groupby(['Country/Region']).sum())




#population_covid_countries = population_data_2016
cov_countries_drop_dup = data['Country/Region'].drop_duplicates()
countries = list(cov_countries_drop_dup[cov_countries_drop_dup.isin(pop_16['Country'])])

# population data for only countries with cov data
pop_cov_countries = pop_16[pop_16['Country'].isin(countries)]

# cov data for only countries with population data
cov_pop_countries = country_data[country_data['Country/Region'].isin(countries)]

#rel_cov = cov_pop_df['Confirmed'] / cov_pop_df['Population']


# Add population column to data
cov_pop_df = cov_pop_countries.sort_values('Country/Region').join(pd.DataFrame({'Population': np.array(pop_cov_countries['Year_2016'])}))

# Sort cov_pop_df by Confirmed
cov_pop_df = cov_pop_df.sort_values(by='Confirmed')

cov_pop_df['Confirmed Relative'] = cov_pop_df['Confirmed'] / cov_pop_df['Population']
cov_pop_df['Recovered Relative'] = cov_pop_df['Recovered'] / cov_pop_df['Population']
cov_pop_df['Deaths Relative'] = cov_pop_df['Deaths'] / cov_pop_df['Population']

# plot relative cov rates per country
fig = plt.figure()
ax = plt.gca()
cov_pop_df.plot.bar(y='Confirmed Relative', log=True, ax=ax, alpha = 0.5, edgecolor='k')
cov_pop_df[['Deaths Relative', 'Recovered Relative']].plot.bar(stacked=True, log=True, ax=ax,
            color=['r', 'limegreen'], alpha = 0.5, edgecolor='k')
plt.xlabel('Country')
plt.ylabel('Number of Cases')
ax.set_xticklabels(country_data['Country/Region'])

plt.show()
