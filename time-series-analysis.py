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
import cartopy.crs as ccrs
import cartopy as ct


make_map_image = False
make_map_animation = False
make_country_ts = False
make_global_ts = False
fit_curve = False
make_country_bar = False
make_country_bar_log = False

#make_map_image = True
#make_map_animation = True
make_country_ts = True
#make_global_ts = True
fit_curve = True
#make_country_bar = True
#make_country_bar_log = True

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


#df.columns = pd.to_datetime(df.columns)
#def to_normal_date(old_dates):
#    old_dates_updated_year = [d[:-2] + '2020' for d in old_dates]
#    new_dates = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in old_dates_updated_year]
#    return new_dates


#col_names[4:] = to_normal_date(col_names[4:])
#
#new_dates_df = pd.to_datetime(con_df_ungrouped.columns[4:])
#con_df_ungrouped.columns = new_dates_df.union(con_df_ungrouped.columns[:4])
#dea_df_ungrouped.columns = con_df_ungrouped.columns
#rec_df_ungrouped.columns = con_df_ungrouped.columns

# list of column names
col_names = list(con_df_ungrouped)

#[d[:-2] + '2020' for d in old_dates]

con_df_ungrouped.columns = col_names
dea_df_ungrouped.columns = col_names
rec_df_ungrouped.columns = col_names


# Combine provinces
con_df = con_df_ungrouped.groupby(['Country/Region'], as_index=False).sum(by=col_names[3:])
rec_df = rec_df_ungrouped.groupby(['Country/Region'], as_index=False).sum(by=col_names[3:])
dea_df = dea_df_ungrouped.groupby(['Country/Region'], as_index=False).sum(by=col_names[3:])

    
# Create dataframe with only date columns
con_df_time_only = con_df[col_names[4:]]
con_df_time_only.columns = pd.to_datetime(con_df_time_only.columns)
dea_df_time_only = dea_df[col_names[4:]]
dea_df_time_only.columns = pd.to_datetime(dea_df_time_only.columns)
rec_df_time_only = rec_df[col_names[4:]]
rec_df_time_only.columns = pd.to_datetime(rec_df_time_only.columns)

# file path for saving figures
fig_path = '..\\figures\\'


if make_country_ts:
    # Plot time-series for a given country
    country = 'China'
    plt.figure()
    ax1 = plt.gca()
    
    if fit_curve:
        # Fit logistic curve
        def logistic(x, L ,x0, k):
            y = L / (1 + np.exp(-k*(x-x0)))
            return y
        
        # initial guess parameters
        p0 = [50000, 20, 10]
#        p0 = [3000, 30, 1]
#        bounds = ([100, 10, 0.01], [10000, 100, 100])
        
        # Country time_series
        data = con_df_time_only.loc[con_df['Country/Region'] == country].transpose()

        # Fit curve
        popt, pcov = curve_fit(logistic, np.arange(len(data)), data.iloc[:,0], 
                               p0)#, bounds=bounds)
        
        # logistic function with optimised parameters
        def logistic_optimised(x):
            L, x0, k = popt
            y = L / (1 + np.exp(-k*(x-x0)))
            return y

        N = len(col_names[4:])
        x = np.linspace(0, N-1, N)
        # dates for dataframe
        x_dates = pd.date_range(start='1/22/2020', periods=N)
        l_opt_df = pd.DataFrame({'Date': x_dates,
                                 'logistic': logistic_optimised(x)})
        l_opt_df = l_opt_df.set_index('Date')
        l_opt_df.plot(ax=ax1, label='Logistic', color='k', linestyle='--')
    
    con_df_time_only.loc[con_df['Country/Region'] == country].transpose().plot(ax=ax1, 
                        label='Confirmed', color='b', legend=True)
    rec_df_time_only.loc[rec_df['Country/Region'] == country].transpose().plot(ax=ax1, label='Recovered', color='limegreen')
    dea_df_time_only.loc[dea_df['Country/Region'] == country].transpose().plot(ax=ax1, label='Deaths', color='r')
    
    if fit_curve:
        legend = ['Logistic', 'Confirmed', 'Recovered', 'Deaths']
    else:
        legend = ['Confirmed', 'Recovered', 'Deaths']
    ax1.legend(legend, loc=2)
    plt.xlabel('Date')
    plt.ylabel('Number of cases')
    plt.title('Covid-19 cases in ' + country)
    plt.tight_layout()
    plt.show()
    
    if fit_curve:
        save_name = 'ts_' + country + '_' + col_names[-1].replace('/', '') + '_l.png'
    else:
        save_name = 'ts_' + country + '_' + col_names[-1].replace('/', '') + '.png'
        
    plt.savefig(fig_path + save_name)


if make_global_ts:
    # Plot total time series
    plt.figure()
    ax2 = plt.gca()
    
    if fit_curve:
         # Fit double logistic curve
        def double_logistic(x, L1, x01, k1, L2, x02, k2):
            y = L1 / (1 + np.exp(-k1*(x - x01))) + L2 / (1 + np.exp(-k2*(x - x02)))
            return y
        
        # global time-series
        data = con_df[col_names[4:]].sum()
        
        # initial guess parameters
        p0 = [10000, 15, 1, 60000, 25, 1]
        bounds = ([0, 5, 0, 40000, 15, 0], [25000, 15, 1000000, 70000, 50, 1000000])
        
        popt, pcov = curve_fit(double_logistic, range(len(data)), data, p0)#, bounds=bounds)
        
        # logistic function with optimised parameters
        def double_logistic_optimised(x):
            L1, x01, k1, L2, x02, k2 = popt
            y = L1 / (1 + np.exp(-k1*(x - x01))) + L2 / (1 + np.exp(-k2*(x - x02)))
            return y
        
        N = len(col_names[4:])
        x = np.linspace(0, N - 1, N)
        # dates for dataframe
        x_dates = pd.date_range(start='1/22/2020', periods=N)
        dl_opt_df = pd.DataFrame({'Date': x_dates,
                                 'Double logistic': double_logistic_optimised(x)})
        dl_opt_df = dl_opt_df.set_index('Date')
        dl_opt_df.plot(ax=ax2, label='Double logistic', color='k', linestyle='--')
    
    con_df_time_only.sum().plot(ax=ax2,
          label='Confirmed', color='b')
    rec_df_time_only.sum().plot(ax=ax2,
          label='Recovered', color='limegreen')
    dea_df_time_only.sum().plot(ax=ax2,
          label='Deaths', color='r')
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Number of cases')
    plt.title('Global Covid-19 cases')
    plt.tight_layout()
    plt.show()
    
    if fit_curve:
        save_name = 'ts_global_dl.png'
    else:
        save_name = 'ts_global.png'
        
    plt.savefig(fig_path + save_name)


if make_country_bar:
    ## Number of cases per country
    
    #latest data
    con_df_latest = con_df.drop(col_names[4:-1], axis=1)
    dea_df_latest = dea_df.drop(col_names[4:-1], axis=1)
    rec_df_latest = rec_df.drop(col_names[4:-1], axis=1)
    
    # combine latest cases into one df
    comb_df = con_df_latest.join(dea_df_latest[col_names[-1]], rsuffix='_D')
    comb_df = comb_df.join(rec_df_latest[col_names[-1]], rsuffix='_R')
    
    comb_df = comb_df.rename(columns={col_names[-1]: "Confirmed",
                                      col_names[-1] + '_D': "Deaths",
                                      col_names[-1] + '_R': "Recovered"})
    
    # sort by confirmed
    comb_df = comb_df.sort_values(by='Confirmed')
    fig = plt.figure(figsize=[19,10])
    ax = plt.gca()
    
    # log?
    log = False
    
    comb_df.plot.bar(y='Confirmed', log=log, ax=ax, alpha = 0.5, edgecolor='k')
    comb_df[['Deaths', 'Recovered']].plot.bar(stacked=True, log=log, ax=ax,
           alpha = 0.5, edgecolor='k', color=['r', 'limegreen'])
    plt.xlabel('Country', fontsize=20)
    plt.ylabel('Number of Cases', fontsize=20)
    plt.yticks(fontsize=15)
    plt.title('Covid-19 cases by country', fontsize=30)
    ax.set_xticklabels(comb_df['Country/Region'])
    plt.tight_layout()
    plt.legend(fontsize=20, loc=2)
    if log == True:
        plt.ylim(bottom=0.5)
    plt.show()
    
    plt.savefig(fig_path + 'bar.png')
    
if make_country_bar_log:
    ## Number of cases per country
    
    #latest data
    con_df_latest = con_df.drop(col_names[4:-1], axis=1)
    dea_df_latest = dea_df.drop(col_names[4:-1], axis=1)
    rec_df_latest = rec_df.drop(col_names[4:-1], axis=1)
    
    # combine latest cases into one df
    comb_df = con_df_latest.join(dea_df_latest[col_names[-1]], rsuffix='_D')
    comb_df = comb_df.join(rec_df_latest[col_names[-1]], rsuffix='_R')
    
    comb_df = comb_df.rename(columns={col_names[-1]: "Confirmed",
                                      col_names[-1] + '_D': "Deaths",
                                      col_names[-1] + '_R': "Recovered"})
    
# sort by confirmed
    comb_df = comb_df.sort_values(by='Confirmed')
    
    fig = plt.figure(figsize=[19,10])
    ax = plt.gca()
    
    # Plot on a log scale?
    log = True
    
    (comb_df['Confirmed'] - 0.0001).plot.bar(log=log, alpha = 0.5, edgecolor='k') # -0.0001 sorts out some weird bug in the plotting probably becasue it turns int into float
    comb_df[['Deaths', 'Recovered']].plot.bar(stacked=True, log=log, ax=ax,
           alpha = 0.5, edgecolor='k',
            color=['r', 'limegreen'])
    plt.xlabel('Country', fontsize=20)
    plt.ylabel('Number of Cases', fontsize=20)
    plt.title('Covid-19 cases by country', fontsize=30)
    plt.yticks(fontsize=15)
    ax.set_xticklabels(comb_df['Country/Region'])
    plt.legend(fontsize=20, loc=2)
    plt.tight_layout()
    if log == True:
        plt.ylim(bottom=0.5)
    plt.show()
    
    plt.savefig(fig_path + 'bar_log.png')



if make_map_image:
    fig = plt.figure(figsize=[19,10])
    
    # choose map projection
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Add map features
    ax.coastlines()
    ax.add_feature(ct.feature.OCEAN)
    ax.add_feature(ct.feature.BORDERS)
    #ax.add_feature(ct.feature.LAND, edgecolor='black')
    #ax.add_feature(ct.feature.LAKES, edgecolor='black')
    #ax.add_feature(ct.feature.RIVERS)
    
    date = col_names[-1] # -1 for latest, 4 for earliest
    
    # size of points
    abs_size = 50000
    def size_func(data):
        return abs_size * data/con_df_ungrouped[col_names[-1]].max()
    
    sizes = size_func(con_df_ungrouped[date])
    scat = plt.scatter(con_df_ungrouped['Long'], con_df_ungrouped['Lat'],
             color='r', s=sizes, marker='o', alpha=0.5,
             transform=ccrs.Geodetic())
    scale_size = abs_size*10000/con_df_ungrouped[col_names[-1]].max()
    plt.scatter([-150],[-30], color='r', s=scale_size, transform=ccrs.Geodetic())
    plt.text(-150,-30, '10,000 people', transform=ccrs.Geodetic(), ha='center',
             fontsize=20)
    date_text = plt.text(-150,0, 'Date: ' + date, transform=ccrs.Geodetic(),
                         ha='center', fontsize=20)
    plt.title('Global Covid-19 cases', fontsize=30)
    ax.set_global() # shows the whole map
    
    plt.savefig(fig_path + 'map_image.png')


if make_map_animation:
    def update_sizes(i, confirmed, scatter):
        scatter.set_sizes(size_func(confirmed[col_names[4 + i]]))
        date_text.set_text('Date: ' + col_names[4 + i])
        return scatter, date_text
    
    n_frames = len(col_names[4:])
    ani = animation.FuncAnimation(fig, update_sizes, frames=range(n_frames),
                                  fargs=(con_df_ungrouped[col_names[4:]], scat),
                                  repeat_delay=3000)
    plt.show()
    
    ani.save(fig_path + 'map_animation.mp4')
    