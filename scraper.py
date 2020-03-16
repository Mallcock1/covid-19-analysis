# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:16:35 2020

@author: smp16mma
"""

import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_19-covid-Confirmed.csv'
DEATH = 'time_series_19-covid-Deaths.csv'
RECOVERED = 'time_series_19-covid-Recovered.csv'

con_df = pd.read_csv(BASE_URL + CONFIRMED)
rec_df = pd.read_csv(BASE_URL + RECOVERED)
dea_df = pd.read_csv(BASE_URL + DEATH)

# list of headings
headings = list(con_df)

# Combine provinces
con_df = con_df.groupby(['Country/Region'], as_index=False).sum(by=headings[3:])
rec_df = rec_df.groupby(['Country/Region'], as_index=False).sum(by=headings[3:])
dea_df = dea_df.groupby(['Country/Region'], as_index=False).sum(by=headings[3:])

ax = plt.gca()
con_df[headings[4:]].sum().plot()
plt.legend()
plt.show()