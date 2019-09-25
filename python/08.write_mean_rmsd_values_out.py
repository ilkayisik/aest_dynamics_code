#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
write root mean squared difference values out
saves: mean_rmsd_rate.csv
mean_rmsd_view.csv
mean_rmsd.csv
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/ilkay.isik/aesthetic_dynamics/aest_dynamics_code/python/functions')
from useful_functions import rmsd
# load the data
# %% set paths and load the data
f_dirs = ['/Users/ilkay.isik/aesthetic_dynamics/data/data_rate.npz',
          '/Users/ilkay.isik/aesthetic_dynamics/data/data_view.npz']
savepath = '/Users/ilkay.isik/aesthetic_dynamics/'
data = []
for i in range(2):
    data.append(np.load(f_dirs[i]))
print (data[0].files)
# %% BP03 or Experiment 2 Version 1 (Rate group)
rt = data[0]['cData_ses1'] # rate test
rrt = data[0]['cData_ses2'] # rate retest
vrt = data[1]['cData_ses2'] # view retest
cTimeSes1 = data[0]['cTime_ses1']
cTime = cTimeSes1[0, 0, :]  # to use in the plots

# experiment parameters
nr_movies = data[0]['nr_movies']
nr_subjects = data[0]['nr_subjects']
nr_subjects2 = data[1]['nr_subjects']
nr_tp = len(cTime)
nr_sessions = 2
# %% Calculate the rate of change with root mean of squared differences
rate_rmsd = np.zeros([nr_sessions, nr_subjects, nr_movies ])
for m in range(nr_movies):  # loop through movies
    for s in range(nr_subjects):  # loop through subjects
        # curves of interest
        t_curve = rt[s, m, :]  # test
        rate_rmsd[0, s, m] = rmsd(t_curve)
        rt_curve = rrt[s, m, :]  # retest
        rate_rmsd[1, s, m] = rmsd(rt_curve)

view_rmsd = np.zeros([nr_subjects, nr_movies])
for m in range(nr_movies):  # loop through movies
    for s in range(nr_subjects):  # loop through subjects
        # curve of interest
        rt_curve = vrt[s, m, :]  # retest
        view_rmsd[s, m] = rmsd(rt_curve)
# %% calculate one mean rmsd value for each subject [for different categories and ses]
# RATE GROUP
rate_mean_rmsd = np.mean(rate_rmsd, axis=2) # per subject
# all means comb
rate_mean_rmsd_comb = np.mean(rate_mean_rmsd, axis=0) 

rate_mean_dnc_test = np.mean(rate_rmsd[0, :, :15], axis=1)
rate_mean_dnc_retest = np.mean(rate_rmsd[1, :, :15], axis=1)
# dnc combined across sessions
rate_mean_dnc_comb = np.mean(np.mean(rate_rmsd[:, :, :15], axis=2), axis=0)

rate_mean_lsp_test = np.mean(rate_rmsd[0, :, 15:], axis=1)
rate_mean_lsp_retest = np.mean(rate_rmsd[1, :, 15:], axis=1)
# lsp combined across sessions
rate_mean_lsp_comb = np.mean(np.mean(rate_rmsd[:, :, 15:], axis=2), axis=0)

# all means comb
view_mean_rmsd = np.mean(view_rmsd, axis=1)
view_mean_dnc = np.mean(view_rmsd[:, :15], axis=1)
view_mean_lsp = np.mean(view_rmsd[:, 15:], axis=1)

# write it out in the csv format
df_rate = pd.DataFrame(rate_mean_rmsd.T)
df_rate.columns = ["mean_rmsd_t", "mean_rmsd_rt"]

df_rate['mean_rmsd_comb'] = rate_mean_rmsd_comb
df_rate['mean_rmsd_dnc_t'] = rate_mean_dnc_test
df_rate['mean_rmsd_dnc_rt'] = rate_mean_dnc_retest
df_rate['mean_rmsd_dnc_comb'] = rate_mean_dnc_comb
df_rate['mean_rmsd_lsp_t'] = rate_mean_dnc_test
df_rate['mean_rmsd_lsp_rt'] = rate_mean_dnc_retest
df_rate['mean_rmsd_lsp_comb'] = rate_mean_lsp_comb
df_rate['group'] = 'rate'
# df_rate.to_csv(savepath + "data/mean_rmsd_rate.csv", index=None)

df_view = pd.DataFrame(view_mean_rmsd.T)
df_view.columns = ["mean_rmsd_rt"]
df_view['mean_rmsd_dnc_rt'] = view_mean_dnc
df_view['mean_rmsd_lsp_rt'] = view_mean_lsp
df_view['group'] = 'view'
# df_view.to_csv(savepath + "data/mean_rmsd_view.csv", index=None)

frames = [df_rate, df_view]
comb = pd.concat(frames, sort=True)
# comb.to_csv(savepath + "data/mean_rmsd.csv", index=None)

