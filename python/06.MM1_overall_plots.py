#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean-minus-one agreement calculations  with overall data
- saves: mm1_overall.csv
Fig06_A_Rate_mm1Agreement_Overall.pdf
Fig06_A_View_mm1Agreement_Overall.pdf
@author: ilkay isik
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/ilkay.isik/aesthetic_dynamics/aest_dynamics_code/python/functions')
import stats as Stats
import matplotlib.pyplot as plt
import seaborn as sns
# %% loading the data
f_dirs = ['/Users/ilkay.isik/aesthetic_dynamics/data/data_rate.npz',
          '/Users/ilkay.isik/aesthetic_dynamics/data/data_view.npz']
savepath = '/Users/ilkay.isik/aesthetic_dynamics/'
data = []
for i in range(2):
    data.append(np.load(f_dirs[i]))
print (data[0].files)

# Load the data
rot = data[0]['oData_ses1']  # rate overall test
rort = data[0]['oData_ses2']  # rate overall retest
vot = data[1]['oData_ses1']  # view overall test
vort = data[1]['oData_ses2']  # view overall retest
# experiment parameters
nr_movies = data[0]['nr_movies']
nr_sub_r = data[0]['nr_subjects']
nr_sub_v = data[1]['nr_subjects']
nr_sub = nr_sub_r  + nr_sub_v # total nr of subject
nr_runs = data[0]['nr_runs']

# in python 3 the result of this is bytes instead of list. therefore applying 
# list comp to do the change to list with strings
cat_lol = data[0]['cat_lol']
cat_lol  = [[mov.decode("utf-8") for mov in group] for group in cat_lol]
# the same here
moi_list = data[0]['moi_list']
moi_list = [mov.decode("utf-8") for mov in moi_list]
# the same here
cats =  data[0]['categories']
cats = [c.decode("utf-8") for c in cats]

nr_movies_per_run = data[0]['nr_movies_per_run']
nr_cats = len(cats)
movie_dur = data[0]['movie_dur']
nr_sessions = 2
movie_count = [len(f) for f in cat_lol]
m_count_cum = np.cumsum(movie_count)  # cumulative sum
m_count = np.insert(m_count_cum, 0, 0)  # insert zero
# %% OVERALL DATA MM1 Calculations
# Rate Test
ov_mean_dict, ov_ci_dict = dict(), dict()
mm1_df_rot = pd.DataFrame()
for i in range(nr_cats):
    temp_dat = rot[:, m_count[i] : m_count[i+1]]
    mm1_rot, corr_rot, mean_rot, ci_rot, ttest_t_rot, ttest_p_rot, cohD_rot = \
    Stats.compute_mm1(temp_dat.T)
    corr_ztrans = Stats.rtoz(corr_rot)
    cat = cats[i]
    ov_mean_dict[cat + '-rate-test'] = [mean_rot]
    ov_ci_dict[cat + '-rate-test'] = [ci_rot[0], ci_rot[1]]
    temp_df_rot = pd.DataFrame({'mm1_corr': corr_rot,
                                'ztrans_mm1': corr_ztrans,
                                'category': [cats[i]] * int(nr_sub_r),
                                'session': ['Test'] * int(nr_sub_r),
                                'sub': range(1, nr_sub_r + 1)})
    mm1_df_rot = mm1_df_rot.append(temp_df_rot)

# Rate Retest
rort[21, 15]=0.99 # there is one subject with lscp values of all 1, so there is 
# an error when correlating their values in MM1, therefore I change the value 
# to be 0.99
mm1_df_rort = pd.DataFrame()
for i in range(nr_cats):
    temp_dat = rort[:, m_count[i]:m_count[i+1]]
    mm1_rort, corr_rort, mean_rort, ci_rort, ttest_t_rort, ttest_p_rort, cohD_rort = \
    Stats.compute_mm1(temp_dat.T)
    corr_ztrans = Stats.rtoz(corr_rort)
    cat = cats[i]
    ov_mean_dict[cat + '-rate-retest'] = [mean_rort]
    ov_ci_dict[cat + '-rate-retest'] = [ci_rort[0], ci_rort[1]]

    temp_df_rort = pd.DataFrame({'mm1_corr': corr_rort,
                                 'ztrans_mm1': corr_ztrans,
                                 'category': [cats[i]] * int(nr_sub_r),
                                 'session': ['Retest'] * int(nr_sub_r),
                                 'sub': range(1, nr_sub_r + 1)})
    mm1_df_rort = mm1_df_rort.append(temp_df_rort)

# View Test
mm1_df_vot = pd.DataFrame()
for i in range(nr_cats):
    temp_dat = vot[:, m_count[i]:m_count[i+1]]
    mm1_vot, corr_vot, mean_vot, ci_vot, ttest_t_vot, ttest_p_vot, cohD_vot = \
    Stats.compute_mm1(temp_dat.T)
    corr_ztrans = Stats.rtoz(corr_vot)
    cat = cats[i]
    ov_mean_dict[cat + '-view-test'] = [mean_vot]
    ov_ci_dict[cat + '-view-test'] = [ci_vot[0], ci_vot[1]]
    temp_df_vot = pd.DataFrame({'mm1_corr': corr_vot,
                                'ztrans_mm1': corr_ztrans,
                                'category': [cats[i]] * int(nr_sub_v),
                                'session': ['Test'] * int(nr_sub_v),
                                'sub': range(nr_sub_r + 1, nr_sub + 1)})
    mm1_df_vot = mm1_df_vot.append(temp_df_vot)

# View Retest
mm1_df_vort = pd.DataFrame()
for i in range(nr_cats):
    temp_dat = vort[:, m_count[i]:m_count[i+1]]
    mm1_vort, corr_vort, mean_vort, ci_vort, ttest_t_vort, ttest_p_vort, cohD_vort = \
    Stats.compute_mm1(temp_dat.T)
    corr_ztrans = Stats.rtoz(corr_vort)
    cat = cats[i]
    ov_mean_dict[cat + '-view-retest'] = [mean_vort]
    ov_ci_dict[cat + '-view-retest'] = [ci_vort[0], ci_vort[1]]
    temp_df_vort = pd.DataFrame({'mm1_corr': corr_vort,
                                 'ztrans_mm1': corr_ztrans,
                                 'category': [cats[i]] * int(nr_sub_v),
                                 'session': ['Retest'] * int(nr_sub_v),
                                 'sub': range(nr_sub_r + 1, nr_sub + 1)})
    mm1_df_vort = mm1_df_vort.append(temp_df_vort)
# %% combine data frames from sessions and groups
frames_o_rate = [mm1_df_rot, mm1_df_rort]
mm1_odf_rate = pd.concat(frames_o_rate)
mm1_odf_rate['group'] = 'Rate'
frames_o_view = [mm1_df_vot, mm1_df_vort]
mm1_odf_view = pd.concat(frames_o_view)
mm1_odf_view['group'] = 'View'

frames_o = [mm1_odf_rate, mm1_odf_view]
mm1_odf = pd.concat(frames_o)

save_mm1_overall = savepath + 'data/mm1_overall.csv'
# mm1_odf.to_csv(save_mm1_overall)
#%% PLOTS for MM1 correlations with OVERALL DATA
# Rate Plot
flatui = ["#F8766D", "#00BFC4"]
sns.set_palette(flatui)

# Overall MM1 Rate Plot
sns.set_style('white')
sns.set_context('paper',font_scale=2 )

# rate: dance-test, dance-retest, lscp-test,  lscp-retest
rate_means = [0.43284665777289516, 0.3080370835208762, 0.4448158724987995, 0.389249779573164]
ci_lower =[0.2887377985799719, 0.1492931223148273, 0.30692199759695254,  0.27754048182854274]
ci_upper = [0.5577767150871706,  0.4513012591773139, 0.5644183868066074, 0.4905738425230475]

plt.figure(figsize=(8, 6))
sns.set_style("ticks")
ax = sns.stripplot(x="category", y="mm1_corr", hue='session',
                   data=mm1_odf_rate ,
                   split=True, jitter=True,
                   edgecolor='black',
                   palette=flatui)
sns.despine()
ax.legend(loc='lower right')
ax.set_xlabel('Category')
ax.set_ylabel('"Mean-minus-one" Agreement')
plt.ylim((-0.6, 1))


ax.plot([-0.204, -0.204], [ci_lower[0], ci_upper[0]], 'k-', lw=2)
ax.plot([0.200, 0.200], [ci_lower[1], ci_upper[1]], 'k-', lw=2)
ax.plot([0.801, 0.801], [ci_lower[2], ci_upper[2]], 'k-', lw=2)
ax.plot([1.2, 1.2], [ci_lower[3], ci_upper[3]], 'k-', lw=2)

ax.scatter(-0.204, rate_means[0], marker="d", s=200, facecolor='k')
ax.scatter(0.200, rate_means[1], marker="d", s=200, facecolor='k')
ax.scatter(0.801, rate_means[2], marker="d", s=200, facecolor='k')
ax.scatter(1.2, rate_means[3], marker="d", s=200, facecolor='k')

# xcoord and y coord has the x and y coord of individual data points 
x_coords = []
y_coords = []
for point_pair in ax.collections:
   for x, y in point_pair.get_offsets():
       x_coords.append(x)
       y_coords.append(y)
       
for i in range(nr_sub_r):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')
    
for i in range(nr_sub_r+25):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')

for i in range(nr_sub, nr_sub+nr_sub_r):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--' )
# add 0 line
ax.axhline(y=0,  color='k', linestyle='--')
fname1 = savepath + '/output/Fig06_A_Rate_mm1Agreement_Overall.pdf'
# plt.savefig(fname1, dpi=900)

#%% Overall MM1 View Plot
# view: dance-test, dance-retest, lscp-test,  lscp-retest
view_means = [0.4037880787387486, 0.42392020718843665, 0.5040671428688499, 0.5421497491984216 ]
ci_low = [0.2772348801175267, 0.30890570944539736, 0.42508246106764114,  0.4599095987239699 ]
ci_upp = [0.5165694440575644,0.5267147786222154, 0.5754334175309381,  0.6151681097064671]

plt.figure(figsize=(8, 6))
sns.set_style("ticks")
ax = sns.stripplot(x="category", y="mm1_corr", hue='session',
                   data=mm1_odf_view,
                   split=True, jitter=True,
                   palette=flatui)
sns.despine()
# ax.legend(loc='lower right')
ax.legend_.remove()
ax.set_xlabel('Category', fontsize=18)
ax.set_ylabel('"Mean-minus-one" Agreement', fontsize=18)
plt.ylim((-0.6, 1))

ax.plot([-0.204, -0.204], [ci_low[0], ci_upp[0]], 'k-', lw=2)
ax.plot([0.200, 0.200], [ci_low[1], ci_upp[1]], 'k-', lw=2)
ax.plot([0.801, 0.801], [ci_low[2], ci_upp[2]], 'k-', lw=2)
ax.plot([1.2, 1.2], [ci_low[3], ci_upp[3]], 'k-', lw=2)

ax.scatter(-0.204,view_means[0], marker="d", s=200, facecolor='k')
ax.scatter(0.200,view_means[1], marker="d", s=200, facecolor='k')
ax.scatter(0.801,view_means[2], marker="d", s=200, facecolor='k')
ax.scatter(1.2,view_means[3], marker="d", s=200, facecolor='k')

x_coords = []
y_coords = []
for point_pair in ax.collections:
   for x, y in point_pair.get_offsets():
       x_coords.append(x)
       y_coords.append(y)

for i in range(nr_sub_v):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')
    
for i in range(nr_sub_v + 25):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')

for i in range(nr_sub, nr_sub+nr_sub_v):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--' )
    
# add 0 line
ax.axhline(y=0,  color='k', linestyle='--')
fname2 = savepath + 'output/Fig06_A_View_mm1Agreement_Overall.pdf'
# plt.savefig(fname2, dpi=900)