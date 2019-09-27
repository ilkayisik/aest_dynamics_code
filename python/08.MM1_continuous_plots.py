# -*- coding: utf-8 -*-
"""
MM1 calculations  with overall data
saves: mm1.continuous.csv
mm1_continous_submeans.csv
mm1_continous_movmeans.csv
Fig06_B_Rate_mm1Agreement_Continous.pdf
Fig06_B_View_mm1Agreement_Continous.pdf
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
# print (data[0].files)
# %% Load the data
rct = data[0]['cData_ses1']  # rate continuous test
rcrt = data[0]['cData_ses2']  # rate continuous retest
timeC_test = data[0]['cTime_ses1']
cTime = timeC_test[0, 0, :]
vcrt = data[1]['cData_ses2']  # view continuous retest
# %% Experiment parameters
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
# %% Continuous DATA MM1 Calculations
# use corr or lad based calculation
my_func = Stats.compute_mm1
rate_cont_mm1_test, rate_cont_mm1_retest = [], []
view_cont_mm1_retest = []
rate_cont_mm1_test_z, rate_cont_mm1_retest_z = [], []
view_cont_mm1_retest_z = []
# Rate Continuous Test
cd_mean_dict, cd_ci_dict = dict(), dict() # cont data mean for movies 
mm1_df_rct = pd.DataFrame()
for i in range(nr_cats): # get the category data
    temp_dat = rct[:, m_count[i] : m_count[i+1], :] # 25, 15, 300
    for m in range(nr_movies_per_run[i]):
        mm1_rct, corr_rct, mean_rct, ci_rct, ttest_t_rct, ttest_p_rct, cohD_rct = \
        my_func(temp_dat[:, m, :].T)
        corr_ztrans = Stats.rtoz(corr_rct)
        movie = cat_lol[i][m]
        cd_mean_dict[movie + '-rate-test'] = mean_rct
        cd_ci_dict[movie + '-rate-test'] = [ci_rct[0], ci_rct[1]]
        temp_df_rct = pd.DataFrame({'mm1_corr': corr_rct,
                                    'ztrans_mm1': corr_ztrans,
                                    'movie': cat_lol[i][m],
                                    'category': [cats[i]] * int(nr_sub_r),
                                    'session': ['test'] * int(nr_sub_r),
                                    'sub': range(1, nr_sub_r + 1)})
        mm1_df_rct = mm1_df_rct.append(temp_df_rct)
        rate_cont_mm1_test.append(corr_rct)
        rate_cont_mm1_test_z.append(corr_ztrans)
rate_cont_mm1_test = np.asarray(rate_cont_mm1_test).T
rate_cont_mm1_test_z = np.asarray(rate_cont_mm1_test_z).T

# Rate Continuous Retest
mm1_df_rcrt = pd.DataFrame()
for i in range(nr_cats): # get the category data
    temp_dat = rcrt[:, m_count[i] : m_count[i+1], :]
    for m in range(nr_movies_per_run[i]):
        mm1_rcrt, corr_rcrt, mean_rcrt, ci_rcrt, ttest_t_rcrt, ttest_p_rcrt, cohD_rcrt = \
        my_func(temp_dat[:, m, :].T)
        corr_ztrans = Stats.rtoz(corr_rcrt)
        movie = cat_lol[i][m]
        cd_mean_dict[movie + '-rate-retest'] = mean_rcrt
        cd_ci_dict[movie + '-rate-retest'] = [ci_rcrt[0], ci_rcrt[1]]
        temp_df_rcrt = pd.DataFrame({'mm1_corr': corr_rcrt,
                                    'ztrans_mm1': corr_ztrans,
                                    'movie': cat_lol[i][m],
                                    'category': [cats[i]] * int(nr_sub_r),
                                    'session': ['retest'] * int(nr_sub_r),
                                    'sub': range(1, nr_sub_r + 1)})
        mm1_df_rcrt = mm1_df_rcrt.append(temp_df_rcrt)
        rate_cont_mm1_retest.append(corr_rcrt)
        rate_cont_mm1_retest_z.append(corr_ztrans)
rate_cont_mm1_retest = np.asarray(rate_cont_mm1_retest).T
rate_cont_mm1_retest_z = np.asarray(rate_cont_mm1_retest_z).T

# View Continuous Retest
mm1_df_vcrt = pd.DataFrame()
for i in range(nr_cats): # get the category data
    temp_dat = vcrt[:, m_count[i] : m_count[i+1], :]
    for m in range(nr_movies_per_run[i]):
        mm1_vcrt, corr_vcrt, mean_vcrt, ci_vcrt, ttest_t_vcrt, ttest_p_vcrt, cohD_vcrt = \
        my_func(temp_dat[:, m, :].T)
        corr_ztrans = Stats.rtoz(corr_vcrt)
        movie = cat_lol[i][m]
        cd_mean_dict[movie + '-view-retest'] = mean_vcrt
        cd_ci_dict[movie + '-view-retest'] = [ci_vcrt[0], ci_vcrt[1]]
        temp_df_vcrt = pd.DataFrame({'mm1_corr': corr_vcrt,
                                     'ztrans_mm1': corr_ztrans,
                                     'movie': cat_lol[i][m],
                                     'category': [cats[i]] * int(nr_sub_r),
                                     'session': ['retest'] * int(nr_sub_r),
                                     'sub': range(nr_sub_r + 1, nr_sub + 1)})
        mm1_df_vcrt = mm1_df_vcrt.append(temp_df_vcrt)
        view_cont_mm1_retest.append(corr_vcrt)
        view_cont_mm1_retest_z.append(corr_ztrans)
view_cont_mm1_retest = np.asarray(view_cont_mm1_retest).T
view_cont_mm1_retest_z = np.asarray(view_cont_mm1_retest_z).T
# %% combine data from sessions and groups
frames_c_rate = [mm1_df_rct, mm1_df_rcrt]
mm1_cdf_rate = pd.concat(frames_c_rate)
mm1_cdf_rate['group'] = 'Rate'

mm1_cdf_view = mm1_df_vcrt
mm1_cdf_view['group'] = 'View'
frames_c = [mm1_cdf_rate, mm1_cdf_view]
mm1_cdf = pd.concat(frames_c)
save_mm1_cont = savepath + 'data/mm1_continuous.csv'
mm1_cdf.to_csv(save_mm1_cont, index=None)
# %% get the mean values calculated for each subject
# Rate-test
mean_rct_sub, ci_rct_sub = [], []
temp_df_rct_sub = pd.DataFrame()
df_rct_sub_mean = pd.DataFrame()

for i in range(nr_cats):
    cat_data = rate_cont_mm1_test[:, m_count[i] : m_count[i+1]]
    for s in range(nr_sub_r):
        corrs = cat_data[s, :]
        sub_mean = Stats.ztor(np.nanmean(Stats.rtoz(corrs)))
        sub_sd = Stats.ztor(np.nanstd(Stats.rtoz(corrs)))
        sub_ci = np.zeros(2)
        sub_ci[0] = Stats.ztor(Stats.rtoz(sub_mean) - 1.96 * (sub_sd) / np.sqrt(nr_movies/2))
        sub_ci[1] = Stats.ztor(Stats.rtoz(sub_mean) + 1.96 * (sub_sd) / np.sqrt(nr_movies/2))
        mean_rct_sub.append(sub_mean)
        ci_rct_sub.append([sub_ci[0], sub_ci[1]])
        temp_df_rct_sub = pd.DataFrame({'category': [cats[i]],
                                        'session': ['Test'],
                                        'sub': str(s + 1),
                                        'mean_mm1': sub_mean,
                                        'std':sub_sd,
                                        'ci_upp': sub_ci[1],
                                        'ci_low': sub_ci[0]})
        df_rct_sub_mean = df_rct_sub_mean.append(temp_df_rct_sub)

# Rate-retest   
mean_rcrt_sub, ci_rcrt_sub = [], []
temp_df_rcrt_sub = pd.DataFrame()
df_rcrt_sub_mean = pd.DataFrame()
for i in range(nr_cats):
    cat_data = rate_cont_mm1_retest[:, m_count[i] : m_count[i+1]]
    for s in range(nr_sub_r):
        corrs = cat_data[s, :]
        sub_mean = Stats.ztor(np.nanmean(Stats.rtoz(corrs)))
        sub_sd = Stats.ztor(np.nanstd(Stats.rtoz(corrs)))
        sub_ci = np.zeros(2)
        sub_ci[0] = Stats.ztor(Stats.rtoz(sub_mean) - 1.96 * (sub_sd) / np.sqrt(nr_movies/2))
        sub_ci[1] = Stats.ztor(Stats.rtoz(sub_mean) + 1.96 * (sub_sd) / np.sqrt(nr_movies/2))
        mean_rcrt_sub.append(sub_mean)
        ci_rcrt_sub.append([sub_ci[0], sub_ci[1]])
        temp_df_rcrt_sub = pd.DataFrame({'category': [cats[i]],
                                         'session': ['Retest'],
                                         'sub': str(s + 1),
                                         'mean_mm1':  sub_mean,
                                         'std': sub_sd,
                                         'ci_upp': sub_ci[1],
                                         'ci_low': sub_ci[0]})
        
        df_rcrt_sub_mean = df_rcrt_sub_mean.append(temp_df_rcrt_sub)
# combine dataframes
frames_c_rate_mean = [df_rct_sub_mean, df_rcrt_sub_mean]
submean_mm1_cdf_rate = pd.concat(frames_c_rate_mean)
submean_mm1_cdf_rate['group'] = 'Rate'   

# View Retest     
mean_vcrt_sub, ci_vcrt_sub = [], []
temp_df_vcrt_sub = pd.DataFrame()
df_vcrt_sub_mean = pd.DataFrame()

for i in range(nr_cats):
    cat_data = view_cont_mm1_retest[:, m_count[i] : m_count[i+1]]
    for s in range(nr_sub_v):
        corrs = cat_data[s, :]
        sub_mean = Stats.ztor(np.nanmean(Stats.rtoz(corrs)))
        sub_sd = Stats.ztor(np.nanstd(Stats.rtoz(corrs)))
        sub_ci = np.zeros(2)
        sub_ci[0] = Stats.ztor(Stats.rtoz(sub_mean) - 1.96 * (sub_sd) / np.sqrt(nr_movies/2))
        sub_ci[1] = Stats.ztor(Stats.rtoz(sub_mean) + 1.96 * (sub_sd) / np.sqrt(nr_movies/2))
        mean_vcrt_sub.append(sub_mean)
        ci_vcrt_sub.append([sub_ci[0], sub_ci[1]])
        temp_df_vcrt_sub = pd.DataFrame({'category': [cats[i]],
                                         'session': ['Retest'],
                                         'sub': str(s + 26),
                                         'mean_mm1': sub_mean,
                                         'std': sub_sd,
                                         'ci_upp': sub_ci[1],
                                         'ci_low': sub_ci[0],
                                         'group':'View'})
        df_vcrt_sub_mean = df_vcrt_sub_mean.append(temp_df_vcrt_sub)
        
submean_mm1_cdf_view = df_vcrt_sub_mean

# save it out
frames = [submean_mm1_cdf_rate, submean_mm1_cdf_view]
mm1_sub_means = pd.concat(frames)
save_mm1_sub_means = savepath + 'data/mm1_continuous_submeans.csv'
mm1_sub_means.to_csv(save_mm1_sub_means, index=None)
# %% get the mean values calculated for each MOVIE
# Rate test
mean_rct_mov, ci_rct_mov = [], []
temp_df_rct_mov = pd.DataFrame()
df_rct_mov_mean = pd.DataFrame()

k=0
for i in range(nr_cats):
    cat_data = rate_cont_mm1_test[:, m_count[i] : m_count[i+1]]
    for m in range(15):
        corrs = cat_data[:, m]
        mov_mean = Stats.ztor(np.nanmean(Stats.rtoz(corrs)))
        mov_sd = Stats.ztor(np.nanstd(Stats.rtoz(corrs)))
        mov_ci = np.zeros(2)
        mov_ci[0] = Stats.ztor(Stats.rtoz(mov_mean) - 1.96 * (mov_sd) / np.sqrt(nr_sub_r))
        mov_ci[1] = Stats.ztor(Stats.rtoz(mov_mean) + 1.96 * (mov_sd) / np.sqrt(nr_sub_r))
        mean_rct_mov.append(mov_mean)
        ci_rct_sub.append([mov_ci[0], mov_ci[1]])
        temp_df_rct_mov = pd.DataFrame({'category': [cats[i]],
                                        'session': ['Test'],
                                        'mov': moi_list[k],
                                        'mean_mm1': mov_mean,
                                        'std':mov_sd,
                                        'ci_upp': mov_ci[1],
                                        'ci_low': mov_ci[0]})
        df_rct_mov_mean = df_rct_mov_mean.append(temp_df_rct_mov)
        k+=1

# Rate retest   
mean_rcrt_mov, ci_rcrt_mov = [], []
temp_df_rcrt_mov = pd.DataFrame()
df_rcrt_mov_mean = pd.DataFrame()

k=0
for i in range(nr_cats):
    cat_data = rate_cont_mm1_retest[:, m_count[i] : m_count[i+1]]
    for m in range(15):
        corrs = cat_data[:, m]
        mov_mean = Stats.ztor(np.nanmean(Stats.rtoz(corrs)))
        mov_sd = Stats.ztor(np.nanstd(Stats.rtoz(corrs)))
        mov_ci = np.zeros(2)
        mov_ci[0] = Stats.ztor(Stats.rtoz(mov_mean) - 1.96 * (mov_sd) / np.sqrt(nr_sub_r))
        mov_ci[1] = Stats.ztor(Stats.rtoz(mov_mean) + 1.96 * (mov_sd) / np.sqrt(nr_sub_r))
        mean_rcrt_mov.append(mov_mean)
        ci_rcrt_sub.append([mov_ci[0], mov_ci[1]])
        temp_df_rcrt_mov = pd.DataFrame({'category': [cats[i]],
                                        'session': ['Retest'],
                                        'mov': moi_list[k],
                                        'mean_mm1':  mov_mean,
                                        'std':mov_sd,
                                        'ci_upp': mov_ci[1],
                                        'ci_low': mov_ci[0]})
        df_rcrt_mov_mean = df_rcrt_mov_mean.append(temp_df_rcrt_mov)
        k+=1
  
# combine dataframes
frames_c_rate_mean = [df_rct_mov_mean, df_rcrt_mov_mean]
movmean_mm1_cdf_rate = pd.concat(frames_c_rate_mean)
movmean_mm1_cdf_rate['group'] = 'Rate'   

# View Retest     
mean_vcrt_mov, ci_vcrt_mov = [], []
temp_df_vcrt_mov = pd.DataFrame()
df_vcrt_mov_mean = pd.DataFrame()
k=0
for i in range(nr_cats):
    cat_data = view_cont_mm1_retest[:, m_count[i] : m_count[i+1]]
    for m in range(15):
        corrs = cat_data[:, m]
        mov_mean = Stats.ztor(np.nanmean(Stats.rtoz(corrs)))
        mov_sd = Stats.ztor(np.nanstd(Stats.rtoz(corrs)))
        mov_ci = np.zeros(2)
        mov_ci[0] = Stats.ztor(Stats.rtoz(mov_mean) - 1.96 * (mov_sd) / np.sqrt(nr_sub_r))
        mov_ci[1] = Stats.ztor(Stats.rtoz(mov_mean) + 1.96 * (mov_sd) / np.sqrt(nr_sub_r))
        mean_vcrt_mov.append(mov_mean)
        ci_vcrt_sub.append([mov_ci[0], mov_ci[1]])
        temp_df_vcrt_mov = pd.DataFrame({'category': [cats[i]],
                                        'session': ['Retest'],
                                        'mov': moi_list[k],
                                        'mean_mm1':  mov_mean,
                                        'std': mov_sd,
                                        'ci_upp': mov_ci[1],
                                        'ci_low': mov_ci[0],
                                        'group':'View'})
        df_vcrt_mov_mean = df_vcrt_mov_mean.append(temp_df_vcrt_mov)
        k+=1

movmean_mm1_cdf_view = df_vcrt_mov_mean


# save it out
frames = [movmean_mm1_cdf_rate, movmean_mm1_cdf_view ]
mm1_mov_means = pd.concat(frames)
save_mm1_mov_means = savepath + 'data/mm1_continous_movmeans.csv'
mm1_mov_means.to_csv(save_mm1_mov_means, index=None)
# %% Calculate means and CI's to use in the plots: SUBJECT BASED
# Rate: dance-test, lscp-test, dance-retest, lscp-retest
rate_sub_means = []
rate_sub_means.append(Stats.ztor(np.nanmean(Stats.rtoz(np.asarray(mean_rct_sub[:25]))))) # dance-test
rate_sub_means.append(Stats.ztor(np.nanmean(Stats.rtoz(np.asarray(mean_rcrt_sub[:25]))))) # dance-retest
rate_sub_means.append(Stats.ztor(np.nanmean(Stats.rtoz(np.asarray(mean_rct_sub[25:]))))) # lscp-test
rate_sub_means.append(Stats.ztor(np.nanmean(Stats.rtoz(np.asarray(mean_rcrt_sub[25:]))))) # lscp-retest

rate_sub_sd = []
rate_sub_sd.append(Stats.ztor(np.nanstd(Stats.rtoz(np.asarray(mean_rct_sub[:25]))))) #dance-test
rate_sub_sd.append(Stats.ztor(np.nanstd(Stats.rtoz(np.asarray(mean_rcrt_sub[:25]))))) #dance-retest
rate_sub_sd.append(Stats.ztor(np.nanstd(Stats.rtoz(np.asarray(mean_rct_sub[25:]))))) #lscp-test
rate_sub_sd.append(Stats.ztor(np.nanstd(Stats.rtoz(np.asarray(mean_rcrt_sub[25:]))))) #lscp-retest

rate_ci_lo, rate_ci_up = [], [] 
for i in range(4):
    rate_ci_lo.append(Stats.ztor(Stats.rtoz(rate_sub_means[i]) - 1.96 * (rate_sub_sd[i]) / np.sqrt(nr_sub_r)))
    rate_ci_up.append(Stats.ztor(Stats.rtoz(rate_sub_means[i]) + 1.96 * (rate_sub_sd[i]) / np.sqrt(nr_sub_r)))

# VIEW:  dance-retest, lscp-retest
view_sub_means = []
view_sub_means.append(Stats.ztor(np.nanmean(Stats.rtoz(np.asarray(mean_vcrt_sub[:25]))))) #dance-test
view_sub_means.append(Stats.ztor(np.nanmean(Stats.rtoz(np.asarray(mean_vcrt_sub[25:]))))) #lscp-test

view_sub_sd = []
view_sub_sd.append(Stats.ztor(np.nanstd(Stats.rtoz(np.asarray(mean_vcrt_sub[:25]))))) #dance-retest
view_sub_sd.append(Stats.ztor(np.nanstd(Stats.rtoz(np.asarray(mean_vcrt_sub[25:]))))) #lscp-retest

view_ci_lo, view_ci_up = [], [] 
for i in range(2):
    view_ci_lo.append(Stats.ztor(Stats.rtoz(view_sub_means[i]) - 1.96 * (view_sub_sd[i]) / np.sqrt(nr_sub_v)))
    view_ci_up.append(Stats.ztor(Stats.rtoz(view_sub_means[i]) + 1.96 * (view_sub_sd[i]) / np.sqrt(nr_sub_v)))
# %% PLOTS for MM1 correlations with CONTINUOUS DATA: Subject Based 
# Rate Plot
flatui = ["#F8766D", "#00BFC4"]
sns.set_palette(flatui)
# sns.palplot(sns.color_palette())
sns.set_style('white')
sns.set_context('paper',font_scale=2 )
sns.set_style("ticks")
plt.figure(figsize=(8, 6))
ax = sns.stripplot(x="category", y="mean_mm1", hue='session',
                   data=submean_mm1_cdf_rate,
                   dodge=True, jitter=True,
                   edgecolor='black',
                   palette=flatui)
sns.despine()

# ax.legend(loc='lower right')
ax.legend_.remove()
ax.set_xlabel('Category')
ax.set_ylabel('"Mean-minus-one" Agreement')
plt.ylim((-0.6, 1))
# Set y ticks ?
# plt.yticks(np.linspace(-0.6, 1, 5, endpoint=True))

ax.plot([-0.204, -0.204], [rate_ci_lo[0], rate_ci_up[0]], 'k-', lw=2) 
ax.plot([0.200, 0.200], [rate_ci_lo[1], rate_ci_up[1]], 'k-', lw=2)
ax.plot([0.801, 0.801], [rate_ci_lo[2], rate_ci_up[2]], 'k-', lw=2)
ax.plot([1.2, 1.2], [rate_ci_lo[3], rate_ci_up[3]], 'k-', lw=2)

ax.scatter(-0.204, rate_sub_means[0], marker="d", s=200, facecolor='k')
ax.scatter(0.200, rate_sub_means[1], marker="d", s=200, facecolor='k')
ax.scatter(0.801, rate_sub_means[2], marker="d", s=200, facecolor='k')
ax.scatter(1.2, rate_sub_means[3], marker="d", s=200, facecolor='k')


x_coords = []
y_coords = []
for point_pair in ax.collections:
   for x, y in point_pair.get_offsets():
       x_coords.append(x)
       y_coords.append(y)
# Line between data points    
for i in range(nr_sub_r):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')
    
for i in range(nr_sub_r + 25):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')

for i in range(nr_sub, nr_sub+nr_sub_r):
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--',)
# add 0 line
ax.axhline(y=0,  color='k', linestyle='--')

fname1 = savepath + 'output/figures/Fig06_B_Rate_mm1Agreement_Continous.pdf'
plt.savefig(fname1)
plt.close()
# %%View Plot
sns.set_style('white')
sns.set_context('paper',font_scale=2 )
# VIEW:  dance-retest, lscp-retest
rate_sub_means = []
ci_sub_lower =[]
ci_sub_upper = []
plt.figure(figsize=(8, 6))
sns.set_style("ticks")
ax = sns.stripplot(x="category", y="mean_mm1", hue='session',
                   data=submean_mm1_cdf_view,
                   split=True, jitter=True,
                   edgecolor='black',
                   palette=flatui)

sns.despine()
# plt.legend(loc='upper left')
ax.legend_.remove()
ax.set_xlabel('Category')
ax.set_ylabel('"Mean-minus-one" Agreement')
plt.ylim((-0.6, 1))

ax.plot([0.005, 0.005], [view_ci_lo[0], view_ci_up[0]], 'k-', lw=2)
ax.plot([1.017, 1.017], [view_ci_lo[1], view_ci_up[1]], 'k-', lw=2)

ax.scatter(0.005, view_sub_means[0], marker="d", s=200, facecolor='k')
ax.scatter(1.017, view_sub_means[1], marker="d", s=200, facecolor='k')

x_coords = []
y_coords = []
for point_pair in ax.collections:
   for x, y in point_pair.get_offsets():
       x_coords.append(x)
       y_coords.append(y)
       
for i in range(nr_sub_v):
  
    ax.plot([x_coords[i], x_coords[i+25]], [y_coords[i], y_coords[i+25]], 
             color='gray', lw=0.5, linestyle='--')
    i+=1
# add 0 line
ax.axhline(y=0,  color='k', linestyle='--')

fname2 = savepath + 'output/figures/Fig06_B_View_mm1Agreement_Continous.pdf'
plt.savefig(fname2, dpi=900)
plt.close()
