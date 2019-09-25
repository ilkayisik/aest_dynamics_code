#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:28:56 2019
@author: ilkay.isik
Reliability across test-retest sessions
Calculate reliability using person corr and lad for 
for continuous ratings
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon
import seaborn as sns
import pandas as pd
import sys
sys.path.append('/Users/ilkay.isik/aesthetic_dynamics/aest_dynamics_code/python/functions')
from useful_functions import norm_data
from distance_functions import l2_norm
from stats import rtoz, ztor
# %% load the data
f_dirs = ['/Users/ilkay.isik/aesthetic_dynamics/data/data_rate.npz',
          '/Users/ilkay.isik/aesthetic_dynamics/data/data_view.npz']
savepath = '/Users/ilkay.isik/aesthetic_dynamics/output/'

data1, data2 = np.load(f_dirs[0]), np.load(f_dirs[1])
# Continuous data test and retest
cdt, cdrt= data1['cData_ses1'], data1['cData_ses2']

# Rate Overall Test
rot = data1['oData_ses1']
rot_d, rot_l = rot[:,0:15], rot[:, 15:]
rort = data1['oData_ses2']
# Rate Overall Retest
rort_d, rort_l = rort[:,0:15], rort[:, 15:]
# View Overall Test
vot = data2['oData_ses1']
vot_d, vot_l = vot[:,0:15], vot[:, 15:]
# View Overall Retest
vort = data2['oData_ses2']
vort_d, vort_l = vort[:,0:15], vort[:, 15:]

# Other experiment parameters
nr_movies = data1['nr_movies']
nr_subjects = data1['nr_subjects']
#nr_tp = len(cTime)
nr_runs = data1['nr_runs']

moi_list = data1['moi_list']
moi_list = [mov.decode("utf-8") for mov in moi_list]
# %% Continuous Rating Reliability with Pearson correlation
pr_r_vals = np.zeros([nr_subjects, nr_movies]) # r values
pr_p_vals = np.zeros([nr_subjects, nr_movies]) # p values
for m in range(nr_movies):  # loop through movies
    for s in range(nr_subjects):  # loop through subjects
        tc = cdt[s, m, :]  # test 
        rtc = cdrt[s, m, :]  # retest
        pr_r_vals[s, m] = pearsonr(tc, rtc)[0]
        pr_p_vals[s, m] = pearsonr(tc, rtc)[1]

# Calculate subject and movie based median and mean r values
pr_meds, pr_means = np.ones([25, 2]), np.ones([25, 2])
# median and mean r per participant for dance
pr_meds[:, 0] = ztor(np.nanmedian(rtoz(pr_r_vals[:, 0:15]), axis=1))
pr_means[:, 0] = ztor(np.nanmean(rtoz(pr_r_vals[:, 0:15]), axis=1))
# median and mean r per participant for lsp
pr_meds[:, 1] = ztor(np.nanmedian(rtoz(pr_r_vals[:, 15:]), axis=1))
pr_means[:, 1] = ztor(np.nanmean(rtoz(pr_r_vals[:, 15:]), axis=1))
# Wilcoxon test across categories
wilcoxon(pr_means[:, 0], pr_means[:, 1])
wilcoxon(pr_meds[:, 0], pr_meds[:, 1])
# mean of the medians
pr_dnc_medmean = ztor(np.nanmean(rtoz(pr_meds[:, 0])))
pr_lsp_medmean = ztor(np.nanmean(rtoz(pr_meds[:, 1])))
# sd of the medians
pr_dnc_med_sd = np.nanstd(rtoz(pr_meds[:, 0]))
pr_lsp_med_sd = np.nanstd(rtoz(pr_meds[:, 1]))
# mean of the means
pr_dnc_meanmean = ztor(np.nanmean(rtoz(pr_means[:, 0])))
pr_lsp_meanmean = ztor(np.nanmean(rtoz(pr_means[:, 1])))
# sd of the means 
pr_dnc_mean_sd = np.nanstd(rtoz(pr_means[:, 0]))
pr_lsp_mean_sd = np.nanstd(rtoz(pr_means[:, 1]))

# conf interval of the medians
dnc_med_ci, lsp_med_ci = np.zeros(2), np.zeros(2)
dnc_med_ci[0] = ztor(rtoz(pr_dnc_medmean) - 1.96 * (pr_dnc_med_sd) / np.sqrt(nr_subjects))
dnc_med_ci[1] = ztor(rtoz(pr_dnc_medmean) + 1.96 * (pr_dnc_med_sd) / np.sqrt(nr_subjects))
lsp_med_ci[0] = ztor(rtoz(pr_lsp_medmean) - 1.96 * (pr_lsp_med_sd) / np.sqrt(nr_subjects))
lsp_med_ci[1] = ztor(rtoz(pr_lsp_medmean) + 1.96 * (pr_lsp_med_sd) / np.sqrt(nr_subjects))

# confidence interval of the means
dnc_mean_ci, lsp_mean_ci = np.zeros(2), np.zeros(2)
dnc_mean_ci[0] = ztor(rtoz(pr_dnc_meanmean) - 1.96 * (pr_dnc_mean_sd) / np.sqrt(nr_subjects))
dnc_mean_ci[1] = ztor(rtoz(pr_dnc_meanmean) + 1.96 * (pr_dnc_mean_sd) / np.sqrt(nr_subjects))
lsp_mean_ci[0] = ztor(rtoz(pr_lsp_meanmean) - 1.96 * (pr_lsp_mean_sd) / np.sqrt(nr_subjects))
lsp_mean_ci[1] = ztor(rtoz(pr_lsp_meanmean) + 1.96 * (pr_lsp_mean_sd) / np.sqrt(nr_subjects))

# %% Continuous Rating Reliability with L2 norm
l2_vals = np.zeros([nr_subjects, nr_movies])
for m in range(nr_movies):  # loop through movies
    for s in range(nr_subjects):  # loop through subjects
        # curves of interest
        tc = cdt[s, m, :]  # test curve
        rtc = cdrt[s, m, :]  # retestcurve
        l2_vals[s, m] = l2_norm(tc, rtc)

# normalization of l2 norm values with absolute min and max
t, rt = np.ones(300), np.ones(300)
rt[rt ==1] = -1
Max, Min = l2_norm(t, rt), l2_norm(t, t)
# l2_vals_norm = norm_data_arr(l2_vals, 1, -1) * -1
# use the theoretical max and min to normalize
l2_vals_norm = norm_data(l2_vals, 1, -1, Max, Min) * -1

# Calculate subject and movie based median and mean r values
l2_meds, l2_means = np.ones([25, 2]), np.ones([25, 2])

# median r per participant for dance
l2_meds[:, 0] = np.nanmedian(l2_vals_norm[:, 0:15], axis=1) 
l2_means[:, 0] = np.nanmean(l2_vals_norm[:, 0:15], axis=1) 

# median r per participant for lsp
l2_meds[:, 1] = np.nanmedian(l2_vals_norm[:, 15:], axis=1)
l2_means[:, 1] = np.nanmean(l2_vals_norm[:, 15:], axis=1)

# wilcoxon test 
wilcoxon(l2_means[:, 0], l2_means[:, 1])
wilcoxon(l2_meds[:, 0], l2_meds[:, 1])

# mean of the medians
l2_dnc_medmean = np.mean(l2_meds[:, 0]) 
l2_lsp_medmean  = np.mean(l2_meds[:, 1])

# mean of the means
l2_dnc_meanmean = np.mean(l2_means[:, 0]) 
l2_lsp_meanmean = np.mean(l2_means[:, 1])

# sd of the medians
l2_dnc_med_sd = np.nanstd(l2_meds[:, 0])
l2_lsp_med_sd = np.nanstd(l2_meds[:, 1])

# sd of the means
l2_dnc_mean_sd = np.nanstd(l2_means[:, 0])
l2_lsp_mean_sd = np.nanstd(l2_means[:, 1])

# CI of the means
l2_dnc_mean_ci, l2_lsp_mean_ci = np.zeros(2), np.zeros(2)
l2_dnc_mean_ci[0] = l2_dnc_meanmean - 1.96 * (l2_dnc_mean_sd) / np.sqrt(nr_subjects)
l2_dnc_mean_ci[1] = l2_dnc_meanmean + 1.96 * (l2_dnc_mean_sd) / np.sqrt(nr_subjects)
l2_lsp_mean_ci[0] = l2_lsp_meanmean - 1.96 * (l2_lsp_mean_sd) / np.sqrt(nr_subjects)
l2_lsp_mean_ci[1] = l2_lsp_meanmean + 1.96 * (l2_lsp_mean_sd) / np.sqrt(nr_subjects)

# %% Overall rel with l2 norm
l2_vals_ov = np.zeros([nr_subjects, 4]) 
for s in range(nr_subjects):
    l2_vals_ov[s, 0] = l2_norm(rot_d[s, :], rort_d[s, :])

for s in range(nr_subjects):
    l2_vals_ov[s, 1] =  l2_norm(rot_l[s, :], rort_l[s, :])

for s in range(nr_subjects):
    l2_vals_ov[s, 2] =  l2_norm(vot_d[s, :], vort_d[s, :])

for s in range(nr_subjects):
    l2_vals_ov[s, 3] = l2_norm(vot_l[s, :], vort_l[s, :])
# Use the theoretical max and min to normalize
t, rt = np.ones(15), np.ones(15)
rt[rt ==1] = -1
Max, Min = l2_norm(t, rt), l2_norm(t, t)
l2_ov_norm = norm_data(l2_vals_ov, 1, -1, Max, Min) * -1

# means
l2_ov_means = np.nanmean(l2_ov_norm, axis=0)

# sds of the means
l2_ov_sds = np.nanstd(l2_ov_norm, axis=0)

# CI of the means
l2_ov_ci_minus = l2_ov_means - 1.96 * (l2_ov_sds) / np.sqrt(nr_subjects)

l2_ov_ci_plus = l2_ov_means + 1.96 * (l2_ov_sds) / np.sqrt(nr_subjects)

wilcoxon(l2_ov_norm[:, 0], l2_ov_norm[:, 1])
wilcoxon(l2_ov_norm[:, 2], l2_ov_norm[:, 3])

# For dance
corr1 = pearsonr(l2_ov_norm[:, 0], l2_meds[:,0])[0]
# For lscp
corr2 = pearsonr(l2_ov_norm[:, 1], l2_meds[:,1])[0]
# %% Reliability PLOT: Figure 4
prvalstp, l2valstp = pr_means, l2_means # tp: to plot
flatui = ["#F8766D", "#00BFC4"]
sns.set_palette(flatui)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.1)
grid = plt.GridSpec(1, 3, wspace=0.3, hspace=0.7)
plt.figure(figsize=(10, 4.5))
plt.subplots_adjust(left=0.10, right=0.95, top=0.9, bottom=0.1)

# Col 01: Continuous rating with Pearson
plt.subplot(grid[0:1, 0])
plt.hist([prvalstp[:, 0], prvalstp[:, 1]], bins=15,
         label=['dance', 'lscp'])
plt.xlim([0, 1]), plt.ylim([0, 8])
plt.text(0.01, 7, "Dance mean: {:.2f}".format(pr_dnc_meanmean), fontsize=11)
plt.text(0.01, 6, "Lscp mean: {:.2f}".format(pr_lsp_meanmean), fontsize=11)
plt.xlabel(r"Mean Pearson's $\it{r}$")
plt.text(-0.2, 8, "(A)", fontsize=12, 
        color='black', fontname='arial', fontweight='bold')
plt.text(-0.15, 4, 'Number of Participants', va='center', rotation='vertical')

# Col 02 : Continuous rating with L2
plt.subplot(grid[0:1, 1])
plt.hist([l2valstp[:, 0], l2valstp[:, 1]], bins=15,
         label=['dance', 'lscp'])
plt.xlim([0, 1]), plt.ylim([0, 8])
plt.text(0.01, 7, "Dance mean: {:.2f}".format(l2_dnc_meanmean), fontsize=11)
plt.text(0.01, 6, "Lscp mean: {:.2f}".format(l2_lsp_meanmean), fontsize=11)
plt.xlabel("Mean $L_{2}$-norm values")
plt.text(-0.2, 8, "(B)", fontsize=12, 
        color='black', fontname='arial', fontweight='bold')

# Col 03  Comparison between overall and continuous ratings
plt.subplot(grid[0:1, 2])
plt.scatter(l2_ov_norm[:, 0], l2_meds[:,0], label='Dance', alpha=0.5)
plt.scatter(l2_ov_norm[:, 1], l2_meds[:,1], label='Landscape', alpha=0.5)
plt.xlabel('$L_{2}$-norm (Overall Reliability)')
plt.xlim([0, 1.1]), plt.ylim([0, 1.1])
plt.plot([0, 1], [0, 1], color='dimgray', ls='dashed')

plt.text(-0.4, 0.53, "   Median $L_{2}-norm$" + '\n(Continuous Reliability)',
         va='center', rotation='vertical')
plt.text(0.01, 0.9, "Dance r : {:.2f}".format(corr1), fontsize=11)
plt.text(0.01, 0.8, "Lscp r: {:.2f}".format(corr2), fontsize=11)
plt.text(-0.2, 1.1, "(D)", fontsize=12, 
        color='black', fontname='arial', fontweight='bold')
plt.legend(loc=4)
sns.despine()

fname = savepath + 'Fig04_Reliability_Pearson_and_L2_histmeans' + '.pdf'
# plt.savefig(fname, dpi=300)
# %%  SUPPLEMENTARY FIGURE: BOX PLOTS 
# Boxplots per person: For Dance and Landscape Continuous reliability values
# To do it with seaborn: Create a data frame with the lad scores
dnc_vals = pr_r_vals[:, 0:15].T
dnc_vals = dnc_vals.flatten('F')
lscp_vals = pr_r_vals[:, 15:].T
lscp_vals = lscp_vals.flatten('F')
concat = np.zeros([375, 2])
concat[:, 0], concat[:, 1] = dnc_vals, lscp_vals
df = pd.DataFrame(concat, columns=['Dance', 'Landscape'])
sub_list = sum([[i] * 15 for i in range(1,26)], []) * 2

df = df.reset_index()
df = pd.melt(df, id_vars='index', value_vars=['Dance', 'Landscape'])
df['Participants'] = pd.Series(sub_list)
df = df.rename(index=str, columns={"value": "Pearson correlation r",
                                   "variable": "Category"})
del df['index']
dance_movs, lscp_movs, movs = moi_list[0:15], moi_list[15:], []
movs.extend(dance_movs * 25)
movs.extend(lscp_movs * 25)
df['Movie'] = movs

# BOX PLOT with r values
flatui = ["#F8766D", "#00BFC4"]
sns.set_palette(flatui)
sns.set_context("paper", font_scale=1.2)
plt.figure(figsize=(7.08, 2.5))
ax = sns.boxplot(x="Participants", y="Pearson correlation r", 
                 hue="Category", data=df)
for i, box in enumerate(ax.artists):
    if i%2 == 0:
        box.set_edgecolor(flatui[0])

    elif i%2!=0:
        box.set_edgecolor(flatui[1])
    box.set_facecolor('white')
sns.despine()
savename = savepath + 'S1Fig_Part01_Boxplot_ContinousRel_PearsonCorr.pdf'
# plt.savefig(savename, dpi=300)

# SUPPLEMENTARY FIGURE Cont.: BOX PLOT with L2 values
dnc_vals = l2_vals_norm[:, 0:15].T
dnc_vals = dnc_vals.flatten('F')
lscp_vals = l2_vals_norm[:, 15:].T
lscp_vals = lscp_vals.flatten('F')
concat = np.zeros([375, 2])
concat[:, 0], concat[:, 1] = dnc_vals, lscp_vals
df = pd.DataFrame(concat, columns=['Dance', 'Landscape'])
sub_list = sum([[i] * 15 for i in range(1,26)], []) * 2

df = df.reset_index()
df = pd.melt(df, id_vars='index', value_vars=['Dance', 'Landscape'])
df['Participants'] = pd.Series(sub_list)
df = df.rename(index=str, columns={"value": "L2 norm",
                                   "variable": "Category"})
del df['index']
dance_movs, lscp_movs, movs = moi_list[0:15], moi_list[15:], []
movs.extend(dance_movs * 25)
movs.extend(lscp_movs * 25)
df['Movie'] = movs

# BOX PLOT with L2 values 
flatui = ["#F8766D", "#00BFC4"]
sns.set_palette(flatui)
sns.set_context("paper", font_scale=1.2)
plt.figure(figsize=(7.08, 2.5))
ax = sns.boxplot(x="Participants", y="L2 norm", 
                 hue="Category", data=df)
for i, box in enumerate(ax.artists):
    if i%2 == 0:
        box.set_edgecolor(flatui[0])

    elif i%2!=0:
        box.set_edgecolor(flatui[1])
    box.set_facecolor('white')
sns.despine()
savename = savepath + 'S1Fig_Part02_Boxplot_ContinousRel_L2norm.pdf'
# plt.savefig(savename, dpi=300)