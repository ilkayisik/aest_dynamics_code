# -*- coding: utf-8 -*-
"""
Fig02:Plot for overall ratings
@author: ilkay.isik
"""
# Load librarires
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
# %% set paths  and  load the data
filedir = '/Users/ilkay.isik/aesthetic_dynamics/data/df_overall.csv'         
savepath = '/Users/ilkay.isik/aesthetic_dynamics/output/figures/'
# %% create several dataframes for different comparisons
# main dataframe
df = pd.read_csv(filedir, sep=',', )
df.loc[df['category'] == 'D', 'category'] = 'Dance'
df.loc[df['category'] == 'L', 'category'] = 'Landscape'
# for rate and view
df_rate = df.loc[df['group'] == 'Rate']
df_view = df.loc[df['group'] == 'View']

# take means per subject
df1 = df_rate.groupby(['subject', 'category', 'session'], as_index=False)['oData'].mean()
# compute stats
df1_stats = df1.groupby(['category', 'session'])['oData'].agg(['mean', 'count', 'std'])
ci95_lo, ci95_hi  = [], []
for i in df1_stats.index:
    m, c, s = df1_stats.loc[i]
    ci95_lo.append(m - 1.96*s/math.sqrt(c))
    ci95_hi.append(m + 1.96*s/math.sqrt(c))
    
df1_stats['ci95_lo'] , df1_stats['ci95_hi'] = ci95_lo, ci95_hi

df1.loc[(df1['category'] == 'Landscape') & (df1['session'] == 'Retest'), 'session'] = 'retest'
df1.loc[(df1['category'] == 'Landscape') & (df1['session'] == 'Test'), 'session'] = 'test'
df1['session'] = df1['session'].astype('category')
df1['session'].cat.reorder_categories(['Test', 'Retest','test', 'retest'], inplace=True)


df2 = df_view.groupby(['subject', 'category', 'session'], as_index=False)['oData'].mean()
# compute stat
df2_stats = df2.groupby(['category', 'session'])['oData'].agg(['mean', 'count', 'std'])
ci95_lo, ci95_hi  = [], []
for i in df2_stats.index:
    m, c, s = df2_stats.loc[i]
    ci95_lo.append(m - 1.96*s/math.sqrt(c))
    ci95_hi.append(m + 1.96*s/math.sqrt(c))
    
df2_stats['ci95_lo'] , df2_stats['ci95_hi'] = ci95_lo, ci95_hi


df2.loc[(df2['category'] == 'Landscape') & (df2['session'] == 'Retest'), 'session'] = 'retest'
df2.loc[(df2['category'] == 'Landscape') & (df2['session'] == 'Test'), 'session'] = 'test'
df2['session'] = df2['session'].astype('category')
df2['session'].cat.reorder_categories(['Test', 'Retest','test', 'retest'], inplace=True)
# %% PLOT
sns.set_style('white')
sns.set_context('paper', font_scale=1.5)
sns.set_style("ticks")

flatui = ["#F8766D", "#AF3C3C", "#00BFC4", "#036D6D"] # color map
sns.set_palette(flatui)
msize=150
j_val = 0.35
d_val = 0.9

plt.figure(figsize=(10, 5))
grid = plt.GridSpec(1, 5, wspace=0.1, hspace=0.2)
# RATE
plt.subplot(grid[0, 0:2])
ax = sns.stripplot(x="category", y="oData", hue='session',
                   data=df1,
                   jitter=j_val, 
                   split=True,
                   dodge=d_val,
                   edgecolor='black',
                   palette=flatui,
                   size=4
                   )
sns.despine()
ax.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('Mean Overall Rating')
plt.ylim((-1, 1.1))

# xcoord and y coord has the x and y coord of individual data points 
x_coords, y_coords = [], []
for point_pair in ax.collections:
   for x, y in point_pair.get_offsets():
       x_coords.append(x)
       y_coords.append(y)
   
x1 = np.median(x_coords[0:25])
x2 = np.median(x_coords[25:50])
x3 = np.median(x_coords[50:75])
x4 = np.median(x_coords[75:100])

ax.plot([x1, x1], [df1_stats['ci95_lo'][1],df1_stats['ci95_hi'][1]], 'k-', lw=2) # test
ax.plot([x2, x2], [df1_stats['ci95_lo'][0],df1_stats['ci95_hi'][0]], 'k-', lw=2) # retest
ax.plot([x3, x3], [df1_stats['ci95_lo'][3],df1_stats['ci95_hi'][3]], 'k-', lw=2) # test
ax.plot([x4, x4], [df1_stats['ci95_lo'][2],df1_stats['ci95_hi'][2]], 'k-', lw=2) # retest

ax.scatter(x1, df1_stats['mean'][1] , marker="d", s=msize, facecolor='k')
ax.scatter(x2, df1_stats['mean'][0] , marker="d", s=msize, facecolor='k')
ax.scatter(x3, df1_stats['mean'][3], marker="d", s=msize, facecolor='k')
ax.scatter(x4, df1_stats['mean'][2], marker="d", s=msize, facecolor='k')
# add 0 line
ax.axhline(y=0,  color='k', linestyle='--')

# VIEW
plt.subplot(grid[0, 3:])
ax = sns.stripplot(x="category", y="oData", hue='session',
                   data=df2,
                   jitter=j_val, 
                   split=True,
                   dodge=d_val,
                   edgecolor='black',
                   palette=flatui,
                   size=4
                   )
sns.despine()
ax.legend_.remove()
plt.ylim((-1, 1.1))
ax.set_xlabel('Category')
ax.set_ylabel(' ')

# xcoord and y coord has the x and y coord of individual data points 
x_coords, y_coords = [], []
for point_pair in ax.collections:
   for x, y in point_pair.get_offsets():
       x_coords.append(x)
       y_coords.append(y)
    
x1 = np.median(x_coords[0:25])
x2 = np.median(x_coords[25:50])
x3 = np.median(x_coords[50:75])
x4 = np.median(x_coords[75:100])

ax.plot([x1, x1], [df2_stats['ci95_lo'][1],df2_stats['ci95_hi'][1]], 'k-', lw=2) # test
ax.plot([x2, x2], [df2_stats['ci95_lo'][0],df2_stats['ci95_hi'][0]], 'k-', lw=2) # retest
ax.plot([x3, x3], [df2_stats['ci95_lo'][3],df2_stats['ci95_hi'][3]], 'k-', lw=2) # test
ax.plot([x4, x4], [df2_stats['ci95_lo'][2],df2_stats['ci95_hi'][2]], 'k-', lw=2) # retest

ax.scatter(x1, df2_stats['mean'][1] , marker="d", s=msize, facecolor='k')
ax.scatter(x2, df2_stats['mean'][0] , marker="d", s=msize, facecolor='k')
ax.scatter(x3, df2_stats['mean'][3], marker="d", s=msize, facecolor='k')
ax.scatter(x4, df2_stats['mean'][2], marker="d", s=msize, facecolor='k')
# add 0 line
ax.axhline(y=0,  color='k', linestyle='--')
plt.tight_layout()
savename = savepath + 'Fig02_overall_rating_plot.pdf'
plt.savefig(savename, dpi=900)
