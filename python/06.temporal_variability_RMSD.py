 # -*- coding: utf-8 -*-
"""
Investigation of the variability of continuous responses
using root mean squared differences [rmsd]
- calculate rmsd values 
- save them as csv [df_rmsd_values.csv]
- histogram of the mean rmsd values per participant [Fig05_A]
- kmeans clustering with rmsd values [Fig05_C]
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from scipy.stats import zscore
import sys
sys.path.append('/Users/ilkay.isik/aesthetic_dynamics/aest_dynamics_code/python/functions')
from useful_functions import rmsd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
# %% set paths and load the data
f_dirs = ['/Users/ilkay.isik/aesthetic_dynamics/data/data_rate.npz',
          '/Users/ilkay.isik/aesthetic_dynamics/data/data_view.npz']
savepath = '/Users/ilkay.isik/aesthetic_dynamics/'
data = []
for i in range(2):
    data.append(np.load(f_dirs[i]))
print (data[0].files)
# %% parameters
cdt = data[0]['cData_ses1']
cdrt = data[0]['cData_ses2']
cdrtv = data[1]['cData_ses2']
cTimeSes1 = data[0]['cTime_ses1']
cTime = cTimeSes1[0, 0, :]  # to use in the plots

nr_movies = data[0]['nr_movies']
nr_sub_r = data[0]['nr_subjects']
nr_sub_v = data[1]['nr_subjects']
nr_sub = nr_sub_r  + nr_sub_v # total nr of subject
nr_runs = data[0]['nr_runs']

# in python 3 the result of this is bytes instead of list. therefore applying 
# list comprehension to do the change to list with strings
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
# %% Calculate the rate of change with root mean of the squared differences
# Rate - test
rmsd_r_t = np.zeros([nr_sub_r, nr_movies])
for m in range(nr_movies):  # loop through movies
    for s in range(nr_sub_r):  # loop through subjects
        t_ts = cdt[s, m, :]  # test
        rmsd_r_t[s, m] = rmsd(t_ts)
# Rate - retest      
rmsd_r_rt = np.zeros([nr_sub_r, nr_movies])
for m in range(nr_movies):  # loop through movies
    for s in range(nr_sub_r):  # loop through subjects
        rt_ts = cdrt[s, m, :]  # retest
        rmsd_r_rt[s, m] = rmsd(rt_ts)

rmsd_v_rt = np.zeros([nr_sub_v, nr_movies])
for m in range(nr_movies):  # loop through movies
    for s in range(nr_sub_v):  # loop through subjects
        rt_ts = cdrtv[s, m, :]  # retest
        rmsd_v_rt[s, m] = rmsd(rt_ts)
        
rmsd_vals = np.zeros([nr_sub_r, nr_movies, 3])

rmsd_vals[:, :, 0] = rmsd_r_t
rmsd_vals[:, :, 1] = rmsd_r_rt
rmsd_vals[:, :, 2] = rmsd_v_rt

rmsd_vals_z = np.copy(rmsd_vals)
rmsd_vals_z = zscore(rmsd_vals)
rmsd_r_t_z = rmsd_vals_z[:, :, 0]
rmsd_r_rt_z = rmsd_vals_z[:, :, 1]
rmsd_v_rt_z = rmsd_vals_z[:, :, 2]
# %% Put rmsd data in dataframe and save as csv for later use
# Rate Test
df_rmsd_rt = pd.DataFrame()
for s in range(nr_sub_r):
    for m in range(nr_movies):
        temp_df = pd.DataFrame({'Category': moi_list[m][0].upper(),
                                'Movie': moi_list[m], 
                                'Subject': s+1,
                                'rmsd': rmsd_r_t[s, m],
                                'rmsd_z': rmsd_r_t_z[s, m],
                                'Session':'Test',
                                'Group':'Rate'},
                                 index=['Sub' + str(s+1)])
        df_rmsd_rt = df_rmsd_rt.append(temp_df)

# Rate Retest
df_rmsd_rrt = pd.DataFrame()
for s in range(nr_sub_r):
    for m in range(nr_movies):
        temp_df = pd.DataFrame({'Category': moi_list[m][0].upper(),
                                'Movie': moi_list[m], 
                                'Subject': s+1,
                                'rmsd': rmsd_r_rt[s, m],
                                'rmsd_z': rmsd_r_rt_z[s, m],
                                'Session':'Retest',
                                'Group': 'Rate'},
                                 index=['Sub' + str(s+1)])
        df_rmsd_rrt = df_rmsd_rrt.append(temp_df)
        
# View Retest
df_rmsd_vrt = pd.DataFrame()
for s in range(nr_sub_v):
    print(s)
    for m in range(nr_movies):
        temp_df = pd.DataFrame({'Category': moi_list[m][0].upper(),
                                'Movie': moi_list[m], 
                                'Subject': s+26,
                                'rmsd': rmsd_v_rt[s, m],
                                'rmsd_z': rmsd_v_rt_z[s, m],
                                'Session':'Retest',
                                'Group':'View'},
                                 index=['Sub' + str(s+1)])
        df_rmsd_vrt = df_rmsd_vrt.append(temp_df)
        
frames = [df_rmsd_rt, df_rmsd_rrt, df_rmsd_vrt ]
df_rmsd = pd.concat(frames)

# save it
save_df_rmsd = savepath + 'data/df_rmsd_values.csv'
df_rmsd.to_csv(save_df_rmsd , index=None)
# %% Simulate time series to compare to real rmsd values
# curve with no change [with zero values]
c_no_change = np.zeros([300])
np.var(c_no_change)

# timeseries with logarithmic increase
x = np.linspace(1, 10, 300)
c_log_change = np.log10(x)

# inner panel in figure 5
# plt.plot(c_log_change)

# timeseries with maximum change [theoretically]
c_max_change = np.asarray([1, -1] * 150)
c_max_change[0] = 0

# increasing line
c_inc_line = np.arange(1,301) / 300

# put the rmsd values of those curves in a dict
sim_vals = dict()
sim_vals['no_change'] = rmsd(c_no_change)
sim_vals['exp_change'] = rmsd(c_log_change)
sim_vals['max_change'] = rmsd(c_max_change)
sim_vals['increasing line'] = rmsd(c_inc_line)
# %% normalization of rmsd values
# which max value should be used to do the normalization
# the real max, max_rmsd or the theoretical max rmsd(c_max_change) ???
theo_max_val = rmsd(c_max_change)
max_rmsd, min_rmsd = np.max(rmsd_vals), np.min(rmsd_vals)
np.argwhere(rmsd_vals == max_rmsd)
real_max_change_curve = cdrt[21, 26, :]
# using the theoretical max
norm_rmsd_vals = (rmsd_vals - min_rmsd) / (theo_max_val - min_rmsd)
np.max(norm_rmsd_vals)
np.argwhere(norm_rmsd_vals == np.max(norm_rmsd_vals))
# %% Fig 5A: Histogram of the rmsd values for dance and landscape MEAN per subj
# Test
flatui = ["#F8766D", "#00BFC4"]
sns.set_palette(flatui)
sns.set_style("white")
sns.set_context('paper',font_scale=1.2 )
plt.style.use('seaborn-deep')
sns.set_style("ticks")

fig, ax = plt.subplots(3, 1, figsize=(5.5, 7), sharex=True, sharey=True)
sns.despine()
ax=ax.flatten()

# Rate Test
x1 = np.mean(rmsd_vals[:, :15, 0], axis=1) # prt_mean_dance
y1 = np.mean(rmsd_vals[:, 15:, 0], axis=1) # prt_mean_lscp

mean_dance, mean_lscp = np.mean(x1), np.mean(y1)
ax[0].hist([x1, y1], bins=15, label=['Dance', 'Landscape'], color=flatui)
ax[0].legend(loc='upper right')

ax[0].text(0.028, 3.2, 'Dance: '+ "%0.3f" % (mean_dance,))
ax[0].text(0.028, 2.5, 'Landscape: '+ "%0.3f" % (mean_lscp,))
ax[0].set_title('Rate Group - Test session')
ax[0].axvline(sim_vals['exp_change'], color='r', linestyle='--')
plt.xlim(0, 0.05)


# Rate Retest
x2 = np.mean(rmsd_vals[:, :15, 1], axis=1) # prt_mean_dance
y2 = np.mean(rmsd_vals[:, 15:, 1], axis=1) # prt_mean_lscp
mean_dance, mean_lscp = np.mean(x2), np.mean(y2)
ax[1].hist([x2, y2], bins=20, label=['Dance', 'Landscape'], color=flatui)
# ax[1].text(0.028, 4.2, 'Means:')
ax[1].text(0.028, 3.2, 'Dance: '+ "%0.3f" % (mean_dance,))
ax[1].text(0.028, 2.5, 'Landscape: '+ "%0.3f" % (mean_lscp,))
ax[1].set_title('Rate - Retest')
ax[1].axvline(sim_vals['exp_change'], color='r', linestyle='--')
ax[1].set_ylabel('Number of participants')

# View Retest
x3 = np.mean(rmsd_vals[:, :15, 2], axis=1) # prt_mean_dance
y3 = np.mean(rmsd_vals[:, 15:, 2], axis=1) # prt_mean_lscp
mean_dance, mean_lscp = np.mean(x3), np.mean(y3)
ax[2].hist([x3, y3], bins=20, label=['Dance', 'Landscape'], color=flatui)

ax[2].text(0.028, 3.2, 'Dance: '+ "%0.3f" % (mean_dance,))
ax[2].text(0.028, 2.5, 'Landscape: '+ "%0.3f" % (mean_lscp,))
ax[2].set_title('View  - Retest')
ax[2].axvline(sim_vals['exp_change'], color='r', linestyle='--')
ax[2].set_xlabel('Average RMSD values')

plt.tight_layout()
fname = savepath + 'output/figures/Fig05_A_MeanRMSD_TestRetest_RateView' + '.pdf'
plt.savefig(fname, dpi=900)

#%% ########################## K-MEANS CLUSTERING  ############################ 
rmsd_dnc = rmsd_vals[:, :15, :]
rmsd_lsp = rmsd_vals[:, 15:, :]

# compute medians per person [only for retest]
med_dnc_rate = np.median(rmsd_dnc[:, :, 1], axis=1)
med_dnc_view = np.median(rmsd_dnc[:, :, 2], axis=1)
med_lsp_rate = np.median(rmsd_lsp[:, :, 1], axis=1)
med_lsp_view = np.median(rmsd_lsp[:, :, 2], axis=1)

med_dnc = np.concatenate((med_dnc_rate, med_dnc_view), axis=0)
med_lsp = np.concatenate((med_lsp_rate, med_lsp_view), axis=0)
rmsd = np.array([med_dnc, med_lsp]).T
# %%take the median of the zscore values for all measures
# Clustering related parameters
cSize = np.arange(2, 11); # try 2-10 clusters
# clusterSess = 1;
clustDist = 'euclidean'; #default
clustRep = 10;
clusters = ('2', '3', '4', '5', '6', '7', '8', '9', '10')

fig, ax = plt.subplots(1, figsize=(5, 5))
sns.despine()
k = 0

silh_scores, mean_silh_scores = [], []
clust = []
for s in cSize:
    clust_mea = {}
    print('Cluster size is: ', str(s))
    km = KMeans(n_clusters=s, init='k-means++', n_init=clustRep, 
                algorithm="auto",)
    km.fit(rmsd)
    clust_mea['size'] = s
    clust_mea['labels'] = km.labels_
    clust_mea['ccenters'] = km.cluster_centers_
    clust_mea['inertia'] = km.inertia_
    print('Silhouette Coefficient: {:0.3f}'.format(silhouette_score(rmsd, km.labels_)))
    
    mean_silh_scores.append(silhouette_score(rmsd, km.labels_,
                                             metric=clustDist))
    silh_scores.append(silhouette_samples(rmsd, km.labels_, 
                                                  metric=clustDist)) 
    clust_mea['silh_coef'] = silhouette_score(rmsd, km.labels_,
                                                metric=clustDist)
    clust_mea['silh_scores'] = silhouette_samples(rmsd, km.labels_, 
                                                  metric=clustDist)
    clust.append(clust_mea)

ax.plot(cSize, mean_silh_scores, 'o') 
ax.set_title('rmsd')
ax.set_ylabel('Silhouette scores')
ax.set_xlabel('nr of clusters')

for csize in [2, 3]:
    print('Cluster size:', csize)
    indices = []
    csize_ind = int(np.argwhere(cSize==csize))
    labels = clust_mea['labels']
    for label in range(csize):
        print("Cluster: ", str(label + 1))
        dat_idx = np.where(labels == label)[0]
        print("Nr of people in this cluster:", str(len(dat_idx)))
        print(dat_idx+1)
        indices.append(np.ndarray.tolist(dat_idx))
        
# Results: Silhouette score suggests two clusters
# %% Fig05-C: Plot showing the actual clusters formed for RMSD
n_clusters = 2
cluster_labels = clust[0]['labels']
n_clusters = 2
X = rmsd
sns.set_style('white')
sns.set_context('talk')
fig, ax = plt.subplots()
sns.despine()
fig.set_size_inches(8,8)
flatui = ["#F8766D", "#00BFC4"]
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

ax.scatter(X[:, 0], X[:, 1], marker='.', s=80, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clust[0]['ccenters']
# Draw white circles at cluster centers
ax.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=100, edgecolor='k')
for i, c in enumerate(centers):
    k = i + 1
    ax.scatter(c[0], c[1], marker='$%d$' %k, alpha=1,
               s=50, edgecolor='k')
ax.set_xlim([0, 0.06])
ax.set_ylim([0, 0.06])
ax.set_title("The visualization of the clustered data with rmsd scores")
ax.set_xlabel("Dance")
ax.set_ylabel("Landscape")
plt.savefig(savepath + 'output/figures/Fig05-C_clusters_with_rmsd.pdf', dpi=300)

