# -*- coding: utf-8 -*-
"""
Fig03: Plot continuous data values for each subject and movie
@author: ilkay.isik
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# %% set paths and load the data
fd_cont = '/Users/ilkay.isik/aesthetic_dynamics/data/df_continuous_rate.csv'
savepath = '/Users/ilkay.isik/aesthetic_dynamics/output/figures/'
# %% Some params
nr_sub = 25
nr_dnc_mov = 15
nr_lsp_mov = 15
nr_mov = nr_dnc_mov + nr_lsp_mov
# define your own color palette with 15 distinct colors
flatui = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#f032e6",
          "#d2f53c", "#fabebe",  "#008080", "#aa6e28", "#800000",
          "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080"]
sns.set_palette(flatui)
# %%
# read in the continuous data as dataframe
df_c = pd.read_csv(fd_cont)
df_c = df_c.rename(columns={'cData':'Rating','cTime':'Duration'})
# get rid of the unnamed column
df_c = df_c.loc[:, ~df_c.columns.str.contains('^Unnamed')]
mov_list = df_c.movName.unique()
# Create dataframes for each subject and put them in a list
df_per_sub = []

for s in range(nr_sub):
    df_sub = df_c[df_c['subject'] == s + 1]
    df_per_sub.append(df_sub)

# Create dataframes for each movie and put them in a list
df_per_mov = []
for m in range(1, nr_dnc_mov + 1):
    mov = 'd_' + '%.2d'%m + '.mp4'
    df_mov = df_c[df_c['movName'] == mov]
    df_per_mov.append(df_mov)

for m in range(1, nr_lsp_mov + 1):
    mov = 'ls_' + '%.2d' %m + '.mp4'
    df_mov = df_c[df_c['movName'] == mov]
    df_per_mov.append(df_mov)

# %% Create plots for each subject
sns.set_style('white')
sns.set_style("ticks")
sns.set_context('talk', font_scale=1.5)

sub_to_plot = 1
#for s in range(nr_sub):
for s in range(sub_to_plot):
    s = 1
    temp_df = df_per_sub[s]
    dnc_df = temp_df[temp_df['category'] == 'D']
    lsp_df = temp_df[temp_df['category'] == 'L']
    
    # dance plot
    sns.relplot(x="Duration", y="Rating",
                hue="movName", col="session", 
                palette=flatui,
                height=4, aspect=2,
                # facet_kws=dict(sharey=False),
                kind="line", legend=False,
                data=dnc_df)
    plt.suptitle('Dance')
    plt.ylim((-1.2, 1.2))
    sub = s + 1
    sname = savepath + 'Fig03_A_Dnc.pdf'
    plt.savefig(sname, dpi=300)
    plt.close()
    
    # landscape plot
    sns.relplot(x="Duration", y="Rating",
                hue="movName", col="session",
                palette=flatui,
                height=4, aspect=2, 
                # facet_kws=dict(sharex=False),
                kind="line",  legend=False,
                data=lsp_df)
    plt.ylim((-1.2, 1.2))
    plt.suptitle('Landscape')
    sname = savepath + 'Fig03_A_Lscp.pdf'
    plt.savefig(sname, dpi=300)
    plt.close()

# %% Create plots for each movie
movs_to_plot = 1, 16
titles = ['dance_movie', 'lscp_movie']
#for m in range(nr_mov):
for i, m in enumerate(movs_to_plot):
    temp_df = df_per_mov[m]
    sns.relplot(x="Duration", y="Rating",
                hue="subject", col="session",
                palette='Dark2',
                height=4, aspect=2,
                # facet_kws=dict(sharex=False), 
                legend=False, kind="line",
                data=temp_df)
    plt.ylim((-1.2, 1.2))
    plt.suptitle(titles[i])
    sname = savepath + 'Fig03_B_' + titles[i] + '.pdf'
    plt.savefig(sname, dpi=300)
    plt.close()
