# -*- coding: utf-8 -*-
"""
Script to create dataframes with overall and continuous rating data
and write out csv files
- separately for Rate and View groups
- combined groups
saves: df_overall_rate, df_overall_view, df_overall
       df_continuous_rate, df_continuous_view, df_continuous
@author: ilkay.isik
"""
# load libraries
import pandas as pd
import numpy as np

# set the paths and load the data
f_dirs = ['/Users/ilkay.isik/aesthetic_dynamics/data/data_rate.npz',
          '/Users/ilkay.isik/aesthetic_dynamics/data/data_view.npz']
savepath = '/Users/ilkay.isik/aesthetic_dynamics/data/'

data = []
for i in range(2):
    data.append(np.load(f_dirs[i]))
print (data[0].files)
# %% experiment parameters
nr_movies = data[0]['nr_movies']
nr_subjects = [data[0]['nr_subjects'], data[1]['nr_subjects']]
nr_runs = data[0]['nr_runs']
cat_lol = data[0]['cat_lol'] # category list of list
nr_movies_per_run = data[0]['nr_movies_per_run']
moi_list = data[0]['moi_list']
# python3 requires:
moi_list = [mov.decode("utf-8") for mov in moi_list]
cats = data[0]['categories']
cats = [c.decode("utf-8") for c in cats]
nr_cats = len(cats)
movie_dur = data[0]['movie_dur']
nr_sessions = 2
group = ['Rate', 'View']
movie_count = [len(f) for f in cat_lol]
m_count_sum = np.cumsum(movie_count)  # cumulative sum
m_count = np.insert(m_count_sum, 0, 0)  # insert zero
timeC = data[0]['cTime_ses1']
cTime = timeC[0, 0, :]
nr_tp = len(cTime)
# %% create data frames for overall data for BP03 and BP04
group = ['Rate', 'View']
group_frames = []
# Place the overall data in a data frame for two sessions
for exp in range(len(data)):
    G = group[exp]
    print(G)
    # Session 1
    df_o_ses1 = pd.DataFrame()

    for s in range(nr_subjects[exp]):
        if G == 'View':
            sub = 'sub-{}'.format(str(s+26).zfill(2))
        elif G == 'Rate':
            sub = 'sub-{}'.format(str(s+1).zfill(2))

        for m in range(nr_movies):
            temp_df_o = pd.DataFrame({'category': moi_list[m][0].upper(),
                                      'movName': moi_list[m],
                                      'subject': s+1,
                                      'sub_code': sub,
                                      'oData': data[exp]['oData_ses1'][s][m],
                                      'oTime': data[exp]['oTime_ses1'][s][m],
                                      'session': 'Test',
                                      'group': G},
                                     index=[s])
            df_o_ses1 = df_o_ses1.append(temp_df_o)  # ignore_index=True
    # Session2
    df_o_ses2 = pd.DataFrame()
    for s in range(nr_subjects[exp]):
        if G == 'View':
            sub = 'sub-{}'.format(str(s+26).zfill(2))
        elif G == 'Rate':
            sub = 'sub-{}'.format(str(s+1).zfill(2))
        for m in range(nr_movies):
            temp_df_o = pd.DataFrame({'category': moi_list[m][0].upper(),
                                      'movName': moi_list[m],
                                      'subject': s+1,
                                      'sub_code': sub,
                                      'oData': data[exp]['oData_ses2'][s][m],
                                      'oTime': data[exp]['oTime_ses2'][s][m],
                                      'session': 'Retest',
                                      'group':G},
                                     index=[s])
            df_o_ses2 = df_o_ses2.append(temp_df_o)  # ignore_index=True

# combined data frame for overall ratings
    frames = [df_o_ses1, df_o_ses2]
    df_o = pd.concat(frames)

    # save the dataframe for one group
    print ('Saving the overall rating dataframe for ' + group[exp] + ' Group')
    save_overall = savepath + 'df_overall_' + group[exp] + '.csv'
    df_o.to_csv(save_overall)
    group_frames.append(df_o)

# combine data: rate and view
df_overall = pd.concat(group_frames)
df_overall = df_overall.reset_index()
df_overall = df_overall.loc[:, ~df_overall.columns.str.contains('^index')]
# %% create data frames for continuous data for BP03 and BP04
# Rate, Session 1
df_c_ses1 = pd.DataFrame()
for s in range(nr_subjects[0]):
    sub = 'sub-{}'.format(str(s+1).zfill(2))
    for m in range(nr_movies):
        temp_df_c = pd.DataFrame({'category': moi_list[m][0].upper(),
                                  'movName': moi_list[m],
                                  'subject': s+1,
                                  'cData': data[0]['cData_ses1'][s][m],
                                  'cTime': cTime},
                                 index=[sub] * nr_tp)
        df_c_ses1 = df_c_ses1.append(temp_df_c)  # ignore_index=True
# Rate, session 2
df_c_ses2 = pd.DataFrame()
for s in range(nr_subjects[0]):
    sub = 'sub-{}'.format(str(s+1).zfill(2))
    for m in range(nr_movies):
        temp_df_c = pd.DataFrame({'category': moi_list[m][0].upper(),
                                  'movName': moi_list[m],
                                  'subject': s+1,
                                  'cData': data[0]['cData_ses2'][s][m],
                                  'cTime': cTime},
                                 index=[sub] * nr_tp)
        df_c_ses2 = df_c_ses2.append(temp_df_c)  # ignore_index=True

# combine data from sessions
frames_c = [df_c_ses1, df_c_ses2]
df_cont = pd.concat(frames_c)

# add session as a column
ses_col = ['Test'] * len(df_c_ses1.index) + ['Retest'] * len(df_c_ses2.index)
df_cont['session'] = ses_col

# add group as a column
exp_col = [group[0]]*(nr_subjects[0]*nr_sessions*nr_movies*nr_tp)
df_cont['group'] = exp_col

# reset index and get rid of the created extra index col
df_cont = df_cont.reset_index()
df_cont = df_cont.rename(columns={"index": "Sub_comb"})

print ('Saving the continuous rating dataframe for Rate Group')
save_cont = savepath + 'df_continuous_rate.csv'
df_cont.to_csv(save_cont)

# View, session 2 (only has continuous data for session 2)
df_c = pd.DataFrame()
for s in range(nr_subjects[1]):
    sub = 'sub-{}'.format(str(s + 26).zfill(2))
    for m in range(nr_movies):
        temp_df_c = pd.DataFrame({'Category': moi_list[m][0].upper(),
                                  'Movie': moi_list[m], 'Subject': s+1,
                                  'C_Data': data[1]['cData_ses2'][s][m],
                                  'C_Time': cTime, 'Session': 'Retest',
                                  'Group': 'View'},
                                 index=[sub] * nr_tp)
        df_c = df_c.append(temp_df_c)  # ignore_index=True

 # reset index and get rid f the created extra index col
df_c = df_c.reset_index()
df_c = df_c.rename(columns={"index": "Sub_comb"})

print ('Saving the continuous dataframe for View Group')
save_cont = savepath + 'df_continuous_view.csv'
df_c.to_csv(save_cont)

# combine the dataframes from BP03 BP04 continuous data
df_cont = pd.concat([df_cont, df_c])
df_cont = df_cont.reset_index()
df_cont = df_cont.loc[:, ~df_cont.columns.str.contains('^index')]
# %% write out to csv
print('Saving combined dataframes')
save_overall = savepath + 'df_overall.csv'
df_overall.to_csv(save_overall, index=False)
save_cont = savepath+'df_continuous.csv'
df_cont.to_csv(save_cont)
