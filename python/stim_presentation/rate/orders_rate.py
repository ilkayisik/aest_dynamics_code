#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
creates stimuli orders for two subjects in reverse order for rate group 
requires the odd numbered subject, will create the even number stim orders too
@author: ilkay.isik
"""
import sys
import os
import pickle
import time
import numpy as np
from psychopy import visual, gui  # visual is needed for gui to work in spyder
# %%
""" Check files """
try:
 # load the text file with the stimuli names
    filename = 'stim.txt'
     # put the names in a list
    stim_list = [line.rstrip('\n') for line in open(filename)]
except:
    print 'Stim file cannot be found'
    sys.exit(1)

# %%
""" GUI operations """

info = {'Subject_number': '00',
        'Experiment_type': ['Rate'],
        'Session_nr': '01',
        'nr_split_per_category': 2,
        #'nr_movie_per_category': np.array([26, 10, 26, 10]),
        'nr_movie_per_category': np.array([15,15]),
        #'movie_category_onsets': np.array([0, 26, 36, 62]),
        'movie_category_onsets': np.array([0, 15]),
        'duration_movie': 30,
        'duration_rating': 10,
        'duration_ITI' : 1,
        'duration_fix' : 1,
        'randomization':['Blocked', 'Intermixed']
        }

dictDlg = gui.DlgFromDict(dictionary=info, title='Pilot_Experiment')

if dictDlg.OK:
    print('Clicked OK, continuing...')
else:
    print('User cancelled, exiting...')
    sys.exit(1)

# %%
""" Determine some parameters based on GUI inputs """
subject_id = info['Experiment_type'] + '_Sub_' + info['Subject_number'] + \
            '_Sess' + info['Session_nr']
subject_id2 = info['Experiment_type'] + '_Sub_' +  "%02d" \
             %(int(subject_id.split('_')[2]) + 1) + '_Sess' + info['Session_nr']

nr_mov_per_cat = info['nr_movie_per_category']
nr_categ = np.size(nr_mov_per_cat)
nr_split = info['nr_split_per_category']
mov_ide_onsets = info['movie_category_onsets']  # movie identifier onset
#nr_runs = nr_categ * nr_split
nr_runs = nr_categ
rand_method = info['randomization']

# %%
""" Create the randomized stimuli lists for each category """
mov_identifiers = range(np.sum(nr_mov_per_cat))
rand_categ_orders = list()

if rand_method is 'Intermixed':
    np.random.shuffle(stim_list)

for i in range(nr_categ):
    onset = mov_ide_onsets[i]
    size = nr_mov_per_cat[i]
    temp = np.random.choice(range(onset, onset + size),
                            size, replace=False)
    rand_categ_orders.append(temp)

# %%
""" Split lists into even chunks """
movie_orders = list()
#for i in range(nr_categ):
#    for j in range(nr_split):
#        s_size = nr_mov_per_cat[i] / nr_split  # split size
#        s_onset = s_size * j  # split onset
#        movie_orders.append(rand_categ_orders[i][s_onset : s_onset+s_size])

movie_orders =  rand_categ_orders
movie_orders2 = movie_orders
movie_orders2 = [i[::-1] for i in movie_orders2]

# %%
""" Insert other codes to indicate ITI (-2), fix-cross(-1) rating states (-1) """

state_orders = list()
for r in range(len(movie_orders)):
    state_orders.append([])
    for m in range(len(movie_orders[r])):
        state_orders[r].extend([-2, -3, movie_orders[r][m], -2, -1])

for r in range(len(state_orders)):
    state_orders[r] = np.array(state_orders[r])

state_orders2 = list()
for r in range(len(movie_orders2)):
    state_orders2.append([])
    for m in range(len(movie_orders2[r])):
        state_orders2[r].extend([-2, -3, movie_orders2[r][m], -2, -1])

for r in range(len(state_orders2)):
    state_orders2[r] = np.array(state_orders2[r])



# %%
""" Create category(run) presentation order """
run_orders = [0, 1]
#run_orders = [0, 2, 1, 3]
run_orders2 = run_orders[::-1]


# %%
""" State durations """

mov_dur = info['duration_movie']
rat_dur = info['duration_rating']
ITI_dur = info['duration_ITI']
fix_dur = info['duration_fix']

state_dur = np.array([ITI_dur, fix_dur, mov_dur, ITI_dur, rat_dur])
state_durations = list()

for i in range(nr_categ):
    nr_mov_per_run = nr_mov_per_cat[i]
    temp = np.tile(state_dur, nr_mov_per_run)
    state_durations.append(temp)

#for i in range(nr_categ):
#    for j in range(nr_split):
#        nr_mov_per_run = nr_mov_per_cat[i] / nr_split
#        temp = np.tile(state_dur, nr_mov_per_run)
#        state_durations.append(temp)

# %%
""" Non-movie states """
nonmovie_identifiers = [-1, -2, -3]

# %%
""" Save the state orders as a pickle file """

out_dir = 'protocols'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# file identifier
date = time.strftime("%b_%d_%Y_%H_%M", time.localtime())

out_name = subject_id + '-' + 'Orders' + '-' + date + '.pickle'
# construct pickle name
pickle_name = os.path.join(out_dir, out_name)

# create a dictionary to save the output variables
output_variables = {'subject_id': subject_id,
                    'date': date,
                    'movie_orders': movie_orders,
                    'state_orders': state_orders,
                    'run_orders': run_orders,
                    'state_durations': state_durations,
                    'nr_movies': len(stim_list),
                    'movie_filenames': stim_list,
                    'nonmovie_identifiers': nonmovie_identifiers,
                    }
print state_orders
pickle.dump(output_variables, open(pickle_name, 'wb'))
print 'Saved as: ' + pickle_name

subject_id = subject_id2
movie_orders = movie_orders2
state_orders = state_orders2
run_orders = run_orders2
#stim_list = stim_list2

out_name = subject_id + '-' + 'Orders' + '-' + date + '.pickle'
# construct pickle name
pickle_name = os.path.join(out_dir, out_name)

# create a dictionary to save the output variables
output_variables = {'subject_id': subject_id,
                    'date': date,
                    'movie_orders': movie_orders,
                    'state_orders': state_orders,
                    'run_orders': run_orders,
                    'state_durations': state_durations,
                    'nr_movies': len(stim_list),
                    'movie_filenames': stim_list,
                    'nonmovie_identifiers': nonmovie_identifiers,
                    }
print state_orders
pickle.dump(output_variables, open(pickle_name, 'wb'))
print 'Saved as: ' + pickle_name
print 'Finished.'
