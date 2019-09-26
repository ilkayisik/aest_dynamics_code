#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Experiment with movies and rating scales.
Windows10, PsychoPy2 v1.84.2
Important notes:
session nr needs to be written with leading zeros: 01 or 02
stim file should contain video files
psychopy uses visual.MovieStim3 to show videos. It downloads ffmpeg library when using for the first time
griffin powermate dial is used for response collection. 
The details on how to make it run with PsychoPy2 can be found Griffin_Powermate_Usage folder
@author: ilkay isik & faruk gulban

This version of the experiment presents the continuous scale based on the
session number determined via the GUI
if session_nr:01 no continuous scale, only the overall scale
if session_nr:02 present the continuous scale + the overall scale
"""
import os
import time
import pickle
import numpy as np
from psychopy import visual, core, event, monitors, gui
from glob import glob
from pprint import pprint

# %%
start_time = time.strftime("%H_%M", time.localtime())
""" Dial (Griffin Powermate) related """

dial_history = []
accel_constant = 1.9  
def tick_limits(tick_pos=(0, 0), lim_min=-20, lim_max=20):
    """Used to prevent tick going out of bounds."""
    if tick_pos[0] <= lim_min:
        return (lim_min, tick_pos[1])
    elif tick_pos[0] >= lim_max:
        return (lim_max, tick_pos[1])
    else:
        return tick_pos

def dial_listener(data, tick_min=-2.0, tick_max=2.0):
    """Listen dial events.
    This function is only relevant when there is a powermate.
    Data is a list with values in it indicating different dial events
    For instance data[1] is 1 if there is a click,
    data[2] is a value between 1 and 255, if it is 1 it is a right
    turn and if it is 255 it is a left turn.
    """
    global new_tik_pos, dial_click, accel_constant, state_start, dresp_time, dial


    # Let the person click the dial and give an overall rating
    # if state ide is -1
    if data[1] == 1 and state_ide[count_s] == -1 and dial:
        dial_click = True

    # Left turn, dial_sync_tik_step will be negative and new_tik_pos
    # will be updated
    elif data[2] > 127 and dial:
        dresp = data[2] - 256  # number that is read from dial
        dresp_time = clock.getTime()-state_start
        # acceleration multiplier
        accel_mult = np.power(accel_constant, np.abs(dresp)-1)
        dial_sync_tik_step = np.multiply(TIK_STEP, dresp * accel_mult)
        new_tik_pos = tuple(np.add(slider_tik.pos, dial_sync_tik_step))
        new_tik_pos = tick_limits(new_tik_pos)
        dial_history.append([new_tik_pos[0], dresp_time])


    # Right turn, dial_sync_tik_step will be positive and new_tik_pos
    # will be updated
    elif data[2] <= 127 and dial:
        dresp = data[2]  # number that is read from dial
        dresp_time = clock.getTime()-state_start
        # acceleration multiplier
        accel_mult = np.power(accel_constant, np.abs(dresp)-1)

        dial_sync_tik_step = np.multiply(TIK_STEP, dresp * accel_mult)
        new_tik_pos = tuple(np.add(slider_tik.pos, dial_sync_tik_step))
        new_tik_pos = tick_limits(new_tik_pos)
        # to save history of the dial
        dial_history.append([new_tik_pos[0], dresp_time])


try:
    from griffin_powermate import GriffinPowermate
    dial = True
    devices = GriffinPowermate.find_all()
    # use the first powermate available
    powermate = devices[0]
    powermate.open()
    powermate.on_event('raw', dial_listener)
    # set the light (0 - 255)
    powermate.set_brightness(0)

except:
    print('----------------------- \n'
          'The dial will not work! \n'
          'Make sure griffin_powermate.py is in the folder '
          'and the dial(powermate) is plugged in! \n'
          '----------------------- \n')

dial_click = False
dial = False  # used to listen to dial only when wanted

# %%
""" GUI operations """

info = {'Subject_number': '00',
        'Experiment_type': 'View',
        'Session_nr': ['01', '02'],
        'full_screen': [False, True]
        }
dictDlg = gui.DlgFromDict(dictionary=info, title='View')

subject_filename = info['Experiment_type'] + '_Sub_' +info['Subject_number'] +\
                   '_Sess' + info['Session_nr'] + '*.pickle'
cwd = os.getcwd()  # get current directory of this python script
protocol_dir = os.path.join(cwd,  'protocols')
pickle_path = os.path.join(protocol_dir,  subject_filename)

if dictDlg.OK:
    print('Clicked OK, continuing...')
else:
    print('User cancelled, exiting...')
    sys.exit(1)

file = open(glob(pickle_path)[-1], 'rb')  # -1 is to take the latest file
exp_info = pickle.load(file)
file.close()

# %%
""" Parameters from pickle """
subject_id = exp_info['subject_id']
movie_filenames = exp_info['movie_filenames']
nr_movies = exp_info['nr_movies']
run_orders = exp_info['run_orders']
movie_orders = exp_info['movie_orders']
state_orders = exp_info['state_orders']
state_durations = exp_info['state_durations']
nr_runs = len(run_orders)
nr_nm = len(exp_info['nonmovie_identifiers'])
nr_mov_per_run = [len(movie_orders[i]) for i in range(nr_runs)]

# %%
""" Monitor """

# set monitor information used in the experimental setup
moni = monitors.Monitor('testMonitor', width=53, distance=60,
                        #gamma=0.6
                        )  # in cm

# set screen (make 'fullscr = True' for fullscreen)
win = visual.Window(size=(1920, 1080),#size=(800,600),
                    screen=0, winType='pyglet',
                    allowGUI=False,  # show mouse
                    fullscr=info['full_screen'],
                    monitor=moni,
                    color='grey',
                    colorSpace='rgb',
                    units='deg',
                    useFBO=True,
                    #waitBlanking=False
                    )

# %%
""" Movie stimuli """

# organize movie paths relative to the script directory
video_paths = list()
for i in range(nr_movies):
    video_paths.append(os.path.join('stimuli', movie_filenames[i]))


def load_movies(idx_run):
    """Load movies which will be displayed in the current run."""
    global nr_movies, movie_orders, video_paths

    # initiate list of lists to store moviestim
    movies = list()
    [movies.append([]) for m in range(nr_movies)]

    for j in movie_orders[idx_run]:
        print 'Loading movie %i' % j
        movies[j] = visual.MovieStim3(win, video_paths[j],
                                      size=(1280, 720),
                                      pos=[0, 0], flipVert=False,
                                      flipHoriz=False, loop=False,
                                      noAudio=True)
    return movies


# %%
""" Text stimuli """

text = visual.TextStim(win=win, height=2, wrapWidth=50,
                       text= 'Instructions... Press enter to start.',
                       )

#create vertices for the letters
plusVert = [(0,0.55),(0,-0.5),(0,0),(0.5,0), (-0.5,0)]
minusVert = [(0.5,0), (-0.5,0)]


slider_text_l = visual.ShapeStim(win,
                                 vertices=minusVert,
                                 fillColor='black',
                                 lineColor='black',
                                 closeShape=False,
                                 lineWidth=3.5,
                                 interpolate=False

                                 )
slider_text_r = visual.ShapeStim(win,
                                 vertices=plusVert,
                                 fillColor='black',
                                 lineColor='black',
                                 closeShape=False,
                                 lineWidth=3.5,
                                 interpolate=False
                                 )

# %%
""" Response stimuli """

TIK_STEP = [1, 0]  # x and y
SLIDER_C_POS = (0, -22.5)  # continuous slider
SLIDER_O_POS = (0, 0)  # overall slider


slider_bar = visual.GratingStim(win=win, tex=None,
                                pos=(0, 0), size=(40+1, 1.5),
                                color='darkgray', interpolate=False)

slider_tik = visual.GratingStim(win=win, tex=None,
                                pos=(0, 0), size=(1, 1.5),
                                color='DarkRed', interpolate=False)

# %%
""" Output Related """
date = time.strftime("%b_%d_%Y_%H_%M", time.localtime())

# check if output folder exists, if not create
out_dir = 'rating_outputs'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# file identifier
out_name = subject_id + '-' + date + '.pickle'

# construct pickle name
pickle_name = os.path.join(out_dir, out_name)

# create a dictionary for the output variables
output_variables = {'subject_id': subject_id,
                    'start_time': start_time,
                    'state_durations': [],
                    'state_orders': state_orders,
                    'run_orders' : run_orders,
                    'movie_orders' : movie_orders,
                    'movie_filenames' : movie_filenames,
                    'date': date,
                    'scale_c': [],
                    'scale_o': [],
                    'end_time': []
                    }

# %%
""" First Trigger """
start_trigger = False
while start_trigger:
    text.draw()
    win.flip()

    for keys in event.getKeys():

        if keys in ['return']:
            start_trigger = False

        elif keys[0] in ['escape', 'q']:
            win.close()
            core.quit()

# %%
""" Render loop """

count_run = 0   # default 0
count_s = 0 # to count states within runs, default 0

while count_run < nr_runs:  # loop over runs

    print 'Run counter: %i' % count_run

    # set run specific parameters
    run_ide = run_orders[count_run]  # run identifier
    state_ide = state_orders[run_ide].astype('int')
    state_dur = state_durations[run_ide].astype(np.float)
    movies = load_movies(run_ide)  # might take some time
    run_duration = np.sum(state_dur[count_s::])
    state_start_offset = np.copy(count_s)  # used in re-started runs, mostly 0

    # ask to start run after loading movies
    start_trigger = True
    text.text = 'Bitte druecken Sie Enter um Block %i zu starten  ' % (count_run+1)
    while start_trigger:
      
        text.draw()
        win.flip()

        for keys in event.getKeys():

            if keys in ['return']:
                start_trigger = False

            elif keys[0] in ['escape', 'q']:
                win.close()
                core.quit()

    # give the system time to settle
    core.wait(0.5)

    # create a clock (start from 0 after each run)
    clock = core.Clock()
    clock.reset()

    while clock.getTime() < run_duration:  # loop over states

        print 'State counter: %i' % count_s

        state_start = clock.getTime()

        # prepare stimuli for the upcoming state
        if state_ide[count_s] >= 0 and info['Session_nr'] == '01':
            dial = False
            print 'State for this movie: %i' % count_s

        elif state_ide[count_s] >= 0 and info['Session_nr'] == '02':
            dial = True
            slider_text_r.pos = (22, -22.5)
            slider_text_l.pos = (-22.5, -22.5)
            slider_bar.pos = SLIDER_C_POS
            slider_tik.pos = SLIDER_C_POS
            new_tik_pos = slider_tik.pos
            dial_history = [[SLIDER_C_POS[0], 0]]
            print 'State for this movie: %i' % count_s

        elif state_ide[count_s] == -1:
            slider_text_r.pos = (22, 0)
            slider_text_l.pos = (-22.5, 0)
            slider_bar.pos = SLIDER_O_POS
            slider_tik.pos = SLIDER_O_POS
            new_tik_pos = slider_tik.pos
            dial_history = [[SLIDER_O_POS[0], 0]]
            dial = True

        while clock.getTime() < np.sum(state_dur[state_start_offset:count_s+1]):

            # determine state draws
            if state_ide[count_s] >= 0 and info['Session_nr'] == '02':
                dial = True
                slider_bar.draw()
                slider_text_r.draw()
                slider_text_l.draw()
                slider_tik.pos = new_tik_pos
                slider_tik.draw()
                movies[state_ide[count_s]].draw()

            elif state_ide[count_s] >= 0 and info['Session_nr'] == '01':
                dial = False
                movies[state_ide[count_s]].draw()

            elif state_ide[count_s] == -1:
                slider_bar.draw()
                slider_text_r.draw()
                slider_text_l.draw()
                slider_tik.pos = new_tik_pos
                slider_tik.draw()

            elif state_ide[count_s] == -2:
                win.clearBuffer()

            elif state_ide[count_s] == -3:
                text.text = '+'
                text.pos = (0, 0)
                if clock.getTime() % 0.5 < 0.25:
                    text.color = 'darkgray'
                else:
                    text.color = 'lightgray'
                text.draw()

            win.flip()

            if state_ide[count_s] == -1 and dial_click:
                state_dur[count_s] = clock.getTime() - state_start
                dial_click = False
                run_duration = np.sum(state_dur)  # update
                dial = False
                break

            # handle key presses each frame
            for keys in event.getKeys(timeStamped=True):
                if info['full_screen']==False and keys[0]in ['escape', 'q'] :
                    win.close()
                    # pprint(output_variables)
                    core.quit()

                elif keys[0]in ['left']:
                    new_tik_pos = tuple(np.subtract(slider_tik.pos, TIK_STEP))
                    new_tik_pos = tick_limits(new_tik_pos)

                elif keys[0]in ['right']:
                    new_tik_pos = tuple(np.add(slider_tik.pos, TIK_STEP))
                    new_tik_pos = tick_limits(new_tik_pos)

        # cleanup operations (including saving) after each state
        dial=False
        if state_ide[count_s] >= 0 and info['Session_nr'] == '02':
            movies[state_ide[count_s]].pause()
            output_variables['scale_c'].append(dial_history)

        elif state_ide[count_s] == -1:
            output_variables['scale_o'].append(dial_history[-1])

        output_variables['state_durations'].append(state_dur)
        pickle.dump(output_variables, open(pickle_name, 'wb'))

        count_s = count_s + 1

    count_run = count_run + 1
    count_s = 0  # reset states
    win.clearBuffer()
    win.flip()


print 'Finished.'

end_time = time.strftime("%H_%M", time.localtime())
output_variables['end_time'] = end_time
pickle.dump(output_variables, open(pickle_name, 'wb'))


# Let participant end the experiment
end_exp = True
#text.text = ('Danke fuer Ihre Teilnahme.\n\nBitte druecken Sie Enter um das Experiment zu beenden')
while end_exp:
    if info['Session_nr'] == '01':
        text.text = 'Der erste Teil des Experiments ist jetzt vorbei.\n\nDanke fuer Ihre Teilnahme'
        text.draw()
        win.flip()

    elif info['Session_nr'] == '02':
        text.text = 'Das Experiment ist jetzt vorbei.\nDanke fuer Ihre Teilnahme'
        text.draw()
        win.flip()

    for keys in event.getKeys():

        if keys in ['return']:
            end_exp = False


win.close()
core.quit()