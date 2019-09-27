# -*- coding: utf-8 -*-
"""
Load the motion energy values and compare scores for dance - landscape 
@author: ilkay isik
"""
import numpy as np
import glob
from scipy.stats import ttest_ind, ks_2samp
# %% Paths for the motion energy
root ='/Users/ilkay.isik/aesthetic_dynamics/'

savepath = root + 'plots/'
nr_dnc_mov, nr_lsp_mov = 15, 15

# Load the data [values in the text files]
egy_dnc, egy_lsp = [], []
dnc_txt_files = glob.glob(root + 'data/motion_energy/d*.txt')
nr_mov = len(dnc_txt_files)
for i in range(nr_mov):
    dnc_txt = np.genfromtxt(dnc_txt_files[i], delimiter=',')
    egy_dnc.append(dnc_txt)
lsp_txt_files = glob.glob(root + 'data/motion_energy/ls*.txt')
for i in range(nr_mov):
    lsp_txt = np.genfromtxt(lsp_txt_files[i], delimiter=',')
    egy_lsp.append(lsp_txt)
# %% Calculate average value per movie
egy_dnc_med, egy_lsp_med = [], []
egy_dnc_mean, egy_lsp_mean = [], []
for i in range(nr_dnc_mov):
    egy_dnc_med.append(np.median(egy_dnc[i]))
    egy_lsp_med.append(np.median(egy_lsp[i]))
    egy_dnc_mean.append(np.mean(egy_dnc[i]))
    egy_lsp_mean.append(np.mean(egy_lsp[i]))

ttest_ind(egy_dnc_med, egy_lsp_med)
ttest_ind(egy_dnc_mean,egy_lsp_mean)
# not t test but kolmogorox smirnov?
ks_2samp(egy_dnc_med, egy_lsp_med)
x = ks_2samp(egy_dnc_mean, egy_lsp_mean) # this is what is reported in the ms
