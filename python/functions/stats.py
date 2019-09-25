#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Custom functions useful when analyzing data.

"""
# %% ----------------------------------------------
from scipy import stats
import numpy as np


def ztor(z):
    """Convert z values to r(correlation).

    Parameters
    ----------
    z : float, np.array, 1D

    Returns
    -------
    r : float, np.array, 1D

    """
    r = (np.exp(2. * z) - 1.) / (np.exp(2. * z) + 1.)
    return r


# %%
 

def rtoz(r):
    """Convert r(correlation) values to z.

    Parameters
    ----------
    r : float, np.array, 1D

    Returns
    -------
    z : float, np.array, 1D

    """
    z = 0.5 * np.log((1. + r) / (1. - r));
    return z

# %%


def rhotoz(r, n):
    """Convert r(correlation) values to z.

    Parameters
    ----------
    rho : float, np.array, 1D
    n : int (sample size)

    Returns
    -------
    z : float, np.array, 1D

    """
    z = (np.sqrt((n - 3) / 1.06)) * (0.5 * np.log((1. + r) / (1. - r)))
    return z

# %%


def compute_mm1(data):
    """Compute mean-minus-1 (MM1).

    Parameters
    ----------
    data : 2D np.array, [nr. of stimuli, nr. of subjects] or [nr tp, nr sub]

    Returns
    -------
    mm1 : m x n matrix of mean responses of all subjects except one

    corr : correlation of each judge's responses with its corresponding
    mm1 vector

    mean : mean of the correlations, computed as a z-score and
           transformed back to an r-value

    ci : 95% confidence interval of mean, via rtoz/ztor

    ttest : results of 1-way, 1-tailed t-test comparison to 0, on rtoz data
            CI's and sd in this structure are still in z!
    cohD : Cohen's D effect size, computed on rtoz data

    """
    [nr_stim, nr_subj] = data.shape
    mm1 = np.zeros([nr_stim, nr_subj])
    corr = np.zeros(nr_subj)

    for s in range(nr_subj):
        # Mean responses of all subjects except one
        mm1[:, s] = np.mean(data[:, np.setdiff1d(range(nr_subj), s)], axis=1)
        temp_corr = np.corrcoef(mm1[:, s], data[:, s])
        corr[s] = temp_corr[0, 1]

    mean = ztor(np.nanmean(rtoz(corr)))  # converted back to r
    # To get the one sided p-value, consider dividing the p value by 2
    [ttest_t, ttest_p] = stats.ttest_1samp(rtoz(corr), 0)  # results in z
    sd = np.nanstd(rtoz(corr))
    cohD = np.nanmean(rtoz(corr)) / sd  # both are in z

    ci = np.zeros(2)
    ci[0] = ztor(rtoz(mean) - 1.96 * (sd) / np.sqrt(nr_subj))
    ci[1] = ztor(rtoz(mean) + 1.96 * (sd) / np.sqrt(nr_subj))

    return mm1, corr, mean, ci, ttest_t, ttest_p, cohD

# %%
    
from ilkay_tools.lad import lad

def compute_mm1_lad(data):
    """Compute mean-minus-1 (MM1).

    Parameters
    ----------
    data : 2D np.array, [nr. of stimuli, nr. of subjects] or [nr tp, nr sub]

    Returns
    -------
    mm1 : m x n matrix of mean responses of all subjects except one

    corr : correlation of each judge's responses with its corresponding
    mm1 vector

    mean : mean of the correlations, computed as a z-score and
           transformed back to an r-value

    ci : 95% confidence interval of mean, via rtoz/ztor

    ttest : results of 1-way, 1-tailed t-test comparison to 0, on rtoz data
            CI's and sd in this structure are still in z!
    cohD : Cohen's D effect size, computed on rtoz data

    """
    [nr_tp, nr_subj] = data.shape
    mm1 = np.zeros([nr_tp, nr_subj])
    lad_vals = np.zeros(nr_subj)

    for s in range(nr_subj):
        # Mean responses of all subjects except one
        mm1[:, s] = np.mean(data[:, np.setdiff1d(range(nr_subj), s)], axis=1)
        temp_lad = lad(mm1[:, s], data[:, s])
        lad_vals[s] = temp_lad

    mean = np.nanmean(lad_vals)  # converted back to r
    # To get the one sided p-value, consider dividing the p value by 2
    [ttest_t, ttest_p] = stats.ttest_1samp(lad_vals, 0)  # results in z
    sd = np.nanstd(lad_vals)
    cohD = np.nanmean(lad_vals) / sd  # both are in z

    ci = np.zeros(2)
    ci[0] = mean - 1.96 * (sd) / np.sqrt(nr_subj)
    ci[1] = mean + 1.96 * (sd) / np.sqrt(nr_subj)

    return mm1, lad_vals, mean, ci, ttest_t, ttest_p, cohD

