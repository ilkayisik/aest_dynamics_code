#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:32:31 2019

@author: ilkay.isik
distance functions
"""

import numpy as np

def l1_norm(ts1, ts2):
    """
    L1_norm (also known as least absolute deviations (LAD) or least absolute 
    errors (LAE)) minimizes the sum of absolute differences between test and 
    retest timeseries.
    
    (After calculating the LAD value we scale it by dividing it to the number of 
    time points.
    After that by multiplying this value with -1 and adding 1 we make sure that 
    the bigger the value the higher the similarity like a correlation value.)
    
    ts1: test time-series

    ts2: retest time-series
    """
    l1_val = np.sum(np.abs(ts1 - ts2))

    return l1_val



def l2_norm(ts1, ts2):
    """
    L2_norm (also know as Least Squares Error (LSE)) minimizes the sum of the
    square of the distances between test and the retest time series.
    
    (After calculating the LSE value we scale it by dividing it to the number of 
    time points.
    After that by multiplying this value with -1 and adding 1 we make sure that 
    the bigger the value the higher the similarity like a correlation value.)
    
    ts1: test time-series

    ts2: retest time-series
    """
    
    l2_val = np.sqrt(np.sum(np.square(ts1 - ts2)))

    return l2_val



def lad(ts1, ts2):
    """compare the similarity of test and retest curves by taking the least
    absolute deviations between them and scaling it by dividing to the
    number of time points
    x1: test time-series

    x2: retest time-series
    """
    rel = (((np.sum(np.abs(ts1 - ts2))) / len(ts1)) * -1 ) + 1

    return rel