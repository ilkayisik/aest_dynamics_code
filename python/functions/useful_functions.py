#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def rmsd(timeseries):
    """Return the root mean squared difference calculated on the timepoints
       of a continuous rating time series"""

    x = np.sqrt(np.mean(np.square(np.diff(timeseries))))

    return x



def norm_data(value, newMax, newMin, Max, Min):
    """
    Scale the value between Max and Min to be between newMax and
    newMin
    """
    out = (newMax - newMin) / (Max - Min) * (value - Max) + newMax

    return out



# another way to normalize the data [Gives the same results as above]
def norm_data2(value, newMax, newMin, Max, Min):
    """
    Scale the value between Max and Min to be between newMax and
    newMin
    """
    out = (newMax - newMin) / (Max - Min) * (value - Min) + newMin

    return out




def norm_data_sc(value, newMax, newMin, Max, Min):
    """
    Scale the value between Max and Min to be between newMax and
    newMin
    """

    out = ((newMax - newMin) / (Max - Min) * (value - Max) + newMax) * -1

    return out




def norm_data_arr(data_arr, newMax, newMin):
    """
    Scale the values between Max and Min to be between newMax and
    newMin
    data_arr:  data matrix with the values to be scaled
    """
    Min, Max = np.min(data_arr), np.max(data_arr)

    # apply the following normalization to every element of this array
    new_arr = ((newMax - newMin) / (Max - Min) * (data_arr - Max) + newMax)

     # return the new array
    return new_arr



def mean_center(arr, axis):
    '''
    Mean centering based on subjects overall ratings

    arr: 2d array with subjest by item

    axis = 0 for row based calculation

    axis = 1 for column based calculation

    '''
    mean_centered_arr = np.zeros_like(arr)
    rows, cols = np.shape(arr)
    if axis == 0:
        for i in range(rows):
            mean_centered_arr[i, :] = arr[i, :] - np.mean(arr[i, :])
    elif axis == 1:
        for i in range(cols):
            mean_centered_arr[:, i] = arr[:, i] - np.mean(arr[:, i])
    return mean_centered_arr
