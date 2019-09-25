#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:46:58 2019

@author: ilkay.isik
"""
import numpy as np
def conf_int(m, std, n, level=1.96):
    """Compute codnfidence intervals

    Parameters
    ----------
    mean : mean values
    
    std: standard deviation
    
    level: level of confidence interval 
           [95%: 1.96, 99%: 2.576, 99.5%:2.807, 99.9%:3.291]
    
    nr_sub: number of participants
    
    
    Returns
    -------
    ci : 2 by 1 matrix of -, + ci interval values

    """
    ci = np.zeros(2)
    ci[0] = m - (level * (std / np.sqrt(n)))
    ci[1] = m + (level * (std / np.sqrt(n)))
    
    return ci    