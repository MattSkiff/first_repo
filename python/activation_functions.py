# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:28:22 2019
Heaviside activation function
@author: skiff
"""
from math import e
import numpy as np

def heaviside(v):
    v_np = np.asarray(v)
    v = v_np.sum()
    print((1 / (1 + e**-v)))
    return;
    
heaviside(2)
