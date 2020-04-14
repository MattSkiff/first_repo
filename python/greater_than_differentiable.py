# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:33:11 2019

@author: user
"""
import math

def greater_approx(x,y,error = 0.000001):
    """
    https://math.stackexchange.com/questions/517482/approximating-a-maximum-function-by-a-differentiable-function
    using a differentiable approximation of modulus
    """
    greater = 0.5*(x+y+math.sqrt(((x-y)+error)**2))
    return greater

def greater_approx(x,y,error = 0.000001):
    """
    https://math.stackexchange.com/questions/517482/approximating-a-maximum-function-by-a-differentiable-function
    using a differentiable approximation of modulus
    """
    x-y
    return greater

import numpy as np

a = [-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0]
np.exp(a) / np.sum(np.exp(a)) 

def step(x):
  return 0 if x < 0 else 1
    