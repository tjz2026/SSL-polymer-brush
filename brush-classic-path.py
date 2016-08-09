#-*- coding: utf-8 -*-
"""
MultiBlock
=========
:copyright: (c) 2016 by Jiuzhou Tang
:license: BSD, see LICENSE for more details.
"""

import numpy as np
#from scipy import optimize
import scipy.io
import matplotlib.pyplot as pl
from numpy.linalg import inv
import scipy as sc
from scipy.integrate import simps
import time
import scipy.optimize # nonlinear solver
#from scipy import interpolate
from sys import exit

def PolymerBrush_SSL(D):
    global Nx
    global delta
    delta=1.0
    Nx=200  # the grid number for x
