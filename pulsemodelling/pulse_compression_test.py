# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:14:03 2016

@author: cpkmanchee

Pulse compression
"""


import numpy as np
import matplotlib.pyplot as plt
import pulsemodel as pm


input_pulse = '/Users/cpkmanchee/Documents/Code/Code Output/laserdynamics/pulsemodelling/laser_system_output/20161207laser_system-01/20161207-142518-01pulse007.pkl'

#grating parameters

N = 1500
AOI = 45

Lmin = 0.4
Lmax = 0.45
L = np.linspace(Lmin,Lmax,10)

p = pm.loadObj(input_pulse)

rms = np.zeros(np.shape(L))

for ind, l in enumerate(L):
    
    output = pm.gratingPair(p,l,N,AOI)
    _,rms[ind] = pm.rmswidth(p.time, np.abs(output)**2)
    
    
plt.plot(L,rms)
    
    




