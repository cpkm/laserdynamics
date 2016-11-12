# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:25:06 2014

@author: cpkmanchee

Notes:
This file requires pulsemodel.py to be run previously
This file uses the functions and classes defined in pulsemodel.py

Everything is in SI units: m,s,W,J etc.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys
import inspect

#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s


plt.ion()                            # Turned on Matplotlib's interactive mode
#

#Define Pulse Object
pulse = Pulse(1.030E-6)
pulse.initializeGrid(12, 20E-12)
T0 = 1E-12
mshape = 1
chirp0 = 0
P_peak = 10E3   #peak power, 10kW corresp. to 1ps pulse, 400mW avg, 40MHz - high end of act. oscillator
pulse.At = P_peak*(sp.exp(-(1/(2*T0**2))*(1+1j*chirp0)*pulse.time**(2*mshape)))


#Define fiber components
smf1 = Fiber(1.2, 'abs', 0.005)
smf1.alpha = 0.0001
smf1.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
smf1.gamma = 0.00045

smf2 = Fiber(0.3, 'abs', 0.005)
smf2.alpha = 0.0001
smf2.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
smf2.gamma = 0.00045

gf1 = FiberGain(0.6, 'abs', 0.005)
gf1.alpha = 0.0001
gf1.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
gf1.gamma = 0.00045


#Plotting
fieldPlot, (t_ax, f_ax) = plt.subplots(2)   #set up plot figure
fieldPlot.suptitle('Pulse propagation profile')

#normalize and scale to power
tau = pulse.time
omega = pulse.freq

t_input, = t_ax.plot(tau,np.abs(pulse.At)**2, 'b--')    #plot time profile
t_ax.set_xlabel('Time (s)')
f_ax.plot(np.fft.fftshift(pulse.freq)/(2*np.pi),np.fft.fftshift(np.abs(pulse.getAf())**2), 'b--')  #plot freq profile
f_ax.set_xlabel('Frequency shift (Hz)')

'''
#Pulse Stats
[pulseCenter0, pulseWidth0] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter0, pulseWidth0)
'''
'''
#Propagation
At = propagateFiber(pulse,smf1)
'''

pulse.At = powerTap(pulse, 0.3)


#Plotting
#normalize and scale to power
t_output, = t_ax.plot(tau,np.abs(pulse.At)**2, 'b-')    #plot time profile
t_ax.set_xlabel('Time (s)')
f_ax.plot(np.fft.fftshift(pulse.freq)/(2*np.pi),np.fft.fftshift(np.abs(pulse.getAf())**2), 'b-')  #plot freq profile
f_ax.set_xlabel('Frequency shift (Hz)')

plt.figlegend((t_input,t_output), ('Input', 'Output'), 'center right')

'''
#Pulse stats
[pulseCenter, pulseWidth] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter, pulseWidth)
'''