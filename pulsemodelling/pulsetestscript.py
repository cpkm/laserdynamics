# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:25:06 2016

@author: cpkmanchee

Notes:
This file requires pulsemodel.py
This file uses the functions and classes defined in pulsemodel.py (used via import)

Everything is in SI units: m,s,W,J etc.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pulsemodel as pm

import os
import inspect

#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s


plt.ion()                            # Turned on Matplotlib's interactive mode
#

#Define Pulse Object
pulse = pm.Pulse(1.03E-6)
pulse.initializeGrid(12, 20E-12)
T0 = 100E-15
mshape = 1
chirp0 = 0
P_peak = 10E0   #peak power, 10kW corresp. to 1ps pulse, 400mW avg, 40MHz - high end of act. oscillator
pulse.At = np.sqrt(P_peak)*(sp.exp(-(1/(2*T0**2))*(1+1j*chirp0)*pulse.time**(2*mshape)))


#Define fiber components
smf1 = pm.Fiber(1.20, 'abs', 0.005)
smf1.alpha = 0.0001
smf1.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
smf1.gamma = 0.00045

smf2 = pm.Fiber(0.3, 'abs', 0.005)
smf2.alpha = 0.0001
smf2.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
smf2.gamma = 0.00045

gf1 = pm.FiberGain(0.6, 'abs', 0.005)
gf1.alpha = 0.0001
gf1.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
gf1.gamma = 0.00045
gf1.sigma_a = np.array([3.04306,0.04966])*1E-24
gf1.sigma_e = np.array([3.17025,0.59601])*1E-24
gf1.lambdas = np.array([0.976,1.030])*1E-6

#large core fiber amp. Nufern ydf-30/250-VIII
lcfa = pm.FiberGain(4)
lcfa.alpha = 0.00345
lcfa.beta = np.array([0.0251222977, 4.5522276126132602e-05, -5.0542788517531417e-08])*(1E-12)**(np.array([2,3,4]))
lcfa.gamma = 0.000251
lcfa.lambdas = np.array([0.976,1.030])*1E-6
lcfa.core_d = 30E-6
lcfa.clad_d = 250E-6
lcfa.N = 2.567986E25
lcfa.sigma_a = np.array([3.04306,0.04966])*1E-24
lcfa.sigma_e = np.array([3.17025,0.59601])*1E-24

lcaP = 25    #large core amp pump poer
lcaF = 500E3    #rep. rate at lca

pumpP = 0.6
Frep = 40E6

Ip = pumpP/(np.pi*(gf1.core_d/2)**2)
Is = np.sum(np.abs(pulse.At)**2)*pulse.dt*Frep/(np.pi*(gf1.core_d/2)**2)
#gf1.gain = pm.calcGain(gf1,Ip,Is) 
'''
Ps = 0.0005 #np.sum(np.abs(pulse.At)**2)*pulse.dt*lcaF
g0 = pm.calcGain(lcfa,lcaP,Ps,'clad')
g1 = pm.calcGain(lcfa,lcaP,Ps,'clad', 'b', 'rk4')
g2 = pm.calcGain(lcfa,lcaP,Ps,'clad', 'f', 'rk4')
'''
'''
#Plotting
tau = pulse.time
omega = pulse.freq

#create plot figure
fieldPlot, (t_ax, f_ax) = plt.subplots(2)   #set up plot figure
fieldPlot.suptitle('Pulse propagation profile')

#plot input
t_input, = t_ax.plot(tau,np.abs(pulse.At)**2, 'b--')    #plot time profile
t_ax.set_xlabel('Time (s)')
f_ax.plot(np.fft.fftshift(pulse.freq)/(2*np.pi),np.fft.fftshift(np.abs(pulse.getAf())**2), 'b--')  #plot freq profile
f_ax.set_xlabel('Frequency shift (Hz)')


#Pulse Stats
[pulseCenter0, pulseWidth0] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter0, pulseWidth0)

#Propagation
At = propagateFiber(pulse,smf1)

#pulse.At = pm.propagateFiber(pulse, gf1)

t01, sig1 = pm.rmswidth(pulse.time,np.abs(pulse.At)**2)
output = pm.gratingPair(pulse, 0.05, 600, 22)
t02, sig2 = pm.rmswidth(pulse.time,np.abs(output)**2)

#input = pulse.At
#pulse.At = output

#plot output
t_output, = t_ax.plot(tau,np.abs(pulse.At)**2, 'b-')    #plot time profile
t_ax.set_xlabel('Time (s)')
f_ax.plot(np.fft.fftshift(pulse.freq)/(2*np.pi),np.fft.fftshift(np.abs(pulse.getAf())**2), 'b-')  #plot freq profile
f_ax.set_xlabel('Frequency shift (Hz)')

plt.figlegend((t_input,t_output), ('Input', 'Output'), 'center right')


#Pulse stats
[pulseCenter, pulseWidth] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter, pulseWidth)
'''