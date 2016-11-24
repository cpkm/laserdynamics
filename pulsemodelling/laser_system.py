# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:18:13 2016

@author: cpkmanchee

Simulate pulse propagation through laser system

Schematic:

1. Oscillator output, 100fs TL pulse, sec2 or gaus shape
2. Stretching fiber
    a. PM1a, ~few meters
    b. RCF, ~100m
    c. PM1b, few meters
    d. return back through c-a
3. PM2, pm fibre after strecher, ~few m
4. GF1, preamp 1, 1m
5. PM3, pm fiber between gain, ~few m
6. GF2, preamp2, same as GF1
7. LCA, large core fiber amp, double clad, 2m


Notes:
This file requires pulsemodel.py
This file uses the functions and classes defined in pulsemodel.py (used via import)

Everything is in SI units: m,s,W,J etc.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pulsemodel as pm

import sys
import inspect


#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s



#Define Pulse Object
pulse = pm.Pulse(1.03E-6)
pulse.initializeGrid(18, 1.5E-9)
T0 = 100E-15
mshape = 1
chirp0 = 0
P_peak = 10E0   #peak power, 10kW corresp. to 1ps pulse, 400mW avg, 40MHz - high end of act. oscillator
pulse.At = np.sqrt(P_peak)*(sp.exp(-(1/(2*T0**2))*(1+1j*chirp0)*pulse.time**(2*mshape)))


#Define fiber components
pm980 = pm.Fiber(5)
pm980.alpha = 0.000576
pm980.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
pm980.gamma = 0.00045
pm980.core_d = 5.5E-6

rcf = pm.Fiber(100)
rcf.alpha = 0.001
rcf.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
rcf.gamma = 0.00045
rcf.core_d = 2.4E-6

gf1 = pm.FiberGain(0.6)
gf1.alpha = 0.000576
gf1.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
gf1.gamma = 0.00045
gf1.sigma_a = np.array([0.93124465,0.06369027])*1E-24
gf1.sigma_e = np.array([1.18207964,0.64375704])*1E-24
gf1.lambdas = np.array([0.976,1.030])*1E-6
gf1.core_d = 5.5E-6

lcfa = pm.FiberGain(2)
lcfa.alpha = 0.000576
lcfa.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
lcfa.gamma = 0.00045
lcfa.sigma_a = np.array([0.93124465,0.06369027])*1E-24
lcfa.sigma_e = np.array([1.18207964,0.64375704])*1E-24
lcfa.lambdas = np.array([0.976,1.030])*1E-6
lcfa.core_d = 30E-6
lcfa.clad_d = 250E-6
#need to check to scale sigma. I think just scale sigma_e by ratio of areas

#Pump parameters
pa1P = 0.6    #preamp1 pump power, CW
pa1F = 40E6    #rep. rate at preamp1

pa2P = 0.6    #preamp2 pump power, CW
pa2F = 500E3    #rep. rate at preamp2

lcaP = 25    #large core amp pump poer
lcaF = 500E3    #rep. rate at lca

Ip = pumpP/(np.pi*(gf1.core_d/2)**2)
Is = np.sum(np.abs(pulse.At)**2)*pulse.dt*Frep/(np.pi*(gf1.core_d/2)**2)
gf1.gain = pm.calcGain(gf1,Ip,Is) 

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

'''
#Pulse Stats
[pulseCenter0, pulseWidth0] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter0, pulseWidth0)
'''
'''
#Propagation
At = propagateFiber(pulse,smf1)
'''
#pulse.At = pm.propagateFiber(pulse, gf1)

t01, sig1 = pm.rmswidth(pulse.time,np.abs(pulse.At)**2)
output = pm.gratingPair(pulse, 1.0, 1500, 45)
t02, sig2 = pm.rmswidth(pulse.time,np.abs(output)**2)

input = pulse.At
pulse.At = output

#plot output
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

