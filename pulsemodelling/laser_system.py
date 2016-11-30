# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:18:13 2016

@author: cpkmanchee

Simulate pulse propagation through laser system

Schematic:

1. Oscillator output, 100fs TL pulse, sec2 or gaus shape
2. Stretching fiber
    a. PM1a, 32.8m
    b. RCF, 144.5m
    c. PM1b, negligible
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
pm980 = pm.Fiber(32.8+2)
pm980.alpha = 0.000576
pm980.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
pm980.gamma = 0.00045
pm980.core_d = 5.5E-6

#reduced core strtching fibre
rcf = pm.Fiber(144.9)
rcf.alpha = 0.001
rcf.beta = np.array([0.1096108, 0.000810048, 0])*(1E-12)**(np.array([2,3,4]))
rcf.gamma = 0.00045
rcf.core_d = 2.9E-6

#gain fiber, nufern ysf-HI
gf1 = pm.FiberGain(0.6)
gf1.alpha = 0.00345
gf1.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
gf1.gamma = 0.00045
gf1.sigma_a = np.array([0.93124465,0.06369027])*1E-24
gf1.sigma_e = np.array([1.18207964,0.64375704])*1E-24
gf1.lambdas = np.array([0.976,1.030])*1E-6
gf1.core_d = 6.0E-6
gf1.N = 6.1813931E25

#large core fiber amp. Nufern ydf-30/250-VIII
lcfa = pm.FiberGain(2)
lcfa.alpha = 0.00345
lcfa.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
lcfa.gamma = 0.00045
lcfa.lambdas = np.array([0.976,1.030])*1E-6
lcfa.core_d = 30E-6
lcfa.clad_d = 250E-6
lcfa.N = 8.7569736E25
X = (lcfa.core_d/lcfa.clad_d)**2
lcfa.sigma_a = np.array([0.93124465*X,0.06369027])*1E-24
lcfa.sigma_e = np.array([1.18207964*X,0.64375704])*1E-24

#passive large mode area, passive matched fiber to lcfa fiber. 30/250
plma = pm.Fiber(2)
plma.alpha = 0.00345
plma.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
plma.gamma = 0.00045
plma.core_d = 30E-6
plma.clad_d = 250E-6


#Pump parameters
pa1P = 0.6    #preamp1 pump power, CW
pa1F = 40E6    #rep. rate at preamp1

pa2P = 0.6    #preamp2 pump power, CW
pa2F = 500E3    #rep. rate at preamp2

lcaP = 25    #large core amp pump poer
lcaF = 500E3    #rep. rate at lca


#Propagation
#stretcher
pulse.At = pm.propagateFiber(pulse,pm980)
pulse.At = pm.propagateFiber(pulse,rcf)
pulse.At = pm.propagateFiber(pulse,rcf)
pulse.At = pm.propagateFiber(pulse,pm980)

#gain1
Ip = pa1P/(np.pi*(gf1.core_d/2)**2)
Is = np.sum(np.abs(pulse.At)**2)*pulse.dt*pa1F/(np.pi*(gf1.core_d/2)**2)
gf1.gain = pm.calcGain(gf1,Ip,Is)

pulse.At = pm.propagateFiber(pulse,gf1)

pm980.initializeGrid(2)
pulse.At = pm.propagateFiber(pulse, pm980)

#gain2
Ip = pa2P/(np.pi*(gf1.core_d/2)**2)
Is = np.sum(np.abs(pulse.At)**2)*pulse.dt*pa2F/(np.pi*(gf1.core_d/2)**2)
gf1.gain = pm.calcGain(gf1,Ip,Is)

pulse.At = pm.propagateFiber(pulse,gf1)

#largecore amp
Ip = lcaP/(np.pi*(lcfa.clad_d/2)**2)
Is = np.sum(np.abs(pulse.At)**2)*pulse.dt*lcaF/(np.pi*(lcfa.core_d/2)**2)
lcfa.gain = pm.calcGain(lcfa,Ip,Is)

pulse.At = pm.propagateFiber(pulse,plma)
pulse.At = pm.propagateFiber(pulse,lcfa)


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

