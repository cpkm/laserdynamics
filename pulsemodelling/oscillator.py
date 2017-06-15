# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:13:47 2017

@author: cpkmanchee

Simulate pulse propagation in oscillator

Schematic:



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
import shutil
import glob
import os
from datetime import datetime


#create save name and folders for output
start_date = datetime.now().strftime("%Y%m%d")
start_time = datetime.now().strftime("%H%M%S")

#output_folder is outside of git repository in: code_folder/Code Output/...
#if file is in code_folder/X/Y/Z, results are in code_folder/Code Output/X/Y/Z
code_folder = '/Users/cpkmanchee/Documents/Code'
output_folder = code_folder + '/Code Output' + os.path.dirname(__file__).split(code_folder)[-1] + '/' + os.path.splitext(os.path.basename(__file__))[0] + '_output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

result_folder = output_folder + '/' + start_date + os.path.splitext(os.path.basename(__file__))[0]

dataset_num = 0
while not not glob.glob((result_folder + '-' + str(dataset_num).zfill(2) + '*')):
    dataset_num = dataset_num + 1
result_folder =  result_folder + '-' + str(dataset_num).zfill(2)
os.makedirs(result_folder)

filebase = result_folder + '/' + start_date + '-' + start_time + '-' + str(dataset_num).zfill(2)
fileext = '.pkl'
output_num = 0
filename =  filebase + 'pulse' + str(output_num).zfill(3) + fileext

shutil.copy(__file__, result_folder + '/' + os.path.basename(__file__))


def savepulse(pulse):
    '''
    to be used locally only
    all file/folder names must be previously defined
    '''
    global output_num, filename    
    
    while not not glob.glob(filename):
        output_num = output_num + 1
        filename = filebase + 'pulse' + str(output_num).zfill(3) + fileext
    pm.saveObj(pulse,filename)


def cavity(pulse):
    '''Define cavity round trip
    '''
    pulse.At = gratingPair(pulse, L, N, AOI, loss = ref_loss, return_coef = False)
    pulse.At = propagateFiber(pulse,smf1)




#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s
rt = 26.3E-9        #cavity round trip time

#Define grating parameters
L = 0.09
N = 600
AOI = 27
ref_loss = 1-(1-0.3)**4

#Define Pulse Object
pulse = pm.Pulse(1.03E-6)
pulse.initializeGrid(18, 1.5E-9)
T0 = 100E-15
mshape = 1
chirp0 = 0
P_peak = 10E0   #peak power, 10kW corresp. to 1ps pulse, 400mW avg, 40MHz - high end of act. oscillator
pulse.At = np.sqrt(P_peak)*(sp.exp(-(1/(2*T0**2))*(1+1j*chirp0)*pulse.time**(2*mshape)))


#Define fiber components
smf1 = pm.Fiber(1.0)
smf1.alpha = 0.000576
smf1.beta = np.array([0.0251222977, 4.5522276126132602e-05, -5.0542788517531417e-08])*(1E-12)**(np.array([2,3,4]))
smf1.gamma = 0.00045
smf1.core_d = 5.5E-6

smf2 = pm.Fiber(1.0)
smf2.alpha = 0.000576
smf2.beta = np.array([0.0251222977, 4.5522276126132602e-05, -5.0542788517531417e-08])*(1E-12)**(np.array([2,3,4]))
smf2.gamma = 0.00045
smf2.core_d = 5.5E-6

smf3 = pm.Fiber(1.0)
smf3.alpha = 0.000576
smf3.beta = np.array([0.0251222977, 4.5522276126132602e-05, -5.0542788517531417e-08])*(1E-12)**(np.array([2,3,4]))
smf3.gamma = 0.00045
smf3.core_d = 5.5E-6


#gain fiber, nufern ysf-HI
ydf1 = pm.FiberGain(0.6)
ydf1.alpha = 0.00345
ydf1.beta = np.array([0.0251222977, 4.5522276126132602e-05, -5.0542788517531417e-08])*(1E-12)**(np.array([2,3,4]))
ydf1.gamma = 0.00045
ydf1.sigma_a = np.array([3.04306,0.04966])*1E-24
ydf1.sigma_e = np.array([3.17025,0.59601])*1E-24
ydf1.lambdas = np.array([0.976,1.030])*1E-6
ydf1.core_d = 6.0E-6
ydf1.N = 1.891669E25

#Pump parameters
p1P = 0.7    #pump power, CW


#save initial pulse
savepulse(pulse)

#Propagation
#stretcher
pulse.At = pm.propagateFiber(pulse,pm980, True)
pulse.At = pm.propagateFiber(pulse,rcf, True)
savepulse(pulse)
pulse.At = pm.propagateFiber(pulse,rcf, True)
pulse.At = pm.propagateFiber(pulse,pm980, True)
savepulse(pulse)

#gain1
Ps = np.sum(np.abs(pulse.At)**2)*pulse.dt*pa1F
gf1.gain = pm.calcGain(gf1,pa1P,Ps)

pulse.At = pm.propagateFiber(pulse,gf1)
savepulse(pulse)

pm980.length = 2
pulse.At = pm.propagateFiber(pulse, pm980, True)
savepulse(pulse)

#gain2
Ps = np.sum(np.abs(pulse.At)**2)*pulse.dt*pa2F
gf1.gain = pm.calcGain(gf1,pa2P,Ps)

pulse.At = pm.propagateFiber(pulse,gf1)
savepulse(pulse)

#largecore amp
Ps = np.sum(np.abs(pulse.At)**2)*pulse.dt*lcaF
lcfa.gain = pm.calcGain(lcfa,lcaP,Ps,'clad', 'back', 'rk4')

pulse.At = pm.propagateFiber(pulse,plma,True)
savepulse(pulse)
pulse.At = pm.propagateFiber(pulse,lcfa)
savepulse(pulse)

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