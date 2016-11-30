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

z_file = result_folder + '/' + start_date + '-' + start_time + 'zgrid' + '-' + str(dataset_num).zfill(2) + '.txt'
f = open(z_file,'a')
f.write(result_folder + '\n' + start_date + ' ' + start_time + '\n')
f.close()


#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s


#Define Pulse Object
pulse = pm.Pulse(1.03E-6)
pulse.initializeGrid(18, 1.5E-9)
T0 = 1000E-15
mshape = 1
chirp0 = 0
P_peak = 1E3   #peak power, 10kW corresp. to 1ps pulse, 400mW avg, 40MHz - high end of act. oscillator
pulse.At = np.sqrt(P_peak)*(sp.exp(-(1/(2*T0**2))*(1+1j*chirp0)*pulse.time**(2*mshape)))


#Define fiber components
pm980 = pm.Fiber(4)
pm980.alpha = 0.000576
pm980.beta = np.array([0.023, 0.00007, 0])*(1E-12)**(np.array([2,3,4]))
pm980.gamma = 0.00045
pm980.core_d = 5.5E-6

#reduced core strtching fibre
rcf = pm.Fiber(5)
rcf.alpha = 0.001
rcf.beta = np.array([0.1096108, 0.000810048, 0])*(1E-12)**(np.array([2,3,4]))
rcf.gamma = 0.00045*(6.0/2.9)**2
rcf.core_d = 2.9E-6


fiber = rcf
pm.saveObj(fiber, filebase + 'fiber' + fileext)

_, t0 = pm.rmswidth(pulse.time, np.abs(pulse.At))
p0 = (np.abs(pulse.At)**2).max()
    
ld = t0**2/(np.abs(fiber.beta[0]))
ln = 1/(p0*fiber.gamma)
    
l_ref = 1/((1/ld)+(1/ln))

dz = l_ref/(np.exp((np.arange(30)-5)/5))
dz_out = np.zeros(np.shape(dz))

f = open(z_file,'a')
f.write('L_ref=' + str(l_ref) + '\npulseid,dz\n')
f.close()

for ind, z_grid in enumerate(dz):
    #loop over different z-grid
    fiber.initializeGrid(fiber.length, 'abs', z_grid)
    dz_out[ind] = np.abs(fiber.z[1]-fiber.z[0])
    output_pulse = pulse.copyPulse()
    output_pulse.At = pm.propagateFiber(pulse, fiber)

    ##Save pulse
    while not not glob.glob(filename):
        output_num = output_num + 1
        filename = filebase + 'pulse' + str(output_num).zfill(3) + fileext
    pm.saveObj(output_pulse,filename)
    
    f = open(z_file,'a')
    f.write(str(output_num).zfill(2) + ',' + str(dz_out[ind]) + '\n')
    f.close()
    ##



#Analyze results

files = glob.glob(filebase + 'pulse*.pkl')
rms = np.zeros(np.shape(files))
peak = np.zeros(np.shape(files))


for ind, f in enumerate(files):
    p = pm.loadObj(f)
    _, rms[ind] = pm.rmswidth(p.time,np.abs(p.At))
    peak[ind] = np.abs(p.At).max()

    

    

'''
##Plotting
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

#plot output
t_output, = t_ax.plot(tau,np.abs(pulse.At)**2, 'b-')    #plot time profile
t_ax.set_xlabel('Time (s)')
f_ax.plot(np.fft.fftshift(pulse.freq)/(2*np.pi),np.fft.fftshift(np.abs(pulse.getAf())**2), 'b-')  #plot freq profile
f_ax.set_xlabel('Frequency shift (Hz)')

plt.figlegend((t_input,t_output), ('Input', 'Output'), 'center right')

'''