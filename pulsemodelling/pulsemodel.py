# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:09:06 2014

@author: cpkmanchee
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys
import inspect



class Pulse:

    T_BIT_DEFAULT = 12      #default time resolution
    T_WIN_DEFAULT = 20E-12  #default window size

    def __init__(self):
        self.time = None
        self.freq = None
        self.At = None
        self.Af = None


    def initializeGrid(self, t_bit_res, t_window):
        nt = 2**t_bit_res    #number of time steps, power of 2 for FFT
        dtau = 2*t_window/nt    #time step size

        self.time = dtau*np.arange(-nt//2, nt//2)       #time array
        self.freq = 2*np.pi*np.fft.fftfreq(nt,dtau)     #frequency array




def propagate(tau, inputField, lengthFib, alpha, gamma, beta):
    '''This function will propagate the input field along the length of...
    a fibre with the given properties...
    tau = time points...
    inputField = time domain input field...
    lengthFib = fibre length...
    alpha = fibre loss parameter...
    gamma = nonlinear parameter...
    beta = dispersion parameters'''
    function_name = sys._getframe().f_code.co_name   
    
    #Check inputs
    inputArgs = inspect.getargspec(eval(function_name)).args
    inputTypes = ['np.ndarray','np.ndarray','float','float','float','']
    if np.size(beta) == 1:
        inputTypes[5] = 'float'
    else:
        inputTypes[5] = 'np.ndarray'

    err = []
    for i in range(len(inputArgs)):
        err.append(checkInput(eval(str(inputArgs[i])),inputTypes[i],int(i+1)))
        

    if err[1:]==err[:-1]:
        #if no errors, check lengths of tau&field
        if len(tau)==len(inputField):
            pass    #all good continue with propagation
        else:
            print('Error: inputField and tau are not the same length')
            return(-1)
    else:
        for i in range(len(err)):
            if err[i] != -1:
                print('Error in function \'' + function_name + '\', Input ' + inputArgs[i]
                + '\n' + err[i])
                
        return(-1)  #if errors, stop 'propagation', return error value -1
        
    #Define dispersion parameters - doesn't need to be done
#    for i in range(len(beta)):
#        name = 'beta' + str(i+2)
#        exec( name + '=' + str(beta[i]))
#        print(str(name) + ' is ' + str(eval(name)))
        
    #Define Grid
    nt = len(tau)   #number of time points
    Tmax = np.max(tau) - np.min(tau)   #window size (time, ps)
    dtau = 2*Tmax/nt    #time step size    
    omega = 2*np.pi*np.fft.fftfreq(nt,dtau)    #frequency array
    
    L = lengthFib
    nz = int(100*L)   #number of spacial steps along fibre z-axis, one every cm
    dz = L/nz   #position step size
    z = dz*np.arange(0, nz)    #position array
    
#    N = np.sqrt(gamma*max(inputField)/(np.abs(beta2)))     #soliton order
    
    #Dispersion operator
    D = (-alpha/2)*dz
    for i in range(len(beta)):
        D += (1j*beta[i]*omega**(i+2)/np.math.factorial(i+2))*dz
    
    #Nonlinear opterator
    N = 1j*gamma*dz
    
    At = inputField*np.exp(np.abs(inputField)**2*N/2)
    count = 0
    for i in range(1,nz):
       
       Af = np.fft.ifft(At)
       Af = Af*np.exp(D)
       At = np.fft.fft(Af)
       At = At*np.exp(N*np.abs(At)**2)
       count += 1
    
    #print('Loop executed {} times' .format(count))
    Af = np.fft.ifft(At)
    Af = Af*np.exp(D)
    At = np.fft.fft(Af)
    outputField = At*np.exp(np.abs(At)**2*N/2)
    
    return(outputField)
    
    
#%%
def checkInput(inputData, requiredType, *inputNum):
    
    if len(inputNum)==1:
        number = inputNum[0]
    else:
        number = '#'

    if not(isinstance(inputData, eval(requiredType))):
        errMsg = 'Input ' + str(number) + ' is type ' + str(type(inputData)) + '\nRequired:' + ' \'' + str(requiredType) + '\'\n' 
    else:
        errMsg = -1
    
    return(errMsg)

#%%
def rmswidth(x,F):
    
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.asarray(x)
    
    if isinstance(F, np.ndarray):
        pass
    else:
        F = np.asarray(F)
        
    dx = [x[i]-x[i-1] for i in range(1,len(x))]
    dx = np.append(dx,dx[-1])
    
    #Normalization integration
    areaF=0
    for i in range(len(x)):
        areaF += dx[i]*F[i]

    #Average value
    mu=0
    for i in range(len(x)):
        mu += x[i]*F[i]*dx[i]/areaF

    #Varience (sd = sqrt(var))
    var = 0
    for i in range(len(x)):
        var += dx[i]*F[i]*(x[i]-mu)**2/areaF
    
    #returns avg and rms width
    return(mu, np.sqrt(var))


#%%
plt.ion()                            # Turned on Matplotlib's interactive mode
#

# Pulse propagation via Nonlinear Schrodinger Equation (NLSE)
# dA/dz = -ib2/2 (d^2A/dtau^2) + b3/6 (d^3 A/dtau^3) -aplha/2 + ig|A|^2*A  
# --> A is field A = sqrt(P0)*u


#Fibre parameters
L = 50.0;     #length in m
beta2 = 0.2  #GVD parameter, ps^2/m
beta3 = 0.0    #TOD, ps^3/m
alpha = 0.001   #loss (gain), 1/m
gamma = 0.003   #nonlinear parameter, 1/(W*m)


#Grid Initialization
nt_order = 11   #exponent for number of time steps, nt = 2^nt_order
Tmax = 20.0   #window size (time, ps)
nz = 1000   #number of spacial steps along fibre z-axis

nt = 2**nt_order    #number of time steps, power of 2 for FFT
dtau = 2*Tmax/nt    #time step size
dz = L/nz   #position step size
tau = dtau*np.arange(-nt//2, nt//2) #time array
omega = 2*np.pi*np.fft.fftfreq(nt,dtau)    #frequency array
z = dz*np.arange(0, nz)    #position array
#At = np.zeros((nz,nt))
#Af = np.zeros((nz,nt))


#Initial Field Parameters
mshape = 1  #1=Gaussian, >1=super Gaussian
chirp0 = 0  #initial chirp
T0 = 1  #initial pulse width
P0 = 6.667  #power, W

At = (sp.exp(-(1/(2*T0**2))*(1+1j*chirp0)*tau**(2*mshape)))  #initial (time) field
Af = np.fft.ifft(At)    #initial (freq) field

#Plotting
fieldPlot, (t_ax, f_ax) = plt.subplots(2)   #set up plot figure
fieldPlot.suptitle('Pulse propagation profile')

#normalize and scale to power
Atplot = sp.sqrt(P0)*At
Afplot = sp.sqrt(P0)*(2*Tmax/(sp.sqrt(2*np.pi)))*Af

t_input, = t_ax.plot(tau,np.abs(Atplot)**2, 'b--')    #plot time profile
t_ax.set_xlim([-5,5])
t_ax.set_xlabel('Time (ps)')
f_ax.plot(np.fft.fftshift(omega)/(2*np.pi),np.fft.fftshift(np.abs(Afplot)**2), 'b--')  #plot freq profile
f_ax.set_xlim(-0.5,0.5)
f_ax.set_xlabel('Frequency shift (THz)')

#Pulse Stats
[pulseCenter0, pulseWidth0] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter0, pulseWidth0)

#Propagation
beta = np.array([beta2, beta3])
At = propagate(tau, At, L, alpha, gamma, beta)

#Plotting
#normalize and scale to power
Af = np.fft.ifft(At)
Atplot = sp.sqrt(P0)*At
Afplot = sp.sqrt(P0)*(2*Tmax/(sp.sqrt(2*np.pi)))*Af

t_output, = t_ax.plot(tau,np.abs(Atplot)**2, 'b-')    #plot time profile
t_ax.set_xlim([-5,5])
f_ax.plot(np.fft.fftshift(omega)/(2*np.pi),np.fft.fftshift(np.abs(Afplot)**2), 'b-')  #plot freq profile
f_ax.set_xlim(-0.5,0.5)

plt.figlegend((t_input,t_output), ('Input', 'Output'), 'center right')

#Pulse stats
[pulseCenter, pulseWidth] = rmswidth(tau, np.abs(Atplot)**2)
print(pulseCenter, pulseWidth)
