# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 23:05:19 2016

@author: cpkmanchee
"""

'''

REGEN Dynamics
Dorring, Opt Exp, 2004

Simulation broken down into pumping time (lowQ) and amplification time (highQ)

 lowQ
 -solve for gain

 highQ
 -solve for pulse amplification
 -solve for gain depletion

 HighQ phase will included many round trips of the pulse in the cavity.


LowQ:

dg/dt = (g0-g)/tau

g2 = g0 + (g1-g0)*exp(-(Td-Tg)/tau))
 

HighQ:

dg/dt = (g0-g)/tau - (g*E)/(Esat*Tr)

dE/dt = E(g-l)/Tr


g0 = small signal gain
g1 = gain at begining of low phase
g2 = gain at begining of high phase
Td = dumping time = 1/f, f is rep rate
Tg = gate time = N*Tr, N is number of round trips
Tr = roundtrip time
Esat = saturation energy = hc/(lamba*(sigma_e+sigma_a))
tau = upper state lifetime

'''


import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys


def rk4(f, x, y0):
    '''
	functional form
	y'(x) = f(x,y)

	y can be array of expressions y = y1, y2,... yn

	f must be function, f(x,y), but can be function of functions
	x = array, x is differentiation variable
	y0 = initial condition

	returns y, integrated array
     '''

    n = np.size(y0)
    N = np.size(x)
    y = np.zeros((N,n))
    
    y[0,:] = y0

    dx = np.gradient(x)

    for i in range(N-1):

            k1 = f(x[i], y[i,:])
            k2 = f(x[i] + dx[i]/2, y[i,:] + k1*dx[i]/2)
            k3 = f(x[i] + dx[i]/2, y[i,:] + k2*dx[i]/2)
            k4 = f(x[i] + dx[i], y[i,:] + k3*dx[i])

            y[i+1,:] = y[i,:] + (k1 + 2*k2 + 2*k3 + k4)*dx[i]/6

    return y


def dy(t,y):

    return np.array([dgHigh(t,y),dEHigh(t,y)])

def g2Low(g1):
    
    return (g0 - (g0-g1)*np.exp(-(Td-Tg)/tau))

def dgHigh(t, y):
    
    g = y[0]
    E = y[1]

    return ((g0-g)/tau - g*E/(E_sat*Tr))

def dEHigh(t, y):
    
    g = y[0]
    E = y[1]

    return (E*(g-alpha)/Tr)
 
#constants
h = 6.62606957E-34	#J*s
c = 299792458.0		#m/s

#cross sections
s_ap = 1.2E-23;     #absorption pump, m^2
s_ep = 1.6E-23;     #emission pump, m^2
s_as = 5.0E-25;     #abs signal, m^2
s_es = 3.0E-24;     #emi signal, m^2

tau = 300E-6    #upper state lifetime, s

#wavelengths
l_p = 0.980E-6;         #pump wavelength, m
l_s = 1.035E-6;         #signal wavelength, m
v_p = c/l_p;            #pump freq, Hz
v_s = c/l_s;            #signal freq, Hz


d = 2.64         #cavity length, m
alpha = 0.05    #cavity losses
g0 = 0.2        #small signal gain
E_seed = 10E-8   #seed pulse energy in J
w_s = 200E-6    #seed spot size
E_sat = (np.pi*w_s**2)*h*c/(l_s*(s_es+s_as))   #saturation energy

frep = 2E3      #rep rate
Td = 1/(frep)   #dumping time

N = 200        # number of roundtrips
Tr = 2*d/c     # round trip time
Tg = N*Tr  	    #gate time, integer round trips


g_cur = 0
E_cur = E_seed


#Low-Q Phase
g_cur = g2Low(g_cur)

#High-Q Phase
t = np.linspace(0,Tg,N, endpoint = False)
y0 = np.array([g_cur,E_cur])
y_out = rk4(dy, t, y0)

plt.plot(t/Tr, y_out[:,1])