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
Esat = saturation energy = hv/(lamba*(sigma_e+sigma_a))
tau = upper state lifetime

'''


import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys


def g2Low(g1):
    
    return (g0 + (g1-g0)*exp(-(Td-Tg)/tau))

#constants
h = 6.62606957E-34	#J*s
c = 299792458.0		#m/s

d = 1.6         #cavity length, m
g0 = 1.5        #small signal gain
tau = 300E-6    #upper state lifetime, s
frep = 1E3      #rep rate
Td = 1/(frep)   #dumping time

N = 20          # number of roundtrips
Tr = 2*d/c      # round trip time