# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:19:14 2016

@author: cpkmanchee
"""

'''
REGEN dynamics simulations
 See A.L Calendron thesis (pg27), Also:
 Svelto, Principles of lasers, Ch 7
 Dorring, Opt Exp, 2004
 Pederson, J. Lightwave tech., 1991
 
 Simulation broken down into pumping time (lowQ) and amplification time (highQ)

 lowQ
 -solve for population inversion

 highQ
 -solve for pulse amplification
 -solve for population depletion
 -apply cavity losses

 HighQ phase will included many round trips of the pulse in the cavity.
 Each round trip will be one time step.


See Derivations Notebook, for Quasi 2-level system

dn/dt = s_ep[nt*(fp-fs) - n*(fp+1)]*Ip/hvp
		+ s_es(1+fs)*n*Is/hvs
		- (n+fs*nt)/tau

dIp/dx = { Nt*s_ep*(n*(1-fp) + nt*(fs-fp))/(1+fs) } *Ip

dIs/dx = { Nt*s_es*n } *Is

h = Planck's const.
vi = frequency (nu) - signal (s), pump (p)
tau = upper laser level lifetime

s_ij = crossection_emmission/abs, pump/signal
fi = s_ai/s_ei		ratio of abs/emi crosssections for signal/pump

n1, n2 = population (ratio) of lower and upper  laser levels, respectively
n1 + n2 = nt = 1

Ni = Nt*ni 			real density of upper/lower states
Nt*n1 + Nt*n2 = Nt = total number density of dopant atoms

n = n2 - fs*n1		represents the system inversion (ratio)
N = Nt*n 			inversion, real density

 '''

import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys


class func:
	def __init__(self, value = None, x = None):
	    self.val = value
	    self.ind = x

	def at(self,x):
		return np.interp(x, self.ind, self.val)


class Xstal:
 	'''
	to be used later maybe

 	creates a crystal object
 	length in mm
 	doping as ratio (3% = 0.03)
 	name is optional

 	z to be array of position along opt. axis
 	n2 to be array (size = len(z) ) of upperstate population ratio
 	'''

 	def __init__(self, length, doping, name = None):
 		self.length = length
 		self.doping = doping
 		self.name = name
 		self.n2 = None
 		self.z = None

 	def createz(self, dz):
 		self.z = np.arange(0,self.length,dz)
 		self.n2 = np.zeros(np.shape(self.z))


def dn(t, Ip, Is, n):
	'''insert dn/dt = 
	In this case, result should be an array, same length as Ip and Is
	'''
	result = (s_ep*(nt*(f_p-f_s) - n*(f_p+1))*Ip/(h*v_p) 
	         - s_es*(1+f_s)*n*Is/(h*v_s)
	         - (n+f_s*nt)/tau_se)
	return result

def dIp(z, Ip, n):
    '''dIp/dx
    n must be a func instance (see class func:)
    '''
    return ((Nt*s_ep/(1+f_s))*(n.at(z)*(1+f_p) + nt*(f_s-f_p)))*Ip

def dIs(z, Is, n):
    ''' dIs/dx 
    n must be a func instance (see class func:)
    '''
    return Nt*s_es*n.at(z)*Is

def dFs(z, Fs, n):
    '''dFs/dx
    n must be a func instance (see class func:)
    '''
    return Nt*s_es*n.at(z)*Fs


def incTime_n(n, dt, Is, Ip):
    a = s_ep*(f_p+1)*Ip/(h*v_p) + s_es*(f_s+1)*Is/(h*v_s) + 1/tau_se
    b = nt*(s_ep*(f_p-f_s)*Ip/(h*v_p) - f_s/tau_se)

    n_new = (n - b/a)*np.exp(-a*dt) + (b/a)

    return n_new

def incTime_Is(Is, dt):
    return Is*np.exp(-alpha*dt/Tr)
 
def calcGain(z, n):
    
    gain_coeffs = Nt*s_es*n
    G = np.exp(Nt*s_es*integrate.trapz(n,z))
    
    return G, gain_coeffs
    
def waitbar(progress):
        sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress*50), progress*100))
        sys.stdout.flush()
	

def rk4(f, x, y0, const_args = [], abs_x = False):
	'''
	functional form
	y'(x) = f(x,y,constants)

	f must be function, f(x,y,const_args)
	x = array
	y0 = initial condition,
	cont_args = additional constants required for f

	returns y, integrated array
	'''

	N = np.size(x)
	y = np.zeros(np.shape(x))
	y[0] = y0
	dx = np.gradient(x)

	if abs_x:
		dx = np.abs(dx)

	for i in range(N-1):
		k1 = f(x[i], y[i], *const_args)
		k2 = f(x[i] + dx[i]/2, y[i] + k1*dx[i]/2, *const_args)
		k3 = f(x[i] + dx[i]/2, y[i] + k2*dx[i]/2, *const_args)
		k4 = f(x[i] + dx[i], y[i] + k3*dx[i], *const_args)

		y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)*dx[i]/6

	return y


'''
def intIp(z, n, Ip):

	return

def intIs(z, n, Is0):

	int_n = sp.integrate.cumtrapz(n,z)
	Is = Is0*np.exp(Nt*s_es*int_n)
'''

#constants
h = 6.62606957E-34	#J*s
c = 299792458.0		#m/s

#cross sections
s_ap = 1.2E-23;     #absorption pump, m^2
s_ep = 1.6E-23;     #emission pump, m^2
s_as = 5.0E-25;     #abs signal, m^2
s_es = 3.0E-24;     #emi signal, m^2

tau_se = 0.3E-3;        #spontaneous emission lifetime, s

#wavelengths
l_p = 0.980E-6;         #pump wavelength, m
l_s = 1.035E-6;         #signal wavelength, m
v_p = c/l_p;            #pump freq, Hz
v_s = c/l_s;            #signal freq, Hz

#calculated constants
f_s = s_as/s_es
f_p = s_ap/s_ep

#parameters
xstal_L = 3.0E-3	#xstal length, m
eta = 0.03		#xstal doping
Nx = 6.3265E27	#host atom density, atoms/m**-3
Nt = eta*Nx		#dopant atom density, atoms/m**-3
nt = 1 			#n1+n2, total atom density ratio == 1

#Beam parameters
wp = 300.0E-6  		#pump beam radius, m
ws = 300.0E-6  		#signal beam radius, m
Pp_pump = 60.0  		#pump power (incident) in W
Ps_seed = 1.0E-2 		#seed power in W

Ip_pump = Pp_pump/(np.pi*wp**2)
Is_seed = Ps_seed/(np.pi*ws**2)

#Cavity parameters
d = 1.6			#total cavity length, m
alpha = 0.05	#total cavity losses, fraction (0.05 = 5%)

#Spacial grid
dz = xstal_L/300	#in m
z = np.arange(0, xstal_L+dz, dz)
z_N = np.size(z)

#Temporal grid
dt = 2*d/c 		#roundtrip cavity time is the time step, ~10-12ns
Tr = dt   		#same as dt, just notation consistency

Frep = 1E3 		#target rep rate
Ng = 100 		#number of round trips during amp
Ncyc = 1  		#number of cycles
Nd = np.int((1/Frep)/(Tr))
Np = Nd-Ng  	#number of rountrips during pumping phase

Td = Nd*Tr 		#1/Td is rep rate, needs to be integer of Tr
Tg = Ng*Tr 		#gate time
Tp = Td - Tg  	#pumping window
T_sim = Ncyc*Td #total simulation time window

t = np.linspace(0,T_sim,Nd)
t_N = np.size(t)

#Output variables
n_out = np.zeros((z_N,t_N))
Ip_out = np.zeros((z_N,t_N))
Is_out = np.zeros((z_N,t_N))
gainCoef_out = np.zeros((z_N,t_N))
G_out = np.zeros(np.shape(t))

Ip_cur = np.zeros(np.shape(z))
Is_cur = np.zeros(np.shape(z))

n = func()
n.val = -f_s*nt*np.ones(np.shape(z))
n.ind = z

n_out[:,0] = n.val

G, gain_coeffs = calcGain(z, n.val)
    
G_out[0] = G
gainCoef_out[:,0] = gain_coeffs
Ip_0 = Ip_pump
Is_0 = 0

for m in range(Np):

    k = m

    Ip_cur = rk4(dIp, z, Ip_0, [n])
    Is_cur = rk4(dIs, z, Is_0, [n])

    n.val = incTime_n(n.val, dt, Is_cur, Ip_cur)
	
    Ip_out[:,k] = Ip_cur
    Is_out[:,k] = Is_cur
    n_out[:,k+1] = n.val
    
    G, gain_coeffs = calcGain(z, n.val)
    
    G_out[k+1] = G
    gainCoef_out[:,k+1] = gain_coeffs
    
    if m%np.int(Nd/1000) == 0:
        waitbar(m/Nd)
        
        
Is_0 = Is_seed

for j in range(Ng):

    i = j + m + 1

    Ip_cur = rk4(dIp, z, Ip_0, [n])
    Is_cur = rk4(dIs, z, Is_0, [n])

    n.val = incTime_n(n.val, dt/2, Is_cur, Ip_cur)
    Is_cur = incTime_Is(Is_cur, dt/2)
    
    Is_0 = Is_cur[-1]

    Ip_cur = rk4(dIp, z, Ip_0, [n])
    Is_cur = np.flipud(rk4(dIs, np.flipud(z), Is_0, [n], abs_x = True))

    n.val = incTime_n(n.val, dt/2, Is_cur, Ip_cur)
    Is_cur = incTime_Is(Is_cur, dt/2)
    
    Is_0 = Is_cur[0]    
    
    Ip_out[:,i] = Ip_cur
    Is_out[:,i] = Is_cur
    
    if i < np.size(n_out,1)-1:
        n_out[:,i+1] = n.val
    
    G, gain_coeffs = calcGain(z, n.val)
    
    G_out[i] = G
    gainCoef_out[:,i] = gain_coeffs
    
    if i%np.int(Nd/1000) == 0:
        waitbar(i/Nd)










