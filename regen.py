# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:19:14 2016

@author: cpkmanchee
"""

'''
REGEN dynamics simulations
 See A.L Calendron thesis (pg27), Also:
 Svelto, Principles of lasers, Ch 7
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

dIp/dx = { Nt*s_ep*(n*(1+fp) + nt*(fs-fp))/(1+fs) } * Ip

dIs/dx = { Nt*s_es*n } * Is

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
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm, trange


class func:
    def __init__(self, value = None, index = None):
        self.val = value
        self.ind = index

    def at(self,x):
        return np.interp(x, self.ind, self.val)

    def diff(self):
        self.gradient = np.gradient(self.val)/np.gradient(self.ind)
            
    def diff_at(self,x):
        return np.interp(x,self.ind,self.gradient)
          


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



def dIp(z, Ip, n, wp):
    '''dIp/dx
    n must be a func instance (see class func:)
    wp must be a func instance (see class func:)
    '''
    return ((Nt*s_ep/(1+f_s))*(n.at(z)*(1+f_p) + nt*(f_s-f_p)))*Ip - (2*wp.diff_at(z)/(wp.at(z)))*Ip


def dFs(z, Fs, n):
    '''dFs/dx
    n must be a func instance (see class func:)
    '''
    return Nt*s_es*n.at(z)*Fs


def incTime_n(n, dt, Fs, Ip):
    a = s_ep*(f_p+1)*Ip/(h*v_p) + s_es*(f_s+1)*Fs/(dt*h*v_s) + 1/tau_se
    b = nt*(s_ep*(f_p-f_s)*Ip/(h*v_p) - f_s/tau_se)

    n_new = (n - b/a)*np.exp(-a*dt) + (b/a)

    return n_new

def incTime_Fs(Fs, dt):
    return Fs*np.exp(-alpha*dt/Tr)
 
def calcGain(z, n):
    
    gain_coeffs = Nt*s_es*n
    G = np.exp(Nt*s_es*integrate.trapz(n,z))
    
    return G, gain_coeffs
    
def waitbar(progress):
        sys.stdout.write("\rProgress: [{0:25s}] {1:.1f}%".format('#' * int(progress*25), progress*100))
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
xstal_L = 5E-3	#xstal length, m
eta = 0.01		#xstal doping
Nx = 6.3265E27	#host atom density, atoms/m**-3
Nt = eta*Nx		#dopant atom density, atoms/m**-3
nt = 1 			#n1+n2, total atom density ratio == 1

#Cavity parameters
d = 2.6			#total cavity length, m
alpha = 0.0066	#total cavity losses, fraction (0.05 = 5%)

#Crystal Spacial grid
dz = xstal_L/300	#in m
z = np.arange(0, xstal_L+dz, dz)
z_N = np.size(z)

#Beam parameters
wp0 = 310.0E-6  		#pump beam radius, m
zRp = 1.5E-3           #pump beam rayleigh parameter
z0p = z[np.int(z.size/2)] #focus of pump, midpoint of crystal
ws = 108.5E-6  		#signal beam radius, m
Pp_pump = 43.5373#50.7865#36.2881#58.0356#	#pump power (incident) in W
Es_seed = 1.0E-8 	#seed energy in J

wp = func(wp0*(1+((z-z0p)/zRp)**2)**(1/2),z) #pump beamwaist
wp.diff()

Ip_pump = Pp_pump/(np.pi*wp.val**2)    #pump intensity
Fs_seed = Es_seed/(np.pi*ws**2)    #seed fluence


#Temporal grid
Frep = 1E3 		#target rep rate
Ng = 100 		#number of round trips during amp
Ncyc = 2  		#number of full pulse cycles
R = 100          #pumping cycle time multiplier

dt = 2*d/c 		#roundtrip cavity time is the time step
Tr = dt   		#same as dt, just notation consistency
dT = R*dt        #time spacing for pumping segment


Tdest = 1/Frep
Tg = Ng*dt                  #gate time
Tp = ((Tdest-Tg)//dT)*dT    #pumping-only time
Np = np.int(Tp/dT)         #number of pumping calculations
Td = Tp + Tg
Nd = Np + Ng        #total calculations per cycle
Nsim = Ncyc*Nd      #total calculations

Tdest = 1/Frep    #estimated Td (cycle time), exact has to be calc. from round trips
Tg = Ng*dt        #gate (amplification) time
Tp = ((Tdest-Tg)//dT)*dT  #pump-only time
Np = np.int(Tp/dT) #number of pump timesteps
Td = Tp + Tg      #full cyclte time
Nd = Np + Ng        #total calculations per cycle
Nsim = Ncyc*Nd    #total simulation steps

tcyc = np.concatenate([np.linspace(0,Tp,Np, endpoint = False), Tp+np.linspace(0,Tg,Ng, endpoint = False)])
t = []
for i in range(Ncyc):
    t = np.concatenate([t,(i+1)*Td+tcyc])

t_N = np.size(t)



#Output variables
n_out = np.zeros((z_N,t_N))
Ip_out = np.zeros((z_N,t_N))
Fs_out = np.zeros((z_N,t_N))
gainCoef_out = np.zeros((z_N,t_N))
G_out = np.zeros(np.shape(t))

Ip_cur = np.zeros(np.shape(z))
Fs_cur = np.zeros(np.shape(z))

#crystal inversion, n
n = func()
n.val = -f_s*nt*np.ones(np.shape(z))
n.ind = z

n_out[:,0] = n.val
G, gain_coeffs = calcGain(z, n.val)
G_out[0] = G
gainCoef_out[:,0] = gain_coeffs
    

#start of main loop
bar=trange(Ncyc)
for k in range(Ncyc):#tqdm(range(Ncyc), desc='Overall'):

    #low Q (pump only) phase    
    Ip_0 = Ip_pump[0]
    Fs_0 = 0

    for j in tqdm(range(Np), desc='Low Q', leave=False, mininterval=1):
        m = j + k*Nd
        
        #calculate pump/signal power
        Ip_cur = rk4(dIp, z, Ip_0, [n,wp])
        Fs_cur = rk4(dFs, z, Fs_0, [n])

        #calculate inversion
        n.val = incTime_n(n.val, dT, Fs_cur, Ip_cur)
    	
        #set updated initial values
        Ip_out[:,m] = Ip_cur
        Fs_out[:,m] = Fs_cur
        n_out[:,m+1] = n.val
        
        #Gain calculations (just for output, does not affect sim)
        G, gain_coeffs = calcGain(z, n.val)
        
        G_out[m+1] = G
        gainCoef_out[:,m+1] = gain_coeffs

            
    #high Q (amplification) phase            
    Fs_0 = Fs_seed

    for i in tqdm(range(Ng), desc='High Q',leave=False,mininterval=1):

        q = i + Np + k*Nd
        
        #calculate pump/signal power
        Ip_cur = rk4(dIp, z, Ip_0, [n,wp])
        Fs_cur = rk4(dFs, z, Fs_0, [n])

        #increment 1/2 time step
        #calculate inversion
        n.val = incTime_n(n.val, dt/2, Fs_cur, Ip_cur)
        #apply cavity loss
        Fs_cur = incTime_Fs(Fs_cur, dt/2)
        
        #update intial condition
        Fs_0 = Fs_cur[-1]

        #calculate pump/seed
        Ip_cur = rk4(dIp, z, Ip_0, [n,wp])
        Fs_cur = np.flipud(rk4(dFs, np.flipud(z), Fs_0, [n], abs_x = True))

        #increment 1/2 time step
        n.val = incTime_n(n.val, dt/2, Fs_cur, Ip_cur)
        Fs_cur = incTime_Fs(Fs_cur, dt/2)
        
        Fs_0 = Fs_cur[0]    
        
        #store output values
        Ip_out[:,q] = Ip_cur
        Fs_out[:,q] = Fs_cur
        
        if q < np.size(n_out,1)-1:
            n_out[:,q+1] = n.val
        
        G, gain_coeffs = calcGain(z, n.val)
        
        G_out[q] = G
        gainCoef_out[:,q] = gain_coeffs
    
    bar.write('Finished cycle %i of %i' %(k+1,Ncyc))

#end of main loop


#outputs
cycle = np.arange(Ncyc)+1
Fs_peak = Fs_out[-1,cycle*Nd - 1]
Es_out = Fs_out[-1,]*(np.pi*ws**2)
Es_peak = Fs_peak*(np.pi*ws**2)




