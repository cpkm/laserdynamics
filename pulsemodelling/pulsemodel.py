# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:02:23 2016

@author: cpkmanchee

Notes:

- classes of Pulse, Fiber, and FiberGain poses all the parameters required for th einput of most functions
- functions should not change class object parameters; instead they should return a value which can be used to 
change the object's parameters in the primary script 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys
import inspect


#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s


class Pulse:
    '''
    Defines a Pulse object
    .time = time array (s)
    .freq = corresponding angular freq array (rad/s)
    .At = time domain Field
    .Af = freq domain Field (redundant, removed)
    .lambda0 = central wavelength of pulse

    Note: At should be used as the primary field. Af should only be reference. 
    Any time field is modified it should be stored as At. Then use getAf() to get current freq domain field.

    '''

    T_BIT_DEFAULT = 12      #default time resolution, 2^12
    T_WIN_DEFAULT = 20E-12  #default window size, 20ps

    def __init__(self, lambda0 = 1.030E-6):
        self.time = None
        self.freq = None
        self.At = None
        self.lambda0 = lambda0

    def initializeGrid(self, t_bit_res, t_window):
        nt = 2**t_bit_res    #number of time steps, power of 2 for FFT
        dtau = 2*t_window/nt    #time step size

        self.time = dtau*np.arange(-nt//2, nt//2)       #time array
        self.freq = 2*np.pi*np.fft.fftfreq(nt,dtau)     #frequency array
        self.nt = nt
        self.dt = dtau

    def getAf(self):
        return ((self.dt*self.nt)/(sp.sqrt(2*np.pi)))*np.fft.ifft(self.At)

    def copyPulse(self, new_At = None):
        '''
        Duplicates pulse, outputs new pulse instance
        Can set new At at same time by sending new_At. If not sent, new_pulse.At is same
        '''

        new_pulse = Pulse(self.lambda0)
        new_pulse.time = self.time
        new_pulse.freq = self.freq
        new_pulse.nt = self.nt
        new_pulse.dt = self.dt

        if new_At == None:
            new_pulse.At = self.At
        else:
            new_pulse.At = new_At

        return new_pulse


class Fiber:
    '''
    Defines a Fiber object
    .length = length of fiber (m)
    .alpha = loss coefficient (m^-1), +alpha means loss
    .beta = dispersion parameters, 2nd 3rd 4th order. array
    .gamma = nonlinear parameter, (W*m)^-1\
    
    can be used for simple gain fiber by using alpha (-alpha = gain-loss)

    .core_d = core diameter
    .clad_d = cladding diameter

    .z is the z-axis array for the fiber

    grid_type specifies whether the z-grid is defined by the grid spacing ('abs' or absolute),
    or number of points ('rel' or relative)
    z_grid is either the grid spacing (abs) or number of grid points (rel)

    '''

    Z_STP_DEFAULT = 0.003  #default grid size, in m, 3mm
    Z_NUM_DEFAULT = 300     #default number of grid points, 300

    CORE_D_DEFAULT = 6E-6    #default core diameter, 6um
    CLAD_D_DEFAULT = 125E-6  #default clad diameter, 125um

    def __init__(self, length = 0, grid_type = 'abs', z_grid = None,  alpha = 0, beta = np.array([0,0,0]), gamma = 0):

        self.length = length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.core_d = self.CORE_D_DEFAULT
        self.clad_d = self.CLAD_D_DEFAULT

        self.initializeGrid(self.length, grid_type, z_grid)

    def initializeGrid(self, length, grid_type = 'abs', z_grid = None):
        '''
        -sets up the z-axis array for the fiber
        -can be called and re-called at any time (even after creation)
        -must provide fiber length, self.length is redefined when initializeGrid is called
        '''

        self.length = length

        if grid_type.lower() == 'abs':
            #grid type is 'absolute', z_grid is grid spacing
            if z_grid == None:
                z_grid = self.Z_STP_DEFAULT 

            nz = self.length//z_grid
            self.z = z_grid*np.arange(0, nz)    #position array

        else:
            # grid type is 'relative', z_grid is number of grid points
            if z_grid == None or z_grid < 1:
                z_grid = self.Z_NUM_DEFAULT

            dz = self.length/z_grid   #position step size
            self.z = dz*np.arange(0, z_grid)    #position array


class FiberGain:
    '''
    Defines a gain Fiber object with gain parameters
    .length = length of fiber (m)
    .alpha = loss coefficient (m^-1)
    .beta = dispersion parameters, 2nd 3rd 4th order. array
    .gamma = nonlinear parameter, (W*m)^-1\
    .gain = fiber gain coefficient (m^-1), same units as alpha, can be z-array or constant
    
    .core_d = core diameter
    .clad_d = cladding diameter

    .sigma_x are 2x2 arrays. col 0 = wavelength, col 1 = sigma, row 0 = pump, row 1= signal
    .tau is excited state lifetime
    .z is the z-axis array for the fiber

    grid_type specifies whether the z-grid is defined by the grid spacing ('abs' or absolute),
    or number of points ('rel' or relative)
    z_grid is either the grid spacing (abs) or number of grid points (rel)

    '''

    Z_STP_DEFAULT = 0.003  #default grid size, in m
    Z_NUM_DEFAULT = 300     #default number of grid points

    CORE_D_DEFAULT = 6E-6    #default core diameter, 6um
    CLAD_D_DEFAULT = 125E-6  #default clad diameter, 125um

    def __init__(self, length = 0, alpha = 0, beta = np.array([0,0,0]), gamma = 0, gain = 0, grid_type = 'abs', z_grid = None):

        self.length = length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gain = gain

        self.sigma_a = np.zeros((2,2))
        self.sigma_e = np.zeros((2,2))

        self.tau = 770E-6
        self.N = 7.1175E25

        self.core_d = self.CORE_D_DEFAULT
        self.clad_d = self.CLAD_D_DEFAULT

        self.initializeGrid(self.length, grid_type, z_grid)


    def initializeGrid(self, length, grid_type = 'abs', z_grid = None):
        '''
        -sets up the z-axis array for the fiber
        -can be called and re-called at any time (even after creation)
        -must provide fiber length, self.length is redefined when initializeGrid is called
        '''

        self.length = length

        if grid_type.lower() == 'abs':
            #grid type is 'absolute', z_grid is grid spacing
            if z_grid == None:
                z_grid = self.Z_STP_DEFAULT 

            nz = self.length//z_grid
            self.z = z_grid*np.arange(0, nz)    #position array

        else:
            # grid type is 'relative', z_grid is number of grid points
            if z_grid == None or z_grid < 1:
                z_grid = self.Z_NUM_DEFAULT

            dz = self.length/z_grid   #position step size
            self.z = dz*np.arange(0, z_grid)    #position array



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


def rmswidth(x,F):
    
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.asarray(x)
    
    if isinstance(F, np.ndarray):
        pass
    else:
        F = np.asarray(F)
        
    dx = np.gradient(x)
    
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


def calcGain(fiber, Ip, Is):
    '''
    Calculate steady state gain over fiber
    Output z-array of gain
    fiber.sigma_x are 2x2 arrays. col 0 = wavelength, col 1 = sigma, row 0 = pump, row 1= signal
    '''
    s_ap = fiber.sigma_a[0,1]
    s_as = fiber.sigma_a[1,1]

    s_ep = fiber.sigma_e[0,1]
    s_es = fiber.sigma_e[1,1]

    v_p = fiber.sigma_a[0,0]
    v_s = fiber.sigma_e[1,0]

    b_p = (s_ap + s_ep)/(h*v_p)
    b_s = (s_as + s_es)/(h*v_s)
    a_p = s_ap/(h*v_p)
    a_s = s_as/(h*v_s)

    tau_se = fiber.tau

    g = np.zeros(np.shape(fiber.z))
    N=fiber.N
    dz = np.gradient(fiber.z)

    for i in range(np.size(g)):

        n = (a_p*Ip + a_s*Is)/(b_p*Ip + b_s*Is + 1/tau_se)
        
        Ip = Ip*np.exp(-(s_ap*N*(1-n) - s_ep*N*n)*dz[i])
        Is = Is*np.exp(-(s_as*N*(1-n) - s_es*N*n)*dz[i]) + n*h*v_s*N*dz[i]/tau_se

        g[i] = (s_es*N*n - s_as*N*(1-n))

    return g


def gratingPair(pulse, L, d, theta, loss = 0):
    '''
    Simulate grating pair
    pulse = input pulse object
    L = grating separation (m)
    d = groove spacing (m) of gratings
    theta = angle of incidence (rad)
    loss = %loss of INTENSITY (not field)

    returns time-domain output field

    '''

    Af = pulse.getAf()
    w0 = 2*np.pi*c/pulse.lambda0
    w = pulse.freq + w0

    phi2 = (-2*2*(np.pi**2)*L*c/(d**2*w**3))*(1-(2*np.pi*c/(d*w) - np.sin(theta))**2)**(-3/2)
    phi3 = (12*(np.pi**2)*c*L/(d**2*w**4*(np.cos(theta))**3))*(1+((2*np.pi*c*np.sin(theta))/(w*d*(np.cos(theta))**2)))

    output_At = np.sqrt(1-loss)*np.fft.fft(Af*np.exp(-1j*(phi2*(w-w0)**2/2 + phi3*(w-w0)**3/6)))
    
    return output_At


def powerTap(pulse, tap, loss = 0):
    '''
    Simulate splitter or tap
    tap is 'output', 'signal' is to cavity. Just semantics though
    signal pulse is (1-tap)

    pulse = input pulse
    tap = tap ratio, ex. 1 == 1%, 50 = %50
    loss = % loss

    tap and loss ratios are of INTENSITY, not field

    Note: tap and signal are 'dephased', differ by factor of i. This is how these work in real life
    
    '''

    At = np.sqrt(1-loss)*pulse.At
    output_tap = 1j*np.sqrt(tap/100)*At
    output_signal = np.sqrt(1-tap/100)*At

    return output_signal, output_tap


def coupler2x2(pulse1, pulse2, tap, loss = 0):
    '''Simulates splitter/coupler
    requires 2 pulses, outputs 2 pulses.
    should be able to send 'null pulse' in out terminal to sim single input

    B(pulse2)-----[=======]-----SignalA, tapB
                  [==2x2==]
    A(pulse1)-----[=======]-----SignalB, tapA

    pulse1 goes to output_sig with (1-tap)
    pulse1 goes to output_tap with tap
    pulse2 goes to output_tap with tap
    pulse2 goes to output_sig with (1-tap)
    '''

    At1 = np.sqrt(1-loss)*pulse1.At
    At2 = np.sqrt(1-loss)*pulse2.At

    output_signal = np.sqrt(1-tap/100)*At1 + 1j*np.sqrt(tap/100)*At2
    output_tap = np.sqrt(1-tap/100)*At2 + 1j*np.sqrt(tap/100)*At1

    return output_signal, output_tap


def opticalFilter(pulse, filter_type, lambda0 = None, bandwidth = 2E-9, loss = 0):
    '''
    Simulate filter, bandpass, longpass, shortpass
    default bandwidth is 2nm

    pulse.lambda0 = central wavelength of PULSE
    lambda0 = central wavelength of FILTER
    w0 is central freq (ang) of FILTER
    '''
    
    Af = pulse.getAf()

    if lambda0 == None:
        lambda0 = pulse.lambda0

    w = pulse.freq + 2*np.pi*c/pulse.lambda0
    w0 = 2*np.pi*c/lambda0

    if filter_type.lower() == 'lpf':
        '''
        long-pass, pass low freq
        w0-w is (+) for w<w0 (pass region)
        '''
        filter_profile = 0.5 * (np.sign(w0-w) + 1)

    elif filter_type.lower() == 'spf':
        '''
        short-pass, pass high freq
        w-w0 is (+) for w>w0 (pass region)
        '''
        filter_profile = 0.5 * (np.sign(w-w0) + 1)

    elif filter_type.lower() == 'bpf':
        '''
        bandpass
        '''
        #prevent divide by 0
        if bandwidth == 0:
            bandwidth = 2E-9

        dw = w0*(bandwidth/lambda0)

        filter_profile = (0.5 * (np.sign(w0-w+dw/2) + 1))*(0.5 * (np.sign(w-w0+dw/2) + 1))

    else:
        '''
        if no filter is specified, only losses are applied (filter is == 1 for all freq)
        '''
        filter_profile = np.ones(np.shape(w))
        
    output_At = np.sqrt(1-loss)*np.fft.fft(Af*filter_profile)

    return output_At


def propagateFiber  (pulse, fiber):
    '''This function will propagate the input field along the length of...
    a fibre with the given properties...

    # Pulse propagation via Nonlinear Schrodinger Equation (NLSE)
    # dA/dz = -ib2/2 (d^2A/dtau^2) + b3/6 (d^3 A/dtau^3) -aplha/2 + ig|A|^2*A  
    # --> A is field A = sqrt(P0)*u

   Requires a Pulse class object and Fiber class object. Fiber can also be FiberGain class

   Inputs:
   pulse = Pulse class object
   fiber = Fiber class object (Fiber or FiberGain)

   Outputs:
    outputField = time domain output field, At
    ''' 
    
    #Pulse inputs
    nt = pulse.nt
    tau = pulse.time
    dtau = pulse.dt
    omega = pulse.freq

    #fiber inputs
    nz = np.size(fiber.z)
    dz = np.gradient(fiber.z)   #position step size

    #compile losses(alpha) and gain appropriately, result should have same dim as fiber.z
    if type(fiber) is Fiber:
        #Fiber does not have inherent gain parameter, thus gain is set to 0
        gain = np.zeros(np.shape(fiber.z))
    elif type(fiber) is FiberGain:
        #FiberGain has gain parameter
        #if fiber.gain is const, this creates a const arrray, if .gain is an array this is simply X 1
        gain = np.ones(np.shape(fiber.z))*fiber.gain
    else:
        #Don't know when this would apply
        gain = np.zeros(np.shape(fiber.z))

    #combined loss and gain, will be array same dim as fiber.z
    #fiber.alpha could be const. or array, result is same dimensionally
    alpha = (fiber.alpha - gain)


    #Dispersion operator, same dim as fiber.z
    D = (-alpha/2)
    for i in range(len(fiber.beta)):
        D += (1j*fiber.beta[i]*omega**(i+2)/np.math.factorial(i+2))
    
    #Nonlinear operator, constant
    N = 1j*fiber.gamma
    
    #Main propagation loop
    At = pulse.At*np.exp(np.abs(pulse.At)**2*N*dz[0]/2)
    for i in range(nz-1):
       
       Af = np.fft.ifft(At)
       Af = Af*np.exp(D[i]*dz[i])
       At = np.fft.fft(Af)
       At = At*np.exp(N*dz[i]*np.abs(At)**2)

    Af = np.fft.ifft(At)
    Af = Af*np.exp(D[-1]*dz[-1])
    At = np.fft.fft(Af)
    outputField = At*np.exp(np.abs(At)**2*N*dz[-1]/2)
    
    return(outputField)
    
    
















#%%
'''

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
'''