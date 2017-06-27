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
import pickle

from tqdm import tqdm


#constants
h = 6.62606957E-34  #J*s
c = 299792458.0     #m/s
mu0 = np.pi*4E-7               #magnetic constant (permeability), N*A-2
eps0 = 1/(np.pi*4E-7*c**2)         #electric constant (permitivity), F*m-1


class func:
    '''general function class to enable interpolation of a variable
    '''
    def __init__(self, value = None, x = None):
        self.val = value
        self.ind = x

    def at(self,x):
        return np.interp(x, self.ind, self.val)


class Pulse:
    '''
    Defines a Pulse object
    .time = time array (s)
    .freq = corresponding angular freq array (rad/s)
    .At = time domain Field, units sqrt(power) i.e. |At|**2 = power
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
        return ((self.dt*self.nt)/(np.sqrt(2*np.pi)))*np.fft.ifft(self.At)

    def getPt(self):
        return np.abs(self.At)**2

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

        if new_At is None:
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

            nz = np.max([2,self.length//z_grid])
            z_grid = self.length/nz
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
    .N is core dopant density
    .z is the z-axis array for the fiber

    Note: N*sigma_a_pump = pump abs coefficient (often quoted by manufacturers in dB/m)

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

        self.sigma_a = np.zeros(2)
        self.sigma_e = np.zeros(2)
    
        self.lambdas = np.zeros(2)

        self.tau = 770E-6
        self.N = 6.1813931E25

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

            nz = np.max([2,self.length//z_grid])
            z_grid = self.length/nz
            self.z = z_grid*np.arange(0, nz)    #position array

        else:
            # grid type is 'relative', z_grid is number of grid points
            if z_grid == None or z_grid < 1:
                z_grid = self.Z_NUM_DEFAULT

            dz = self.length/z_grid   #position step size
            self.z = dz*np.arange(0, z_grid)    #position array



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


def saveObj(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)


def loadObj(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)

    return obj


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
    
    
def calcZGrid(fiber,pulse, res = 'med'):
    '''Autocalculation of zgrid of Fiber.
    fiber = class Fiber() or FiberGain() instance
    pulse = class Pulse() instance
    res = resolution, either numeric or string
    '''
    if isinstance(res, str):
        if res.lower() == 'low':
            n = 10
        elif res.lower() == 'med':
            n = 35
        elif res.lower() == 'high':
            n = 100
        else:
            n = 35
        
    elif isinstance(res, int) or isinstance(res, float):
        n = res//1
    
    if n==0:
        n = 1

    _, t0 = rmswidth(pulse.time, np.abs(pulse.At))
    p0 = (np.abs(pulse.At)**2).max()
    
    ld = t0**2/(np.abs(fiber.beta[0]))
    ln = 1/(p0*fiber.gamma)
    l_ref = 1/((1/ld)+(1/ln))
    
    return l_ref/n
    
    
    

def calcGain(fiber, Pp, Ps, pump_scheme = 'core', pump_dir = 'forward', method = 'simple'):
    '''
    Calculate steady state gain over fiber
    Output z-array of gain
    fiber.sigma_x are 2x2 arrays. col 0 = wavelength, col 1 = sigma, row 0 = pump, row 1= signal
    '''

    s_ap = fiber.sigma_a[0]
    s_as = fiber.sigma_a[1]

    s_ep = fiber.sigma_e[0]
    s_es = fiber.sigma_e[1]

    v_p = c/fiber.lambdas[0]
    v_s = c/fiber.lambdas[1]

    b_p = (s_ap + s_ep)/(h*v_p)
    b_s = (s_as + s_es)/(h*v_s)
    a_p = s_ap/(h*v_p)
    a_s = s_as/(h*v_s)

    tau_se = fiber.tau
    dv_ase = (53E-9)*(v_s**2/c)

    MFA = np.pi*(fiber.core_d/2)**2
    G_s = MFA/(np.pi*(fiber.core_d/2)**2)
    Is = Ps/(np.pi*(fiber.core_d/2)**2)

    #define pump overlap, core pumped or clad pumped
    if pump_scheme.lower() in {'clad', 'cladding'}:
        G_p = MFA/(np.pi*(fiber.clad_d/2)**2)
        Ip = Pp/(np.pi*(fiber.clad_d/2)**2)
    else:
        G_p = MFA/(np.pi*(fiber.core_d/2)**2)
        Ip = Pp/(np.pi*(fiber.core_d/2)**2)


    g = np.zeros(np.shape(fiber.z))
    N=fiber.N
    dz = np.gradient(fiber.z)


    if method.lower() in {'simple', 's'}:
    #simple integration for intensities and n
        for i in tqdm(range(np.size(g)),desc='Calculate Gain',leave=False):

            n = (a_p*Ip + a_s*Is)/(b_p*Ip + b_s*Is + 1/tau_se)
            
            Ip = Ip*np.exp(-G_p*(s_ap*N*(1-n) - s_ep*N*n)*dz[i])
            Is = Is*np.exp(-G_s*(s_as*N*(1-n) - s_es*N*n)*dz[i])

            g[i] = (s_es*N*n - s_as*N*(1-n))


    elif method.lower() in {'rk4', 'r'}:
    #iterative rk4 method, significantly longer than 'simple'
    #likely only necessary for double clad fiber
        I0p = Ip
        I0s = Is
        Ip = I0p*np.ones(np.shape(fiber.z))
        Isig = I0s*np.ones(np.shape(fiber.z))
        Iasef = np.zeros(np.shape(fiber.z))
        Iaseb = np.zeros(np.shape(fiber.z))
        
        dIp =   lambda z, I, n: -G_p*(s_ap*N*(1-n.at(z)) - s_ep*N*n.at(z))*I 
        dIsig = lambda z, I, n: -G_s*(s_as*N*(1-n.at(z)) - s_es*N*n.at(z))*I 
        dIase = lambda z, I, n: -G_s*(s_as*N*(1-n.at(z)) - s_es*N*n.at(z))*I + n.at(z)*h*v_s*N*s_es*dv_ase/MFA

        n = func()
        n.ind = fiber.z
        n.val = np.zeros(np.shape(n.ind))

        loop_num = 0
        gain_err = 1
        c_gain = 0
        
        max_loops = 500
        min_err = 1E-4

        while (loop_num < max_loops and gain_err > min_err):

            p_gain = c_gain
            
            for i in tqdm(range(np.size(fiber.z)),desc='Calculate Gain',leave=False):
                Is = Isig + Iasef + Iaseb
                n.val[i] = (a_p*Ip[i] + a_s*Is[i])/(b_p*Ip[i] + b_s*Is[i] + 1/tau_se)
                
                if i < np.size(fiber.z)-1:
                    zf = np.array([fiber.z[i],fiber.z[i+1]])
                    zb = np.array([fiber.z[-i-2],fiber.z[-i-1]])
    
                    if pump_dir.lower().startswith('b'):
                            Ip[-i-2],_ = np.flipud(rk4(dIp, np.flipud(zb), Ip[-i-1], [n], True))
                    else:
                        _,Ip[i+1] = rk4(dIp, zf, Ip[i], [n])
                    _,Isig[i+1] = rk4(dIsig, zf, Isig[i], [n])
                    _,Iasef[i+1] = rk4(dIase, zf, Iasef[i], [n])
                    Iaseb[-i-2],_ = np.flipud(rk4(dIase, np.flipud(zb), Iaseb[-i-1], [n], True))

            g = (s_es*N*n.val - s_as*N*(1-n.val))

            c_gain = g.sum()
            gain_err = np.abs((c_gain-p_gain)/c_gain)

            loop_num = loop_num + 1
            

        print(gain_err, loop_num)
        

    else:
        g.fill(1)
        print('Unknown method. Gain set to 1')
    
    return g


def gratingPair(pulse, L, N, AOI, loss = 0, return_coef = False):
    '''
    ***NEED TO FIX EQN's, theta not AOI*** see ettiene thesis and Backus 1998
    Simulate grating pair
    pulse = input pulse object
    L = grating separation (m), use (-) L for stretcher, (+) L for compressor geometry
    N = lns/mm of gratings
    AOI = angle of incidence (deg)
    loss = %loss of INTENSITY (not field)

    theta = diffraction angle (assumed -1 order, as is standard)
    d = groove spacing

    returns time-domain output field

    '''
    m = 1
    g = AOI*np.pi/180    #convert AOI into rad
    d = 1E-3/N    #gives grove spacing in m

    Af = np.fft.ifft(pulse.At)
    w0 = 2*np.pi*c/pulse.lambda0
    omega = pulse.freq
    theta = np.arcsin(m*2*np.pi*c/(w0*d) - np.sin(g)) 

    phi2 = (-m**2*2*4*(np.pi**2)*L*c/(d**2*w0**3))*(1/np.cos(theta)**3)
    phi3 = (-3*phi2/w0)*(1+(2*np.pi*c*m*np.sin(theta)/(w0*d*np.cos(theta)**2)))
    phi4 = ((2*phi3)**2/(3*phi2)) + phi2*(2*np.pi*c*m/(w0**2*d*np.cos(theta)**2))**2

    output_At = np.sqrt(1-loss)*np.fft.fft(Af*np.exp(1j*(phi2*(omega)**2/2 + phi3*(omega)**3/6 + phi4*(omega)**4/24)))
    
    if return_coef:
        return output_At, np.array([phi2,phi3,phi4])
    else:
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
    set either pulse to None for 'splitter' behaviour

    B(pulse2)-----[=======]-----SignalA, tapB
                  [==2x2==]
    A(pulse1)-----[=======]-----SignalB, tapA

    pulse1 goes to output_sig with (1-tap)
    pulse1 goes to output_tap with tap
    pulse2 goes to output_tap with tap
    pulse2 goes to output_sig with (1-tap)

    tap can be concidered output coupler value
    '''
    if pulse1 is None:
        At1 = 0
    else:
        At1 = np.sqrt(1-loss)*pulse1.At

    if pulse2 is None:
        At2 = 0
    else:
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
    
    Af = np.fft.ifft(pulse.At)
    
    if lambda0 == None:
        lambda0 = pulse.lambda0

    w = pulse.freq + 2*np.pi*c/pulse.lambda0
    w0 = 2*np.pi*c/lambda0

    if filter_type.lower() == 'lpf':
        '''
        long-pass, pass low freq
        w0-w is (+) for w<w0 (pass region)
        '''
        filter_profile = 0.5*(np.sign(w0-w) + 1)

    elif filter_type.lower() == 'spf':
        '''
        short-pass, pass high freq
        w-w0 is (+) for w>w0 (pass region)
        '''
        filter_profile = 0.5*(np.sign(w-w0) + 1)

    elif filter_type.lower() == 'bpf':
        '''
        bandpass
        '''
        dw = w0*(bandwidth/lambda0)

        filter_profile = (0.5*(np.sign(w0-w+dw/2) + 1))*(0.5 * (np.sign(w-w0+dw/2) + 1))

    else:
        '''
        if no filter is specified, only losses are applied (filter is == 1 for all freq)
        '''
        filter_profile = np.ones(np.shape(w))
        
    output_At = np.sqrt(1-loss)*np.fft.fft(Af*filter_profile)

    return output_At


def propagateFiber (pulse, fiber, autodz = False):
    '''This function will propagate the input field along the length of...
    a fibre with the given properties...

    # Pulse propagation via Nonlinear Schrodinger Equation (NLSE)
    # dA/dz = -ib2/2 (d^2A/dtau^2) + b3/6 (d^3 A/dtau^3) -aplha/2 + ig|A|^2*A  
    # --> A is field A = sqrt(P0)*u

    Requires a Pulse class object and Fiber class object. Fiber can also be FiberGain class

    Inputs:
    pulse = Pulse class object
    fiber = Fiber class object (Fiber or FiberGain)
    autodz = boolean to calculate fiber dz
   
    Outputs:
    outputField = time domain output field, At
    
    Warning: setting autodz = True will modify fiber object!!!
    autodz uses calcZGrid to calculate dz based on the input pulse and fiber
    Should not be used for gain fiber!!!, since gain calc depends on dz as well
    ''' 
    if autodz == False:
        pass
    else:
        if autodz is True:
            res = 'med'
        else:
            try:
                res = autodz//1
            except TypeError:
                res = 'med'

        dz = calcZGrid(fiber,pulse,res)
        fiber.initializeGrid(fiber.length, 'abs', dz)
    
    #Pulse inputs
    #nt = pulse.nt
    #tau = pulse.time
    #dtau = pulse.dt
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

    #Define Dispersion operator: D = G + B, G = gain/loss, B = dispersion
    G = -alpha/2 + 0j*alpha
    B = 0
    
    #Dispersion components
    for i in range(len(fiber.beta)):
        B += (1j*fiber.beta[i]*omega**(i+2)/np.math.factorial(i+2))
    
    #Nonlinear operator, constant
    N = 1j*fiber.gamma
    
    #Main propagation loop
    At = pulse.At*np.exp(np.abs(pulse.At)**2*N*dz[0]/2)
    for i in tqdm(range(nz-1),desc='Progagate Fiber',leave=False):
        
        D = G[i] + B
       
        Af = np.fft.ifft(At)
        Af = Af*np.exp(D*dz[i])
        At = np.fft.fft(Af)
        At = At*np.exp(N*dz[i]*np.abs(At)**2)

    #Final Propagation steps
    Af = np.fft.ifft(At)
    Af = Af*np.exp(D[-1]*dz[-1])
    At = np.fft.fft(Af)
    outputField = At*np.exp(np.abs(At)**2*N*dz[-1]/2)
    
    return outputField
    

def saturableAbs(pulse,sat_int,spot_size,mod_depth=1,loss=0):
    ''' Simulate saturable absorber.
    sat_int = saturation intensity, J/m**2
    spot_size = beam diameter
    mod_depth = modulation depth, ratio e.g. 1% -> 0.01
    loss = non saturable losses

    small signal -> refl ~ 1-loss-mod_depth
    high signal --> refl ~ 1-loss
    '''
    intensity = np.abs(pulse.At)**2/(np.pi*(spot_size/2)**2)
    outputField = np.sqrt(1-loss)*pulse.At*np.sqrt((1-mod_depth/(1+intensity/sat_int)))

    return outputField




