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

 '''

import numpy as np
import scipy as sp



 class Xstal:
 	'''
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



kgw = Xstal(3, 0.03, 'Yb:KGW')


#Spacial grid
dz = 0.01	#in mm
z = 
