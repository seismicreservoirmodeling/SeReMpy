#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:43:29 2020

@author: dariograna
"""

#% Rock Physics Model Driver %%
# In this script we apply different rock physics model to a synthetic well
# log of porosity. 
# The rock physics models include:
# Wyllie
# Raymer
# Soft sand model
# Stiff sand model
# Inclusion model for spherical pores
# Berryman inclusion model for ellptical pores
# We assume that the solid is 100% quartz and the fluid is 100% water. For
# mixtures of minerals and fluids, the elastic properties can be computed
# using the function Matrix Fluid Model.
# 

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from RockPhysics import *


#% Available data and parameters
# Load data (porosity and depth)
x = np.loadtxt('Data/data1.dat')
Depth = x[:,0].reshape(-1, 1)
Phi = x[:,1].reshape(-1, 1)


# Initial parameters
Kmat = 36
Gmat = 45
Rhomat = 2.65
Kfl = 2.25
Gfl = 0
Rhofl = 1


#% Empirical models
# Initial parameters
Vpmat, Vsmat = VelocityDefinitions(Kmat, Gmat, Rhomat)
Vpfl, Vsfl = VelocityDefinitions(Kfl, Gfl, Rhofl)

# Wyllie model
VpW = WyllieModel(Phi, Vpmat, Vpfl)
# Raymer model
VpR = RaymerModel(Phi, Vpmat, Vpfl)

# figures
plt.figure(1)
plt.plot(VpW, Depth, color='k', linewidth=2.0)
plt.plot(VpR, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(1.5, 6.5) 
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Depth')
plt.legend(('Wyllie model', 'Raymer model'))
plt.show()


plt.figure(2)
plt.scatter(Phi, VpW, 50, Phi, 'o')
plt.scatter(Phi, VpR, 50, Phi, 'd')
plt.grid()
plt.xlim(0, 0.3) 
plt.ylim(1.5, 6.5) 
plt.xlabel('Porosity')
plt.ylabel('P-wave velocity (km/s)')
plt.legend(('Wyllie model', 'Raymer model'))

#% Granular media models
# Initial parameters
criticalporo = 0.4
coordnumber = 7
pressure = 0.02

# Density
Rho = DensityModel(Phi, Rhomat, Rhofl)

# Soft sand model
VpSoft, VsSoft = SoftsandModel(Phi, Rho, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure)
# Stiff sand model
VpStiff, VsStiff = StiffsandModel(Phi, Rho, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure)

# figures
plt.figure(3)
plt.subplot(121)
plt.plot(VpSoft, Depth, color='k', linewidth=2.0)
plt.plot(VpStiff, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(1.5, 6.5) 
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Depth')
plt.subplot(122)
plt.plot(VsSoft, Depth, color='k', linewidth=2.0)
plt.plot(VsStiff, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(0.5, 4.5) 
plt.xlabel('S-wave velocity (km/s)')
plt.ylabel('Depth')
plt.legend(('Soft sand model', 'Stiff sand model'))
plt.show()

plt.figure(4)
plt.scatter(Phi, VpSoft, 50, Phi, 'o')
plt.scatter(Phi, VpStiff, 50, Phi, 'd')
plt.grid()
plt.xlim(0, 0.3) 
plt.ylim(1.5, 6.5) 
plt.xlabel('Porosity')
plt.ylabel('P-wave velocity (km/s)')
plt.legend(('Soft sand model', 'Stiff sand model'))


#% Includion models 
# Initial parameters
Ar = 0.2# for elliptical inclusion model

# Density
Rho = DensityModel(Phi, Rhomat, Rhofl)

# Spherical inclusion model
VpSph, VsSph = SphericalInclusionModel(Phi, Rho, Kmat, Gmat, Kfl)
# Elliptical inclusion model
VpEll, VsEll = BerrymanInclusionModel(Phi, Rho, Kmat, Gmat, Kfl, Ar)

# figures
plt.figure(5)
plt.subplot(121)
plt.plot(VpSph, Depth, color='k', linewidth=2.0)
plt.plot(VpEll, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(1.5, 6.5) 
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Depth')
plt.subplot(122)
plt.plot(VsSph, Depth, color='k', linewidth=2.0)
plt.plot(VsEll, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(0.5, 4.5) 
plt.xlabel('S-wave velocity (km/s)')
plt.ylabel('Depth')
plt.legend(('Spherical pores', 'Elliptical pores (Ar=0.2)'))
plt.show()

plt.figure(4)
plt.scatter(Phi, VpSph, 50, Phi, 'o')
plt.scatter(Phi, VpEll, 50, Phi, 'd')
plt.grid()
plt.xlim(0, 0.3) 
plt.ylim(1.5, 6.5) 
plt.xlabel('Porosity')
plt.ylabel('P-wave velocity (km/s)')
plt.legend(('Spherical pores', 'Elliptical pores (Ar=0.2)'))

