#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:08:52 2021

@author: dariograna

Reference: Grana and De Figueiredo, 2021, SeReMpy
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from RockPhysics import *
from Facies import *


#% Application1
# data
x = np.loadtxt('Data/1Ddatalog.dat')
Clay = x[:,0].reshape(-1, 1)
Depth = x[:,1].reshape(-1, 1)
Facies = x[:,2].reshape(-1, 1)
Phi = x[:,3].reshape(-1, 1)
Rho = x[:,4].reshape(-1, 1)
Sw = x[:,5].reshape(-1, 1)
Vp = x[:,7].reshape(-1, 1)
Vs = x[:,8].reshape(-1, 1)


# Rock physics
# Initial parameters
criticalporo = 0.55
coordnumber = 12
pressure = 0.04
Kminc = np.array([36,21])
Gminc = np.array([45,7])
Rhominc = np.array([2.68,2.65])
Kflc = np.array([2.5,0.7])
Rhoflc = np.array([1.03,0.7])
Volminc = np.hstack([1-Clay, Clay])
Sflc = np.hstack([Sw, 1-Sw])
patchy = 0

# Rock physics model
# Solid and fluid
[Kmat, Gmat, Rhomat, Kfl, Rhofl] = MatrixFluidModel(Kminc, Gminc, Rhominc, Volminc, Kflc, Rhoflc, Sflc, patchy)
# Density
RhoMod = DensityModel(Phi, Rhomat, Rhofl)
# Stiff sand model
[VpStiff, VsStiff] = StiffsandModel(Phi, RhoMod, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure)

# figures
plt.figure(1)
ax = plt.subplot(171)
plt.plot(Phi, Depth, color='k', linewidth=2.0)
plt.grid()
plt.xlim(0, 0.4) 
plt.xlabel('Porosity')
plt.ylabel('Depth (km)')
plt.ylim(max(Depth),min(Depth))
yticks = ax.get_yticks() 
ax = plt.subplot(172)
plt.plot(Clay, Depth, color='k', linewidth=2.0)
plt.grid()
plt.xlim(0, 1) 
plt.xlabel('Clay volume')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
ax = plt.subplot(173)
plt.plot(Sw, Depth, color='k', linewidth=2.0)
plt.grid()
plt.xlim(0, 1) 
plt.xlabel('Water saturation')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
ax = plt.subplot(174)
plt.plot(VpStiff, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(3, 5) 
plt.xlabel('P-wave velocity (km/s)')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
ax = plt.subplot(175)
plt.plot(VsStiff, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(1.5, 3.5) 
plt.xlabel('S-wave velocity (km/s)')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
ax = plt.subplot(176)
plt.plot(RhoMod, Depth, color='r', linewidth=2.0)
plt.grid()
plt.xlim(2, 2.5) 
plt.xlabel('Density (g/cm^3)')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))


# Facies
# Initial parameters
Facies = Facies-1
Facies = Facies.astype(int)
data =  np.hstack([Vp, Vs, Rho])
ns = data.shape[0]
nv = data.shape[1]

# Gaussian model (2 components)
nf = max(np.unique(Facies))+1
fp = np.zeros((nf, 1))
mup = np.zeros((nf, nv))
sp = np.zeros((nv, nv, nf))
for k in range(nf):
    fp[k,0] = np.sum(Facies == k) / ns
    mup[k, :] = np.mean(data[Facies[:,0] == k, :],axis=0)
    sp[:,:,k] = np.cov(np.transpose(data[Facies[:,0] == k, :]))

# Facies classification
fmap, fpost = BayesGaussFaciesClass(data, fp, mup, sp)

# figure
ax = plt.subplot(177)
plt.pcolor(np.arange(2), Depth, fmap)
plt.xlabel('Predicted facies')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
plt.show()
