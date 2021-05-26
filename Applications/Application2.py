#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:43:01 2021

@author: dariograna

Reference: Grana and de Figueiredo, 2021, SeReMpy
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from context import SeReMpy
from SeReMpy.Geostats import *
from SeReMpy.Facies import *


#% Application2
# data
x = np.loadtxt('Data/1Ddatalog.dat')
Depth = x[:,1].reshape(-1, 1)
Facies = x[:,2].reshape(-1, 1)
Phi = x[:,3].reshape(-1, 1)
Vp = x[:,7].reshape(-1, 1)

# available data - porosity
dcoords = Depth[::10]
dphi = Phi[::10]
nd = len(dcoords)
# grid of coordinates of the location to be estimated
xcoords = Depth
ns = len(xcoords)

# parameters random variable - porosity
phimean = 0.23
phivar = 0.001
l = 0.01
krigtype = 'gau'

# kriging
xsk = np.zeros((ns, 1))
for i in range(ns):
    # simple kiging
    xsk[i,0] = SimpleKriging(xcoords[i,:], dcoords, dphi, phimean, phivar, l, krigtype)[0]
     
# Sequential Gaussian Simulation
krig = 1
nsim = 10
sgsim = np.zeros((ns,nsim))
for i in range(nsim):
    sim = SeqGaussianSimulation(xcoords, dcoords, dphi, phimean, phivar, l, krigtype, krig)
    sgsim[:,i] = sim[:,0]

# figures
plt.figure(2)
ax = plt.subplot(151)
plt.plot(xsk[:,0], Depth, color='r', linewidth=2.0)
plt.plot(dphi, dcoords, 'k*')
plt.grid()
plt.xlim(0, 0.4) 
plt.xlabel('Porosity')
plt.ylabel('Depth (km)')
plt.ylim(max(Depth),min(Depth))
yticks = ax.get_yticks() 
ax = plt.subplot(152)
for i in range(nsim):
    plt.plot(sgsim[:,i], Depth, linewidth=2.0)
plt.plot(dphi, dcoords, 'k*')
plt.grid()
plt.xlim(0, 0.4) 
plt.xlabel('Porosity')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))


# available data - Vp
dcoords = Depth[::10]
dvp = Vp[::10]
nd = len(dcoords)
# grid of coordinates of the location to be estimated
xcoords = Depth
ns = len(xcoords)

# parameters random variable
vpmean = 4
vpvar = 0.03
l = 0.01
krigtype = 'gau'

# kriging
xsk = np.zeros((ns, 1))
for i in range(ns):
    # simple kiging
    xsk[i,0] = SimpleKriging(xcoords[i,:], dcoords, dvp, vpmean, vpvar, l, krigtype)[0]
     
# Sequential Gaussian Simulation
krig = 1
nsim = 10
sgsim = np.zeros((ns,nsim))
for i in range(nsim):
    sim = SeqGaussianSimulation(xcoords, dcoords, dvp, vpmean, vpvar, l, krigtype, krig)
    sgsim[:,i] = sim[:,0]

# figures
ax = plt.subplot(153)
plt.plot(xsk[:,0], Depth, color='r', linewidth=2.0)
plt.plot(dvp, dcoords, 'k*')
plt.grid()
plt.xlim(3,6) 
plt.xlabel('P-wave velocity (km/s)')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
ax = plt.subplot(154)
for i in range(nsim):
    plt.plot(sgsim[:,i], Depth, linewidth=2.0)
plt.plot(dvp, dcoords, 'k*')
plt.grid()
plt.xlim(3,6) 
plt.xlabel('P-wave velocity (km/s)')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))


# Markov chain simulation
nsim = 10
Facies = Facies-1
Facies = Facies.astype(int)
nf = max(np.unique(Facies))+1
T = np.zeros((nf, nf))
for i in range(ns-1):
    T[Facies[i], Facies[i+1]] = T[Facies[i], Facies[i+1]] + 1
T = T / np.sum(T,axis=1).reshape(-1, 1)

# simulation
fsim = MarkovChainSimulation(T, ns, nsim)

# figures
ax = plt.subplot(155)
plt.pcolor(np.arange(nsim+1), Depth, fsim)
plt.xlabel('Facies realizations')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.ylim(max(Depth),min(Depth))
plt.show()



