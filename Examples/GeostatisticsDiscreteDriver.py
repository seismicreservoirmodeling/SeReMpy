#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:02:21 2020

@author: dariograna

Reference: Grana and de Figueiredo, 2021, SeReMpy
"""

#% Geostatistics Discrete Driver %%
# In this script we illustrate kriging and sequential simulation with 
# two examples: 
# Example 1: example with 4 density measurements
# Example 2: example with 15 elevation measurements from Yellowstone

from scipy.io import loadmat
import scipy.spatial.distance
import matplotlib.pyplot as plt
import numpy as np

from context import SeReMpy
from SeReMpy.Geostats import *

#% Example 1
# available data (4 measurements)
dcoords = np.array([[5, 18], [15, 13], [11, 4], [1, 9]])
fvalues = np.array([[0, 1, 1, 0]])
fvalues = np.transpose(fvalues)
# coordinates of the location to be estimated
xcoords = np.array([10, 10])

# parameters random variable
nf = 2
pprior = np.array([0.5, 0.5])
l = 9
krigtype = 'exp'

# plot
plt.figure(1)
plt.scatter(dcoords[:,0], dcoords[:,1], 100, fvalues, 'o')
plt.plot(xcoords[0], xcoords[1], 'ks')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

# indicator kriging
ikp, ikmap = IndicatorKriging(xcoords, dcoords, fvalues, nf, pprior, float(l), krigtype)


# simulation
nsim = 1000
isim = np.zeros((nsim, 1))
for i in range(nsim):
    isim[i,0] = RandDisc(ikp)

# plot results
plt.figure(2)
plt.hist(isim)
plt.plot(ikmap, 0, '*r')
plt.grid()
plt.xlabel('Discrete property')
plt.ylabel('Frequency')
plt.show()


#% Example 2 (elevation Yellowstone)
E = np.loadtxt('Data/ElevationData.dat')
nx = 123
ny = 292
X = E[:,0].reshape(nx, ny)
Y = E[:,1].reshape(nx, ny)
Z = E[:,2].reshape(nx, ny)
T = E[:,3].reshape(nx, ny)
d = np.loadtxt('Data/data6reduced.dat')
dx = d[:,0].reshape(-1, 1)
dy = d[:,1].reshape(-1, 1)
dz = d[:,2].reshape(-1, 1)
dt = d[:,3].reshape(-1, 1)

# available data (15 measurements)
dcoords = np.hstack([dx,dy])
nd = dcoords.shape[0]

# discrete property definition
zmean = 2476
df = np.zeros(dz.shape)
df[dz > zmean] = 1
df = df.astype(int)

# grid of coordinates of the location to be estimated
xcoords = np.transpose(np.vstack([X.reshape(-1), Y.reshape(-1)]))
n = xcoords.shape[0]

# parameters random variable
pprior = np.array([0.5, 0.5])
l = 12.5
krigtype = 'exp'

 # plot
plt.figure(3)
plt.scatter(dcoords[:,0], dcoords[:,1], 50, df, 'o')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

# kriging
ikp = np.zeros((n, nf))
ikmap = np.zeros((n,1))
for i in range(n):
    ikp[i,:],ikmap[i,0] = IndicatorKriging(xcoords[i,:], dcoords, df, nf, pprior, l, krigtype)
ikp = np.reshape(ikp, (X.shape[0], X.shape[1], nf))
ikmap = np.reshape(ikmap, (X.shape[0], X.shape[1]))

# Sequential Indicator Simulation
nsim = 3
sisim = np.zeros((X.shape[0], X.shape[1], nsim))
for i in range(nsim):
    sim = SeqIndicatorSimulation(xcoords, dcoords, df, nf, pprior, l, krigtype)
    sisim[:,:,i] = np.reshape(sim, (X.shape[0], X.shape[1]))


# # plot results
plt.figure(4)
plt.subplot(221)
plt.pcolor(X,Y, ikp[:,:,0])
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Probability valleys', rotation=270)
plt.title('Indicator Kriging Probability of facies 0')
plt.subplot(222)
plt.pcolor(X,Y, ikmap)
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Peaks Valleys', rotation=270)
plt.title('Indicator Kriging most likely facies')
plt.subplot(223)
plt.pcolor(X,Y, sisim[:,:,0])
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Peaks Valleys', rotation=270)
plt.title('SIS Realization 1')
plt.subplot(224)
plt.pcolor(X,Y, sisim[:,:,1])
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Peaks Valleys', rotation=270)
plt.title('SIS Realization 2')
plt.show()


#% Markov chain simulation
# initial parameters
nsim = 3
ns = 100
# vertical axis
z = np.arange(ns)

# Transition matrix T1 (equal propotions, equal transitions)
T1 = np.array([[0.5, 0.5], [0.5, 0.5]])
# Transition matrix T2 (equal propotions, asymmetrix transitions)
T2 = np.array([[0.9, 0.1], [0.1, 0.9]])
# Transition matrix T3 (different propotions, asymmetrix transitions)
T3 = np.array([[0.1, 0.9], [0.1, 0.9]])

# simulation
fsim1 = MarkovChainSimulation(T1, ns, nsim)
fsim2 = MarkovChainSimulation(T2, ns, nsim)
fsim3 = MarkovChainSimulation(T3, ns, nsim)

# plot realzations
plt.figure(5)
plt.subplot(131)
plt.pcolor(np.arange(nsim+1), z, fsim1)
plt.xlabel('Facies realizations')
plt.ylabel('Relative depth (m)')
plt.title('Transition matrix T1')
plt.subplot(132)
plt.pcolor(np.arange(nsim+1), z, fsim2)
plt.xlabel('Facies realizations')
plt.title('Transition matrix T2')
plt.subplot(133)
plt.pcolor(np.arange(nsim+1), z, fsim3)
plt.xlabel('Facies realizations')
plt.title('Transition matrix T3')
plt.show()

