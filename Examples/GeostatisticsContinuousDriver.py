#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:25:49 2020

@author: dariograna

Reference: Grana and de Figueiredo, 2021, SeReMpy
"""

#% Geostatistics Continuous Driver %%
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
dvalues = np.array([[3.1, 3.9, 4.1, 3.2]])
dvalues = np.transpose(dvalues)
# coordinates of the location to be estimated
xcoords = np.array([10, 10])

# parameters random variable
xmean = 3.5
xvar = 0.1
l = 9
krigtype = 'exp'

# plot
plt.figure(1)
plt.scatter(dcoords[:,0], dcoords[:,1], 100, dvalues, 'o')
plt.plot(xcoords[0], xcoords[1], 'ks')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()


# simple kiging
xsk, xvar = SimpleKriging(xcoords, dcoords, dvalues, xmean, xvar, l, krigtype)

# ordinary kiging
xok, xvar = OrdinaryKriging(xcoords, dcoords, dvalues, xvar, l, krigtype)

# Gaussian simulation
krig = 0
nsim = 100
gsim = np.zeros((nsim, 1))
for i in range(nsim):
    gsim[i,0] = GaussianSimulation(xcoords, dcoords, dvalues, xmean, xvar, l, krigtype, krig)

# plot results
plt.figure(2)
plt.hist(gsim)
plt.plot(xsk, 0, '*r')
plt.plot(xok, 0, 'sb')
plt.plot(np.mean(gsim), 0, 'og')
plt.grid()
plt.xlabel('Property')
plt.ylabel('Frequency')
plt.show()


# #% Example 2 (elevation Yellowstone)
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


# # available data (15 measurements)
dcoords = np.hstack([dx,dy])
nd = dcoords.shape[0]
# # grid of coordinates of the location to be estimated
xcoords = np.transpose(np.vstack([X.reshape(-1), Y.reshape(-1)]))
n = xcoords.shape[0]

# parameters random variable
zmean = 2476
zvar = 8721
l = 12.5
krigtype = 'exp'

# # plot
plt.figure(3)
plt.scatter(dcoords[:,0], dcoords[:,1], 50, dz, 'o')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

# kriging
xsk = np.zeros((n, 1))
xok = np.zeros((n, 1))
for i in range(n):
    # simple kiging
    xsk[i,0] = SimpleKriging(xcoords[i,:], dcoords, dz, zmean, zvar, l, krigtype)[0]
    # ordinary kiging
    xok[i,0] = OrdinaryKriging(xcoords[i,:], dcoords, dz, zvar, l, krigtype)[0]
xsk = np.reshape(xsk, X.shape)
xok = np.reshape(xok, X.shape)

# Sequential Gaussian Simulation
krig = 1
nsim = 3
sgsim = np.zeros((X.shape[0], X.shape[1], nsim))
for i in range(nsim):
    sim = SeqGaussianSimulation(xcoords, dcoords, dz, zmean, zvar, l, krigtype, krig)
    sgsim[:,:,i] = np.reshape(sim, (X.shape[0], X.shape[1]))


# plot results
plt.figure(4)
plt.subplot(221)
plt.pcolor(X,Y, xsk)
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Elevation', rotation=270)
plt.title('Simple Kriging')
plt.subplot(222)
plt.pcolor(X,Y, xok)
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Elevation', rotation=270)
plt.title('Ordinary Kriging')
plt.subplot(223)
plt.pcolor(X,Y, sgsim[:,:,0])
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Elevation', rotation=270)
plt.title('SGS Realization 1')
plt.subplot(224)
plt.pcolor(X,Y, sgsim[:,:,1])
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Elevation', rotation=270)
plt.title('SGS Realization 2')
plt.show()


