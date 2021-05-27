#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:43:50 2021

@author: dariograna

Reference: Grana and de Figueiredo, 2021, SeReMpy
"""

from scipy.io import loadmat
import scipy.spatial.distance
import matplotlib.pyplot as plt
import numpy as np

from context import SeReMpy
from SeReMpy.Geostats import *


#% Application3  (temperature Yellowstone)
E = np.loadtxt('Data/ElevationData.dat')
nx = 123
ny = 292
X = E[:,0].reshape(nx, ny)
Y = E[:,1].reshape(nx, ny)
Z = E[:,2].reshape(nx, ny)
T = E[:,3].reshape(nx, ny)
d = np.loadtxt('Data/data6full.dat')
dx = d[:,0].reshape(-1, 1)
dy = d[:,1].reshape(-1, 1)
dz = d[:,2].reshape(-1, 1)
dt = d[:,3].reshape(-1, 1)

# # available data (100 measurements)
dcoords = np.hstack([dx,dy])
nd = dcoords.shape[0]
# # grid of coordinates of the location to be estimated
xcoords = np.transpose(np.vstack([X.reshape(-1), Y.reshape(-1)]))
n = xcoords.shape[0]

# parameters random variable
tmean = 7.45
tvar = 0.45
l = 25
krigtype = 'exp'

# kriging
xsk = np.zeros((n, 1))
for i in range(n):
    # simple kiging
    xsk[i,0] = SimpleKriging(xcoords[i,:], dcoords, dt, tmean, tvar, l, krigtype)[0]
xsk = np.reshape(xsk, X.shape)

# Sequential Gaussian Simulation
krig = 1
nsim = 1
sgsim = np.zeros((X.shape[0], X.shape[1], nsim))
for i in range(nsim):
    sim = SeqGaussianSimulation(xcoords, dcoords, dt, tmean, tvar, l, krigtype, krig)
    sgsim[:,:,i] = np.reshape(sim, (X.shape[0], X.shape[1]))


# plot results
# # plot
plt.figure(3)
plt.subplot(311)
plt.pcolor(X,Y, T)
plt.scatter(dcoords[:,0], dcoords[:,1], 20, dt, 'o')
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Elevation', rotation=270)
plt.title('Temperature data')
plt.subplot(312)
plt.pcolor(X,Y, xsk)
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Elevation', rotation=270)
plt.title('Simple Kriging')
plt.subplot(313)
plt.pcolor(X,Y, sgsim[:,:,0])
plt.xlabel('X')
plt.ylabel('Y')
cbar = plt.colorbar()
cbar.set_label('Temperature', rotation=270)
plt.title('SGS Realization')
plt.show()


