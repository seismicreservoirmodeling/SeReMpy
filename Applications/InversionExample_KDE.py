#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:59:11 2023

@author: Dario Grana and Mingliang Liu
"""

# Bayesian Petrophysical Inversion Driver 
# In this script we apply the Bayesian petrophysical inversion method
# (Grana and Della Rossa, 2010) to predict the petrophysical properties 
# from seismic data.

import time
import numpy as np
import scipy.io as sio
from Inversion import RickerWavelet
import matplotlib.pyplot as plt
from GeosPIn3D import SpatialCovariance, BayesPetroInversion3D_KDE
from Utils import plot_slices


# Load seismic data from the 'SeismicData3D.mat' file
seismic_data = sio.loadmat('Data3D/SeismicData3D.mat')
near = seismic_data['near']
mid = seismic_data['mid']
far = seismic_data['far']
X = seismic_data['Y']
Y = seismic_data['X']
Z = seismic_data['Z']

# Define the chosen inline and crossline indices
interx = 34
intery = 39

# Additional initial parameters
nm = near.shape[2] + 1
nxl = near.shape[0]
nil = near.shape[1]
nv = 3
theta = [15, 30, 45]
ntheta = len(theta)
TimeSeis = Z[0, 0, :]
dt = TimeSeis[1] - TimeSeis[0]
varerr = 1e-4
sigmaerr = varerr * np.eye(ntheta * (nm - 1))

# Wavelet parameters
freq = 45
ntw = 64
wavelet, tw = RickerWavelet(ntw, freq, dt)

# Plot seismic data
cmap = plt.cm.viridis
fig = plt.figure(1, figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
plot_slices(np.unique(X), np.unique(Y), np.unique(Z), near, interx, intery, cmap=cmap, ax=ax1)
ax1.set_xlabel('Inline')
ax1.set_ylabel('Crossline')
ax1.set_zlabel('Time (s)')
ax1.set_title('Near')

ax2 = fig.add_subplot(132, projection='3d')
plot_slices(np.unique(X), np.unique(Y), np.unique(Z), near, interx, intery, cmap=cmap, ax=ax2)
ax2.set_xlabel('Inline')
ax2.set_ylabel('Crossline')
ax2.set_zlabel('Time (s)')
ax2.set_title('Mid')

ax3 = fig.add_subplot(133, projection='3d')
plot_slices(np.unique(X), np.unique(Y), np.unique(Z), near, interx, intery, cmap=cmap, ax=ax3)
ax3.set_xlabel('Inline')
ax3.set_ylabel('Crossline')
ax3.set_zlabel('Time (s)')
ax3.set_title('Far')

plt.tight_layout()
plt.show()



# Prior model (filtered well logs)
vppriormean = 4
vspriormean = 2.4
rhopriormean = 2.3
Vpprior = vppriormean * np.ones((nxl, nil, nm))
Vsprior = vspriormean * np.ones((nxl, nil, nm))
Rhoprior = rhopriormean * np.ones((nxl, nil, nm))

# Spatial correlation matrix
corrlength = 5 * dt
sigma0 = np.array([[0.0034, 0.0037, 0.0014],
                   [0.0037, 0.0042, 0.0012],
                   [0.0014, 0.0012, 0.0015]])
sigmaprior = SpatialCovariance(corrlength, dt, nm, sigma0)

# Rock physics parameters (assuming the data is already loaded)
training_data = sio.loadmat('Data3D/RockPhysicsTrain.mat')
PhiTrain = training_data['PhiTrain']
ClayTrain = training_data['ClayTrain']
SwTrain = training_data['SwTrain']
VpTrain = training_data['VpTrain']
VsTrain = training_data['VsTrain']
RhoTrain = training_data['RhoTrain']
FaciesTrain = training_data['FaciesTrain']
petrotrain = np.column_stack((PhiTrain, ClayTrain, SwTrain))
npetro = petrotrain.shape[1]
elastrain = np.column_stack((VpTrain, VsTrain, RhoTrain))
nd = elastrain.shape[1]
nf = np.max(np.unique(FaciesTrain))

# KDE parameters
ndiscr = 25
phigrid = np.linspace(0.01, 0.4, ndiscr)
claygrid = np.linspace(0, 0.8, ndiscr)
swgrid = np.linspace(0, 1, ndiscr)

vpgrid = np.linspace(3.2, 4.6, ndiscr)
vsgrid = np.linspace(2, 3, ndiscr)
rhogrid = np.linspace(2, 2.6, ndiscr)

h = 5  # kernel factor

# Seismic inversion
start_t = time.time()
Vpmap, Vsmap, Rhomap, Phimap, Claymap, Swmap, Time = BayesPetroInversion3D_KDE(near, mid, far, TimeSeis, Vpprior, Vsprior, Rhoprior, sigmaprior, elastrain, petrotrain, vpgrid, vsgrid, rhogrid, phigrid, claygrid, swgrid, sigmaerr, wavelet, theta, nv, h)
end_t = time.time()
elapsed_t = end_t - start_t
print('Computational time: ', elapsed_t) # seconds
X, Y, Z = np.meshgrid(np.arange(1, nil+1), np.arange(1, nxl+1), Time)


# Plot results
fig = plt.figure(2, figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
plot_slices(np.unique(Y), np.unique(X), np.unique(Z[:, :, 1:]), Phimap, interx, intery, cmap=cmap, ax=ax1)
ax1.set_xlabel('Crossline')
ax1.set_ylabel('Time (s)')
ax1.set_title('Porosity')

ax2 = fig.add_subplot(132, projection='3d')
plot_slices(np.unique(Y), np.unique(X), np.unique(Z[:, :, 1:]), Claymap, interx, intery, cmap=cmap, ax=ax2)
ax2.set_xlabel('Crossline')
ax2.set_ylabel('Time (s)')
ax2.set_title('Clay volume')

ax3 = fig.add_subplot(133, projection='3d')
plot_slices(np.unique(Y), np.unique(X), np.unique(Z[:, :, 1:]), Swmap, interx, intery, cmap=cmap, ax=ax3)
ax3.set_xlabel('Crossline')
ax3.set_ylabel('Time (s)')
ax3.set_title('Water saturation')

plt.tight_layout()
plt.show()

