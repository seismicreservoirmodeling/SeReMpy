# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:59:11 2023

@author: Dario Grana and Mingliang Liu
"""

# Geostatistical Petrophysical inversion Driver %%
# In this script we apply the Ensenmble petrophysical inversion method
# (Liu and Grana, 2018) to predict the petrophysical properties 
# from seismic data.

import time
import numpy as np
import scipy.io as sio
from Inversion import RickerWavelet
import matplotlib.pyplot as plt
from GeosPIn3D import GeosPetroInversion3D
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
# Prior model (filtered well logs)
phipriormean = 0.2
claypriormean = 0.23
swpriormean = 0.6
phiprior = phipriormean * np.ones((nxl, nil, nm))
clayprior = claypriormean * np.ones((nxl, nil, nm))
swprior = swpriormean * np.ones((nxl, nil, nm))

# Prior correlation matrix of petrophysical properties
corrpetro = np.array([[1.0, -0.6, -0.5],
                      [-0.6, 1.0, 0.2],
                      [-0.5, 0.2, 1.0]])

# Prior standard deviation vector of petrophysical properties
stdpetro = np.array([0.035, 0.055, 0.09])

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

# Input parameters
nsim = 500
niter = 4
vertcorr = 20
horcorr = 25

# Call the GeosPetroInversion3D function
start_t = time.time()
Phimap, Claymap, Swmap, Time = GeosPetroInversion3D(near, mid, far, TimeSeis, phiprior, clayprior, swprior, stdpetro, corrpetro, elastrain, petrotrain, wavelet, theta, nv, sigmaerr, vertcorr, horcorr, nsim, niter)
end_t = time.time()
elapsed_t = end_t - start_t
print('Computational time: ', elapsed_t) # seconds
# Create meshgrid for plotting
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