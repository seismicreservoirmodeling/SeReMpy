#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:19:51 2020

"""

#% Ensemble Smoother Petrophysical Inversion Driver %%
# In this script we apply the Ensemble Smoother inversion method
# (Liu and Grana, 2018) to predict the petrophysical properties 
# from seismic data.

from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from Inversion import *
from Geostats import CorrelatedSimulation
from RockPhysics import LinearizedRockPhysicsModel

#% Available data and parameters
# Load data (seismic data, reference petroelastic properties, and time)
ds = np.loadtxt('Data/data5seis.dat')
TimeSeis = ds[:,0].reshape(-1, 1)
Snear = ds[:,1].reshape(-1, 1)
Smid = ds[:,2].reshape(-1, 1)
Sfar = ds[:,3].reshape(-1, 1)
dl = np.loadtxt('Data/data5log.dat')
Phi = dl[:,0].reshape(-1, 1)
Clay = dl[:,1].reshape(-1, 1)
Sw = dl[:,2].reshape(-1, 1)
Time = dl[:,3].reshape(-1, 1)
Vp = dl[:,4].reshape(-1, 1)
Vs = dl[:,5].reshape(-1, 1)
Rho = dl[:,6].reshape(-1, 1)

#% Initial parameters
# number of samples (elastic properties)
nm = Snear.shape[0]+1
# number of samples (seismic data)
nd = Snear.shape[0]
# number of variables
nv = 3
# reflection angles 
theta = np.linspace(15,45,3)
ntheta = len(theta)
# time sampling
dt = TimeSeis[1] - TimeSeis[0]
# error variance
varerr = 10 ** -4
sigmaerr = varerr * np.eye(ntheta * (nm - 1))
# number of realizations
nsim = 500

#% Wavelet
# wavelet 
freq = 45
ntw = 64
wavelet, tw = RickerWavelet(freq, dt, ntw)

# matrix associated to the linear rock physics operator
R = np.zeros((nd, nv + 1))
X = np.hstack([Phi,Clay,Sw, np.ones(Phi.shape)])
R[0, :] = (np.linalg.lstsq(X,Vp,rcond=None)[0]).T
R[1, :] = (np.linalg.lstsq(X,Vs,rcond=None)[0]).T
R[2, :] = (np.linalg.lstsq(X,Rho,rcond=None)[0]).T

#% Plot seismic data
plt.figure(1)
plt.subplot(131)
plt.plot(Snear, TimeSeis, 'k')
plt.grid()
plt.ylim(max(TimeSeis),min(TimeSeis))
plt.xlabel('Near')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Smid, TimeSeis, 'k')
plt.grid()
plt.ylim(max(TimeSeis),min(TimeSeis))
plt.xlabel('Mid')
plt.subplot(133)
plt.plot(Sfar, TimeSeis, 'k')
plt.grid()
plt.ylim(max(TimeSeis),min(TimeSeis))
plt.xlabel('Far')
plt.show()

#% Prior model (filtered well logs)
nfilt = 3
cutofffr = 0.04
b, a = signal.butter(nfilt, cutofffr)
Phiprior = signal.filtfilt(b, a, np.squeeze(Phi))
Clayprior = signal.filtfilt(b, a, np.squeeze(Clay))
Swprior = signal.filtfilt(b, a, np.squeeze(Sw))
mprior = np.hstack([Phiprior[:,np.newaxis], Clayprior[:,np.newaxis], Swprior[:,np.newaxis]])

#% Spatial correlation matrix
corrlength = 5 * dt
trow = np.matlib.repmat(np.arange(0, nm * dt, dt), nm, 1)
tcol = np.matlib.repmat(trow[0,:].reshape(nm,1), 1, nm)
tdis = abs(trow - tcol)
sigmatime = np.exp(-(tdis / corrlength) ** 2)
sigma0 = np.cov(np.hstack([Phi, Clay, Sw]).T)
sigmaprior = np.kron(sigma0, sigmatime)

#% Prior realizations
Phisim = np.zeros((nm, nsim))
Claysim = np.zeros((nm, nsim))
Swsim = np.zeros((nm, nsim))
SeisPred = np.zeros((nd * ntheta, nsim))
for i in range(nsim):
    msim = CorrelatedSimulation(mprior, sigma0, sigmatime)
    Phisim[:,i] = msim[:,0]
    Claysim[:,i] = msim[:,1]
    Swsim[:,i] = msim[:,2]
Phisim[Phisim < 0] = 0
Phisim[Phisim > 0.4] = 0.4
Claysim[Claysim < 0] = 0
Claysim[Claysim > 0.8] = 0.8
Swsim[Swsim < 0] = 0
Swsim[Swsim > 1] = 1
Vpsim, Vssim, Rhosim = LinearizedRockPhysicsModel(Phisim, Claysim, Swsim, R)
for i in range(nsim):
    seis, TimeSeis = SeismicModel(Vpsim[:,i], Vssim[:,i], Rhosim[:,i], Time, theta, wavelet)
    SeisPred[:,i] = seis[:,0]
    
    
# plot of prior models
plt.figure(2)
plt.subplot(131)
plt.plot(Phisim, Time, 'b')
plt.plot(Phi, Time, 'k')
plt.plot(Phiprior, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Porosity')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Claysim, Time, 'b')
plt.plot(Clay, Time, 'k')
plt.plot(Clayprior, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Clay volume')
plt.subplot(133)
plt.plot(Sw, Time, 'k')
plt.plot(Swprior, Time, 'r')
plt.plot(Swsim, Time, 'b')
plt.plot(Sw, Time, 'k')
plt.plot(Swprior, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Water saturation')
plt.legend(['Reference model', 'Prior mean','Prior Realizations'],loc ="lower right")
plt.show()


#% ESMDA petrophysical inversion
niter = 4
alpha = 1 / niter   # sum alpha = 1
PriorModels = np.vstack([Phisim, Claysim, Swsim])
SeisData = np.vstack([Snear,Smid,Sfar])
PostModels = PriorModels
for j in range(niter):
    PostModels, KalmanGain = EnsembleSmootherMDA(PostModels, SeisData, SeisPred, alpha, sigmaerr)
    Phipost = PostModels[0:nm, :]
    Claypost = PostModels[nm :2 * nm, :]
    Swpost = PostModels[2 * nm :, :]
    Phipost[Phipost < 0] = 0
    Phipost[Phipost > 0.4] = 0.4
    Claypost[Claypost < 0] = 0
    Claypost[Claypost > 0.8] = 0.8
    Swpost[Swpost < 0] = 0
    Swpost[Swpost > 1] = 1
    Vppost, Vspost, Rhopost = LinearizedRockPhysicsModel(Phipost, Claypost, Swpost, R)
    for i in range(nsim):
        seis, TimeSeis = SeismicModel(Vppost[:,i], Vspost[:,i], Rhopost[:,i], Time, theta, wavelet)
        SeisPred[:,i] = seis[:,0]

# posterior mean models
mpost = np.mean(PostModels, axis=1)
mpost = mpost.reshape(len(mpost),1)
Phimean = mpost[0:nm, 0]
Claymean = mpost[nm :2 * nm, 0]
Swmean = mpost[2 * nm :, 0]

#% Plot results
plt.figure(3)
plt.subplot(131)
plt.plot(Phipost, Time, 'b')
plt.plot(Phi, Time, 'k')
plt.plot(Phimean, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Porosity')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Claypost, Time, 'b')
plt.plot(Clay, Time, 'k')
plt.plot(Claymean, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Clay volume')
plt.subplot(133)
plt.plot(Sw, Time, 'k')
plt.plot(Swmean, Time, 'r')
plt.plot(Swpost, Time, 'b')
plt.plot(Sw, Time, 'k')
plt.plot(Swmean, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Water saturation')
plt.legend(['Reference model', 'Posterior mean','Posterior Realizations'],loc ="lower right")
plt.show()

