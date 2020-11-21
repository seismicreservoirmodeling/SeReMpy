#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:19:26 2020

@author: dariograna
"""

#% Ensemble Smoother Seismic Inversion Driver %%
# In this script we apply the Ensemble Smoother inversion method
# (Liu and Grana, 2018) to predict the elastic properties (P- and S-wave
# velocity and density) from seismic data.

from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from Inversion import *
from Geostats import CorrelatedSimulation

#% Available data and parameters
# Load data (seismic data and time)
s = loadmat('Data/data3.mat')
TimeSeis = s['TimeSeis']
Snear = s['Snear']
Smid = s['Smid']
Sfar = s['Sfar']
Vp = s['Vp']
Vs = s['Vs']
Rho = s['Rho']
Time = s['Time']


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
Vpprior = signal.filtfilt(b, a, np.squeeze(Vp))
Vsprior = signal.filtfilt(b, a, np.squeeze(Vs))
Rhoprior = signal.filtfilt(b, a, np.squeeze(Rho))
mprior = np.hstack([Vpprior[:,np.newaxis], Vsprior[:,np.newaxis], Rhoprior[:,np.newaxis]])

#% Spatial correlation matrix
corrlength = 5 * dt
trow = np.matlib.repmat(np.arange(0, nm * dt, dt), nm, 1)
tcol = np.matlib.repmat(trow[0,:].reshape(nm,1), 1, nm)
tdis = abs(trow - tcol)
sigmatime = np.exp(-(tdis / corrlength) ** 2)
sigma0 = np.cov(np.hstack([Vp, Vs, Rho]).T)
sigmaprior = np.kron(sigma0, sigmatime)

#% Prior realizations
Vpsim = np.zeros((nm, nsim))
Vssim = np.zeros((nm, nsim))
Rhosim = np.zeros((nm, nsim))
SeisPred = np.zeros((nd * ntheta, nsim))
for i in range(nsim):
    msim = CorrelatedSimulation(mprior, sigma0, sigmatime)
    Vpsim[:,i] = msim[:,0]
    Vssim[:,i] = msim[:,1]
    Rhosim[:,i] = msim[:,2]
    seis, TimeSeis = SeismicModel(Vpsim[:,i], Vssim[:,i], Rhosim[:,i], Time, theta, wavelet)
    SeisPred[:,i] = seis[:,0]
    
# plot of prior models
plt.figure(2)
plt.subplot(131)
plt.plot(Vpsim, Time, 'b')
plt.plot(Vp, Time, 'k')
plt.plot(Vpprior, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Vssim, Time, 'b')
plt.plot(Vs, Time, 'k')
plt.plot(Vsprior, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('S-wave velocity (km/s)')
plt.subplot(133)
plt.plot(Rhosim, Time, 'b')
plt.plot(Rho, Time, 'k')
plt.plot(Rhoprior, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Density (g/cm^3)')
plt.show()

#% ESMDA seismic inversion
niter = 4
alpha = 1 / niter   # sum alpha = 1
PriorModels = np.vstack([Vpsim, Vssim, Rhosim])
SeisData = np.vstack([Snear,Smid,Sfar])
PostModels = PriorModels
for j in range(niter):
    PostModels, KalmanGain = EnsembleSmootherMDA(PostModels, SeisData, SeisPred, alpha, sigmaerr)
    Vppost = PostModels[0:nm, :]
    Vspost = PostModels[nm :2 * nm, :]
    Rhopost = PostModels[2 * nm :, :]
    for i in range(nsim):
        seis, TimeSeis = SeismicModel(Vppost[:,i], Vspost[:,i], Rhopost[:,i], Time, theta, wavelet)
        SeisPred[:,i] = seis[:,0]

# posterior mean models
mpost = np.mean(PostModels, axis=1)
mpost = mpost.reshape(len(mpost),1)
Vpmean = mpost[0:nm, 0]
Vsmean = mpost[nm :2 * nm, 0]
Rhomean = mpost[2 * nm :, 0]

#% Plot results
plt.figure(3)
plt.subplot(131)
plt.plot(Vppost, Time, 'b')
plt.plot(Vp, Time, 'k')
plt.plot(Vpmean, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Vspost, Time, 'b')
plt.plot(Vs, Time, 'k')
plt.plot(Vsmean, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('S-wave velocity (km/s)')
plt.subplot(133)
plt.plot(Rhopost, Time, 'b')
plt.plot(Rho, Time, 'k')
plt.plot(Rhomean, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Density (g/cm^3)')
plt.show()