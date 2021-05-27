#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:42:46 2020

@author: dariograna

Reference: Grana and de Figueiredo, 2021, SeReMpy
"""

#% Seismic inversion Driver %%
# In this script we apply the Bayesian linearized AVO inversion method
# (Buland and Omre, 2003) to predict the elastic properties (P- and S-wave
# velocity and density) from seismic data.

from numpy import matlib
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from context import SeReMpy
from SeReMpy.Inversion import *

#% Available data and parameters
# Load data (seismic data and time)
ds = np.loadtxt('Data/data3seis.dat')
TimeSeis = ds[:,0].reshape(-1, 1)
Snear = ds[:,1].reshape(-1, 1)
Smid = ds[:,2].reshape(-1, 1)
Sfar = ds[:,3].reshape(-1, 1)
dl = np.loadtxt('Data/data3log.dat')
Vp = dl[:,0].reshape(-1, 1)
Vs = dl[:,1].reshape(-1, 1)
Rho = dl[:,2].reshape(-1, 1)
Time = dl[:,3].reshape(-1, 1)

#% Initial parameters
# number of samples (elastic properties)
nm = Snear.shape[0]+1
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

#% Spatial correlation matrix
corrlength = 5 * dt
trow = np.matlib.tile(np.arange(0, nm * dt, dt), (nm, 1))
tcol = np.matlib.tile(trow[0,:].reshape(nm,1), (1, nm))
tdis = abs(trow - tcol)
sigmatime = np.exp(-(tdis / corrlength) ** 2)
sigma0 = np.cov(np.hstack([np.log(Vp), np.log(Vs), np.log(Rho)]).T)
sigmaprior = np.kron(sigma0, sigmatime)

#% Seismic inversion
Seis = np.vstack([Snear, Smid, Sfar])
mmap, mlp, mup, t = SeismicInversion(Seis, TimeSeis, Vpprior, Vsprior, Rhoprior, sigmaprior, sigmaerr, wavelet, theta, nv)

Vpmap = mmap[0:nm,0]
Vsmap = mmap[nm:2*nm,0]
Rhomap = mmap[2*nm :,0]
Vplp = mlp[0:nm,0]
Vslp = mlp[nm:2*nm,0]
Rholp = mlp[2*nm:,0]
Vpup = mup[0:nm,0]
Vsup = mup[nm:2*nm,0]
Rhoup = mup[2*nm:,0]

#% Plot results
plt.figure(2)
plt.subplot(131)
plt.plot(Vp, Time, 'k')
plt.plot(Vpprior, Time, 'b')
plt.plot(Vpmap, Time, 'r')
plt.plot(Vplp, Time, 'r--')
plt.plot(Vpup, Time, 'r--')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Vs, Time, 'k')
plt.plot(Vsprior, Time, 'b')
plt.plot(Vsmap, Time, 'r')
plt.plot(Vslp, Time, 'r--')
plt.plot(Vsup, Time, 'r--')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('S-wave velocity (km/s)')
plt.subplot(133)
plt.plot(Rho, Time, 'k')
plt.plot(Rhoprior, Time, 'b')
plt.plot(Rhomap, Time, 'r')
plt.plot(Rholp, Time, 'r--')
plt.plot(Rhoup, Time, 'r--')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('Density (g/cm^3)')
plt.show()
