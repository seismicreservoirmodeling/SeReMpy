#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:09:34 2021

@author: dariograna

Reference: Grana and De Figueiredo, 2021, SeReMpy
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from Inversion import *
from numpy import matlib


#% Application2
# Load data (seismic data and time)
E = np.loadtxt('Data/2Ddataelas.dat')
nx = 67
ny = 85
Vp = E[:,0].reshape(nx, ny)
Vs = E[:,1].reshape(nx, ny)
Rho = E[:,2].reshape(nx, ny)
Time = E[:,3].reshape(nx, ny)
S = np.loadtxt('Data/2Ddataseis.dat')
nx = 66
ny = 85
Snear = S[:,0].reshape(nx, ny)
Smid = S[:,1].reshape(nx, ny)
Sfar = S[:,2].reshape(nx, ny)
TimeSeis = S[:,3].reshape(nx, ny)

#% Initial parameters
# number of traces
ntr = Snear.shape[1]
# number of samples (elastic properties)
nm = Snear.shape[0]+1
# number of variables
nv = 3
# reflection angles 
theta = np.linspace(15,45,3)
ntheta = len(theta)
# time sampling
dt = TimeSeis[1,0] - TimeSeis[0,0]
# error variance
varerr = 10 ** -4
sigmaerr = varerr * np.eye(ntheta * (nm - 1))


#% Wavelet
# wavelet 
freq = 45
ntw = 64
wavelet, tw = RickerWavelet(freq, dt, ntw)

#% Prior model (filtered well logs)
nfilt = 3
cutofffr = 0.04
b, a = signal.butter(nfilt, cutofffr)
Vpprior = np.zeros((nm,ntr))
Vsprior = np.zeros((nm,ntr))
Rhoprior = np.zeros((nm,ntr))
for i in range(ntr):
    Vpprior[:,i] = signal.filtfilt(b, a, np.squeeze(Vp[:,i]))
    Vsprior[:,i] = signal.filtfilt(b, a, np.squeeze(Vs[:,i]))
    Rhoprior[:,i] = signal.filtfilt(b, a, np.squeeze(Rho[:,i]))

#% Spatial correlation matrix
corrlength = 5 * dt
trow = np.matlib.repmat(np.arange(0, nm * dt, dt), nm, 1)
tcol = np.matlib.repmat(trow[0,:].reshape(nm,1), 1, nm)
tdis = abs(trow - tcol)
sigmatime = np.exp(-(tdis / corrlength) ** 2)
sigma0 = np.cov(np.hstack([np.log(Vp[:,0].reshape(-1,1)), np.log(Vs[:,0].reshape(-1,1)), np.log(Rho[:,0].reshape(-1,1))]).T)
sigmaprior = np.kron(sigma0, sigmatime)

#% Seismic inversion
Vpmap = np.zeros((nm,ntr))
Vsmap = np.zeros((nm,ntr))
Rhomap = np.zeros((nm,ntr))
for i in range(ntr):
    Seis = np.vstack([Snear[:,i], Smid[:,i], Sfar[:,i]])
    Seis = np.reshape(Seis, (Seis.shape[0]*(nm-1),1))
    mmap, mlp, mup, t = SeismicInversion(Seis, TimeSeis[:,i], Vpprior[:,i], Vsprior[:,i], Rhoprior[:,i], sigmaprior, sigmaerr, wavelet, theta, nv)
    Vpmap[:,i] = mmap[0:nm,0]
    Vsmap[:,i] = mmap[nm:2*nm,0]
    Rhomap[:,i] = mmap[2*nm :,0]

# plots data and results
plt.figure(2)
plt.subplot(321)
img = plt.pcolor(np.linspace(1,ntr,ntr), TimeSeis[:,0], Snear, cmap = 'gray')
plt.xlabel('Trace number')
plt.ylabel('Time (s)')
cbar = plt.colorbar()
cbar.set_label('Seismic', rotation=270)
plt.title('Near')
plt.ylim(max(TimeSeis[:,0]),min(TimeSeis[:,0]))
plt.subplot(323)
plt.pcolor(np.linspace(1,ntr,ntr), TimeSeis[:,0], Smid, cmap = 'gray')
plt.xlabel('Trace number')
plt.ylabel('Time (s)')
cbar = plt.colorbar()
cbar.set_label('Seismic', rotation=270)
plt.title('Mid')
plt.ylim(max(TimeSeis[:,0]),min(TimeSeis[:,0]))
plt.subplot(325)
plt.pcolor(np.linspace(1,ntr,ntr), TimeSeis[:,0], Sfar, cmap = 'gray')
plt.xlabel('Trace number')
plt.ylabel('Time (s)')
cbar = plt.colorbar()
cbar.set_label('Seismic', rotation=270)
plt.title('Far')
plt.ylim(max(TimeSeis[:,0]),min(TimeSeis[:,0]))
plt.subplot(322)
Time = Time[:,0];
plt.pcolor(np.linspace(1,ntr,ntr), Time, Vpmap)
plt.xlabel('Trace number')
plt.ylabel('Time (s)')
cbar = plt.colorbar()
cbar.set_label('Velocity', rotation=270)
plt.title('P-wave velocity (m/s)')
plt.ylim(max(Time),min(Time))
plt.subplot(324)
plt.pcolor(np.linspace(1,ntr,ntr), Time, Vsmap)
plt.xlabel('Trace number')
plt.ylabel('Time (s)')
cbar = plt.colorbar()
cbar.set_label('velocity', rotation=270)
plt.title('S-wave velocity (m/s)')
plt.ylim(max(Time),min(Time))
plt.subplot(326)
plt.pcolor(np.linspace(1,ntr,ntr), Time, Rhomap)
plt.xlabel('Trace number')
plt.ylabel('Time (s)')
cbar = plt.colorbar()
cbar.set_label('Density', rotation=270)
plt.title('Density (g/cm^3)')
plt.ylim(max(Time),min(Time))
plt.show()