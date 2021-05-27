#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:36:59 2020

@author: dariograna

Reference: Grana and de Figueiredo, 2021, SeReMpy
"""

#% Seismic Model Driver %%
# In this script we apply the convolutional model of a wavelet and the
# linearized approximation of Zoeppritz equations to compute synthetic
# seismograms for different reflection angles
# The model parameterization is expressed in terms of P- and S-wave
# velocity and density.

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from context import SeReMpy
from SeReMpy.Inversion import *

#% Available data and parameters
# Load data (elastic properties and depth)
dl = np.loadtxt('Data/data2.dat')
Depth = dl[:,0].reshape(-1, 1)
Rho = dl[:,1].reshape(-1, 1)
Vp = dl[:,3].reshape(-1, 1)
Vs = dl[:,4].reshape(-1, 1)

#% Initial parameters
# number of variables
nv = 3
# reflection angles 
theta = np.linspace(15,45,3)

# travel time
dt = 0.001
t0 = 1.8
temp = np.cumsum(np.diff(Depth,axis=0)/Vp[1:])
TimeLog = np.append(t0, t0+2*temp)
Time = np.arange(TimeLog[0],TimeLog[-1],dt)

# time-interpolated elastic log
Vp = np.interp(Time, TimeLog, np.squeeze(Vp))
Vs = np.interp(Time, TimeLog, np.squeeze(Vs))
Rho = np.interp(Time, TimeLog, np.squeeze(Rho))

# number of samples (seismic properties)
nd = len(Vp) - 1

#% Wavelet
# wavelet 
freq = 45
ntw = 64
wavelet, tw = RickerWavelet(freq, dt, ntw)


#% Plot elastic data
plt.figure(1)
plt.subplot(131)
plt.plot(Vp, Time, 'k')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Time (s)')
plt.subplot(132)
plt.plot(Vs, Time, 'k')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('S-wave velocity (km/s)')
plt.subplot(133)
plt.plot(Rho, Time, 'k')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('Density (g/cm^3)')
plt.show()

#% Synthetic seismic data
Seis, TimeSeis = SeismicModel(Vp, Vs, Rho, Time, theta, wavelet)
Snear = Seis[0:nd]
Smid = Seis[nd:2 * nd]
Sfar = Seis[2 * nd :]


#% Plot seismic data
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