#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:12:09 2021

@author: dariograna
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from Inversion import *
from Geostats import *
from RockPhysics import *
from numpy import matlib


#% Application5
# Load data (seismic data and time)
s = loadmat('Data/1Ddatab.mat')
TimeSeis = s['TimeSeis']
Snear = s['Snear']
Smid = s['Smid']
Sfar = s['Sfar']
Phi = s['Phi']
Sw = s['Sw']
Time = s['Time']

#% Initial parameters
# number of samples (elastic properties)
nm = Snear.shape[0]+1
# number of samples (seismic data)
nd = Snear.shape[0]
# number of variables
nv = 2
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

# rock physics parameters
criticalporo = 0.4
coordnumber = 9
pressure = 0.04
Kmat = 30
Gmat = 60
Rhomat = 2.6
Kw = 2.5
Ko = 0.7
Rhow = 1.03
Rhoo = 0.7

#% Prior model (filtered well logs)
nfilt = 3
cutofffr = 0.04
b, a = signal.butter(nfilt, cutofffr)
Phiprior = signal.filtfilt(b, a, np.squeeze(Phi))
Swprior = signal.filtfilt(b, a, np.squeeze(Sw))
mprior = np.vstack([Phiprior[:,np.newaxis], Swprior[:,np.newaxis]])
d = np.vstack([Snear,Smid,Sfar])



#% Spatial correlation matrix
corrlength = 5 * dt
trow = np.matlib.repmat(np.arange(0, nm * dt, dt), nm, 1)
tcol = np.matlib.repmat(trow[0,:].reshape(nm,1), 1, nm)
tdis = abs(trow - tcol)
sigmatime = np.exp(-(tdis / corrlength) ** 2)
sigma0 = np.cov(np.hstack([Phi, Sw]).T)
sigmaprior = np.kron(sigma0, sigmatime)
InvCovmatrixPrior = np.linalg.pinv(sigmaprior)

# %  error model
Errnear = 0.2 * np.var(Snear)# var(noise) = var(data)/SNR
Errmid = 0.2 * np.var(Smid)# var(noise) = var(data)/SNR
Errfar = 0.2 * np.var(Sfar)# var(noise) = var(data)/SNR
Errvar = np.diag(np.array([Errnear, Errmid, Errfar]))
InvCovErr = np.linalg.pinv(np.kron(Errvar, np.eye(nd)))

# proposal variance
varfrac = 0.001# variance fraction in the proposal

# Initial model 
mtrend = np.hstack([Phiprior[:,np.newaxis], Swprior[:,np.newaxis]])
msim = CorrelatedSimulation(mtrend, varfrac*sigma0, sigmatime)
Phiinit = msim[:,0]
Swinit = msim[:,1]
Phiinit[Phiinit < 0] = 0
Phiinit[Phiinit > 0.4] = 0.4
Swinit[Swinit < 0] = 0
Swinit[Swinit > 1] = 1
    
Kfl = Swinit*Kw+(1-Swinit)*Ko;
Rhofl = Swinit*Rhow+(1-Swinit)*Rhoo;
Rho = DensityModel(Phiinit, Rhomat, Rhofl);
Vp, Vs = StiffsandModel(Phiinit, Rho, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure);
Seis, TimeSeis = SeismicModel (Vp, Vs, Rho, Time, theta, wavelet);
Snearinit = Seis[0:nd];
Smidinit = Seis[nd:2*nd];
Sfarinit = Seis[2*nd:];

mold = np.vstack([Phiinit[:,np.newaxis],Swinit[:,np.newaxis]])
dold = np.vstack([Snearinit,Smidinit,Sfarinit])

Logpriorold = np.dot((mold - mprior).T , np.dot(InvCovmatrixPrior, (mold - mprior)))
Loglikeold = np.dot((dold - d).T, np.dot(InvCovErr, (dold - d)))

niter = 10 ** 4
chain = np.zeros((nm*nv, niter))
chainlike = np.zeros((niter,1))
chain[:,0] = mold[:,0]
chainlike[0] = Loglikeold
ar = 0


for j in range(niter-1):
    Phiold = mold[0:nm,0]
    Swold = mold[nm:,0]
    mtrend = np.hstack([Phiold[:,np.newaxis], Swold[:,np.newaxis]])
    mprop = CorrelatedSimulation(mtrend, varfrac * sigma0, sigmatime)
    Phiprop = mprop[:,0]
    Swprop = mprop[:,1]
    Phiprop[Phiprop < 0] = 0
    Phiprop[Phiprop > 0.4] = 0.4
    Swprop[Swprop < 0] = 0
    Swprop[Swprop > 1] = 1

    Kfl = Swprop*Kw+(1-Swprop)*Ko;
    Rhofl = Swprop*Rhow+(1-Swprop)*Rhoo;
    Rho = DensityModel(Phiprop, Rhomat, Rhofl);
    Vp, Vs = StiffsandModel(Phiprop, Rho, Kmat, Gmat, Kfl, criticalporo, coordnumber, pressure);
    Seis, TimeSeis = SeismicModel (Vp, Vs, Rho, Time, theta, wavelet);
    Snearprop = Seis[0:nd];
    Smidprop= Seis[nd:2*nd];
    Sfarprop= Seis[2*nd:];
   
    mprop = np.vstack([Phiprop[:,np.newaxis],Swprop[:,np.newaxis]])
    dprop = np.vstack([Snearprop, Smidprop,Sfarprop])
    
    Logpriorprop = np.dot((mprop - mprior).T, np.dot(InvCovmatrixPrior, (mprop - mprior)))
    Loglikeprop = np.dot((dprop - d).T, np.dot(InvCovErr, (dprop - d)))

    # Likelihood and prior
    MetrHast = np.exp(-1 / 2 * (Logpriorprop - Logpriorold)) * np.exp(-1 / 2 * (Loglikeprop - Loglikeold))

    u = np.random.rand(1)

    if u <= np.minimum(MetrHast, 1):    # if prop>old then MetrHast>1 then min(MetrHast,1)=1 then u<=1  then we accept
        # if MetrHast< 1  then  min(MetrHast,1)=MetrHast (A) if u<MetrHast
        # then accept (B) if u>MetrHast then reject
        mold = mprop
        Logpriorold = Logpriorprop
        Loglikeold = Loglikeprop
        ar = ar + 1

    chain[:,j+1]= mold[:,0]
    chainlike[j+1] = Loglikeold

Acceptance_Ratio = ar / niter * 100

postmodels = chain[:, 1000:]
postphi = postmodels[0:nm,:]
postsw = postmodels[nm:,:]
Phipredicted = np.mean(postphi, axis=1)
Swpredicted = np.mean(postsw, axis=1)

plt.figure(3)
plt.subplot(131)
ax = plt.subplot(141)
plt.plot(Snear-0.1, TimeSeis, 'k')
plt.plot(Smid, TimeSeis, 'k')
plt.plot(Sfar+0.1, TimeSeis, 'k')
plt.grid()
plt.ylim(max(TimeSeis),min(TimeSeis))
plt.xlabel('Near')
plt.ylabel('Time (s)')
ax = plt.subplot(132)
plt.plot(postphi, Time, 'b')
plt.plot(Phi, Time, 'k')
plt.plot(Phipredicted, Time, 'r')
plt.grid()
plt.ylim(max(Time),min(Time))
plt.xlabel('Porosity')
yticks = ax.get_yticks() 
ax.set_yticks(yticks) 
ax.set_yticklabels([])
ax = plt.subplot(133)
plt.plot(postsw, Time, 'b')
plt.plot(Sw, Time, 'k')
plt.plot(Swpredicted, Time, 'r')
plt.grid()
plt.ylim([max(Time), min(Time)])
plt.xlabel('Water saturation')
ax.set_yticks(yticks) 
ax.set_yticklabels([])
plt.show()

