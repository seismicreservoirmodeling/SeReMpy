#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 07:24:51 2020

@author: dariograna
"""

#% Rock physics inversion Driver %%
# In this script we apply the Bayesian Rock phyisics inversion to predict 
# the petrophysical properties (porosity, clay volume, and water saturation
# We implement 4 different options:
# Gaussian distribution and linear model 
# Gaussian mixture distribution and linear model (Grana, 2016)
# Gaussian mixture distribution and non-linear model (Grana and Della Rossa, 2010)
# Non-parametric distribution  and non-linear model (Grana, 2018).
# The linear rock physics model is a multi-linear regression and it is
# estimated from a training dataset.
# In this implementation of the non-linear model we assume that the joint
# distribution of model and % data can be estimated from a training dataset 
# (generated, for example, using a rock physics model)

import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import multi_dot
from scipy import stats
from scipy.io import loadmat
import matplotlib.pyplot as plt
from RockPhysics import *
from Inversion import *


#% Available data and parameters
# Load data (seismic data and time)
x = np.loadtxt('Data/data4.dat')
Clay = x[:,0].reshape(-1, 1)
Depth = x[:,1].reshape(-1, 1)
Facies = x[:,2].reshape(-1, 1)
Phi = x[:,3].reshape(-1, 1)
Rho = x[:,4].reshape(-1, 1)
Rhorpm = x[:,5].reshape(-1, 1)
Sw = x[:,6].reshape(-1, 1)
Vp = x[:,7].reshape(-1, 1)
Vprpm = x[:,8].reshape(-1, 1)
Vs = x[:,9].reshape(-1, 1)
Vsrpm = x[:,10].reshape(-1, 1)

Facies = Facies-1
Facies = Facies.astype(int)

# training dataset
mtrain = np.hstack([Phi, Clay, Sw])
nv = mtrain.shape[1]
dtrain = np.hstack([Vprpm, Vsrpm, Rhorpm])
nd = dtrain.shape[1]
nf = max(np.unique(Facies))+1

# domain to evaluate the posterior PDF
phidomain = np.arange(0,0.405,0.005)
cdomain = np.arange(0,0.81,0.01)
swdomain = np.arange(0,1.01,0.01)
P, V, S = np.mgrid[0:0.405:0.005, 0:0.81:0.01, 0:1.01:0.01]
mdomain = np.stack((P,V,S), axis=3)

# measured data (elastic logs)
dcond = np.hstack([Vp, Vs, Rho])
ns = dcond.shape[0]

# matrix associated to the linear rock physics operator
R = np.zeros((nd, nv + 1))
X = np.hstack([mtrain, np.ones(Phi.shape)])
R[0, :] = (np.linalg.lstsq(X,Vprpm)[0]).T
R[1, :] = (np.linalg.lstsq(X,Vsrpm)[0]).T
R[2, :] = (np.linalg.lstsq(X,Rhorpm)[0]).T

# Error
sigmaerr = 10 ** -2 * np.eye(nd)


#% Gaussian linear case
# prior model
mum = np.mean(mtrain,axis=0)
mum = mum.reshape(1,nv)
sm = np.cov(mtrain.T)

# linearization
G = R[:,0:nv]
datacond = dcond - R[:,-1].T

# # inversion
[mupost, sigmapost, Ppost] = RockPhysicsLinGaussInversion(mum, sm, G, mdomain, datacond, sigmaerr)

# posterior mean
Phipost = mupost[:, 0]
Cpost = mupost[:, 1]
Swpost = mupost[:, 2]
Philp = mupost[:, 0] - 1.96 * np.sqrt(sigmapost[0,0])
Clp = mupost[:, 1] - 1.96 * np.sqrt(sigmapost[1,1])
Swlp = mupost[:, 2] - 1.96 * np.sqrt(sigmapost[2,2])
Phiup = mupost[:, 0] + 1.96 * np.sqrt(sigmapost[0,0])
Cup = mupost[:, 1] + 1.96 * np.sqrt(sigmapost[1,1])
Swup = mupost[:, 2] + 1.96 * np.sqrt(sigmapost[2,2])

# marginal posterior distributions
Ppostphi = np.zeros((ns, len(phidomain)))
Ppostclay = np.zeros((ns, len(cdomain)))
Ppostsw = np.zeros((ns, len(swdomain)))
for i in range(ns):
    Ppostphi[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=2)), axis=1)
    Ppostclay[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=2)), axis=0)
    Ppostsw[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=1)), axis=0)
    Ppostphi[i,:]= Ppostphi[i,:] / sum(Ppostphi[i,:])
    Ppostclay[i,:]= Ppostclay[i,:] / sum(Ppostclay[i,:])
    Ppostsw[i,:]= Ppostsw[i,:] / sum(Ppostsw[i,:])

# plots
plt.figure(1)
plt.subplot(131)
plt.pcolor(phidomain, Depth, Ppostphi)
plt.colorbar()
plt.plot(Phi, Depth, 'k')
plt.plot(Phipost, Depth, 'r')
plt.ylabel('Depth (m)')
plt.xlabel('Porosity')
plt.xlim([0, 0.4])
plt.ylim([max(Depth), min(Depth)])
plt.subplot(132)
plt.pcolor(cdomain, Depth, Ppostclay)
plt.colorbar()
plt.plot(Clay, Depth, 'k')
plt.plot(Cpost, Depth, 'r')
plt.xlabel('Clay volume')
plt.xlim([0, 0.8])
plt.ylim([max(Depth), min(Depth)])
plt.subplot(133)
plt.pcolor(swdomain, Depth, Ppostsw)
plt.colorbar()
plt.plot(Sw, Depth, 'k')
plt.plot(Swpost, Depth, 'r')
plt.ylabel('Depth (m)')
plt.xlabel('Water saturation')
plt.xlim([0, 1])
plt.ylim([max(Depth), min(Depth)])
plt.show()


#% Gaussian mixture linear case
# prior model
pf = np.zeros((nf,1))
mum = np.zeros((nf,nv))
sm = np.zeros((nv,nv,nf))
for k in range(nf):
    pf[k,0] = np.sum(Facies == k) / ns
    mum[k,:] = np.mean(mtrain[Facies[:,0] == k,:],axis=0)
    sm[:,:,k] = np.cov(mtrain[Facies[:,0] == k,:].T)
                       
mupost, sigmapost, pfpost, Ppost = RockPhysicsLinGaussMixInversion(pf, mum, sm, G, mdomain, datacond, sigmaerr)

# marginal posterior distributions
Ppostphi = np.zeros((ns, len(phidomain)))
Ppostclay = np.zeros((ns, len(cdomain)))
Ppostsw = np.zeros((ns, len(swdomain)))
Phimap = np.zeros((ns, 1))
Cmap = np.zeros((ns, 1))
Swmap = np.zeros((ns, 1))
for i in range(ns):
    Ppostphi[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=2)), axis=1)
    Ppostclay[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=2)), axis=0)
    Ppostsw[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=1)), axis=0)
    Ppostphi[i,:]= Ppostphi[i,:] / sum(Ppostphi[i,:])
    Ppostclay[i,:]= Ppostclay[i,:] / sum(Ppostclay[i,:])
    Ppostsw[i,:]= Ppostsw[i,:] / sum(Ppostsw[i,:])
    Phimapind = np.argmax(Ppostphi[i,:])
    Cmapind = np.argmax(Ppostclay[i,:])
    Swmapind = np.argmax(Ppostsw[i,:])
    Phimap[i,0]= phidomain[Phimapind]
    Cmap[i,0]= cdomain[Cmapind]
    Swmap[i,0]= swdomain[Swmapind]
    

# plots
plt.figure(2)
plt.subplot(131)
plt.pcolor(phidomain, Depth, Ppostphi)
plt.colorbar()
plt.plot(Phi, Depth, 'k')
plt.plot(Phimap, Depth, 'r')
plt.ylabel('Depth (m)')
plt.xlabel('Porosity')
plt.xlim([0, 0.4])
plt.ylim([max(Depth), min(Depth)])
plt.subplot(132)
plt.pcolor(cdomain, Depth, Ppostclay)
plt.colorbar()
plt.plot(Clay, Depth, 'k')
plt.plot(Cmap, Depth, 'r')
plt.xlabel('Clay volume')
plt.xlim([0, 0.8])
plt.ylim([max(Depth), min(Depth)])
plt.subplot(133)
plt.pcolor(swdomain, Depth, Ppostsw)
plt.colorbar()
plt.plot(Sw, Depth, 'k')
plt.plot(Swmap, Depth, 'r')
plt.ylabel('Depth (m)')
plt.xlabel('Water saturation')
plt.xlim([0, 1])
plt.ylim([max(Depth), min(Depth)])
plt.show()


#% Gaussian mixture case
# The joint Gaussian mixture distribution is estimated from the training dataset
mupost, sigmapost, pfpost, Ppost = RockPhysicsGaussMixInversion(Facies, mtrain, dtrain, mdomain, dcond, sigmaerr)
# The joint Gaussian distribution can also be used
# mupost, sigmapost, Ppost = RockPhysicsGaussInversion(mtrain, dtrain, mdomain, dcond, sigmaerr);

# marginal posterior distributions
Ppostphi = np.zeros((ns, len(phidomain)))
Ppostclay = np.zeros((ns, len(cdomain)))
Ppostsw = np.zeros((ns, len(swdomain)))
Phimap = np.zeros((ns, 1))
Cmap = np.zeros((ns, 1))
Swmap = np.zeros((ns, 1))
for i in range(ns):
    Ppostphi[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=2)), axis=1)
    Ppostclay[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=2)), axis=0)
    Ppostsw[i,:]= np.sum(np.squeeze(np.sum(np.squeeze(Ppost[i,:,:,:]), axis=1)), axis=0)
    Ppostphi[i,:]= Ppostphi[i,:] / sum(Ppostphi[i,:])
    Ppostclay[i,:]= Ppostclay[i,:] / sum(Ppostclay[i,:])
    Ppostsw[i,:]= Ppostsw[i,:] / sum(Ppostsw[i,:])
    Phimapind = np.argmax(Ppostphi[i,:])
    Cmapind = np.argmax(Ppostclay[i,:])
    Swmapind = np.argmax(Ppostsw[i,:])
    Phimap[i,0]= phidomain[Phimapind]
    Cmap[i,0]= cdomain[Cmapind]
    Swmap[i,0]= swdomain[Swmapind]
    

# plots
plt.figure(3)
plt.subplot(131)
plt.pcolor(phidomain, Depth, Ppostphi)
plt.colorbar()
plt.plot(Phi, Depth, 'k')
plt.plot(Phimap, Depth, 'r')
plt.ylabel('Depth (m)')
plt.xlabel('Porosity')
plt.xlim([0, 0.4])
plt.ylim([max(Depth), min(Depth)])
plt.subplot(132)
plt.pcolor(cdomain, Depth, Ppostclay)
plt.colorbar()
plt.plot(Clay, Depth, 'k')
plt.plot(Cmap, Depth, 'r')
plt.xlabel('Clay volume')
plt.xlim([0, 0.8])
plt.ylim([max(Depth), min(Depth)])
plt.subplot(133)
plt.pcolor(swdomain, Depth, Ppostsw)
plt.colorbar()
plt.plot(Sw, Depth, 'k')
plt.plot(Swmap, Depth, 'r')
plt.ylabel('Depth (m)')
plt.xlabel('Water saturation')
plt.xlim([0, 1])
plt.ylim([max(Depth), min(Depth)])
plt.show()


# % Non-parametric case (Kernel density estimation)
## Inefficient implementation ##

# # phidomain = np.arange(0,0.425,0.025)
# # cdomain = np.arange(0,0.85,0.05)
# # swdomain = np.arange(0,1.05,0.05)
# P, V, S, VP, VS, R= np.mgrid[0:0.42:0.02, 0:0.85:0.05, 0:1.05:0.05, min(Vp):max(Vp):(max(Vp)-min(Vp))/30, min(Vs):max(Vs):(max(Vs)-min(Vs))/30, min(Rho):max(Rho):(max(Rho)-min(Rho))/30]
# jointdomain = np.vstack([P.ravel(), V.ravel(), S.ravel(), VP.ravel(), VS.ravel(), R.ravel()])
# datadomain = np.vstack([VP[0,0,0,:,0,0], VS[0,0,0,0,:,0], R[0,0,0,0,0,:]])
# phidomain=P[:,0,0,0,0,0]
# cdomain=V[0,:,0,0,0,0]
# swdomain=S[0,0,:,0,0,0]
# jointdim = P.shape 
# mdim = P[:,:,:,0,0,0].shape 
# # # inversion
# Ppost = RockPhysicsKDEInversion(mtrain, dtrain, jointdomain, datadomain, dcond, jointdim, mdim)

