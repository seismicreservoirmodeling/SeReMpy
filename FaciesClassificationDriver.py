#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:31:23 2020

"""

#% Facies Classification Driver %%
# In this script we illustrate Bayesian facies classification using 
# two assumptions: 
# Example 1: Gaussian model
# Example 2: non-parametric model estimated using KDE

from scipy.io import loadmat
import scipy.spatial.distance
from scipy import stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from Geostats import *
from Facies import *

# available data
x = np.loadtxt('Data/data4.dat')
Depth = x[:,1].reshape(-1, 1)
Facies = x[:,2].reshape(-1, 1)
Rho = x[:,4].reshape(-1, 1)
Vp = x[:,7].reshape(-1, 1)
Facies = Facies-1
Facies = Facies.astype(int)
data =  np.hstack([Vp, Rho])
ns = data.shape[0]
nv = data.shape[1]

# domain
v,r = np.mgrid[3:5.005:0.005, 2:2.81:.01]
domain = np.dstack((v, r))

#% Gaussian model (2 components)
nf = max(np.unique(Facies))+1
fp = np.zeros((nf, 1))
mup = np.zeros((nf, nv))
sp = np.zeros((nv, nv, nf))
for k in range(nf):
    fp[k,0] = np.sum(Facies == k) / ns
    mup[k, :] = np.mean(data[Facies[:,0] == k, :],axis=0)
    sp[:,:,k] = np.cov(np.transpose(data[Facies[:,0] == k, :]))

# likelihood function
GaussLikeFun = np.zeros((domain.shape[0], domain.shape[1], nf))
for k in range(nf):
    lf = multivariate_normal.pdf(domain, mup[k,:], sp[:,:,k])
    GaussLikeFun[:,:,k]  = lf / np.sum(lf.reshape(-1))

# plot likelihood
plt.figure(1)
plt.plot(data[:,0], data[:,1],'.k')
for k in range(nf):
    plt.contour(v, r, GaussLikeFun[:,:,k])
plt.grid()
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Density (g/cm^3)')
plt.show()

# classification
fmap, fpost = BayesGaussFaciesClass(data, fp, mup, sp)

# confusion matrix (absolute frequencies)
confmat = ConfusionMatrix(Facies, fmap, nf)

# plot results
plt.figure(2)
plt.subplot(141)
plt.plot(Vp, Depth, 'k')
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Depth (m)')
plt.grid()
plt.ylim(max(Depth),min(Depth))
plt.subplot(142)
plt.plot(Rho, Depth, 'k')
plt.xlabel('Density (g/cm^3)')
plt.grid()
plt.ylim(max(Depth),min(Depth))
plt.subplot(143)
plt.plot(fpost, Depth, 'k')
plt.xlabel('Facies probability')
plt.grid()
plt.ylim(max(Depth),min(Depth))
plt.subplot(144)
plt.pcolor(np.arange(2), Depth, fmap)
plt.xlabel('Predicted facies')
plt.ylabel('Depth (m)')
plt.ylim(max(Depth),min(Depth))
plt.show()


#% Non parametric model (2 components)
# training data
dtrain = data
ftrain = Facies

dspace = np.vstack([v.ravel(), r.ravel()])

# likelihood function
KDElikefun = np.zeros((v.shape[0], v.shape[1], nf))
d = dtrain.T
f = ftrain.T
for k in range(nf):    
    kde = stats.gaussian_kde(d[:, f[0,:] == k])
    lf = kde(dspace)
    lf = lf/np.sum(lf)
    KDElikefun[:,:,k]  = np.reshape(lf.T, v.shape)

# plot likelihood
plt.figure(3)
plt.plot(data[:,0], data[:,1],'.k')
for k in range(nf):
    plt.contour(v, r, KDElikefun[:,:,k])
plt.grid()
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Density (g/cm^3)')
plt.show()


# classification
fmap, fpost = BayesKDEFaciesClass(data, dtrain, ftrain, fp, dspace)

# confusion matrix (absolute frequencies)
confmat = ConfusionMatrix(Facies, fmap, nf)

# plot results
plt.figure(4)
plt.subplot(141)
plt.plot(Vp, Depth, 'k')
plt.xlabel('P-wave velocity (km/s)')
plt.ylabel('Depth (m)')
plt.grid()
plt.ylim(max(Depth),min(Depth))
plt.subplot(142)
plt.plot(Rho, Depth, 'k')
plt.xlabel('Density (g/cm^3)')
plt.grid()
plt.ylim(max(Depth),min(Depth))
plt.subplot(143)
plt.plot(fpost, Depth, 'k')
plt.xlabel('Facies probability')
plt.grid()
plt.ylim(max(Depth),min(Depth))
plt.subplot(144)
plt.pcolor(np.arange(2), Depth, fmap)
plt.xlabel('Predicted facies')
plt.ylabel('Depth (m)')
plt.ylim(max(Depth),min(Depth))
plt.show()