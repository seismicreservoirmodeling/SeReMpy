#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:06:47 2020

"""
import numpy as np
from scipy.stats import multivariate_normal
from scipy import stats

def BayesGaussFaciesClass(data, fprior, muprior, sigmaprior):
    """
    BAYES GAUSS FACIES CLASS
    Computes the Bayesian facies classification of
    the data assuming a Gaussian distribution.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    data : array_like
        Input data (ns, nv).
    fprior : array_like
        Prior facies proportions (nf, 1).
    muprior : array_like
        Prior means on input variables (nf, nv).
    sigmaprior : array_like
        Prior covariance matrices on input
        variables (nv, nv, nf).
    Returns
    -------
    fmap : array_like
        Facies maximum a posteriori (ns, 1).
    fpost : array_like
        Posterior facies probability (ns, nf).    

    """

    # initial parameters
    ns = data.shape[0]
    nf = muprior.shape[0]

    # conditional probability
    fmap = np.zeros((ns, 1))
    fpost = np.zeros((ns, nf))
    for i in range(ns):
        for k in range(nf):
            fpost[i,k] = fprior[k,0] * multivariate_normal.pdf(data[i,:], muprior[k,:], sigmaprior[:,:,k])
        # probability
        fpost[i,:] = fpost[i,:] / np.sum(fpost[i,:])
        # maximum a posteriori
        fmap[i,0] = np.argmax(fpost[i,:])
    return fmap, fpost
    
    
def BayesKDEFaciesClass(data, dtrain, ftrain, fprior, domain):
    """
    BAYES KDE FACIES CLASS
    Computes the Bayesian facies classification of
    the data assuming a non-parametric distribution.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    data : array_like
        Input data (ns, nv).
    dtrain : array_like
        Training data (ntrain, nv).
    ftrain : array_like
        Training facies (ntrain, 1).
    fprior : array_like
        Prior facies proportions (nf, 1).
    domain : array_like
        Discretized domain of input variables
        (generated using meshgrid).
    Returns
    -------
    fmap : array_like
        Facies maximum a posteriori (ns, 1).
    fpost : array_like
        Posterior facies probability (ns, nf).
    """

    # initial parameters
    ns = data.shape[0]
    nf = np.max(np.unique(ftrain))+1
    nd = domain.shape[1]


    # joint distribution
    Pjoint = np.zeros((nd,nf))
    d = dtrain.T
    f = ftrain.T
    for k in range(nf):     
        kde = stats.gaussian_kde(d[:, f[0,:] == k])
        lf = kde(domain)
        Pjoint[:,k] = lf/np.sum(lf)

    # conditional distribution
    fmap = np.zeros((ns, 1))
    fpost = np.zeros((ns, nf))
    for i in range(ns):
        ind = np.argmin(np.sum((domain.T - data[i,:]) ** 2, axis=1))
        for k in range(nf):
            fpost[i,k] = fprior[k,0] * Pjoint[ind, k]
        # probability
        fpost[i,:] = fpost[i,:] / np.sum(fpost[i,:])
        # maximum a posteriori
        fmap[i,0] = np.argmax(fpost[i,:])
    
    return fmap, fpost
    
    
def ConfusionMatrix(ftrue, fpred, nf):
    """
    CONFUSION MATRIX
    Computes the confusion matrix of a discrete classification.
    Written by Dario Grana (August 2020)
    
    Parameters
    ----------
    ftrue : array_like
        true model
    fpred : array_like
        predicted model
    nf : int
        number of possible outcomes (e.g. number of facies)
    Returns
    -------
    confmat : array_like
        confusion matrix (absolute frequencies)
    """

     ns = ftrue.shape[0]
     ftrue = ftrue.astype(int)
     fpred = fpred.astype(int)
     confmat = np.zeros((nf, nf))
     for i in range(ns):
         confmat[ftrue[i], fpred[i]] = confmat[ftrue[i], fpred[i]] + 1
     return confmat
    