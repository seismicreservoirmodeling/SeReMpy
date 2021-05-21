#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:30:38 2020

"""
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
from scipy import stats


def AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv):
    """
    AKI RICHARDS COEFFICIENTS MATRIX computes the Aki Richards coefficient
    matrix
    Parameters:
          Vp = P-wave velocity profile
          Vs = S-wave velocity profile
          theta = vector of reflection angles
          nv = number of model variables 
    Returns:
           A = Aki Richards coefficients matrix

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    # initial parameters
    nsamples = Vp.shape[0]
    ntheta = len(theta)
    A = np.zeros(( (nsamples-1)*ntheta, nv*(nsamples-1)))

    # average velocities at the interfaces
    avgVp = 1 / 2 * (Vp[0:- 1] + Vp[1:])
    avgVs = 1 / 2 * (Vs[0:- 1] + Vs[1:])

    # reflection coefficients (Aki Richards linearized approximation)
    for i in range(ntheta):
        cp = 1 / 2 * (1 + np.tan(theta[i]*np.pi / 180) ** 2) * np.ones(nsamples - 1)
        cs = -4 * (avgVs ** 2) / (avgVp ** 2) * np.sin(theta[i]*np.pi / 180) ** 2
        cr = 1 / 2 * (1 - 4 * (avgVs ** 2) / (avgVp ** 2) * np.sin(theta[i]*np.pi / 180) ** 2)
        Acp = np.diag(cp)
        Acs = np.diag(cs)
        Acr = np.diag(cr)
        A[ i*(nsamples-1) : (i+1)*(nsamples-1), : ] = np.hstack([Acp, Acs, Acr])
   
    return A
    
def DifferentialMatrix(nt, nv):
    """
    DIFFERENTIAL MATRIX computes the differential matrix for discrete
    differentiation
    Parameters:
          nt = number of samples
          nv = numbr of model variables
    Returns:
           D = differential matrix

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    I = np.eye(nt)
    B = np.zeros((nt, nt))
    B[1:, 0:- 1] = -np.eye(nt-1)
    I = (I + B)
    J = I[1:,:];
    D = np.zeros(((nt-1)*nv, nt*nv))
    for i in range(nv):
        D[ i*(nt-1):(i+1)*(nt-1),i*nt:(i+1)*nt] = J
        
    return D
    
def RickerWavelet(freq, dt, ntw):
    """
    RICKER WAVELET computes the Ricker wavelet
    Parameters:
          freq = dominant frequency
          dt = time sampling rate
          ntw = number of samples of the wavelet
    Returns:
           w = wavelet
           tw = wavelet time

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    tmin = -dt * np.round(ntw / 2)
    tw = tmin + dt * np.arange(0, ntw)
    w = (1 - 2. * (np.pi ** 2 * freq ** 2) * tw ** 2) * np.exp(-(np.pi ** 2 * freq ** 2) * tw ** 2)
    
    return w, tw 

def SeismicInversion(Seis, TimeSeis, Vpprior, Vsprior, Rhoprior, sigmaprior, sigmaerr, wavelet, theta, nv):
    """
    SEISMIC INVERSION computes the posterior distribution of elastic
    properties according to the Bayesian linearized AVO inversion (Buland and
    Omre, 2003)
    Parameters:
          Seis = vector of seismic data of size (nsamples x nangles, 1)
          TimeSeis = vector of seismic time of size (nsamples, 1)
          Vpprior = vector of prior (low frequency) Vp model (nsamples+1, 1)
          Vsprior = vector of prior (low frequency) Vs model (nsamples+1, 1)
          Rhoprior = vector of prior (low frequency) density model (nsamples+1, 1)
          sigmaprior = prior covariance matrix (nv*(nsamples+1),nv*(nsamples+1))
          sigmaerr = covariance matrix of the error (nv*nsamples,nv*nsamples)
          theta = vector of reflection angles (1,nangles)
          nv = number of model variables
    Returns:
           mmap = MAP of posterior distribution (nv*(nsamples+1),1)
           mlp = P2.5 of posterior distribution (nv*(nsamples+1),1)
           mup = P97.5 of posterior distribution (nv*(nsamples+1),1)
           Time = time vector of elastic properties (nsamples+1,1)

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.2
    Grana and De Figueiredo, 2021, SeReMpy - Equations 5 and 6
    """

    # parameters
    ntheta = len(theta)

    # logarithm of the prior
    logVp = np.log(Vpprior)
    logVs = np.log(Vsprior)
    logRho = np.log(Rhoprior)
    mprior = np.hstack([logVp, logVs, logRho])
    mprior = mprior.reshape(len(mprior),1)
    nm = logVp.shape[0]
    

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vpprior, Vsprior, theta, nv)

    # Differential matrix 
    D = DifferentialMatrix(nm, nv)

    # Wavelet matrix
    W = WaveletMatrix(wavelet, nm, ntheta)

    # forward operator
    G = multi_dot([W,A,D])

    # Bayesian Linearized AVO inverison analytical solution (Buland and Omre, 2003)
    # mean of d
    mdobs = np.dot(G, mprior)
    # covariance matrix
    sigmadobs = multi_dot([G, sigmaprior, G.T]) + sigmaerr

    # posterior mean
    mpost = mprior + np.dot((np.dot(G, sigmaprior)).T , np.linalg.lstsq(sigmadobs, Seis - mdobs,rcond=None)[0])
    # posterior covariance matrix
    sigmapost = sigmaprior - np.dot((np.dot(G, sigmaprior)).T , np.linalg.lstsq(sigmadobs, np.dot(G,sigmaprior),rcond=None)[0])

    # statistical estimators posterior distribution
    varpost = np.diag(sigmapost).reshape(sigmapost.shape[0],1)
    mmap = np.exp(mpost - varpost)
    mlp = np.exp(mpost - 1.96 * np.sqrt(varpost))
    mup = np.exp(mpost + 1.96 * np.sqrt(varpost))

    # time
    dt = TimeSeis[1] - TimeSeis[0]
    Time = np.arange(TimeSeis[1] - dt / 2, TimeSeis[-1] + dt / 2 + dt, dt)
    
    return mmap, mlp, mup, Time
    
def SeismicModel(Vp, Vs, Rho, Time, theta, wavelet):
    """
    SEISMIC MODEL computes synthetic seismic data according to a linearized
    seismic model based on the convolution of a wavelet and the linearized
    approximation of Zoeppritz equations
    Parameters:
          Vp = P-wave velocity profile
          Vs = S-wave velocity profile
          Rho = Density profile
          theta = vector of reflection angles 
          wavelet = wavelet
    Returns:
           Seis = vector of seismic data of size (nsamples x nangles, 1)
           Time = seismic time  (nsamples, 1)

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    # initial parameters
    ntheta = len(theta)
    nm = Vp.shape[0]

    # number of variables
    nv = 3

    # logarithm of model variables
    logVp = np.log(Vp)
    logVs = np.log(Vs)
    logRho = np.log(Rho)
    m = np.hstack([logVp, logVs, logRho])
    m = m.reshape(len(m),1)

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv)

    # Differential matrix 
    D = DifferentialMatrix(nm, nv)
    mder = np.dot(D, m)

    # Reflectivity coefficients matrix
    Cpp = np.dot(A, mder)

    # Wavelet matrix
    W = WaveletMatrix(wavelet, nm, ntheta)

    # Seismic data matrix
    Seis = np.dot(W, Cpp)

    # Time seismic measurements
    TimeSeis = 1 / 2 * (Time[0:- 1] + Time[1:])
    
    return Seis, TimeSeis
    
def WaveletMatrix(wavelet, nsamples, ntheta):
    """
    WAVELET MATRIX computes the wavelet matrix for discrete convolution
    Parameters:
          w = wavelet
          nsamples = numbr of samples
          ntheta = number of angles
    Returns:
           W = wavelet matrix

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    W = np.zeros((ntheta*(nsamples-1), ntheta*(nsamples-1)))
    indmaxwav = np.argmax(wavelet)
    
    for i in range(ntheta):
        wsub = convmtx(wavelet, (nsamples - 1))
        wsub = wsub.T
        W[ i*(nsamples-1):(i+1)*(nsamples-1), i*(nsamples-1):(i+1)*(nsamples-1)] = wsub[indmaxwav:indmaxwav+(nsamples-1),:]
    
    return W
    

def convmtx(w, ns):
    """    
    CONVMTX  computes the Toeplitz matrix for discrete convolution
    Parameters:
          w = wavelet
          ns = numbr of samples
    Returns:
           C = Toeplitz matrix

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    if len(w) < ns:
        a = np.r_[w[0], np.zeros(ns-1)]
        b = np.r_[w, np.zeros(ns-1)]
    else:
        b = np.r_[w[0], np.zeros(ns - 1)]
        a = np.r_[w, np.zeros(ns - 1)]
    C = toeplitz(a, b)
    return C  

def RockPhysicsGaussInversion(mtrain, dtrain, mdomain, dcond, sigmaerr):
    """
    ROCK PHYSICS GAUSSIAN INVERSION computes the posterior distribution of
    petrophysical properties conditioned on elastic properties assuming a
    Gaussian distribution.
    The joint distribution of the Bayesian inversion approach is estimated
    from a training dataset 
    Parameters:
          mtrain = training dataset of petrophysical properties (ntrain, nm)
          dtrain = training dataset of elastic properties (ntrain, nd)
          mdomain = discretized domain of petrophysical properties 
                    (generated using meshgrid)
          dcond = measured data (nsamples, nd)
          sigmaerr = covariance matrix of the error (nd, nd)
    Returns:
           mupost = posterior mean (nsamples x nv, 1)
           sigmapost = posterior covariance matrix (nv, nv) 
           Ppost = joint posterior distribution 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.4
    """

    # initial parameters
    nv = mtrain.shape[1]
    ns = dcond.shape[0]
    datatrain = np.hstack([mtrain, dtrain])

    # joint distribution
    mjoint = np.mean(datatrain, axis=0)
    mum = mjoint[0:nv]
    mud = mjoint[nv:]
    sjoint = np.cov(datatrain.T)
    sm = sjoint[0:nv,0:nv]
    sd = sjoint[nv:,nv:]
    smd = sjoint[0:nv,nv:]
    sdm = sjoint[nv:,0:nv]

    # posterior distribution
    mupost = np.zeros((ns, nv))
    Ppost = np.zeros((ns, *mdomain.shape[0:nv]))
    # posterior covariance matrix
    kmatrix = np.dot(smd, np.linalg.pinv(sd + sigmaerr)) 
    sigmapost = sm - np.dot( kmatrix,  sdm )
    # [~,posdefcheck] = chol(sigmapost);
    # [V,D]=eig(sigmapost);
    # d=diag(D);
    # d(d<=0)=eps;
    # sigmapost= V*diag(d)*V';
    # analytical solution
    for i in range(ns):
        # posterior mean
        mupost[i, :] = mum + (np.dot( kmatrix,  (dcond[i, :] - mud.T).T)).T 
        # posterior PDF
        Ppost[i,:,:,:] = multivariate_normal.pdf(mdomain, mupost[i,:], sigmapost)

    return mupost, sigmapost, Ppost
    
def RockPhysicsGaussMixInversion(ftrain, mtrain, dtrain, mdomain, dcond, sigmaerr):
    """
    ROCK PHYSICS GAUSS MIX INVERSION computes the posterior distribution of
    petrophysical properties conditioned on elastic properties assuming a
    Gaussian mixture distribution.
    The joint distribution of the Bayesian inversion approach is estimated
    from a training dataset 
    Parameters:
          ftrain = training dataset of facies (ntrain, 1)
          mtrain = training dataset of petrophysical properties (ntrain, nm)
          dtrain = training dataset of elastic properties (ntrain, nd)
          mdomain = discretized domain of petrophysical properties 
                    (generated using meshgrid)
          dcond = measured data (nsamples, nd)
          sigmaerr = covariance matrix of the error (nd, nd)
    Returns:
           mupost = posterior mean (nsamples x nv, 1)
           sigmapost = posterior covariance matrix (nv, nv) 
           fpost = posterior weights (facies proportions) (nsamples, 1) 
           Ppost = joint posterior distribution 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.4
    """

    # initial parameters
    nv = mtrain.shape[1]
    nd = dtrain.shape[1]
    nf = max(np.unique(ftrain))+1
    ns = dcond.shape[0]
    datatrain = np.hstack([mtrain, dtrain])

    # joint distribution
    pf = np.zeros((nf, 1))
    mjoint = np.zeros((nf, nv + nd))
    mum = np.zeros((nf, nv))
    mud = np.zeros((nf, nd))
    sjoint = np.zeros((nv + nd, nv + nd, nf))
    sm = np.zeros((nv, nv, nf))
    sd = np.zeros((nd, nd, nf))
    smd = np.zeros((nv, nd, nf))
    sdm = np.zeros((nd, nv, nf))
    for k in range(nf):
        pf[k,0] = np.sum(ftrain[:,0] == k) / len(ftrain)
        mjoint[k,:] = np.mean(datatrain[ftrain[:,0] == k, :],axis=0)
        mum[k,:] = mjoint[k,0:nv]
        mud[k,:] = mjoint[k,nv:]
        sjoint[:,:,k] = np.cov(np.transpose(datatrain[ftrain[:,0] == k, :]))
        sm[:,:,k] = sjoint[0:nv,0:nv,k]
        sd[:,:,k] = sjoint[nv:,nv:,k]
        smd[:,:,k] = sjoint[0:nv,nv:,k]
        sdm[:,:,k] = sjoint[nv:,0:nv,k]

    # posterior distribution 
    mupost = np.zeros((ns, nv, nf))
    sigmapost = np.zeros((nv, nv, nf))
    kmatrix = np.zeros((nv, nv, nf))
    pfpost = np.zeros((ns, nf))
    Ppost = np.zeros((ns, *mdomain.shape[0:nv]))
    # posterior covariance matrices
    for k in range(nf):
        kmatrix[:,:,k] = np.dot(smd[:,:,k], np.linalg.pinv(sd[:,:,k] + sigmaerr)) 
        sigmapost[:,:,k] = sm[:,:,k] - np.dot( kmatrix[:,:,k],  sdm[:,:,k] )
        #     [~,posdefcheck] = chol(sigmapost(:,:,k));
        #     if posdefcheck~=0
        #         [V,D]=eig(sigmapost(:,:,k));
        #         d=diag(D);
        #         d(d<=0)=eps;
        #         sigmapost(:,:,k)= V*diag(d)*V';
        #     end

    for i in range(ns):
        for k in range(nf):
            # posterior means
            mupost[i,:,k] = mum[k,:] + (np.dot( kmatrix[:,:,k],  (dcond[i, :] - mud[k,:].T).T)).T 
            # posterior weights
            pfpost[i,k] = pf[k,0] * (multivariate_normal.pdf(dcond[i, :], mud[k,:], sd[:,:,k])).T
        den = np.sum(pfpost[i, :])
        lh = 0
        for k in range(nf):
            pfpost[i,k] = pfpost[i,k]  / den
            lh = lh + pfpost[i,k] * multivariate_normal.pdf(mdomain, mupost[i,:,k], sigmapost[:,:,k])
        # posterior PDF
        Ppost[i,:,:,:] = lh / sum(lh.ravel())
    
    return mupost, sigmapost, pfpost, Ppost
    
def RockPhysicsLinGaussInversion(mum, sm, G, mdomain, dcond, sigmaerr):
    """
    ROCK PHYSICS LINEAR GAUSSIAN INVERSION computes the posterior 
    distribution petrophysical properties conditioned on elastic properties 
    assuming a Gaussian distribution and a linear rock physics model
    Parameters:
          mum = prior mean of petrophysical propertiees (nv, 1)
          sm = prior covariance matrix of petrophysical propertiees (nv, nv)
          G = rock physics operator matrix
          mdomain = discretized domain of petrophysical properties 
                    (generated using meshgrid)
          dcond = measured data (nsamples, nd)
          sigmaerr = covariance matrix of the error (nd, nd)
    Returns:
           mupost = posterior mean (nsamples x nv, 1)
           sigmapost = posterior covariance matrix (nv, nv) 
           Ppost = joint posterior distribution 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.4
    """

    # initial parameters
    nv = mum.shape[1]
    ns = dcond.shape[0]
    
    # analytical calculations
    mud = np.dot(G, mum.T)
    sd = multi_dot([G, sm, G.T])
    smd = np.dot(sm, G.T)
    sdm = np.dot(G, sm)

    # posterior distribution
    mupost = np.zeros((ns, nv))
    Ppost = np.zeros((ns, *mdomain.shape[0:nv]))
    # posterior covariance matrix
    kmatrix = np.dot(smd, np.linalg.pinv(sd + sigmaerr)) 
    sigmapost = sm - np.dot( kmatrix,  sdm )
    # analytical solution
    for i in range(ns):
        # posterior mean
        mupost[i, :] = mum + (np.dot( kmatrix,  (dcond[i, :] - mud.T).T)).T 
        # posterior PDF
        Ppost[i,:,:,:] = multivariate_normal.pdf(mdomain, mupost[i,:], sigmapost)

    return mupost, sigmapost, Ppost
    
def RockPhysicsLinGaussMixInversion(pf, mum, sm, G, mdomain, dcond, sigmaerr):
    """
    ROCK PHYSICS LINEAR GAUSS MIX INVERSION computes the posterior 
    distribution petrophysical properties conditioned on elastic properties 
    assuming a Gaussian mixture distribution and a linear rock physics model
    Parameters:
          pf =  prior weights (facies proportions) (nf, 1)
          mum = prior means of petrophysical propertiees (nf, nv)
          sm = prior covariance matrices of petrophysical propertiees (nv, nv, nf)
          G = rock physics operator matrix
          mdomain = discretized domain of petrophysical properties 
                    (generated using meshgrid)
          dcond = measured data (nsamples, nd)
          sigmaerr = covariance matrix of the error (nd, nd)
    Returns:
           mupost = posterior mean (nsamples, nv, nf)
           sigmapost = posterior covariance matrix (nv, nv, nf) 
           fpost = posterior weights (nsamples, nf) 
           Ppost = joint posterior distribution 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.4
    """

    # initial parameters
    nv = mum.shape[1]
    nf = mum.shape[0]
    nd = dcond.shape[1]
    ns = dcond.shape[0]

    # analytical calculations
    mud = np.zeros((nf, nd))
    sd = np.zeros((nd, nd, nf))
    smd = np.zeros((nv, nd, nf))
    sdm = np.zeros((nd, nv, nf))
    for k in range(nf):
        mud[k,:] = np.dot(G, mum[k, :].T)
        sd[:,:,k] = multi_dot([G, sm[:,:, k], G.T]) 
        smd[:,:,k] = np.dot(sm[:,:, k], G.T)
        sdm[:,:,k] = np.dot(G, sm[:,:, k])

    # posterior distribution
    mupost = np.zeros((ns, nv, nf))
    sigmapost = np.zeros((nv, nv, nf))
    kmatrix = np.zeros((nv, nv, nf))
    pfpost = np.zeros((ns, nf))
    Ppost = np.zeros((ns, *mdomain.shape[0:nv]))
    # posterior covariance matrices
    for k in range(nf):
        kmatrix[:,:,k] = np.dot(smd[:,:,k], np.linalg.pinv(sd[:,:,k] + sigmaerr)) 
        sigmapost[:,:,k] = sm[:,:,k] - np.dot( kmatrix[:,:,k],  sdm[:,:,k] )

    for i in range(ns):
        for k in range(nf):
            # posterior means
            mupost[i,:,k] = mum[k,:] + (np.dot( kmatrix[:,:,k],  (dcond[i, :] - mud[k,:].T).T)).T 
            # posterior weights
            pfpost[i,k] = pf[k,0] * (multivariate_normal.pdf(dcond[i, :], mud[k,:], sd[:,:,k])).T
        den = np.sum(pfpost[i, :])
        lh = 0
        for k in range(nf):
            pfpost[i,k] = pfpost[i,k]  / den
            lh = lh + pfpost[i,k] * multivariate_normal.pdf(mdomain, mupost[i,:,k], sigmapost[:,:,k])
        # posterior PDF
        Ppost[i,:,:,:] = lh / sum(lh.ravel())
    
    return mupost, sigmapost, pfpost, Ppost


def RockPhysicsKDEInversion(mtrain, dtrain, jointdimain, datadomain, dcond, jointdim, mdim):
    """
    ROCK PHYSICS KDE INVERSION computes the posterior distribution of
    petrophysical properties conditioned on elastic properties assuming a
    non-parametric distribution.
    The joint distribution of the Bayesian inversion approach is estimated
    from a training dataset using Kernel Density Estimation
    Parameters:
          mtrain = training dataset of petrophysical properties (ntrain, nm)
          dtrain = training dataset of elastic properties (ntrain, nd)
          jointdimain = discretized domain of all properties 
          datadomain = discretized vectors of elastic properties 
          dcond = measured data (nsamples, nd)
          jointdim  = dimension of joint distribution
          mdim  = dimension of post distribution
    Returns:
           Ppost = posterior distribution 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.4
    """

    ## Inefficient implementation in Python -- refer to MATLAB code ##

    # number of training datapoint
    nd = dtrain.shape[1]
    ns = dcond.shape[0]
    datatrain = np.hstack([mtrain, dtrain])
    
    kernel = stats.gaussian_kde(datatrain.T)
    
    Pjoint = np.reshape(kernel(jointdimain).T, jointdim)
    
    Ppost = np.zeros((ns, *mdim))
    for i in range(ns):
        ind = np.zeros((nd,1))      
        for k in range(nd):
            ind[k] = np.argmin(((dcond[i,k] - datadomain[k,:]))**2)
        ind = ind.astype(int)
        Ppost[i,:,:,:] = np.squeeze(Pjoint[:,:,:,ind[0],ind[1],ind[2]]) / sum(Pjoint[:,:,:,ind[0],ind[1],ind[2]].ravel())  
    
    return Ppost
    
    
def EnsembleSmootherMDA(PriorModels, SeisData, SeisPred, alpha, sigmaerr):
    """
    ENSEMBLE SMOOTHER MDA computes the updated realizations of the
    model variables conditioned on the assimilated data using the 
    Ensemble Smoother Multiple Data Assimilation
    Parameters:
          PriorModels = prior models realizations (nm, ne) 
          SeisData = measured seismic data (nd, 1)
          SeisPred = predicted data (nd, ne)
          alpha = inflation coefficient 
          sigmaerr = covariance matrix of the error (nd, nd)
    Returns:
           PostModels = updated models realizations (nm, ne) 
           KalmanGain = Kalman Gain Matrix'

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.6
    """

    # initial parameters
    nd, ne = SeisPred.shape
    # data perturbation
    SeisPert = np.matlib.repmat(SeisData, 1, ne) + np.dot(np.sqrt(alpha * sigmaerr), np.random.randn(nd, ne))
    # mean models
    mum = np.mean(PriorModels, axis=1)
    mud = np.mean(SeisPred, axis=1)
    # covariance matrices
    smd = 1 / (ne - 1) * np.dot((PriorModels - mum.reshape(len(mum),1)),(SeisPred - mud.reshape(len(mud),1)).T)
    sdd = 1 / (ne - 1) * np.dot((SeisPred - mud.reshape(len(mud),1)), (SeisPred - mud.reshape(len(mud),1)).T)
    # Kalman Gain
    KalmanGain = np.dot(smd,  np.linalg.pinv(sdd + alpha * sigmaerr))
    # Updated models
    PostModels = PriorModels + np.dot(KalmanGain, (SeisPert - SeisPred))
    
    return PostModels, KalmanGain
    
def InvLogitBounded(w, minv, maxv):
    """
    INVERSE LOGIT BOUNDED computes the inverse logit tranformation for
    bounded variables
    Parameters:
          w = initial variable
          minv = lower bound 
          maxv = upper bound 
    Returns:
           index = transformed variable 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.6
    """

    # tranformation
    v = (np.exp(w) * maxv + minv) / (np.exp(w) + 1)
    
    return v

def LogitBounded(v, minv, maxv):
    """
    LOGIT BOUNDED computes the logit tranformation for
    bounded variables
    Parameters:
          v = initial variable
          minv = lower bound of the domain
          maxv = upper bound of the domain
    Returns:
           index = transformed variable 

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.6
    """

    w = np.log((v - minv) / (maxv - v))
    
    return w
    