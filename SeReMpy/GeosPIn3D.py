#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:08:27 2023

@author: Dario Grana and Mingliang Liu
"""

import numpy as np
from scipy.linalg import pinvh
from scipy.fftpack import fftn, ifftn
from numpy.random import multivariate_normal
from scipy.sparse import eye, diags, lil_matrix
from RockPhysics import LinearizedRockPhysicsModel
from Inversion import EnsembleSmootherMDA


def AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv):
    # AkiRichardsCoefficientsMatrix computes the Aki Richards coefficient matrix
    # INPUT Vp = P-wave velocity profile
    #       Vs = S-wave velocity profile
    #       theta = vector of reflection angles
    #       nv = number of model variables 
    # OUTPUT A = Aki Richards coefficients matrix

    # initial parameters
    nsamples = len(Vp)
    ntheta = len(theta)
    A = diags(np.zeros((ntheta, nv*(nsamples-1))), format='lil')

    # average velocities at the interfaces
    avgVp = 0.5 * (Vp[:-1] + Vp[1:])
    avgVs = 0.5 * (Vs[:-1] + Vs[1:])

    # reflection coefficients (Aki Richards linearized approximation)
    for i in range(ntheta):
        cp = 0.5 * (1 + np.tan(np.deg2rad(theta[i])) ** 2) * np.ones(nsamples - 1)
        cs = -4 * avgVs ** 2 / avgVp ** 2 * np.sin(np.deg2rad(theta[i])) ** 2
        cr = 0.5 * (1 - 4 * avgVs ** 2 / avgVp ** 2 * np.sin(np.deg2rad(theta[i])) ** 2)
        Acp = diags(cp, format='lil')
        Acs = diags(cs, format='lil')
        Acr = diags(cr, format='lil')
        A[i*(nsamples-1):(i+1)*(nsamples-1), :] = diags([Acp, Acs, Acr], format='lil')

    return A.tocsr()  # Convert to CSR format for better performance in matrix operations


def BayesPetroInversion3D_GMM(near, mid, far, TimeSeis, vpprior, vsprior, rhoprior, sigmaprior, elastrain, petrotrain, faciestrain, vpgrid, vsgrid, rhogrid, phigrid, claygrid, swgrid, sigmaerr, wavelet, theta, nv, rpsigmaerr):
    # BayesPetroInversion3D_GMM computes the posterior distribution of 
    # petrophysical properties using Bayesian linearized AVO inversion 
    # (Buland and Omre, 2003), a Gaussian mixture rock physics likelihood
    # (Grana and Della Rossa, 2010) and Chapman Kolmogorov equation
    # INPUT near = seismic near angle (nxl x nil x nd)
    #       mid = seismic mid angle (nxl x nil x nd)
    #       far = seismic far angle (nxl x nil x nd)
    #       TimeSeis = seismic time vector (nd x 1)
    #       vpprior = prior Vp model (nxl x nil x nm, with nm = nd+1)
    #       vsprior = prior Vs model (nxl x nil x nm, with nm = nd+1)
    #       rhoprior = prior density model (nxl x nil x nm, with nm = nd+1)
    #       elastrain = matrix with training elastic data [Vp, Vs, density]
    #                    (ns x 3)
    #       petrotrain = matrix with training petrophysics data 
    #                    [porosoty, clay, saturation](ns x 3)
    #       faciestrain = vector with training facies data 
    #       vpgrid = vector of discretized Vp grid (ndiscr x 1)
    #       vsgrid = vector of discretized Vs grid (ndiscr x 1)
    #       rhogrid = vector of discretized density grid (ndiscr x 1)
    #       phigrid = vector of discretized porosity grid (ndiscr x 1)
    #       claygrid = vector of discretized clay grid (ndiscr x 1)
    #       swgrid = vector of discretized satruration grid (ndiscr x 1)
    #       sigmaerr = covariance matrix of the error (nv*nsamples x nv*nsamples)
    #       wavelet = wavelet vector 
    #       theta = vector of reflection angles 
    #       nv = number of model variables
    #       rpsigmaerr = rock physics  error variance
    # OUTPUT Vpmap = Predicted Vp (nxl x nil x nm)
    #       Vpmap = Predicted Vp (nxl x nil x nm)
    #       Rhomap = Predicted density (nxl x nil x nm)
    #       Phimap = Predicted porosity (nxl x nil x nm)
    #       Claymap = Predicted clay (nxl x nil x nm)
    #       Swmap = Predicted saturation (nxl x nil x nm)
    #       Time =  time vector (nm x 1)

    nxl, nil, nm = near.shape[0], near.shape[1], near.shape[2] + 1
    ndiscr = len(phigrid)
    dt = TimeSeis[1] - TimeSeis[0]
    Time = np.arange(TimeSeis[0] - dt/2, TimeSeis[-1] + dt, dt)
    Vpmap = np.zeros((nxl, nil, nm))
    Vsmap = np.zeros((nxl, nil, nm))
    Rhomap = np.zeros((nxl, nil, nm))
    Phimap = np.zeros((nxl, nil, nm))
    Claymap = np.zeros((nxl, nil, nm))
    Swmap = np.zeros((nxl, nil, nm))

    # rock physics likelihood
    petrogrid = np.column_stack([p.ravel() for p in np.meshgrid(phigrid, claygrid, swgrid)])
    elasgrid = np.column_stack([p.ravel() for p in np.meshgrid(vpgrid, vsgrid, rhogrid)])
    Ppetro = RockPhysicsGMM(faciestrain, petrotrain, elastrain, petrogrid, elasgrid, rpsigmaerr)

    # inversion
    for i in range(nxl):
        print(f'Percentage progress: {round(i/nxl*100)} %')
        for j in range(nil):
            Seis = np.column_stack([near[i, j], mid[i, j], far[i, j]])
            mmap, mtrans, strans, Time = SeismicInversion3D(Seis, TimeSeis, vpprior[i, j], vsprior[i, j], rhoprior[i, j], sigmaprior, sigmaerr, wavelet, theta, nv)
            Vpmap[i, j] = mmap[:nm]
            Vsmap[i, j] = mmap[nm:2*nm]
            Rhomap[i, j] = mmap[2*nm:]
            vptrans = mtrans[:nm]
            vstrans = mtrans[nm:2*nm]
            rhotrans = mtrans[2*nm:]
            sigmatrans = np.diag(strans[np.array([round(nm/2), nm + round(nm/2), 2*nm + round(nm/2)])])
            Pseis = np.zeros((elasgrid.shape[0], nm))
            for k in range(nm):
                Pseis[:, k] = multivariate_normal.pdf(elasgrid, [vptrans[k], vstrans[k], rhotrans[k]], sigmatrans)
                Pseis[:, k] /= Pseis[:, k].sum()
            Ppost = Ppetro.T @ Pseis
            Ppostmarg = Ppost.reshape((ndiscr, ndiscr, ndiscr, nm))
            Pphi = np.sum(np.sum(Ppostmarg, axis=2), axis=1)
            Pclay = np.sum(np.sum(Ppostmarg, axis=2), axis=0)
            Psw = np.sum(np.sum(Ppostmarg, axis=1), axis=0)
            Pphi /= Pphi.sum()
            Pclay /= Pclay.sum()
            Psw /= Psw.sum()
            Phimap[i, j] = phigrid[np.argmax(Pphi)]
            Claymap[i, j] = claygrid[np.argmax(Pclay)]
            Swmap[i, j] = swgrid[np.argmax(Psw)]

    return Vpmap, Vsmap, Rhomap, Phimap, Claymap, Swmap, Time


def BayesPetroInversion3D_KDE(near, mid, far, TimeSeis, vpprior, vsprior, rhoprior, sigmaprior, elastrain, petrotrain, vpgrid, vsgrid, rhogrid, phigrid, claygrid, swgrid, sigmaerr, wavelet, theta, nv, h):
    nxl = near.shape[0]
    nil = near.shape[1]
    nm = near.shape[2] + 1
    ndiscr = len(phigrid)
    dt = TimeSeis[1] - TimeSeis[0]
    Time = np.arange(TimeSeis[0] - dt / 2, TimeSeis[-1] + dt, dt)
    Vpmap = np.zeros((nxl, nil, nm))
    Vsmap = np.zeros_like(Vpmap)
    Rhomap = np.zeros_like(Vpmap)
    Phimap = np.zeros_like(Vpmap)
    Claymap = np.zeros_like(Vpmap)
    Swmap = np.zeros_like(Vpmap)

    # rock physics likelihood
    petrogrid = np.column_stack((phigrid, claygrid, swgrid))
    elasgrid = np.column_stack((vpgrid, vsgrid, rhogrid))
    hm = np.array([
        (np.max(phigrid) - np.min(phigrid)) / h,
        (np.max(claygrid) - np.min(claygrid)) / h,
        (np.max(swgrid) - np.min(swgrid)) / h])
    hd = np.array([
        (np.max(vpgrid) - np.min(vpgrid)) / h,
        (np.max(vsgrid) - np.min(vsgrid)) / h,
        (np.max(rhogrid) - np.min(rhogrid)) / h])
    vpgrid, vsgrid, rhogrid = np.meshgrid(vpgrid, vsgrid, rhogrid)
    elasevalpoints = np.column_stack((vpgrid.ravel(), vsgrid.ravel(), rhogrid.ravel()))
    Ppetro = RockPhysicsKDE(petrotrain, elastrain, petrogrid, elasgrid, elasevalpoints, hm, hd)

    # inversion
    for i in range(nxl):
        print('Percentage progress:', int(i / nxl * 100), '%')
        for j in range(nil):
            Seis = np.concatenate((near[i, j], mid[i, j], far[i, j]), axis=None)[:-1]
            mmap, mtrans, strans, Time = SeismicInversion3D(Seis, TimeSeis, vpprior[i, j], vsprior[i, j], rhoprior[i, j], sigmaprior, sigmaerr, wavelet, theta, nv)
            Vpmap[i, j] = mmap[:nm]
            Vsmap[i, j] = mmap[nm:2 * nm]
            Rhomap[i, j] = mmap[2 * nm:]
            vptrans = mtrans[:nm]
            vstrans = mtrans[nm:2 * nm]
            rhotrans = mtrans[2 * nm:]
            sigmatrans = np.array([[strans[round(nm / 2), 0], 0, 0],
                                  [0, strans[nm + round(nm / 2), 0], 0],
                                  [0, 0, strans[2 * nm + round(nm / 2), 0]]])
            Pseis = np.zeros((len(elasevalpoints), nm))
            for k in range(nm):
                Pseis[:, k] = multivariate_normal.pdf(elasevalpoints, mean=[vptrans[k], vstrans[k], rhotrans[k]], cov=sigmatrans)
                Pseis[:, k] /= np.sum(Pseis[:, k])
            Ppost = Ppetro.T @ Pseis
            Ppostmarg = np.zeros((ndiscr, ndiscr, ndiscr, nm))
            Pphi = np.zeros((nm, ndiscr))
            Pclay = np.zeros((nm, ndiscr))
            Psw = np.zeros((nm, ndiscr))
            for k in range(nm):
                Ppostmarg[:, :, :, k] = Ppost[:, k].reshape((ndiscr, ndiscr, ndiscr))
                Pphi[k, :] = np.sum(np.sum(Ppostmarg[:, :, :, k], axis=2), axis=1)
                Pclay[k, :] = np.sum(np.sum(Ppostmarg[:, :, :, k], axis=2), axis=0)
                Psw[k, :] = np.sum(np.sum(Ppostmarg[:, :, :, k], axis=1), axis=0)
                Pphi[k, :] /= np.sum(Pphi[k, :])
                Pclay[k, :] /= np.sum(Pclay[k, :])
                Psw[k, :] /= np.sum(Psw[k, :])
                ii = np.argmax(Pphi[k, :])
                jj = np.argmax(Pclay[k, :])
                kk = np.argmax(Psw[k, :])
                Phimap[i, j, k] = phigrid[ii]
                Claymap[i, j, k] = claygrid[jj]
                Swmap[i, j, k] = swgrid[kk]

    return Vpmap, Vsmap, Rhomap, Phimap, Claymap, Swmap, Time


def CorrelationFunction3D(Lv, Lh, nxl, nil, nm):
    # CorrelationFunction3D computes the 3D spatial correlation function using FFT-MA simulations
    # INPUT Lv = vertical correlation parameter
    #       Lh = horizontal correlation parameter
    #       nxl = number of crossline
    #       nil = number of inline
    #       nm = number of samples
    # OUTPUT corrfun = Spatial Correlation model

    ordem = 3
    desvio = 0.25
    corrfun = np.zeros((nxl, nil, nm))

    for i in range(nxl):
        for j in range(nil):
            for k in range(nm):
                r = np.sqrt(((i - nxl // 2) / (3 * Lv))**2 + ((j - nil // 2) / (3 * Lh))**2 + ((k - nm // 2) / (3 * Lh))**2)
                if r < 1:
                    value = 1 - 1.5 * r + 0.5 * r**3
                else:
                    value = 0
                winval = np.exp(-((abs((i - nxl // 2)) / (desvio * nxl))**ordem + abs((j - nil // 2)) / (desvio * nil)**ordem + abs((k - nm // 2)) / (desvio * nm)**ordem))
                corrfun[i, j, k] = value * winval

    return corrfun


def DifferentialMatrix(nt, nv):
    # DifferentialMatrix computes the differential matrix for discrete differentiation
    # INPUT nt = number of samples
    #       nv = number of model variables
    # OUTPUT D = differential matrix

    I = eye(nt)
    B = diags([-1, 1], [0, 1], shape=(nt - 1, nt), format='lil')
    I = (I + B)[1:]
    D = diags(np.zeros(((nt - 1) * nv, nt * nv)), format='lil')

    for i in range(nv):
        D[i * (nt - 1):(i + 1) * (nt - 1), i * nt:(i + 1) * nt] = I

    return D


def EpanechnikovKernel(u):
    return (np.abs(u) <= 1) * (3/4) * (1 - u**2)


def ForwardGeopModel1D(Phi, Clay, Sw, regcoef, Time, theta, wavelet, nm, nv, nsim):
    # ForwardGeopModel1D computes predicted seismic data
    # INPUT Phi = porosity model (nxl x nil x nm)
    #       Clay = clay model (nxl x nil x nm)
    #       Sw = saturation model (nxl x nil x nm)
    #       regcoef = Regression coefficients rock physics model (3x4)
    #       Time = time vector (nd x 1)
    #       theta = vector of reflection angles 
    #       wavelet = wavelet vector 
    #       nm = number of samples
    #       nv = number of model variables
    #       nsim = number of simulations
    # OUTPUT SeisPred = Predicted seismic (nxl x nil x nd)

    nd = nm - 1
    ntheta = len(theta)
    SeisPred = np.zeros((ntheta * nd, nsim))
    Vp, Vs, Rho = LinearizedRockPhysicsModel(Phi, Clay, Sw, regcoef)
    D = DifferentialMatrix(nm, nv)
    W = WaveletMatrix(wavelet, nm, ntheta)

    for k in range(nsim):
        SeisPred[:, k], _ = SeismicModel3D(Vp[:, k], Vs[:, k], Rho[:, k], Time, theta, W, D)

    return SeisPred


def ForwardGeopModel3D(Phi, Clay, Sw, petrotrain, elastrain, Time, theta, wavelet, nxl, nil, nm, nv, nsim):
    # ForwardGeopModel3D computes predicted seismic data
    # INPUT Phi = porosity model (nxl x nil x nm)
    #       Clay = clay model (nxl x nil x nm)
    #       Sw = saturation model (nxl x nil x nm)
    #       petrotrain = matrix with training petrophysics data [porosoty, clay, saturation] (ns x 3)
    #       elastrain = matrix with training elastic data [Vp, Vs, density] (ns x 3)
    #       Time = time vector (nd x 1)
    #       theta = vector of reflection angles 
    #       wavelet = wavelet vector 
    #       nxl = number of crossline
    #       nil = number of inline
    #       nm = number of samples
    #       nv = number of model variables
    #       nsim = number of simulations
    # OUTPUT SeisPred = Predicted seismic (nxl x nil x nd x nsim)
    #        regcoef = Regression coefficients rock physics model (3 x (nv+1))

    nd = nm - 1
    ntheta = len(theta)
    SeisPred = np.zeros((nxl, nil, ntheta * nd, nsim))
    regcoef = np.zeros((nv, nv + 1))
    regx = np.hstack((petrotrain, np.ones((petrotrain.shape[0], 1))))
    
    for i in range(nv):
        regcoef[i, :] = np.linalg.lstsq(regx, elastrain[:, i], rcond=None)[0]

    Vp, Vs, Rho = LinearizedRockPhysicsModel(Phi, Clay, Sw, regcoef)
    D = DifferentialMatrix(nm, nv)
    W = WaveletMatrix(wavelet, nm, ntheta)

    for i in range(nxl):
        print('Percentage progress of simulation: {:.1f} %'.format(i / nxl * 100))
        for j in range(nil):
            for k in range(nsim):
                SeisPred[i, j, :, k], _ = SeismicModel3D(Vp[i, j, :, k], Vs[i, j, :, k], Rho[i, j, :, k], Time, theta, W, D)

    return SeisPred, regcoef


def GeosPetroInversion3D(near, mid, far, TimeSeis, phiprior, clayprior, swprior, stdpetro, corrpetro, elastrain, petrotrain, wavelet, theta, nv, sigmaerr, vertcorr, horcorr, nsim, niter):
    nxl = near.shape[0]
    nil = near.shape[1]
    nd = near.shape[2]
    nm = nd + 1
    dt = TimeSeis[1] - TimeSeis[0]
    Time = np.arange(TimeSeis[0] - dt / 2, TimeSeis[-1] + dt, dt)
    Phimap = np.zeros((nxl, nil, nm))
    Claymap = np.zeros_like(Phimap)
    Swmap = np.zeros_like(Phimap)

    ## Prior realizations
    Phisim, Claysim, Swsim = ProbFieldSimulation3D(vertcorr, horcorr, phiprior, clayprior, swprior, stdpetro, corrpetro, nxl, nil, nm, nsim)
    SeisSim, regcoef = ForwardGeopModel3D(Phisim, Claysim, Swsim, petrotrain, elastrain, Time, theta, wavelet, nxl, nil, nm, nv, nsim)

    ## ESMDA petrophysical inversion
    alpha = 1 / niter

    for i in range(nxl):
        print('Percentage progress of inversion:', int(i / nxl * 100), '%')
        for j in range(nil):
            PostModels = np.vstack((Phisim[i, j].reshape(-1, nsim),
                                    Claysim[i, j].reshape(-1, nsim),
                                    Swsim[i, j].reshape(-1, nsim)))
            SeisData = np.hstack((near[i, j], mid[i, j], far[i, j])).reshape(-1, 1)
            SeisPred = SeisSim[i, j].T
            for h in range(niter):
                PostModels, _ = EnsembleSmootherMDA(PostModels, SeisData, SeisPred, alpha, sigmaerr)
                Phipost = PostModels[:nm, :]
                Claypost = PostModels[nm:2 * nm, :]
                Swpost = PostModels[2 * nm:, :]
                Phipost[Phipost < 0] = 0
                Phipost[Phipost > 0.4] = 0.4
                Claypost[Claypost < 0] = 0
                Claypost[Claypost > 0.8] = 0.8
                Swpost[Swpost < 0] = 0
                Swpost[Swpost > 1] = 1
                SeisPred = ForwardGeopModel1D(Phipost, Claypost, Swpost, regcoef, Time, theta, wavelet, nm, nv, nsim)

            # posterior mean models
            mpost = np.mean(PostModels, axis=1)
            Phimap[i, j] = mpost[:nm]
            Claymap[i, j] = mpost[nm:2 * nm]
            Swmap[i, j] = mpost[2 * nm:]

    return Phimap, Claymap, Swmap, Time


def ProbFieldSimulation3D(vertcorr, horcorr, phiprior, clayprior, swprior, stdpetro, corrpetro, nxl, nil, nm, nsim):
    # ProbFieldSimulation3D simulates spatially correlated realizations of 
    # petrophysical properties using PFS simulations
    # INPUT vertcorr = vertical correlation parameter
    #       horcorr = horizontal correlation parameter
    #       phiprior = prior porosity mean
    #       clayprior = prior clay mean
    #       swprior = prior saturation mean
    #       stdpetro = vector with prior standard deviation of petrophysical properties (3 x 1)
    #       corrpetro = prior correlation matrix of petrophysical properties (3 x 3)
    #       nxl = number of crossline
    #       nil = number of inline
    #       nm = number of samples
    #       nsim = number of simulations
    # OUTPUT Phisim = Simulated porosity (nxl x nil x nm x nsim)
    #        Claysim = Simulated clay (nxl x nil x nm x nsim)
    #        Swsim = Simulated saturation (nxl x nil x nm x nsim)

    Phisim = np.zeros((nxl, nil, nm, nsim))
    Claysim = np.zeros((nxl, nil, nm, nsim))
    Swsim = np.zeros((nxl, nil, nm, nsim))
    corrfun = CorrelationFunction3D(vertcorr, horcorr, nxl, nil, nm)
    corrprior = (np.diag(np.sqrt(np.diag(corrpetro))) @ corrpetro) / np.diag(np.sqrt(np.diag(corrpetro)))

    for k in range(nsim):
        uncorrsim = multivariate_normal([0, 0, 0], corrprior, nxl * nil * nm)
        noisephi = np.reshape(uncorrsim[:, 0], (nxl, nil, nm))
        noiseclay = np.reshape(uncorrsim[:, 1], (nxl, nil, nm))
        noisesw = np.reshape(uncorrsim[:, 2], (nxl, nil, nm))
        Phisim[:, :, :, k] = phiprior + stdpetro[0] * np.real(ifftn(np.sqrt(np.abs(fftn(corrfun))) * fftn(noisephi)))
        Claysim[:, :, :, k] = clayprior + stdpetro[1] * np.real(ifftn(np.sqrt(np.abs(fftn(corrfun))) * fftn(noiseclay)))
        Swsim[:, :, :, k] = swprior + stdpetro[2] * np.real(ifftn(np.sqrt(np.abs(fftn(corrfun))) * fftn(noisesw)))

    Phisim[Phisim < 0.01] = 0.01
    Phisim[Phisim > 0.4] = 0.4
    Claysim[Claysim < 0] = 0
    Claysim[Claysim > 0.8] = 0.8
    Swsim[Swsim < 0] = 0
    Swsim[Swsim > 1] = 1

    return Phisim, Claysim, Swsim


def RockPhysicsGMM(ftrain, mtrain, dtrain, mdomain, dcond, sigmaerr):
    # RockPhysicsGMM computes the rock physics likelihood assuming
    # Gaussian mixture distribution from a training dataset
    # INPUT ftrain = vector with training facies data 
    #       mtrain = matrix with training petrophysics data 
    #                [porosoty, clay, saturation](ns x 3)
    #       dtrain = matrix with training elastic data [Vp, Vs, density]
    #                (ns x 3)
    #       mdomain = petrophysical domain (created using ndgrid)
    #       dcond = elastic domain (created using ndgrid)
    #       sigmaerr = rock physics error variance
    # OUTPUT Ppost = petrophysical joint distribution 

    # initial parameters
    nv = mtrain.shape[1]
    nd = dtrain.shape[1]
    nf = int(np.max(ftrain))
    ns = dcond.shape[0]
    datatrain = np.concatenate((mtrain, dtrain), axis=1)

    # joint distribution
    pf = np.zeros(nf)
    mjoint = np.zeros((nf, nv + nd))
    mum = np.zeros((nf, nv))
    mud = np.zeros((nf, nd))
    sjoint = np.zeros((nv + nd, nv + nd, nf))
    sm = np.zeros((nv, nv, nf))
    sd = np.zeros((nd, nd, nf))
    smd = np.zeros((nv, nd, nf))
    sdm = np.zeros((nd, nv, nf))
    
    for k in range(1, nf + 1):
        pf[k - 1] = np.sum(ftrain == k) / len(ftrain)
        mjoint[k - 1, :] = np.mean(datatrain[ftrain == k], axis=0)
        mum[k - 1, :] = mjoint[k - 1, :nv]
        mud[k - 1, :] = mjoint[k - 1, nv:]
        sjoint[:, :, k - 1] = np.cov(datatrain.T)
        sm[:, :, k - 1] = sjoint[:nv, :nv, k - 1]
        sd[:, :, k - 1] = sjoint[nv:, nv:, k - 1]
        smd[:, :, k - 1] = sjoint[:nv, nv:, k - 1]
        sdm[:, :, k - 1] = sjoint[nv:, :nv, k - 1]

    # posterior distribution 
    mupost = np.zeros((ns, nv, nf))
    sigmapost = np.zeros((nv, nv, nf))
    pfpost = np.zeros((ns, nf))
    Ppost = np.zeros((ns, mdomain.shape[0]))
    
    # posterior covariance matrices
    for k in range(1, nf + 1):
        sigmapost[:, :, k - 1] = sm[:, :, k - 1] - smd[:, :, k - 1] @ np.linalg.inv(sd[:, :, k - 1] + sigmaerr) @ sdm[:, :, k - 1]

    for i in range(ns):
        for k in range(1, nf + 1):
            # posterior means
            mupost[i, :, k - 1] = mum[k - 1] + smd[:, :, k - 1] @ np.linalg.inv(sd[:, :, k - 1] + sigmaerr) @ (dcond[i] - mud[k - 1])
            # posterior weights
            pfpost[i, k - 1] = pf[k - 1] * multivariate_normal.pdf(dcond[i], mean=mud[k - 1], cov=sd[:, :, k - 1])

        if np.sum(pfpost[i]) > 0:
            pfpost[i] /= np.sum(pfpost[i])

        lh = np.zeros(mdomain.shape[0])
        for k in range(1, nf + 1):
            lh += pfpost[i, k - 1] * multivariate_normal.pdf(mdomain, mean=mupost[i, :, k - 1], cov=sigmapost[:, :, k - 1])
        
        # posterior PDF
        if np.sum(lh > 0):
            Ppost[i] = lh / np.sum(lh)
        else:
            Ppost[i] = 0

    return Ppost


def RockPhysicsKDE(mtrain, dtrain, mdomain, ddomain, dcond, hm, hd):
    # ROCK PHYSICS KDE INVERSION computes the posterior distribution of
    # petrophysical properties conditioned on elastic properties assuming a
    # non-parametric distribution.
    # The joint distribution of the Bayesian inversion approach is estimated
    # from a training dataset using Kernel Density Estimation
    # INPUT mtrain = training dataset of petrophysical properties (ntrain, nm)
    #       dtrain = training dataset of elastic properties (ntrain, nd)
    #       mdomain = discretized domain of petrophysical properties (ndiscr, nm)
    #       ddomain = discretized domain of elastic properties (ndiscr, nd)
    #       dcond = measured data (nsamples, nd)
    #       hm = kernel bandwidths hs of petrophysical properties (nm, 1)
    #       hd = kernel bandwidths of elastic properties (nd, 1)
    # OUTPUT Ppost = joint posterior distribution 

    # number of training datapoint
    nt = mtrain.shape[0]
    # number of data points
    ns = dcond.shape[0]

    # multidimensional grids for model variables
    vecm = np.meshgrid(*mdomain.T, indexing='ij')
    mgrid = np.column_stack([m.flatten() for m in vecm])
    # multidimensional grids for data variables
    vecd = np.meshgrid(*ddomain.T, indexing='ij')
    dgrid = np.column_stack([d.flatten() for d in vecd])

    # kernel density estimation
    hmmat = np.tile(hm, (nt, 1, mgrid.shape[0]))
    prodm = np.prod(EpanechnikovKernel((mgrid.T[:, :, None] - mtrain.T[:, None, :]) / hmmat), axis=1)
    
    hdmat = np.tile(hd, (nt, 1, dgrid.shape[0]))
    prodd = np.prod(EpanechnikovKernel((dgrid.T[:, :, None] - dtrain.T[:, None, :]) / hdmat), axis=1)

    # joint distribution
    Pjoint = prodm.T @ prodd / (nt * np.prod(hm) * np.prod(hd))
    Pjoint /= np.sum(Pjoint)

    # posterior distribution
    Ppost = np.zeros((ns, mgrid.shape[0]))
    for i in range(ns):
        indcond = np.argmin(np.sum((dgrid - dcond[i, :]) ** 2, axis=1))
        if np.sum(Pjoint[:, indcond]) > 0:
            Ppost[i, :] = Pjoint[:, indcond] / np.sum(Pjoint[:, indcond])
        else:
            Ppost[i, :] = Pjoint[:, indcond]

    return Ppost


def SeismicInversion3D(Seis, TimeSeis, Vpprior, Vsprior, Rhoprior, sigmaprior, sigmaerr, wavelet, theta, nv):
    # SeismicInversion3D computes the posterior distribution of elastic
    # properties according to the Bayesian linearized AVO inversion (Buland and
    # Omre, 2003)
    # INPUT Seis = vector of seismic data of size (nsamples x nangles, 1)
    #       TimeSeis = vector of seismic time of size (nsamples, 1)
    #       Vpprior = vector of prior (low frequency) Vp model (nsamples+1, 1)
    #       Vsprior = vector of prior (low frequency) Vs model (nsamples+1, 1)
    #       Rhoprior = vector of prior (low frequency) density model (nsamples+1, 1)
    #       sigmaprior = prior covariance matrix (nv*(nsamples+1),nv*(nsamples+1))
    #       sigmaerr = covariance matrix of the error (nv*nsamples,nv*nsamples)
    #       theta = vector of reflection angles (1,nangles)
    #       nv = number of model variables
    # OUTPUT mmap = MAP of posterior distribution (nv*(nsamples+1),1)
    #        mtrans = transformed mean (logGaussian to Gaussian) (nv*(nsamples+1),1)
    #        strans = transformed variance (logGaussian to Gaussian) (nv*(nsamples+1),1)
    #        Time = time vector of elastic properties (nsamples+1,1)

    # parameters
    ntheta = len(theta)

    # logarithm of the prior
    logVp = np.log(Vpprior)
    logVs = np.log(Vsprior)
    logRho = np.log(Rhoprior)
    mprior = np.concatenate([logVp, logVs, logRho])
    nm = logVp.shape[0]

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vpprior, Vsprior, theta, nv)

    # Differential matrix 
    D = DifferentialMatrix(nm, nv)

    # Wavelet matrix
    W = WaveletMatrix(wavelet, nm, ntheta)

    # forward operator
    G = np.dot(np.dot(W, A), D)

    # Bayesian Linearized AVO inversion analytical solution (Buland and Omre, 2003)
    # mean of d
    mdobs = np.dot(G, mprior)
    # covariance matrix
    sigmadobs = np.dot(np.dot(G, sigmaprior), G.T) + sigmaerr

    # posterior mean
    mpost = mprior + np.dot(np.dot(np.dot(sigmaprior.T, G.T), pinvh(sigmadobs)), (Seis - mdobs))
    # posterior covariance matrix
    sigmapost = sigmaprior - np.dot(np.dot(np.dot(np.dot(sigmaprior.T, G.T), pinvh(sigmadobs)), G), sigmaprior)

    # statistical estimators posterior distribution
    mmap = np.exp(mpost - np.diag(sigmapost))

    # posterior distribution
    mtrans = np.exp(mpost + 0.5 * np.diag(sigmapost))
    strans = np.exp(2 * mpost + np.diag(sigmapost)) * (np.exp(np.diag(sigmapost)) - 1)

    # time
    dt = TimeSeis[1] - TimeSeis[0]
    Time = np.arange(TimeSeis[0] - dt/2, TimeSeis[-1] + dt + dt/2, dt)

    return mmap, mtrans, strans, Time


def SeismicModel3D(Vp, Vs, Rho, Time, theta, W, D):
    # SeismicModel3D computes synthetic seismic data according to a linearized
    # seismic model based on the convolution of a wavelet and the linearized
    # approximation of Zoeppritz equations
    # INPUT Vp = P-wave velocity profile
    #       Vs = S-wave velocity profile
    #       Rho = Density profile
    #       theta = vector of reflection angles
    #       W = wavelet matrix
    #       D = differential matrix
    # OUTPUT Seis = vector of seismic data of size (nsamples x nangles, 1)
    #        Time = seismic time  (nsamples, 1)
    #        Cpp  = reflectivity coefficients matrix (nsamples x nangles, nsamples x nangles)

    # number of variables
    nv = 3

    # logarithm of model variables
    logVp = np.log(Vp)
    logVs = np.log(Vs)
    logRho = np.log(Rho)
    m = np.concatenate((logVp, logVs, logRho))

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv)

    # Reflectivity coefficients matrix
    mder = np.dot(D, m)
    Cpp = np.dot(A, mder)

    # Seismic data matrix
    Seis = np.dot(W, Cpp)

    # Time seismic measurements
    TimeSeis = 0.5 * (Time[:-1] + Time[1:])

    return Seis, TimeSeis, Cpp


def WaveletMatrix(wavelet, nsamples, ntheta):
    # WaveletMatrix computes the wavelet matrix for discrete convolution
    # INPUT w = wavelet
    #       nsamples = numbr of samples
    #       ntheta = number of angles
    # OUTUPT W = wavelet matrix

    W = lil_matrix((ntheta * (nsamples - 1), ntheta * (nsamples - 1)))
    _, indmaxwav = np.argmax(wavelet)

    for i in range(1, ntheta + 1):
        wsub = np.convolve(wavelet, np.ones(nsamples - 1), 'full').reshape(-1, 1)
        indsub = np.arange((i - 1) * (nsamples - 1), i * (nsamples - 1))
        W[indsub, indsub] = wsub[indmaxwav : indmaxwav + nsamples - 1]

    return W


def SpatialCovariance(corrlength, dt, nm, sigma0):
    # SpatialCovariance computes the spatial covariance matrix
    # INPUT corrlength = correlation length
    #       dt = time interval
    #       nm = number of samples
    #       sigma0 = covariance matrix
    # OUTPUT sigmaprior = spatial covariance matrix

    trow = np.tile(np.arange(0, nm) * dt, (nm, 1))
    tcol = np.tile(np.arange(0, nm) * dt, (nm, 1)).T
    tdis = np.abs(trow - tcol)
    sigmatime = np.exp(-(tdis / corrlength) ** 2)
    sigma = np.kron(sigma0, sigmatime)

    return sigma
