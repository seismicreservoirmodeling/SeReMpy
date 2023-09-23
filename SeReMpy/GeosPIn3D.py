#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:08:27 2023

@author: Dario Grana and Mingliang Liu
"""

import numpy as np
import scipy.sparse
from scipy.linalg import pinvh
from scipy.fftpack import fftn, ifftn
from scipy.linalg import toeplitz
# from numpy.random import multivariate_normal
from scipy.stats import multivariate_normal
from scipy.sparse import eye, diags, lil_matrix


def AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv):
    # AkiRichardsCoefficientsMatrix computes the Aki Richards coefficient matrix
    # INPUT Vp = P-wave velocity profile
    #       Vs = S-wave velocity profile
    #       theta = vector of reflection angles
    #       nv = number of model variables 
    # OUTPUT A = Aki Richards coefficients matrix

    # initial parameters
    nt, ntrace = Vp.shape
    ntheta = len(theta)
    A = np.zeros(((nt - 1) * ntheta, nv * (nt - 1), ntrace))

    # average velocities at the interfaces
    avgVp = 1 / 2 * (Vp[:-1] + Vp[1:])
    avgVs = 1 / 2 * (Vs[:-1] + Vs[1:])

    # reflection coefficients (Aki Richards linearized approximation)
    for i in range(ntheta):
        cp = 1 / 2 * (1 + np.tan(theta[i] * np.pi / 180) ** 2) * np.ones(nt - 1)
        cs = -4 * (avgVs ** 2) / (avgVp ** 2) * np.sin(theta[i] * np.pi / 180) ** 2
        cr = 1 / 2 * (1 - 4 * (avgVs ** 2) / (avgVp ** 2) * np.sin(theta[i] * np.pi / 180) ** 2)
        Acp = np.array([np.diag(cp) for _ in range(ntrace)]).transpose(1, 2, 0)
        Acs = np.array([np.diag(cs[:, i]) for i in range(ntrace)]).transpose(1, 2, 0)
        Acr = np.array([np.diag(cr[:, i]) for i in range(ntrace)]).transpose(1, 2, 0)
        A[i * (nt - 1): (i + 1) * (nt - 1), :, :] = np.concatenate([Acp, Acs, Acr], axis=1)

    return A


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
    # rock physics likelihood
    M1, M2, M3 = np.meshgrid(phigrid, claygrid, swgrid, indexing='ij')
    M1 = M1.flatten(order='F').reshape(-1, 1)
    M2 = M2.flatten(order='F').reshape(-1, 1)
    M3 = M3.flatten(order='F').reshape(-1, 1)
    petrogrid = np.hstack([M1, M2, M3])

    M1, M2, M3 = np.meshgrid(vpgrid, vsgrid, rhogrid, indexing='ij')
    M1 = M1.flatten(order='F').reshape(-1, 1)
    M2 = M2.flatten(order='F').reshape(-1, 1)
    M3 = M3.flatten(order='F').reshape(-1, 1)
    elasgrid = np.hstack([M1, M2, M3])
    Ppetro = RockPhysicsGMM(faciestrain, petrotrain, elastrain, petrogrid, elasgrid, rpsigmaerr)

    # inversion
    for i in range(nxl):
        print(f'{i}/{nxl}')
        # print(f'Percentage progress: {round(i / nxl * 100)} %')
        Seis = np.hstack([near[i], mid[i], far[i]])
        mmap, mtrans, strans, Time = SeismicInversion3D(Seis.T, TimeSeis, vpprior[i].T, vsprior[i].T, rhoprior[i].T,
                                                        sigmaprior, sigmaerr, wavelet, theta, nv)
        m_inv = mtrans.reshape(nv, -1, nil)
        Pseis = [[multivariate_normal.pdf(elasgrid, mean=m_inv[:, jjj, iii],
                                          cov=np.diag([strans[nm//2, iii], strans[nm+nm//2, iii], strans[2*nm+nm//2, iii]]))
                  for iii in range(nil)] for jjj in range(nm)]
        Pseis = np.array(Pseis)#.transpose(2, 0, 1)
        Ppost = np.einsum('ij,kli->jkl', Ppetro, Pseis, optimize=True)
        Ppost = Ppost.reshape(ndiscr, ndiscr, ndiscr, nm, nil)
        # marginal posterior distributions
        Pphi = np.sum(Ppost, axis=(0, 1))
        Pclay = np.sum(Ppost, axis=(0, 2))
        Psw = np.sum(Ppost, axis=(1, 2))

        Pphi = Pphi / np.sum(Pphi, axis=0)
        Pclay = Pclay / np.sum(Pclay, axis=0)
        Psw = Psw / np.sum(Psw, axis=0)

        Phimap[i] = phigrid[np.argmax(Pphi, axis=0), 0].T
        Claymap[i] = claygrid[np.argmax(Pclay, axis=0), 0].T
        Swmap[i] = swgrid[np.argmax(Psw, axis=0), 0].T

    return Vpmap, Vsmap, Rhomap, Phimap, Claymap, Swmap, Time


def BayesPetroInversion3D_KDE(near, mid, far, TimeSeis, vpprior, vsprior, rhoprior, sigmaprior, elastrain, petrotrain, vpgrid, vsgrid, rhogrid, phigrid, claygrid, swgrid, sigmaerr, wavelet, theta, nv, h):
    nxl, nil, nm = near.shape[0], near.shape[1], near.shape[2] + 1
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
    vpgrid, vsgrid, rhogrid = np.meshgrid(vpgrid, vsgrid, rhogrid, indexing='ij')
    M1 = vpgrid.flatten(order='F').reshape(-1, 1)
    M2 = vsgrid.flatten(order='F').reshape(-1, 1)
    M3 = rhogrid.flatten(order='F').reshape(-1, 1)
    elasevalpoints = np.hstack([M1, M2, M3])
    Ppetro = RockPhysicsKDE(petrotrain, elastrain, petrogrid, elasgrid, elasevalpoints, hm, hd)

    # inversion
    for i in range(nxl):
        print(f'{i}/{nxl}')
        # print(f'Percentage progress: {round(i / nxl * 100)} %')
        Seis = np.hstack([near[i], mid[i], far[i]])
        mmap, mtrans, strans, Time = SeismicInversion3D(Seis.T, TimeSeis, vpprior[i].T, vsprior[i].T, rhoprior[i].T,
                                                        sigmaprior, sigmaerr, wavelet, theta, nv)
        m_inv = mtrans.reshape(nv, -1, nil)
        Pseis = [[multivariate_normal.pdf(elasevalpoints, mean=m_inv[:, jjj, iii],
                                          cov=np.diag([strans[nm // 2, iii], strans[nm + nm // 2, iii],
                                                       strans[2 * nm + nm // 2, iii]]))
                  for iii in range(nil)] for jjj in range(nm)]
        Pseis = np.array(Pseis)  # .transpose(2, 0, 1)
        Ppost = np.einsum('ij,kli->jkl', Ppetro, Pseis, optimize=True)
        Ppost = Ppost.reshape(ndiscr, ndiscr, ndiscr, nm, nil)
        # marginal posterior distributions
        Pphi = np.sum(Ppost, axis=(1, 2))
        Pclay = np.sum(Ppost, axis=(0, 2))
        Psw = np.sum(Ppost, axis=(0, 1))

        Pphi = Pphi / np.sum(Pphi, axis=0)
        Pclay = Pclay / np.sum(Pclay, axis=0)
        Psw = Psw / np.sum(Psw, axis=0)

        Phimap[i] = phigrid[np.argmax(Pphi, axis=0)].T
        Claymap[i] = claygrid[np.argmax(Pclay, axis=0)].T
        Swmap[i] = swgrid[np.argmax(Psw, axis=0)].T

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

    I = np.eye(nt)
    B = np.zeros((nt, nt))
    B[1:, 0:- 1] = -np.eye(nt - 1)
    I = (I + B)
    J = I[1:, :]
    D = np.zeros(((nt - 1) * nv, nt * nv))
    for i in range(nv):
        D[i * (nt - 1):(i + 1) * (nt - 1), i * nt:(i + 1) * nt] = J

    return D


def EpanechnikovKernel(u):
    return (np.abs(u) <= 1) * (3/4) * (1 - u**2)


def SeismicModel1D(Vp, Vs, Rho, Time, theta, W, D):
    # SeismicModel1D computes synthetic seismic data according to a linearized
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
    mder = D @ m
    Cpp = A @ mder

    # Seismic data matrix
    Seis = W @ Cpp

    # Time seismic measurements
    TimeSeis = 0.5 * (Time[:-1] + Time[1:])

    return Seis, TimeSeis, Cpp

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
        SeisPred[:, k], _, _ = SeismicModel1D(Vp[:, k], Vs[:, k], Rho[:, k], Time, theta, W, D)

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

    for i in range(nsim):
        print('Percentage progress of simulation: {:.1f} %'.format(i / nsim * 100))
        SeisPred[:, :, :, i], _, _ = SeismicModel3D(Vp[:, :, :, i], Vs[:, :, :, i], Rho[:, :, :, i], Time, theta, W, D)

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

    for h in range(niter):
        for i in range(nxl):
            print('Percentage progress of inversion:', (h*nxl+i) / (niter*nxl) * 100, '%')
            PostModels = np.concatenate((Phisim[i], Claysim[i], Swsim[i]), axis=1)
            SeisData = np.concatenate((near[i], mid[i], far[i]), axis=-1)
            SeisData = np.expand_dims(SeisData, axis=-1)
            SeisPred = SeisSim[i]
            PostModels, _ = EnsembleSmootherMDA(PostModels, SeisData, SeisPred, alpha, sigmaerr)
            Phipost = PostModels[:, :nm, :]
            Claypost = PostModels[:, nm:2 * nm, :]
            Swpost = PostModels[:, 2 * nm:, :]
            Phipost[Phipost < 0] = 0
            Phipost[Phipost > 0.4] = 0.4
            Claypost[Claypost < 0] = 0
            Claypost[Claypost > 0.8] = 0.8
            Swpost[Swpost < 0] = 0
            Swpost[Swpost > 1] = 1


            Phisim[i] = Phipost
            Claysim[i] = Claypost
            Swsim[i] = Swpost


        Vp, Vs, Rho = LinearizedRockPhysicsModel(Phisim, Claysim, Swsim, regcoef)
        D = DifferentialMatrix(nm, nv)
        W = WaveletMatrix(wavelet, nm, len(theta))

        for i in range(nsim):
            SeisSim[:, :, :, i], _, _ = SeismicModel3D(Vp[:, :, :, i], Vs[:, :, :, i], Rho[:, :, :, i], Time, theta, W, D)

    # posterior mean models
    Phimap = np.mean(Phisim, axis=-1)
    Claymap = np.mean(Claysim, axis=-1)
    Swmap = np.mean(Swsim, axis=-1)

    return Phimap, Claymap, Swmap, Time



def LinearizedRockPhysicsModel(Phi, Clay, Sw, R):
    # LINEARIZED ROCK PHYSICS MODEL implements a linear rock physics model
    # based on a multilinear regression
    # INPUT Phi = Porosity
    #       Clay = Clay volume
    #       Sw = Shear modulus of dry rock
    #       R = regression coefficients matrix (estimated with regress.m)
    # OUTUPT Vp = P-wave velocity
    #        Vs = S-wave velocity
    #        Rho = Density

    # Written by Dario Grana (August 2020)

    # multilinear regression
    Vp = R[0, 0] * Phi + R[0, 1] * Clay + R[0, 2] * Sw + R[0, 3]
    Vs = R[1, 0] * Phi + R[1, 1] * Clay + R[1, 2] * Sw + R[1, 3]
    Rho = R[2, 0] * Phi + R[2, 1] * Clay + R[2, 2] * Sw + R[2, 3]

    return Vp, Vs, Rho


def EnsembleSmootherMDA(PriorModels, SeisData, SeisPred, alpha, sigmaerr):
    # ENSEMBLE SMOOTHER MDA computes the updated realizations of the
    # model variables conditioned on the assimilated data using the
    # Ensemble Smoother Multiple Data Assimilation
    # INPUT PriorModels = prior models realizations (nm, ne)
    #       SeisData = measured seismic data (nil, nd, 1)
    #       SeisPred = predicted data (nd, ne)
    #       alpha = inflation coefficient
    #       sigmaerr = covariance matrix of the error (nd, nd)
    # OUTPUT PostModels = updated models realizations (nm, ne)
    #        KalmanGain = Kalman Gain Matrix

    # initial parameters
    nil, nd, ne = SeisPred.shape
    # data perturbation
    SeisPert = SeisData + np.sqrt(alpha * sigmaerr) @ np.random.randn(nil, nd, ne)
    # mean models
    mum = np.expand_dims(np.mean(PriorModels, axis=-1), axis=-1)
    mud = np.expand_dims(np.mean(SeisPred, axis=-1), axis=-1)
    # covariance matrices
    smd = 1 / (ne - 1) * np.einsum('ijk,ilk->ijl', PriorModels - mum, SeisPred - mud, optimize=True)
    sdd = 1 / (ne - 1) * np.einsum('ijk,ilk->ijl', SeisPred - mud, SeisPred - mud, optimize=True)
    # Kalman Gain
    sigmadobs = sdd + alpha * np.expand_dims(sigmaerr, axis=0)
    sigmainv = np.linalg.pinv(sigmadobs)
    KalmanGain = np.einsum('ijk,ikl->ijl', smd, sigmainv, optimize=True)
    # Updated models
    mdelta = np.einsum('ijk,ikl->ijl', KalmanGain, SeisPert - SeisPred, optimize=True)
    PostModels = PriorModels + mdelta

    return PostModels, KalmanGain


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
    # corrprior = (np.diag(np.sqrt(np.diag(corrpetro))) @ corrpetro) / np.diag(np.sqrt(np.diag(corrpetro)))

    for k in range(nsim):
        uncorrsim = np.random.multivariate_normal([0, 0, 0], corrpetro, nxl * nil * nm)
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
        mjoint[k - 1, :] = np.mean(datatrain[ftrain[:, 0] == k, :], axis=0)
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
        sigmapost[:, :, k - 1] = sm[:, :, k - 1] - smd[:, :, k - 1] @ np.linalg.inv(sd[:, :, k - 1] + sigmaerr) @ sdm[:,
                                                                                                                  :,
                                                                                                                  k - 1]

    for i in range(ns):
        for k in range(1, nf + 1):
            # posterior means
            mupost[i, :, k - 1] = mum[k - 1] + smd[:, :, k - 1] @ np.linalg.inv(sd[:, :, k - 1] + sigmaerr) @ (
                        dcond[i] - mud[k - 1])
            # posterior weights
            pfpost[i, k - 1] = pf[k - 1] * multivariate_normal.pdf(dcond[i], mean=mud[k - 1], cov=sd[:, :, k - 1])

        if np.sum(pfpost[i]) > 0:
            pfpost[i] /= np.sum(pfpost[i])

        lh = np.zeros(mdomain.shape[0])
        for k in range(1, nf + 1):
            lh += pfpost[i, k - 1] * multivariate_normal.pdf(mdomain, mean=mupost[i, :, k - 1],
                                                             cov=sigmapost[:, :, k - 1])

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
    mgridrep = np.tile(mgrid.T, [nt, 1, 1])
    datamrep = np.tile(mtrain[..., np.newaxis], [1, 1, mgrid.shape[0]])
    # multidimensional grids for data variables
    vecd = np.meshgrid(*ddomain.T, indexing='ij')
    dgrid = np.column_stack([d.flatten() for d in vecd])
    dgridrep = np.tile(dgrid.T, [nt, 1, 1])
    datadrep = np.tile(dtrain[..., np.newaxis], [1, 1, dgrid.shape[0]])
    # kernel density estimation
    hmmat = np.tile(hm[..., np.newaxis], [nt, 1, mgrid.shape[0]])
    prodm = np.prod(EpanechnikovKernel((mgridrep - datamrep) / hmmat), axis=1)
    
    hdmat = np.tile(hd[..., np.newaxis], [nt, 1, dgrid.shape[0]])
    prodd = np.prod(EpanechnikovKernel((dgridrep - datadrep) / hdmat), axis=1)

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
    mprior = np.concatenate([logVp, logVs, logRho], axis=0)
    nm = logVp.shape[0]

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vpprior, Vsprior, theta, nv)

    # Differential matrix 
    D = DifferentialMatrix(nm, nv)

    # Wavelet matrix
    W = WaveletMatrix(wavelet, nm, ntheta)

    # forward operator
    G = np.einsum('mi,ijk,jn->mnk', W, A, D, optimize=True)

    # Bayesian Linearized AVO inversion analytical solution (Buland and Omre, 2003)
    # mean of d
    mdobs = np.einsum('ijk,jk->ik', G, mprior, optimize=True)
    # covariance matrix
    sigmadobs = np.einsum('ijk,jm,mnk->ink', G, sigmaprior, G.transpose(1, 0, 2), optimize=True)
    sigmadobs = sigmadobs + np.expand_dims(sigmaerr, axis=-1)
    sigmainv = np.linalg.pinv(sigmadobs.transpose(2, 0, 1)).transpose(1, 2, 0)
    # posterior mean
    ddelta = Seis - mdobs
    mdelta = np.einsum('ij,jkm,knm,nm->im', sigmaprior, G.transpose(1, 0, 2), sigmainv, ddelta, optimize=True)
    mpost = mprior + mdelta
    # posterior covariance matrix
    sigmadelta = np.einsum('ij,jkm,knm,npm,pq->iqm', sigmaprior, G.transpose(1, 0, 2), sigmainv, G, sigmaprior, optimize=True)
    sigmapost = np.expand_dims(sigmaprior, axis=-1) - sigmadelta

    # statistical estimators posterior distribution
    sigmapost_diag = np.diagonal(sigmapost, axis1=0, axis2=1).T
    mmap = np.exp(mpost - sigmapost_diag)

    # posterior distribution
    mtrans = np.exp(mpost + 0.5 * sigmapost_diag)
    strans = np.exp(2 * mpost + sigmapost_diag) * (np.exp(sigmapost_diag) - 1)

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

    nxl, nil, nm = Vp.shape
    ntrace = nxl * nil
    Vp = Vp.reshape(ntrace, nm).T
    Vs = Vs.reshape(ntrace, nm).T
    Rho = Rho.reshape(ntrace, nm).T
    # number of variables
    nv = 3

    # logarithm of model variables
    logVp = np.log(Vp)
    logVs = np.log(Vs)
    logRho = np.log(Rho)
    m = np.concatenate((logVp, logVs, logRho), axis=0)

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv)

    # Reflectivity coefficients matrix
    mder = D @ m
    Cpp = np.einsum('ijk,jk->ik', A, mder, optimize=True)
    # Seismic data matrix
    Seis = W @ Cpp

    # Time seismic measurements
    TimeSeis = 0.5 * (Time[:-1] + Time[1:])

    Seis = np.reshape(Seis.T, (nxl, nil, -1))
    Cpp = np.reshape(Cpp.T, (nxl, nil, -1))

    return Seis, TimeSeis, Cpp


def convmtx(w, ns):
    if len(w) < ns:
        a = np.r_[w[0], np.zeros(ns - 1)]
        b = np.r_[w, np.zeros(ns - 1)]
    else:
        b = np.r_[w[0], np.zeros(ns - 1)]
        a = np.r_[w, np.zeros(ns - 1)]
    C = toeplitz(a, b)

    return C


def WaveletMatrix(wavelet, ns, ntheta):
    # WaveletMatrix computes the wavelet matrix for discrete convolution
    # INPUT w = wavelet
    #       ns = numbr of samples
    #       ntheta = number of angles
    # OUTUPT W = wavelet matrix

    W = np.zeros((ntheta * (ns - 1), ntheta * (ns - 1)))
    indmaxwav = np.argmax(wavelet)

    for i in range(ntheta):
        wsub = convmtx(wavelet, (ns - 1))
        wsub = wsub.T
        W[i * (ns - 1):(i + 1) * (ns - 1), i * (ns - 1):(i + 1) * (ns - 1)] = wsub[indmaxwav:indmaxwav + (ns - 1), :]

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
