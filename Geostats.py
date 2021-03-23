# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:05:53 2020

@author: dariograna
"""

import numpy as np
import numpy.matlib
from numpy.linalg import matrix_power
import scipy .spatial


def CorrelatedSimulation(mprior, sigma0, sigmaspace):

    # CORRELATED SIMULATION generates 1D stochastic realizations of correlated
    # multiple random variables with a spatial correlation model
    # INPUT mprior = prior trend (nsamples, nvariables)
    #       sigma0 = stationary covariance matrix (nvariables, nvariables)
    #       sigmaspace =spatial covariance matrix (nsamples, nsamples)
    # OUTPUT msim = stochastic realization (nsamples, nvariables)

    # Written by Dario Grana (August, 2020)

    # initial parameters
    nm = mprior.shape[1]
    ns = mprior.shape[0]

    # spatial covariance matrix
    sigma = np.kron(sigma0, sigmaspace)

    # spatially correlated realization
    mreal = np.random.multivariate_normal(mprior.T.flatten(), sigma)
    msim = np.zeros((ns, nm))
    for i in range(nm):
        msim[:,i] = mreal[i*ns:(i+1)*ns]
    
    return msim


def ExpCov(h, l):

    # EXP COV computes the exponential covariance function
    # INPUT h = distance
    #       l = correlation length (or range)
    # OUTPUT C = covariance

    # Written by Dario Grana (August, 2020)

    # covariance function
    C = np.exp(-3 * h / l)
    
    return C


def GauCov(h, l):

    # GAU COV computes the Gaussian covariance function
    # INPUT h = distance
    #       l = correlation length (or range)
    # OUTPUT C = covariance

    # Written by Dario Grana (August, 2020)

    # covariance function
    C = np.exp(-3 * h ** 2 / l ** 2)
    
    return C


def SphCov(h, l):

    # SPH COV computes the spherical covariance function
    # INPUT h = distance
    #       l = correlation length (or range)
    # OUTPUT C = covariance

    # Written by Dario Grana (August, 2020)

    # covariance function
    C = np.zeros(h.shape)
    #C(h <= l).lvalue = 1 - 3 / 2 * h(h <= l) / l + 1 / 2 * h(h <= l) ** 3 / l ** 3
    C[h <= l] = 1 - 3 / 2 * h[h <= l] / l + 1 / 2 * h[h <= l] ** 3 / l ** 3

    return C


def GaussianSimulation(xcoord, dcoords, dvalues, xmean, xvar, l, krigtype, krig):

    # GAUSSIAN SIMULATION  generates a realization of the random variable 
    # conditioned on the available measurements
    # INPUT xcoord = coordinates of the location for the estimation (1, ndim)
    #       dcoords = coordinates of the measurements (ns, ndim)
    #       dvalues = values of the measurements (ns, 1)
    #       xmean = prior mean
    #       xvar = prior variance
    #       h = distance
    #       l = correlation length
    #       type = function ype ('exp', 'gau', 'sph')
    #       krig = kriging type (0=simple, 1=ordinary)
    # OUTPUT sgsim = realization

    # Written by Dario Grana (August, 2020)

    if krig == 0:
        krigmean, krigvar = SimpleKriging(xcoord, dcoords, dvalues, xmean, xvar, l, krigtype)
    else:
        krigmean, krigvar = OrdinaryKriging(xcoord, dcoords, dvalues, xvar, l, krigtype)

    # realization
    sgsim = krigmean + np.sqrt(krigvar) * np.random.randn(1)
    
    return sgsim


def IndicatorKriging(xcoord, dcoords, dvalues, nf, pprior, l, krigtype):

    # INDICATOR KRIGING computes the indicator kriging estimate and variance 
    # INPUT xcoord = coordinates of the location for the estimation (1, ndim)
    #       dcoords = coordinates of the measurements (ns, ndim)
    #       dvalues = values of the measurements (ns, 1)
    #       nf = number of possible outcomes (e.g. number of facies)
    #       pprior = prior probability (1,nf)
    #       h = distance 
    #       l = correlation range, for different range for each facies, l is an array with nf components
    #       type = function type ('exp', 'gau', 'sph') for different type for each facies, type is an array wuth nf components
    # OUTPUT ikp = indicator kriging probability
    #        ikmap = maximum a posteriori of indicator kriging probability

    # Written by Dario Grana (August, 2020)
    
    # If l and krigtype are single parameters, use it for all facies
    if type(l)==float:
        l = np.tile(l, (nf, 1))
    if type(krigtype)==str:
        krigtype = np.tile(krigtype, (nf, 1))

    # indicator variables
    nd = dvalues.shape[0]
    indvar = np.zeros((nd, nf))
    for i in range(nd):
        indvar[i, dvalues[i].astype(int)] = 1

    # kriging weights
    xdtemp = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.vstack((xcoord, dcoords))))
   
    distvect = xdtemp[1:,0]
    distmatr = xdtemp[1:,1:]
    varprior = np.zeros((1,nf))
    krigvect = np.zeros((nd,nf))
    krigmatr = np.zeros((nd,nd,nf))
    wkrig = np.zeros((nd, nf))
    for j in range(nf):
        varprior[:,j]= pprior[j] * (1 - pprior[j])
        krigvect[:,j]= varprior[:,j] * SpatialCovariance1D(distvect, l[j], krigtype[j])
        krigmatr[:,:,j] = varprior[:,j] * SpatialCovariance1D(distmatr, l[j], krigtype[j])
        wkrig[:,j] = np.linalg.lstsq(krigmatr[:,:,j], krigvect[:,j])[0]
        

    # indicator kriging probability
    ikp = np.zeros((1, nf))
    for j in range(nf):
        ikp[0,j] = pprior[j] + sum(wkrig[:,j] * (indvar[:,j]- pprior[j]))

    # Should we only normalize ikp, do we have to truncate?        
    #ikp[ikp<0] = 0;ikp[ikp>1] = 1
    ikp = ikp/ikp.sum()
    ikmap = np.argmax(ikp, axis=1)
    
    return ikp, ikmap


def OrdinaryKriging(xcoord, dcoords, dvalues, xvar, l, krigtype):

    # ORDINARY KRIGING computes the ordinary kriging estimate and variance 
    # INPUT xcoord = coordinates of the location for the estimation (1, ndim)
    #       dcoords = coordinates of the measurements (ns, ndim)
    #       dvalues = values of the measurements (ns, 1)
    #       xvar = prior variance
    #       h = distance
    #       l = correlation length
    #       type = function ype ('exp', 'gau', 'sph')
    # OUTPUT xok = kriging estimate
    #        xvarok = kriging variance

    # Written by Dario Grana (August, 2020)

    # kriging matrix and vector
    nd = dcoords.shape[0]
    krigmatr = np.ones((nd + 1, nd + 1))
    krigvect = np.ones((nd + 1, 1))
    xdtemp = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.vstack((xcoord, dcoords))))
    distvect = xdtemp[1:,0]
    distmatr = xdtemp[1:,1:]
    krigvect[0:-1,0] = xvar * SpatialCovariance1D(distvect, l, krigtype)
    krigmatr[0:-1,0:-1] = xvar * SpatialCovariance1D(distmatr, l, krigtype)
    krigmatr[-1,-1] = 0
    # to avoid numerical issue, specially with Gaussian variogram model
    krigmatr = krigmatr + 0.000001*xvar*np.eye(krigmatr.shape[0])

    # kriging weights
    wkrig = np.linalg.lstsq(krigmatr, krigvect)[0]

    # kriging mean
    # xok = mean(dvalues)+sum(wkrig(1:end-1).*(dvalues-mean(dvalues)));
    xok = np.sum(wkrig[0:- 1] * dvalues)
    # kriging variance
    xvarok = xvar - np.sum(wkrig * krigvect)
    
    return xok, xvarok


def SimpleKriging(xcoord, dcoords, dvalues, xmean, xvar, l, krigtype):

    # SIMPLE KRIGING computes the simple kriging estimate and variance 
    # INPUT xcoord = coordinates of the location for the estimation (1, ndim)
    #       dcoords = coordinates of the measurements (ns, ndim)
    #       dvalues = values of the measurements (ns, 1)
    #       xmean = prior mean
    #       xvar = prior variance
    #       h = distance
    #       l = correlation length
    #       type = function ype ('exp', 'gau', 'sph')
    # OUTPUT xsk = kriging estimate
    #        xvarsk = kriging variance

    # Written by Dario Grana (August, 2020)

    # kriging matrix and vector
    nd = dcoords.shape[0]
    krigmatr = np.ones((nd, nd))
    krigvect = np.ones((nd, 1))
    xdtemp = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.vstack((xcoord, dcoords))))
    distvect = xdtemp[1:,0]
    distmatr = xdtemp[1:,1:]
    krigvect[:,0] = xvar * SpatialCovariance1D(distvect, l, krigtype)
    krigmatr = xvar * SpatialCovariance1D(distmatr, l, krigtype)
    # to avoid numerical issue, specially with Gaussian variogram model
    krigmatr = krigmatr + 0.000001*xvar*np.eye(krigmatr.shape[0])

    # kriging weights
    wkrig = np.linalg.lstsq(krigmatr, krigvect)[0]

    # kriging mean
    xsk = xmean + np.sum(wkrig * (dvalues - xmean))
    # kriging variance
    xvarsk = xvar - np.sum(wkrig * krigvect)
    
    return xsk, xvarsk


def MarkovChainSimulation(T, ns, nsim):

    # MARKOV CHAIN SIMULATION simulates 1D realizations of a discrete random 
    # variable based on a stationary first-order Markov chain with given
    # transition probability matrix 
    # INPUT T = transition  probability matrix 
    #       ns = number of samples
    #       nsim = number of simulations
    # OUTUPT fsim = realizations (ns, nsim)

    # Written by Dario Grana (August 2020)

    fsim = np.zeros((ns, nsim))
    fsim = fsim.astype(int)
    Tpow = matrix_power(T, 100)
    fprior = np.zeros((1,T.shape[1]))
    fprior[0,:] = Tpow[0,:]
    for j in range(nsim):
        fsim[0, j] = RandDisc(fprior)
        for i in range(1,ns):
            fcond = np.zeros((1,T.shape[1]))
            fcond[0,:] = T[fsim[i-1,j], :]
            fsim[i, j] = RandDisc(fcond)
    
    return fsim


def RadialCorrLength(lmin, lmax, azim, theta):

    # RADIAL CORR LENGTH computes the radial correlation length 
    # INPUT lmin = minimum correlation length
    #       lmax = aaximum correlation length
    #       azim = azimuth
    #       theta = radial coordinate
    # OUTPUT l = radial correlation length

    # Written by Dario Grana (August, 2020)

    # covariance function
    l = np.sqrt((lmin ** 2 * lmax ** 2) / (lmax ** 2 * (np.sin(azim - theta)) ** 2 + lmin ** 2 * (np.cos(azim - theta)) ** 2))
                                              
    return l


def SpatialCovariance1D(h, l, krigtype):

    # SPATIAL COVARIANCE 1D computes the 1D spatial covariance function 
    # INPUT l = correlation length
    #       h = distance
    #       type = function ype ('exp', 'gau', 'sph')
    # OUTPUT C = covariance

    # Written by Dario Grana (August, 2020)

    # covariance function
    if krigtype == 'exp':
        C = ExpCov(h, l)
    elif krigtype == 'gau':
        C = GauCov(h, l)
    elif krigtype == 'sph':
        C = SphCov(h, l)  
    else:
        print('error')   
        
    return C


def SpatialCovariance2D(lmin, lmax, azim, theta, h, krigtype):

    # SPATIAL COVARIANCE 2D computes the 2D anisotropic spatial covariance 
    # function 
    # INPUT lmin = minimum correlation length
    #       lmax = aaximum correlation length
    #       azim = azimuth
    #       theta = radial coordinate
    #       h = distance
    #       type = function ype ('exp', 'gau', 'sph')
    # OUTPUT C = covariance

    # Written by Dario Grana (August, 2020)

    # covariance function
    if krigtype == 'exp':
        C = ExpCov(h, RadialCorrLength(lmin, lmax, azim, theta))
    elif krigtype == 'gau':
        C = GauCov(h, RadialCorrLength(lmin, lmax, azim, theta))
    elif krigtype == 'sph':
        C = SphCov(h, RadialCorrLength(lmin, lmax, azim, theta))
    else:
        print('error')    
        
    return C    


def SeqGaussianSimulation(xcoords, dcoords, dvalues, xmean, xvar, l, krigtype, krig):

    # SEQ GAUSSIAN SIMULATION  generates a realization of the random variable 
    # conditioned on the available measurements using Sequential Gaussian
    # Simulation
    # INPUT xcoords = coordinates of the locations for the estimation (np, ndim)
    #       dcoords = coordinates of measurements (nd,ndim)
    #       dvalues = values of measurements (nd,1)
    #       xmean = prior mean
    #       xvar = prior variance
    #       h = distance
    #       l = correlation length
    #       type = function ype ('exp', 'gau', 'sph')
    #       krig = kriging type (0=simple, 1=ordinary)
    # OUTPUT sgsim = realization

    # Written by Dario Grana (August, 2020)

    # initial parameters
    n = xcoords.shape[0]
    nd = dcoords.shape[0]

    # maximum number of conditioning data
    nmax = 12

    # Data assignment to the simulation grid (-999 or measurements)
    sgsim = -999 * np.ones((n, 1))
    for i in range(nd):
        ind = np.argmin(np.sum((xcoords - dcoords[i, :]) ** 2, axis=1))
        sgsim[ind] = dvalues[i]

    # random path of locations
    npl = n - nd
    nonsimcoords = xcoords[sgsim[:,0] == -999, :]
    pathind = np.random.permutation(range(npl))
    pathcoords = nonsimcoords[pathind, :]
    simval = np.zeros((npl, 1))

    # sequential simulation
    for i in range(npl):
        if dcoords.shape[0] < nmax:
            dc = dcoords
            dz = dvalues
        else:
            # conditioning data selection 
            dv = []
            dv = np.sqrt(np.sum((dcoords - pathcoords[i, :]) ** 2, axis=1))
            ind = np.argsort(dv)
            dc = dcoords[ind[0:nmax-1],:]
            dz = dvalues[ind[0:nmax-1]]

        # kriging
        if krig == 0:
            krigmean, krigvar = SimpleKriging(pathcoords[i,:], dc, dz, xmean, xvar, l, krigtype)
        else:
            krigmean, krigvar = OrdinaryKriging(pathcoords[i,:], dc, dz, xvar, l, krigtype)
  
        # realization
        simval[pathind[i],0] = krigmean + np.sqrt(krigvar) * np.random.randn(1)
        # Adding simulated value the vector of conditioning data
        dcoords = np.vstack((dcoords, pathcoords[i, :]))
        dvalues = np.vstack((dvalues, simval[pathind[i]]))

    # Assigning the sampled values to the simulation grid
    sgsim[sgsim[:,0] == -999, 0] = simval[:,0]     
    
    return sgsim


def SeqIndicatorSimulation(xcoords, dcoords, dvalues, nf, pprior, l, krigtype):

    # SEQ INDICATOR SIMULATION  generates a realization of the discrete random 
    # variable conditioned on the available measurements using Sequential 
    # Indicator Simulation
    # INPUT xcoords = coordinates of the locations for the estimation (np, ndim)
    #       dcoords = coordinates of measurements (nd,ndim)
    #       dvalues = values of the measurements (ns, 1)
    #       nf = number of possible outcomes (e.g. number of facies)
    #       pprior = prior probability (1,nf)
    #       h = distance
    #       l = correlation range, for different range for each facies, l is an array with nf components
    #       type = function type ('exp', 'gau', 'sph') for different type for each facies, type is an array wuth nf components
    # OUTPUT sgsim = realization

    # Written by Dario Grana (August, 2020)

    # initial parameters
    n = xcoords.shape[0]
    nd = dcoords.shape[0]

    # maximum number of conditioning data
    nmax = 12

    # Data assignment to the simulation grid (-999 or measurements)
    sgsim = -999 * np.ones((n, 1))
    for i in range(nd):
        ind = np.argmin(np.sum((xcoords - dcoords[i, :]) ** 2, axis=1))
        sgsim[ind] = dvalues[i]


    # random path of locations
    npl = n - nd
    nonsimcoords = xcoords[sgsim[:,0] == -999, :]
    pathind = np.random.permutation(range(npl))
    pathcoords = nonsimcoords[pathind, :]
    simval = np.zeros((npl, 1))

    # sequential simulation
    for i in range(npl):
        if dcoords.shape[0] < nmax:
            dc = dcoords
            dz = dvalues
        else:
            # conditioning data selection 
            dv = []
            dv = np.sqrt(np.sum((dcoords - pathcoords[i, :]) ** 2, axis=1))
            ind = np.argsort(dv)
            dc = dcoords[ind[0:nmax-1],:]
            dz = dvalues[ind[0:nmax-1]]
            dz = dz.astype(int)
            
        ikprob, ikmap = IndicatorKriging(pathcoords[i,:], dc, dz, nf, pprior, l, krigtype)

        # realization        
        simval[pathind[i]] = RandDisc(ikprob)
        # Adding simulated value the vector of conditioning data
        dcoords = np.vstack((dcoords, pathcoords[i, :]))
        dvalues = np.vstack((dvalues, simval[pathind[i]]))

    # Assigning the sampled values to the simulation grid
    sgsim[sgsim[:,0] == -999, 0] = simval[:,0]     
    
    return sgsim


def RandDisc(p):

    # RANDDISC samples a discrete random variable with a given probability
    # mass function
    # INPUT p = probabilities 
    # OUTUPT index = sampled value

    # Written by Dario Grana (August 2020)

    u = np.random.rand(1)
    index = 0
    s = p[0,0]
    while ((u > s) and (index < p.shape[1])):
        index = index + 1
        s = s + p[0,index]
    
    return index
