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
    """
    CORRELATED SIMULATION
    Generates 1D stochastic realizations of correlated
    multiple random variables with a spatial correlation model.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    mprior : array_like
        Prior trend (nsamples, nvariables).
    sigma0 : array_like
        Stationary covariance matrix (nvariables, nvariables).
    sigmaspace : array_like
        Spatial covariance matrix (nsamples, nsamples).

    Returns
    -------
    msim : array_like
        Stochastic realization (nsamples, nvariables).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.6
    """

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
    """
    EXP COV
    Computes the exponential covariance function.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    h : float or array_like
        Distance.
    l : float or array_like
        Correlation length (or range).
        
    Returns
    -------
    C : array_like
        Covariance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.2
    """

    # covariance function
    C = np.exp(-3 * h / l)
    
    return C

def GauCov(h, l):
    """
    GAU COV
    Computes the Gaussian covariance function.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    h : float or array_like
        Distance.
    l : float or array_like
        Correlation length (or range).

    Returns
    -------
    C : array_like
        Covariance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.2
    """

    # covariance function
    C = np.exp(-3 * h ** 2 / l ** 2)
    
    return C

def SphCov(h, l):
    """
    SPH COV
    Computes the spherical covariance function.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    h : float or array_like
        Distance.
    l : float or array_like
        Correlation length (or range).

    Returns
    -------
    C : array_like
        Covariance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.2
    """

    # covariance function
    C = np.zeros(h.shape)
    #C(h <= l).lvalue = 1 - 3 / 2 * h(h <= l) / l + 1 / 2 * h(h <= l) ** 3 / l ** 3
    C[h <= l] = 1 - 3 / 2 * h[h <= l] / l + 1 / 2 * h[h <= l] ** 3 / l ** 3

    return C

def GaussianSimulation(xcoord, dcoords, dvalues, xmean, xvar, l, krigtype, krig):
    """
    GAUSSIAN SIMULATION
    Generates a realization of the random variable conditioned on
    the available measurements.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    xcoord : array_like
        Coordinates of the location for the estimation (1, ndim).
    dcoords : array_like
        Coordinates of the measurements (ns, ndim).
    dvalues : array_like
        Values of the measurements (ns, 1).
    xmean : float
        Prior mean.
    xvar : float
        Prior variance.
    h : float
        Distance.
    l : float
        Correlation length.
    krigtype : str
        Function type ('exp', 'gau', 'sph').
    krig : int
        Kriging type (0=simple, 1=ordinary).

    Returns
    -------
    sgsim : array_like
        Realization (nsamples, nvariables).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.5
    """

    if krig == 0:
        krigmean, krigvar = SimpleKriging(xcoord, dcoords, dvalues, xmean, xvar, l, krigtype)
    else:
        krigmean, krigvar = OrdinaryKriging(xcoord, dcoords, dvalues, xvar, l, krigtype)

    # realization
    sgsim = krigmean + np.sqrt(krigvar) * np.random.randn(1)
    
    return sgsim

def IndicatorKriging(xcoord, dcoords, dvalues, nf, pprior, l, krigtype):
    """
    INDICATOR KRIGING
    Computes the indicator kriging estimate and variance.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    xcoord : array_like
        Coordinates of the location for the estimation (1, ndim).
    dcoords : array_like
        Coordinates of the measurements (ns, ndim).
    dvalues : array_like
        Values of the measurements (ns, 1).
    nf : int
        Number of possible outcomes (e.g. number of facies).
    pprior : array_like
        Prior probability (1, nf).
    h : float or array_like
        Distance.
    l : float or array_like
        Correlation range, for different range for each facies
        (array with nf components).
    krigtype : str
        Function type ('exp', 'gau', 'sph') for different type for each facies,
        (array with nf components).

    Returns
    -------
    ikp : array_like
        Indicator kriging probability.
    ikmap : array_like
        Maximum a posteriori of indicator kriging probability.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 4.1
    """

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
        wkrig[:,j] = np.linalg.lstsq(krigmatr[:,:,j], krigvect[:,j],rcond=None)[0]
        

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
    """
    ORDINARY KRIGING
    Computes the ordinary kriging estimate and variance.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    xcoord : array_like
        Coordinates of the location for the estimation (1, ndim).
    dcoords : array_like
        Coordinates of the measurements (ns, ndim).
    dvalues : array_like
        Values of the measurements (ns, 1).
    xvar : float
        Prior variance.
    l : float
        Correlation length
    krigtype : str
        Function type ('exp', 'gau', 'sph').

    Returns
    -------
    xok : array_like
        Kriging estimate.
    xvarok : array_like
        Kriging variance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.4
    """

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
    wkrig = np.linalg.lstsq(krigmatr, krigvect,rcond=None)[0]

    # kriging mean
    # xok = mean(dvalues)+sum(wkrig(1:end-1).*(dvalues-mean(dvalues)));
    xok = np.sum(wkrig[0:- 1] * dvalues)
    # kriging variance
    xvarok = xvar - np.sum(wkrig * krigvect)
    
    return xok, xvarok

def SimpleKriging(xcoord, dcoords, dvalues, xmean, xvar, l, krigtype):
    """
    SIMPLE KRIGING
    Computes the simple kriging estimate and variance.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    xcoord : array_like
        Coordinates of the location for the estimation (1, ndim).
    dcoords : array_like
        Coordinates of the measurements (ns, ndim).
    dvalues : array_like
        Values of the measurements (ns, 1).
    xmean : float
        Prior mean.
    xvar : float
        Prior variance.
    l : float
        Correlation length.
    krigtype : str
        Function type ('exp', 'gau', 'sph').

    Returns
    -------
    xsk : array_like
        Kriging estimate.
    xvarsk : array_like
        Kriging variance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.4
    """

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
    wkrig = np.linalg.lstsq(krigmatr, krigvect,rcond=None)[0]

    # kriging mean
    xsk = xmean + np.sum(wkrig * (dvalues - xmean))
    # kriging variance
    xvarsk = xvar - np.sum(wkrig * krigvect)
    
    return xsk, xvarsk
    
def MarkovChainSimulation(T, ns, nsim):
    """
    MARKOV CHAIN SIMULATION
    Simulates 1D realizations of a discrete random variable based on
    a stationary first-order Markov chain with given transition probability matrix.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    T : array_like
        Transition  probability matrix.
    ns : int
        Number of samples.
    nsim : int
        Number of simulations.

    Returns
    -------
    fsim : array_like
        Realizations (ns, nsim).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 4.4
    """

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
    """
    RADIAL CORR LENGTH
    Computes the radial correlation length.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    lmin : float
        Minimum correlation length.
    lmax : float
        Maximum correlation length.
    azim : float
        Azimuth.
    theta : float
        Radial coordinate.

    Returns
    -------
    l : float
        Radial correlation length.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.2
    """

    # covariance function
    l = np.sqrt((lmin ** 2 * lmax ** 2) / (lmax ** 2 * (np.sin(azim - theta)) ** 2 + lmin ** 2 * (np.cos(azim - theta)) ** 2))
                                              
    return l

def SpatialCovariance1D(h, l, krigtype):
    """
    SPATIAL COVARIANCE 1D
    Computes the 1D spatial covariance function.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    l : float
        Correlation length.
    h : float
        Distance.
    krigtype : str
        Function type ('exp', 'gau', 'sph').

    Returns
    -------
    C : array_like
        Covariance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.2
    """

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
    """
    SPATIAL COVARIANCE 2D
    Computes the 2D anisotropic spatial covariance function.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    lmin : float
        Minimum correlation length.
    lmax : float
        Maximum correlation length.
    azim : float
        Azimuth.
    theta : float
        Radial coordinate.
    h : float
        Distance.
    krigtype : str
        Function type ('exp', 'gau', 'sph').

    Returns
    -------
    C : array_like
        Covariance.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.2
    """

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
    """
    SEQ GAUSSIAN SIMULATION
    Generates a realization of the random variable conditioned on
    the available measurements using Sequential Gaussian Simulation.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    xcoord : array_like
        Coordinates of the location for the estimation (np, ndim).
    dcoords : array_like
        Coordinates of the measurements (nd, ndim).
    dvalues : array_like
        Values of the measurements (nd, 1).
    xmean : float or array (for local variable mean)
        Prior mean.
    xvar : float
        Prior variance.
    l : float
        Correlation length.
    krigtype : str
        Function type ('exp', 'gau', 'sph').
    krig : int
        Kriging type (0=simple, 1=ordinary).

    Returns
    -------
    sgsim : array_like
        Realization.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 3.5
    """

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

    # if the xmean is a single value, transform to an array
    if type(xmean) == float:
        xmean = xmean*np.ones((n, 1))    

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
            krigmean, krigvar = SimpleKriging(pathcoords[i,:], dc, dz, xmean[pathind[i]], xvar, l, krigtype)            
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
    """
    SEQ INDICATOR SIMULATION
    Generates a realization of the discrete random variable conditioned on
    the available measurements using Sequential Indicator Simulation.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    xcoord : array_like
        Coordinates of the location for the estimation (np, ndim).
    dcoords : array_like
        Coordinates of the measurements (nd, ndim).
    dvalues : array_like
        Values of the measurements (ns, 1).
    nf : int
        Number of possible outcomes (e.g. number of facies).
    pprior : array_like
        Prior probability (1, nf).
    l : float or array_like
        Correlation range, for different range for each facies
        (array with nf components).
    krigtype : str
        Function type ('exp', 'gau', 'sph') for different type for each facies,
        (array with nf components).

    Returns
    -------
    sgsim : array_like
        Realization.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 4.2
    """

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
    """
    RANDDISC
    Samples a discrete random variable with a given probability mass function.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    p : array_like
        Probabilities.
        
    Returns
    -------
    index : array_like
        Sampled value.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 4.4
    """
    u = np.random.rand(1)
    index = 0
    s = p[0,0]
    while ((u > s) and (index < p.shape[1])):
        index = index + 1
        s = s + p[0,index]
    
    return index

def NonParametricToUniform(data2transform, reference_variables, gridsize=0.05):
    """    
    STEPWISE CONDITIONAL TRANSFORMATION (non-par to uniform)
    Tranform a non-parametric distributed variables to a uniformly distributed variables following the stepwise transformation approach

    REFRERENCE:
    Direct Multivariate Simulation - A stepwise conditional transformation for multivariate geostatistical simulation
    de Figueiredo et al., 2020

    Written by Leandro P. de Figueiredo (May 2021)    

    Parameters
    ----------
    data2transform : array_like
        Non-parametric distributed variables to be transformed to a uniform distribution, each line is a simulation value and each column is a different variable.
    reference_variables : array_like
        Non-parametric distributed variables to be used as the reference distribution, each line is a simulation value/point and each column is a different variable..
    gridsize : float
        Grid size for conditioning. Low values may cause not enought points to compute the conditional distribution. High values may cause a non accurate transformation.
        
    Returns
    -------
    variable_uniform : array_like
        Uniformly distributed variables related to data2transform.
    """
    # Treatment to ensure that the method works with inputs arrays of shape (n,) or (n,n_variables)
    n_points = data2transform.shape[0]
    n_variables = int(data2transform.ravel().shape[0]/data2transform.shape[0])
    if n_variables == 1:
        data2transform = data2transform.reshape( ( data2transform.shape[0], n_variables) )
        reference_variables = reference_variables.reshape( ( reference_variables.shape[0], n_variables) )

    # Normalize the input variables
    min2norm = reference_variables.min(axis=0)
    reference_variables = reference_variables - np.tile(min2norm, (reference_variables.shape[0], 1))
    max2norm = reference_variables.max()
    reference_variables = reference_variables / np.tile(max2norm, (reference_variables.shape[0], 1))
    data2transform = data2transform - np.tile(min2norm, (data2transform.shape[0], 1))
    data2transform = data2transform / np.tile(max2norm, (data2transform.shape[0], 1))

    variable_uniform = np.zeros( data2transform.shape )
    for i in np.arange(data2transform.shape[0]):

        reference_variables_filtered = reference_variables   

        for var in np.arange(n_variables):
            # Compute the empirical cumulative distribution for the variable var
            empirical_cumulative = np.sort(reference_variables_filtered[:,var], axis=None)            
            if empirical_cumulative.shape[0]>1:
                # If we have data/statistics/enough data
                empirical_cumulative = empirical_cumulative + np.arange(empirical_cumulative.shape[0])*0.000000001 #infinitesimal line to avoid numerical issues in interp
                # Apply cumulative distribution to the varible:
                variable_uniform[i,var] = np.interp(data2transform[i,var], empirical_cumulative, np.arange(empirical_cumulative.shape[0])/empirical_cumulative.shape[0])
            else:
                variable_uniform[i,var] = 0.5   

            # Filter the data to obtain the statistics of the distribution given by the drawn variables variable_uniform[i,0:var]
            distance_in_axis = abs( reference_variables_filtered[:,var] - data2transform[i,var] )
            index = np.nonzero( distance_in_axis > gridsize )            
            reference_variables_filtered = np.delete(reference_variables_filtered, index, axis=0)

    return variable_uniform


def UniformToNonParametric(data2transform, reference_variables, gridsize=0.05):
    """    
    STEPWISE CONDITIONAL TRANSFORMATION (uniform to non-par)
    Tranform a uniformly distributed  variables to a non-parametric target distributed variables following the stepwise transformation approach

    REFRERENCE:
    Direct Multivariate Simulation - A stepwise conditional transformation for multivariate geostatistical simulation
    de Figueiredo et al., 2020

    Written by Leandro P. de Figueiredo (May 2021)    

    Parameters
    ----------
    data2transform : array_like
        Uniformly distributed variables to be transformed to a non parametric distribution, each line is a simulation value/point and each column is a different variable.
    reference_variables : array_like
        Non-parametric distributed variables to be used as the reference distribution, each line is a simulation value/point and each column is a different variable..
    gridsize : float
        Grid size for conditioning. Low values may cause not enought points to compute the conditional distribution. High values may cause a non accurate transformation.
        
    Returns
    -------
    variable_uniform : array_like
        Uniformly distributed transformed variables of data2transform.
    """
    # Treatment to ensure that the method works with inputs arrays of shape (n,) or (n,n_variables)
    n_points = data2transform.shape[0]
    n_variables = int(data2transform.ravel().shape[0]/data2transform.shape[0])
    if n_variables == 1:
        data2transform = data2transform.reshape( ( data2transform.shape[0], n_variables) )
        reference_variables = reference_variables.reshape( ( reference_variables.shape[0], n_variables) )

    # Normalize the input variables
    min2norm = reference_variables.min(axis=0)
    reference_variables = reference_variables - np.tile(min2norm, (reference_variables.shape[0], 1))
    max2norm = reference_variables.max()
    reference_variables = reference_variables / np.tile(max2norm, (reference_variables.shape[0], 1))

    num_point_without_statistic = 0
    variable_nonParametric = np.zeros( data2transform.shape )
    for i in np.arange(data2transform.shape[0]):

        reference_variables_filtered = reference_variables   

        for var in np.arange(n_variables):
            # Compute the empirical cumulative distribution for the variable var
            empirical_cumulative = np.sort(reference_variables_filtered[:,var], axis=None)            

            if empirical_cumulative.shape[0]>1:
                # If we have data/statistics/enough data
                empirical_cumulative = empirical_cumulative + np.arange(empirical_cumulative.shape[0])*0.000000001 #infinitesimal line to avoid numerical issues in interp
                # Apply cumulative distribution to the varible:
                variable_nonParametric[i,var] = np.interp(data2transform[i,var], np.arange(empirical_cumulative.shape[0])/empirical_cumulative.shape[0], empirical_cumulative)
            else:
                # If we do not have data/statistics/enough data, draw from the marginal instead of conditional
                num_point_without_statistic = num_point_without_statistic + 1
                empirical_cumulative = np.sort(reference_variables[:,var], axis=None)
                variable_nonParametric[i,var] = np.interp(data2transform[i,var], np.arange(empirical_cumulative.shape[0])/empirical_cumulative.shape[0], empirical_cumulative)

            # Filter the data to obtain the statistics of the distribution given by the drawn variables variable_uniform[i,0:var]
            distance_in_axis = abs( reference_variables_filtered[:,var] - variable_nonParametric[i,var] )
            index = np.nonzero( distance_in_axis > gridsize )            
            reference_variables_filtered = np.delete(reference_variables_filtered, index, axis=0)


    variable_nonParametric = variable_nonParametric * np.tile(max2norm, (variable_nonParametric.shape[0], 1))
    variable_nonParametric = variable_nonParametric + np.tile(min2norm, (variable_nonParametric.shape[0], 1))   

    return variable_nonParametric