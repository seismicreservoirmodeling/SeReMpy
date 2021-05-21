#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:40:50 2020

@author: dariograna
"""

import numpy as np

def BerrymanInclusionModel(Phi, Rho, Kmat, Gmat, Kfl, Ar):
    """
    BERRYMAN INCLUSION MODEL
    Berryman's inclusion model for prolate and oblate spheroids.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Rho : float
        Density of the saturated rock (g/cc).
    Kmat : float
        Bulk modulus of the solid phase (GPa).
    Gmat : float
        Shear modulus of the solid phase (GPa).
    Kfl : float
        Bulk modulus of the fluid phase (GPa).
    Ar : float
        Aspect ratio (unitless).

    Returns
    -------
    Vp : float or array_like 
        P-wave velocity (km/s).
    Vs : float or array_like
        S-wave velocity (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.5
    """

    # inclusion properties 
    Kinc = Kfl
    Ginc = 0

    # Berryman's formulation
    Poisson = (3 * Kmat - 2 * Gmat) / (2 * (3 * Kmat + Gmat))
    theta = Ar / (1 - Ar ** 2) ** (3/ 2) * (np.arccos(Ar) - Ar * np.sqrt(1 - Ar ** 2))
    g = Ar ** 2 / (1 - Ar ** 2) * (3* theta - 2)
    R = (1 - 2* Poisson) / (2 - 2* Poisson)
    A = (Ginc / Gmat) - 1
    B = 1/ 3* (Kinc / Kmat - Ginc / Gmat)
    F1 = 1 + A * (3/ 2* (g + theta) - R * (3/ 2* g + 5/ 2* theta - 4/ 3))
    F2 = 1 + A * (1 + 3/ 2* (g + theta) - R / 2* (3* g + 5 * theta)) + B * (3 - 4* R) + A / 2* (A + 3* B) * (3 - 4* R) * (g + theta - R * (g - theta + 2* theta ** 2))
    F3 = 1 + A * (1 - (g + 3 / 2 * theta) + R * (g + theta))
    F4 = 1 + A / 4* (g + 3* theta - R * (g - theta))
    F5 = A * (R * (g + theta - 4/ 3) - g) + B * theta * (3 - 4 * R)
    F6 = 1 + A * (1 + g - R * (theta + g)) + B * (1 - theta) * (3 - 4 * R)
    F7 = 2 + A / 4 * (9* theta + 3* g - R * (5* theta + 3* g)) + B * theta * (3 - 4* R)
    F8 = A * (1 - 2* R + g / 2* (R - 1) + theta / 2* (5* R - 3)) + B * (1 - theta) * (3 - 4* R)
    F9 = A * (g * (R - 1) - R * theta) + B * theta * (3 - 4* R)
    Tiijj = 3 * F1 / F2
    Tijij = Tiijj / 3 + 2/ F3 + 1/ F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    P = Tiijj / 3
    Q = (Tijij - P) / 5

    # elastic moduli
    Ksat = ((Phi * (Kinc - Kmat) * P) * 4 / 3* Gmat + Kmat * (Kmat + 4 / 3* Gmat)) / (Kmat + 4 / 3* Gmat - (Phi * (Kinc - Kmat) * P))
    psi = (Gmat * (9 * Kmat + 8* Gmat)) / (6 * (Kmat + 2 * Gmat))
    Gsat = (psi * (Phi * (Ginc - Gmat) * Q) + Gmat * (Gmat + psi)) / (Gmat + psi - (Phi * (Ginc - Gmat) * Q))

    # velocities
    Vp = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    Vs = np.sqrt(Gsat / Rho)
    
    return Vp, Vs

def DensityModel(Phi, Rhomat, Rhofl):
    """
    DENSITY MODEL
    Linear porosity-density relation.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Rhomat : float
        Density of the solid phase (g/cc).
    Rhofl : float
        Density of the fluid phase (g/cc).

    Returns
    -------
    Rho : float or array_like
        Density of saturated rock (g/cc).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.1
    """

    Rho = (1 - Phi) * Rhomat + Phi * Rhofl
    
    return Rho

def GassmannModel(Phi, Kdry, Gdry, Kmat, Kfl):
    """
    GASSMANN MODEL
    Gassmann's equations.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Kdry : float
        Bulk modulus of dry rock (GPa).
    Gdry : float
        Shear modulus of dry rock (GPa).
    Kmat : float
        Bulk modulus of solid phase (GPa).
    Kfl : float
        Bulk modulus of the fluid phase (GPa).

    Returns
    -------
    Ksat : float or array_like
        Bulk modulus of saturated rock (GPa).
    Gsat : float or array_like
        Shear modulus of saturated rock (GPa).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.6
    """

    # Bulk modulus of saturated rock
    Ksat = Kdry + ((1 - Kdry / Kmat) ** 2) / (Phi / Kfl + (1 - Phi) / Kmat - Kdry / (Kmat ** 2))
    # Shear modulus of saturated rock
    Gsat = Gdry
    
    return Ksat, Gsat

def LinearizedRockPhysicsModel(Phi, Clay, Sw, R):
    """
    LINEARIZED ROCK PHYSICS MODEL
    Linear rock physics model based on multilinear regression.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Clay : float
        Clay volume (unitless).
    Sw : float
        Water saturation (unitless)
    R : float
        Regression coefficients matrix
        estimated with regress.m

    Returns
    -------
    Vp : float or array_like
        P-wave velocity (km/s).
    Vs : float or array_like
        S-wave velocity (km/s).
    Rho : float or array_like
        Density (g/cc).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.1
    """

    # multilinear regression
    Vp = R[0, 0] * Phi + R[0, 1] * Clay + R[0, 2] * Sw + R[0, 3]
    Vs = R[1, 0] * Phi + R[1, 1] * Clay + R[1, 2] * Sw + R[1, 3]
    Rho = R[2, 0] * Phi + R[2, 1] * Clay + R[2, 2] * Sw + R[2, 3]
    
    return Vp, Vs, Rho

def MatrixFluidModel(Kminc, Gminc, Rhominc, Volminc, Kflc, Rhoflc, Sflc, patchy):
    """
    MATRIX FLUID MODEL
    Computes elastic moduli and density of the solid phase
    and fluid phase using Voigt-Reuss averages.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Kminc : array_like
        1D array of mineral bulk moduli (GPa).
    Gminc : array_like
        1D array of mineral shear moduli (GPa).
    Rhominc : array_like
        1D array of mineral densities (g/cc).
    Volminc : array_like
        2D array of mineral volumes.
    Kflc : array_like
        1D array of fluid bulk moduli (GPa).
    Rhoflc : array_like
        1D array of fluid densities (g/cc ).
    Sflc : array_like
        2D array of fluid saturations.
    patchy : int
        Saturation model: 1=Patchy, 0=Homogeneous

    Returns
    -------
    Kmat : array_like
        Bulk modulus of matrix phase (GPa).
    Gmat : array_like
        Shear modulus of matrix phase (GPa).
    Rhomat : array_like
        Density of matrix phase (g/cc).
    Kfl : array_like
        bulk modulus of fluid phase (GPa).
    Rhofl : array_like
        density of fluid phase (g/cc).

    Notes
    -----
    Kminc, Gminc and Rhominc for a 2-mineral assemblage can be
    entered as [36, 21], [45, 7], [2.6, 2.3], i.e. elements in the 0 position are related to the first mineral component,
    elements in the 1 position are related to the second mineral
    components etc.
    Volminc is a 2D array entered as [mineral1, mineral2] where
    mineral1 and mineral2 are vectors (1D arrays) with length n(n = number of samples).
    Kflc, Rhoflc for 2 fluids are entered as [2.25 0.8] and Rhoflc as [1.0 0.7] for brine and oil.
    Sflc is a 2D array entered as [Sw, 1-Sw] with Sw being the saturation log with number of samples equal to n.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.2
    """

    # number of samples
    n = Volminc.shape[0]
    # initialization variables
    KmatV = np.zeros((n, 1))
    KmatR = np.zeros((n, 1))
    Kmat = np.zeros((n, 1))
    GmatV = np.zeros((n, 1))
    GmatR = np.zeros((n, 1))
    Gmat = np.zeros((n, 1))
    Rhomat = np.zeros((n, 1))
    Kfl = np.zeros((n, 1))
    Rhofl = np.zeros((n, 1))

    for i in range(n):
        # Voigt average (bulk)
        KmatV[i] = np.sum((Volminc[i,:] * Kminc) / np.sum(Volminc[i,:]))
        # Reuss average (bulk)
        KmatR[i]= 1. / np.sum((Volminc[i,:] / Kminc) / np.sum(Volminc[i,:]))
        # Voigt-Reuss-Hill average (bulk)
        Kmat[i]= 0.5 * (KmatV[i] + KmatR[i])
        # Voigt average (shear)
        GmatV[i] = np.sum((Volminc[i,:] * Gminc) / np.sum(Volminc[i,:]))
        # Reuss average (shear)
        GmatR[i]= 1. / np.sum((Volminc[i,:] / Gminc) / np.sum(Volminc[i,:]))
        # Voigt-Reuss-Hill average (shear)
        Gmat[i] = 0.5 * (GmatV[i] + GmatR[i])
        # linear average for matrix density
        Rhomat[i] = np.sum((Volminc[i,:] * Rhominc) / np.sum(Volminc[i,:]))
        if patchy == 0:
            # Reuss average for fluid
            Kfl[i] = 1 / np.sum(Sflc[i,:] / Kflc)
        else:
            # Voigt average for fluid
            Kfl[i] = np.sum(Sflc[i,:] * Kflc)
        # linear average for fluid density
        Rhofl[i] = np.sum(Sflc[i,:] * Rhoflc)

    return Kmat, Gmat, Rhomat, Kfl, Rhofl

def RaymerModel(Phi, Vpmat, Vpfl):
    """
    RAYMER MODEL
    Raymer's equation.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Vpmat : float or array_like
        P-wave velocity of the solid phase (km/s).
    Vpfl : float or array_like
        P-wave velocity of the fluid phase (km/s).

    Returns
    -------
    Vp : float or array_like
        P-wave velocity of saturated rock (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.1
    """

    # Raymer  
    Vp = (1 - Phi) ** 2 * Vpmat + Phi * Vpfl
    
    return Vp

def SoftsandModel(Phi, Rho, Kmat, Gmat, Kfl, critporo, coordnum, press):
    """
    SOFT SAND MODEL
    Dvorkin's soft sand model.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Rho : float
        Density of the saturated rock (g/cc).
    Kmat : float
        Bulk modulus of the solid phase (GPa).
    Gmat : float
        Shear modulus of the solid phase (GPa).
    Kfl : float
        Bulk modulus of the fluid phase (GPa).
    critporo : float
        Critical porosity (unitless).
    coordnum : int
        Coordination number (unitless)
    pressure : float
        Effective pressure (GPa).

    Returns
    -------
    Vp : float or array_like
        P-wave velocity (km/s).
    Vs : float or array_like
        S-wave velocity (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.4
    """

    # Hertz-Mindlin
    Poisson = (3 * Kmat - 2 * Gmat) / (6 * Kmat + 2 * Gmat)
    KHM = ((coordnum ** 2 * (1 - critporo) ** 2 * Gmat **2 * press) / (18 * np.pi ** 2 * (1 - Poisson) **2)) **(1 / 3)
    GHM = (5 - 4 * Poisson) / (10 - 5 * Poisson) * ((3 * coordnum ** 2 * (1 - critporo) ** 2 * Gmat **2 * press) / (2 * np.pi ** 2 * (1 - Poisson) **2)) **(1 / 3)
    # f = friction
    # GHM = (2+3*f-Poisson*(1+3f))./(10-5*Poisson).*((3*coordnumber^2*(1-criticalporo)^2*Gmat.^2*pressure)./(2*np.pi^2*(1-Poisson).^2)).^(1/3);

    # Modified Hashin-Shtrikmann lower bounds
    Kdry = 1. / ((Phi / critporo) / (KHM + 4 / 3 * GHM) + (1 - Phi / critporo) / (Kmat + 4 / 3 * GHM)) - 4 / 3 * GHM
    psi = (9 * KHM + 8 * GHM) / (KHM + 2 * GHM)
    Gdry = 1. / ((Phi / critporo) / (GHM + 1 / 6 * psi * GHM) + (1 - Phi / critporo) / (Gmat + 1 / 6 * psi * GHM)) - 1 / 6 * psi * GHM

    # Gassmann
    [Ksat, Gsat] = GassmannModel(Phi, Kdry, Gdry, Kmat, Kfl)

    # Velocities
    Vp = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    Vs = np.sqrt(Gsat / Rho)
    
    return Vp, Vs

def SphericalInclusionModel(Phi, Rho, Kmat, Gmat, Kfl):
    """
    SPHERICAL INCLUSION MODEL
    Inclusion model for spherical pores.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Rho : float
        Density of the saturated rock (g/cc).
    Kmat : float
        Bulk modulus of the solid phase (GPa).
    Gmat : float
        Shear modulus of the solid phase (GPa).
    Kfl : float
        Bulk modulus of the fluid phase (GPa).

    Returns
    -------
    Vp : float or array_like
        P-wave velocity (km/s).
    Vs : float or array_like
        S-wave velocity (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.5
    """

    # elastic moduli of the dry rock
    Kdry = 4 * Kmat * Gmat * (1 - Phi) / (3 * Kmat * Phi + 4 * Gmat)
    Gdry = Gmat * (9 * Kmat + 8 * Gmat) * (1 - Phi) / ((9 * Kmat + 8 * Gmat + 6 * (Kmat + 2 * Gmat) * Phi))

    # Gassmann
    [Ksat, Gsat] = GassmannModel(Phi, Kdry, Gdry, Kmat, Kfl)

    # Velocities
    Vp = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    Vs = np.sqrt(Gsat / Rho)
    
    return Vp, Vs

def StiffsandModel(Phi, Rho, Kmat, Gmat, Kfl, critporo, coordnum, press):
    """
    STIFF SAND MODEL
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Rho : float
        Density of the saturated rock (g/cc).
    Kmat : float
        Bulk modulus of the solid phase (GPa).
    Gmat : float
        Shear modulus of the solid phase (GPa).
    Kfl : float
        Bulk modulus of the fluid phase (GPa).
    critporo : float
        Critical porosity (unitless).
    coordnum : int
        Coordination number (unitless)
    pressure : float
        Effective pressure (GPa).

    Returns
    -------
    Vp : float or array_like
        P-wave velocity (km/s).
    Vs : float or array_like
        S-wave velocity (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.4
    """

    # Hertz-Mindlin
    Poisson = (3 * Kmat - 2 * Gmat) / (6 * Kmat + 2 * Gmat)
    KHM = ((coordnum ** 2 * (1 - critporo) ** 2 * Gmat ** 2 * press) / (18 * np.pi ** 2 * (1 - Poisson) ** 2)) ** (1 / 3)
    GHM = (5 - 4 * Poisson) / (10 - 5 * Poisson) * ((3 * coordnum ** 2 * (1 - critporo) ** 2 * Gmat ** 2 * press) / (2 * np.pi ** 2 * (1 - Poisson) ** 2)) ** (1 / 3)

    # Modified Hashin-Shtrikmann upper bounds
    Kdry = 1. / ((Phi / critporo) / (KHM + 4 / 3 * Gmat) + (1 - Phi / critporo) / (Kmat + 4 / 3 * Gmat)) - 4 / 3 * Gmat
    psi = (9 * Kmat + 8 * Gmat) / (Kmat + 2 * Gmat)
    Gdry = 1. / ((Phi / critporo) / (GHM + 1 / 6 * psi * Gmat) + (1 - Phi / critporo) / (Gmat + 1 / 6 * psi * Gmat)) - 1 / 6 * psi * Gmat

    # Gassmann
    [Ksat, Gsat] = GassmannModel(Phi, Kdry, Gdry, Kmat, Kfl)

    # Velocities
    Vp = np.sqrt((Ksat + 4 / 3 * Gsat) / Rho)
    Vs = np.sqrt(Gsat / Rho)
    
    return Vp, Vs

def VelocityDefinitions(K, G, Rho):
    """
    VELOCITY DEFINITIONS
    Definitions of P- and S-wave velocity.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    K : float
        Bulk modulus (GPa)
    G : float
        Shear modulus (GPa)
    Rho : float
        Density (g/cc)

    Returns
    -------
    Vp : float or array_like
        P-wave velocity (km/s).
    Vs : float or array_like
        S-wave velocity (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.1
    """

    # definitions
    Vp = np.sqrt((K + 4 / 3 * G) / Rho)
    Vs = np.sqrt(G / Rho)
    
    return Vp, Vs

def WyllieModel(Phi, Vpmat, Vpfl):
    """
    WYLLIE MODEL
    Wyllie's equation.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Phi : float or array_like
        Porosity (unitless).
    Vpmat : float or array_like
        P-wave velocity of the solid phase (km/s).
    Vpfl : float or array_like
        P-wave velocity of the fluid phase (km/s).
        
    Returns
    -------
    Vp : float or array_like
        P-wave velocity of saturated rock (km/s).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 2.1
    """

    # Wyllie 
    Vp = 1 / ((1 - Phi) / Vpmat + Phi / Vpfl)
    
    return Vp


