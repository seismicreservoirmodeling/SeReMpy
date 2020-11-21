#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:40:50 2020

@author: dariograna
"""

import numpy as np

def BerrymanInclusionModel(Phi, Rho, Kmat, Gmat, Kfl, Ar):

    # BERRYMAN INCLUSION MODEL implements Berryman's inclusion model for
    # prolate and oblate spheroids
    # INPUT Phi = Porosity
    #       Rho = Density of the saturated rock
    #       Kmat = Bulk modulus of the solid phase
    #       Gmat = Shear modulus of the solid phase
    #       Kfl = Bulk modulus of the fluid phase
    #       Ar = Aspect ratio
    # OUTUPT Vp = P=wave velocity
    #        Vs = S-wave velocity

    # Written by Dario Grana (August 2020)

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

    # DENSITY MODEL implements the linear porosity-density relation
    # INPUT Phi = Porosity
    #       Rhomat = Density of the solid phase
    #       Rhofl = Density of the fluid phase
    # OUTUPT Rho = Density of saturated rock

    # Written by Dario Grana (August 2020)

    Rho = (1 - Phi) * Rhomat + Phi * Rhofl
    
    return Rho

def GassmannModel(Phi, Kdry, Gdry, Kmat, Kfl):

    # GASSMANN MODEL implements Gassmann's equations 
    # INPUT Phi = Porosity
    #       Kdry = Bulk modulus of dry rock
    #       Gdry = Shear modulus of dry rock
    #       Kmat = Bulk modulus of solid phase
    #       Kfl = Bulk modulus of fluid rock
    # OUTUPT Ksat = Bulk modulus of saturated rock
    #        Gsat = Shear modulus of saturated rock

    # Written by Dario Grana (August 2020)

    # Bulk modulus of saturated rock
    Ksat = Kdry + ((1 - Kdry / Kmat) ** 2) / (Phi / Kfl + (1 - Phi) / Kmat - Kdry / (Kmat ** 2))
    # Shear modulus of saturated rock
    Gsat = Gdry
    
    return Ksat, Gsat

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

def MatrixFluidModel(Kminc, Gminc, Rhominc, Volminc, Kflc, Rhoflc, Sflc, patchy):

    # MATRIX FLUID MODEL computes elastic moduli and density of the solid phase
    # and fluid phase using Voigt-Reuss averages
    # INPUT Kminc = Row vector of bulk moduli of minerals in GPa (ex [36 21])
    #       Gminc = Row vector of shear moduli of minerals in GPa (ex [45 7])
    #       Rhominc = Row vector of densities of minerals in g/cc (ex [2.6 2.3])
    #       Volminc = Matrix of volumes. Each column is a mineral volume log
    #                 (ex [vquartz 1-vclay])
    #       Kflc = Row vector of bulk moduli of fluid components in GPa 
    #               (ex [2.25 0.8 0.1])
    #       Rhoflc = Row vector of densities of fluid components in g/cc 
    #               (ex [1.03 0.7 0.02])
    #       Volminc = Matrix of saturations. Each column is a saturation log
    #                 (ex [sw so sg 1-vclay])
    #       patchy = binary variable: 1=Patchy; 0=Homegeneous
    # OUTUPT Kmat = bulk modulus of matrix phase
    #        Gmat = shear modulus of matrix phase
    #        Rhomat = density of matrix phase
    #        Kfl = bulk modulus of fluid phase
    #        Rhofl = density of fluid phase

    # Written by Dario Grana (August 2020)

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
        Gmat[i] = 0.5 * (GmatV(i) + GmatR(i))
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

    # RAYMER MODEL implements Raymer's equation 
    # INPUT Phi = Porosity
    #       Vpmat = P-wave velocity of the solid phase
    #       Vpfl = P-wave velocity of the fluid phase
    # OUTUPT Vp = P-wave velocity of saturated rock

    # Written by Dario Grana (August 2020)

    # Raymer  
    Vp = (1 - Phi) ** 2 * Vpmat + Phi * Vpfl
    
    return Vp

def SoftsandModel(Phi, Rho, Kmat, Gmat, Kfl, critporo, coordnum, press):

    # SOFT SAND MODEL implements Dvorkin's soft sand model
    # INPUT Phi = Porosity
    #       Rho = Density of the saturated rock
    #       Kmat = Bulk modulus of the solid phase
    #       Gmat = Shear modulus of the solid phase
    #       Kfl = Bulk modulus of the fluid phase
    #       critporo = critical porosity
    #       coordnum = coordination number
    #       pressure = effective pressure in GPA
    # OUTUPT Vp = P=wave velocity
    #        Vs = S-wave velocity

    # Written by Dario Grana (August 2020)

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

    # SPHERICAL INCLUSION MODEL implements the inclusion model for spherical
    # pores
    # INPUT Phi = Porosity
    #       Rho = Density of the saturated rock
    #       Kmat = Bulk modulus of the solid phase
    #       Gmat = Shear modulus of the solid phase
    #       Kfl = Bulk modulus of the fluid phase
    # OUTUPT Vp = P=wave velocity
    #        Vs = S-wave velocity

    # Written by Dario Grana (August 2020)

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

    # STIFF SAND MODEL implements Dvorkin's soft sand model
    # INPUT Phi = Porosity
    #       Rho = Density of the saturated rock
    #       Kmat = Bulk modulus of the solid phase
    #       Gmat = Shear modulus of the solid phase
    #       Kfl = Bulk modulus of the fluid phase
    #       critporo = critical porosity
    #       coordnum = coordination number
    #       press = effective pressure in GPA
    # OUTUPT Vp = P=wave velocity
    #        Vs = S-wave velocity

    # Written by Dario Grana (August 2020)

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

    # VELOCITY DEFINITIONS implements the definitions of P- and S-wave velocity
    # INPUT K = Bulk modulus
    #       G = Shear modulus
    #       Rho = Density
    # OUTUPT Vp = P-wave velocity
    #        Vs = S-wave velocity

    # Written by Dario Grana (August 2020)

    # definitions
    Vp = np.sqrt((K + 4 / 3 * G) / Rho)
    Vs = np.sqrt(G / Rho)
    
    return Vp, Vs

def WyllieModel(Phi, Vpmat, Vpfl):

    # WYLLIE MODEL implements Wyllie's equation 
    # INPUT Phi = Porosity
    #       Vpmat = P-wave velocity of the solid phase
    #       Vpfl = P-wave velocity of the fluid phase
    # OUTUPT Vp = P-wave velocity of saturated rock

    # Written by Dario Grana (August 2020)

    # Wyllie 
    Vp = 1 / ((1 - Phi) / Vpmat + Phi / Vpfl)
    
    return Vp


