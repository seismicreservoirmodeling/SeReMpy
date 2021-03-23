# Facies.py

ok

# Geostats.py

### GaussianSimulation

* check if types are correct (xcoord, dcoords are 2d arrays; but xvar, xmean, f, l etc --> are they floats?)

### GaussianSimulation, IndicatorKriging, SeqIndicatorSimulation,

* l and type are arrays here? or vectors (1d arrays)?

### RandDisc

* check types

# Inversion.py

### RickerWavelet

* dt (sample rate), secs or msecs?

### InvLogitBounded

* check types of w, index (float? int? arrays?)

# RockPhysics.py

### LinearizedRockPhysicsModel

* Sw: is this shear modulus really or water saturation?
* where is regress.m (or its python equivalent) to estimate regression coefficients matrix?

### MatrixFluidModel

* Kflc, Rhoflc, etc specify if they are 1D or 2D array
* Sflc = 2D array? (n, m) with n = number of samples and m = number of volume fractions available, like fluid1, fluid2, shale, qtz

### RaymerModel

if phi is entered as an array, then it all works correctly; but what happens if Vpmat and/or Vpfl are entered as arrays like:

This is ok:

    RaymerModel(0.3, 5000, 1600)
    Out[54]: 2929.9999999999995

This is ok:

    RaymerModel(np.linspace(0.1,0.3), 5000, 1600)
    Out[55]: 
    array([4210.   , [...] 2952.124, 2930.   ])

but this?

    RaymerModel(0.3, np.linspace(4000,5000), 1600)

or this?

    RaymerModel(0.3, np.linspace(4000,5000), np.linspace(1400,1600))

***

v1 = np.random.rand(10)
v2 = 1-v1

Kminc = np.array([36, 21])
Gminc = np.array([45, 7])
Rhominc = np.array([2.2, 2.7])
Volminc = np.column_stack((v1, v2))
Kflc = np.array([2.25, 0.1])
Rhoflc = np.array([1.03, 0.7])

s1 = np.random.rand(10)
s2 = 1-s1
Sflc = np.column_stack((s1, s2))


MatrixFluidModel(Kminc, Gminc, Rhominc, Volminc, Kflc, Rhoflc, Sflc, 0)



Kminc = [36, 21]
Gminc = [45, 7]
Rhominc = [2.2, 2.7]
Kflc = [2.25, 0.1, 0.06]
Rhoflc = [1.03, 0.7, 0.1]
s1 = np.random.rand(10)
s2 = np.random.rand(10)*.1
s3 = 1-s1-s2
Sflc = np.column_stack((s1, s2, s3))

MatrixFluidModel(Kminc, Gminc, Rhominc, Volminc, Kflc, Rhoflc, Sflc, 0)