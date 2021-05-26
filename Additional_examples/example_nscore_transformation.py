
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from context import SeReMpy
from SeReMpy.Geostats import *

# data
x = np.loadtxt('./Data/data1.dat')
reference_log = x[:,1].reshape(-1, 1)

# NSCORE TRANSFORMATION - Transform th non-parametric reference log/data to a Gaussian distributed varible
referencelog_uniform = NonParametricToUniform(reference_log, reference_log)
referencelog_gaussian = norm.ppf(referencelog_uniform) # The variogram should be computed from referencelog_gaussian for reference
# truncate values min and max in 3std to avoid numerical issues
referencelog_gaussian[referencelog_gaussian<-3]=-3
referencelog_gaussian[referencelog_gaussian>3]=3

fig, axs = plt.subplots(1, 3)
axs[0].hist(reference_log[:,0], bins=15)
axs[1].hist(referencelog_uniform[:,0], bins=15)
axs[2].hist(referencelog_gaussian[:,0], bins=15)

# NSCORE BACK TRANSFORMATION - Sequential Gaussian Simulation of a non-parametric distribution
xcoords = np.arange(reference_log.shape[0]).reshape((reference_log.shape[0],1))    
cond_depth = np.array([]).reshape((0,1))    
cond_values = np.array([]).reshape((0,1))    

# Simulate a Gaussian simulation with zero mean and std 1
simulation_gaussian = SeqGaussianSimulation(xcoords, cond_depth, cond_values, 0.0, 1.0, 10.0, 'exp', 0)                      
# Transform from Gaussian to uniform
simulation_uniform = norm.cdf(simulation_gaussian)
# Then, transform from uniform to target non parametric distribution
simulation_non_parametric = UniformToNonParametric(simulation_uniform, reference_log)


plt.figure()
plt.plot(simulation_gaussian)
plt.plot(simulation_uniform)
plt.plot(simulation_non_parametric)

fig, axs = plt.subplots(1, 3)
axs[0].hist(simulation_gaussian[:,0], bins=15)
axs[1].hist(simulation_uniform[:,0], bins=15)
axs[2].hist(reference_log[:,0], bins=15, fc=(1, 0, 0, 0.5))
axs[2].hist(simulation_non_parametric[:,0], bins=15, fc=(0, 0, 0, 0.5))

plt.show()

