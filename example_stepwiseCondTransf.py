#
# # EXAMPLE OF RUNNING STEPWISE CONDITIONAL TRANSFORMATION # # 
# 
import numpy as np
import matplotlib.pyplot as plt

from Geostats import NonParametricToUniform, UniformToNonParametric

# Load example data following a non-parametric six-variate distribution
data2transform = np.genfromtxt('Data/NonParametricVariables.csv', delimiter=',')
data2transform = data2transform[1:,:]

# number of point to use only part of the data
n_ptos = 5000
# transform the non parametric distributed variable to follow a uniform distribution
variable_uniform = NonParametricToUniform(data2transform[0:n_ptos,:], data2transform)
# transform back the uniform distributed variables to follow the non parametric distribution
variable_non_parametric = UniformToNonParametric(variable_uniform, data2transform)

# show plots
fig, axs = plt.subplots(1, 2)
axs[0].hist(data2transform[:,0], bins=15)
axs[1].hist(variable_uniform[:,0], bins=15)

fig, axs = plt.subplots(1, 3)
axs[0].scatter(data2transform[:,0], data2transform[:,1], c=data2transform[:,2], s=20, marker='o')
axs[1].scatter(variable_uniform[:,0], variable_uniform[:,1], c=variable_uniform[:,2], s=20, marker='o')
axs[2].scatter(variable_non_parametric[:,0], variable_non_parametric[:,1], c=variable_non_parametric[:,2], s=20, marker='o')

plt.show()