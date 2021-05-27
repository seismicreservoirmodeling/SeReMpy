#
# # EXAMPLE OF RUNNING STEPWISE CONDITIONAL TRANSFORMATION # # 
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from context import SeReMpy
from SeReMpy.Geostats import NonParametricToUniform, UniformToNonParametric


# Load example data following a non-parametric six-variate distribution
data2transform = np.genfromtxt('Data/NonParametricVariables.csv', delimiter=',')
data2transform = data2transform[1:,:]

# number of point to use only part of the data
n_ptos = 10000
data2transform = data2transform[0:n_ptos,:]
# transform the non parametric distributed variable to follow a uniform distribution
variable_uniform = NonParametricToUniform(data2transform, data2transform)
# transform back the uniform distributed variables to follow the non parametric distribution
variable_non_parametric = UniformToNonParametric(variable_uniform, data2transform)
# to generate new non-parametric distributed variables, only replace variable_uniform for a uniformly distributed random variables.

# show plots
fig, axs = plt.subplots(1, 2)
axs[0].hist(data2transform[:,0], bins=15)
axs[0].set_title('Non-parametric distributed variable')
axs[1].hist(variable_uniform[:,0], bins=15)
axs[1].set_title('Uniform distributed variable')

## Kernel density distribution in a meshgrid
xmin = data2transform[:,0].min()
xmax = data2transform[:,0].max()
ymin = data2transform[:,1].min()
ymax = data2transform[:,1].max()
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
color_map = plt.cm.hot
color_map = color_map.reversed()

fig, ax = plt.subplots(1,2)
xy = np.vstack([data2transform[:,0], data2transform[:,1]])
kernel = stats.gaussian_kde(xy)
Z = np.reshape(kernel(positions).T, X.shape)
ax[0].imshow(np.rot90(Z), cmap=color_map,extent=[xmin, xmax, ymin, ymax],aspect='auto')
ax[0].set_xlim([xmin, xmax])
ax[0].set_ylim([ymin, ymax])
ax[0].set_title('Non-parametric reference data')
ax[0].set_ylabel('z2')
ax[0].set_xlabel('z1')

xy = np.vstack([variable_non_parametric[:,0], variable_non_parametric[:,1]])
kernel = stats.gaussian_kde(xy)
Z = np.reshape(kernel(positions).T, X.shape)
ax[1].imshow(np.rot90(Z), cmap=color_map,extent=[xmin, xmax, ymin, ymax],aspect='auto')
ax[1].set_xlim([xmin, xmax])
ax[1].set_ylim([ymin, ymax])
ax[1].set_title('Transformed data distributed variable')
ax[1].set_ylabel('z2')
ax[1].set_xlabel('z1')

## Kernel density distribution in the points
# (from: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib)
fig, ax = plt.subplots(1,2)
xy = np.vstack([data2transform[:,0], data2transform[:,1]])
z = stats.gaussian_kde(xy)(xy)
ax[0].scatter(xy[0,:], xy[1,:], c=z, s=10)
ax[0].set_title('Non-parametric reference data')
ax[0].set_ylabel('z2')
ax[0].set_xlabel('z1')

xy = np.vstack([variable_non_parametric[:,0], variable_non_parametric[:,1]])
z = stats.gaussian_kde(xy)(xy)
ax[1].scatter(xy[0,:], xy[1,:], c=z, s=10)
ax[1].set_title('Transformed data distributed variable')
ax[1].set_ylabel('z2')
ax[1].set_xlabel('z1')

plt.show()