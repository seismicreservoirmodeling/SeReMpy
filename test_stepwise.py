import numpy as np
import matplotlib.pyplot as plt

from Geostats import NonParametricToUniform


data2transform = np.genfromtxt('Data/NonParametricVariables.csv', delimiter=',')
data2transform = data2transform[1:,:]


variable_uniform = NonParametricToUniform(data2transform[0:1000,:], data2transform[0:1000,:])

fig, axs = plt.subplots(1, 2)

axs[0].hist(data2transform[:,0], bins=15)
axs[1].hist(variable_uniform[:,0], bins=15)


#plt.scatter(data2transform[:,0], data2transform[:,1], c=data2transform[:,2], s=20, marker='o')

plt.show()