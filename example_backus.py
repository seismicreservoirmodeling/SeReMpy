
import numpy as np
import matplotlib.pyplot as plt

from RockPhysics import BackusAverageIsotropic

n_ptos = 500

Vp = 3500 + 200 * np.random.randn(n_ptos, )
Vs = 2500 + 100 * np.random.randn(n_ptos, )
Rho =  2.5 + 0.1 * np.random.randn(n_ptos, )

window_depth = 1
d_depth = 0.1

Vp_backus, Vs_backus, Rho_backus = BackusAverageIsotropic(Vp, Vs, Rho, window_depth, d_depth)

fig, axs = plt.subplots(3)
axs[0].plot(Vp)
axs[0].plot(Vp_backus)
axs[1].plot(Vs)
axs[1].plot(Vs_backus)
axs[2].plot(Rho)
axs[2].plot(Rho_backus)
plt.show()