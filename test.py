import fundamental_constants as fc
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt

t_squared = []
bessel = []
for T in np.logspace(5, 11, 100):

    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))

    bessel.append(special.kn(2, 1. / theta))
    t_squared.append(theta ** 2)

plt.plot(np.logspace(5, 11, 100), t_squared)
plt.plot(np.logspace(5, 11, 100), bessel)
plt.xscale('log')
plt.yscale('log')
plt.show()