import numpy as np
import matplotlib.pyplot as plt

alpha = 1.
T = 1e10
r = 10
beta_list = np.linspace(0.01, 0.99, 100)
e_list = []

for beta in beta_list:
    c_3 = 3.12e-13 * T * r / beta
    c_1 = 1.5 * c_3
    x = 9 * c_3 * np.power(alpha, 2) / 2
    epsilon = 0.5 * (((18 * np.power(alpha, 2) - np.power(x, 2)) / (2 * x)) - 5)
    e_list.append(epsilon)

plt.plot(beta_list, e_list)
plt.show()

gamma_list = []

for beta in beta_list:
    gamma = (32 - 24 * beta - 3 * np.power(beta, 2)) / (24 - 21 * beta)
    gamma_list.append(gamma)

plt.plot(beta_list, gamma_list)
plt.show()
