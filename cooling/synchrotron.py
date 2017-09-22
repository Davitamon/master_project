import numpy as np
import fundamental_constants as fc
import scipy.special as special
from scipy.optimize import fsolve
import general_functions as gf
import scipy.integrate as integrate


def _integrand(nu, T, n_lep, B):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    nu_0 = fc.e * B / (2 * np.pi * fc.m_e * fc.c)
    x_M = 2 * nu / (3 * nu_0 * np.power(theta, 2))
    I = 4.0505 / np.power(x_M, 1 / 6.) * (1 + 0.4 / np.power(x_M, 1 / 4.) + 0.5316 / np.power(x_M, 1 / 2.)) * np.exp(-1.8899 * np.power(x_M, 1 / 3.))
    # print(theta, nu, nu_0)
    # print(I, special.kn(2, 1. / theta), x_M, np.exp(-1.8899 * np.power(x_M, 1 / 3.)))
    return 4.43e-30 * 4 * np.pi * nu * n_lep * I / special.kn(2, 1. / theta)


def _to_solve(nu, T, n_lep, B, H):
    return _integrand(nu, T, n_lep, B) - 2 * np.pi * fc.k * T * np.power(nu, 2) / (H * np.power(fc.c, 2))


def sychrotron_self_absorbed_freq(T, H, n_lep, B):
    abs_freq = fsolve(_to_solve, np.array([3.e12]), args=(T, n_lep, B, H))[0]
    # print('absfreq', '%e' % abs_freq)
    return abs_freq


def cooling(T, H, nu_c, n_lep, B):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    nu_0 = fc.e * B / (2 * np.pi * fc.m_e * fc.c)
    a_1 = 2 / (3 * nu_0 * np.power(theta, 2))
    a_2 = 0.4 / np.power(a_1, 1/ 4.)
    a_3 = 0.5316 / np.power(a_1, 1 / 2.)
    a_4 = 1.8899 * np.power(a_1, 1 / 3.)
    self_abs = 2 * np.pi * fc.k * T * np.power(nu_c, 3) / (3 * H * np.power(fc.c, 2))
    thin = 6.76e-28 * n_lep / (special.kn(2, 1. / theta) * np.power(a_1, 1 / 6.)) *\
           (1 / np.power(a_4, 11 / 2.) * gf.gamma_function(11 / 2., a_4 * np.power(nu_c, 1 / 3.))
            + a_2 / np.power(a_4, 19 / 4.) * gf.gamma_function(19 / 4., a_4 * np.power(nu_c, 1 / 3.))
            + a_3 / np.power(a_4, 4) * (np.power(a_4, 3) * nu_c
                                        + 3 * np.power(a_4, 2) * np.power(nu_c, 2 / 3.)
                                        + 6 * a_4 * np.power(nu_c, 1 / 3.)
                                        + 6) * np.exp(-a_4 * np.power(nu_c, 1 / 3.)))
    # thin1 = 6.76e-28 * n_lep / (special.iv(2, 1. / theta) * np.power(a_1, 1 / 6.))
    # thin2 = 1 / np.power(a_4, 11 / 2.) * gf.gamma_function(11 / 2., a_4 * np.power(nu_c, 1 / 3.))
    # thin3 = a_2 / np.power(a_4, 19 / 4.) * gf.gamma_function(19 / 4., a_4 * np.power(nu_c, 1 / 3.))
    # thin4 = a_3 / np.power(a_4, 4) * (np.power(a_4, 3) * nu_c
    #                                     + 3 * np.power(a_4, 2) * np.power(nu_c, 2 / 3.)
    #                                     + 6 * a_4 * np.power(nu_c, 1 / 3.)
    #                                     + 6) * np.exp(-a_4 * np.power(nu_c, 1 / 3.))
    return self_abs + thin

if __name__ == '__main__':
    print(sychrotron_self_absorbed_freq(1e7, 1, 3.063466e+14, 1.345503e+05))