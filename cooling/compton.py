import fundamental_constants as fc
import general_functions as gf
import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate


def enhancement(nu, T, n_lep, H):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    A = 1 + 4 * theta + 16 * np.power(theta, 2)
    eta_max = 3 * fc.k * T / (fc.h * nu)
    j_m = np.log(eta_max) / np.log(A)
    tau_es = 2 * n_lep * fc.thomson * H
    s = tau_es + np.power(tau_es, 2)
    eta = np.exp(s * (A - 1)) * (1 - gf.incomplete_gamma_function(j_m + 1, A * s)) + eta_max * gf.incomplete_gamma_function(j_m + 1, s)
    return eta


def synchrotron_enhancement(nu_c, T, n_lep, H):
    return enhancement(nu_c, T, n_lep, H)


def _free_free_opacity(nu, T, rho):
    Z = 1.4
    mu_e = 1.23
    mu_i = 1.14
    return 1.34e56 * (np.power(Z, 2) / (mu_e * mu_i)) * rho * np.power(T, -1.2) * np.power(nu, -3) * (1 - np.exp(-fc.h * nu / (fc.k * T)))


def _to_solve(nu, T, rho):
    return _free_free_opacity(nu, T, rho) - 0.4


def _solve_brehms_cutoff(T, rho):
    return fsolve(_to_solve, 1e6, args=(T, rho))[0]


def bremsstrahlung_enhancement(T, rho, n_lep, H):
    nu_br = _solve_brehms_cutoff(T, rho)
    integrated_enhancement = integrate.quad(enhancement, nu_br, fc.k * T / fc.h, args=(T, n_lep, H))[0]
    return integrated_enhancement / (fc.k * T / fc.h - nu_br)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    freqs = np.logspace(-10, 10, 1000)
    print(_solve_brehms_cutoff(1e11, 1e-11))

