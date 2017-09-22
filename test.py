import numpy as np
import fundamental_constants as fc
import pair_eq
from cooling import synchrotron, bremsstrahlung, compton


def calc_chi(cooling, H, tau, T, rho):
    p_rad = cooling * H / (2 * fc.c) * (tau + 2 / np.sqrt(3))
    p_gas = rho * fc.k * T * (1 / (fc.amu * 1.23) + 1 / (1.14 * fc.amu))
    return p_gas / (p_rad + p_gas)

# input parameters
m = 10.
mdot = 1.e-5
r = 10.
alpha = 0.1
H = 10.
beta_m = 0.5
beta = 0.5
T = 1.e10

chi_curr = 0.1
chi_prev = 0.4
while abs(chi_curr - chi_prev) > 1e-6:
    beta = chi_prev * beta_m

    # helper variables
    c_3 = 3.12e-13 * T * r / beta
    c_1 = 1.5 * c_3
    x = 9 * c_3 * np.power(alpha, 2)
    epsilon = 0.5 * (((18 * np.power(alpha, 2) - np.power(x, 2)) / (2 * x)) - 5)

    # primitive variables
    v = -2.12e10 * alpha * c_1 / np.sqrt(r)
    c_s_square = 4.5e20 * c_3 / r
    rho = 3.79e-5 * mdot / (alpha * c_1 * m * np.sqrt(c_3 * np.power(r, 3)))
    B = 6.55e8 * np.sqrt((1 - beta_m) * mdot / (alpha * c_1 * m) * np.sqrt(c_3 / np.power(r, 5)))
    gamma = (32 - 24 * beta - 3 * np.power(beta, 2)) / (24 - 21 * beta)
    f = ((5 / 3 - gamma) / (gamma - 1)) / epsilon
    q_plus = 1.84e21 * epsilon * mdot * np.sqrt(c_3) / (np.power(m, 2) * np.power(r, 4))

    # number densities
    n_prot = rho / fc.m_H
    z = pair_eq.dens(T)
    n_el = n_prot * (1 + z)
    n_lep = n_prot * (1 + 2 * z)
    n_pos = n_lep - n_el

    print('%e, %e, %e, %f, %f, %f' % (T, epsilon, beta, c_1, c_3, x))

    nu_c = synchrotron.sychrotron_self_absorbed_freq(T, H, n_lep, B)
    brem = bremsstrahlung.cooling(T, n_prot, n_lep, n_el, n_pos)
    sync = synchrotron.cooling(T, H, nu_c, n_lep, B)

    sync_comp = compton.synchrotron_enhancement(nu_c, T, n_lep, H)
    brem_comp = compton.bremsstrahlung_enhancement(T, rho, n_lep, H)

    tau_es = 2 * n_lep * fc.thomson * H
    tau_abs = (H / (4 * fc.sb * np.power(T, 4))) * (brem + sync)
    tau = tau_es + tau_abs

    # print(sync_comp, brem_comp)
    brem *= brem_comp
    sync *= sync_comp
    # total = (4 * fc.sb * np.power(T, 4) / H) / (1.5 * tau + np.sqrt(3) + (4 * fc.sb * np.power(T, 4) / H) * np.power(brem + sync, -1))
    total = brem + sync
    # print()
    # print('%10e, %10e' % (brem, sync))
    chi_prev = chi_curr
    chi_curr = calc_chi(total, H, tau, T, rho)
    # print('%e, %.2f, %.2f, %e, %e, %e' % (T, chi_prev, chi_curr, tau, tau_es, tau_abs))