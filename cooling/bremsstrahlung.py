import fundamental_constants as fc
import numpy as np


def lepton_ion(T, n_prot, n_lep):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    if theta < 1.:
        F_ei = 4 * np.sqrt(2 * theta / np.power(np.pi, 3)) * (1 + 1.781 * np.power(theta, 1.34))
    else:
        F_ei = 9 * theta / (2 * np.pi) * (np.log(1.123 * theta + 0.48) + 1.5)
    return 1.48e-22 * n_prot * n_lep * F_ei


def lepton_lepton_same_charge(T, n_el, n_pos):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    if theta < 1.:
        return 2.56e-22 * (np.power(n_el, 2) + np.power(n_pos, 2)) * np.power(theta, 1.5) * (1 + 1.1 * theta + np.power(theta, 2) - 1.25 * np.power(theta, 2.5))
    else:
        return 3.42e-22 * (np.power(n_el, 2) + np.power(n_pos, 2)) * theta * (np.log(1.123 * theta) + 1.28)


def lepton_lepton_opposite_charge(T, n_el, n_pos):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    if theta < 1.:
        return 3.43e-22 * n_el * n_pos * (np.power(theta, 0.5) + 1.7 * np.power(theta, 2))
    else:
        return 6.84e-22 * n_el * n_pos * theta * (np.log(1.123 * theta) + 1.24)


def cooling(T, n_prot, n_lep, n_el, n_pos):
    return lepton_ion(T, n_prot, n_lep) + lepton_lepton_same_charge(T, n_el, n_pos) + lepton_lepton_opposite_charge(T, n_el, n_pos)

if __name__ == '__main__':
    pass
