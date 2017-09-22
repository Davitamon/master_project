import fundamental_constants as fc
import numpy as np


def pair_annihilation(T, n_el, n_pos):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    r_e = np.power(fc.e, 2) / (fc.m_e * np.power(fc.c, 2))
    g = np.power(1 + 2 * np.power(theta, 2) / np.log(1.12 * theta + 1.3), -1)
    return np.pi * fc.c * np.power(r_e, 2) * n_el * n_pos * g


def e_e_pair_production(T, n_el):
    theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
    r_e = np.power(fc.e, 2) / (fc.m_e * np.power(fc.c, 2))
    if theta < 1.:
        return fc.c * np.power(r_e, 2) * np.power(n_el, 2) * 2e-4 * np.power(theta, 1.5) * np.exp(-2. / theta) * (1 + 0.015 * theta)
    else:
        return fc.c * np.power(r_e, 2) * np.power(n_el, 2) * (112 / (27 * np.pi)) * np.power(fc.fine_structure, 2) * np.power(np.log(theta), 3) * np.power(1 + 0.058 / theta, -1)


def dens(T):
    if isinstance(T, float) or isinstance(T, int):
        theta = fc.k * T / (fc.m_e * np.power(fc.c, 2))
        g = np.power(1 + 2 * np.power(theta, 2) / np.log(1.12 * theta + 1.3), -1)
        if theta < 1.:
            el_pos_dens_ratio = (2e-4 * np.power(theta, 1.5) * np.exp(-2. / theta) * (1 + 0.015 * theta)) / (np.pi * g)
        else:
            el_pos_dens_ratio = ((112 / (27 * np.pi)) * np.power(fc.fine_structure, 2) * np.power(np.log(theta), 3) * np.power(1 + 0.058 / theta, -1)) / (np.pi * g)
        return el_pos_dens_ratio / (1 - el_pos_dens_ratio)
    elif isinstance(T, list) or isinstance(T, tuple) or isinstance(T, np.ndarray):
        z_list = []
        for temp in T:
            theta = fc.k * temp / (fc.m_e * np.power(fc.c, 2))
            g = np.power(1 + 2 * np.power(theta, 2) / np.log(1.12 * theta + 1.3), -1)
            if theta < 1.:
                el_pos_dens_ratio = (2e-4 * np.power(theta, 1.5) * np.exp(-2. / theta) * (1 + 0.015 * theta)) / (np.pi * g)
            else:
                el_pos_dens_ratio = ((112 / (27 * np.pi)) * np.power(fc.fine_structure, 2) * np.power(np.log(theta), 3) * np.power(1 + 0.058 / theta, -1)) / (np.pi * g)
            z_list.append(el_pos_dens_ratio / (1 - el_pos_dens_ratio))
        return z_list


if __name__ == '__main__':
    T = np.logspace(9, 11, 100)
    print(type(T))
    z = dens(T)
    import matplotlib.pyplot as plt
    plt.plot(T, z)
    plt.xscale('log')
    plt.show()