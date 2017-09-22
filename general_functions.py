import scipy.integrate as integrate
import numpy as np
import fundamental_constants as fc


def _gamma_integrand(x, a):
    return np.power(x, a - 1) * np.power(np.e, -x)


def gamma_function(a, x_0):
    result = integrate.quad(_gamma_integrand, x_0, np.inf, args=(a,))
    value = result[0]
    abserr = result[1]
    return value


def incomplete_gamma_function(a, x_0):
    result = integrate.quad(_gamma_integrand, 0, x_0, args=(a,))
    value = result[0] # / gamma_function(a, 0.)
    abserr = result[1]
    return value


if __name__ == '__main__':
    print(incomplete_gamma_function(7/2, 5))
