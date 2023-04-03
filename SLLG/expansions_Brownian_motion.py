import numpy as np
from math import floor, ceil, log, sqrt, pi, exp
import matplotlib.pyplot as plt
import warnings


def param_LC_Brownian_motion(tt, yy, T):
    if np.any(np.amax(tt) > T) | np.any(np.amin(tt) < 0):
        warnings.warn("Warning...........tt not within [0,T]")
    tt = tt/T  # rescale on [0,1]
    L = ceil(log(len(yy), 2))  # number of levels (even if last does not have all basis functions)
    yy = np.append(yy, np.zeros(2**L-len(yy)))
    W = yy[0] * tt
    for l in range(1, L + 1):
        for j in range(1, 2 ** (l - 1) + 1):
            eta_n_i = 0 * tt
            ran1 = np.where((tt >= (2 * j - 2) / (2 ** l)) & (tt <= (2 * j - 1) / (2 ** l)))
            ran2 = np.where((tt >= (2 * j - 1) / (2 ** l)) & (tt <= (2 * j) / (2 ** l)))
            eta_n_i[ran1] = tt[ran1] - (2 * j - 2) / 2 ** l
            eta_n_i[ran2] = - tt[ran2] + (2 * j) / 2 ** l
            W = W + yy[2 ** (l - 1) + j - 1] * 2 ** ((l - 1) / 2) * eta_n_i
    
    W = W*np.sqrt(T)  # rescale y direction to actually approximate standard Browninan motion on [0,T]
    return W

# and the LG expansion of the INTegral of W i.e. \sum_i y_i \int_0^t eta_i(s) ds where eta_i are the wavelet-hat function basis
def param_LC_Brownian_motion_int(tt, yy, T):
    if np.any(np.amax(tt) > T) | np.any(np.amin(tt) < 0):
        warnings.warn("Warning...........tt not within [0,T]")
    tt = tt/T  # rescale on [0,1]
    N = int(log(len(yy), 2))
    W = yy[0] * 0.5*tt**2
    for n in range(1, N + 1):
        Cn = 2 ** (-(3*n - 1) / 2)
        for i in range(1, 2 ** (n - 1) + 1):
            eta_n_i = 0 * tt
            ran1 = np.where((tt >= (i - 1) / (2 ** (n-1))) & (tt <= (i - 0.5) / (2 ** (n-1))))
            ran2 = np.where((tt >= (i - 0.5) / (2 ** (n-1))) & (tt <= i / (2 ** (n-1))))
            ran3 = np.where(tt >= i / (2 ** (n-1)))
            eta_n_i[ran1] = (2**(n-1) * tt[ran1] - i + 1)**2
            eta_n_i[ran2] = 0.25 + 2*(2**(n-1)*tt[ran2] - i + 1) - 1 - (2**(n-1)*tt[ran2]-i+1)**2 + 0.25
            eta_n_i[ran3] = 0.5
            W = W + yy[2 ** (n - 1) + i - 1] * Cn * eta_n_i
    W = W*np.power(T, 3./2)  # rescale y direction to actually approximate standard Browninan motion on [0,T]
    return W

def param_KL_Brownian_motion(tt, yy):
    W = 0 * tt
    for n in range(len(yy)):
        W = W + sqrt(2) * 2 / (pi * (2 * n - 1)) * np.sin(0.5 * (2 * n - 1) * tt) * yy[n]

    return W

# # test
# N = 10
# Nt = 1000
# tt = np.linspace(0, 1, Nt)
# dd = np.zeros(500)
# for n in range(500):
#     yy = np.random.normal(0, 1, 2**N)
#     vv = param_LC_Brownaina_motion(tt, yy)
#     dd[n] = vv[-1]
# plt.hist(dd,30, density=True)
# xx = np.linspace(-4, 4, 1000)
# plt.plot(xx, 1/sqrt(2*pi)*np.exp(-xx**2/2))
# plt.show()
