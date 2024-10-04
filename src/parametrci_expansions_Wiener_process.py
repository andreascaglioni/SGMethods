import warnings
import numpy as np
from math import floor, ceil, log, sqrt, pi, exp

""" A Collection of parametric expanisons of the Wiener process and related
    random fields."""


def param_LC_Brownian_motion(tt, yy, T):
    """The classical Levy-Cielsielsky constuction of the WIener process used 
        here as a parametric expansion.

    Args:
        tt (1D array float): Discret times of evaluation in [0,T]
        yy (2D array float): The scalar parameters of the expansion. 
            Each row consts of the scalar components (each in R) of a parameteir
            vector.
        T (float): final time of approximation

    Returns:
        2D array flaat: each row gives the samples on tt for a single parameter 
        vector in yy
    """

    # Check input
    if np.any(np.amax(tt) > T) | np.any(np.amin(tt) < 0):
        warnings.warn("Warning...........tt not within [0,T]")
    
    tt = tt/T  # rescale on [0,1]

    # number of levels (nb last level may not have all basis functions!)
    L = ceil(log(len(yy), 2))  
    yy = np.append(yy, np.zeros(2**L-len(yy)))  # zero padding to fill level L

    W = yy[0] * tt
    for l in range(1, L + 1):
        for j in range(1, 2 ** (l - 1) + 1):
            eta_n_i = 0 * tt
            # define part of currect basis function corepsonding to (0, 1/2)
            ran1 = np.where(\
                (tt >= (2 * j - 2) / (2 ** l)) & (tt <= (2 * j - 1) / (2 ** l))\
                    )
            eta_n_i[ran1] = tt[ran1] - (2 * j - 2) / 2 ** l
            # define part of currect basis function corepsonding to (0, 1/2, 1)
            ran2 = np.where(\
                (tt >= (2 * j - 1) / (2 ** l)) & (tt <= (2 * j) / (2 ** l))\
                    )
            eta_n_i[ran2] = - tt[ran2] + (2 * j) / 2 ** l
            W = W + yy[2 ** (l - 1) + j - 1] * 2 ** ((l - 1) / 2) * eta_n_i
    W = W*np.sqrt(T)  # NB Revert scaling above to go to times in [0,T] 
    return W