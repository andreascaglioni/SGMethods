""" 
This module provides functions for parametric expansions of the Wiener process.

Functions:
    param_LC_Brownian_motion(tt, yy, T):
        Constructs the Wiener process using the classical Levy-Ciesielski 
        construction.
    param_KL_Brownian_motion(tt, yy):
        Constructs the Wiener process using the Karhunen-Loeve expansion.
"""

from math import ceil, log, sqrt, pi
import warnings
import numpy as np


def param_LC_Brownian_motion(tt, yy, T):
    """The classical Lèvy-Cielsielsky constuction of the Wiener process is used 
    as a parametric expansion as follows:

    .. math::

        W(y, t) = y_0 \eta_{0,1}(t) + 
        \sum_{l=1}^L \sum_{j=1}^{2^{l-1}} y_{l,j} \eta_{l,j}(t),
    
    where :math:`\eta_{l,j}` is the Faber-Schauder basis on :math:`[0,T]` (i.e.
    a wavelet basis of hat functions) and :math:`y = (y_{l,j})_{l,j}` is a 
    sequence of real numbers that replace i.i.d. standard Gaussians.

    Args:
        tt (numpy.ndarray[float]): Discret times of evaluation in :math:`[0,T]`.
        yy (numpy.ndarray[float]): The parameter vectors of the expansion. 
            Each row consts of the scalar components (each in R) of a parameteir
            vector.
        T (float): Final (positive) time of approximation.

    Returns:
        numpy.ndarray[float]: Each row gives the samples on tt for the 
        corresponding row (parameter vector) in ``yy``.
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

    W = W*np.sqrt(T)  # revert scaling above to go to times in [0,T]

    return W

def param_KL_Brownian_motion(tt, yy):
    """The Karhunen_Loeve expansion of th Brownian motion. Can be computed 
    exactly.

    Args:
        tt (numpy.ndarray[float]): Discrete times of evaluation in [0,1].
        yy (numpy.ndarray[float]): Each row is a parameter vector of the 
            expansion. Each component is a real numbers that replace i.i.d.
            standard Gaussians.

    Returns:
        numpy.ndarray[float]: Each row gives the samples of the Wiener process 
        on ``tt`` for the corresponding row (parameter vector) in ``yy``.
    """

    W = 0 * tt
    for n in range(len(yy)):
        W = W + pi*(n+0.5)*sqrt(2)*np.sin((n+0.5)*pi*tt)*yy[n]
    return W
