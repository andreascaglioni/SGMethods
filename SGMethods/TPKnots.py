import numpy as np


def TPKnots(ScalarKnots, numKnotsDir):
    '''INPUT
    ScalarKnots (function) ScalarKnots(n) generates n 1D knots
    numKnotsDir (1D array of size N) number of knots in each direction
    OUTPUT
    kk (list of np 1D arrays) knots along each direction
    knotsGrid (np array of shape #TP nodes x N) the TP grid where each row is a node'''

    kk = ()
    for i in range(len(numKnotsDir)):
        kk += (ScalarKnots(numKnotsDir[i]),)
    # knotsGridmat = np.meshgrid(*kk, indexing='ij')
    # knotsGrid = knotsGridmat[0].flatten()
    # for n in range(1, len(kk)):
    #     tmp = knotsGridmat[n].flatten()
    #     knotsGrid = np.vstack((knotsGrid, tmp))
    # knotsGrid = knotsGrid.T
    return kk  # TUPLE, ndarray
