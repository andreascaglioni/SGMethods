import numpy as np


def TPKnots(ScalarKnots, numKnotsDir):
    """Generate tensor product nodes

    Args:
        ScalarKnots (function): ScalarKnots(n) generates set pf n 1D nodes
        numKnotsDir (array): Number of nodes in each direction

    Returns:
        tuple of np.array : Nodes along each direction
    """

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
