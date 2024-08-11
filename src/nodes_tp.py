import numpy as np


def TPKnots(ScalarKnots, numKnotsDir):
    """Generate tensor product nodes

    Args:
        ScalarKnots (function): ScalarKnots(n) generates set pf n 1D nodes 
        numKnotsDir (array int): Number of nodes in each direction

    Returns:
        tuple of array double: Nodes along each direction
    """

    kk = ()
    for i in range(len(numKnotsDir)):
        kk += (ScalarKnots(numKnotsDir[i]),)
    
    # TODO: remove
    # knotsGridmat = np.meshgrid(*kk, indexing='ij')
    # knotsGrid = knotsGridmat[0].flatten()
    # for n in range(1, len(kk)):
    #     tmp = knotsGridmat[n].flatten()
    #     knotsGrid = np.vstack((knotsGrid, tmp))
    # knotsGrid = knotsGrid.T

    return kk
