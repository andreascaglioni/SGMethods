import numpy as np
import sys, os
from multiprocessing import Pool
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/src'))
from src.sparse_grid_interpolant import SGInterpolant
from src.utils import findMid, lexicSort


def compute_a_post_estim_RM(oldRM, old_estimators, I, knots, lev2knots, F, SG, \
                            uOnSG, yyRnd, L2errParam, uInterp, \
                                interpolationType="linear", NParallel=1):
    """Compute pointwise a-posteriori error estimator on the reduced margin as 
    in [Guignard, Noible (2018)]. Recycles values computed before if possible.

    Args:
        oldRM (2D array int): Old reduced margin (each row is a multi-index)
        old_estimators (array): poitwise estimator on old reduced margin
        I (MidSet): Current multi-index set
        knots (function): Nodes function
        lev2knots (function): level-to-know function
        F (function): Function to be interpolated
        SG (SGInterpolant): Current interpolation operator instance
        uOnSG (array double): Values F on a previous sparse grid 
        yyRnd (ND array double): Random parameters for estimation norm
        L2errParam (function): Compute L2 error in paramter space
        uInterp (array): values of the function to be itnerpolated on current 
            sparse grid
        interpolationType (str, optional): Type of interpolant (see class 
            SGInterpolant). Defaults to "linear".
        NParallel (int, optional): Number of parallel computations. Defaults 1

    Returns:
        array double: Pointwise estimator on reduced margin
    """
    assert(oldRM.shape[0] == old_estimators.size)
    dimReducedMargin = I.reducedMargin.shape[0]
    estimator_reduced_margin = np.zeros(dimReducedMargin)
    # first recycle from previous list of estimators, if any
    if(oldRM.size==0):
        toCompute = range(I.reducedMargin.shape[0])
    else:
        # NB did the margin dimensionality grow?
        dimDiff = I.reducedMargin.shape[1] - oldRM.shape[1]
        assert(dimDiff>=0)
        if dimDiff>0: # if yes, 
            oldRM = np.hstack( (oldRM, np.zeros((oldRM.shape[0], dimDiff), \
                                                dtype=int)) )
        toCompute = []  # if I don't find a mid in oldRM, compute it later
        for i in range(dimReducedMargin):
            currMid = I.reducedMargin[i]
            isinOldRM, pos = findMid(currMid, oldRM)
            if(isinOldRM):
                estimator_reduced_margin[i] = old_estimators[pos]
            else:
                toCompute.append(i)
    
    # for the estimators corresponding to new mids, compute explicitly
    for i in toCompute:
        currMid = I.reducedMargin[i, :]
        IExt = lexicSort( np.vstack((I.midSet, currMid)) )  
        # TODO write lexSortInsert(mid, midSet)
        interpolantExt = SGInterpolant(IExt, knots, lev2knots, \
                                       interpolationType=interpolationType, \
                                        NParallel=NParallel, verbose=False)
        uOnSGExt = interpolantExt.sampleOnSG(F, oldXx=SG, oldSamples=uOnSG)
        uInterpExt = interpolantExt.interpolate(yyRnd, uOnSGExt)
        estimator_reduced_margin[i] = L2errParam(uInterpExt, uInterp)
    return estimator_reduced_margin