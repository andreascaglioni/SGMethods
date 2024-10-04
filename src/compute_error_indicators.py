import numpy as np
import sys, os
from multiprocessing import Pool
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/src'))
from src.sparse_grid_interpolant import SGInterpolant
from src.utils import findMid, lexicSort
from src.tp_inteprolants import TPPwLinearInterpolator


# TODO: Write tests
def compute_GG_indicators_RM(oldRM, old_estimators, Lambda, knots, lev2knots, \
                             F, SG, uOnSG, yyRnd, L2errParam, uInterp, \
                            TP_inteporlant=TPPwLinearInterpolator, NParallel=1):
    """Compute refinement error indicators for adaptive sparse grid 
    interpolation as in [Gerstner, Griebel (2003)].
    Compute them only on the reduced margin of the current multi-index set.
    Their sum (even on the whole maring) is NOT an error estimator!
    Recycle the previously computed values if the margin did not change in a 
    neighbourhood.

    Args:
        oldRM (2D int array): Old reduced margin (each row is a multi-index)
        old_estimators (double array): poitwise estimator on old reduced margin
        I (MidSet): Current multi-index set
        knots (function): Nodes function
        lev2knots (function): level-to-knots function
        F (function): Function to be interpolated
        SG (SGInterpolant): Current interpolation operator instance
        uOnSG (array double): Values F on a previous sparse grid 
        yyRnd (ND array double): Random parameters for estimation norm
        L2errParam (function): Compute L2 error in paramter space
        uInterp (ND array double): values of the function to be itnerpolated on
            current sparse grid
        interpolationType (str, optional): Type of interpolant (see class 
            SGInterpolant). Defaults to "linear".
        NParallel (int, optional): Number of parallel computations. Defaults 1

    Returns:
        array double: Pointwise estimator on reduced margin
    """
    
    # Check input
    assert(oldRM.shape[0] == old_estimators.size)

    cardinality_reduced_margin = Lambda.reducedMargin.shape[0]
    estimator_reduced_margin = np.zeros(cardinality_reduced_margin)
    
    # Recycle from previous list of estimators, if any
    if(oldRM.size==0):
        toCompute = range(Lambda.reducedMargin.shape[0])
    else:
        # Adapt if margin dimension grew
        dimDiff = Lambda.reducedMargin.shape[1] - oldRM.shape[1]
        assert(dimDiff >= 0)  # TODO: haldle general case? Usefull?
        if dimDiff>0:
            oldRM = np.hstack( (oldRM, np.zeros((oldRM.shape[0], dimDiff), \
                                                dtype=int)) )
        
        # Compute later all other mids
        toCompute = []  
        for i in range(cardinality_reduced_margin):
            currMid = Lambda.reducedMargin[i]
            isinOldRM, pos = findMid(currMid, oldRM)  # TODO change
            if(isinOldRM):
                estimator_reduced_margin[i] = old_estimators[pos]
            else:
                toCompute.append(i)
    
    # Compute the rest
    for i in toCompute:
        currMid = Lambda.reducedMargin[i, :]
        IExt = lexicSort( np.vstack((Lambda.midSet, currMid)) )  
        # TODO write and use here lexSortInsert(mid, midSet)
        # TODO Pass only lambda I : interpolant_give_midset(I)
        interpolantExt = SGInterpolant(IExt, knots, lev2knots, \
                                       TPInterpolant=TP_inteporlant, \
                                        NParallel=NParallel, verbose=False)
        uOnSGExt = interpolantExt.sampleOnSG(F, oldXx=SG, oldSamples=uOnSG)
        uInterpExt = interpolantExt.interpolate(yyRnd, uOnSGExt)
        estimator_reduced_margin[i] = L2errParam(uInterpExt, uInterp)  
        # TODO pass only more general norm
    return estimator_reduced_margin