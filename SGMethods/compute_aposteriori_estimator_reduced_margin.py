import numpy as np
from math import sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.SGInterpolant import SGInterpolant
from utils.utils import coordUnitVector, lexicSort
from multiprocessing import Pool


def findMid(mid, midSet):
    if(midSet.size==0):
        return False, -1
    assert(midSet.shape[1] == mid.size)
    boolVec = np.all(midSet==mid, axis=1)
    pos = np.where(boolVec)[0]
    assert(pos.size<2)
    if len(pos)==0:
        return False, -1
    else:
        return True, pos[0]

def compute_aposteriori_estimator_reduced_margin(oldRM, old_estimators, I, knots, lev2knots, F, SG, uOnSG, yyRnd, L2errParam, uInterp, interpolationType="linear", NParallel=1):
    """for each element in *reduced* margin, compute 
    # norm{\Delta_nu u} 
    # where we use 
    # \Delta_nu u =  I_{\Lambda \cup \nu}[u] - I_{\Lambda}[u]
    """
    

    dimReducedMargin = I.reducedMargin.shape[0]
    estimator_reduced_margin = np.zeros(dimReducedMargin)

    # first recycle from previous list of estimators, if any
    if(oldRM.size==0):
        toCompute = range(I.reducedMargin.shape[0])
    else:
        dimDiff = I.reducedMargin.shape[1] - oldRM.shape[1]
        assert(dimDiff>=0)
        if dimDiff>0:
            oldRM = np.hstack( (oldRM, np.zeros((oldRM.shape[0], dimDiff), dtype=int)) )
        toCompute = []  # if I dont find a mid in oldRM, list it here to compute it later
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
        interpolantExt = SGInterpolant(IExt, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel, verbose=False)
        uOnSGExt = interpolantExt.sampleOnSG(F, oldXx=SG, oldSamples=uOnSG)
        uInterpExt = interpolantExt.interpolate(yyRnd, uOnSGExt)
        estimator_reduced_margin[i] = L2errParam(uInterpExt, uInterp)
    return estimator_reduced_margin