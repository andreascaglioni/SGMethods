import numpy as np
from math import sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.SGInterpolant import SGInterpolant
from utils.utils import coordUnitVector, lexicSort

def compute_aposteriori_estimator_reduced_margin(I, knots, lev2knots, F, dimF, SG, uOnSG, yyRnd, physicalNorm, NRNDSamples, uInterp, interpolationType = "linear", NParallel=1):
    """for each element in *reduced* margin, compute 
    # norm{\Delta_nu u} 
    # where we use 
    # \Delta_nu u =  I_{\Lambda \cup \nu}[u] - I_{\Lambda}[u]"""
    dimReducedMargin = I.reducedMargin.shape[0]
    estimator_reduced_margin = np.zeros(dimReducedMargin)
    for i in range(dimReducedMargin):
        currMid = I.reducedMargin[i, :]
        IExt = lexicSort( np.vstack((I.midSet, currMid)) )
        interpolantExt = SGInterpolant(IExt, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        uOnSGExt = interpolantExt.sampleOnSG(F, dimF, SG, uOnSG)
        uInterpExt = interpolantExt.interpolate(yyRnd, uOnSGExt)
        errSamples = np.array([physicalNorm(uInterpExt[n] - uInterp[n]) for n in range(NRNDSamples)])
        estimator_reduced_margin[i] = sqrt(np.mean(np.square(errSamples)))
    return estimator_reduced_margin