import numpy as np
from math import sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from utils.utils import coordUnitVector, lexicSort
"""An example using the [Gerstner-Griebel] adaptivity"""


def compute_estimator_reduced_margin(I, knots, lev2knots, F, dimF, SG, uOnSG, yyRnd, physicalNorm, interpolationType = "liner"):
    """for each element in reduced margin, compute 
    # norm{\Delta_nu u} 
    # where we compute 
    # \Delta_nu u =  I_{\Lambda \cup \nu}[u] - I_{\Lambda}[u]"""
    estimator_reduced_margin = np.array([], dtype=float)
    for i in range(len(I.ids_reduced_margin)):
        currMid = I.margin[I.ids_reduced_margin[i], :]
        IExt = lexicSort( np.vstack((I.midSet, currMid)) )
        interpolantExt = SGInterpolant(IExt, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        uOnSGExt = interpolant.sampleOnSG(F, dimF, SG, uOnSG)
        uInterpExt = interpolantExt.interpolate(yyRnd, uOnSGExt)
        errSamples = np.array([physicalNorm(uInterpExt[n] - uInterp[n]) for n in range(NRNDSamples)])
        estimator_reduced_margin[i] = sqrt(np.mean(np.square(errSamples)))
    return estimator_reduced_margin

N = 2
dimF = 3
def F(x):
    return np.sin(np.sum(x))+1 * np.array([1., 2., 3.])
def physicalNorm(x):
    return np.linalg.norm(x, 2)

# interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = unboundedKnotsNested
NParallel = 1

# adaptive loop
NLCExpansion = 2**10
NRNDSamples = 100
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
maxNumNodes = 50
I = midSet(maxN=2)
oldSG = None
uOnSG = None
while True:
    # COMPUTE
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType="linear", NParallel=NParallel)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # ESTIMATE: 
    estimator_reduced_margin = compute_estimator_reduced_margin(I, knots, lev2knots, F, dimF, interpolant.SG, uOnSG, yyRnd, physicalNorm)
    # MARK: maximum
    idMax = np.argmax(estimator_reduced_margin)
    #REFINE
    I.update(idMax)
    if(interpolant.numNodes > maxNumNodes):
        break
    oldSG = interpolant.SG