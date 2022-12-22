import numpy as np
from math import sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant

"""An example using the [Gerstner-Griebel] adaptivity"""

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
    oldSG = interpolant.SG
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)

    # ESTIMATE: for each element in reduced margin, compute 
    # norm{\Delta_nu u} where we compute 
    # \Delta_nu u =  I_{\Lambda \cup \nu}[u] - I_{\Lambda}[u]
    
    estimator_reduced_margin = np.array([], dtype=float)
    # check only mids in margin
    for i in range(I.dimMargin):
        currMid = I.margin[i,:]
        # check if mid is in reduced margin
        for n in range(I.dimMargin):
            en = np.zeros(I.dimMargin).astype(int)
            en[n] = 1
            if(not(currMid[n] == 0 or ((currMid-en).tolist() in I.midSet.tolist()))):
                assert(currMid[n] == 0 or ((currMid-en).tolist() in self.midSet.tolist()))
        
        
        IExtend = np.vstack((I.midSet, currMid))
        IExtend = IExtend[np.lexsort(np.rot90(IExtend))]
        interpolantExt = SGInterpolant(IExtend, knots, lev2knots, interpolationType="linear", NParallel=NParallel)
        uOnSGExt = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
        uInterpExt = interpolantExt.interpolate(yyRnd, uOnSGExt)
        
        errSamples = np.array([physicalNorm(uInterpExt[n] - uInterp[n]) for n in range(NRNDSamples)])
        # interp_I_ext_samples - interp_I_samples
        estimator_margin[i] = sqrt(np.mean(np.square(errSamples)))
    
    # MARK: maximum
    idMax = np.argmax(estimator_margin)

    #REFINE
    I.update(idMax)
    if(interpolant.numNodes > maxNumNodes):
        break