import numpy as np
from SGMethods.MidSets import LLG_optimal_midset_freeN
from SGMethods.SGInterpolant import SGInterpolant
import cProfile

beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))
from SGMethods.ScalarNodes import unboundedKnotsNested

knots = unboundedKnotsNested
lev2knots = lambda nu: 2**(nu+1)-1
NLCExpansion = 2**10
NRNDSamples = 100
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])

def F(x):
    return np.sum(x)*np.array([1,2])
print(F(yyRnd[0,:]))
dimF=2

oldSG = None
uOnSG = None
for w in range(4,6):
    print("********************* w=", w)
    I = LLG_optimal_midset_freeN(w, rhoFun)
    # print(I)
    print("effective dim:", I.shape[1])
    print("initializing interpolant...")

    # cProfile.run('SGInterpolant(I, knots, lev2knots, pieceWise=True, NParallel=1)')

    interpolant = SGInterpolant(I, knots, lev2knots, pieceWise=True, NParallel=1)
    print("sampling...")
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    print("interpolating...")
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    oldSG = interpolant.SG
