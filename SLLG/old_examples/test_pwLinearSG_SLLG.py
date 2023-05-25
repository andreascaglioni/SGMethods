from __future__ import division
from signal import Sigmasks
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from multiprocessing import Pool
from math import log, pi, sqrt
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet, SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant

"""we approximate the SLLG problem with piecewise linear sparse grids using two different sequences of midsets
- the knapsack-optimal midset (effectivedimension grows with number of mids)
- the simplex with fixed effective dimensions N=1,2,4,8,16,32,64
"""

FEMOrder = 1
BDFOrder = 1
T = 1
Nh = 32
NTau = Nh * 2
NRNDSamples = 32
NParallel = 32
maxNumNodes = 200

def F(x):
    return sample_LLG_function_noise(x, Nh, NTau, T, FEMOrder, BDFOrder)
def physicalError(u, uExa):
    return error_sample(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder, False)

# error computations
NLCExpansion = 2**10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
# ww = (2*pi)**(-0.5*NLCExpansion) * np.exp(-0.5*np.square(np.linalg.norm(yyRnd, 2, axis=1)))
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)
dimF = uExa[0].size

# choose interpolant
lev2knots = lambda nu: 2**(nu+1)-1
knots = unboundedKnotsNested
beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))
ProfitFun = lambda x : 2*np.sum(x) + np.sum(np.log2(rhoFun(x.size)) , where=(x>0))
maxNV = [1,2,4,8,16, 32, 64]
for maxN in maxNV:
    err = np.array([])
    nNodes = np.array([])
    w=0
    I = midSet(maxN)
    oldSG = None
    uOnSG = None
    while(True):
        print("Computing w  = ", w)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, pieceWise=True, NParallel=NParallel)
        if(interpolant.numNodes > maxNumNodes):
            break
        print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
        uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
        uInterp = interpolant.interpolate(yyRnd, uOnSG)
        # compute error
        errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
        errCurr = sqrt(np.mean(np.square(errSamples)))
        err = np.append(err, errCurr)
        nNodes = np.append(nNodes, interpolant.numNodes)
        print("Error:", err[w])
        np.savetxt('convergenge_pwLin_aniso_SLLG_'+str(maxN)+'.csv',np.array([nNodes, err]))
        oldSG = interpolant.SG
        # update midset for next iteration
        P = np.apply_along_axis(ProfitFun, 1, I.margin)
        idMax = np.argmin(P)
        I.update(idMax)
        print(I.midSet)
        w+=1