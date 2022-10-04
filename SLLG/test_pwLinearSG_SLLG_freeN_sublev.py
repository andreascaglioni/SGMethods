from __future__ import division
from signal import Sigmasks
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from multiprocessing import Pool
from math import log, pi
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import LLG_optimal_midset_freeN, SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant

"""we approximate the SLLG problem with piecewise linear sparse grids using two different sequences of midsets
- the knapsack-optimal midset (effectivedimension grows with number of mids)
- the simplex with fixed effective dimensions N=1,2,4,8,16

midset for dimension adaptive version is determined as sublevel set
"""

FEMOrder = 1
BDFOrder = 1
T = 1
Nh = 16
NTau = Nh * 2
NRNDSamples = 2
NParallel = 2
maxNumNodes = 500

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

################################################ dimension-adapted SG
print("########################### Dimension-adapted SG")
# choose interpolant
lev2knots = lambda nu: 2**(nu+1)-1
knots = unboundedKnotsNested
beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))

err = np.array([])
nNodes = np.array([])
w=0
while(True):
    print("Computing w  = ", w)
    #midset
    I = LLG_optimal_midset_freeN(w, rhoFun)
    interpolant = SGInterpolant(I, knots, lev2knots, pieceWise=True, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    print("Number effective dimensions:", I.shape[1])
    print("Number of nodes:", interpolant.numNodes)
    if w ==0 :
        oldSG = None
        uOnSG = None
    # sample and interpolate
    print("sampling...")
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    print("interpolating...")
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # compute error
    print("compting error...")
    errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
    errCurr = np.mean(errSamples) # np.amax(np.multiply(errSamples,  ww))

    err = np.append(err, errCurr)
    nNodes = np.append(nNodes, interpolant.numNodes)
    print("Error:", err[w])
    np.savetxt('convergenge_pwLin_aniso_SLLG_freeN_sublev.csv',np.array([nNodes, err]))
    oldSG = interpolant.SG

    w+=1

################################################ Smolyak SG with N fixed
lev2knots = lambda nu: 2**(nu+1)-1
knots = unboundedKnotsNested
NN = [1,2,4,8,16]
for N in NN:
    print("########################### Smolyak with fixed-dimension N =", N)
    err = []
    nNodes = []
    w=0
    while(True):
        print("Computing n = ", w)
        I = SmolyakMidSet(w, N)
        interpolant = SGInterpolant(I, knots, lev2knots, pieceWise=True)
        print(interpolant.numNodes, "nodes")
        if(interpolant.numNodes > maxNumNodes):
            break
        if w ==0 :
            oldSG = None
            uOnSG = None
        uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
        uInterp = interpolant.interpolate(yyRnd, uOnSG)
        errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
        print(errSamples)
        errCurr = np.mean(errSamples) # np.amax(np.multiply(errSamples,  ww))
        print("Error:", errCurr)
        err = np.append(err, errCurr)
        nNodes = np.append(nNodes, interpolant.numNodes)
        oldSG = interpolant.SG
        w+=1
    np.savetxt('convergenge_pwLin_smolyak_N_'+str(N)+'_SLLG.csv',np.array([nNodes, err]))