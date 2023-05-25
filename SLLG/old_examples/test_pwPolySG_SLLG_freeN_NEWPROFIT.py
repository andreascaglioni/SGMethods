from __future__ import division
from signal import Sigmasks
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from multiprocessing import Pool
from math import log, pi, sqrt, factorial, log2
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet, SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant

"""we approximate the SLLG problem with piecewise QUADRATIC sparse grids using
the knapsack-optimal midset (effective dimension grows with number of mids)
"""

T = 1
FEMOrder = 1
BDFOrder = 1
Nh = 32  # 8  # 
NTau = Nh * 2
NRNDSamples = 128  #4  # 
NParallel = 32
maxNumNodes = 550  # 64  # 
p = 3  # degrere + 1
interpolationType = "quadratic"

def F(x):
    return sample_LLG_function_noise(x, Nh, NTau, T, FEMOrder, BDFOrder)
def physicalError(u, uExa):
    return error_sample(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder, False)

# error computations
NLCExpansion = 2**10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)
dimF = uExa[0].size

# choose interpolant
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))
# ProfitFun = lambda x : (p+1) * np.sum(x) + p * np.sum(np.log2(rhoFun(x.size)) , where=(x>0))

Cbar = 2
def ProfitFun(x):
    logRho = beta*np.ceil(np.log2(np.linspace(1,x.size,x.size)))
    C1 = np.sum(x)
    C2 = np.sum(logRho, where=(x==1))
    C3 = np.sum(-log2(Cbar) + p*x + p*logRho, where=(x>1))
    return C1 + C2 + C3

err = np.array([])
nNodes = np.array([])
effectiveDim = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
while(True):
    print("Computing w  = ", w)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
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
    effectiveDim = np.append(effectiveDim, I.N)
    print("Error:", err[w])
    np.savetxt('convergenge_pwQuadratic_SLLG_NEWPROFIT.csv',np.array([nNodes, err, effectiveDim]))
    oldSG = interpolant.SG
    # update midset for next iteration, doubling the number of collocation nodes
    current_num_cps = nNodes[-1]
    while(current_num_cps <= sqrt(2)* nNodes[-1]):
        P = np.apply_along_axis(ProfitFun, 1, I.margin)
        idMax = np.argmin(P)
        I.update(idMax)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        current_num_cps = interpolant.numNodes
    w+=1