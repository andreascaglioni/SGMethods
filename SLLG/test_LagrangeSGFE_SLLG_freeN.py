from __future__ import division
import sys, os
from scipy.io import loadmat
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from multiprocessing import Pool

"""Test lagrange sparse grid interpolant with leja nodes on SLLG. 
number of dimensions is free reference solution is without truncation"""

T = 1
FEMOrder = 1
BDFOrder = 1
Nh = 32
NTau = Nh * 2
NRNDSamples = 128
NParallel = 32
maxNumNodes = 257

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
lev2knots = lambda n: n+1
mat = loadmat('SGMethods/knots_weighted_leja_2.mat')
wLejaArray = np.ndarray.flatten(mat['X'])
knots = lambda n : wLejaArray[0:n:]
beta=0.5
rhoFun = lambda N : 2 * 2**(beta*np.ceil(np.log2(np.linspace(1,N,N)))) # factor 2* is important 
ProfitFun = lambda x : np.sum(np.log(rhoFun(x.size)) * x) 


err = np.array([])
nNodes = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
while(True):
    print("Computing w  = ", w)
    print(I.midSet)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType="polynomial", NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    print(interpolant.numNodes, "nodes")
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
    errCurr = sqrt(np.mean(np.square(errSamples)))
    err = np.append(err, errCurr)
    nNodes = np.append(nNodes, interpolant.numNodes)
    print("Error:", err[w])
    np.savetxt('convergenge_SLLG_freeN.csv',np.array([nNodes, err]))
    oldSG = interpolant.SG
    # update midset for next iteration
    P = np.apply_along_axis(ProfitFun, 1, I.margin)
    idMax = np.argmin(P)
    I.update(idMax)
    w+=1