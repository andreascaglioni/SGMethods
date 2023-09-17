from __future__ import division
from multiprocessing import Pool
from math import log, pi, sqrt, factorial, log2
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))

from SLLG.profits import ProfitMixL1holo
from SLLG.sample_LLG_function_noise_2 import sample_LLG_function_noise_2
from SLLG.error_sample_pb2_L2norm import error_sample_pb2_L2norm
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant

""" test the L1 holomoprhy based profits """

T = 1
FEMOrder = 1
BDFOrder = 1
Nh = 32  # 8  # 
NTau = Nh * 8
NRNDSamples = 1024  # 4  #   
NParallel = 8
maxNumNodes = 1500  # 64  # 
p = 3  # NBB degrere + 1
interpolationType = "quadratic"

def F(x):
    return sample_LLG_function_noise_2(x, Nh, NTau, T, FEMOrder, BDFOrder)
def physicalError(u, uExa):
    return error_sample_pb2_L2norm(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder)

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
# Profits
Profit = lambda nu :ProfitMixL1holo(nu, p)

err = np.array([])
nNodes = np.array([])
nDims = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
while(True):
    print("Computing w  = ", w)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    midSetMaxDim = np.amax(I.midSet, axis=0)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N, "\nMax midcomponents:", midSetMaxDim)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # compute error
    errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
    errCurr = sqrt(np.mean(np.square(errSamples)))
    err = np.append(err, errCurr)
    print("Error:", err[w])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    
    np.savetxt('convergenge_SG_ProfitL1Mix.csv',np.transpose(np.array([nNodes, err, nDims])), delimiter=',')
    
    oldSG = interpolant.SG

    current_num_cps = nNodes[-1]
    while(current_num_cps <= sqrt(2)* nNodes[-1]):
        P = Profit(I.margin)
        idMax = np.argmax(P)
        I.update(idMax)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        current_num_cps = interpolant.numNodes
        if(current_num_cps > maxNumNodes):
            break
    w+=1

print("# nodes:", nNodes)
print("# dimen:", nDims)
print("error:", err)
print("rates:", -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:]))
plt.plot(nNodes, err, '.-', nNodes, np.power(nNodes, -0.5), '.k-', nNodes, np.power(nNodes, -0.25), 'k-', )
plt.show()
plt.plot(nNodes, nDims, '.-', nNodes, nNodes, 'k-')
plt.show()