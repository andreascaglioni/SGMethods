from __future__ import division
from multiprocessing import Pool
from math import log, pi, sqrt, factorial, log2
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))

# from SLLG.sample_LLG_function_noise_2_time_1 import sample_LLG_function_noise_2_time_1
# from SLLG.error_sample_pb2_time_1 import error_sample_pb2_time_1
from SLLG.sample_LLG_function_noise_2 import sample_LLG_function_noise_2
from SLLG.error_sample_pb2_L2norm import error_sample_pb2_L2norm
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant

"""Approximate the SLLG problem 2  (square space domain) with piecewise polynomial
sparse grids using the knapsack-optimal midset (effective dimension grows with
 number of mids)
"""

T = 1
FEMOrder = 1
BDFOrder = 1
Nh = 32  # 8  # 
NTau = Nh * 8
NRNDSamples = 1024  # 4  #   
NParallel = 32
maxNumNodes = 1500  # 64  # 
p = 2  # NBB degrere + 1
interpolationType = "linear"

# def F(x):
#     return sample_LLG_function_noise_2_time_1(x, Nh, NTau, T, FEMOrder, BDFOrder)
# def physicalError(u, uExa):
#     return error_sample_pb2_time_1(u, uExa, Nh, Nh, FEMOrder)
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
def Profit(nu):
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims)))) 
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)
    repSizeSupp = repRho
    C1 = 1
    C2 = 1
    v1 = np.prod(C1* np.power(repRho*repSizeSupp, -1), axis=1, where=(nu==1))
    v2 = np.prod(C2*np.power(2**nu * repRho*repSizeSupp, -p), axis=1, where=(nu>1))
    w = np.prod((2**(nu+1)-2)*(p-1)+1, axis=1)
    return v1 * v2 / w

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
    
    np.savetxt('convergenge_pwLin_pb2_smoothingProfitL2Long.csv',np.transpose(np.array([nNodes, err, nDims])), delimiter=',')
    
    oldSG = interpolant.SG

    # P = Profit(I.margin)
    # idMax = np.argmax(P)
    # I.update(idMax)
    # update midset for next iteration, doubling the number of collocation nodes
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