from __future__ import division
import numpy as np
from math import sqrt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.TPPwCubicInterpolator import TPPwCubicInterpolator
from SGMethods.ScalarNodes import unboundedKnotsNested


def computeL2Error(uExa, Iu):  # NBB if rnd points normally distributed, need no gaussian weight!
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : np.linalg.norm(x, ord=2, axis=1)
    errSample = np.zeros(uExa.shape[0])
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))

# choose function
N = 2
dimF = 3
anisoVec = np.array([1,10.])
def F(x):
    return np.sin(np.sum(x/anisoVec))* np.array([1., 2., 3.])

# error computations
NRNDSamples = int(1.e2)
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
fOnxx = np.zeros((NRNDSamples, dimF))
for w in range(NRNDSamples):
    fOnxx[w] = F(xxRND[w, :])

# choose interpolant
def lev2knots(nu):  # NBB avoid case #nodes = 2^{1+1}-1= 3 (not enough for cubic)
    if(nu >= 0):
        return 2**(nu+2)-1
    else:
        return 1
knots = lambda m : unboundedKnotsNested(m, p=4)
maxNumNodes = 100000

# convergence test
err = []
nNodes = []
oldSG = None
FOnSG = None
w=2
while True:
    kk = knots(lev2knots(w))
    print(kk.size)
    nodesTuple = (kk, kk)
    FVals = np.zeros((kk.size, kk.size, dimF))
    for i  in range(kk.size):
        for j in range(kk.size):
            xCurr = np.array([kk[i], kk[j]])
            FVals[i, j] = F(xCurr)

    interpolant = TPPwCubicInterpolator(nodesTuple, FVals)
    numNodes = (kk.size)**N
    if(numNodes > maxNumNodes):
        break
    nNodes.append(numNodes)
    fInterp = interpolant(xxRND)
    err.append(computeL2Error(fInterp, fOnxx))
    w+=1


print(err)
err = np.array(err)
nNodes = np.array(nNodes)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)