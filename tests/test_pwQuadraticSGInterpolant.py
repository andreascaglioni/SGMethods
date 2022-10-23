from __future__ import division
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from SGMethods.MidSets import anisoSmolyakMidSet, SmolyakMidSet, TPMidSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested


def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : np.linalg.norm(x, ord=2, axis=1)
    errSample = np.zeros(uExa.shape[0])
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))

# choose function
N = 3
dimF = 3
anisoVec = np.array([1,10., 100])
def F(x):
    return np.sin(np.sum(x/anisoVec))* np.array([1., 2., 3.])

# error computations
NRNDSamples = int(1.e5)
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
fOnxx = np.zeros((NRNDSamples, dimF))
for w in range(NRNDSamples):
    fOnxx[w] = F(xxRND[w, :])

# choose interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = lambda m : unboundedKnotsNested(m, p=2)
maxNumNodes = 1000

# convergence test
err = []
nNodes = []
oldSG = None
FOnSG = None
w=0
while True:
    # print("Computing n = ", w)
    # I = TPMidSet(w, N)
    I = anisoSmolyakMidSet(w, N, anisoVec)
    interpolant = SGInterpolant(I, knots, lev2knots, "quadratic")
    SG = interpolant.SG
    if(interpolant.numNodes > maxNumNodes):
        break
    nNodes.append(interpolant.numNodes)
    FOnSG = interpolant.sampleOnSG(F, dimF, oldSG, FOnSG)
    fInterp = interpolant.interpolate(xxRND, FOnSG)
    err.append(computeL2Error(fInterp, fOnxx))
    oldSG = interpolant.SG
    w+=1

print(err)
err = np.array(err)
nNodes = np.array(nNodes)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', nNodes, 1/nNodes, '-k')
#############################################################################################
# choose interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = lambda m : unboundedKnotsNested(m, p=2)
maxNumNodes = 1000

# convergence test
err = []
nNodes = []
oldSG = None
FOnSG = None
w=0
while True:
    # print("Computing n = ", w)
    I = TPMidSet(w, N)
    # I = SmolyakMidSet(w, N)
    interpolant = SGInterpolant(I, knots, lev2knots, "quadratic")
    SG = interpolant.SG
    if(interpolant.numNodes > maxNumNodes):
        break
    nNodes.append(interpolant.numNodes)
    FOnSG = interpolant.sampleOnSG(F, dimF, oldSG, FOnSG)
    fInterp = interpolant.interpolate(xxRND, FOnSG)
    err.append(computeL2Error(fInterp, fOnxx))
    oldSG = interpolant.SG
    w+=1

print(err)
err = np.array(err)
nNodes = np.array(nNodes)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', nNodes, 1/nNodes, '-k')
plt.show()
