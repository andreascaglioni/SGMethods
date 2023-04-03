from __future__ import division
import numpy as np
from math import sqrt, log2, pi, factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from multiprocessing import Pool
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import TPMidSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion


NParallel = 16
NRNDSamples = 512
maxNumNodes = 8000
p = 3 # NBB degree+1
N=2

# choose function
Nt = 1.e3
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    W = param_LC_Brownian_motion(tt, x[0:2], 1)
    return np.sin(W)

def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))

# error computations
NLCExpansion = 2**10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = np.array(pool.map(F, yyRnd))
dimF = uExa[0].size

# choose interpolant
lev2knots = lambda n: 2**(n+1)-1
# def lev2knots(n):
#     w=np.where(n==0)
#     n = n+1
#     n[w]=0
#     return 2**(n+1)-1

knots = lambda m : unboundedKnotsNested(m, p=p)

# convergence test
err = np.array([])
nNodes = np.array([])
oldSG = None
uOnSG = None
w=0
while True:
    print("Computing w  = ", w)
    I = TPMidSet(w, N)
    # I = SmolyakMidSet(w, N)
    interpolant = SGInterpolant(I, knots, lev2knots, "quadratic")
    if(interpolant.numNodes > maxNumNodes):
        break
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", N)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)

    # ERROR
    errCurr = computeL2Error(uExa, uInterp)
    err = np.append(err, errCurr)
    nNodes = np.append(nNodes, interpolant.numNodes)
    print("Error:", err[w])
    
    oldSG = interpolant.SG   
    w+=1

tt = np.linspace(0, 1, Nt)
plt.plot(tt, uExa[0,:], '.-', tt, uInterp[0,:], '.-')
plt.show()

print(err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-')
plt.loglog(nNodes, np.power(nNodes, -0.5))
plt.loglog(nNodes, np.power(nNodes, -0.25))
plt.show()

