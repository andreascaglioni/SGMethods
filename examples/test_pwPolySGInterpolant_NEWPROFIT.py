from __future__ import division
import numpy as np
from math import sqrt, log2, pi, factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from multiprocessing import Pool
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.multi_index_sets import midSet, TPMidSet
from SGMethods.sparse_grid_interpolant import SGInterpolant
from SGMethods.nodes_1d import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion


NParallel = 16
NRNDSamples = 512
maxNumNodes = 250
interpolationType = 'quadratic'
p = 3 # NBB degree+1

# choose function
Nt = 1.e3
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    # x = x[0:16]
    # N = x.size
    # ww = 2**(-0.5*np.ceil(np.log2(np.linspace(1, N ,N))))
    # out = np.sum( np.square(x) * ww )*np.array([1., 2.])
    # return out
    W = param_LC_Brownian_motion(tt, x, 1)
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
beta=0.5
Cbar = 2
def ProfitFun(x):
    logRho = beta*np.ceil(np.log2(np.linspace(1,x.size,x.size)))
    C1 = np.sum(x)
    C2 = np.sum(logRho, where=(x==1))
    C3 = np.sum(-log2(Cbar) + p*x + p*logRho, where=(x>1))
    C4 = np.count_nonzero(x)* (-log2(factorial(p)) + p)
    return C1 + C2 + C3

# convergence test
err = np.array([])
nNodes = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
while True:
    print("Computing w  = ", w)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # ERROR
    errCurr = computeL2Error(uExa, uInterp)
    err = np.append(err, errCurr)
    nNodes = np.append(nNodes, interpolant.numNodes)
    print("Error:", err[w])
    oldSG = interpolant.SG
    # update midset for next iteration
    P = np.apply_along_axis(ProfitFun, 1, I.margin)
    idMax = np.argmin(P)
    # print("new Mid", I.margin[idMax, :])
    I.update(idMax)


    # ncps = interpolant.numNodes
    # while interpolant.numNodes < sqrt(2)*ncps:
    #     P = np.apply_along_axis(ProfitFun, 1, I.margin)
    #     idMax = np.argmin(P)
    #     # print("new Mid", I.margin[idMax, :])
    #     I.update(idMax)
    #     interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    w+=1

print(I.midSet)
print(np.amax(I.midSet, axis=0))

# tt = np.linspace(0, 1, Nt)
# plt.plot(tt, uExa[0,:], '.-', tt, uInterp[0,:], '.-')
# plt.show()

print(err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-')
plt.loglog(nNodes, np.power(nNodes, -0.5))
plt.loglog(nNodes, np.power(nNodes, -0.25))
plt.show()