from __future__ import division
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from SGMethods.MidSets import SmolyakMidSet, anisoSmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from ParametricPoisson import sampleParametricPoisson, computeErrorPoisson
import scipy.io
from multiprocessing import Pool

# choose function
N = 5
nH = 16
orderDecayCoefficients = 2
u, V, mesh = sampleParametricPoisson([0.], nH, orderDecayCoefficients)
dimF = V.dim()
def F(y):
    return sampleParametricPoisson(y, nH, orderDecayCoefficients)[0].vector()[:]

# choose interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = unboundedKnotsNested
normAn = 1/(np.linspace(1, N, N)**orderDecayCoefficients)
tau = 0.5/normAn
anisoVector = 0.5*np.log(tau + np.sqrt(1+tau*tau))

# error computations
NRNDSamples = 128
NParallel = 8
yyRnd = np.random.normal(0, 1, [NRNDSamples, N])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)

maxNumNodes = 100
# convergence test
err = np.array([])
nNodes = np.array([])
w=0
while(True):
    print("Computing w  = ", w)
    I = anisoSmolyakMidSet(w*min(anisoVector), N, anisoVector)
    # I = SmolyakMidSet(w, N)
    interpolant = SGInterpolant(I, knots, lev2knots, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    print(interpolant.numNodes, "nodes")
    if w ==0 :
        oldSG = None
        uOnSG = None
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    
    err = np.append(err, computeErrorPoisson(yyRnd, uInterp, uExa, V))
    nNodes = np.append(nNodes, interpolant.numNodes)
    
    print("Error:", err[w])
    oldSG = interpolant.SG
    w+= 1

print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', nNodes, 1/nNodes, '-k')
plt.show()