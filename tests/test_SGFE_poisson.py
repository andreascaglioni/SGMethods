from __future__ import division
import sys
sys.path.append('/home/ascaglio/workspace/SGMethods')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from SGMethods.MidSets import anisoSmolyakMidSet, SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from ParametricPoisson import sampleParametricPoisson, computeErrorPoisson

# choose function
N = 2
nH = 16
u, V, mesh = sampleParametricPoisson([0], nH)
dimF = V.dim()
F = lambda y: sampleParametricPoisson(y, nH)[0].vector()[:]

# define aniso vector
normAn = 1/(1.65*np.linspace(1, N, N)**2)
tau = 0.5/normAn
anisoVector = 0.5*np.log(tau + np.sqrt(1+tau*tau))

# choose interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = unboundedKnotsNested

# error computations
nLevels = 5
NRNDSamples = 2
yyRnd = np.random.normal(0, 1, [NRNDSamples, N])
uExa = []
for n in range(NRNDSamples):
    uExa.append(F(yyRnd[n,:]))

# convergence test
err = np.zeros(nLevels)
nNodes = np.zeros(nLevels)
for w in range(0, len(err)):
    print("Computing n = ", w)
    I = anisoSmolyakMidSet(w, N, anisoVector)
    interpolant = SGInterpolant(I, knots, lev2knots)
    SG = interpolant.SG
    nNodes[w] = interpolant.numNodes
    if w ==0 :
        oldSg = None
        FOnSG = None
    uOnSG = interpolant.sampleOnSG(F, dimF)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    err[w] = computeErrorPoisson(yyRnd, uInterp, uExa, V)
    oldSG = interpolant.SG

print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', nNodes, 1/nNodes, '-k')
plt.show()