import numpy as np
from math import sin, sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import SmolyakMidSet
from SGMethods.MLInterpolant import MLInterpolant
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

nLevels = 4  # number of levels -1
kk = np.linspace(0,nLevels-1, nLevels)
hh = 2**(-kk-3)
FExa = lambda y : sin(np.sum(y))
FApprox = lambda y, k : FExa(y) + 2**(-k-3)
d=3
cost = np.power(hh, -d)  # the cost of 1 sample in space
dimF = 1
# error computation
NSamples = 1000
dimParam = 2
yyRnd = np.random.normal(0, 1, [NSamples, dimParam])
samplesFExa = list(map(FExa, yyRnd))
samplesFExa = np.reshape(np.array(samplesFExa), (-1, 1))
def compute_error(FAppr):
    errSamples = FAppr - samplesFExa
    return sqrt(1/NSamples * np.sum(np.square(errSamples)))

# Convegence test
p = 2  # NBB degrere + 1
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
err = []
totalCost = []
for nLevels in range(1, 11):
    SLInterpolSequence = []
    for k in range(nLevels):
        midCurr = SmolyakMidSet(k, N=dimParam)
        SLInterpolSequence.append(SGInterpolant(midCurr, knots, lev2knots, interpolationType))
    interpolant = MLInterpolant(SLInterpolSequence, dimF)
    FOnSGML = interpolant.sample(FApprox)
    # interpolate at some point
    FInterp = interpolant.interpolate(yyRnd, FOnSGML)
    # compute error
    err.append(compute_error(FInterp))
    totalCostCurr = 0
    kk = np.linspace(0,nLevels-1, nLevels)
    hh = 2**(-kk-3)
    cost = np.power(hh, -d)
    for k in range(nLevels):
        totalCostCurr += SLInterpolSequence[k].numNodes * (cost[nLevels-k-1])
    totalCost.append(totalCostCurr)
    
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(yyRnd[:,0], yyRnd[:,1], samplesFExa)
    # ax.scatter(yyRnd[:,0], yyRnd[:,1], FInterp)
    # plt.show()
plt.loglog(cost, err, '.-')
plt.loglog(cost, np.power(cost, -1/d), 'k-')
plt.show()