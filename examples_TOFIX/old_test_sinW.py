import numpy as np
from math import sqrt
from multiprocessing import Pool
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.multi_index_sets import midSet
from SGMethods.sparse_grid_interpolant import SGInterpolant
from SGMethods.nodes_1d import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.profits import ProfitMix


# NUMERICS PARAMETERS
NRNDSamples = 256
np.random.seed(1607)
maxNumNodes = 128
p=2
interpolationType =  "linear"
lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p=p)
Profit = lambda nu : ProfitMix(nu, p)

# FUNCTION SAMPLER
Nt = 100
tt = np.linspace(0, 1, Nt)
dt = 1/Nt
def F(x):
    return np.sin(param_LC_Brownian_motion(tt, x, 1))

# ERROR COPMUTATION
def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)  # L2 norm in time 
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))
yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])  # infinite paramter vector
uExa = np.array(list(map(F, yyRnd)))

# CONVERGENCE TEST
err = np.array([])
nNodes = np.array([])
nDims = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
while True:
    print("Computing w  = ", w)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=1)
    if(interpolant.numNodes > maxNumNodes):
        break
    midSetMaxDim = np.amax(I.midSet, axis=0)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N, "\nMax midcomponents:", midSetMaxDim)
    uOnSG = interpolant.sampleOnSG(F, oldXx=oldSG, oldSamples=uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # COMPUTE ERROR
    err = np.append(err, computeL2Error(uExa, uInterp))
    print("Error:", err[-1])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    oldSG = interpolant.SG
    ncps = interpolant.numNodes
    while interpolant.numNodes < sqrt(2)*ncps:
        P = Profit(I.margin)
        idMax = np.argmax(P)
        I.update(idMax)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType)
        if(interpolant.numNodes > maxNumNodes):
            break
    w+=1
# RESULTS
print("Error: ", err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-')
plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')
plt.show()