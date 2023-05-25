import numpy as np
from math import sqrt, factorial, log2, pi
from multiprocessing import Pool
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.compute_aposteriori_estimator_reduced_margin import compute_aposteriori_estimator_reduced_margin
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion



NRNDSamples = 100
maxNumNodes = 400
p=2
interpolationType =  "linear"


Nt = 100
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    return np.sin(param_LC_Brownian_motion(tt, x, 1))

def physicalNorm(x):
    if(len(x.shape)==1):
        x = np.reshape(x, (1, -1))
    return sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)  # L2 norm in time 

def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    errSample = physicalNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))

np.random.seed(1607)
yyRnd = np.random.normal(0, 1, [NRNDSamples, 2**10])  # infinite paramter vector
print("Parallel random sampling")
NParallel = 1
if NParallel == 1:
    dimF = F(yyRnd[0,:]).size
    uExa = np.zeros((NRNDSamples, dimF))
    for i in range(NRNDSamples):
        uExa[i,:] = F(yyRnd[i,:])
else:
    pool = Pool(NParallel)
    uExa = np.array(pool.map(F, yyRnd))
dimF = uExa.shape[1]

# Choose interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p=p)



# convergence test
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
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # ERROR
    err = np.append(err, computeL2Error(uExa, uInterp))
    print("Error:", err[-1])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    
    np.savetxt('sinW_example_linear_adaptive.csv',np.transpose(np.array([nNodes, err, nDims])), delimiter=',')

    oldSG = interpolant.SG

    # ESTIMATE
    estimator_reduced_margin = compute_aposteriori_estimator_reduced_margin(I, knots, lev2knots, F, dimF, interpolant.SG, uOnSG, yyRnd, physicalNorm, NRNDSamples, uInterp, interpolationType=interpolationType, NParallel=NParallel)
    # MARK
    idMax = np.argmax(estimator_reduced_margin)  # NBB index in REDUCED margin
    mid = I.reducedMargin[idMax, :]
    idxMargin = np.where(( I.margin==mid).all(axis=1) )[0][0]
    # REFINE
    I.update(idxMargin)
    w+=1
    


print(err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-')
plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')
plt.show()