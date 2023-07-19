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
maxNumNodes = 150
p=2
interpolationType =  "linear"
NParallel = 16
Nt = 100
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    return np.sin(param_LC_Brownian_motion(tt, x, 1))

def L2errParam(u, uExa):
    assert(uExa.shape == u.shape)
    if(len(u.shape)==1):
        u = np.reshape(u, (1, -1))
        uExa = np.reshape(uExa, (1, -1))
    errSample = sqrt(dt)*np.linalg.norm(u-uExa, ord=2, axis=1) 
    return sqrt(np.mean(np.square(errSample)))

np.random.seed(1607)
yyRnd = np.random.normal(0, 1, [NRNDSamples, 2**10])  # infinite paramter vector
print("Parallel random sampling reference solution...")
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
err_estimator = np.array([])
w=0
I = midSet(trackReducedMargin=True)
oldSG = None
uOnSG = None

oldRM = np.array([])
error_indicators_RM = np.array([])
while True:
    print("Computing w  = ", w)
    # COMPUTE
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    midSetMaxDim = np.amax(I.midSet, axis=0)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N, "\nMax midcomponents:", midSetMaxDim)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # COMPUTE ERROR
    err = np.append(err, L2errParam(uExa, uInterp))
    print("Error:", err[-1])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    oldSG = interpolant.SG

    # ESTIMATE
    error_indicators_RM = compute_aposteriori_estimator_reduced_margin(oldRM, error_indicators_RM, I, knots, lev2knots, F, interpolant.SG, uOnSG, yyRnd, L2errParam, uInterp, interpolationType=interpolationType, NParallel=NParallel)
    oldRM = I.reducedMargin
    err_estimator = np.append(err_estimator, np.sum(error_indicators_RM))
    np.savetxt('sinW_example_linear_adaptive.csv',np.transpose(np.array([nNodes, err, nDims, err_estimator])), delimiter=',')
    print("error indicators reduced margin", np.sum(error_indicators_RM))

    # MARK
    # I consider the correspondin profit i.e. I divide the norm of hierarchica surpluses by the number of corresponding added nodes (work)
    RM = I.reducedMargin
    work_RM = np.prod((2**(RM+1)-2)*(p-1)+1, axis=1)
    idMax = np.argmax(error_indicators_RM / work_RM)  # NBB index in REDUCED margin
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