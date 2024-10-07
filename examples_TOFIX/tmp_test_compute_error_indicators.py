import numpy as np
from math import sqrt, factorial, log2, pi
from multiprocessing import Pool
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from sgmethods.MidSet import MidSet
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.nodes_1d import unboundedKnotsNested
from sgmethods.compute_error_indicators import compute_GG_indicators_RM
from sgmethods.parametric_expansions_Wiener_process import param_LC_Brownian_motion
from sgmethods.tp_inteprolants import TPPwLinearInterpolator


# Parameters
np.random.seed(1607)  # Changing the seed may make the test fail
T = 1
NRNDSamples = 1024
maxNumNodes = 150

NParallel = 16
Nt = 100
tt = np.linspace(0, T, Nt)
dt = 1/Nt

def F(x):  # function to interpolate
    return np.sin(param_LC_Brownian_motion(tt, x, T))

def compute_error(u, uExa):  # Error function
    assert(uExa.shape == u.shape)
    if(len(u.shape)==1):
        u = np.reshape(u, (1, -1))
        uExa = np.reshape(uExa, (1, -1))
    errSample = sqrt(dt)*np.linalg.norm(u-uExa, ord=2, axis=1) 
    return sqrt(np.mean(np.square(errSample)))

# Comptue reference
yyRnd = np.random.normal(0, 1, [NRNDSamples, 2**10])
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
p = 2


# convergence test
err = np.array([])
nNodes = np.array([])
nDims = np.array([])
err_estimator = np.array([])
w=0
I = MidSet(track_reduced_margin=True)
oldSG = None
uOnSG = None
oldRM = np.array([])
error_indicators_RM = np.array([])
while True:
    print("Computing w  = ", w)

    # COMPUTE
    interpolant = SGInterpolant(I.mid_set, knots, lev2knots, 
                                TPPwLinearInterpolator,  NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    midSetMaxDim = np.amax(I.mid_set, axis=0)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.get_dim(), "\nMax midcomponents:", midSetMaxDim)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    oldSG = interpolant.SG  # Save for next iteration
    uInterp = interpolant.interpolate(yyRnd, uOnSG)

    # Compute reference error
    err = np.append(err, compute_error(uExa, uInterp))
    print("Error:", err[-1])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.get_dim())

    # ESTIMATE
    error_indicators_RM = compute_GG_indicators_RM(oldRM, error_indicators_RM,
                                                   I, knots, lev2knots, F, 
                                                   interpolant.SG, uOnSG, yyRnd,
                                                   compute_error, uInterp, 
                                                   TP_inteporlant=TPPwLinearInterpolator, 
                                                   NParallel=NParallel)
    oldRM = I.reduced_margin
    err_estimator = np.append(err_estimator, np.sum(error_indicators_RM))
    # np.savetxt('sinW_example_linear_adaptive.csv',np.transpose(np.array([nNodes, err, nDims, err_estimator])), delimiter=',')
    print("error indicators reduced margin", np.sum(error_indicators_RM))

    # MARK
    # I consider the correspondin profit i.e. I divide the norm of hierarchica surpluses by the number of corresponding added nodes (work)
    RM = I.reduced_margin
    work_RM = np.prod((2**(RM+1)-2)*(p-1)+1, axis=1)  # TODO relate to lev2knots istead of hard-coding
    idMax = np.argmax(error_indicators_RM / work_RM)  # NBB index in REDUCED margin
    mid = I.reduced_margin[idMax, :]
    idxMargin = np.where(( I.margin==mid).all(axis=1) )[0][0]
    
    # REFINE
    I.enlarge_from_margin(idxMargin)
    w+=1
    
    plt.loglog(nNodes, err, '.-')
    plt.loglog(nNodes, err_estimator, '.-')
    plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
    plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')
    plt.pause(0.05)


print(err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
# plt.loglog(nNodes, err, '.-')
# plt.loglog(nNodes, err_estimator, '.-')
# plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
# plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')
# plt.draw()