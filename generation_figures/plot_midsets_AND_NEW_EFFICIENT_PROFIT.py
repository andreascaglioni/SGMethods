from __future__ import division
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from multiprocessing import Pool
from math import log, pi, sqrt, log2
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet, SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant



maxNumMids = 103
p=2
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))

def compute_midset(ProfitFun):
    w=0
    I = midSet()
    while True:
        maxDim = np.amax(I.midSet, axis = 0)
        print("dim", I.N, "card", I.numMids, "maxDim", maxDim)  #  , "card margin", I.margin.shape[0])
        nMids = I.numMids
        if(nMids > maxNumMids):
            break 
        tmp_nMids = nMids
        while(tmp_nMids <= sqrt(2) * nMids):
            # P = np.apply_along_axis(ProfitFun, 1, I.margin)
            P = ProfitFun(I.margin)
            idMax = np.argmin(P)
            I.update(idMax)
            tmp_nMids = I.numMids
        w+=1
    return I




######################################################### OLD PROFIT
# ProfitFunOld = lambda x : (p+1) * np.sum(x) + p * np.sum(np.log2(rhoFun(x.size)) , where=(x>0))
def ProfitFunOld(x):
    # INPUT x np array shaped (# mids, dimension mids)
    # OUTPUT profit np array shaped (# mids, )
    if(len(x.shape) == 1):
        x = np.reshape(x, (1,-1))
    nMids = x.shape[0]
    nDims = x.shape[1]
    logRho = beta*np.ceil(np.log2(np.linspace(1,nDims,nDims)))
    logRhoReshape = np.reshape(logRho, (1, -1))
    repLogRho = np.repeat(logRhoReshape, nMids, axis=0)
    return (p+1) * np.sum(x, axis=1) + p * np.sum(repLogRho, axis=1, where=(x>0))

I = compute_midset(ProfitFunOld)
# plt.plot(I.midSet[:,0], I.midSet[:,1], '.', markersize=20)
# plt.show()





######################################################### NEW PROFIT
# Cbar = 2
# def ProfitFun(x):
#     # INPUT x np array shaped (# mids, dimension mids)
#     # OUTPUT profit np array shaped (# mids, )
#     if(len(x.shape) == 1):
#         x = np.reshape(x, (1,-1))
#     nMids = x.shape[0]
#     nDims = x.shape[1]
#     logRho = beta*np.ceil(np.log2(np.linspace(1,nDims,nDims)))
#     C1 = np.sum(x, axis=1)
#     logRhoReshape = np.reshape(logRho, (1, -1))
#     repLogRho = np.repeat(logRhoReshape, nMids, axis=0)
#     C2 = np.sum(repLogRho, axis=1, where=(x==1))
#     vecVals = -log2(Cbar) +p*x + p * repLogRho
#     C3 = np.sum(vecVals, axis=1, where=(x>1))
#     return C1 + C2 + C3
# I = compute_midset(ProfitFun)
# plt.plot(I.midSet[:,0], I.midSet[:,1], '.', markersize=20)
# plt.show()
