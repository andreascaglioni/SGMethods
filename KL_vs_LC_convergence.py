import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.expansions_Brownian_motion import param_KL_Brownian_motion, param_LC_Brownian_motion


def compute_LCCoords(tt, W, nCoords):
    # assume tt.size = 2**k +1 for some k\in\N
    # assume nCoords \leq tt.size
    
    ntt = tt.size
    L = log(nCoords-1, 2)
    coordsLC = np.zeros(nCoords)
    coordsLC[0] = W[-1] # do l=0 separately
    for l in range(1, L+1): # from l=1 onwards
        for j in range(1, 2**(l-1)+1): # there are 2**l-1 basis funs per level
            pos = 
            coordsLC[2**(l-1)+j] = W[pos]





T=1
nT = 1000
tt = np.linspace(0, T, 1024+1)
dt = T/tt.size
nRefs = 9
nExa = 2**(nRefs+2)
nSamples = 32
err = np.zeros((nSamples, nRefs))
JKLs = 2**np.linspace(0, nRefs-1, nRefs).astype(int)
print(JKLs)
for nSample in range(nSamples):
    #generate sample
    YY = np.random.normal(0, 1, nExa)
    # generate almost exact Wiener process
    W = param_KL_Brownian_motion(tt, YY)
    for nRef, JKL in enumerate(JKLs):
        
        # generate KL
        # WKL = param_KL_Brownian_motion(tt, YY[0:JKL])
        LCCoords = compute_LCCoords(tt, W)
        WLC = param_LC_Brownian_motion(tt, LCCoords, T)






        # compute error sample
        # errCurrKL = np.linalg.norm(W - WKL, 2) * sqrt(dt)
        errCurrLC = np.linalg.norm(W - WLC, 2) * sqrt(dt)
        print(errCurrLC)
        err[nSample, nRef] = errCurrLC
print(err)
# comptue RMSE
err = np.sqrt(np.mean(np.square(err), axis=0))
plt.loglog(JKLs, err, label="RMSE KL")
plt.plot(JKLs, 1/JKLs, 'k-', label="N^{-1}")
plt.plot(JKLs, 1/np.sqrt(JKLs), 'k-', label="N^{-1/2}")
plt.xlabel("Number of expansion terms")
plt.title("RMSE with L^2 error in time")
plt.legend()
plt.show()