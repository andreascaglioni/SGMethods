import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, log
from scipy import integrate
from copy import deepcopy
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
"""geerate a Wiener process, compute the KLE and LCE truncations. Compare convergence as number of terms 
increases in different norms"""


def sample_Wiener(tt):
    assert abs(tt[0]) < 1e-10
    W = np.zeros_like(tt)
    W[0] = 0
    for nt in range(1, len(tt)):
        dW = np.random.normal(0,sqrt(tt[nt] - tt[nt-1]))
        W[nt] = W[nt-1] + dW
    return W

def L2err(y1, y2, tt):
    assert(y1.shape == y2.shape)
    assert(y1.shape == tt.shape)
    assert(len(tt.shape) == 1)
    dt = np.diff(tt)
    dy = y2[:-1]-y1[:-1]
    return sqrt(np.dot(dt, np.square(dy)))

def compute_KLE(WW, tt, nY):
    jj = np.reshape(np.arange(1,nY+1), (nY,1))
    phi = sqrt(2)*np.sin((jj-0.5)*pi*tt)  # matrix of basis functions of size #basis funs x #timesteps
    # compute coefficient YY (1D array of size nY) as <W, phi_i>_{L^2(0,1)} = int_0^1 W phi_i
    YY = np.zeros(nY)
    for i in range(nY):
        YY[i] =  integrate.simps(WW*phi[i,:], tt)
    # compute the truncated KLE
    WWKL = np.dot(phi.T, YY)
    return WWKL

def compute_LCE(WW, tt, nY):
    # use recursive formula for LCE    
    # coeff 0,1
    xNew = 1
    LCECoeffs = np.array([np.interp(xNew, tt, WW)])  # nb the basis function is 1 at t=1
    WWTmp = WW - param_LC_Brownian_motion(tt, LCECoeffs, T=1)
    xxEval = np.array([xNew])
    yyEval = LCECoeffs
    # other coeff.s
    L = int(log(nY, 2))
    JMax = nY - 2**L
    for ell in range(1, L+1):
        maxBasisFun = 2**(-(ell-1)/2)*0.5
        for j in range(0, 2**(ell-1)):
            if j == ell-1 and j > JMax:
                break
            xNew = (j+0.5)*2**-(ell-1)
            yInterp = np.interp(xNew, tt, WWTmp)
            LCECoeffs = np.append(LCECoeffs, yInterp/maxBasisFun)
            xxEval = np.append(xxEval, xNew)
            yyEval = np.append(yyEval, yInterp)
            WWTmp = WW - param_LC_Brownian_motion(tt, LCECoeffs, T=1)
    return param_LC_Brownian_motion(tt, LCECoeffs, T=1)

########## EXAMPLE 1: just plot a Wiener process sample path and truncated KLE, LCE
"""
# sample Wiener process with definition
nt = 2**10+1
tt = np.linspace(0,1,nt)
WW = sample_Wiener(tt)
#compute KLE
nY = 1000
WWKL = compute_KLE(WW, tt, nY)
#compute LCE
nY = 1000
WWLC = compute_LCE(WW, tt, nY)
# plot
print("error KLE:", L2err(WW, WWKL, tt))
print("error LCE:", L2err(WW, WWLC, tt))
plt.plot(tt, WW, label="classic")
plt.plot(tt, WWKL, label="KLE")
plt.plot(tt, WWLC, label="LCE")
plt.legend()
plt.show()
"""

########## EXAMPLE 2: COMPARE CONVERGENCE OF KLE AND LCE IN  L2(\Omega, L^2(0,T))
# approximate the probabiltiy L2 norm with a Monte Carlo estimate
nt = 2**10+1
tt = np.linspace(0,1,nt)
NMC = 100
nRefs = 7
errKL = np.zeros((NMC, nRefs))
errLC = np.zeros_like(errKL)
for nSample in range(NMC):
    WW = sample_Wiener(tt)
    NNY = 2**np.arange(0,nRefs+1)
    for nRef in range(len(NNY)):
        nY = NNY[nRef]
        WWKL = compute_KLE(WW, tt, nY)
        WWLC = compute_LCE(WW, tt, nY)

        plt.plot(tt, WW, label="Wiener process")
        plt.plot(tt, WWLC, label="LC")
        plt.show()

        errCurrLC = L2err(WW, WWLC, tt)
        errKL[nSample, nRef] = L2err(WW, WWKL, tt)
        errLC[nSample, nRef] = L2err(WW, WWLC, tt)
errKL = np.mean(errKL, axis=0)  # MC average
errLC = np.mean(errLC, axis=0)  # MC average

print("error refinement KLE:", errKL)
print("error refinement LCE:", errLC)

plt.loglog(NNY, errKL, '.-')
plt.loglog(NNY, errLC, '.-')
plt.loglog(NNY, 1/NNY, 'k-')
plt.loglog(NNY, np.power(NNY, -0.5), 'k-')
plt.show()



# compare convergence truncation in L2 (KLE wins because it is optimal)
# then in L1
# then in Linfty