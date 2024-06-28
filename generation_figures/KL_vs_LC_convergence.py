import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log, pi
from scipy.integrate import simps
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.expansions_Brownian_motion import param_KL_Brownian_motion, param_LC_Brownian_motion


def sample_Wiener_process(tt):
    # generate samples Wiener process with classical sampling algorithm
    WW = np.zeros(tt.size)
    for n in range(1, tt.size):
        WW[n] = WW[n-1] + np.random.normal(0, sqrt(tt[n]-tt[n-1]))
    return WW

def compute_KL_corrdimates(tt, WW, N):
    # compute the KL coordinates of the Wiener process
    yy = np.zeros(N)
    for n in range(N):
        basisF = sqrt(2)*np.sin((n+0.5)*pi*tt)
        yy[n] = simps(WW*basisF, tt) / (pi*(n+0.5))
    return yy

def compute_LCCoords(tt, WW, N):
    # assume tt.size = 2**k +1  for some k\in\N
    d = int(log(tt.size-1, 2))
    assert(tt.size == 2**d+1)
    # assume tt.size \geq N
    assert(N <= tt.size)
    # assume N is compatible with LC levels
    L = int(log(N, 2))
    assert(N == 2**L)
    ##################################
    coordsLC = np.zeros(N)
    coordsLC[0] = WW[-1]  # do l=0 separately
    for l in range(1, L+1):  # from l=1 to L
        WLCCurr = param_LC_Brownian_motion(tt, coordsLC, T)
        for j in range(1, 2**(l-1)+1):  # from j = 1 to 2**(l-1)
            pos = (2*j-1)*2**(-l+d)
            valBFPos = 2**(-(l+1)*0.5)
            coordsLC[2**(l-1)+j-1] = (WW[pos] - WLCCurr[pos]) / valBFPos
    return coordsLC
def LpNorm(x, p, dt):
    if p == float('inf'):
        return np.amax(np.abs(x))
    else:
        return np.linalg.norm(x, p) * dt**(1/p)

############################# MAIN ##############################
# PARAMETERS
T=1
tt = np.linspace(0, T, 2**12+1)
dt = T/tt.size
p = float('inf')  # Lp error in time
nRefs = 7
nMCSamples = 128
errKL = np.zeros((nMCSamples, nRefs))
errLC = np.zeros((nMCSamples, nRefs))
NCoords = 2**np.linspace(0, nRefs-1, nRefs).astype(int)  # number of KLE terms convergence test

# CONVERGENCE TEST
for nSample in range(nMCSamples):
    # sample on tt a sample path of Wiener process
    WW = sample_Wiener_process(tt)
    for nRef, NCoord in enumerate(NCoords):  # couter, value
        # generate KL
        yyKL = compute_KL_corrdimates(tt, WW, NCoord)
        WKL = param_KL_Brownian_motion(tt, yyKL)

        # generate LC
        LCCoords = compute_LCCoords(tt, WW, NCoord)
        WLC = param_LC_Brownian_motion(tt, LCCoords, T)

        # compute error sample
        #KL
        errCurrKL = LpNorm(WW-WKL, p, dt)
        errKL[nSample, nRef] = errCurrKL
        # print("error KL:", errCurrKL)
        # LC
        errCurrLC = LpNorm(WW-WLC, p, dt) 
        errLC[nSample, nRef] = errCurrLC
        # print("error LC:", errCurrLC)

        # plt.plot(tt, WKL, 'r-', label="KL")
        # # plt.plot(tt, WLC, 'b-', label="LC")
        # plt.plot(tt, WW, 'k-', label="True")
        # plt.show()

# comptue RMSE
errKL = np.sqrt(np.mean(np.square(errKL), axis=0))
errLC = np.sqrt(np.mean(np.square(errLC), axis=0))

np.savetxt('conv_KL_p_inf.csv', np.array([NCoords, errKL]).T , delimiter=',')
np.savetxt('conv_LC_p_inf.csv', np.array([NCoords, errLC]).T , delimiter=',')

print("RMSE KL:", errKL)
print("RMSE LC:", errLC)
plt.loglog(NCoords, errKL, label="RMSE KL")
plt.loglog(NCoords, errLC, label="RMSE LC")
# plt.plot(NCoords, 1/NCoords, 'k-', label="N^{-1}")
plt.plot(NCoords, 1/np.sqrt(NCoords), 'k-', label="N^{-1/2}")
plt.xlabel("Number of expansion terms")
plt.title("RMSE KL and LC with L^p error in time, p ="+str(p))
plt.legend()
plt.show()