import numpy as np 
import matplotlib.pyplot as plt
from math import ceil, sqrt
from SLLG.expansions_Brownian_motion import param_KL_Brownian_motion, param_LC_Brownian_motion


"""Compare convergence truncated KLE and LCE of Wiener process"""

def compute_L2_norm(W,dt):
    # NB assume equispaced
    n = W.size
    A = np.diag( np.full(n, 2/3) )
    A[0,0] = 1/3
    A[-1,-1] = 1/3
    A = A + np.diag(np.full(n-1,1/6), 1)
    A = A + np.diag(np.full(n-1,1/6), -1)
    A = A*dt
    tmp = np.dot(A, W)
    return sqrt(np.dot(W, tmp))


logntt = 13
ntt = 2**logntt+1
tt = np.linspace(0,1,ntt)
yy = np.random.normal(0,1,2**logntt)
W = param_KL_Brownian_motion(tt, yy)
eL2=np.zeros(logntt)
eLinf=np.zeros(logntt)
for i in range(logntt):
    yyT = yy[0:2**i+1]
    WKL = param_KL_Brownian_motion(tt, yyT)
    eL2[i] = compute_L2_norm(W-WKL, tt[1]-tt[0])
    eLinf[i] = np.amax(W-WKL)
print(eL2)
print(eLinf)
ttV = 2**np.linspace(0, logntt-1, logntt)
plt.loglog(ttV, eL2, '.-')
plt.loglog(ttV, eLinf, '.-')
plt.show()

