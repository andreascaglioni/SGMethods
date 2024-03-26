import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy import integrate

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


# first sample Wiener process with definition
nt = 10000
tt = np.linspace(0,1,nt)
WW = sample_Wiener(tt)

#compute KLE
nY = 1000
WWKL = compute_KLE(WW, tt, nY)

# PLOT PATHS
plt.plot(tt, WW, label="classic")
plt.plot(tt, WWKL, label="KLE")
plt.legend()
plt.show()
print(L2err(WW, WWKL, tt))

# PLOT CONVERGENCE KLE in L2
err = []
NNY = 2**np.arange(1,8)
for nY in NNY:
    WWKL = compute_KLE(WW, tt, nY)
    errCurr = L2err(WW, WWKL, tt)
    err.append(errCurr)
print("error refinement KLE:", err)
plt.loglog(NNY, err, '.-')
plt.loglog(NNY, 1/NNY, 'k-')
plt.loglog(NNY, 1/np.sqrt(NNY), 'k-')
plt.show()



# compute LCE coefficients
# convergence truncation in L2 (KLE wins because it is optimal)
# then in L1
# then in Linfty