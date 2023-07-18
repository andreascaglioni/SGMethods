from __future__ import division
from multiprocessing import Pool
from math import log, pi, sqrt, factorial, log2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.sample_LLG_function_noise_2 import sample_LLG_function_noise_2
from SLLG.error_sample_pb2_L2norm import error_sample_pb2_L2norm


""" Compute derivatives of parameter-to-solution amp of SLLG. 
The results are stored in dm, a 2d array. Each row corresponds to a different parameter. 
Each colum correpsond to decreasing finite differences"""

T=1
Nh=32
Ntau = Nh*8
FEMOrder = BDFOrder = 1

D = 32
nE = 3
ee = np.power(2, -5-np.linspace(1,nE,nE))
y0 = np.zeros(D)
mY = sample_LLG_function_noise_2(y0, Nh, Ntau, T, FEMOrder, BDFOrder)

def compute_row(dCurr):
    r = np.zeros(ee.size)
    for idxEps in range(ee.size):
        epsCurr = ee[idxEps]
        yCurr = np.copy(y0)
        yCurr[dCurr] = yCurr[dCurr] + epsCurr
        mYCurr = sample_LLG_function_noise_2(yCurr, Nh, Ntau, T, FEMOrder, BDFOrder)
        r[idxEps] = error_sample_pb2_L2norm(mYCurr, mY, Nh, Ntau, Nh, Ntau, T, FEMOrder, BDFOrder) / epsCurr
    print("d=", dCurr, "Dm ", r)
    return r  # np.reshape(r, (1,-1))

# parallel
dd = np.linspace(0,D-1,D).astype(int)
pool = Pool(D)
dm = pool.map(compute_row, dd)
dm = np.array(dm)

# non parallel
# for dCurr in range(D):
#     dm[dCurr, :] = compute_row(dCurr)

np.savetxt('derivatives_paramM.csv', dm, delimiter=',')


# for dCurr in range(D):
#     plt.semilogx(ee, dm[dCurr, :], '.-', label="d="+str(dCurr))
#     plt.legend()
#     plt.pause(0.05)
# plt.show()


ll = 2**(0.5*np.ceil(np.log2(np.linspace(1,D,D))))
plt.loglog(dm[:,-1])
plt.loglog(np.power(ll, -3))
plt.show()
