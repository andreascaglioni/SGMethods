import numpy as np
from multiprocessing import Pool
from dolfin import *
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.error_sample import error_sample
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise


T = 1
FEMOrder = 1
BDFOrder = 1
Nh = 32 # 8 # 
NTau = Nh * 4
NRNDSamples = 64 # 4 # 
NParallel = 16
def F(x):
    return sample_LLG_function_noise(x, Nh, NTau, T, FEMOrder, BDFOrder)
def physicalError(u, uExa):
    return error_sample(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder, False)

# error computations
NLCExpansion = 2**10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)
dimF = uExa[0].size

# CONVEGENCE TRUNCATION
err = np.array([])
ll = np.linspace(0,6,7, dtype=int)
nn = 2**ll
for k in range(nn.size):
    print("Computing k  = ", k)
    nCurr = nn[k]
    yyRndTruncation = yyRnd[:, 0:nCurr]
    pool = Pool(NParallel)
    uTruncation = pool.map(F, yyRndTruncation)
    errSamples = np.array([ physicalError(uTruncation[i], uExa[i]) for i in range(NRNDSamples) ])
    errCurr = sqrt(np.mean(np.square(errSamples)))
    print("error:", errCurr)
    err = np.append(err, errCurr)
    np.savetxt('convergenge_SLLG_truncation.csv',np.array([nn[0:k+1], err]))

print(err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nn[1::]/nn[:-1:])
print("Rate:",  rates)
plt.loglog(nn, err, '.-')
plt.plot()