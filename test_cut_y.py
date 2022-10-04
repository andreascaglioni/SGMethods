


from __future__ import division
from signal import Sigmasks
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
from multiprocessing import Pool
from math import log, pi
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant


FEMOrder = 1
BDFOrder = 1
T = 1
Nh = 16
NTau = Nh * 2
def F(x):
    return sample_LLG_function_noise(x, Nh, NTau, T, FEMOrder, BDFOrder)
def physicalError(u, uExa):
    return error_sample(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder, False)



levMax = 6
NLCExpansion = 2**levMax
yy = np.random.normal(0, 1, NLCExpansion)
print(yy)
exa = F(yy)
err=[]
for l in range(levMax):
    ycurr = yy[0:2**l:]
    print(ycurr)
    u = F(ycurr)
    err.append(physicalError(u, exa))
    print(err[l])
print(err)
