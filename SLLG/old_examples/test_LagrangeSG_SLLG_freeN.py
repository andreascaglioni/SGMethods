from __future__ import division
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from SGMethods.MidSets import anisoSmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant
import scipy.io
from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample
from multiprocessing import Pool

"""We test the sparse grid interpolant on SLLG. 
The convergence of 2 SG interpolants are compared:
1. anisotrpoic midset with rho_n = 2^n
2. isotropic midset"""

FEMOrder = 1
BDFOrder = 1
T = 1
Nh = 32
NTau = Nh * 2
NParallel = 32
NRNDSamples = 32
maxNumNodes = 500

def F(x):
    return sample_LLG_function_noise(x, Nh, NTau, T, FEMOrder, BDFOrder)
def physicalError(u, uExa):
    return error_sample(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder, False)

dimF = F([0]).size

# choose interpolant
lev2knots = lambda n: n+1
mat = scipy.io.loadmat('SGMethods/knots_weighted_leja_2.mat')
wLejaArray = np.ndarray.flatten(mat['X'])
knots = lambda n : wLejaArray[0:n:]

# error computations
NLCExpansion = 2**10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
# ww = np.exp(- np.square(np.linalg.norm(yyRnd, 2, axis=1)))
uExa = np.zeros((NRNDSamples, dimF))
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)

##################### convergence test ANISOTROPIC
anisoVector = lambda L: np.append(1, 2**(0.5*np.floor(np.log2(np.linspace(1,L,L)))))
beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))

err = np.array([])
nNodes = np.array([])
w=0
I = midSet()
while(True):
    print("Computing w  = ", w)
    I = anisoSmolyakMidSet(w*min(anisoVector), N, anisoVector)
    interpolant = SGInterpolant(I, knots, lev2knots, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    print(interpolant.numNodes, "nodes")
    if w ==0 :
        oldSG = None
        uOnSG = None
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
    errCurr = np.amax(np.multiply(errSamples,  ww))
    err = np.append(err, errCurr)
    nNodes = np.append(nNodes, interpolant.numNodes)
    print("Error:", err[w])
    np.savetxt('convergenge_SLLG_aniso.csv',np.array([nNodes, err]))
    oldSG = interpolant.SG
    w+=1
print("Error:", err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', nNodes, 1.e-5/nNodes, '-k')