import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from SGMethods.MidSets import  SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant
import scipy.io
from numpy.linalg import lstsq

from SLLG.sample_LLG_function_noise import sample_LLG_function_noise
from SLLG.error_sample import error_sample

from multiprocessing import Pool


"""Probe the anisotropy of param to sol map of SLLG
We fix 1 at time the parameters. We do Lagrange interpolation. The convergence rate gives an idea of the corresponding regularity.
This is just heuristic!"""

################## parameters for the anisotropy test ##################
N = 16
FEMOrder = 1
BDFOrder = 1
T = 1
Nh = 32
NTau = Nh * 2

maxNumNodes = 15
NRNDSamples = 1024
NPrallelPool = 32

def physicalError(u, uExa):
    return error_sample(u, uExa, Nh, NTau, Nh, NTau, T, FEMOrder, BDFOrder, False)

def F(y):
    return sample_LLG_function_noise(y, Nh, NTau, T, FEMOrder, BDFOrder)
dimF = F([0]).size


################## the rest should not be edited ##################

def bindedVector(N, p, x):
    """Returns a np vector of length N of zeros except in position p, where it takes the value x"""
    v = np.zeros(N)
    v[p] = x
    return v

def ConvergenceTest1D(maxNumNodes, dimF, dimTest):
    xxRND = np.random.normal(0, 1, NRNDSamples)
    xxRNDVec = np.zeros((xxRND.size, N))
    xxRNDVec[:, dimTest] = xxRND
    uExa = np.zeros((NRNDSamples, dimF))
    print("Parallel random sampling")
    pool = Pool(NPrallelPool)
    uExa = pool.map(F, xxRNDVec)

    # for w in range(NRNDSamples):
    #     uExa[w] = FSample(xxRNDVec[w])
    
    ww = np.exp(- np.square(xxRND))
    lev2knots = lambda n: n+1
    mat = scipy.io.loadmat('SGMethods/knots_weighted_leja_2.mat')
    wLejaArray = np.ndarray.flatten(mat['X'])
    knots = lambda n : wLejaArray[0:n:]
    err = np.array([])
    w=0
    F1D = lambda x : F(bindedVector(N, dimTest, x))
    while True:
        print("Computing with ", w+1, "nodes" )
        I = SmolyakMidSet(w, 1)
        interpolant = SGInterpolant(I, knots, lev2knots)
        if(interpolant.numNodes>maxNumNodes):
            break
        if w ==0 :
            oldSG = None
            FOnSG = None
        FOnSG = interpolant.sampleOnSG(F1D, dimF, oldSG, FOnSG)
        uInterp = interpolant.interpolate(xxRND, FOnSG)
        errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
        err = np.append(err, np.amax(np.multiply(errSamples,  ww)))
        if(err[-1]<1.e-14):
            break
        oldSG = interpolant.SG
        w+=1
    return err

def fitToExponential(x, y):
    y = np.log(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    expC = -m
    multC = np.exp(c)
    return multC, expC

errMat = []
for n in range(N):
    print("Testing direction", n)
    # Fcurr = lambda x: F(bindedVector(N,n,x))
    err = ConvergenceTest1D(maxNumNodes, dimF, n)
    nNodes = np.linspace(1, err.size, err.size)
    print("Error: ", err)
    rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
    print("Rate:",  rates)
    plt.loglog(nNodes, err, '.-', label=str(n))
    multC, expC = fitToExponential(nNodes, err)
    xx=np.linspace(nNodes[0], nNodes[-1])
    ss = "e="+ ('%.2f' % expC ) +" C=" + ('%.2f' % multC)
    plt.loglog(xx, multC*np.exp(-expC*xx), '-k', label=ss)
    errMat.append(list(err))
    np.savetxt('probe_SLLG_regularity_parameters_Nh32.csv',np.array(errMat))

plt.legend()
plt.show()
