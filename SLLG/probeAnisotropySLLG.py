import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from SGMethods.MidSets import  SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant
import scipy.io
from numpy.linalg import lstsq



"""Probe the anisotropy of a function of N parameters by 1D interpolation.
We fix 1 at time the parameters. We do Lagrange interpolation. The convergence rate gives an idea of the corresponding regularity.
This is just heuristic!"""

################## parameters for the anisotropy test ##################
N = 3
dimF = 3
F = lambda x: np.sin(x[0]+0.1*x[1]+0.01*x[2]) * np.array([1., 2., 3.])
maxNumNodes = 20

def physicalError(u, uExa):
    return np.linalg.norm(u - uExa, ord=2, axis=1)

################## the rest should not be edited ##################

def bindedVector(N, p, x):
    v = np.zeros(N)
    v[p] = x
    return v

def ConvergenceTest1D(maxNumNodes, F, dimF):
    NRNDSamples = 1000
    xxRND = np.random.normal(0, 1, NRNDSamples)
    uExa = np.zeros((NRNDSamples, dimF))
    for w in range(NRNDSamples):
        uExa[w] = F(xxRND[w])
    ww = np.exp(- np.square(xxRND))
    lev2knots = lambda n: n+1
    mat = scipy.io.loadmat('SGMethods/knots_weighted_leja_2.mat')
    wLejaArray = np.ndarray.flatten(mat['X'])
    knots = lambda n : wLejaArray[0:n:]
    err = np.array([])
    w=0
    while True:
        print("Computing n = ", w)
        I = SmolyakMidSet(w, 1)
        interpolant = SGInterpolant(I, knots, lev2knots)
        SG = interpolant.SG
        if(interpolant.numNodes>maxNumNodes):
            break
        if w ==0 :
            oldSG = None
            FOnSG = None
        FOnSG = interpolant.sampleOnSG(F, dimF, oldSG, FOnSG)
        uInterp = interpolant.interpolate(xxRND, FOnSG)
        err = np.append(err, np.amax(physicalError(uInterp, uExa) * ww))
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


for n in range(N):
    Fcurr = lambda x: F(bindedVector(N,n,x))
    err = ConvergenceTest1D(maxNumNodes, Fcurr, dimF)
    nNodes = np.linspace(1, err.size, err.size)


    print("Error: ", err)
    rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
    print("Rate:",  rates)
    plt.loglog(nNodes, err, '.-', label=str(n))
    multC, expC = fitToExponential(nNodes, err)
    xx=np.linspace(nNodes[0], nNodes[-1])
    ss = "e="+ ('%.2f' % expC ) +" C=" + ('%.2f' % multC)
    plt.loglog(xx, multC*np.exp(-expC*xx), '-k', label=ss)

plt.legend()
plt.show()
