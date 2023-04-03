import numpy as np
from math import sqrt, factorial, log2
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion





NRNDSamples = 128
maxNumNodes = 500
p=2
interpolationType =  "linear"

Nt = 1.e1
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    W = param_LC_Brownian_motion(tt, x, 1)
    return np.sin(W)

lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p=p)

def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))
NLCExpansion = 2**10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
print("Parallel random sampling")
NParallel = 16
pool = Pool(NParallel)
uExa = np.array(pool.map(F, yyRnd))
dimF = uExa[0].size

beta=0.5
Cbar = 2
# log(P) compared to theory

def ProfitFun(x):
    logRho = beta*np.ceil(np.log2(np.linspace(1,x.size,x.size)))
    C1 = - np.sum(x)  # work^-1
    C2 = np.sum(-logRho, where=(x==1))  # value for components =1
    C3 = np.sum(log2(Cbar) - log2(factorial(p)) - p*x - p*logRho, where=(x>1))  # value for other components
    return C1 + C2 + C3

def ProfitFunOld(x):
    logRho = beta*np.ceil(np.log2(np.linspace(1,x.size,x.size)))
    C1 = - np.sum(x)  # work^-1
    C2 = np.sum(log2(Cbar) -log2(factorial(p)) - p*x - p*logRho, where=(x>=1))
    return C1 + C2

def convergenceTest(ProfitFun):
    # convergence test
    err = np.array([])
    nNodes = np.array([])
    w=0
    I = midSet()
    oldSG = None
    uOnSG = None
    while True:
        print("Computing w  = ", w)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        if(interpolant.numNodes > maxNumNodes):
            break
        print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
        uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
        uInterp = interpolant.interpolate(yyRnd, uOnSG)
        # ERROR
        errCurr = computeL2Error(uExa, uInterp)
        print("Error:", errCurr)
        err = np.append(err, errCurr)
        nNodes = np.append(nNodes, interpolant.numNodes)
        oldSG = interpolant.SG
        # update midset for next iteration
        P = np.apply_along_axis(ProfitFun, 1, I.margin)
        idMax = np.argmax(P)
        # print("new Mid", I.margin[idMax, :])
        I.update(idMax)
        # ncps = interpolant.numNodes
        # while interpolant.numNodes < sqrt(2)*ncps:
        #     P = np.apply_along_axis(ProfitFun, 1, I.margin)
        #     idMax = np.argmin(P)
        #     # print("new Mid", I.margin[idMax, :])
        #     I.update(idMax)
        #     interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        w+=1
    print(I.midSet)
    print(np.amax(I.midSet, axis=0))

    # tt = np.linspace(0, 1, Nt)
    # plt.plot(tt, uExa[0,:], '.-', tt, uInterp[0,:], '.-')
    # plt.show()

    print(err)
    rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
    print("Rate:",  rates)
    plt.loglog(nNodes, err, '.-')
    plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
    plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')

convergenceTest(ProfitFun)
convergenceTest(ProfitFunOld)
plt.show()