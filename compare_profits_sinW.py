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


NRNDSamples = 100
maxNumNodes = 300
p=2
interpolationType =  "linear"

Nt = 10
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    return np.sin(param_LC_Brownian_motion(tt, x, 1))

lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p=p)

def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)  # L2 norm in time
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))

yyRnd = np.random.normal(0, 1, [NRNDSamples, 2**10])  # infitnite paramter vector
print("Parallel random sampling")
NParallel = 1
# pool = Pool(NParallel)
# uExa = np.array(pool.map(F, yyRnd))

# version wo parallel for profiling
dimF = F(yyRnd[0,:]).size
uExa = np.zeros((NRNDSamples, dimF))
for i in range(NRNDSamples):
    uExa[i,:] = F(yyRnd[i,:])


# Profits
def Profit(nu):
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims))))
    rhoReshape = np.reshape(rho, (1, -1))
    repRho = np.repeat(rhoReshape, nMids, axis=0)
    M = 2**nu*repRho
    return np.power(np.prod(p*M, axis=1, where=(nu==1)), -1) * np.power(np.prod(M, axis=1, where=(nu>1)), -p)

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
        err = np.append(err, computeL2Error(uExa, uInterp))
        print("Error:", err[-1])
        nNodes = np.append(nNodes, interpolant.numNodes)
        oldSG = interpolant.SG
        # update midset for next iteration
        # P = np.apply_along_axis(ProfitFun, 1, I.margin)
        P = ProfitFun(I.margin)
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
    # print(I.midSet)
    # print(np.amax(I.midSet, axis=0))

    # tt = np.linspace(0, 1, Nt)
    # plt.plot(tt, uExa[0,:], '.-', tt, uInterp[0,:], '.-')
    # plt.show()

    print(err)
    rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
    print("Rate:",  rates)
    plt.loglog(nNodes, err, '.-')
    plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
    plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')

convergenceTest(Profit)
# convergenceTest(ProfitFunOld)
plt.show()