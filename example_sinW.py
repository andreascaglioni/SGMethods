import numpy as np
from math import sqrt, factorial, log2, pi
from multiprocessing import Pool
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion


NRNDSamples = 100
maxNumNodes = 450
p=2
interpolationType =  "linear"


Nt = 100
tt = np.linspace(0, 1, Nt)
dt = 1/Nt

def F(x):
    return np.sin(param_LC_Brownian_motion(tt, x, 1))

def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    spaceNorm = lambda x : sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)  # L2 norm in time 
    errSample = spaceNorm(uExa-Iu)
    return sqrt(np.mean(np.square(errSample)))

np.random.seed(1607)
yyRnd = np.random.normal(0, 1, [NRNDSamples, 2**10])  # infinite paramter vector
print("Parallel random sampling")
NParallel = 1
# pool = Pool(NParallel)
# uExa = np.array(pool.map(F, yyRnd))

# version wo parallel for profiling
dimF = F(yyRnd[0,:]).size
uExa = np.zeros((NRNDSamples, dimF))
for i in range(NRNDSamples):
    uExa[i,:] = F(yyRnd[i,:])


lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p=p)

# Profits
def Profit(nu):
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims))))
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)
    # c = sqrt(pi**p * 0.5**(-(p+1)) * (2*p+1)**(0.5*(2*p-1)) ) * np.sqrt(1 + (2**(2*p+2)-4)/(2**(nu+1)))
    # C1 = 1+ c * 3**(-p)/factorial(p)
    # C2 = c * (1+2**(-p))/factorial(p)
    C1=1
    C2=1
    v1 = np.prod(C1/repRho, axis=1, where=(nu==1))
    v2 = np.prod(C2*np.power(2**nu * repRho, -p) ,axis=1, where=(nu>1))
    w = np.prod((2**(nu+1)-2)*(p-1)+1 ,axis=1)
    return v1 * v2 / w


def Profit2(nu):
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims))))
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)

    v = np.prod(np.power(2**nu * repRho, -p) ,axis=1, where=(nu>0))
    w = np.prod((2**(nu+1)-2)*(p-1)+1 ,axis=1)
    return v/w

def convergenceTest(ProfitFun):
    # convergence test
    err = np.array([])
    nNodes = np.array([])
    nDims = np.array([])
    w=0
    I = midSet()
    oldSG = None
    uOnSG = None
    while True:
        print("Computing w  = ", w)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=1)
        if(interpolant.numNodes > maxNumNodes):
            break
        print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
        uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
        uInterp = interpolant.interpolate(yyRnd, uOnSG)
        # ERROR
        err = np.append(err, computeL2Error(uExa, uInterp))
        print("Error:", err[-1])
        nNodes = np.append(nNodes, interpolant.numNodes)
        nDims = np.append(nDims, interpolant.N)
        midSetMaxDim = np.amax(I.midSet, axis=0)

        oldSG = interpolant.SG
        # update midset for next iteration
        # P = np.apply_along_axis(ProfitFun, 1, I.margin)
        P = ProfitFun(I.margin)
        idMax = np.argmax(P)

        # print("new Mid", I.margin[idMax, :])
        # I.update(idMax)
        ncps = interpolant.numNodes
        while interpolant.numNodes < sqrt(2)*ncps:
            P = ProfitFun(I.margin)
            idMax = np.argmax(P)
            # print("new Mid", I.margin[idMax, :])
            I.update(idMax)
            interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
            if(interpolant.numNodes > maxNumNodes):
                break
        w+=1
    
    print(midSetMaxDim)

    # tt = np.linspace(0, 1, Nt)
    # plt.plot(tt, uExa[0,:], '.-', tt, uInterp[0,:], '.-')
    # plt.show()

    print(err)
    rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
    print("Rate:",  rates)
    plt.loglog(nNodes, err, '.-')
    plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
    plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')

convergenceTest(Profit2)

plt.show()