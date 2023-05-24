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
maxNumNodes = 1000
p=2
interpolationType =  "linear"

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
    c = sqrt(pi**p * 0.5**(-(p+1)) * (2*p+1)**(0.5*(2*p-1)) ) * np.sqrt(1 + (2**(2*p+2)-4)/(2**(nu+1)))
    C1 = 1+ c * 3**(-p)/factorial(p)
    C2 = c * (1+2**(-p))/factorial(p)
    v1 = np.prod(C1/repRho, axis=1, where=(nu==1))
    v2 = np.prod(C2*np.power(2**nu * repRho, -p) ,axis=1, where=(nu>1))
    w = np.prod((2**(nu+1)-2)*(p-1)+1 ,axis=1)
    return v1 * v2 / w

# convergence test
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
    
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, interpolant.N)
    P = Profit(I.margin)
    idMax = np.argmax(P)
    I.update(idMax)
    w+=1

for tau in np.linspace(0.4, 0.7, 10):
    CTau = 2**(nDims**(1-0.5*tau))
    plt.loglog(nNodes, CTau * nNodes**(1-1/tau), label=str(tau))
plt.loglog(nNodes, 1/nNodes, 'k-', label="n^{-1}")
plt.legend()
plt.show()