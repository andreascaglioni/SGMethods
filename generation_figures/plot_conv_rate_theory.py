import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, factorial, pi
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion

maxNumNodes = 2000
NParallel = 1
p=2
interpolationType =  "linear"
lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p=p)

def Profit(nu):
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims))))
    rhoReshape = np.reshape(rho, (1, -1))
    repRho = np.repeat(rhoReshape, nMids, axis=0)
    c = sqrt(pi**p * 0.5**(-(p+1)) * (2*p+1)**(0.5*(2*p-1)) ) * np.sqrt(1 + (2**(2*p+2)-4)/(2**(nu+1)))
    C1 = 1+ c * 3**(-p)/factorial(p)
    C2 = c * (1+2**(-p))/factorial(p)
    v1 = np.prod(C1/repRho, axis=1, where=(nu==1))
    v2 = np.prod(C2*np.power(2**nu*repRho, -p) ,axis=1, where=(nu>1))
    w = np.prod((2**(nu+1)-2)*(p-1)+1 ,axis=1)
    return v1 * v2 / w


nNodes = np.array([])
nDims = np.array([])
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
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, interpolant.N)
    oldSG = interpolant.SG
    P = Profit(I.margin)
    idMax = np.argmax(P)
    I.update(idMax)

for tau in np.linspace(0.1, 1, 10):
        CTau = np.exp(1/tau * (1-nDims**(1-0.5*tau))/(1-2**(1-0.5*tau)))
        plt.loglog(nNodes, CTau * nNodes**(1-1/tau), label="tau="+str(tau))
plt.legend()    
plt.show()