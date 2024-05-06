import numpy as np 
from math import sin, sqrt
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import anisoSmolyakMidSet, midSet
from SGMethods.SGInterpolant import SGInterpolant

"""Tutorial on sparse grid interpolation with SGMethods. 
We interpolate an infiite-dimensional (domain) parametetric map"""

# the parametric function to interpolate
weightFun = lambda N : np.power(np.linspace(1, N, N), -2)
def F(y):
    weightCUrr = weightFun(y.size)
    return sin(np.dot(weightCUrr, y))

# Numerics parameters
p=2  # polynomial degree
interpolationType =  "linear"
lev2knots = lambda n: 2**(n+1)-1
knots = lambda n : unboundedKnotsNested(n, p)

# Error computation
def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    return sqrt(np.mean(np.square(uExa - Iu)))
yyRnd = np.random.normal(0, 1, [256, 1000])  # 256 random parameters of (effectively) infinite paramter vector
uExa = np.array(list(map(F, yyRnd)))  # Random sample of exact function

# define the sparse grid interpolant
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))
midSet = anisoSmolyakMidSet(w=5, N=10, a=anisoVec(10))  # anisotropic Smolyak multi-index set
interpolant = SGInterpolant(midSet, knots, lev2knots, interpolationType=interpolationType)
uOnSG = interpolant.sampleOnSG(F)
uInterp = interpolant.interpolate(yyRnd, uOnSG)
error = computeL2Error(uExa, uInterp)
print("Error:", error)


############################################################################################################
# Convergence test
maxNumNodes = 128
err = np.array([])
nNodes = np.array([])
nDims = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
Profit = lambda nu : 1/np.product(nu+1)
while True:
    print("Computing w  = ", w)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType)  # define spoarse gridinterpolant 
    if(interpolant.numNodes >= maxNumNodes):  # check termination condition 
        break
    print("# nodes:", interpolant.numNodes, "Number effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, oldXx=oldSG, oldSamples=uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # Compute error
    err = np.append(err, computeL2Error(uExa, uInterp))
    print("Error:", err[-1])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    oldSG = interpolant.SG
    ncps = interpolant.numNodes
    # Refine the sparse grid by enriching the multi-index set. Use a profit-based criterion
    while interpolant.numNodes < sqrt(2)*ncps:
        P = Profit(I.margin)  # compute profit on margin
        idMax = np.argmax(P)  # find profit maximizer
        I.update(idMax)  # include it in multi-index set
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType)  # update sparse grid interpolant
        if(interpolant.numNodes > maxNumNodes):  # refinement termination condition
            break
    w+=1
# Sumamrize results
print("Errors: ", err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rates of convergence:",  rates)
plt.loglog(nNodes, err, '.-')
plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')
plt.show()



