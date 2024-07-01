import numpy as np 
from math import sin, sqrt
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import anisoSmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant

"""Tutorial on sparse grid interpolation with SGMethods. 
We see how to use SGMethods to  interpolate an infinite-dimensional (domain) parametetric map.
We first carry out a single interpolation, then a convergence test."""

# High-dimensional function
weightFun = lambda N : np.power(np.linspace(1, N, N), -2)  # Weight function
def F(y):
    weightCurr = weightFun(y.size)
    return sin(np.dot(weightCurr, y))

# Sparse grid parameters
p = 2  # Polynomial degree + 1 
TPInterpolant = (lambda nodesTuple, fOnNodes : 
                 RegularGridInterpolator(nodesTuple, fOnNodes, method='linear', bounds_error=False, fill_value=None))
lev2knots = lambda n: 2**(n+1)-1  # Level-to-knots function
knots = lambda n : unboundedKnotsNested(n, p)  # Nodes family

# Error computation
def computeL2Error(uExa, Iu):
    assert(uExa.shape == Iu.shape)
    return sqrt(np.mean(np.square(uExa - Iu)))
yyRnd = np.random.normal(0, 1, [256, 1000])  # 256 random sample of (effectively) infinite parameter vector
uExa = np.array(list(map(F, yyRnd)))  # Random sample of exact function

# Sparse grid interpolation
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))
midSet = anisoSmolyakMidSet(w=10, N=10, a=anisoVec(10))  # Anisotropic Smolyak multi-index set in 10 dimensions
interpolant = SGInterpolant(midSet, knots, lev2knots, TPInterpolant=TPInterpolant)
uOnSG = interpolant.sampleOnSG(F)  # Sample the function on the sparse grid
uInterp = interpolant.interpolate(yyRnd, uOnSG)  # Compute the interpolated value
error = computeL2Error(uExa, uInterp)
print("Error:", error)