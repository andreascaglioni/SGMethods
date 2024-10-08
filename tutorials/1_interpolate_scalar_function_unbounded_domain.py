import numpy as np 
from math import sin, sqrt
from scipy.interpolate import RegularGridInterpolator
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from sgmethods.nodes_1d import cc_nodes
from sgmethods.multi_index_sets import aniso_smolyak_mid_set
from sgmethods.tp_inteprolants import TPPwLinearInterpolator
from sgmethods.sparse_grid_interpolant import SGInterpolant


""" SGMethods Tutorial 0: Interpolate a scalar function.
We use SGMethods to interpolate a function F : [-1,1]^N rightarrow R.
The domain has dimension N in N, possibly high.
"""

# High-dimensional function
weightFun = lambda N : np.power(np.linspace(1, N, N), -2)  # Weight function
def F(y):
    weightCurr = weightFun(y.size)
    return sin(np.dot(weightCurr, y))

# Sparse grid parameters. 
p = 2  # Polynomial degree of the interpolant is p-1 ine ach parameter
lev2knots = lambda nu: np.where(nu==0, 1, 2**(nu-1)+1)  # Level-to-knots function
knots = lambda n : cc_nodes(n)  # Nodes family

# Error computation
def computeL2Error(uExa, Iu):
    assert(uExa.shape[0] == Iu.shape[0])
    return sqrt(np.mean(np.square(uExa - Iu)))
yyRnd = np.random.normal(0, 1, [256, 1000])  # 256 random sample of (effectively) infinite parameter vector
uExa = np.array(list(map(F, yyRnd)))  # Random sample of exact function

# Sparse grid interpolation
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))
midSet = aniso_smolyak_mid_set(w=5, N=10, a=anisoVec(10))  # Anisotropic Smolyak multi-index set in 10 dimensions
interpolant = SGInterpolant(midSet, knots, lev2knots, TPInterpolant=TPInterpolant)
uOnSG = interpolant.sampleOnSG(F)  # Sample the function on the sparse grid
uInterp = interpolant.interpolate(yyRnd, uOnSG)  # Compute the interpolated value
error = computeL2Error(uExa, uInterp)
