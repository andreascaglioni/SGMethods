""" We use SGMethods to interpolate a function with an infinite-dimensional 
domain.
"""

from math import sin, sqrt
import sys
# import os
import numpy as np
from sgmethods.nodes_1d import unbounded_nodes_nested
from sgmethods.multi_index_sets import aniso_smolyak_mid_set
from sgmethods.tp_inteprolants import TPPwLinearInterpolator
from sgmethods.sparse_grid_interpolant import SGInterpolant

sys.path.insert(1, 'sgmethods')

# Weight function
def weight_fun(N):
    """Assign different importance to variables in  following function."""
    np.power(np.linspace(1, N, N), -2)


def f(y):
    """High-dimensional function."""
    return sin(np.dot(weight_fun(y.size), y))

# Sparse grid parameters
p = 2  # Polynomial degree + 1
lev2knots = lambda n: 2**(n+1)-1  # Level-to-knots function

# Error computation
def compute_L2_error(exact_u, interpol_u):
    """ Compute the L2 error between the exact and the interpolated function."""
    assert exact_u.shape[0] == interpol_u.shape[0]
    return sqrt(np.mean(np.square(exact_u - interpol_u)))

# Generate 256 random sample of (effectively) infinite parameter vector
yy_rnd = np.random.normal(0, 1, [256, 1000])
uExa = np.array(list(map(f, yy_rnd)))  # Random sample of exact function

# Sparse grid interpolation
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))

# Anisotropic Smolyak multi-index set in 10 dimensions
mid_set = aniso_smolyak_mid_set(w=5, N=10, a=anisoVec(10))  

# Define the sparse grid interpolant
interpolant = SGInterpolant(mid_set, unbounded_nodes_nested, lev2knots,\
                            tp_interpolant=TPPwLinearInterpolator)

uOnSG = interpolant.sample_on_SG(f)  # Sample the function on the sparse grid

u_interpol = interpolant.interpolate(yy_rnd, uOnSG)  # Interpolate on rnd sample

error = compute_L2_error(uExa, u_interpol)
print(error)
