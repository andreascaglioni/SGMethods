""" Interpolate a scalar function with an infinite-dimensional bounded domain
using SGMthods and estimate the uniform (:math:`L^{\infty}`) error with the
Monte Carlo method.
We interpolate using Lagrange interpolation on Clenshaw-Curtis nodes.
"""

from math import sin
import numpy as np
import sys, os 

# NB to import from sgmethods modules, we add the code directory to the path.
# If the project is given as a package, this is not necessary.
sys.path.insert(0, os.path.abspath('./'))

from sgmethods.nodes_1d import cc_nodes
from sgmethods.multi_index_sets import aniso_smolyak_mid_set
from sgmethods.tp_interpolants import TPLagrangeInterpolator
from sgmethods.sparse_grid_interpolant import SGInterpolant


# 1 PROBLEM DEFINITION
# Define a "weight vector" to assign different importance to different scalar
# variables.
# Since we approximate a function of a sequence (and not a finite dimensional
# vector), the weight vector should be a sequence itself. To implement this, we
# define it though its finite-dimensiaonal truncations.
weight_fun = lambda N : np.power(np.linspace(1, N, N), -2)
# Define the function to be approximated
f = lambda y : sin(np.dot(weight_fun(y.size), y))

# 2 SPARSE GRID INTERPOLATION PARAMETERS
lev2knots = lambda n: np.where(0, 1, 2**n+1)  # Level-to-knots function
# Anisotropic Smolyak multi-index set in 10 dimensions
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))  # to define multi-index set
N = 8  # Number approximated scalar parameters
mid_set = aniso_smolyak_mid_set(w=4, N=N, a=anisoVec(N))

# 3 PRE-PROCESSING FOR ERROR COMPUTATION
# Measure the error in in the uniform (i.e. L^{infty}) norm, similar to the
# maximum differente between the exact and the interpolated function.
def compute_uniform_error(exact_u, interpol_u):
    assert exact_u.shape[0] == interpol_u.shape[0]
    return np.amax(np.abs(exact_u - interpol_u))
# Generate random sample of (effectively) infinite parameter vector.
yy_rnd = np.random.uniform(-1, 1, [256, 1000])
# Take a random sample of the exact function
u_exa = np.array(list(map(f, yy_rnd)))

# 4 SPARSE GRID INTERPOLATION
# Construct the sparse grid interpolant with the data defiend above. The
# interpolation method is Lagernace, i.e. global polynomials. This means that
# the polynomial degree increases with the number of nodes.
interpolant = SGInterpolant(mid_set, cc_nodes, lev2knots,\
                            tp_interpolant=TPLagrangeInterpolator)

# Sample the function to approxiomate on the sparse grid.
uOnSG = interpolant.sample_on_SG(f)

 # Interpolate the data on the random sample (the same used in 3) by sampling
 # the interpolant on it.
u_interpol = interpolant.interpolate(yy_rnd, uOnSG)

# Finally approximately compute the error.
error = compute_uniform_error(u_exa, u_interpol)
print(error)
