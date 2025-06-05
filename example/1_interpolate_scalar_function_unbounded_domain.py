""" Interpolate a function with an infinite-dimensional and non-compact domain
with SGMthods and estimate the L2 error with the Monte Carlo method.
Unlike the preivous example, it does not make sense to approximate the error in
the uniform norm. Rather, we use the L^2_{mu} norm, i.e. a root intergral square
norm where the integrla is weighted with a Gaussian measure. This makes the 
error function (difference between exact function and interpolation) less and
less important away from the origin.
We interpolate using piecewise linear interpolation on a suitable nodes family.
"""

from math import sin, sqrt
import numpy as np
import sys, os

# NB to import from sgmethods modules, we add the code directory to the path.
# If the project is given as a package, this is not necessary.
sys.path.insert(0, os.path.abspath('./'))

from sgmethods.nodes_1d import opt_guass_nodes_nest
from sgmethods.multi_index_sets import aniso_smolyak_mid_set
from sgmethods.tp_interpolants import TPPwLinearInterpolator
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
p = 2  # Polynomial degree + 1
lev2knots = lambda n: 2**(n+1)-1  # Level-to-knots function
# Anisotropic Smolyak multi-index set in 10 dimensions
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))  # to define multi-index set
mid_set = aniso_smolyak_mid_set(w=5, N=10, a=anisoVec(10))

# 3 PRE-PROCESSING FOR ERROR COMPUTATION
# Measure the error in in the L2 norm, similar to the root mean square error.
def compute_L2_error(exact_u, interpol_u):
    assert exact_u.shape[0] == interpol_u.shape[0]
    return sqrt(np.mean(np.square(exact_u - interpol_u)))
# Generate random sample of (effectively) infinite parameter vector.
yy_rnd = np.random.normal(0, 1, [256, 1000])
# Take a random sample of the exact function
uExa = np.array(list(map(f, yy_rnd)))

# 4 SPARSE GRID INTERPOLATION
# Construct the sparse grid interpolant with the data defiend above. The
# interpolation method is piecewise linear.
interpolant = SGInterpolant(mid_set, opt_guass_nodes_nest, lev2knots,\
                            tp_interpolant=TPPwLinearInterpolator)

# Sample the function to approxiomate on the sparse grid.
uOnSG = interpolant.sample_on_SG(f)

 # Interpolate the data on the random sample (the same used in 3) by sampling
 # the interpolant on it.
u_interpol = interpolant.interpolate(yy_rnd, uOnSG)

# Finally approximately compute the error.
error = compute_L2_error(uExa, u_interpol)
print(error)
