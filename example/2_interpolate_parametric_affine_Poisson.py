""" We consider a parametric Poisson problem where the parameter is introduced
through the diffusion coefficient, a function of the space variable and an
affine function of a parametric vector.
We use SGMethods to approximate the parameter-to-solution map 
:math:`y \mpasto u(y)`,
where :math:`u(y)` is the solution of the Poisson problem for parameter 
:math:`y`.
We use the Fenics finite elements library to approximate the solution of the
Poisson problem for fixed parameters and compute the approximation error of
these sampels.
"""

from math import pi
import numpy as np
import sys, os

# NB to import from sgmethods modules, we add the code directory to the path.
# If the project is given as a package, this is not necessary.
sys.path.insert(0, os.path.abspath('./'))

from sgmethods.nodes_1d import cc_nodes
from sgmethods.multi_index_sets import aniso_smolyak_mid_set
from sgmethods.tp_inteprolants import TPLagrangeInterpolator
from sgmethods.sparse_grid_interpolant import SGInterpolant

# Import functions for sampleing the parameter-to-solution map and computing
# the error.
from parametric_affine_poisson import sample_parametric_poisson,\
    compute_error_samples_poisson


# 1 PROBLEM DEFINITION
n_h = 16  # Mesh resolution

order_decay_coeffs = 2
def f(y):
    sample_parametric_poisson(y, n_h, order_decay_coeffs)

# 2 SPARSE GRID INTERPOLATION PARAMETERS
lev2knots = lambda n: np.where(0, 1, 2**n+1)  # Level-to-knots function
# Anisotropic Smolyak multi-index set in 10 dimensions
anisoVec = lambda N : 2**(np.linspace(0, N-1, N))  # to define multi-index set
# Define the anisotrpic multi-index set needed for good (i.e.
# dimensiona-independent) approximation.
N = 8  # Number approximated scalar parameters
C = pi**2/6 * 1.1  # normalization diffusion coefficient
a_min = 1 - -pi**2/(6*C)  # minimum diffusion coefficient
gamma = lambda N : 1/(C*a_min*np.linspace(1, N, N)**order_decay_coeffs)
tau = lambda N : 1/(2*gamma(N))
anisoVec = lambda N : 0.5* np.log(1+tau(N))  
mid_set = aniso_smolyak_mid_set(w=3, N=N, a=anisoVec(N))

# 3 PRE-PROCESSING FOR ERROR COMPUTATION
# Measure the error in in the uniform (i.e. L^{infty}) norm, similar to the
# maximum differente between the exact and the interpolated function.
def compute_uniform_error(exact_u, interpol_u):
    assert exact_u.shape[0] == interpol_u.shape[0]
    return np.amax(np.abs(exact_u - interpol_u))
# Generate random sample of (effectively) infinite parameter vector.
yy_rnd = np.random.uniform(-1, 1, [128, 1000])
# Take a random sample of the exact function
u_exa = np.array(list(map(f, yy_rnd)))

# 4 SPARSE GRID INTERPOLATION
# Construct the sparse grid interpolant with the data defiend above. The
# interpolation method is Lagernace, i.e. global polynomials. This means that
# the polynomial degree increases with the number of nodes.
print("Sparse grid interpolation:")
interpolant = SGInterpolant(mid_set, cc_nodes, lev2knots,\
                            tp_interpolant=TPLagrangeInterpolator)

# Sample the function to approxiomate on the sparse grid.
uOnSG = interpolant.sample_on_SG(f)

 # Interpolate the data on the random sample (the same used in 3) by sampling
 # the interpolant on it.
u_interpol = interpolant.interpolate(yy_rnd, uOnSG)

# 5 ERROR COMPUTATION
error_samples = compute_error_samples_poisson(u_interpol, u_exa, n_h)
errr_uniform = np.amax(error_samples)
print("Number of collocation nodes:", interpolant.num_nodes)
print("Error:", errr_uniform)

# 6 REFINED SPARSE GRID INTERPOLATION
print("Refined sparse grid interpolation:")
mid_set = aniso_smolyak_mid_set(w=4, N=N, a=anisoVec(N))
interpolant = SGInterpolant(mid_set, cc_nodes, lev2knots,\
                            tp_interpolant=TPLagrangeInterpolator)
uOnSG = interpolant.sample_on_SG(f, old_xx=interpolant.SG, old_samples=uOnSG)

# 7 REFINED ERROR COMPUTATION
error_samples = compute_error_samples_poisson(u_interpol, u_exa, n_h)
errr_uniform = np.amax(error_samples)
print("Number of collocation nodes:", interpolant.num_nodes)
print("Error:", errr_uniform)
