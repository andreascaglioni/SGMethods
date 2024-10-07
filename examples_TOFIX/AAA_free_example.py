from math import sqrt
import numpy as np
from sgmethods.parametric_expansions_Wiener_process import param_LC_Brownian_motion
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.tp_inteprolants import TPPwLinearInterpolator
from sgmethods.multi_index_sets import computeMidSetFast
from sgmethods.nodes_1d import unboundedNodesOptimal
import matplotlib.pyplot as plt


# Define parameters and function to interpolate
N = 4  # Number of parameters
nt = 10  # number of timesptes
T = 1  # Final approximation time
tt = np.linspace(0, T, nt)
F = lambda y : np. power(param_LC_Brownian_motion(tt, y, T=T), 2)

#Compute random saples for error
np.random.seed(0)  # If changed, some tests may fail
rnd_samples = np.random.normal(0, 1, [5000, N])
f_rnd = np.array(list(map(F, rnd_samples)))

# Parameters interpolant
lev2knots = lambda nu : 2**(nu+1)-1
p = 2  # always equals inteprolant degree + 1
kk = lambda n : unboundedNodesOptimal(n, p=p)
TPInterpol = lambda nodes_tuple, f_on_nodes : \
    TPPwLinearInterpolator(nodes_tuple, f_on_nodes)
ell = lambda nu: np.where(nu==0, 0, np.floor(np.log2(nu)).astype(int)+1) # level
Profit = lambda nu : np.prod(\
    np.where(nu==1, 2.**(-ell(nu)-nu)/2, (2.**(-p*nu*(1+ell(nu))))/2) \
                    )  # computations
# Convergence test
err = np.zeros(0)
number_samples = np.zeros_like(err)
ii = np.linspace(0, 9, 10, dtype=int)
for iter in ii:
    # Define multi-index set
    Lambda = computeMidSetFast(Profit, 11.**(-iter), N=N)

    # Define Interpolant and sample, interpolate
    interpolant = SGInterpolant(Lambda, kk, lev2knots, 
                                TPInterpolant=TPInterpol, NParallel=1, 
                                verbose=False)
    f_on_SG = interpolant.sampleOnSG(F)
    f_interpolated = interpolant.interpolate(rnd_samples, f_on_SG)

    # Cmpute error and print result current iteration
    errSamples = np.linalg.norm(f_rnd - f_interpolated, 2, axis=1) * sqrt(1/nt)
    err_curr = sqrt(np.mean(np.square(errSamples)))
    print("iter = ", iter)
    print("Nodes:", interpolant.numNodes)
    print("Error:", err_curr)
    number_samples = np.append(number_samples, interpolant.numNodes)
    err = np.append(err, err_curr)

plt.loglog(number_samples, err, 'o-')
plt.loglog(number_samples, 2*number_samples**(-0.2)*np.exp(-0.003*number_samples), 'k-')
plt.show()

# Test assertion: Respectively
# - number of collocation nodes increases with w
# - error decreases with w
assert(np.all(np.diff(number_samples) >= 0))
assert(np.all(err< 2*number_samples**(-0.2)*np.exp(-0.003*number_samples)))