"""Tests for the module sparse_grid_interpolant.py"""

from math import sqrt
import numpy as np
from sgmethods.nodes_1d import equispaced_nodes, cc_nodes,\
    optimal_gaussian_nodes
from sgmethods.tp_inteprolants import TPPwLinearInterpolator,\
    TPPwCubicInterpolator
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.multi_index_sets import smolyak_mid_set, compute_mid_set_fast
from sgmethods.parametric_expansions_wiener_process import\
    param_LC_Brownian_motion


def test_sg_interpolant_exact_degree1():
    """Interpolate exactly a polynomial of degree 1"""

    profit = lambda x : 2. + x[0] + x[0]*x[1]
    Lambda = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    kk = lambda n : equispaced_nodes(n)
    lev2knots = lambda n : n+1
    # default piecewise linear interpolation
    interpolant = SGInterpolant(Lambda, kk, lev2knots)
    samples_sg = interpolant.sample_on_SG(profit)
    xRnd = np.random.uniform(-1, 1, [10, 2])
    IP_x = interpolant.interpolate(xRnd, samples_sg)
    P_x = np.zeros((xRnd.shape[0]))
    for n in range(xRnd.shape[0]):
        P_x[n] = profit(xRnd[n, :])
    assert np.linalg.norm(IP_x-P_x) < 1.e-4

def test_SGInterpolant_exact_polyDegree3():
    """Interpolate exactly a polynomial of degree 3"""  

    P = lambda x : 2. + x[0] + x[0]*x[1] + x[1]**3 + x[0]*x[1]**3
    Lambda = np.array([[0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [1, 3]])
    kk = lambda n : equispaced_nodes(n)
    lev2knots = lambda n : 3*n+1
    interpolant = SGInterpolant(Lambda, kk, lev2knots, \
                                tp_interpolant=TPPwCubicInterpolator)
    samples_SG = interpolant.sample_on_SG(P)
    xRnd = np.random.uniform(-1, 1, [10, 2])
    IP_x = interpolant.interpolate(xRnd, samples_SG)
    P_x = np.zeros((xRnd.shape[0]))
    for n in range(xRnd.shape[0]):
        P_x[n] = P(xRnd[n, :])
    assert np.linalg.norm(IP_x-P_x) < 1.e-4


def test_SGinterpolant_exp():
    """Inteprolate an exponential functio function (not exactly a polynomial)"""

    # Define function to interpolate
    N = 4
    F = lambda x : np.exp(np.sum(x))

    #Compute random saples for error
    np.random.seed(0)  # If changed, some tests may fail
    rnd_samples = np.random.uniform(-1, 1, [5000, N])
    f_rnd = np.array(list(map(F, rnd_samples)))

    # Parameters interpolant
    lev2knots = lambda nu : nu+1  # np.where(nu == 0, 1, 2**nu+1)
    kk = lambda n : cc_nodes(n)
    TPInterpol = lambda nodes_tuple, f_on_nodes : \
        TPPwLinearInterpolator(nodes_tuple, f_on_nodes)

    # Convergence test
    err = np.zeros(0)
    number_samples = np.zeros_like(err)
    ww = np.linspace(0, 6, 7, dtype=int)
    for w in ww:
        # Define Interpolant
        Lambda = smolyak_mid_set(w, N=N)
        interpolant = SGInterpolant(Lambda, kk, lev2knots, 
                                    tp_interpolant=TPInterpol, n_parallel=1, 
                                    verbose=False)
        f_on_SG = interpolant.sample_on_SG(F)
        f_interpolated = interpolant.interpolate(rnd_samples, f_on_SG)

        # Cmpute error
        err_curr = np.amax(np.abs(f_interpolated - f_rnd))
        print("w = ", w)
        print("Nodes:", interpolant.num_nodes)
        print("Error:", err_curr)
        number_samples = np.append(number_samples, interpolant.num_nodes)
        err = np.append(err, err_curr)

    # Test assertion: Respectively
    # - number of collocation nodes increases with w
    # - error decreases with w
    # - error decreases exponentially with number of collocation nodes, up to 
    #   algebraic term
    assert(np.all(np.diff(number_samples) > 0))
    assert(np.all(np.diff(err) < 0))
    assert(np.all(err< 31*number_samples**(0.02)*np.exp(-0.005*number_samples)))


def test_SGinterpolant_Gaussian_RF():
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
    kk = lambda n : optimal_gaussian_nodes(n, p=p)
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
        Lambda = compute_mid_set_fast(Profit, 11.**(-iter), N=N)

        # Define Interpolant and sample, interpolate
        interpolant = SGInterpolant(Lambda, kk, lev2knots, 
                                    tp_interpolant=TPInterpol, n_parallel=1, 
                                    verbose=False)
        f_on_SG = interpolant.sample_on_SG(F)
        f_interpolated = interpolant.interpolate(rnd_samples, f_on_SG)

        # Cmpute error and print result current iteration
        errSamples = np.linalg.norm(f_rnd - f_interpolated, 2, axis=1) * sqrt(1/nt)
        err_curr = sqrt(np.mean(np.square(errSamples)))
        print("iter = ", iter)
        print("Nodes:", interpolant.num_nodes)
        print("Error:", err_curr)
        number_samples = np.append(number_samples, interpolant.num_nodes)
        err = np.append(err, err_curr)

    # Test assertion: Respectively
    # - number of collocation nodes increases with w
    # - error decreases with w
    assert(np.all(np.diff(number_samples) >= 0))
    assert(np.all(err< 2*number_samples**(-0.2)*np.exp(-0.003*number_samples)))