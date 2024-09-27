import numpy as np
from src.nodes_1d import equispacedNodes
from src.tp_piecewise_quadratic import TPPwQuadraticInterpolator
from src.tp_piecewise_cubic import TPPwCubicInterpolator
from src.sparse_grid_interpolant import SGInterpolant


def test_SGInterpolant_exact_polyDegree1():
    """Interpolate exactly a polynomial of degree 1"""

    P = lambda x : 2. + x[0] + x[0]*x[1]
    Lambda = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    kk = lambda n : equispacedNodes(n)
    lev2knots = lambda n : n+1
    interpolant = SGInterpolant(Lambda, kk, lev2knots)
    samples_SG = interpolant.sampleOnSG(P)
    xRnd = np.random.uniform(-1, 1, [10, 2])
    IP_x = interpolant.interpolate(xRnd, samples_SG)
    P_x = np.zeros((xRnd.shape[0]))
    for n in range(xRnd.shape[0]):
        P_x[n] = P(xRnd[n, :])
    assert np.linalg.norm(IP_x-P_x) < 1.e-4


def test_SGInterpolant_exact_polyDegree2():
    """Interpolate exactly a polynomial of degree 2"""    
    P = lambda x : 2. + x[0] + x[0]*x[1] + x[1]**2 + x[0]*x[1]**2
    Lambda = np.array([[0, 0],
                        [0, 1],
                        [0, 2],
                        [1, 0],
                        [1, 1],
                        [1, 2]])
    kk = lambda n : equispacedNodes(n)
    lev2knots = lambda n : 2*n+1
    interpolant = SGInterpolant(Lambda, kk, lev2knots, \
                                TPInterpolant=TPPwQuadraticInterpolator)
    samples_SG = interpolant.sampleOnSG(P)
    xRnd = np.random.uniform(-1, 1, [10, 2])
    IP_x = interpolant.interpolate(xRnd, samples_SG)
    P_x = np.zeros((xRnd.shape[0]))
    for n in range(xRnd.shape[0]):
        P_x[n] = P(xRnd[n, :])
    print(np.linalg.norm(IP_x-P_x))
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
    kk = lambda n : equispacedNodes(n)
    lev2knots = lambda n : 3*n+1
    interpolant = SGInterpolant(Lambda, kk, lev2knots, \
                                TPInterpolant=TPPwCubicInterpolator)
    samples_SG = interpolant.sampleOnSG(P)
    xRnd = np.random.uniform(-1, 1, [10, 2])
    IP_x = interpolant.interpolate(xRnd, samples_SG)
    P_x = np.zeros((xRnd.shape[0]))
    for n in range(xRnd.shape[0]):
        P_x[n] = P(xRnd[n, :])
    assert np.linalg.norm(IP_x-P_x) < 1.e-4


def test_SGInterpolant_approx_exp():
    """Approximate exponential function. Check that error on random sample 
    decreases with refinement midset"""
    f = lambda x : np.exp(np.sum(x))