import numpy as np 
from fenics import set_log_level, DOLFIN_EPS, Constant, Expression, UnitSquareMesh, FunctionSpace, DirichletBC, TrialFunction, TestFunction, inner, grad, dx, solve, Function, norm
from math import sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import CCNodes
from SGMethods.TPLagrangeInterpolator import TPLagrangeInterpolator
from SGMethods.MidSets import anisoSmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant

"""Tutorial on sparse grid interpolation with SGMethods. 
We see how to use SGMethods to approximate the parametric affine diffusion Poisson problem."""

def sampleParametricPoisson(yy, nH, orderDecayCoefficients):
    """Function to sample the parametric affined diffusion Poisson problem."""
    set_log_level(31)  # reduce fenics logging
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS 
    u0 = Constant(0.0)
    f = Constant(1.)
    # Diffusion coefficient a(y,x) = 1 + 1/C * \sum_{n=1}^N y_n \frac{sin(x[0]*2*pi*n)}{n^2}, where C = 1.65 makes the functions uniformly positive
    strA = "1."
    for n in range(len(yy)):
        strA = strA + "+" + "sin(x[0]*2.*pi*" + str(n+1) + ")/pow(" + str(n+1) + ","+ str(orderDecayCoefficients) +")" + "*" + str(yy[n])
    strA = "exp(" + strA + ")"
    a = Expression(strA, degree=2)
    mesh = UnitSquareMesh(nH, nH)
    V = FunctionSpace(mesh, "Lagrange", 1)
    bc = DirichletBC(V, u0, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    L = inner(a*grad(u), grad(v))*dx
    rhs = f*v*dx
    u = Function(V)
    solve(L == rhs, u, bc)
    return u, V, mesh

def computeErrorSamplesPoisson(u, uExa, V):
    """compute error parametric poisson samples as \norm{u(y)-uExa(y)}{H^1_0}
    for all y"""
    assert(len(u) == len(uExa))
    errSamples = np.zeros(len(u))
    errFun = Function(V)
    for n in range(len(u)):
        errFun.vector()[:] = u[n] - uExa[n]
        errSamples[n] = norm(errFun, 'H1')
    return errSamples

# Sparse grid parameters
TPInterpolant = (lambda nodesTuple, fOnNodes :
                    TPLagrangeInterpolator(nodesTuple, fOnNodes))






# # Parametric Poisson sampler given a parameter y
# weightFun = lambda N : np.power(np.linspace(1, N, N), -2)  # Weight function
# def F(y):
#     weightCurr = weightFun(y.size)
#     return sin(np.dot(weightCurr, y))

# # Sparse grid parameters
# p = 2  # Polynomial degree + 1 
# TPInterpolant = (lambda nodesTuple, fOnNodes : 
#                  RegularGridInterpolator(nodesTuple, fOnNodes, method='linear', bounds_error=False, fill_value=None))
# lev2knots = lambda n: 2**(n+1)-1  # Level-to-knots function
# knots = lambda n : unboundedKnotsNested(n, p)  # Nodes family

# # Error computation
# def computeL2Error(uExa, Iu):
#     assert(uExa.shape == Iu.shape)
#     return sqrt(np.mean(np.square(uExa - Iu)))
# yyRnd = np.random.normal(0, 1, [256, 1000])  # 256 random sample of (effectively) infinite parameter vector
# uExa = np.array(list(map(F, yyRnd)))  # Random sample of exact function

# # Sparse grid interpolation
# anisoVec = lambda N : 2**(np.linspace(0, N-1, N))
# midSet = anisoSmolyakMidSet(w=10, N=10, a=anisoVec(10))  # Anisotropic Smolyak multi-index set in 10 dimensions
# interpolant = SGInterpolant(midSet, knots, lev2knots, TPInterpolant=TPInterpolant)
# uOnSG = interpolant.sampleOnSG(F)  # Sample the function on the sparse grid
# uInterp = interpolant.interpolate(yyRnd, uOnSG)  # Compute the interpolated value
# error = computeL2Error(uExa, uInterp)
# print("Error:", error)