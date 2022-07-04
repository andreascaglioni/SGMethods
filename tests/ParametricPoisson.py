from fenics import *
import numpy as np
from math import exp


def sampleParametricPoisson(yy, nH):
    """The parametric poisson problem with log a(y) = a0 + \sum_n an yn is solved with FE.
    the unfirm norm of the an decays quadratically"""
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS 
    u0 = Constant(0.0)
    f = Constant(1.)
    strA = "1."
    for n in range(len(yy)):
        strA = strA + "+" + "1/(1.65)*sin(x[0]*2.*pi*" + str(n+1) + ")/(" + str(n+1) + "*" + str(n+1) + ")" + "*" + str(yy[n])
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

def computeErrorPoisson(yy, u, uExa, V):
    """compute error parametric poisson problem as
    \norm{u-uExa} = max_{y\in yy} \nomr{u(y)-uExa(y)}{H^1_0}"""
    assert(len(u) == len(uExa))
    assert(len(u) == yy.shape[0])
    errSamples = np.zeros(len(u))
    errFun = Function(V)
    for n in range(len(u)):
        errFun.vector()[:] = u[n] - uExa[n]
        errSamples[n] = norm(errFun, 'H1')
    ww = np.exp(- np.square(np.linalg.norm(yy, 2, axis=1)))
    return np.max( errSamples * ww)


