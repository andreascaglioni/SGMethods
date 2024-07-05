from math import pi
import numpy as np
from fenics import set_log_level, DOLFIN_EPS, Constant, Expression,UnitSquareMesh, FunctionSpace, DirichletBC, \
    TrialFunction, TestFunction, inner, grad, dx, solve, Function, norm


def sampleParametricPoisson(yy, nH, orderDecayCoefficients):
    """Sample the parameter-to-finite element solution map of the affine diffusion parametric Poisson Problem 
        -\nabla \cdot (a(\by) \nabla u(\by)) = 1 on D
                                           u = 0 on \partial D
    Domain D is 2D unit square;
    Diffusion a(\by) is affine in y, coercive on D
    Parameters are in [-1,1], any length of yy is allowed
    Mesh on D is structured with nH edges on each side
    Args:
        yy (double array): Vector of scalar parameters in [-1,1]
        nH (int): Number of mesh edges on each edge of square domain (structured mesh)
        orderDecayCoefficients (positive int): Order of decay terms affine diffusion
    
    Returns:
        np.array: Corrdinates of the solution (w.r.t. selected hat functions basis)
    """
    
    assert(np.min(yy) >= -1 and np.max(yy) <= 1)
    set_log_level(31)  # reduce fenics logging
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS 
    u0 = Constant(0.0)  # boundary condition
    f = Constant(1.)  # right-hand-side
    # Diffusion coefficient 
    C = pi**2/6 * 1.1  # makes the function uniformly positive
    strA = "1."
    for n in range(len(yy)):
        strA = strA + "+" + "sin(x[0]*2.*pi*" + str(n+1) + ")/("+str(C)+"*pow(" + str(n+1) + ","+ \
            str(orderDecayCoefficients) + "))" + "*" + str(yy[n])
    a = Expression(strA, degree=2)
    # Define computational objects
    mesh = UnitSquareMesh(nH, nH)
    V = FunctionSpace(mesh, "Lagrange", 1)
    bc = DirichletBC(V, u0, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    L = inner(a*grad(u), grad(v))*dx
    rhs = f*v*dx
    u = Function(V)
    solve(L == rhs, u, bc)  # solve linear system
    return u.vector()[:]

def computeErrorSamplesPoisson(u, uExa, nH):
    """Compute H^1_0(D) norm (see brevious function) of difference of two functions giventhrough their coordinates in\
          a hat functions basis
    Args:
        u (double array): Corrdinates of fucntion 1 
        uExa (double array): Corrdinates of Function 2
        nH (int): Number of mesh edges on each edge of square domain (structured mesh)
    
    Returns:
        double: error in the H^1_0(D) norm
    """

    assert(len(u) == len(uExa))
    mesh = UnitSquareMesh(nH, nH)
    V = FunctionSpace(mesh, "Lagrange", 1)
    errSamples = np.zeros(len(u))
    errFun = Function(V)
    for n in range(len(u)):
        errFun.vector()[:] = u[n] - uExa[n]
        errSamples[n] = norm(errFun, 'H1')
    return errSamples