""" Use the Dolfin finte elements library to perform some tasks for Example 2.
In particular:
* Compute a finninte element solution of the Poisson problem with a
diffusion coefficient that depends on a parameter y, here fixed;
* Compute the error of the finite element solution in the H^1_0(D) norm."""

from math import pi
import numpy as np
from fenics import set_log_level, DOLFIN_EPS, Constant, Expression,\
    UnitSquareMesh, FunctionSpace, DirichletBC, TrialFunction, TestFunction,\
        inner, grad, dx, solve, Function, norm\

# TODO find out if possible to have conditional imprort depending on
# user-selected parameter when running pip over requirements.txt
# from fenics import set_log_level, DOLFIN_EPS, Constant, Expression,\
#     UnitSquareMesh, FunctionSpace, DirichletBC, TrialFunction, TestFunction,\
#     inner, grad, dx, solve, Function, norm

def parametric_affine_diffusion(yy, order_decay_diff):
    """ Affine parametric diffusion for Poisson problem:

    .. math::

        a(y, x) = 1 + sum_{n=1}^{N} y_n* \frac{1}{C n^{\gamma}}}\sin(x_1*2.*\pi*n)
    
    Args:
        yy (numpoy.ndarray[float]): Vector of scalar parameters in 
            :math:`[-1,1]`
        order_decay_diff (int): Positive. Order of decay of the terms of the 
            affine diffusion (:math:`\gamma` in the formula above).
    
    Returns:
        fenics Expression: Affine parametric diffusion as a Fenics Expression,
        for the fixed given parameter ``yy``.
    """

    # Define the diffusion coefficient for the input parameter yy. The result is
    # a function of space only.
    assert np.min(yy) >= -1 and np.max(yy) <= 1
    norm_const = pi**2/6 * 1.1  # Makes the function uniformly positive
    str_a = "1."
    for n, y_n in enumerate(yy):
        str_a = str_a + "+" + "sin(x[0]*2.*pi*" + str(n+1) + ")/("+ \
            str(norm_const)+"*pow(" + str(n+1) + ","+ \
            str(order_decay_diff) + "))" + "*" + str(y_n)
    return Expression(str_a, degree=2)

def sample_parametric_poisson(yy, n_h, order_decay_diff):
    """Sample the parameter-to-finite element solution map of the affine 
    diffusion parametric Poisson problem.
    """
    #     -\nabla \cdot (a(\by) \nabla u(\by)) = 1 on D
    #                                        u = 0 on \partial D
    # Domain D is 2D unit square;
    # Diffusion a(\by) is coercive on D for any \by (n.b. coercivity constant needs not be unform in \by!)
    # Parameters are in [-1,1], any length of yy is allowed
    # Mesh on D is structured with nH edges on each side
    # Args:
    #     yy (double array): Vector of scalar parameters
    #     nH (int): Number of mesh edges on each edge of square domain (structured mesh)
    #     orderDecayCoefficients (positive int): Order of decay terms affine diffusion
    
    # Returns:
    #     np.array: Corrdinates of the solution (w.r.t. selected hat functions basis)
    # """

    set_log_level(31)  # reduce fenics logging
    # Problem data
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS\
             or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
    u0 = Constant(0.0)  # boundary condition
    f = Constant(1.)  # right-hand-side
    # Parametric diffusion as a Dolfin Expression
    a = parametric_affine_diffusion(yy, order_decay_diff)
    # Computational objects
    mesh = UnitSquareMesh(n_h, n_h)
    V = FunctionSpace(mesh, "Lagrange", 1)
    bc = DirichletBC(V, u0, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    L = inner(a*grad(u), grad(v))*dx
    rhs = f*v*dx
    u = Function(V)
    # Solve linear system
    solve(L == rhs, u, bc)
    return u.vector()[:]  # return only coordinates!

def compute_error_samples_poisson(u_approx, u_exact, n_h):
    """Compute H^1_0(D) norm (see brevious function) of difference of two
    functions giventhrough their coordinates in a hat functions basis.

    Args:
        u (double array): Corrdinates of fucntion 1.
        uExa (double array): Corrdinates of Function 2.
        nH (int): Number of mesh edges on each edge of square domain.
    
    Returns:
        double: error in the H^1_0(D) norm.
    """

    assert len(u_approx) == len(u_exact)
    mesh = UnitSquareMesh(n_h, n_h)
    V = FunctionSpace(mesh, "Lagrange", 1)
    err_samples = np.zeros(len(u_approx))
    err_fun = Function(V)
    for n in range(len(u_approx)):
        err_fun.vector()[:] = u_approx[n] - u_exact[n]
        err_samples[n] = norm(err_fun, 'H1')
    return err_samples