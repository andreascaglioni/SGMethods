from __future__ import division
from scipy.io import savemat
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SLLG'))
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.bdfllg_func import *
from math import ceil


def sample_LLG_function_noise(param, Nh, Ntau, T, r, p):
    set_log_level(31)
    alpha = 1.4

    quadrature_degree = 6
    
    tt = np.linspace(0, T, Ntau+1)
    W = param_LC_Brownian_motion(tt, param, T)
    tau = T / Ntau

    PETScOptions.clear()
    PETScOptions.set('ksp_type', 'gmres')
    PETScOptions.set("ksp_rtol", 1e-10)
    PETScOptions.set("pc_type", "ilu")

    H = (0, 0, 10)

    # coefficient g
    sqnx = '((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))'
    C = '400'
    vcomp0 = C+'*exp(-1.0/(0.25-' + sqnx + '))*(x[0]-0.5)'
    vcomp1 = C+'*exp(-1.0/(0.25-' + sqnx + '))*(x[1]-0.5)'
    vcomp2 = 'sqrt(1-'+C+'*'+C+'*exp(-2.0/(0.25-'+sqnx+'))*('+sqnx+'))'
    c1 = '('+sqnx+'<=0.25) ? ' + vcomp0 + ' : 0'
    c2 = '('+sqnx+'<=0.25) ? ' + vcomp1 + ' : 0'
    c3 = '('+sqnx+'<=0.25) ? ' + vcomp2 + ' : 1'
    gex = Expression((c1, c2, c3), degree=r)

    # CONSTANT IC
    c1 = '-sqrt(2./3.)'
    c2 = 'sqrt(1./6.)'
    c3 = '-sqrt(1./6.)'
    minit = Expression((c1, c2, c3), degree=r+1)

    Pr = FiniteElement('P', tetrahedron, r)
    mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 0.1), Nh, Nh, ceil(Nh/10.))
    Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
    element = MixedElement([Pr3, Pr])
    VV = FunctionSpace(mesh, element)
    V3 = FunctionSpace(mesh, Pr3)
    V = FunctionSpace(mesh, Pr)

    g = interpolate(gex, V3)

    mapr = bdfllg_func(r, p, alpha, T, tau, minit, VV, V3, V, W, g, m1=[], quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)

    dofs = np.array([])
    for niter in range(len(mapr)):
        dofs = np.concatenate((dofs, mapr[niter].vector()[:]))

    return dofs
