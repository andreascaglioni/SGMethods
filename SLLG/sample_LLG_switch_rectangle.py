from __future__ import division
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.bdfllg_func import *


def sample_LLG_switch_rectangle(param, Nh, Ntau, T, r, p):
    set_log_level(31)
    alpha = 1.4
    quadrature_degree = 6
    PETScOptions.clear()
    PETScOptions.set('ksp_type', 'gmres')
    PETScOptions.set("ksp_rtol", 1e-10)
    PETScOptions.set("pc_type", "ilu")
    # external magnetic field
    # H = (0, 0, -20.)
    H = (0., 0., 0.)
    # coefficient g
    g1 = '0.1*(-0.5)*cos(pi*x[0])'
    g2 = '0.1*(-0.5)*cos(pi*x[1])'
    g3 = '0.1*sqrt(1.-0.25*(cos(pi*x[0])*cos(pi*x[0])+cos(pi*x[1])*cos(pi*x[1])))'
    gex = Expression((g1, g2, g3), degree=r)
    # CONSTANT IC
    m1 = '0.'
    m2 = '0.'
    m3 = '1.'
    minit = Expression((m1, m2, m3), degree=r+1)
    tau = T / Ntau
    tt = np.linspace(0, T, Ntau+1)
    W = param_LC_Brownian_motion(tt, param, T)
    Pr = FiniteElement('P', triangle, r)
    mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), Nh, 5*Nh)
    Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
    element = MixedElement([Pr3, Pr])
    VV = FunctionSpace(mesh, element)
    V3 = FunctionSpace(mesh, Pr3)
    V = FunctionSpace(mesh, Pr)
    g = interpolate(gex, V3)
    mapr = bdfllg_func(r, p, alpha, T, tau, minit, VV, V3, V, W, g, m1=[], quadrature_degree=quadrature_degree, Hinput=H)
    dofs = np.array([])
    for niter in range(len(mapr)):
        dofs = np.concatenate((dofs, mapr[niter].vector()[:]))
    return dofs