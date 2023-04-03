import numpy as np
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from dolfin import *
from math import ceil
from SLLG.bdfllg_func import bdfllg_func
from SLLG.error_sample import error_sample


###################################################################################
def sample_LLG_function_noise(W, h, k, T, r, p):
    set_log_level(31)
    alpha = 1.4
    quadrature_degree = 6
    PETScOptions.clear()
    PETScOptions.set('ksp_type', 'gmres')
    PETScOptions.set("ksp_rtol", 1e-10)
    PETScOptions.set("pc_type", "ilu")
    
    # magnetic field
    H = (0, 0, 0)

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

    tt = np.linspace(0, T, int(T/k))
    Pr = FiniteElement('P', tetrahedron, r)
    Nh = int(1/h)
    mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 0.1), Nh, Nh, ceil(Nh/10.))
    Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
    element = MixedElement([Pr3, Pr])
    VV = FunctionSpace(mesh, element)
    V3 = FunctionSpace(mesh, Pr3)
    V = FunctionSpace(mesh, Pr)
    g = interpolate(gex, V3)
    mapr = bdfllg_func(r, p, alpha, T, k, minit, VV, V3, V, W, g, m1=[], quadrature_degree=quadrature_degree, Hinput=H)
    dofs = np.array([])
    for niter in range(len(mapr)):
        dofs = np.concatenate((dofs, mapr[niter].vector()[:]))
    return dofs

############################################################################


T=1

######################## TEST
LL = 2**np.linspace(0, 3, 4, dtype=int) 
nMCSample = 5 
nH = 10*np.ones(7, dtype=int)
nHRef = 10
nK = 2**(np.linspace(1,5,5))*8
nKRef = 2**6*8

######################## PRODUCTION
# LL = 2**np.linspace(0, 6, 7, dtype=int)
# nMCSample = 128
# nH = 2**(np.linspace(1,7,7, dtype=int))
# nK = nH*8*int(T)
# nHRef = 2**8
# nKRef = nHRef*8


hh = 1./nH
kk = T/nK
hRef = 1./nHRef
kRef = T/nKRef
ttRef = np.linspace(0, T, nKRef+1)  # NBB count also 0 and T

LSError = np.zeros((LL.size, hh.size))
for nL in range(LL.size):
    print("Conputing L=", LL[nL])
    L = LL[nL]

    # MC SAMPLING
    errMCSamples = np.zeros((nMCSample, hh.size))
    for nY in range(nMCSample):
        print("Computing sample", nY)
        yy = np.random.normal(0,1, (2**L))
        WReference = param_LC_Brownian_motion(ttRef, yy, T)
        print("Computing Reference...")
        mRef = sample_LLG_function_noise(WReference, hRef, kRef, T, 1, 1)

        #CONVERGENCE TEST current MC sample
        for nRefine in range(hh.size):
            print("Computing refinement ", nRefine)
            WCurr = WReference[::int(kk[nRefine]/kRef)]
            mCurr = sample_LLG_function_noise(WCurr, hh[nRefine], kk[nRefine], T, 1, 1)
            errMCSamples[nRefine] = error_sample(mCurr, mRef, nH[nRefine], nK[nRefine], nHRef, nKRef, T, 1, 1, False)

    np.savetxt('FEError_MC_samples_L_'+str(L)+'.txt', LSError)
    LSError[nL, :] = np.sum(errMCSamples,axis=0) / nMCSample
np.savetxt('MSError_FE_wrt_L.txt', LSError)