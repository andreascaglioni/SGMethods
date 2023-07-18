import numpy as np
from math import sqrt, factorial, log2, pi
from multiprocessing import Pool
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.compute_aposteriori_estimator_reduced_margin import compute_aposteriori_estimator_reduced_margin
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.bdfllg_func import *
from SLLG.error_sample_fast import error_sample_fast


"""Adaptive approximation of SLLG 
Genster-Griebel algorithm"""

######################## PROBLEM COEFFICIENTS #########################
alpha = 1.4
T=1
# const IC
m1 = '0.'
m2 = '0.'
m3 = '1.'
# External magnetic field
H = (0, 0, 0)
# g coefficient non constant
g1 = '-0.5*cos(pi*x[0])'
g2 = '-0.5*cos(pi*x[1])'
g3 = 'sqrt(1-('+g1+')*('+g1+')-('+g2+')*('+g2+'))'

######################### NUMERICS COEFFICIENTS #########################
maxNumNodes = 1500  # 64  #  
p = 2  # NBB degrere + 1
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
np.random.seed(567)
NRNDSamples =  1024  #  64  # 
NParallel = 32

FEMOrder = 1
BDFOrder = 1
Nh = 32  # 8
Ntau = Nh * 4
PETScOptions.clear()
PETScOptions.set('ksp_type', 'gmres')
PETScOptions.set("ksp_rtol", 1e-8)
PETScOptions.set("pc_type", "ilu")
PETScOptions.set("ksp_initial_guess_nonzero", "true")
quadrature_degree = 6

######################### PRE-PROCESSING #########################
gex = Expression((g1, g2, g3), degree=FEMOrder+1)
minit = Expression((m1, m2, m3), degree=FEMOrder+1)
Pr = FiniteElement('P', triangle, FEMOrder)
tau = float(T) / Ntau
tt = np.linspace(0, T, Ntau+1)
mesh = UnitSquareMesh(Nh, Nh)
Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), FEMOrder, dim=3)
element = MixedElement([Pr3, Pr])
V3 = FunctionSpace(mesh, Pr3)
V = FunctionSpace(mesh, Pr)
VV = FunctionSpace(mesh, element)
g = interpolate(gex, V3)

def F(x):
    W = param_LC_Brownian_motion(tt, x, T)
    mRef = bdfllg_func(FEMOrder, BDFOrder, alpha, T, tau, minit, VV, V3, V, W, g, quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)
    dofs = np.array([])
    for niter in range(len(mRef)):
        dofs = np.concatenate((dofs, mRef[niter].vector()[:]))
    return dofs

def physicalError(u, uExa):
    return error_sample_fast(u, mesh, V3, tt, uExa, V3, tt, BDFOrder)

def L2ErrorParam(u, uExa):
    assert(uExa.shape == u.shape)
    errSample = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
    return sqrt(np.mean(np.square(errSample)))

######################### COMPUTE REFERENCE #########################
yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = np.array(pool.map(F, yyRnd))

######################### RUN CONVERGENCE TEST #########################
err = np.array([])
nNodes = np.array([])
nDims = np.array([])
error_estimator = np.array([])
w=0
I = midSet(trackReducedMargin=True)
oldSG = None
uOnSG = None
oldRM = np.array([])
indicators_reduced_margin = np.array([])

while True:
    print("Computing w  = ", w)
    # COMPUTE
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    midSetMaxDim = np.amax(I.midSet, axis=0)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N, "\nMax midcomponents:", midSetMaxDim)
    uOnSG = interpolant.sampleOnSG(F, oldXx=oldSG, oldSamples=uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    
    # COMPUTE ERROR
    err = np.append(err, L2ErrorParam(uInterp, uExa))
    print("Error:", err[-1])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    oldSG = interpolant.SG

    # ESTIMATE
    indicators_reduced_margin = compute_aposteriori_estimator_reduced_margin(oldRM, indicators_reduced_margin, I, knots, lev2knots, F, interpolant.SG, uOnSG, yyRnd, L2ErrorParam, uInterp, interpolationType=interpolationType, NParallel=NParallel)
    error_estimator = np.append(error_estimator, np.sum(indicators_reduced_margin))
    oldRM = I.reducedMargin

    np.savetxt('example_SLLG_adaptive_SG.csv',np.transpose(np.array([nNodes, err, nDims, error_estimator])), delimiter=',')
    print("error indicators reduced margin", np.sum(indicators_reduced_margin))
    
    # MARK
    idMax = np.argmax(indicators_reduced_margin)  # NBB index in REDUCED margin
    mid = I.reducedMargin[idMax, :]
    idxMargin = np.where(( I.margin==mid).all(axis=1) )[0][0]
    
    # REFINE
    I.update(idxMargin)
    w+=1 

print(err)
rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-')
plt.loglog(nNodes, np.power(nNodes, -0.5), '-k')
plt.loglog(nNodes, np.power(nNodes, -0.25), '-k')
plt.show()