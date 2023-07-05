from __future__ import division
from multiprocessing import Pool
from math import log, pi, sqrt, factorial, log2
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))

from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SLLG.bdfllg_func import *
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.profits import ProfitMix
from SLLG.error_sample_fast import error_sample_fast


"""When writing a multilevel example, you want FE and SG errors to be of similar order, otherwise it makes no sense to go multilevel
Here we look at SG convergence as usual
"""


######################## PROBLEM COEFFICIENTS #########################
alpha = 1.4
T=1
# const IC
# m1 = '0.'
# m2 = '0.'
# m3 = '1.'

# NONconst IC
m1 = '-0.5*cos(pi*x[0])'
m2 = '-0.5*cos(pi*x[1])'
m3 = 'sqrt(1-('+m1+')*('+m1+')-('+m2+')*('+m2+'))'

# External magnetic field
H = (0, 0, 0)

## g coefficient non constant
# g1 = '-0.5*cos(pi*x[0])'
# g2 = '-0.5*cos(pi*x[1])'
# g3 = 'sqrt(1-('+g1+')*('+g1+')-('+g2+')*('+g2+'))'

## g coeff non-const and smaller in modulus
g1 = '0.2*(-0.5)*cos(pi*x[0])'
g2 = '0.2*(-0.5)*cos(pi*x[1])'
g3 = '0.2*sqrt(1- 0.25*cos(pi*x[0])*cos(pi*x[0]) - 0.25*cos(pi*x[1])*cos(pi*x[1]) )'

######################### NUMERICS COEFFICIENTS #########################
maxNumNodes = 1024  # 64  # 
p = 2  # NBB degrere + 1
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
Profit = lambda nu : ProfitMix(nu, p)
np.random.seed(567)
NRNDSamples =  128  #64  #   
NParallel = 32

FEMOrder = 1
BDFOrder = 1
Nh = 32  # 8  # 
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
    
######################### COMPUTE REFERENCE
yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)

######################### RUN CONVERGENCE TEST
err = np.array([])
nNodes = np.array([])
nDims = np.array([])
w=0
I = midSet()
oldSG = None
uOnSG = None
while(True):
    print("Computing w  = ", w)
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    midSetMaxDim = np.amax(I.midSet, axis=0)
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N, "\nMax midcomponents:", midSetMaxDim)
    uOnSG = interpolant.sampleOnSG(F, oldXx=oldSG, oldSamples=uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    
    # compute error    
    errSamples = np.array([ physicalError(uInterp[n], uExa[n]) for n in range(NRNDSamples) ])
    errCurr = sqrt(np.mean(np.square(errSamples)))
    err = np.append(err, errCurr)
    print("Error:", err[w])
    nNodes = np.append(nNodes, interpolant.numNodes)
    nDims = np.append(nDims, I.N)
    
    np.savetxt('multilevel_preprocessing_conv_SG_nonconstIC.csv',np.transpose(np.array([nNodes, err, nDims])), delimiter=',')
    
    oldSG = interpolant.SG

    # update midset for next iteration, doubling the number of collocation nodes
    current_num_cps = nNodes[-1]
    while(current_num_cps <= sqrt(2)* nNodes[-1]):
        P = Profit(I.margin)
        idMax = np.argmax(P)
        I.update(idMax)
        interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
        current_num_cps = interpolant.numNodes
        if(current_num_cps > maxNumNodes):
            break
    w+=1

print("# nodes:", nNodes)
print("# dimen:", nDims)
print("error:", err)
print("rates:", -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:]))
plt.plot(nNodes, err, '.-', nNodes, np.power(nNodes, -0.5), '.k-', nNodes, np.power(nNodes, -0.25), 'k-', )
plt.show()
plt.plot(nNodes, nDims, '.-', nNodes, nNodes, 'k-')
plt.show()