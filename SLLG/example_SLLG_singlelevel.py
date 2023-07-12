import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.MLInterpolant import MLInterpolant
from SLLG.error_sample_fast import error_sample_fast
from SLLG.profits import ProfitMix
from SLLG.bdfllg_func import *
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion


"""SINGLE LEVEL example for the parametric LLG problem. 
Using PROBLEM 2"""

######################## PROBLEM COEFFICIENTS #########################
alpha = 1.4
T=1

# NONconst IC
m1 = '-0.5*cos(pi*x[0])'
m2 = '-0.5*cos(pi*x[1])'
m3 = 'sqrt(1-('+m1+')*('+m1+')-('+m2+')*('+m2+'))'

# External magnetic field
H = (0, 0, 0)

## g coeff non-const and smaller in modulus
g1 = '0.2*(-0.5)*cos(pi*x[0])'
g2 = '0.2*(-0.5)*cos(pi*x[1])'
g3 = '0.2*sqrt(1- 0.25*cos(pi*x[0])*cos(pi*x[0]) - 0.25*cos(pi*x[1])*cos(pi*x[1]) )'

######################### NUMERICS COEFFICIENTS #########################
nRefs = 6  # the first level has coarse SG and FE space; the alst one has fine SG and FE space
NRNDSamples = 128  # 4  #  
NParallel = 32  # 4  # 

p=2
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
Profit = ProfitMix

# The size of the sparse grid in each ML expansion must balance the FE error at different resulutions. In another script we did a pre-rpocesssing that determins the convergence of FE and SG alone and therefore allow to determine the SG sizes
NNLevels = np.linspace(1,nRefs,nRefs, dtype=int)
SGCards = 2**(2*NNLevels)

FEMOrder = BDFOrder = 1
Nh = 2**(np.linspace(0, nRefs-1, nRefs, dtype=int))  # NB in linspace the final value is INCLUDED
Ntau = Nh*4
Nhref = Nh[-1]*2
Ntauref = Nhref*4

np.random.seed(567)
PETScOptions.clear()
PETScOptions.set('ksp_type', 'gmres')
PETScOptions.set("ksp_rtol", 1e-8)
PETScOptions.set("pc_type", "ilu")
PETScOptions.set("ksp_initial_guess_nonzero", "true")
quadrature_degree = 6

######################### PRE-PROCESSING #########################
print("Preprocessing FE spaces...")
gex = Expression((g1, g2, g3), degree=FEMOrder+1)
minit = Expression((m1, m2, m3), degree=FEMOrder+1)
Pr = FiniteElement('P', triangle, FEMOrder)
tauref = float(T) / Ntauref
ttref = np.linspace(0, T, Ntauref+1)
NttRef = Ntauref+1
meshRef = UnitSquareMesh(Nhref, Nhref)
Pr3Ref = VectorElement('Lagrange', meshRef.ufl_cell(), FEMOrder, dim=3)
elementRef = MixedElement([Pr3Ref, Pr])
V3ref = FunctionSpace(meshRef, Pr3Ref)
dimV3Ref = V3ref.dim()
VRef = FunctionSpace(meshRef, Pr)
VVRef = FunctionSpace(meshRef, elementRef)
gRef = interpolate(gex, V3ref)
tau = float(T) / Ntau
tt = []
Ntt = Ntau+1
mesh = []
nElements=[]
Pr3 = []
element = []
V3 = []
dimV3 = []
V = []
VV = []
dimVV = []
g = []
for i in range(Nh.size):
    tt.append(np.linspace(0, T, Ntau[i]+1))
    mesh.append(UnitSquareMesh(Nh[i], Nh[i]))
    nElements.append(mesh[i].num_cells())
    Pr3.append(VectorElement('Lagrange', mesh[i].ufl_cell(), FEMOrder, dim=3))
    element.append(MixedElement([Pr3[i], Pr]))
    V3.append(FunctionSpace(mesh[i], Pr3[i]))
    dimV3.append(V3[i].dim())
    V.append(FunctionSpace(mesh[i], Pr))
    VV.append(FunctionSpace(mesh[i], element[i]))
    dimVV.append(VV[i].dim())
    g.append(interpolate(gex, V3[i]))

# reference solution sampler
def F(y):
    Wref = param_LC_Brownian_motion(ttref, y, T)
    mRef = bdfllg_func(FEMOrder, BDFOrder, alpha, T, tauref, minit, VVRef, V3ref, VRef, Wref, gRef, quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)
    dofsRef = np.array([])
    for niter in range(len(mRef)):
        dofsRef = np.concatenate((dofsRef, mRef[niter].vector()[:]))
    return dofsRef

# approximate solution sampler
def FApprox(y, k):
    W = param_LC_Brownian_motion(tt[k], y, T)
    mApprx = bdfllg_func(FEMOrder, BDFOrder, alpha, T, tau[k], minit, VV[k], V3[k], V[k], W, g[k], quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)
    dofsApprx = np.array([])
    for niter in range(len(mApprx)):
        dofsApprx = np.concatenate((dofsApprx, mApprx[niter].vector()[:]))
    return dofsApprx

################################# ERROR ESTIMATION #################################
assert((FEMOrder==1) and (BDFOrder==1))  # error computation implemented only for order 1
def errorSingleLevel(uExa, uApprox, currFELevel):
    errSamples = np.zeros(len(uExa))
    for ny in range(len(uExa)):
        errSamples[ny] = error_sample_fast(uApprox[ny], mesh[currFELevel], V3[currFELevel], tt[currFELevel], uExa[ny], V3ref, ttref, BDFOrder)
    errSamples = np.array(errSamples)
    return sqrt(np.mean(np.square(errSamples)))

############################### CONVERGENCE TEST ##########################################
yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])
print("Parallel random sampling reference solution...")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)

#compute sequence of approximations, error
totalCost = []
err = []
for i in range(NNLevels.size):
    nLevels = NNLevels[i]
    print("Level", nLevels)

    # generate SG interpolant w correct # nodes
    MidSetObj = midSet()
    interpolant = SGInterpolant(MidSetObj.getMidSet(), knots, lev2knots, interpolationType)  # has only 1 collocation node
    while(interpolant.numNodes < SGCards[i]):
        P = Profit(MidSetObj.margin, p)
        idMax = np.argmax(P)
        MidSetObj.update(idMax)
        interpolant = SGInterpolant(MidSetObj.getMidSet(), knots, lev2knots, interpolationType)

    # sample on SG
    FCurr = lambda y : FApprox(y, i)
    uOnSG = interpolant.sampleOnSG(FCurr)
    # inteprolate
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # error
    errCurr = errorSingleLevel(uExa, uInterp, i)
    err = np.append(err, errCurr)
    print("Error:", err[-1])

    costCurr = interpolant.numNodes * (Ntau[i] * dimVV[i])
    totalCost.append(costCurr)

    np.savetxt('convergenge_singlelevel_NONconstIC_g_02.csv',np.transpose(np.array([totalCost, err])), delimiter=',')
    
# OUTPUT
print("error:", err)
print("cost:", totalCost)
plt.loglog(totalCost, err, '.-')
plt.plot()