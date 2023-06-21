from __future__ import division
import multiprocessing
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.bdfllg_func import *
from SLLG.error_sample_fast import error_sample_fast


"""When writing a multilevel example, you want FE and SG errors to be of similar order, otherwise it makes no sense to go multilevel"""


######################## PROBLEM COEFFICIENTS
alpha = 1.4
T=1
# const IC
m1 = '0.'
m2 = '0.'
m3 = '1.'
# External magnetic field
H = (0, 0, -20.)
## g coefficient non constant, factor 0.1 to make nice plots
g1 = '0.1*(-0.5)*cos(pi*x[0])'
g2 = '0.1*(-0.5)*cos(pi*x[1])'
g3 = '0.1*sqrt(1.-0.25*(cos(pi*x[0])*cos(pi*x[0])+cos(pi*x[1])*cos(pi*x[1])))'

######################### NUMERICS COEFFICIENTS
FEMOrder=1  # FEM order
BDFOrder=1
Nh = np.array([1,2,4,8, 16, 32])  # np.array([2, 4, 8, 16, 32, 64, 128])
Ntau = Nh*4
Nhref = 64  # 256
Ntauref = Nhref*4
PETScOptions.clear()
PETScOptions.set('ksp_type', 'gmres')
PETScOptions.set("ksp_rtol", 1e-8)
PETScOptions.set("pc_type", "ilu")
PETScOptions.set("ksp_initial_guess_nonzero", "true")
quadrature_degree = 6
np.random.seed(567)
NMCSamples = 8
NParallel = 8

######################### PRE-PROCESSING
gex = Expression((g1, g2, g3), degree=FEMOrder+1)
minit = Expression((m1, m2, m3), degree=FEMOrder+1)
Pr = FiniteElement('P', triangle, FEMOrder)
tauref = float(T) / Ntauref
ttref = np.linspace(0, T, Ntauref+1)
meshref = RectangleMesh(Point(0, 0), Point(0.2, 1), Nhref, 5*Nhref)
Pr3Ref = VectorElement('Lagrange', meshref.ufl_cell(), FEMOrder, dim=3)
elementRef = MixedElement([Pr3Ref, Pr])
V3ref = FunctionSpace(meshref, Pr3Ref)
VRef = FunctionSpace(meshref, Pr)
VVRef = FunctionSpace(meshref, elementRef)
gRef = interpolate(gex, V3ref)
tau = float(T) / Ntau
tt = []
mesh = []
nElements=[]
Pr3 = []
element = []
V3 = []
V = []
VV = []
g = []
for i in range(Nh.size):
    tt.append(np.linspace(0, T, Ntau[i]+1))
    mesh.append(RectangleMesh(Point(0, 0), Point(0.2, 1), Nh[i], 5*Nh[i]))
    nElements.append(mesh[i].num_cells())
    Pr3.append(VectorElement('Lagrange', mesh[i].ufl_cell(), FEMOrder, dim=3))
    element.append(MixedElement([Pr3[i], Pr]))
    V3.append(FunctionSpace(mesh[i], Pr3[i]))
    V.append(FunctionSpace(mesh[i], Pr))
    VV.append(FunctionSpace(mesh[i], element[i]))
    g.append(interpolate(gex, V3[i]))
######################### GENERATE RANDOM FIELD PARAM
yyRnd = np.random.normal(0, 1, [NMCSamples, 1000])

######################### CONVERGENCE TEST FUNCTION
def convergence_test(yy):
    # REFERENCE
    Wref = param_LC_Brownian_motion(ttref, yy, T)
    mRef = bdfllg_func(FEMOrder, BDFOrder, alpha, T, tauref, minit, VVRef, V3ref, VRef, Wref, gRef, quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)
    dofsRef = np.array([])
    for niter in range(len(mRef)):
        dofsRef = np.concatenate((dofsRef, mRef[niter].vector()[:]))
    # CONVERGENCE TEST
    err = []
    for i in range(Nh.size):
        WCurr = np.zeros(int(Ntau[i] + 1))
        for j in range(int(Ntau[i]) + 1):
            WCurr[j] = Wref[int(j * (Ntauref / Ntau[i]))]
        mCurr = bdfllg_func(FEMOrder, BDFOrder, alpha, T, tau[i], minit, VV[i], V3[i], V[i], WCurr, g[i], quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)
        dofsCurr = np.array([])
        for niter in range(len(mCurr)):
            dofsCurr = np.concatenate((dofsCurr, mCurr[niter].vector()[:]))
        err.append(error_sample_fast(dofsCurr, mesh[i], V3[i], tt[i], dofsRef, V3ref, ttref, BDFOrder))
    return np.array(err)

######################### RUN IN PARALLEL
pool = multiprocessing.Pool(NParallel)
errSamples = np.zeros((NMCSamples, Nh.size))
errSamples = pool.map(convergence_test, yyRnd)
err = np.sqrt(np.mean(np.square(errSamples), axis=0))

np.savetxt('multilevel_preprocessing_conv_FE.csv',np.transpose(np.array([nElements, err])), delimiter=',')

print("Error", err)
plt.loglog(nElements, err, '.-')
plt.xlabel("Number of mesh elements")
plt.show()