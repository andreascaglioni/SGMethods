from __future__ import division
from scipy.io import savemat
from expansions_Brownian_motion import param_LC_Brownian_motion
from bdfllg_func import *
from compute_H1_error import compute_H1_error
from compute_magnitude_error import compute_magnitude_error
from math import ceil


"""Just consider a sample path and test converngece of BDF+FE for time and space approximation"""

addr_results = 'test_convegence_SLLG_problem_2/'
set_log_level(31)

# parameters current run:
T = 1
alpha = 1.4
p = 1
r = 1
Nh = np.array([2,4,8,16,32])  # np.array([2, 4, 8, 16, 32, 64, 128])
Ntau = Nh*8
Nhref = 64  # 256
Ntauref = Nhref*8

assert Nhref >= np.amax(Nh) and Ntauref > np.amax(Ntau)
assert Nh.size == Ntau.size
assert p < 3  # to be implemented

np.random.seed(567)
param = np.random.normal(0, 1, 1000)

PETScOptions.clear()
PETScOptions.set('ksp_type', 'gmres')
PETScOptions.set("ksp_rtol", 1e-8)
PETScOptions.set("pc_type", "ilu")
PETScOptions.set("ksp_initial_guess_nonzero", "true")
quadrature_degree = 6

H = (0, 0, 0)
# coefficient g
g1 = '-0.5*cos(pi*x[0])'
g2 = '-0.5*cos(pi*x[1])'
g3 = 'sqrt(1-('+g1+')*('+g1+')-('+g2+')*('+g2+'))'
gex = Expression((g1, g2, g3), degree=r)
# CONSTANT IC
m1 = '0.'
m2 = '0.'
m3 = '1.'
minit = Expression((m1, m2, m3), degree=r+1)

##################### REFERENCE SOLUTION ################
tau = float(T) / Ntau
tauref = float(T) / Ntauref
ttref = np.linspace(0, T, Ntauref+1)

Wref = param_LC_Brownian_motion(ttref, param, T)
# Wiener Processes on coarser time grids
W = np.ndarray([len(Ntau)], dtype=np.object)
for i in range(len(Ntau)):
    W[i] = np.zeros(int(Ntau[i] + 1))
    for j in range(int(Ntau[i]) + 1):
        W[i][j] = Wref[int(j * (Ntauref / Ntau[i]))]

ttref = np.linspace(0, T, Ntauref + 1)
meshref = UnitSquareMesh(Nhref, Nhref)

Pr = FiniteElement('P', triangle, r)
Pr3 = VectorElement('Lagrange', meshref.ufl_cell(), r, dim=3)
element = MixedElement([Pr3, Pr])
VV = FunctionSpace(meshref, element)
V3ref = FunctionSpace(meshref, Pr3)
V = FunctionSpace(meshref, Pr)
g = interpolate(gex, V3ref)
mref = bdfllg_func(r, p, alpha, T, tauref, minit, VV, V3ref, V, Wref, g, quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=True)

matdofsref = np.ndarray((V3ref.dim(), Ntauref + 1))  # needed later to compute error
for n in range(Ntauref + 1):
    matdofsref[:, n] = mref[n].vector()[:]

################## CONVERGENCE TEST ################
mapr, err, errmag = [], [], []
for i in range(0, len(Ntau)):
    print("Computing for NTau", Ntau[i], "Nh", Nh[i], flush=True)
    ttcurr = np.linspace(0, T, Ntau[i] + 1)
    
    mesh = UnitSquareMesh(Nhref, Nhref)

    Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
    element = MixedElement([Pr3, Pr])
    VV = FunctionSpace(mesh, element)
    V3 = FunctionSpace(mesh, Pr3)
    V = FunctionSpace(mesh, Pr)
    g = interpolate(gex, V3)

    mapr = bdfllg_func(r, p, alpha, T, tau[i], minit, VV, V3, V, W[i], g, quadrature_degree=quadrature_degree, Hinput=H)

    
    matdofscurr = np.ndarray((V3.dim(), Ntau[i] + 1))
    for n in range(Ntau[i] + 1):
        matdofscurr[:, n] = mapr[n].vector()[:]
    errH1Time = compute_H1_error(p, matdofsref, V3ref, ttref, matdofscurr, V3, ttcurr)
    err.append(np.amax(errH1Time))

    errmagtime = compute_magnitude_error(Ntau[i], mapr, V)
    errmag.append(np.amax(errmagtime))
np.savetxt('convergence_test_problem_2.txt', np.column_stack((Nh, Ntau, err, errmag)), delimiter=',')