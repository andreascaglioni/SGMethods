from __future__ import division
import matplotlib.pyplot as plt
import multiprocessing
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.bdfllg_func import *

"""Example of magnetization switching wiht noise. There is a flat domain and the
constant initial maginetization constant and is directed perperdicularly. There 
is a constant external maginetic field parallel to the m_0 but in the opposite
direction. """


r=1  # FEM order
p=1
Nh = 16  #256  #  
Ntau = Nh * 4
alpha = 1.4
T=1
np.random.seed(5295)
NParallel = 8

# External magnetic field
H = (0, 0, -10.)
## g non const rescaled 0.1
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
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), Nh, 5*Nh)
Pr = FiniteElement('P', triangle, r)
Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
element = MixedElement([Pr3, Pr])
VV = FunctionSpace(mesh, element)
V3 = FunctionSpace(mesh, Pr3)
V = FunctionSpace(mesh, Pr)
g = interpolate(gex, V3)
quadrature_degree = 6
PETScOptions.clear()
PETScOptions.set('ksp_type', 'gmres')
PETScOptions.set("ksp_rtol", 1e-10)
PETScOptions.set("pc_type", "ilu")

NMCSamples = 32
W = []
for n in range(NMCSamples):
    param = np.random.normal(0,1,10000)  # parameter for LC noise
    W.append(param_LC_Brownian_motion(tt, param, T))

def sample_LLG(n):
    print("computing", n)
    WCurr = W[n]
    mapr = bdfllg_func(r, p, alpha, T, tau, minit, VV, V3, V, WCurr, g, m1=[], quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=False)
    # extract, for every t, the MAXIMUM z component over D
    Max = np.zeros(len(mapr))
    Min = np.zeros(len(mapr))
    for i in range(len(mapr)):
        mCurr = mapr[i]  # sol at curent timestep
        mZ = mCurr.sub(2,deepcopy=True)  # z component
        valsZ = mZ.vector().get_local()  # vertices values
        Max[i] = np.amax(valsZ)
        Min[i] = np.amin(valsZ)
    np.savetxt('swsitch_rectange_min_max_sample_'+str(n)+'.csv',np.transpose(np.array([tt, WCurr, Min, Max])), delimiter=',')

pool = multiprocessing.Pool(NParallel)
pool.map(sample_LLG, range(0, NMCSamples))