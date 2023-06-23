from __future__ import division
from expansions_Brownian_motion import param_LC_Brownian_motion
from bdfllg_func import *
import matplotlib.pyplot as plt


"""Example of magnetization switching wiht noise. There is a flat domain and the
constant initial maginetization constant and is directed perperdicularly. There 
is a constant external maginetic field parallel to the m_0 but in the opposite
direction. """


r=1  # FEM order
p=1
Nh = 16
Ntau = Nh * 64
alpha = 1.4
T=1
np.random.seed(256)
param = np.random.normal(0,1,10000)  # parameter for LC noise
# External magnetic field
H = (0, 0, -20.)
## g constant
# g1 = '0.'
# g2 = '0.'
# g3 = '0.'
## g coefficient non constant
# g1 = '-0.5*cos(pi*x[0])'
# g2 = '-0.5*cos(pi*x[1])'
# g3 = 'sqrt(1-('+g1+')*('+g1+')-('+g2+')*('+g2+'))'
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
W = param_LC_Brownian_motion(tt, param, T)
Pr = FiniteElement('P', triangle, r)
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), Nh, 5*Nh)

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
mapr = bdfllg_func(r, p, alpha, T, tau, minit, VV, V3, V, W, g, m1=[], quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=True)

# SAVE SOLUTION TO FILE
with HDF5File(mesh.mpi_comm(), "m_switching_rectangle_01_seed256.h5", "w") as file:
    for niter in range(len(mapr)):
        file.write(mapr[niter], "/function", niter)

# Max = np.zeros(len(mapr))
# Min = np.zeros(len(mapr))
# for i in range(len(mapr)):
#     mCurr = mapr[i]  # sol at curent timestep
#     mZ = mCurr.sub(2,deepcopy=True)  # z component
#     valsZ = mZ.vector().get_local()  # vertices values
#     Max[i] = np.amax(valsZ)
#     Min[i] = np.amin(valsZ)

# plt.plot(Min, '-', label="min")
# plt.plot(Max, '-', label="max")
# plt.legend()
# plt.show()
