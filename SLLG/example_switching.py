from __future__ import division
from expansions_Brownian_motion import param_LC_Brownian_motion
from bdfllg_func import *

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
np.random.seed(5295)
param = np.random.normal(0,1,10000)  # parameter for LC noise
# External magnetic field
H = (0, 0, -20.)
## g constant
# g1 = '0.'
# g2 = '1.'
# g3 = '0.'
## g coefficient non constant
g1 = '-0.5*cos(pi*x[0])'
g2 = '-0.5*cos(pi*x[1])'
g3 = 'sqrt(1-('+g1+')*('+g1+')-('+g2+')*('+g2+'))'

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
# mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 0.1), Nh, Nh, ceil(Nh/10.))
# mesh = UnitSquareMesh(Nh, Nh)
mesh = RectangleMesh(Point(0, 0), Point(0, 0, 0), Nh, 5*Nh)

Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
element = MixedElement([Pr3, Pr])
VV = FunctionSpace(mesh, element)
V3 = FunctionSpace(mesh, Pr3)
V = FunctionSpace(mesh, Pr)

g = interpolate(gex, V3)
# file_write = XDMFFile("plot_g.xdmf")
# file_write.write(g)
# file_write.close()

quadrature_degree = 6
PETScOptions.clear()
PETScOptions.set('ksp_type', 'gmres')
PETScOptions.set("ksp_rtol", 1e-10)
PETScOptions.set("pc_type", "ilu")
mapr = bdfllg_func(r, p, alpha, T, tau, minit, VV, V3, V, W, g, m1=[], quadrature_degree=quadrature_degree, Hinput=H, VERBOSE=True)

# SAVE SOLUTION TO FILE
with HDF5File(mesh.mpi_comm(), "m_switching_rectange.h5", "w") as file:
    for niter in range(len(mapr)):
        file.write(mapr[niter], "/function", niter)
