from dolfin import *
import matplotlib.pyplot as plt
import math

address_read = "/home/andrea/workspace/SGMethods/m_switching_rectange.h5"
address_write = "m_plot.xdmf"

Nh = 16
Ntau = Nh * 64
r = 1  # FEM order
# mesh = UnitSquareMesh(Nh, Nh)
mesh = RectangleMesh(Point(0, 0), Point(0.2,1), Nh, 5*Nh)

Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
V3 = FunctionSpace(mesh, Pr3)

file_read = HDF5File(mesh.mpi_comm(), address_read, "r")
file_write = XDMFFile(address_write)
file_write.parameters["flush_output"] = True
m = Function(V3)
for i in range(0, Ntau+1):
    file_read.read(m, "/function/vector_{}".format(i))
    # plot(m)
    # plt.show()
    timestamp = file_read.attributes("function/vector_{}".format(i))["timestamp"]
    assert timestamp == i

    m.rename('m', 'm')
    file_write.write(m, i)
file_write.close()


