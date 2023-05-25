from math import ceil
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SLLG'))
from SLLG.compute_H1_error import compute_H1_error
from SLLG.compute_dtm_error import compute_dtm_error


def error_sample_pb2_time_1(dofscurr, dofsref, Nhref, Nh, r):

    # 1 PREPARE REFERENCE
    meshref = UnitSquareMesh(Nhref, Nhref)
    Pr3 = VectorElement('Lagrange', meshref.ufl_cell(), r, dim=3)
    V3ref = FunctionSpace(meshref, Pr3)
    dimV3ref = V3ref.dim()
    if len(dofsref) != dimV3ref:
        print("mismatch reference sol:", len(dofsref), " != ", dimV3ref)
        assert(len(dofsref) == dimV3ref)

    # 2 PREPARE CURRENT
    mesh = meshref = UnitSquareMesh(Nh, Nh)
    Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3)
    V3 = FunctionSpace(mesh, Pr3)
    dimV3 = V3.dim()
    if len(dofscurr) != dimV3:
        print("mismatch current sol:", len(dofscurr), " != ", dimV3)
        assert(len(dofscurr) == dimV3)

    # workaround to avoid "PETSC ERROR: Logging has not been enabled."
    Pr = FiniteElement('P', triangle, 1)
    Vtmp = FunctionSpace(mesh, Pr)
    tmp = assemble(TestFunction(Vtmp)*dx)
    # end workaround

    # 3 COMPUTE SPACE ERROR FOR EACH TIMESTEP
    if V3.dim() != V3ref.dim():
        M = PETScDMCollection.create_transfer_matrix(V3, V3ref)
    err = Function(V3ref)
    tmp = Function(V3)
    if V3.dim() == V3ref.dim():
        err.vector()[:] = dofsref - dofscurr
    else:
        tmp.vector()[:] = dofscurr
        v2 = M * tmp.vector()
        err.vector()[:] = dofsref - v2[:]
    return norm(err, 'H1')