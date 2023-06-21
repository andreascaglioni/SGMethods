import numpy as np
from dolfin import *
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SLLG'))
from SLLG.compute_H1_error import compute_H1_error
from SLLG.compute_dtm_error import compute_dtm_error

"""Compute a fininte element error in space + time given a reference solution. NB spaces must be ensted! Also pass fininte element sapces for faster computations"""
def error_sample_fast(dofscurr, mesh, V3, tt, dofsref, V3ref, ttref, p, DT_ERROR=False):

    Nt = len(tt)
    tau = tt[1]-tt[0]
    Ntref = len(ttref)
    tauref = ttref[1]-ttref[0]
    # 1 PREPARE REFERENCE
    dimV3ref = V3ref.dim()
    if len(dofsref) != Ntref*dimV3ref:
        print("mismatch reference sol:", len(dofsref), " != ", Ntref, " * ", dimV3ref)
        assert(len(dofsref) == Ntref*dimV3ref)
    matdofsref = np.ndarray((dimV3ref, Ntref))
    for n in range(Ntref):
        matdofsref[:, n] = dofsref[n*dimV3ref:(n+1)*dimV3ref:]
    # 2 PREPARE CURRENT
    dimV3 = V3.dim()
    if len(dofscurr) != Nt*dimV3:
        print("mismatch current sol:", len(dofscurr), " != ", Nt, " * ", dimV3)
        assert(len(dofscurr) == Nt*dimV3)
    matdofscurr = np.ndarray((dimV3, Nt))
    for n in range(Nt):
        matdofscurr[:, n] = dofscurr[n*dimV3:(n+1)*dimV3:]

    # workaround to avoid "PETSC ERROR: Logging has not been enabled."
    Pr = FiniteElement('P', triangle, 1)
    Vtmp = FunctionSpace(mesh, Pr)
    tmp = assemble(TestFunction(Vtmp)*dx)
    # end workaround

    # 3 COMPUTE SPACE ERROR FOR EACH TIMESTEP
    errtime = compute_H1_error(p, matdofsref, V3ref, ttref, matdofscurr, V3, tt)

    if DT_ERROR:
        errtimecurr_dtm = compute_dtm_error(p, matdofsref, V3ref, tauref, ttref, matdofscurr, V3, tau, tt)
        errtime += errtimecurr_dtm

    # 4 RETURN L2 ERROR IN TIME
    return sqrt ( tauref * np.sum(np.square(errtime)) )  # NB errtime was on REFERENCE time grid
